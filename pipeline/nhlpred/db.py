"""Postgres writes. A prediction can change until puck drop and never after."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import psycopg

SCHEMA = (Path(__file__).parent / "schema.sql").read_text()


def connect() -> psycopg.Connection:
    return psycopg.connect(os.environ["DATABASE_URL"], autocommit=False)


def init(conn) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def _n(v):
    return None if v is None or pd.isna(v) else v


def upsert_predictions(conn, df: pd.DataFrame, source: str, model_version: str) -> int:
    """Insert or refresh predictions. Rows whose game has started are left alone."""
    sql = """
    INSERT INTO predictions (game_id, season, game_type, game_date, start_utc, home, away,
                             p_home, elo_diff, model_version, source, predicted_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
    ON CONFLICT (game_id) DO UPDATE SET
      p_home = EXCLUDED.p_home, elo_diff = EXCLUDED.elo_diff,
      model_version = EXCLUDED.model_version, predicted_at = EXCLUDED.predicted_at,
      start_utc = EXCLUDED.start_utc, game_date = EXCLUDED.game_date
    WHERE predictions.source = 'live' AND predictions.start_utc > now() AND predictions.home_win IS NULL
    """
    n = 0
    with conn.cursor() as cur:
        for r in df.itertuples():
            cur.execute(sql, (int(r.game_id), int(r.season), int(r.game_type), r.game_date,
                              r.start_utc.to_pydatetime(), r.home, r.away, float(r.p_home),
                              _n(r.elo_diff), model_version, source))
            n += cur.rowcount
    conn.commit()
    return n


def seed_backtest(conn, df: pd.DataFrame, model_version: str) -> int:
    """Load settled walk-forward predictions. Never overwrites an existing row."""
    sql = """
    INSERT INTO predictions (game_id, season, game_type, game_date, start_utc, home, away, p_home,
                             model_version, source, predicted_at, market_p_home, home_score,
                             away_score, overtime, home_win, settled_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'backtest', now(), %s, %s, %s, %s, %s, now())
    ON CONFLICT (game_id) DO NOTHING
    """
    rows = [(int(r.game_id), int(r.season), int(r.game_type), r.game_date, r.start_utc.to_pydatetime(),
             r.home, r.away, float(r.p_logit), model_version, _n(r.market_p_home), int(r.home_score),
             int(r.away_score), bool(r.overtime), bool(r.home_win)) for r in df.itertuples()]
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
        n = cur.rowcount
    conn.commit()
    return n


def set_market(conn, market: pd.DataFrame) -> int:
    """Attach the latest pre-game market price to unsettled live predictions."""
    sql = """
    UPDATE predictions SET market_p_home = %s, market_quoted_utc = %s
    WHERE game_id = %s AND source = 'live' AND home_win IS NULL AND %s < start_utc
      AND (market_quoted_utc IS NULL OR market_quoted_utc < %s)
    """
    n = 0
    with conn.cursor() as cur:
        for r in market.itertuples():
            q = r.market_quoted_utc.to_pydatetime()
            cur.execute(sql, (float(r.market_p_home), q, int(r.game_id), q, q))
            n += cur.rowcount
    conn.commit()
    return n


def settle(conn, finals: pd.DataFrame) -> int:
    sql = """
    UPDATE predictions SET home_score = %s, away_score = %s, overtime = %s, home_win = %s, settled_at = now()
    WHERE game_id = %s AND home_win IS NULL
    """
    n = 0
    with conn.cursor() as cur:
        for r in finals.itertuples():
            cur.execute(sql, (int(r.home_score), int(r.away_score), bool(r.overtime),
                              bool(r.home_score > r.away_score), int(r.game_id)))
            n += cur.rowcount
    conn.commit()
    return n


def unsettled_window(conn):
    row = conn.execute(
        "SELECT min(game_date), max(game_date) FROM predictions WHERE home_win IS NULL AND start_utc < now()"
    ).fetchone()
    return row


def log_run(conn, model_version, n_train, trained_through, n_predicted, n_settled, coefficients) -> None:
    conn.execute(
        "INSERT INTO model_runs (model_version, n_train, trained_through, n_predicted, n_settled, coefficients)"
        " VALUES (%s, %s, %s, %s, %s, %s)",
        (model_version, n_train, trained_through, n_predicted, n_settled, json.dumps(coefficients)),
    )
    conn.commit()
