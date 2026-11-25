"""Load source tables: the Hugging Face dataset plus the public NHL schedule API."""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATASET = "jjerlee/nhl-analytics"
HF_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main/data/{{name}}.parquet"
NHL_API = "https://api-web.nhle.com/v1"
RAW_DIR = Path(os.environ.get("NHLPRED_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "raw"))

# Franchises that changed abbreviation; ratings and form carry over.
FRANCHISE = {"PHX": "UTA", "ARI": "UTA"}

_session = requests.Session()
_session.headers["User-Agent"] = "nhlpred/0.1"


def download(names=("games", "odds")) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        r = _session.get(HF_URL.format(name=name), timeout=120)
        r.raise_for_status()
        (RAW_DIR / f"{name}.parquet").write_bytes(r.content)


def load_games() -> pd.DataFrame:
    """Completed games, one row per game, oldest first."""
    g = pd.read_parquet(RAW_DIR / "games.parquet")
    g = g.dropna(subset=["home_score", "away_score"]).copy()
    g["start_utc"] = pd.to_datetime(g["time_start"], utc=True)
    g["game_date"] = pd.to_datetime(g["game_date"]).dt.date
    g["game_type"] = (g["game_id"] // 10000 % 100).astype(int)  # 2 regular, 3 playoffs
    g["home"] = g["home_team"].replace(FRANCHISE)
    g["away"] = g["away_team"].replace(FRANCHISE)
    g["home_win"] = (g["home_score"] > g["away_score"]).astype(float)
    g["overtime"] = (g["period_ending"] != "REG").astype(int)
    keep = [
        "game_id", "season", "game_type", "game_date", "start_utc", "home", "away",
        "home_score", "away_score", "home_win", "overtime",
        "xg_for", "xg_against", "hdcf_for", "hdcf_against", "gsax_for", "gsax_against",
    ]
    return g[keep].sort_values(["start_utc", "game_id"]).reset_index(drop=True)


def american_to_prob(odds: pd.Series) -> np.ndarray:
    o = odds.astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def load_market() -> pd.DataFrame:
    """Vig-free home-win probability from the two-way moneyline, one row per game.

    The dataset keeps only the latest snapshot per book, and about one in six of
    those was captured after puck drop (a live in-game line). Those are dropped:
    only quotes timestamped before the scheduled start count as a pre-game price.
    """
    o = pd.read_parquet(RAW_DIR / "odds.parquet")
    o = o[o["odds_description"] == "MONEY_LINE_2_WAY"].copy()
    o["start_utc"] = pd.to_datetime(o["start_time_utc"], utc=True)
    o["quoted_utc"] = pd.to_datetime(o["last_updated_utc"], utc=True)
    o = o[o["quoted_utc"] < o["start_utc"]]
    ph, pa = american_to_prob(o["home_odds"]), american_to_prob(o["away_odds"])
    o["market_p_home"] = ph / (ph + pa)
    o = o[(ph + pa > 1.0) & (ph + pa < 1.12)]  # discard malformed quotes
    out = (
        o.groupby("game_id")
        .agg(market_p_home=("market_p_home", "mean"), market_quoted_utc=("quoted_utc", "max"),
             market_books=("betting_partner", lambda s: ",".join(sorted(set(s)))))
        .reset_index()
    )
    return out


def _schedule_week(day: date) -> list[dict]:
    r = _session.get(f"{NHL_API}/schedule/{day.isoformat()}", timeout=30)
    r.raise_for_status()
    rows = []
    for d in r.json().get("gameWeek", []):
        for g in d.get("games", []):
            if g.get("gameType") not in (2, 3):
                continue
            rows.append({
                "game_id": int(g["id"]),
                "season": int(g["season"]),
                "game_type": int(g["gameType"]),
                "game_date": date.fromisoformat(d["date"]),
                "start_utc": pd.Timestamp(g["startTimeUTC"]),
                "home": FRANCHISE.get(g["homeTeam"]["abbrev"], g["homeTeam"]["abbrev"]),
                "away": FRANCHISE.get(g["awayTeam"]["abbrev"], g["awayTeam"]["abbrev"]),
                "state": g.get("gameState"),
                "schedule_state": g.get("gameScheduleState"),
                "home_score": g["homeTeam"].get("score"),
                "away_score": g["awayTeam"].get("score"),
                "overtime": int(g.get("gameOutcome", {}).get("lastPeriodType", "REG") != "REG"),
            })
    return rows


def load_schedule(start: date, end: date) -> pd.DataFrame:
    """Games from the NHL API between two dates inclusive (final, live and future)."""
    rows, day = [], start
    while day <= end:
        rows += _schedule_week(day)
        day += timedelta(days=7)
    df = pd.DataFrame(rows).drop_duplicates("game_id") if rows else pd.DataFrame()
    if df.empty:
        return df
    df = df[(df["game_date"] >= start) & (df["game_date"] <= end)]
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    return df.sort_values(["start_utc", "game_id"]).reset_index(drop=True)


FINAL_STATES = {"OFF", "FINAL"}


def with_recent_results(games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Append finished games the dataset has not published yet (score only).

    The dataset refreshes once a day, so last night's results can be missing in
    the morning. Ratings and rest need them; the shot-quality columns stay empty
    for those rows and the rolling averages skip them.
    """
    if schedule.empty:
        return games
    done = schedule[schedule["state"].isin(FINAL_STATES) & ~schedule["game_id"].isin(games["game_id"])].copy()
    if done.empty:
        return games
    done["home_win"] = (done["home_score"] > done["away_score"]).astype(float)
    done = done.reindex(columns=games.columns)
    out = pd.concat([games, done], ignore_index=True)
    return out.sort_values(["start_utc", "game_id"]).reset_index(drop=True)


def load_live_market() -> pd.DataFrame:
    """Today's two-way moneyline straight from the NHL's partner odds feed.

    Same source the dataset archives, read directly so the price is captured
    before puck drop rather than whenever the dataset's own job happens to run.
    """
    cols = ["game_id", "market_p_home", "market_quoted_utc", "market_books"]
    rows = []
    for country in ("US", "CA"):
        try:
            r = _session.get(f"{NHL_API}/partner-game/{country}/now", timeout=30)
            r.raise_for_status()
            d = r.json()
        except (requests.RequestException, ValueError):
            continue
        quoted = pd.Timestamp(d.get("lastUpdatedUTC"))
        book = (d.get("bettingPartner") or {}).get("name", country)
        for g in d.get("games", []):
            line = {}
            for side in ("homeTeam", "awayTeam"):
                for o in g[side].get("odds", []):
                    if o.get("description") == "MONEY_LINE_2_WAY":
                        line[side] = o.get("value")
            if len(line) < 2 or quoted >= pd.Timestamp(g["startTimeUTC"]):
                continue
            ph, pa = american_to_prob(pd.Series([line["homeTeam"]]))[0], american_to_prob(pd.Series([line["awayTeam"]]))[0]
            if not 1.0 < ph + pa < 1.12:
                continue
            rows.append({"game_id": int(g["gameId"]), "market_p_home": ph / (ph + pa),
                         "market_quoted_utc": quoted, "market_books": book})
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    return (df.groupby("game_id")
              .agg(market_p_home=("market_p_home", "mean"), market_quoted_utc=("market_quoted_utc", "max"),
                   market_books=("market_books", lambda s: ",".join(sorted(set(s)))))
              .reset_index())
