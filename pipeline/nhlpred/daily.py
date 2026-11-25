"""The scheduled job: settle, retrain, predict, record the market price.

Run: python -m nhlpred.daily            (needs DATABASE_URL)
     python -m nhlpred.daily --seed     (first run: also load the backtest history)
     python -m nhlpred.daily --dry-run  (print predictions, write nothing)

Safe to run several times a day. Later runs pick up fresher market prices and any
schedule changes; a prediction stops changing at its scheduled start.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from . import backtest, data, db, model
from .features import FEATURES, build_features

KIND = "logit"
HORIZON_DAYS = 1  # predict today's and tomorrow's games
LOOKBACK_DAYS = 5  # pull recent finals the dataset may not have yet
ARTIFACTS = Path(__file__).resolve().parents[2] / "data" / "model"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    if not args.no_download:
        data.download()
    today = datetime.now(ZoneInfo("America/New_York")).date()
    now = datetime.now(timezone.utc)

    conn = None if args.dry_run else db.connect()
    lookback = LOOKBACK_DAYS
    if conn:
        db.init(conn)
        oldest, _ = db.unsettled_window(conn)
        if oldest:
            lookback = max(lookback, (today - oldest).days + 1)

    schedule = data.load_schedule(today - timedelta(days=lookback), today + timedelta(days=HORIZON_DAYS))
    games = data.with_recent_results(data.load_games(), schedule)

    upcoming = schedule[
        (schedule["start_utc"] > now) & (schedule["schedule_state"] == "OK")
        & ~schedule["state"].isin(data.FINAL_STATES) & ~schedule["game_id"].isin(games["game_id"])
    ]
    frame = pd.concat([games, upcoming.reindex(columns=games.columns)], ignore_index=True)
    feats = build_features(frame)

    train = feats[feats["home_win"].notna() & feats["home_gd_10"].notna() & feats["away_gd_10"].notna()]
    fitted = model.fit(KIND, train)
    todo = feats[feats["game_id"].isin(upcoming["game_id"])].copy()
    todo["p_home"] = model.predict(KIND, fitted, todo) if len(todo) else []

    logit = fitted.named_steps["logisticregression"]
    coefs = {f: round(float(c), 4) for f, c in zip(FEATURES, logit.coef_[0])}
    coefs["intercept"] = round(float(logit.intercept_[0]), 4)
    trained_through = train["game_date"].max()

    print(f"trained on {len(train)} games through {trained_through}; {len(todo)} upcoming")
    for r in todo.itertuples():
        print(f"  {r.game_date} {r.away:>3} @ {r.home:<3}  P(home)={r.p_home:.3f}  elo_diff={r.elo_diff:+.0f}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted, "features": list(FEATURES), "version": model.MODEL_VERSION,
                 "trained_through": str(trained_through), "n_train": len(train)}, ARTIFACTS / "model.joblib")

    if args.dry_run:
        return

    if args.seed:
        backtest.run()
        hist = pd.read_parquet(backtest.OUT / "predictions.parquet")
        print("seeded backtest rows:", db.seed_backtest(conn, hist, model.MODEL_VERSION))

    finals = schedule[schedule["state"].isin(data.FINAL_STATES)]
    n_settled = db.settle(conn, finals)
    n_pred = db.upsert_predictions(conn, todo, "live", model.MODEL_VERSION)
    n_market = db.set_market(conn, data.load_market()) + db.set_market(conn, data.load_live_market())
    db.log_run(conn, model.MODEL_VERSION, len(train), trained_through, n_pred, n_settled, coefs)
    print(f"settled {n_settled}, wrote {n_pred} predictions, attached {n_market} market prices")
    conn.close()


if __name__ == "__main__":
    main()
