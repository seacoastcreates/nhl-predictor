"""Walk-forward backtest: for each season, train on every earlier season, predict it.

Run: python -m nhlpred.backtest
Writes data/backtest/summary.json and data/backtest/predictions.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, model
from .features import build_features

OUT = Path(__file__).resolve().parents[2] / "data" / "backtest"
FIRST_TEST_SEASON = 20172018
KINDS = ("elo", "logit", "gbm")


def run() -> dict:
    games = data.load_games()
    feats = build_features(games)
    feats = feats[feats["home_gd_10"].notna() & feats["away_gd_10"].notna()]
    seasons = sorted(s for s in feats["season"].unique() if s >= FIRST_TEST_SEASON)

    parts = []
    for season in seasons:
        train, test = feats[feats["season"] < season], feats[feats["season"] == season].copy()
        for kind in KINDS:
            test[f"p_{kind}"] = model.predict(kind, model.fit(kind, train), test)
        test["p_home_rate"] = train["home_win"].mean()
        parts.append(test)
    pred = pd.concat(parts, ignore_index=True).merge(data.load_market(), on="game_id", how="left")

    y = pred["home_win"].to_numpy()
    summary = {"overall": {}, "by_season": {}, "vs_market": {}}
    for kind in KINDS + ("home_rate",):
        summary["overall"][kind] = model.metrics(y, pred[f"p_{kind}"])
    for season, d in pred.groupby("season"):
        summary["by_season"][int(season)] = {k: model.metrics(d["home_win"], d[f"p_{k}"]) for k in KINDS}

    m = pred[pred["market_p_home"].notna()]
    summary["vs_market"]["market"] = model.metrics(m["home_win"], m["market_p_home"])
    for kind in KINDS:
        summary["vs_market"][kind] = model.metrics(m["home_win"], m[f"p_{kind}"])
    summary["vs_market"]["seasons"] = sorted(int(s) for s in m["season"].unique())

    # calibration of the logistic model in 5-point bins
    bins = np.clip((pred["p_logit"] * 20).astype(int), 0, 19)
    cal = pred.groupby(bins).agg(p=("p_logit", "mean"), actual=("home_win", "mean"), n=("home_win", "size"))
    summary["calibration_logit"] = [
        {"bin": float(b) / 20, "predicted": float(r.p), "actual": float(r.actual), "n": int(r.n)}
        for b, r in cal.iterrows()
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    keep = ["game_id", "season", "game_type", "game_date", "start_utc", "home", "away", "home_score",
            "away_score", "home_win", "overtime", "p_elo", "p_logit", "p_gbm", "market_p_home"]
    pred[keep].to_parquet(OUT / "predictions.parquet", index=False)
    return summary


if __name__ == "__main__":
    s = run()
    fmt = lambda m: f"n={m['n']:>5}  logloss={m['log_loss']:.4f}  brier={m['brier']:.4f}  acc={m['accuracy']:.3f}"
    print("Walk-forward, all test seasons")
    for k, v in s["overall"].items():
        print(f"  {k:<10} {fmt(v)}")
    print(f"Games with a pre-game market price ({s['vs_market']['seasons']})")
    for k, v in s["vs_market"].items():
        if k != "seasons":
            print(f"  {k:<10} {fmt(v)}")
    print("Logistic model by season")
    for season, v in s["by_season"].items():
        print(f"  {season}  {fmt(v['logit'])}")
