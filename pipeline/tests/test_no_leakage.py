"""A game's features must not depend on that game's own outcome."""
import numpy as np
import pandas as pd

from nhlpred import data
from nhlpred.features import FEATURES, build_features


def test_features_ignore_own_result():
    games = data.load_games().tail(3000).reset_index(drop=True)
    base = build_features(games)
    ids = games["game_id"].iloc[-200:]

    flipped = games.copy()
    m = flipped["game_id"].isin(ids)
    flipped.loc[m, "home_win"] = 1.0 - flipped.loc[m, "home_win"]
    flipped.loc[m, "home_score"] = np.where(flipped.loc[m, "home_win"] == 1.0, 9.0, 0.0)
    flipped.loc[m, "away_score"] = 9.0 - flipped.loc[m, "home_score"]
    for c in ("xg_for", "xg_against", "hdcf_for", "hdcf_against", "gsax_for", "gsax_against"):
        flipped.loc[m, c] = flipped.loc[m, c] * 2 + 1
    alt = build_features(flipped)

    # the first altered game sees only untouched history, so it must match exactly
    first = ids.iloc[0]
    a = base.loc[base["game_id"] == first, FEATURES].to_numpy(dtype=float)
    b = alt.loc[alt["game_id"] == first, FEATURES].to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True)


def test_unplayed_game_gets_features():
    games = data.load_games().tail(2000).reset_index(drop=True)
    future = games.tail(1).copy()
    future["game_id"] += 100000
    future["start_utc"] += np.timedelta64(3, "D")
    future[["home_score", "away_score", "home_win"]] = np.nan
    out = build_features(pd.concat([games, future], ignore_index=True))
    row = out[out["game_id"] == future["game_id"].iloc[0]]
    assert row[FEATURES].notna().all(axis=1).iloc[0]
