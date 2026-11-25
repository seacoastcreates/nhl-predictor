"""Pre-game features. Every value for a game uses only games that started before it.

Upcoming games are appended to the frame with empty scores and pass through the
same code as history, so training and serving cannot drift apart.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

ELO_BASE = 1500.0
ELO_K = 6.0
ELO_HOME = 40.0
ELO_CARRY = 0.7  # share of rating kept between seasons

WINDOWS = (10, 40)
TEAM_STATS = ("gd", "xgd", "xg_share", "hd_share", "gsax", "win")

FEATURES = (
    ["elo_diff", "rest_diff", "home_b2b", "away_b2b", "playoff", "season_progress"]
    + [f"{s}_{w}_diff" for w in WINDOWS for s in TEAM_STATS]
)


def add_elo(g: pd.DataFrame) -> pd.DataFrame:
    """Margin-adjusted Elo; stores the home-minus-away rating before each game."""
    rating: dict[str, float] = {}
    last_season: dict[str, int] = {}
    diffs = np.zeros(len(g))
    cols = zip(g["home"], g["away"], g["season"], g["home_score"], g["away_score"], g["game_type"])
    for i, (h, a, season, hs, as_, gtype) in enumerate(cols):
        for t in (h, a):
            if t not in rating:
                rating[t] = ELO_BASE
            elif last_season[t] != season:
                rating[t] = ELO_BASE + ELO_CARRY * (rating[t] - ELO_BASE)
            last_season[t] = season
        diff = rating[h] - rating[a]
        diffs[i] = diff
        if pd.isna(hs) or pd.isna(as_):
            continue
        d = diff + ELO_HOME
        expected = 1.0 / (1.0 + 10 ** (-d / 400.0))
        won = 1.0 if hs > as_ else 0.0
        margin = abs(hs - as_)
        winner_edge = d if won else -d
        mult = (0.6686 * math.log(margin) + 0.8048) * (2.05 / (winner_edge * 0.001 + 2.05))
        k = ELO_K * (1.5 if gtype == 3 else 1.0)
        delta = k * mult * (won - expected)
        rating[h] += delta
        rating[a] -= delta
    g = g.copy()
    g["elo_diff"] = diffs
    return g


def _team_long(g: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game, one from each team's point of view."""
    def side(team, opp, gf, ga, xgf, xga, hdf, hda, gsax, win):
        return pd.DataFrame({
            "game_id": g["game_id"], "start_utc": g["start_utc"], "season": g["season"],
            "team": g[team], "is_home": team == "home",
            "gd": g[gf] - g[ga],
            "xgd": g[xgf] - g[xga],
            "xg_share": g[xgf] / (g[xgf] + g[xga]),
            "hd_share": g[hdf] / (g[hdf] + g[hda]),
            "gsax": g[gsax],
            "win": win,
        })
    home = side("home", "away", "home_score", "away_score", "xg_for", "xg_against",
                "hdcf_for", "hdcf_against", "gsax_for", g["home_win"])
    away = side("away", "home", "away_score", "home_score", "xg_against", "xg_for",
                "hdcf_against", "hdcf_for", "gsax_against", 1.0 - g["home_win"])
    return pd.concat([home, away], ignore_index=True).sort_values(["team", "start_utc", "game_id"])


def build_features(g: pd.DataFrame) -> pd.DataFrame:
    """Return one row per game with FEATURES plus the identifying columns."""
    g = g.sort_values(["start_utc", "game_id"]).reset_index(drop=True)
    g = add_elo(g)

    long = _team_long(g)
    by_team = long.groupby("team", sort=False)
    for w in WINDOWS:
        for s in TEAM_STATS:
            # shift(1) keeps the current game out of its own average
            long[f"{s}_{w}"] = by_team[s].transform(
                lambda x, w=w: x.shift(1).rolling(w, min_periods=5).mean()
            )
    prev_start = by_team["start_utc"].shift(1)
    rest = (long["start_utc"] - prev_start).dt.total_seconds() / 86400.0
    new_season = by_team["season"].shift(1) != long["season"]
    long["rest"] = rest.where(~new_season, 7.0).clip(upper=7.0).fillna(7.0)
    long["b2b"] = (long["rest"] < 1.5).astype(int)
    long["season_game_no"] = long.groupby(["team", "season"]).cumcount()

    stat_cols = [f"{s}_{w}" for w in WINDOWS for s in TEAM_STATS] + ["rest", "b2b", "season_game_no"]
    h = long[long["is_home"]].set_index("game_id")[stat_cols].add_prefix("home_")
    a = long[~long["is_home"]].set_index("game_id")[stat_cols].add_prefix("away_")
    out = g.join(h, on="game_id").join(a, on="game_id")

    for w in WINDOWS:
        for s in TEAM_STATS:
            out[f"{s}_{w}_diff"] = out[f"home_{s}_{w}"] - out[f"away_{s}_{w}"]
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    out["playoff"] = (out["game_type"] == 3).astype(int)
    out["season_progress"] = ((out["home_season_game_no"] + out["away_season_game_no"]) / 164.0).clip(upper=1.0)
    return out
