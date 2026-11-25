"""Candidate models. Each returns P(home team wins), overtime and shootout included."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

MODEL_VERSION = "logit-v1"


def make_model(kind: str):
    if kind == "elo":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
    if kind == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
    if kind == "gbm":
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.01, num_leaves=4, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=10.0,
            random_state=7, verbose=-1,
        )
    raise ValueError(kind)


def columns(kind: str) -> list[str]:
    return ["elo_diff"] if kind == "elo" else list(FEATURES)


def fit(kind: str, train: pd.DataFrame):
    m = make_model(kind)
    m.fit(train[columns(kind)], train["home_win"].astype(int))
    return m


def predict(kind: str, model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[columns(kind)])[:, 1]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "accuracy": float(((p > 0.5) == (y == 1)).mean()),
    }
