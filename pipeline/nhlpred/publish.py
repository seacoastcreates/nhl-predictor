"""Push the day's fitted model and a model card to the Hugging Face Hub.

Run: python -m nhlpred.publish   (needs HF_TOKEN; exits quietly without it)
"""
from __future__ import annotations

import os

import joblib
from huggingface_hub import HfApi

from .daily import ARTIFACTS

REPO = os.environ.get("HF_MODEL_REPO", "36Dunes/nhl-win-probability")

CARD = """---
license: mit
library_name: scikit-learn
tags: [tabular-classification, sports, nhl, hockey]
datasets: [jjerlee/nhl-analytics]
---
# NHL win probability ({version})

Regularised logistic regression that returns the probability the home team wins an
NHL game (overtime and shootout included). Retrained every morning; this file was
fitted on {n_train:,} games through {trained_through}.

- Live predictions and track record: https://nhl.36dunes.com
- Code: https://github.com/seacoastcreates/nhl-predictor
- Walk-forward backtest (2017-18 onward, 12,244 games): log loss 0.6658, accuracy 58.9%.
  On games with a pre-game moneyline the market scores 0.6691 against the model's 0.6736.

`model.joblib` is a dict with `model` (a scikit-learn pipeline), `features` (column
order), `version`, `trained_through` and `n_train`. It is a pickle: load it only if you
trust this repository.

Features: {features}
"""


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; skipping publish")
        return
    art = joblib.load(ARTIFACTS / "model.joblib")
    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(ARTIFACTS / "model.joblib"), path_in_repo="model.joblib", repo_id=REPO,
                    commit_message=f"Model trained through {art['trained_through']}")
    card = CARD.format(version=art["version"], n_train=art["n_train"], trained_through=art["trained_through"],
                       features=", ".join(f"`{f}`" for f in art["features"]))
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=REPO,
                    commit_message="Update model card")
    print("published", REPO)


if __name__ == "__main__":
    main()
