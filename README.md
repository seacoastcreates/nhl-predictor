# Open Net

NHL win probabilities, published before every game and scored in public.

A model retrains each morning on every NHL game since 2013, predicts the day's slate,
and is graded against the final score and the betting market. Live site:
https://nhl.36dunes.com

## How it fits together

```
GitHub Actions (4 runs a day)            Neon Postgres            Next.js on Vercel
  download dataset + NHL schedule   ->   predictions table   ->   slate, track record,
  settle, retrain, predict, odds         model_runs table         calibration, backtest
  push model to Hugging Face
```

- `pipeline/nhlpred/data.py` loads the [jjerlee/nhl-analytics](https://huggingface.co/datasets/jjerlee/nhl-analytics)
  dataset (MIT), the NHL schedule API and the pre-game moneyline.
- `pipeline/nhlpred/features.py` builds pre-game features: margin-adjusted Elo, rolling
  10- and 40-game form (goal difference, expected-goal share, high-danger share,
  goaltending above expected), rest and back-to-backs.
- `pipeline/nhlpred/backtest.py` runs the walk-forward evaluation.
- `pipeline/nhlpred/daily.py` is the scheduled job. `db.py` holds the write rules.
- `web/` is the site.

## Results

Walk-forward backtest, 2017-18 onward: each season predicted by a model trained only
on earlier seasons (12,244 games).

| | Log loss | Accuracy |
|---|---|---|
| Always the home-win rate | 0.6905 | 53.7% |
| Elo only | 0.6702 | 58.6% |
| Gradient boosting (LightGBM) | 0.6684 | 58.8% |
| **Logistic regression (shipped)** | **0.6658** | **58.9%** |

On the 2,705 games with a pre-game moneyline, the market scores 0.6691 and the model
0.6736. The market is sharper, as expected; the site tracks the gap going forward.

## Things the data taught me

- About one in six odds rows in the dataset was captured after puck drop. Those
  in-game lines "predict" at 79% accuracy, so they are filtered out, and the daily job
  reads the odds feed directly to capture prices before the game.
- The `shots` and `hits` columns in the team box scores are not usable (they average
  about 3 a game), so the model relies on goals and the expected-goal measures.
- The dataset publishes once a day, so last night's scores are filled in from the NHL
  API to keep ratings and rest days current.

## Guarantees

- A prediction can change until the scheduled start. After that the upsert's `WHERE`
  clause refuses the write.
- Backtest rows are labelled `backtest` and never count toward the live record.
- `pipeline/tests/test_no_leakage.py` checks that a game's features do not move when
  that game's own result is changed.

## Run it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e pipeline pytest
.venv/bin/python -m nhlpred.backtest                 # prints the table above
DATABASE_URL=postgresql://localhost/nhlpred_dev .venv/bin/python -m nhlpred.daily --seed
cd web && echo 'DATABASE_URL=postgresql://localhost/nhlpred_dev' > .env.local && npm run dev
```

Not betting advice.
