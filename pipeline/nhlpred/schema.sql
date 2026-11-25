CREATE TABLE IF NOT EXISTS predictions (
  game_id            bigint PRIMARY KEY,
  season             integer NOT NULL,
  game_type          integer NOT NULL,           -- 2 regular season, 3 playoffs
  game_date          date NOT NULL,              -- local (Eastern) date of the game
  start_utc          timestamptz NOT NULL,
  home               text NOT NULL,
  away               text NOT NULL,
  p_home             double precision NOT NULL,  -- model P(home wins), OT/SO included
  elo_diff           double precision,
  model_version      text NOT NULL,
  source             text NOT NULL,              -- 'live' or 'backtest'
  predicted_at       timestamptz NOT NULL,
  market_p_home      double precision,           -- vig-free two-way moneyline
  market_quoted_utc  timestamptz,
  home_score         integer,
  away_score         integer,
  overtime           boolean,
  home_win           boolean,
  settled_at         timestamptz
);
CREATE INDEX IF NOT EXISTS predictions_start_idx ON predictions (start_utc);
CREATE INDEX IF NOT EXISTS predictions_date_idx ON predictions (game_date);

CREATE TABLE IF NOT EXISTS model_runs (
  id             bigserial PRIMARY KEY,
  run_at         timestamptz NOT NULL DEFAULT now(),
  model_version  text NOT NULL,
  n_train        integer NOT NULL,
  trained_through date NOT NULL,
  n_predicted    integer NOT NULL,
  n_settled      integer NOT NULL,
  coefficients   jsonb NOT NULL
);
