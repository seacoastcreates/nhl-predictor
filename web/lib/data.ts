import postgres from "postgres";
import { cacheLife } from "next/cache";

const url = process.env.DATABASE_URL ?? "";
const local = /localhost|127\.0\.0\.1/.test(url);
const sql = postgres(url, { ssl: local ? false : "require", max: 3, prepare: false, idle_timeout: 20 });

export type Game = {
  game_id: string;
  game_date: string;
  time_et: string;
  home: string;
  away: string;
  p_home: number;
  market_p_home: number | null;
  home_score: number | null;
  away_score: number | null;
  overtime: boolean | null;
  home_win: boolean | null;
  started: boolean;
  source: "live" | "backtest";
};

export type Record = {
  n: number;
  accuracy: number | null;
  log_loss: number | null;
  n_market: number;
  model_ll_mkt: number | null;
  market_ll: number | null;
  market_acc: number | null;
  model_acc_mkt: number | null;
};

export type SeasonRow = { season: number; n: number; model: number; baseline: number; market: number | null };
export type CalRow = { predicted: number; actual: number; n: number };

const GAME_COLS = sql`
  game_id::text, to_char(game_date, 'YYYY-MM-DD') AS game_date,
  to_char(start_utc AT TIME ZONE 'America/New_York', 'FMHH12:MI AM') AS time_et,
  home, away, p_home, market_p_home, home_score, away_score, overtime, home_win,
  start_utc <= now() AS started, source`;

// Log loss of the home-win probability column `p` against the result.
const ll = (p: ReturnType<typeof sql>) =>
  sql`-avg(CASE WHEN home_win THEN ln(${p}) ELSE ln(1 - ${p}) END)`;

async function record(source: "live" | "backtest"): Promise<Record> {
  const [r] = await sql<Record[]>`
    SELECT count(*)::int AS n,
           avg(((p_home > 0.5) = home_win)::int)::float AS accuracy,
           ${ll(sql`p_home`)}::float AS log_loss,
           count(market_p_home)::int AS n_market,
           (${ll(sql`p_home`)} FILTER (WHERE market_p_home IS NOT NULL))::float AS model_ll_mkt,
           (${ll(sql`market_p_home`)} FILTER (WHERE market_p_home IS NOT NULL))::float AS market_ll,
           (avg(((market_p_home > 0.5) = home_win)::int) FILTER (WHERE market_p_home IS NOT NULL))::float AS market_acc,
           (avg(((p_home > 0.5) = home_win)::int) FILTER (WHERE market_p_home IS NOT NULL))::float AS model_acc_mkt
    FROM predictions WHERE home_win IS NOT NULL AND source = ${source}`;
  return r;
}

export async function getDashboard() {
  "use cache";
  cacheLife("minutes");

  const [slate, recent, live, backtest, seasons, calibration, runs] = await Promise.all([
    sql<Game[]>`
      WITH d AS (
        SELECT coalesce(
          (SELECT min(game_date) FROM predictions WHERE source = 'live' AND home_win IS NULL),
          (SELECT max(game_date) FROM predictions WHERE source = 'live')) AS day)
      SELECT ${GAME_COLS} FROM predictions, d
      WHERE source = 'live' AND game_date = d.day ORDER BY start_utc, game_id`,
    sql<Game[]>`
      SELECT ${GAME_COLS} FROM predictions WHERE home_win IS NOT NULL
      ORDER BY start_utc DESC, game_id DESC LIMIT 12`,
    record("live"),
    record("backtest"),
    sql<SeasonRow[]>`
      SELECT season, count(*)::int AS n,
             ${ll(sql`p_home`)}::float AS model,
             ${ll(sql`0.54`)}::float AS baseline,
             CASE WHEN count(market_p_home) >= 100
               THEN (${ll(sql`market_p_home`)} FILTER (WHERE market_p_home IS NOT NULL))::float END AS market
      FROM predictions WHERE home_win IS NOT NULL GROUP BY season ORDER BY season`,
    sql<CalRow[]>`
      SELECT avg(p_home)::float AS predicted, avg(home_win::int)::float AS actual, count(*)::int AS n
      FROM predictions WHERE home_win IS NOT NULL
      GROUP BY floor(p_home * 20) HAVING count(*) >= 40 ORDER BY 1`,
    sql<{ run_at: string; n_train: number; trained_through: string }[]>`
      SELECT to_char(run_at AT TIME ZONE 'America/New_York', 'Mon FMDD, FMHH12:MI AM') AS run_at,
             n_train, to_char(trained_through, 'Mon FMDD, YYYY') AS trained_through
      FROM model_runs ORDER BY id DESC LIMIT 1`,
  ]);

  return { slate: [...slate], recent: [...recent], live, backtest, seasons: [...seasons],
           calibration: [...calibration], run: runs[0] ?? null };
}
