import { Suspense } from "react";
import { getDashboard, type Game, type Record } from "@/lib/data";
import { CalibrationChart, SeasonChart } from "./charts";

const REPO = "https://github.com/seacoastcreates/nhl-predictor";
const DATASET = "https://huggingface.co/datasets/jjerlee/nhl-analytics";

const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;
const longDate = (iso: string) =>
  new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { timeZone: "UTC", weekday: "long", month: "long", day: "numeric" });
const shortDate = (iso: string) =>
  new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { timeZone: "UTC", month: "short", day: "numeric" });

function Verdict({ g }: { g: Game }) {
  if (g.home_win === null) return null;
  const hit = g.p_home > 0.5 === g.home_win;
  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium">
      <span aria-hidden className={`inline-grid size-4 place-items-center rounded-full text-[10px] text-white ${hit ? "bg-good" : "bg-critical"}`}>
        {hit ? "✓" : "✕"}
      </span>
      {hit ? "Correct" : "Missed"}
    </span>
  );
}

function GameCard({ g }: { g: Game }) {
  const pAway = 1 - g.p_home;
  const homePick = g.p_home > 0.5;
  const final = g.home_win !== null;
  const mktAway = g.market_p_home === null ? null : 1 - g.market_p_home;
  const mktFav = g.market_p_home === null ? null : g.market_p_home > 0.5 ? g.home : g.away;
  const mktP = g.market_p_home === null ? null : Math.max(g.market_p_home, 1 - g.market_p_home);

  const Team = ({ abbr, p, pick, score, align }: { abbr: string; p: number; pick: boolean; score: number | null; align: "left" | "right" }) => (
    <div className={align === "right" ? "text-right" : ""}>
      <div className="text-[11px] uppercase tracking-wide text-ink-2">{align === "left" ? "Away" : "Home"}</div>
      <div className={`flex items-baseline gap-2 ${align === "right" ? "flex-row-reverse" : ""}`}>
        <span className="text-xl font-semibold tracking-tight">{abbr}</span>
        {score !== null && <span className="tnum text-xl text-ink-2">{score}</span>}
      </div>
      <div className={`tnum text-sm ${pick ? "font-semibold" : "text-ink-2"}`}>{pct(p)}</div>
    </div>
  );

  return (
    <article className="rounded-xl border border-line bg-surface p-4">
      <div className="mb-3 flex items-center justify-between text-xs text-ink-2">
        <span>{final ? `Final${g.overtime ? " (OT/SO)" : ""}` : g.started ? "In progress" : `${g.time_et} ET`}</span>
        <Verdict g={g} />
      </div>
      <div className="flex items-end justify-between">
        <Team abbr={g.away} p={pAway} pick={!homePick} score={g.away_score} align="left" />
        <Team abbr={g.home} p={g.p_home} pick={homePick} score={g.home_score} align="right" />
      </div>
      <div className="relative mt-3 pt-2">
        <div className="flex h-2.5 gap-0.5" title={`Model: ${g.away} ${pct(pAway)}, ${g.home} ${pct(g.p_home)}`}>
          <div className={`rounded-l-full ${homePick ? "bg-track" : "bg-model"}`} style={{ width: pct(pAway, 1) }} />
          <div className={`flex-1 rounded-r-full ${homePick ? "bg-model" : "bg-track"}`} />
        </div>
        {mktAway !== null && (
          <div className="absolute top-0 h-[18px] w-0.5 -translate-x-1/2 rounded bg-market" style={{ left: pct(mktAway, 1) }} aria-hidden />
        )}
      </div>
      <div className="mt-2 flex items-center gap-1.5 text-xs text-ink-2">
        {mktFav ? (
          <><span aria-hidden className="inline-block h-3 w-0.5 rounded bg-market" />Market: {mktFav} {pct(mktP!)}</>
        ) : (
          <span>Market price not posted yet</span>
        )}
      </div>
    </article>
  );
}

function Tile({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <div className="text-xs text-ink-2">{label}</div>
      <div className="tnum mt-1 text-2xl font-semibold tracking-tight">{value}</div>
      <div className="mt-1 text-xs text-ink-2">{note}</div>
    </div>
  );
}

function LiveTiles({ r }: { r: Record }) {
  if (r.n === 0) {
    return (
      <div className="rounded-xl border border-dashed border-line p-4 text-sm text-ink-2">
        The live record starts with the first slate below. Results are graded the morning after each game.
      </div>
    );
  }
  const hits = Math.round((r.accuracy ?? 0) * r.n);
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      <Tile label="Live record" value={`${hits}-${r.n - hits}`} note={`${pct(r.accuracy ?? 0, 1)} of ${r.n} winners picked`} />
      <Tile label="Live log loss" value={(r.log_loss ?? 0).toFixed(4)} note="Lower is better. A coin flip scores 0.6931" />
      <Tile label="Market, same games" value={r.market_ll === null ? "n/a" : r.market_ll.toFixed(4)}
        note={r.n_market ? `Model ${r.model_ll_mkt!.toFixed(4)} on the ${r.n_market} games with a price` : "No pre-game prices captured yet"} />
    </div>
  );
}

function Section({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="mt-12">
      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      {sub && <p className="mt-1 max-w-2xl text-sm text-ink-2">{sub}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

async function Dashboard() {
  const d = await getDashboard();
  const b = d.backtest;
  const slateDone = d.slate.length > 0 && d.slate.every((g) => g.home_win !== null);

  return (
    <>
      <p className="mt-3 text-xs text-ink-2">
        {d.run ? `Model retrained ${d.run.run_at} ET on ${d.run.n_train.toLocaleString()} games through ${d.run.trained_through}.` : "Waiting for the first model run."}
      </p>

      <Section
        title={d.slate.length ? `${slateDone ? "Latest slate" : "Next slate"}: ${longDate(d.slate[0].game_date)}` : "Next slate"}
        sub="The blue side is the model's pick and the bar shows its win probability. The orange tick marks where the betting market puts the same game. Predictions lock at puck drop.">
        {d.slate.length ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {d.slate.map((g) => <GameCard key={g.game_id} g={g} />)}
          </div>
        ) : (
          <p className="text-sm text-ink-2">No games on the schedule in the next two days.</p>
        )}
      </Section>

      <Section title="Live track record" sub="Only predictions published here before the game count. Nothing is edited after the fact.">
        <LiveTiles r={d.live} />
      </Section>

      <Section
        title="Backtest"
        sub={`Before going live the model was tested the honest way: for each season since 2017-18 it was trained only on earlier seasons, then asked to predict that season. That is ${b.n.toLocaleString()} games it had never seen.`}>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Tile label="Winners picked" value={pct(b.accuracy ?? 0, 1)} note="Picking the home team every time gets 53.7%" />
          <Tile label="Log loss" value={(b.log_loss ?? 0).toFixed(4)} note="Coin flip 0.6931, home-ice baseline 0.6905" />
          <Tile label="Model vs market" value={`${b.model_ll_mkt?.toFixed(4)} vs ${b.market_ll?.toFixed(4)}`}
            note={`Log loss on the ${b.n_market.toLocaleString()} games with a pre-game price. The market is sharper.`} />
          <Tile label="Winners, same games" value={`${pct(b.model_acc_mkt ?? 0, 1)} vs ${pct(b.market_acc ?? 0, 1)}`} note="Model vs market favourite" />
        </div>
        <div className="mt-3 grid gap-3 lg:grid-cols-2">
          <div className="rounded-xl border border-line bg-surface p-4">
            <h3 className="text-sm font-semibold">When the model says 60%, does the home team win 60% of the time?</h3>
            <p className="mb-2 mt-0.5 text-xs text-ink-2">Dots on the dashed line mean the probabilities can be taken at face value.</p>
            <CalibrationChart rows={d.calibration} />
          </div>
          <div className="rounded-xl border border-line bg-surface p-4">
            <h3 className="text-sm font-semibold">Log loss by season (lower is better)</h3>
            <p className="mb-2 mt-0.5 text-xs text-ink-2">The model beats the naive baseline every season and trails the market by a small margin.</p>
            <SeasonChart rows={d.seasons} />
          </div>
        </div>
      </Section>

      <Section title="Most recent results">
        <div className="overflow-x-auto rounded-xl border border-line bg-surface">
          <table className="tnum w-full min-w-[560px] text-sm">
            <thead className="text-left text-xs text-ink-2">
              <tr className="border-b border-line">
                {["Date", "Game", "Score", "Model pick", "Market", "Result", "Source"].map((h) => (
                  <th key={h} className="px-4 py-2.5 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {d.recent.map((g) => {
                const homePick = g.p_home > 0.5;
                return (
                  <tr key={g.game_id} className="border-b border-line last:border-0">
                    <td className="px-4 py-2.5 text-ink-2">{shortDate(g.game_date)}</td>
                    <td className="px-4 py-2.5">{g.away} @ {g.home}</td>
                    <td className="px-4 py-2.5">{g.away_score}-{g.home_score}{g.overtime ? " OT" : ""}</td>
                    <td className="px-4 py-2.5">{homePick ? g.home : g.away} {pct(homePick ? g.p_home : 1 - g.p_home)}</td>
                    <td className="px-4 py-2.5 text-ink-2">
                      {g.market_p_home === null ? "n/a" : `${g.market_p_home > 0.5 ? g.home : g.away} ${pct(Math.max(g.market_p_home, 1 - g.market_p_home))}`}
                    </td>
                    <td className="px-4 py-2.5"><Verdict g={g} /></td>
                    <td className="px-4 py-2.5 text-ink-2">{g.source === "live" ? "Live" : "Backtest"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-5xl px-5 pb-20 pt-12">
      <header>
        <div className="text-sm font-semibold tracking-tight text-model">Open Net</div>
        <h1 className="mt-2 max-w-3xl text-3xl font-semibold tracking-tight sm:text-4xl">
          NHL win probabilities, published before every game and scored in public.
        </h1>
        <p className="mt-3 max-w-2xl text-ink-2">
          A model retrains each morning on every NHL game since 2013, predicts the day&apos;s slate, then gets graded against the
          final score and against the bookmakers.
        </p>
      </header>

      <Suspense fallback={<p className="mt-12 text-sm text-ink-2">Loading the latest predictions…</p>}>
        <Dashboard />
      </Suspense>

      <Section title="How it works">
        <div className="grid gap-x-10 gap-y-5 text-sm leading-6 text-ink-2 sm:grid-cols-2">
          <p>
            <span className="font-medium text-ink">Data.</span> Game results and shot-quality stats come from the open{" "}
            <a className="underline" href={DATASET}>jjerlee/nhl-analytics</a> dataset on Hugging Face, which refreshes daily. The
            schedule and overnight scores come from the NHL&apos;s public API.
          </p>
          <p>
            <span className="font-medium text-ink">Features.</span> A margin-adjusted Elo rating, each team&apos;s last 10 and 40
            games of goal difference, expected-goal share, high-danger chance share and goaltending above expected, plus rest days
            and back-to-backs. Every feature uses only games that started earlier.
          </p>
          <p>
            <span className="font-medium text-ink">Model.</span> Regularised logistic regression. A gradient-boosted model was
            tested on the same walk-forward splits and scored worse (0.6684 vs 0.6658 log loss), so the simpler model ships.
          </p>
          <p>
            <span className="font-medium text-ink">Honesty rules.</span> A prediction can update until the scheduled start, then
            the database refuses changes. Market prices only count if they were quoted before puck drop. Backtest rows are labelled
            and kept out of the live record.
          </p>
        </div>
        <p className="mt-6 text-sm text-ink-2">
          Hockey is the closest major sport to a coin flip, so a good model lands near 59% and nobody reliably beats the market.
          This is a forecasting project, not betting advice. <a className="underline" href={REPO}>Source on GitHub</a>.
        </p>
      </Section>
    </main>
  );
}
