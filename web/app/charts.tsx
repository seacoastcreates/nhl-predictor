"use client";

import { useState } from "react";
import type { CalRow, SeasonRow } from "@/lib/data";

const W = 520;
const H = 300;
const M = { top: 16, right: 76, bottom: 34, left: 44 };
const IW = W - M.left - M.right;
const IH = H - M.top - M.bottom;

const pct = (v: number) => `${Math.round(v * 100)}%`;
const seasonLabel = (s: number) => `${String(s).slice(2, 4)}-${String(s).slice(6, 8)}`;

function Tip({ x, y, children }: { x: number; y: number; children: React.ReactNode }) {
  const left = (x / W) * 100;
  return (
    <div
      className="pointer-events-none absolute z-10 w-max max-w-[220px] rounded-md border border-line bg-surface px-2.5 py-1.5 text-xs shadow-md"
      style={{
        left: `${left}%`,
        top: `${(y / H) * 100}%`,
        transform: `translate(${left > 60 ? "calc(-100% - 10px)" : "10px"}, -50%)`,
      }}
    >
      {children}
    </div>
  );
}

function Key({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <svg width="16" height="8" aria-hidden>
        <line x1="0" y1="4" x2="16" y2="4" stroke={color} strokeWidth="2" strokeDasharray={dashed ? "3 3" : undefined} />
      </svg>
      {label}
    </span>
  );
}

export function CalibrationChart({ rows }: { rows: CalRow[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const lo = 0.25;
  const hi = 0.8;
  const sx = (v: number) => M.left + ((v - lo) / (hi - lo)) * (W - M.left - 20);
  const sy = (v: number) => M.top + IH - ((v - lo) / (hi - lo)) * IH;
  const ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
  const maxN = Math.max(...rows.map((r) => r.n));
  const h = hover === null ? null : rows[hover];

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
        aria-label="Calibration: predicted home win probability against how often the home team actually won">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={sx(hi)} y1={sy(t)} y2={sy(t)} stroke="var(--line)" />
            <text x={M.left - 8} y={sy(t) + 4} textAnchor="end" fontSize="11" fill="var(--text-secondary)">{pct(t)}</text>
            <text x={sx(t)} y={H - 12} textAnchor="middle" fontSize="11" fill="var(--text-secondary)">{pct(t)}</text>
          </g>
        ))}
        <line x1={sx(lo)} y1={sy(lo)} x2={sx(hi)} y2={sy(hi)} stroke="var(--muted-mark)" strokeWidth="2" strokeDasharray="3 3" />
        {rows.map((r, i) => (
          <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
            <circle cx={sx(r.predicted)} cy={sy(r.actual)} r="14" fill="transparent" />
            <circle cx={sx(r.predicted)} cy={sy(r.actual)} r={4 + 5 * Math.sqrt(r.n / maxN)}
              fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth="2" opacity={hover === null || hover === i ? 1 : 0.45} />
          </g>
        ))}
      </svg>
      {h && (
        <Tip x={sx(h.predicted)} y={sy(h.actual)}>
          <div className="font-medium">Model said {pct(h.predicted)}</div>
          <div className="text-ink-2">Home team won {pct(h.actual)} of {h.n.toLocaleString()} games</div>
        </Tip>
      )}
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
        <Key color="var(--muted-mark)" label="Perfect calibration" dashed />
        <span>Dot size shows number of games. Axes: predicted (across), actual (up).</span>
      </div>
    </div>
  );
}

export function SeasonChart({ rows }: { rows: SeasonRow[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const vals = rows.flatMap((r) => [r.model, r.baseline, ...(r.market === null ? [] : [r.market])]);
  const lo = Math.floor(Math.min(...vals) * 100 - 0.5) / 100;
  const hi = Math.ceil(Math.max(...vals) * 100 + 0.5) / 100;
  const step = IW / Math.max(rows.length - 1, 1);
  const sx = (i: number) => M.left + i * step;
  const sy = (v: number) => M.top + ((hi - v) / (hi - lo)) * IH;
  const ticks: number[] = [];
  for (let t = lo; t <= hi + 1e-9; t += 0.01) ticks.push(Math.round(t * 100) / 100);
  const path = (get: (r: SeasonRow) => number | null) =>
    rows.map((r, i) => [i, get(r)] as const).filter(([, v]) => v !== null)
      .map(([i, v], k) => `${k ? "L" : "M"}${sx(i)},${sy(v as number)}`).join(" ");
  const last = rows.length - 1;
  const lastMarket = rows[last]?.market;
  const h = hover === null ? null : rows[hover];

  // keep the two end labels from overlapping
  let yModel = sy(rows[last].model);
  let yMarket = lastMarket == null ? null : sy(lastMarket);
  if (yMarket !== null && Math.abs(yModel - yMarket) < 13) {
    const mid = (yModel + yMarket) / 2;
    const modelAbove = yModel <= yMarket;
    yModel = mid + (modelAbove ? -7 : 7);
    yMarket = mid + (modelAbove ? 7 : -7);
  }

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img"
        aria-label="Log loss by season for the model, the betting market and a home-ice baseline"
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          const x = ((e.clientX - box.left) / box.width) * W;
          setHover(Math.max(0, Math.min(last, Math.round((x - M.left) / step))));
        }}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={M.left + IW} y1={sy(t)} y2={sy(t)} stroke="var(--line)" />
            <text x={M.left - 8} y={sy(t) + 4} textAnchor="end" fontSize="11" fill="var(--text-secondary)">{t.toFixed(2)}</text>
          </g>
        ))}
        {rows.map((r, i) => (
          <text key={r.season} x={sx(i)} y={H - 12} textAnchor="middle" fontSize="11" fill="var(--text-secondary)">
            {seasonLabel(r.season)}
          </text>
        ))}
        {h && <line x1={sx(hover!)} x2={sx(hover!)} y1={M.top} y2={M.top + IH} stroke="var(--text-secondary)" strokeWidth="1" />}
        <path d={path((r) => r.baseline)} fill="none" stroke="var(--muted-mark)" strokeWidth="2" strokeDasharray="3 3" />
        <path d={path((r) => r.market)} fill="none" stroke="var(--series-2)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        <path d={path((r) => r.model)} fill="none" stroke="var(--series-1)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        {rows.map((r, i) => (i === last || i === hover) && (
          <g key={r.season}>
            <circle cx={sx(i)} cy={sy(r.model)} r="4.5" fill="var(--series-1)" stroke="var(--surface-1)" strokeWidth="2" />
            {r.market !== null && <circle cx={sx(i)} cy={sy(r.market)} r="4.5" fill="var(--series-2)" stroke="var(--surface-1)" strokeWidth="2" />}
          </g>
        ))}
        <text x={sx(last) + 10} y={yModel + 4} fontSize="11" fill="var(--text-primary)">Model</text>
        {yMarket !== null && <text x={sx(last) + 10} y={yMarket + 4} fontSize="11" fill="var(--text-primary)">Market</text>}
      </svg>
      {h && (
        <Tip x={sx(hover!)} y={M.top + IH / 2}>
          <div className="font-medium">{seasonLabel(h.season)} season · {h.n.toLocaleString()} games</div>
          <div className="tnum mt-1 grid grid-cols-[auto_auto] gap-x-3 text-ink-2">
            <span>Model</span><span className="text-right text-ink">{h.model.toFixed(4)}</span>
            {h.market !== null && (<><span>Market</span><span className="text-right text-ink">{h.market.toFixed(4)}</span></>)}
            <span>Home-ice baseline</span><span className="text-right text-ink">{h.baseline.toFixed(4)}</span>
          </div>
        </Tip>
      )}
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
        <Key color="var(--series-1)" label="Model" />
        <Key color="var(--series-2)" label="Betting market" />
        <Key color="var(--muted-mark)" label="Home-ice baseline (always 54%)" dashed />
      </div>
    </div>
  );
}
