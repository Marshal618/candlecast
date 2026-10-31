import { useState } from 'react'
import {
  api,
  type Challenger,
  type CrossSectionResponse,
  type CrossSectionSignal,
  type Fundamental,
  type IcStats,
  type KronosName,
  type ModelName,
} from './api'
import { LongShortChart } from './charts'
import { FUNDAMENTAL_HINTS, MODEL_COLORS, signalColor, signalLabel } from './theme'
import { pct, rate } from './format'
import { RunNote, Slider, Stat } from './ui'
import { estimate, runningLabel, UNIVERSES, useProgress, useUniverses } from './universes'

const CHALLENGERS: Challenger[] = ['naive', 'garch', 'chronos']
const FUNDAMENTAL_GROUPS: [string, Fundamental[]][] = [
  ['earnings & shares (yahoo, point-in-time)', ['surprise', 'revisions', 'value', 'size', 'issuance']],
  ['sec filings (point-in-time from filing dates)', ['profitability', 'investment', 'accruals', 'book']],
]

const label = (s: CrossSectionSignal) => signalLabel(s.name)
const color = (s: CrossSectionSignal) => signalColor(s.name)
const num = (x: number, d = 3) => (x >= 0 ? '+' : '') + x.toFixed(d)
// |t| > 2 is the conventional "unlikely to be luck" bar for a mean IC
const tone = (s: IcStats) => (s.t_stat >= 2 ? 'up' : s.t_stat <= -2 ? 'down' : '')
const bps = (x: number | null) => (x == null ? 'no edge' : x >= 1000 ? '>999 bps' : `${x.toFixed(0)} bps`)
const SECTOR_SHORT: Record<string, string> = {
  'Communication Services': 'comms',
  'Consumer Discretionary': 'discr',
  'Consumer Staples': 'staples',
  Energy: 'energy',
  Financials: 'fin',
  'Health Care': 'health',
  Industrials: 'indus',
  'Information Technology': 'tech',
  Materials: 'mater',
  'Real Estate': 'reits',
  Utilities: 'utils',
  Unclassified: 'other',
}
const short = (sector: string) => SECTOR_SHORT[sector] ?? sector.slice(0, 6).toLowerCase()

export function ResearchView({ onError }: { onError: (msg: string | null) => void }) {
  const [symbols, setSymbols] = useState(UNIVERSES.sectors.join(', '))
  const [interval, setBarInterval] = useState<'1d' | '1h'>('1d')
  const [kronos, setKronos] = useState<KronosName | null>('mini')
  const [rivals, setRivals] = useState<Challenger[]>(['naive', 'garch', 'chronos'])
  const [funds, setFunds] = useState<Fundamental[]>([])
  const [horizon, setHorizon] = useState(5)
  const [periods, setPeriods] = useState(24)
  const [buckets, setBuckets] = useState(3)
  const [lookback, setLookback] = useState(64)
  const [samples, setSamples] = useState(8)
  const [cost, setCost] = useState('5')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<CrossSectionResponse | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const { presets, asOf } = useUniverses()
  const prog = useProgress(busy)

  const list = [...new Set(symbols.split(/[\s,]+/).map((s) => s.trim().toUpperCase()).filter(Boolean))]
  const models: ModelName[] = [...(kronos ? [kronos] : []), ...rivals]
  const toggleRival = (m: Challenger) => setRivals((cur) => (cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m]))
  const toggleFund = (f: Fundamental) => setFunds((cur) => (cur.includes(f) ? cur.filter((x) => x !== f) : [...cur, f]))

  const run = async () => {
    setBusy(true)
    onError(null)
    try {
      const r = await api.crosssection({
        symbols: list,
        interval,
        models,
        lookback,
        horizon,
        periods,
        buckets: Math.min(buckets, list.length),
        samples,
        cost_bps: Number(cost) || 0,
        fundamentals: funds,
      })
      setResult(r)
      setSelected(r.signals[0]?.name ?? null)
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const sel = result?.signals.find((s) => s.name === selected) ?? result?.signals[0]

  return (
    <>
      <div className="sim-setup">
        <div className="panel-title">Cross-sectional research</div>
        <p className="muted small">
          every {horizon} bars, each model forecasts every symbol and the universe is ranked by expected return. the
          question isn't "will it go up" but "does the ranking predict which ones beat the others", scored as RankIC and
          as a long-top / short-bottom portfolio after costs, next to two classic price factors.
        </p>
        <div className="setup-grid">
          <label>
            universe ({list.length} symbols, 4–700)
            <input className="field" value={symbols} onChange={(e) => setSymbols(e.target.value)} spellCheck={false} />
            <div className="quick">
              {Object.entries(presets).map(([name, u]) => (
                <button key={name} onClick={() => setSymbols(u.join(', '))}>
                  {name}
                </button>
              ))}
            </div>
            {list.length > 30 && asOf && (
              <span className="muted small">
                index lists are today's members (as of {asOf}): names that fell out or were delisted are missing, which flatters backtests
              </span>
            )}
          </label>
          <label>
            bars
            <div className="segmented">
              {(['1d', '1h'] as const).map((i) => (
                <button key={i} className={interval === i ? 'active' : ''} onClick={() => setBarInterval(i)}>
                  {i}
                </button>
              ))}
            </div>
          </label>
          <label>
            kronos
            <div className="segmented">
              {([null, 'mini', 'small', 'base'] as (KronosName | null)[]).map((m) => (
                <button key={m ?? 'off'} className={kronos === m ? 'active' : ''} onClick={() => setKronos(m)}>
                  {m ?? 'off'}
                </button>
              ))}
            </div>
          </label>
          <label>
            challengers
            <div className="rivals">
              {CHALLENGERS.map((m) => (
                <button key={m} className={rivals.includes(m) ? 'active' : ''} onClick={() => toggleRival(m)}>
                  <i className="dot" style={{ background: MODEL_COLORS[m] }} />
                  {m}
                </button>
              ))}
            </div>
          </label>
          {FUNDAMENTAL_GROUPS.map(([title, group]) => (
            <label key={title}>
              {title}
              <div className="rivals">
                {group.map((f) => (
                  <button key={f} className={funds.includes(f) ? 'active' : ''} onClick={() => toggleFund(f)} title={FUNDAMENTAL_HINTS[f]}>
                    <i className="dot" style={{ background: signalColor(f) }} />
                    {signalLabel(f)}
                  </button>
                ))}
              </div>
            </label>
          ))}
          <Slider label="rebalance every (bars)" value={horizon} min={1} max={60} onChange={setHorizon} />
          <Slider label="rebalances" value={periods} min={4} max={60} onChange={setPeriods} />
          <Slider label="buckets" value={Math.min(buckets, Math.max(list.length, 2))} min={2} max={Math.max(2, Math.min(10, list.length))} onChange={setBuckets} />
          <Slider label="lookback" value={lookback} min={32} max={512} onChange={setLookback} />
          <Slider label="paths per forecast" value={samples} min={1} max={30} onChange={setSamples} />
          <label>
            cost per unit turnover (bps)
            <input className="field" inputMode="decimal" value={cost} onChange={(e) => setCost(e.target.value)} />
          </label>
        </div>
        <button className="primary start" disabled={busy || list.length < 4 || models.length === 0} onClick={run}>
          {busy ? runningLabel(prog, 'running') : '▶ Run research'}
        </button>
        {!busy && estimate(periods * list.length, samples, horizon, kronos != null) && (
          <span className="muted small"> {estimate(periods * list.length, samples, horizon, kronos != null)}</span>
        )}
      </div>

      {result && sel && (
        <>
          <div className="stats">
            <Stat label="rebalances" value={String(result.periods)} hint={`every ${result.horizon} bars`} />
            <Stat label="symbols" value={String(result.symbols.length)} hint={`${result.buckets} buckets`} />
            <Stat label="dispersion" value={pct(result.dispersion, 2).slice(1)} hint="avg cross-sectional σ of returns" />
            <Stat label="universe / period" value={pct(result.universe.mean, 2)} hint="equal-weight, dotted line" />
            <Stat label="runtime" value={`${(result.latency_ms / 1000).toFixed(0)}s`} />
          </div>
          <RunNote dropped={result.dropped} unavailable={result.unavailable} />

          <div className="panel-title">Signals</div>
          <table className="tok-table league research">
            <thead>
              <tr>
                <th>signal</th>
                <th>rank ic</th>
                <th>t</th>
                <th>ic &gt; 0</th>
                <th>beyond factors</th>
                {result.sectors && <th>within sectors</th>}
                <th>l/s net / period</th>
                <th>l/s sharpe</th>
                <th>max dd</th>
                <th>turnover</th>
                <th>break-even</th>
                <th>ρ reversal</th>
                <th>ρ momentum</th>
              </tr>
            </thead>
            <tbody>
              {result.signals.map((s) => (
                <tr key={s.name} className={s.name === sel.name ? 'me' : ''} onClick={() => setSelected(s.name)}>
                  <td>
                    <i className="dot" style={{ background: color(s) }} /> {label(s)}
                    {s.coverage != null && (
                      <span className="muted" title="share of (rebalance, symbol) cells with real data; gaps sit mid-ranking">
                        {' '}· {rate(s.coverage)} covered
                      </span>
                    )}
                  </td>
                  <td className={tone(s.rank_ic)}>{num(s.rank_ic.mean)}</td>
                  <td className={tone(s.rank_ic)}>{num(s.rank_ic.t_stat, 2)}</td>
                  <td>{rate(s.rank_ic.positive)}</td>
                  <td className={s.rank_ic_residual ? tone(s.rank_ic_residual) : ''}>
                    {s.rank_ic_residual ? `${num(s.rank_ic_residual.mean)} (t ${num(s.rank_ic_residual.t_stat, 1)})` : '—'}
                  </td>
                  {result.sectors && (
                    <td className={s.sector ? tone(s.sector.rank_ic) : ''}>
                      {s.sector ? `${num(s.sector.rank_ic.mean)} (t ${num(s.sector.rank_ic.t_stat, 1)})` : '—'}
                    </td>
                  )}
                  <td className={s.long_short.net_mean > 0 ? 'up' : 'down'}>{pct(s.long_short.net_mean, 2)}</td>
                  <td>{s.long_short.sharpe.toFixed(2)}</td>
                  <td>{pct(s.long_short.max_drawdown, 1)}</td>
                  <td>{rate(s.long_short.turnover)}</td>
                  <td className={s.long_short.breakeven_bps != null && s.long_short.breakeven_bps < Number(cost) * 2 ? 'down' : ''}>
                    {bps(s.long_short.breakeven_bps)}
                  </td>
                  <td>{s.corr.reversal != null ? num(s.corr.reversal, 2) : '—'}</td>
                  <td>{s.corr.momentum != null ? num(s.corr.momentum, 2) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">
            rank ic = spearman correlation between a signal's ranking and the realized next-{result.horizon}-bar returns,
            averaged over rebalances; green / red when |t| &gt; 2. "beyond factors" scores the ranking left after regressing
            out momentum and reversal.{result.sectors && ' "within sectors" ranks signal and returns inside each GICS sector, so a sector bet scores zero.'}{' '}
            break-even = cost per unit traded that would eat the whole gross edge (red when under twice your cost). fundamentals (dotted) only use data public by each rebalance; funds and crypto
            have none. l/s = long the top bucket, short the bottom, equal weight, net of costs. ρ = average
            rank correlation with each factor. click a row for its buckets.
          </p>

          <div className="panel-title">Long-short, cumulative net</div>
          <LongShortChart result={result} selected={sel.name} />
          <div className="legend">
            {result.signals.map((s) => (
              <span key={s.name}>
                <i className="dot" style={{ background: color(s) }} /> {label(s)}
              </span>
            ))}
            <span className="muted">··· equal-weight universe</span>
          </div>

          <div className="token-grid">
            <div>
              <div className="panel-title">Buckets: {label(sel)}</div>
              <Bars
                values={sel.bucket_returns}
                labels={sel.bucket_returns.map((_, i) => (i === 0 ? 'low' : i === sel.bucket_returns.length - 1 ? 'high' : `q${i + 1}`))}
                format={(v) => pct(v, 2)}
              />
              <p className="muted small">
                average next-{result.horizon}-bar return of each bucket, lowest score to highest. a real signal climbs left
                to right.
              </p>
            </div>
            <div>
              <div className="panel-title">Rank ic per rebalance</div>
              <Bars values={sel.rank_ic_series.map((v) => v ?? 0)} format={(v) => v.toFixed(2)} />
              <p className="muted small">
                one bar per rebalance, oldest first. ic {rate(sel.rank_ic.positive)} of the time above zero, icir{' '}
                {sel.rank_ic.icir.toFixed(2)}.
              </p>
            </div>
          </div>

          <div className="token-grid">
            <div>
              <div className="panel-title">IC decay: {label(sel)}</div>
              <Bars
                values={sel.ic_decay}
                labels={sel.ic_decay.map((_, k) => (k === 0 ? 'next' : `+${k * result.horizon} bars`))}
                format={(v) => num(v)}
              />
              <p className="muted small">
                each ranking scored against the next period's returns, then the ones after. an ic that lasts can be
                traded less often for the same edge. rank autocorrelation {num(sel.rank_autocorr, 2)}: how much of each
                ranking carries over to the next (1 = never changes).
              </p>
            </div>
            {sel.sector && (
              <div>
                <div className="panel-title">Sector exposure: {label(sel)}</div>
                <Bars
                  values={Object.values(sel.sector.exposure)}
                  labels={Object.keys(sel.sector.exposure).map(short)}
                  format={(v) => pct(v, 0)}
                />
                <p className="muted small">
                  average net weight of the long-short in each sector (long side sums to +100%, short to −100%). bars
                  near zero mean it picks stocks; tall bars mean it's partly a sector bet. sector-neutral l/s: sharpe{' '}
                  {sel.sector.long_short.sharpe.toFixed(2)}, {pct(sel.sector.long_short.net_mean, 2)} / period vs{' '}
                  {sel.long_short.sharpe.toFixed(2)}, {pct(sel.long_short.net_mean, 2)} plain.
                </p>
              </div>
            )}
          </div>
          {sel.sector && Object.keys(sel.sector.by_sector).length > 0 && (
            <>
              <div className="panel-title">Rank ic inside each sector: {label(sel)}</div>
              <Bars
                values={Object.values(sel.sector.by_sector).map((x) => x.mean)}
                labels={Object.entries(sel.sector.by_sector).map(([k, x]) => `${short(k)} (${x.n})`)}
                format={(v) => num(v)}
              />
              <p className="muted small">
                the signal ranked only against stocks in the same sector (sectors with 5+ members; count in brackets).
                one sector carrying the whole ic is fragile.
              </p>
            </>
          )}
        </>
      )}
      {!result && !busy && (
        <div className="empty">
          // no research run yet. sector ETFs at 5-bar rebalances take about 20s for all four models.
        </div>
      )}
    </>
  )
}

// Zero-centred bar strip: up bars above the axis, down bars below
function Bars({ values, labels, format }: { values: number[]; labels?: string[]; format: (v: number) => string }) {
  const max = Math.max(...values.map(Math.abs), 1e-9)
  return (
    <div className="bars">
      {values.map((v, i) => (
        <div key={i} className="bar-col" title={`${labels?.[i] ?? `#${i + 1}`}: ${format(v)}`}>
          <div className="bar-half">{v > 0 && <div className="bar up" style={{ height: `${(v / max) * 100}%` }} />}</div>
          <div className="bar-half">{v < 0 && <div className="bar down" style={{ height: `${(-v / max) * 100}%` }} />}</div>
          {labels && (
            <div className="bar-label">
              {labels[i]}
              <br />
              {format(v)}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
