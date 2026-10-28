import { useState } from 'react'
import {
  api,
  type Challenger,
  type CrossSectionResponse,
  type CrossSectionSignal,
  type Factor,
  type IcStats,
  type KronosName,
  type ModelName,
} from './api'
import { LongShortChart } from './charts'
import { FACTOR_COLORS, FACTOR_LABELS, MODEL_COLORS, MODEL_LABELS } from './theme'
import { pct, rate } from './format'
import { Slider, Stat } from './ui'
import { UNIVERSES } from './universes'

const CHALLENGERS: Challenger[] = ['naive', 'garch', 'chronos']

const isFactor = (s: CrossSectionSignal): s is CrossSectionSignal & { name: Factor } => s.kind === 'factor'
const label = (s: CrossSectionSignal) => (isFactor(s) ? FACTOR_LABELS[s.name] : MODEL_LABELS[s.name as ModelName])
const color = (s: CrossSectionSignal) => (isFactor(s) ? FACTOR_COLORS[s.name] : MODEL_COLORS[s.name as ModelName])
const num = (x: number, d = 3) => (x >= 0 ? '+' : '') + x.toFixed(d)
// |t| > 2 is the conventional "unlikely to be luck" bar for a mean IC
const tone = (s: IcStats) => (s.t_stat >= 2 ? 'up' : s.t_stat <= -2 ? 'down' : '')

export function ResearchView({ onError }: { onError: (msg: string | null) => void }) {
  const [symbols, setSymbols] = useState(UNIVERSES.sectors.join(', '))
  const [interval, setBarInterval] = useState<'1d' | '1h'>('1d')
  const [kronos, setKronos] = useState<KronosName | null>('mini')
  const [rivals, setRivals] = useState<Challenger[]>(['naive', 'garch', 'chronos'])
  const [horizon, setHorizon] = useState(5)
  const [periods, setPeriods] = useState(24)
  const [buckets, setBuckets] = useState(3)
  const [lookback, setLookback] = useState(64)
  const [samples, setSamples] = useState(8)
  const [cost, setCost] = useState('5')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<CrossSectionResponse | null>(null)
  const [selected, setSelected] = useState<string | null>(null)

  const list = [...new Set(symbols.split(/[\s,]+/).map((s) => s.trim().toUpperCase()).filter(Boolean))]
  const models: ModelName[] = [...(kronos ? [kronos] : []), ...rivals]
  const toggleRival = (m: Challenger) => setRivals((cur) => (cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m]))

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
            universe ({list.length} symbols, 4–30)
            <input className="field" value={symbols} onChange={(e) => setSymbols(e.target.value)} spellCheck={false} />
            <div className="quick">
              {Object.entries(UNIVERSES).map(([name, u]) => (
                <button key={name} onClick={() => setSymbols(u.join(', '))}>
                  {name}
                </button>
              ))}
            </div>
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
          {busy ? `▶ running ${periods * list.length * models.length} forecasts…` : '▶ Run research'}
        </button>
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

          <div className="panel-title">Signals</div>
          <table className="tok-table league research">
            <thead>
              <tr>
                <th>signal</th>
                <th>rank ic</th>
                <th>t</th>
                <th>ic &gt; 0</th>
                <th>beyond factors</th>
                <th>l/s net / period</th>
                <th>l/s sharpe</th>
                <th>max dd</th>
                <th>turnover</th>
                <th>ρ reversal</th>
                <th>ρ momentum</th>
              </tr>
            </thead>
            <tbody>
              {result.signals.map((s) => (
                <tr key={s.name} className={s.name === sel.name ? 'me' : ''} onClick={() => setSelected(s.name)}>
                  <td>
                    <i className="dot" style={{ background: color(s) }} /> {label(s)}
                  </td>
                  <td className={tone(s.rank_ic)}>{num(s.rank_ic.mean)}</td>
                  <td className={tone(s.rank_ic)}>{num(s.rank_ic.t_stat, 2)}</td>
                  <td>{rate(s.rank_ic.positive)}</td>
                  <td className={s.rank_ic_residual ? tone(s.rank_ic_residual) : ''}>
                    {s.rank_ic_residual ? `${num(s.rank_ic_residual.mean)} (t ${num(s.rank_ic_residual.t_stat, 1)})` : '—'}
                  </td>
                  <td className={s.long_short.net_mean > 0 ? 'up' : 'down'}>{pct(s.long_short.net_mean, 2)}</td>
                  <td>{s.long_short.sharpe.toFixed(2)}</td>
                  <td>{pct(s.long_short.max_drawdown, 1)}</td>
                  <td>{rate(s.long_short.turnover)}</td>
                  <td>{s.corr.reversal != null ? num(s.corr.reversal, 2) : '—'}</td>
                  <td>{s.corr.momentum != null ? num(s.corr.momentum, 2) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">
            rank ic = spearman correlation between a signal's ranking and the realized next-{result.horizon}-bar returns,
            averaged over rebalances; green / red when |t| &gt; 2. "beyond factors" scores the ranking left after regressing
            out momentum and reversal. l/s = long the top bucket, short the bottom, equal weight, net of costs. ρ = average
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
