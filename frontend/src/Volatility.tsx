import { useMemo, useState } from 'react'
import { LineStyle } from 'lightweight-charts'
import { api, type Challenger, type KronosName, type ModelName, type VolForecast, type VolatilityResponse } from './api'
import { VolChart, type VolLine } from './charts'
import { COLORS, MODEL_COLORS, MODEL_LABELS } from './theme'
import { pct } from './format'
import { RunNote, Slider, Stat } from './ui'
import { estimate, runningLabel, UNIVERSES, useProgress, useUniverses } from './universes'

const CHALLENGERS: Challenger[] = ['naive', 'garch', 'chronos']
const EWMA_COLOR = COLORS.benchmark

const label = (f: VolForecast) => (f.name === 'ewma' ? 'ewma λ=0.94 (baseline)' : MODEL_LABELS[f.name as ModelName])
const color = (f: VolForecast) => (f.name === 'ewma' ? EWMA_COLOR : MODEL_COLORS[f.name as ModelName])
const num = (x: number, d = 2) => (x >= 0 ? '+' : '') + x.toFixed(d)
// The baseline is the bar: green when a model beats it with |t| > 2, red when it loses as clearly
const tone = (f: VolForecast) => (f.kind === 'baseline' ? '' : f.dm_t >= 2 ? 'up' : f.dm_t <= -2 ? 'down' : '')

export function VolatilityView({ onError }: { onError: (msg: string | null) => void }) {
  const [symbols, setSymbols] = useState(UNIVERSES['mega caps'].join(', '))
  const [interval, setBarInterval] = useState<'1d' | '1h'>('1d')
  const [kronos, setKronos] = useState<KronosName | null>('mini')
  const [rivals, setRivals] = useState<Challenger[]>(['garch', 'chronos'])
  const [horizon, setHorizon] = useState(5)
  const [periods, setPeriods] = useState(40)
  const [lookback, setLookback] = useState(64)
  const [samples, setSamples] = useState(16)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<VolatilityResponse | null>(null)
  const [symbol, setSymbol] = useState(0)
  const [calibrated, setCalibrated] = useState(true)
  const { presets, asOf } = useUniverses()
  const prog = useProgress(busy)

  const list = [...new Set(symbols.split(/[\s,]+/).map((s) => s.trim().toUpperCase()).filter(Boolean))]
  const models: ModelName[] = [...(kronos ? [kronos] : []), ...rivals]
  const toggleRival = (m: Challenger) => setRivals((cur) => (cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m]))

  const run = async () => {
    setBusy(true)
    onError(null)
    try {
      setResult(await api.volatility({ symbols: list, interval, models, lookback, horizon, periods, samples }))
      setSymbol(0)
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const lines = useMemo<VolLine[]>(() => {
    if (!result) return []
    const col = (rows: (number | null)[][]) => rows.map((r) => r[symbol] ?? null)
    return [
      { values: col(result.realized), color: COLORS.actual, width: 2 },
      ...result.forecasts.map((f) => ({
        values: col(calibrated ? f.calibrated : f.raw),
        color: color(f),
        style: f.kind === 'baseline' ? LineStyle.Dashed : LineStyle.Solid,
      })),
    ]
  }, [result, symbol, calibrated])

  return (
    <>
      <div className="sim-setup">
        <div className="panel-title">Volatility test</div>
        <p className="muted small">
          direction is close to unpredictable from prices; the size of moves isn't, because volatility clusters. every{' '}
          {horizon} bars, each model's spread of sampled paths is read as a forecast of the next {horizon} bars' variance
          and scored against what was realized, next to an ewma baseline that needs no model at all.
        </p>
        <div className="setup-grid">
          <label>
            symbols ({list.length}, 1–700)
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
          <Slider label="horizon / rebalance every (bars)" value={horizon} min={1} max={60} onChange={setHorizon} />
          <Slider label="rebalances" value={periods} min={6} max={120} onChange={setPeriods} />
          <Slider label="lookback" value={lookback} min={32} max={512} onChange={setLookback} />
          <Slider label="paths per forecast" value={samples} min={4} max={50} onChange={setSamples} />
        </div>
        <button className="primary start" disabled={busy || list.length < 1 || models.length === 0} onClick={run}>
          {busy ? runningLabel(prog, 'running') : '▶ Run volatility test'}
        </button>
        {!busy && estimate(periods * list.length, samples, horizon, kronos != null) && (
          <span className="muted small"> {estimate(periods * list.length, samples, horizon, kronos != null)}</span>
        )}
      </div>

      {result && (
        <>
          <div className="stats">
            <Stat label="scored rebalances" value={String(result.scored_periods)} hint={`first ${result.calibration_periods} only calibrate`} />
            <Stat label="symbols" value={String(result.symbols.length)} />
            <Stat label="avg realized vol" value={pct(result.avg_realized_vol, 0).slice(1)} hint="annualized" />
            <Stat label="runtime" value={`${(result.latency_ms / 1000).toFixed(0)}s`} />
          </div>
          <RunNote dropped={result.dropped} unavailable={result.unavailable} />

          <div className="panel-title">Forecasts</div>
          <table className="tok-table league research">
            <thead>
              <tr>
                <th>forecast</th>
                <th title="realized vol / forecast vol, raw. 1 = right size, above 1 = bands too narrow">vol ratio</th>
                <th title="QLIKE loss of the raw spread, lower is better">qlike raw</th>
                <th title="QLIKE after walk-forward rescaling by past realized/forecast ratios">qlike cal</th>
                <th title="1 - calibrated QLIKE / ewma's; above 0 beats the baseline">skill vs ewma</th>
                <th title="Diebold-Mariano t of the per-period loss gap; positive = lower loss than ewma">dm t</th>
                <th title="Mincer-Zarnowitz R² of log realized on log forecast (scale-free)">mz r²</th>
                <th title="Mincer-Zarnowitz slope; 1 = right sensitivity">slope</th>
                <th title="rank ic across symbols: which names will be wildest">rank ic (t)</th>
                <th title="average correlation over time within each symbol: when a name gets wilder">timing</th>
              </tr>
            </thead>
            <tbody>
              {result.forecasts.map((f) => (
                <tr key={f.name}>
                  <td>
                    <i className="dot" style={{ background: color(f) }} /> {label(f)}
                  </td>
                  <td className={Math.abs(f.vol_ratio - 1) > 0.25 ? 'down' : ''}>{f.vol_ratio.toFixed(2)}×</td>
                  <td>{f.qlike.toFixed(3)}</td>
                  <td>{f.qlike_cal.toFixed(3)}</td>
                  <td className={tone(f)}>{f.kind === 'baseline' ? '—' : num(f.skill * 100, 1) + '%'}</td>
                  <td className={tone(f)}>{f.kind === 'baseline' ? '—' : num(f.dm_t)}</td>
                  <td>{f.mz_r2.toFixed(2)}</td>
                  <td>{f.mz_slope.toFixed(2)}</td>
                  <td>
                    {num(f.rank_ic.mean)} ({f.rank_ic.t_stat.toFixed(1)})
                  </td>
                  <td>{num(f.timing_corr)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">
            target = realized variance, the sum of squared per-bar log returns over the next {result.horizon} bars. qlike
            (patton 2011) is the loss that ranks variance forecasts fairly despite that noisy target. "cal" first rescales
            each period's forecasts by the average realized / forecast ratio of earlier periods only, so a model isn't
            punished for a constant scale error, just like the conformal bands. green / red when the diebold-mariano |t|
            &gt; 2 against ewma.
          </p>

          <div className="panel-title">
            Forecast vs realized:{' '}
            <select className="field inline" value={symbol} onChange={(e) => setSymbol(Number(e.target.value))}>
              {result.symbols.map((s, i) => (
                <option key={s} value={i}>
                  {s}
                </option>
              ))}
            </select>{' '}
            <span className="segmented inline">
              {[true, false].map((c) => (
                <button key={String(c)} className={calibrated === c ? 'active' : ''} onClick={() => setCalibrated(c)}>
                  {c ? 'calibrated' : 'raw'}
                </button>
              ))}
            </span>
          </div>
          <VolChart interval={result.interval} times={result.origin_times} lines={lines} />
          <div className="legend">
            <span>
              <i className="dot" style={{ background: COLORS.actual }} /> realized
            </span>
            {result.forecasts.map((f) => (
              <span key={f.name}>
                <i className="dot" style={{ background: color(f) }} /> {label(f)}
              </span>
            ))}
          </div>
          <p className="muted small">
            annualized vol at each rebalance. each point is the forecast made at that date and the vol realized over the{' '}
            {result.horizon} bars after it.
          </p>
        </>
      )}
      {!result && !busy && (
        <div className="empty">// no volatility test yet. mega caps × kronos-mini + garch + chronos, 40 rebalances, take about two minutes.</div>
      )}
    </>
  )
}
