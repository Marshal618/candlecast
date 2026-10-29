import { useEffect, useMemo, useState } from 'react'
import { LineStyle } from 'lightweight-charts'
import {
  api,
  type Construction,
  type KronosName,
  type PortfolioRequest,
  type PortfolioResponse,
  type SignalName,
  type Trial,
} from './api'
import { CurvesChart, type Curve } from './charts'
import { COLORS, signalColor, signalLabel } from './theme'
import { pct, rate } from './format'
import { Slider, Stat } from './ui'
import { UNIVERSES } from './universes'

// Fundamentals are rebuilt point-in-time; ETFs and crypto have none and sit mid-ranking on them
const OTHER_SIGNALS: SignalName[] = ['reversal', 'momentum', 'surprise', 'revisions', 'value', 'size', 'naive', 'garch', 'chronos']
const METHODS: { id: Construction['method']; label: string }[] = [
  { id: 'equal', label: 'equal' },
  { id: 'inverse_vol', label: 'inverse vol' },
  { id: 'mean_variance', label: 'mean-variance' },
]

const describe = (c: Trial['config']) =>
  `${c.signals.map((s) => `${s.weight === 1 ? '' : `${s.weight}×`}${s.name}`).join(' + ')} · ${c.weighting === 'ic' ? 'ic-wtd · ' : ''}${
    METHODS.find((m) => m.id === c.construction.method)?.label
  } · ${c.construction.side === 'long_only' ? 'long-only' : 'l/s'}${c.construction.trade_rate < 1 ? ` · trade ${c.construction.trade_rate}` : ''}`

export function PortfolioView({ onError }: { onError: (msg: string | null) => void }) {
  const [symbols, setSymbols] = useState(UNIVERSES['mega caps'].join(', '))
  const [interval, setBarInterval] = useState<'1d' | '1h'>('1d')
  const [kronos, setKronos] = useState<KronosName>('mini')
  const [weights, setWeights] = useState<Partial<Record<SignalName, number>>>({ mini: 1, reversal: 1 })
  const [weighting, setWeighting] = useState<'manual' | 'ic'>('manual')
  const [icWindow, setIcWindow] = useState(12)
  const [c, setC] = useState<Construction>({
    method: 'inverse_vol',
    side: 'long_short',
    select: 0.3,
    risk_window: 120,
    vol_target: 0.1,
    max_gross: 2,
    trade_rate: 1,
  })
  const [horizon, setHorizon] = useState(5)
  const [periods, setPeriods] = useState(60)
  const [lookback, setLookback] = useState(64)
  const [samples, setSamples] = useState(8)
  const [cost, setCost] = useState('5')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<PortfolioResponse | null>(null)
  const [recipeName, setRecipeName] = useState('')
  const [saved, setSaved] = useState<Trial[]>([])

  useEffect(() => {
    api.portfolio.recipes().then(setSaved).catch(() => {})
  }, [])

  const list = [...new Set(symbols.split(/[\s,]+/).map((s) => s.trim().toUpperCase()).filter(Boolean))]
  const available: SignalName[] = [kronos, ...OTHER_SIGNALS]
  const active = available.filter((s) => weights[s] != null)
  const set = <K extends keyof Construction>(k: K, v: Construction[K]) => setC((cur) => ({ ...cur, [k]: v }))
  const toggle = (s: SignalName) =>
    setWeights((cur) => {
      const next = { ...cur }
      if (next[s] != null) delete next[s]
      else next[s] = 1
      return next
    })
  // Switching Kronos size carries its weight over
  const pickKronos = (k: KronosName) => {
    setWeights((cur) => {
      const next = { ...cur }
      if (next[kronos] != null) {
        next[k] = next[kronos]
        delete next[kronos]
      }
      return next
    })
    setKronos(k)
  }

  const request = (): PortfolioRequest => ({
    symbols: list,
    interval,
    signals: active.map((s) => ({ name: s, weight: weights[s] ?? 1 })),
    weighting,
    ic_window: icWindow,
    construction: c,
    lookback,
    horizon,
    periods,
    samples,
    cost_bps: Number(cost) || 0,
  })

  const run = async () => {
    setBusy(true)
    onError(null)
    try {
      setResult(await api.portfolio.run(request()))
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const save = async () => {
    if (!result || !recipeName.trim()) return
    try {
      await api.portfolio.save(result.trial_id, recipeName.trim())
      setSaved(await api.portfolio.recipes())
      setRecipeName('')
    } catch (e) {
      onError((e as Error).message)
    }
  }

  const curves = useMemo<Curve[]>(
    () =>
      result
        ? [
            { values: result.benchmark, color: COLORS.benchmark, style: LineStyle.Dotted },
            { values: result.equity, color: COLORS.actual, width: 2 },
          ]
        : [],
    [result],
  )

  const models = active.filter((s) => !['reversal', 'momentum'].includes(s))
  return (
    <>
      <div className="sim-setup">
        <div className="panel-title">Portfolio lab</div>
        <p className="muted small">
          blend signals into one ranking, size positions with a shrunk covariance risk model, target a volatility, trade
          gradually, pay costs. every run is logged as a trial for this universe, and the sharpe is deflated by how many
          you've tried. save a recipe to race it in the Sim.
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
            kronos size
            <div className="segmented">
              {(['mini', 'small', 'base'] as KronosName[]).map((m) => (
                <button key={m} className={kronos === m ? 'active' : ''} onClick={() => pickKronos(m)}>
                  {m}
                </button>
              ))}
            </div>
          </label>
        </div>

        <div className="panel-title">Signals</div>
        <div className="blend">
          {available.map((s) => (
            <div key={s} className={`blend-row ${weights[s] != null ? 'on' : ''}`}>
              <button onClick={() => toggle(s)}>
                <i className="dot" style={{ background: signalColor(s) }} />
                {signalLabel(s)}
              </button>
              {weights[s] != null && (
                <input
                  className="field"
                  type="number"
                  step={0.5}
                  min={-5}
                  max={5}
                  value={weights[s]}
                  onChange={(e) => setWeights((cur) => ({ ...cur, [s]: Number(e.target.value) }))}
                  title="blend weight; negative flips the signal"
                />
              )}
            </div>
          ))}
        </div>
        <div className="setup-grid">
          <label>
            blend weighting
            <div className="segmented">
              <button className={weighting === 'manual' ? 'active' : ''} onClick={() => setWeighting('manual')}>
                fixed
              </button>
              <button className={weighting === 'ic' ? 'active' : ''} onClick={() => setWeighting('ic')}>
                by trailing ic
              </button>
            </div>
            <span className="muted small">
              {weighting === 'ic'
                ? 'each rebalance weights signals by their positive rank ic over past rebalances only'
                : 'weights above, applied to z-scored ranks'}
            </span>
          </label>
          {weighting === 'ic' && <Slider label="ic window (rebalances)" value={icWindow} min={3} max={60} onChange={setIcWindow} />}
        </div>

        <div className="panel-title">Construction</div>
        <div className="setup-grid">
          <label>
            method
            <div className="segmented">
              {METHODS.map((m) => (
                <button key={m.id} className={c.method === m.id ? 'active' : ''} onClick={() => set('method', m.id)}>
                  {m.label}
                </button>
              ))}
            </div>
          </label>
          <label>
            side
            <div className="segmented">
              <button className={c.side === 'long_short' ? 'active' : ''} onClick={() => set('side', 'long_short')}>
                long-short
              </button>
              <button className={c.side === 'long_only' ? 'active' : ''} onClick={() => set('side', 'long_only')}>
                long-only
              </button>
            </div>
          </label>
          {c.method !== 'mean_variance' && (
            <Slider label="share in each leg" value={c.select} min={0.05} max={1} step={0.05} onChange={(v) => set('select', v)} />
          )}
          <Slider label="risk window (bars)" value={c.risk_window} min={20} max={250} step={10} onChange={(v) => set('risk_window', v)} />
          <label>
            vol target (annual)
            <div className="segmented">
              {[null, 0.05, 0.1, 0.15, 0.2].map((v) => (
                <button key={v ?? 'off'} className={c.vol_target === v ? 'active' : ''} onClick={() => set('vol_target', v)}>
                  {v == null ? 'off' : `${v * 100}%`}
                </button>
              ))}
            </div>
          </label>
          <Slider label="max gross" value={c.max_gross} min={0.5} max={4} step={0.25} onChange={(v) => set('max_gross', v)} />
          <Slider label="trade rate" value={c.trade_rate} min={0.05} max={1} step={0.05} onChange={(v) => set('trade_rate', v)} />
          <label>
            cost per unit turnover (bps)
            <input className="field" inputMode="decimal" value={cost} onChange={(e) => setCost(e.target.value)} />
          </label>
        </div>

        <details className="advanced">
          <summary>Walk-forward</summary>
          <div className="setup-grid">
            <Slider label="rebalance every (bars)" value={horizon} min={1} max={60} onChange={setHorizon} />
            <Slider label="rebalances" value={periods} min={4} max={120} onChange={setPeriods} />
            <Slider label="model lookback" value={lookback} min={32} max={512} onChange={setLookback} />
            <Slider label="paths per forecast" value={samples} min={1} max={30} onChange={setSamples} />
          </div>
        </details>

        <button className="primary start" disabled={busy || list.length < 4 || active.length === 0} onClick={run}>
          {busy
            ? models.length
              ? `▶ running (new forecasts: up to ${periods * list.length * models.length})…`
              : '▶ running…'
            : '▶ Run portfolio'}
        </button>
      </div>

      {result && <PortfolioResult r={result} curves={curves} recipeName={recipeName} setRecipeName={setRecipeName} save={save} />}
      {!result && !busy && (
        <div className="empty">
          // no run yet. model forecasts are cached, so after the first run, changing construction is instant.
        </div>
      )}

      {saved.length > 0 && (
        <>
          <div className="panel-title">Saved recipes</div>
          <table className="tok-table league">
            <thead>
              <tr>
                <th>name</th>
                <th>recipe</th>
                <th>bars</th>
                <th>sharpe</th>
                <th>ann</th>
                <th>final blend</th>
              </tr>
            </thead>
            <tbody>
              {saved.map((t) => (
                <tr key={t.id}>
                  <td>{t.name}</td>
                  <td className="muted">{describe(t.config)}</td>
                  <td>
                    {t.config.interval} / {t.config.horizon}
                  </td>
                  <td>{t.metrics.sharpe.toFixed(2)}</td>
                  <td>{pct(t.metrics.ann_return, 1)}</td>
                  <td>
                    {Object.entries(t.final_blend ?? {})
                      .filter(([, w]) => w)
                      .map(([k, w]) => `${k} ${(w as number).toFixed(2)}`)
                      .join(', ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">pick one as the portfolio bot when you start a Sim game.</p>
        </>
      )}
    </>
  )
}

function PortfolioResult({
  r,
  curves,
  recipeName,
  setRecipeName,
  save,
}: {
  r: PortfolioResponse
  curves: Curve[]
  recipeName: string
  setRecipeName: (s: string) => void
  save: () => void
}) {
  const m = r.metrics
  const d = r.deflation
  const last = r.weights.at(-1) ?? []
  const maxW = Math.max(...last.map(Math.abs), 1e-9)
  const order = last.map((w, i) => [w, i] as const).sort((a, b) => b[0] - a[0])
  return (
    <>
      <div className="stats">
        <Stat label="ann return" value={pct(m.ann_return, 1)} tone={m.ann_return >= 0 ? 'up' : 'down'} hint={`universe ${pct(m.bench_ann_return, 1)}`} />
        <Stat label="ann vol" value={pct(m.ann_vol, 1).slice(1)} hint={`ex-ante ${pct(m.ex_ante_vol, 1).slice(1)}`} />
        <Stat label="sharpe" value={m.sharpe.toFixed(2)} hint={`${r.periods} rebalances`} />
        <Stat label="max drawdown" value={pct(m.max_drawdown, 1)} />
        <Stat label="turnover" value={rate(m.turnover)} hint={`cost drag ${pct(m.cost_drag, 2).slice(1)}/yr`} />
        <Stat label="gross / net" value={`${m.avg_gross.toFixed(2)} / ${m.avg_net >= 0 ? '+' : ''}${m.avg_net.toFixed(2)}`} />
      </div>

      <div className={`deflation ${d.dsr >= 0.95 ? 'pass' : 'fail'}`}>
        <div>
          <div className="results-title">deflated sharpe {rate(d.dsr)}</div>
          <div className="muted small">
            probability the true sharpe beats the best of {d.trials} trial{d.trials === 1 ? '' : 's'} on this universe by luck
            alone (luck alone reaches sharpe ≈ {d.luck_sharpe.toFixed(2)}). without deflation: {rate(d.psr)}. 95% is the usual
            bar; every new configuration you try raises it.
          </div>
        </div>
        <div className="save">
          <input className="field" placeholder="recipe name" value={recipeName} onChange={(e) => setRecipeName(e.target.value)} />
          <button disabled={!recipeName.trim()} onClick={save}>
            save recipe
          </button>
        </div>
      </div>

      <div className="panel-title">Equity, net of costs</div>
      <CurvesChart interval={r.interval} times={r.origin_times} curves={curves} />
      <div className="legend">
        <span>
          <i className="dot" style={{ background: COLORS.actual }} /> portfolio
        </span>
        <span className="muted">··· equal-weight universe (long only)</span>
      </div>

      <div className="token-grid">
        <div>
          <div className="panel-title">Latest weights</div>
          <table className="tok-table">
            <tbody>
              {order.map(([w, i]) => (
                <tr key={r.symbols[i]}>
                  <td>{r.symbols[i]}</td>
                  <td className={w > 0 ? 'up' : w < 0 ? 'down' : 'muted'} style={{ width: '70%' }}>
                    <span className={`wbar ${w < 0 ? 'neg' : ''}`} style={{ width: `${(Math.abs(w) / maxW) * 50}%` }} />
                    {pct(w, 1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <div className="panel-title">Blend</div>
          <div className="meters">
            {r.signals.map((s, k) => (
              <div key={s} className="meter">
                <span className="meter-label">
                  <i className="dot" style={{ background: signalColor(s) }} />
                  {signalLabel(s)}
                </span>
                <div className="meter-track">
                  <div className="meter-fill" style={{ width: `${Math.abs(r.final_blend[s] ?? 0) * 100}%`, background: signalColor(s) }} />
                </div>
                <span className="meter-value">{(r.final_blend[s] ?? 0).toFixed(2)}</span>
                <span className="muted small span3">
                  avg {(r.blend.reduce((a, b) => a + b[k], 0) / r.blend.length).toFixed(2)} over the run
                </span>
              </div>
            ))}
          </div>

          <div className="panel-title">Best trials on this universe</div>
          <table className="tok-table">
            <thead>
              <tr>
                <th>#</th>
                <th>recipe</th>
                <th>sharpe</th>
              </tr>
            </thead>
            <tbody>
              {r.study.map((t) => (
                <tr key={t.id} className={t.id === r.trial_id ? 'me' : ''}>
                  <td>{t.id}</td>
                  <td className="muted">{t.name ? <b>{t.name}</b> : describe(t.config)}</td>
                  <td>{t.metrics.sharpe.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}
