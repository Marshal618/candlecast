import { useEffect, useState } from 'react'
import {
  api,
  type BacktestResponse,
  type Candle,
  type ForecastResponse,
  type Health,
  type Interval,
  type ModelName,
  type StrategyMode,
  type JournalReport,
  type Scorecard,
  type StrategyResult,
  type TokensResponse,
} from './api'
import { BacktestChart, EquityChart, PriceChart, TokenChart, type TokenOverlay } from './charts'
import { pct, rate } from './format'
import { Slider, Stat } from './ui'
import { SimView } from './Sim'
import './App.css'

const INTERVALS: Interval[] = ['5m', '15m', '1h', '1d']
const MODELS: ModelName[] = ['mini', 'small', 'base']
const DEFAULT_WATCHLIST = ['AAPL', 'NVDA', 'SPY', 'BTC-USD', 'ETH-USD']
const WATCHLIST_KEY = 'kd.watchlist'

function loadWatchlist(): string[] {
  try {
    const raw = localStorage.getItem(WATCHLIST_KEY)
    const parsed = raw ? JSON.parse(raw) : null
    return Array.isArray(parsed) ? parsed : DEFAULT_WATCHLIST
  } catch {
    return DEFAULT_WATCHLIST
  }
}

function saveWatchlist(list: string[]) {
  try {
    localStorage.setItem(WATCHLIST_KEY, JSON.stringify(list))
  } catch {
    /* storage unavailable; watchlist just won't persist */
  }
}


export default function App() {
  const [symbol, setSymbol] = useState('AAPL')
  const [input, setInput] = useState('AAPL')
  const [interval, setBarInterval] = useState<Interval>('1d')
  const [model, setModel] = useState<ModelName>('small')
  const [horizon, setHorizon] = useState(20)
  const [lookback, setLookback] = useState(400)
  const [samples, setSamples] = useState(10)
  const [temperature, setTemperature] = useState(1.0)
  const [topP, setTopP] = useState(0.9)
  const [tab, setTab] = useState<'forecast' | 'backtest' | 'tokens' | 'journal' | 'sim'>('forecast')
  const [journal, setJournal] = useState<JournalReport | null>(null)
  const [journalScope, setJournalScope] = useState<'symbol' | 'all'>('all')
  const [tokens, setTokens] = useState<TokensResponse | null>(null)
  const [overlay, setOverlay] = useState<TokenOverlay>('surprisal')

  const [watchlist, setWatchlist] = useState<string[]>(loadWatchlist)
  const [health, setHealth] = useState<Health | null>(null)
  const [candles, setCandles] = useState<Candle[]>([])
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null)
  const [mode, setMode] = useState<StrategyMode>('long_only')
  const [enterP, setEnterP] = useState(0.6)
  const [costBps, setCostBps] = useState(5)
  const [strategy, setStrategy] = useState<StrategyResult | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const maxContext = health?.models[model]?.max_context ?? 512

  useEffect(() => {
    api.health().then(setHealth).catch(() => setError('Backend unreachable. Is the API running? Start it with ./dev.sh'))
  }, [])

  useEffect(() => {
    setLookback((lb) => Math.min(lb, maxContext))
  }, [maxContext])

  useEffect(() => {
    let cancelled = false
    setForecast(null)
    setBacktest(null)
    setStrategy(null)
    setTokens(null)
    setError(null)
    setBusy('Loading candles…')
    api
      .candles(symbol, interval)
      .then((r) => !cancelled && setCandles(r.candles))
      .catch((e) => {
        if (cancelled) return
        setCandles([])
        setError(e.message)
      })
      .finally(() => !cancelled && setBusy(null))
    return () => {
      cancelled = true
    }
  }, [symbol, interval])

  const params = { symbol, interval, model, lookback, samples, temperature, top_p: topP }
  const strategyParams = { mode, enter_p: enterP, cost_bps: costBps }

  // Re-score the existing backtest forecasts when the rules change; no model calls involved
  useEffect(() => {
    if (!backtest) return
    const ctrl = new AbortController()
    const timer = setTimeout(() => {
      api
        .strategy(
          {
            symbol: backtest.symbol,
            interval: backtest.interval,
            horizon: backtest.horizon,
            windows: backtest.windows.map((w) => ({ origin_time: w.origin_time, p_up: w.p_up })),
            strategy: { mode, enter_p: enterP, cost_bps: costBps },
          },
          ctrl.signal,
        )
        .then(setStrategy)
        .catch((e) => e.name !== 'AbortError' && setError(e.message))
    }, 150)
    return () => {
      clearTimeout(timer)
      ctrl.abort()
    }
  }, [backtest, mode, enterP, costBps])

  const runForecast = async () => {
    setError(null)
    setBusy(`Running Kronos-${model}…`)
    try {
      setForecast(await api.forecast({ ...params, horizon }))
      setTab('forecast')
      api.health().then(setHealth).catch(() => {})
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  const runBacktest = async () => {
    setError(null)
    setBusy(`Backtesting Kronos-${model} (this can take a while)…`)
    try {
      const r = await api.backtest({
        ...params,
        horizon,
        windows: 20,
        samples: Math.min(samples, 5),
        strategy: strategyParams,
      })
      setStrategy(r.strategy)
      setBacktest(r)
      setTab('backtest')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  const loadJournal = async () => {
    setError(null)
    setBusy('Scoring journal…')
    try {
      setJournal(await api.journal(journalScope === 'symbol' ? { symbol } : {}))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  useEffect(() => {
    if (tab === 'journal') loadJournal()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, journalScope, symbol])

  const runTokens = async () => {
    setError(null)
    setBusy(`Inspecting Kronos-${model} tokens…`)
    try {
      setTokens(await api.tokens({ symbol, interval, model, lookback }))
      setTab('tokens')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  const submitSymbol = (e: React.FormEvent) => {
    e.preventDefault()
    const s = input.trim().toUpperCase()
    if (s) setSymbol(s)
  }

  const toggleWatch = () => {
    const next = watchlist.includes(symbol) ? watchlist.filter((s) => s !== symbol) : [...watchlist, symbol]
    setWatchlist(next)
    saveWatchlist(next)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>CANDLECAST</h1>
        <span className="tag">{symbol}</span>
        <span className="tag">{interval}</span>
        <span className="muted header-meta">
          {health ? (
            <>
              <i className={`led ${busy ? 'busy' : 'ok'}`} /> {health.device} · {health.loaded_models.join(',') || 'idle'}
            </>
          ) : (
            <>
              <i className="led off" /> offline
            </>
          )}
        </span>
      </header>

      <aside className="sidebar">
        <form onSubmit={submitSymbol} className="symbol-form">
          <span className="prompt">&gt;</span>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="AAPL, BTC-USD" spellCheck={false} />
          <button type="submit">Go</button>
        </form>

        <div className="watchlist">
          <div className="section-title">
            Watchlist
            <button className="link" onClick={toggleWatch}>
              {watchlist.includes(symbol) ? `− ${symbol}` : `+ ${symbol}`}
            </button>
          </div>
          {watchlist.map((s) => (
            <button
              key={s}
              className={`watch-item ${s === symbol ? 'active' : ''}`}
              onClick={() => {
                setSymbol(s)
                setInput(s)
              }}
            >
              {s}
            </button>
          ))}
        </div>

        <div className="section-title">Interval</div>
        <label>
          <div className="segmented">
            {INTERVALS.map((i) => (
              <button key={i} className={i === interval ? 'active' : ''} onClick={() => setBarInterval(i)}>
                {i}
              </button>
            ))}
          </div>
        </label>

        <div className="section-title">Model</div>
        <label>
          <div className="segmented">
            {MODELS.map((m) => (
              <button key={m} className={m === model ? 'active' : ''} onClick={() => setModel(m)}>
                {m}
              </button>
            ))}
          </div>
        </label>
        <Slider label="horizon" value={horizon} min={1} max={120} onChange={setHorizon} />
        <Slider label="lookback" value={lookback} min={32} max={maxContext} onChange={setLookback} />
        <details className="advanced">
          <summary>Sampling</summary>
          <Slider label="paths" value={samples} min={1} max={50} onChange={setSamples} />
          <Slider label="temp" value={temperature} min={0.1} max={2} step={0.1} onChange={setTemperature} />
          <Slider label="top_p" value={topP} min={0.1} max={1} step={0.05} onChange={setTopP} />
        </details>

        <div className="actions">
          <button className="primary" disabled={!!busy || candles.length === 0} onClick={runForecast}>
            ▶ Run
          </button>
          <button disabled={!!busy || candles.length === 0} onClick={runBacktest}>
            Backtest
          </button>
          <button className="span2" disabled={!!busy || candles.length === 0} onClick={runTokens}>
            Inspect tokens
          </button>
        </div>
      </aside>

      <main className="main">
        <div className="tabs">
          <button className={tab === 'forecast' ? 'active' : ''} onClick={() => setTab('forecast')}>
            Forecast
          </button>
          <button className={tab === 'backtest' ? 'active' : ''} onClick={() => setTab('backtest')}>
            Backtest
          </button>
          <button className={tab === 'tokens' ? 'active' : ''} onClick={() => setTab('tokens')}>
            Tokens
          </button>
          <button className={tab === 'journal' ? 'active' : ''} onClick={() => setTab('journal')}>
            Journal
          </button>
          <button className={tab === 'sim' ? 'active' : ''} onClick={() => setTab('sim')}>
            Sim
          </button>
          <span className="status">{busy ? `> ${busy}` : ''}</span>
        </div>

        {error && <div className="error">{error}</div>}

        {tab === 'sim' ? (
          <SimView watchlist={watchlist} onError={setError} />
        ) : tab === 'journal' ? (
          <JournalView
            report={journal}
            scope={journalScope}
            setScope={setJournalScope}
            symbol={symbol}
            refresh={loadJournal}
            busy={!!busy}
          />
        ) : tab === 'tokens' ? (
          tokens ? (
            <TokensView t={tokens} candles={candles} overlay={overlay} setOverlay={setOverlay} interval={interval} />
          ) : (
            <div className="empty">// no token inspection yet. click "Inspect tokens" to see how Kronos reads this chart.</div>
          )
        ) : tab === 'forecast' ? (
          <>
            <div className="stats">
              <Stat label="last" value={candles[candles.length - 1]?.close.toFixed(2) ?? '—'} />
              <Stat
                label={`p(up) +${forecast?.forecast.length ?? horizon}`}
                value={forecast ? rate(forecast.stats.p_up) : '—'}
                tone={forecast ? (forecast.stats.p_up >= 0.5 ? 'up' : 'down') : undefined}
              />
              <Stat
                label="exp move"
                value={forecast ? pct(forecast.stats.expected_return, 2) : '—'}
                tone={forecast ? (forecast.stats.expected_return >= 0 ? 'up' : 'down') : undefined}
              />
              <Stat label="band p10–p90" value={forecast ? pct(forecast.stats.band_width_pct).slice(1) : '—'} />
              <Stat label="latency" value={forecast ? `${(forecast.latency_ms / 1000).toFixed(1)}s` : '—'} />
            </div>
            {forecast && (
              <p className="muted small journal-note">
                {forecast.journal.new
                  ? `▸ logged to journal as #${forecast.journal.id}. it will be scored automatically once its ${forecast.forecast.length} bars close.`
                  : `▸ journal already holds #${forecast.journal.id} for this bar, model and horizon. the first forecast counts, so re-runs aren't logged.`}
              </p>
            )}
            <PriceChart candles={candles} forecast={forecast} interval={interval} />
            <p className="legend">
              <span className="sw candle-hist" /> history
              <span className="sw candle-proj" /> projected (hollow up / solid down)
              <span className="sw median" /> median
              <span className="sw band" /> p10–p90
            </p>
          </>
        ) : backtest ? (
          <>
            <div className="stats">
              <Stat
                label="hit rate"
                value={rate(backtest.metrics.hit_rate)}
                hint={`baseline ${rate(backtest.metrics.baseline_hit_rate)}`}
                tone={backtest.metrics.hit_rate > backtest.metrics.baseline_hit_rate ? 'up' : 'down'}
              />
              <Stat
                label="close mae"
                value={pct(backtest.metrics.mae_pct, 2).slice(1)}
                hint={`naive ${pct(backtest.metrics.naive_mae_pct, 2).slice(1)}`}
                tone={backtest.metrics.mae_pct < backtest.metrics.naive_mae_pct ? 'up' : 'down'}
              />
              <Stat
                label="r² oos"
                value={pct(backtest.metrics.r2_oos, 1)}
                hint={backtest.metrics.ridge_r2_oos == null ? 'vs zero forecast' : `ridge ${pct(backtest.metrics.ridge_r2_oos, 1)}`}
                tone={backtest.metrics.r2_oos > Math.max(0, backtest.metrics.ridge_r2_oos ?? 0) ? 'up' : 'down'}
              />
              <Stat
                label="ridge hit"
                value={backtest.metrics.ridge_hit_rate == null ? '—' : rate(backtest.metrics.ridge_hit_rate)}
                hint="linear ml baseline"
              />
              <Stat label="coverage" value={rate(backtest.metrics.coverage_p10_p90)} hint="ideal ≈ 80%" />
              <Stat label="windows" value={String(backtest.metrics.windows)} />
              <Stat label="latency" value={`${(backtest.latency_ms / 1000).toFixed(1)}s`} />
            </div>
            <BacktestChart result={backtest} interval={interval} />
            <p className="legend">
              <span className="sw actual" /> actual <span className="sw median" /> forecast @ horizon
              <span className="sw band" /> p10–p90 · dots: green hit / red miss
            </p>
            <p className="muted small">
              baseline = majority direction · naive = no change · r² oos vs a zero-return forecast (gu, kelly &amp; xiu);
              negative = worse than predicting nothing · ridge = walk-forward regularized regression on lagged returns,
              volatility, volume.
            </p>

            <div className="panel-title">Strategy</div>
            <div className="strategy-controls">
              <div className="segmented">
                <button className={mode === 'long_only' ? 'active' : ''} onClick={() => setMode('long_only')}>
                  long only
                </button>
                <button className={mode === 'long_short' ? 'active' : ''} onClick={() => setMode('long_short')}>
                  long / short
                </button>
              </div>
              <Slider label="long if p(up) ≥" value={enterP} min={0.5} max={1} step={0.05} onChange={setEnterP} />
              <Slider label="cost bps" value={costBps} min={0} max={50} onChange={setCostBps} />
            </div>
            <p className="muted small rule">
              rule: at each window origin, {mode === 'long_short' ? 'long' : 'go long'} if p(up) ≥ {enterP.toFixed(2)}
              {mode === 'long_short' ? `, short if p(up) ≤ ${(1 - enterP).toFixed(2)}` : ''}, else flat. fill at next
              bar open, hold {backtest.horizon} bars, pay {costBps} bps per side.
            </p>
            {strategy && <StrategyReport s={strategy} />}
            {strategy && <EquityChart result={strategy} interval={interval} />}
            <p className="legend">
              <span className="sw actual" /> strategy <span className="sw benchmark" /> buy &amp; hold · arrows: entries
            </p>
          </>
        ) : (
          <div className="empty">// no backtest yet. run one to score past forecasts against reality.</div>
        )}

        <p className="disclaimer">research tool · not financial advice · forecasts are probabilistic and often wrong</p>
      </main>
    </div>
  )
}

const fmtDate = (t: number) =>
  new Date(t * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

function JournalView({
  report,
  scope,
  setScope,
  symbol,
  refresh,
  busy,
}: {
  report: JournalReport | null
  scope: 'symbol' | 'all'
  setScope: (s: 'symbol' | 'all') => void
  symbol: string
  refresh: () => void
  busy: boolean
}) {
  const controls = (
    <div className="journal-controls">
      <div className="segmented">
        <button className={scope === 'all' ? 'active' : ''} onClick={() => setScope('all')}>
          all symbols
        </button>
        <button className={scope === 'symbol' ? 'active' : ''} onClick={() => setScope('symbol')}>
          {symbol} only
        </button>
      </div>
      <button disabled={busy} onClick={refresh}>
        ↻ score now
      </button>
    </div>
  )
  if (!report) return controls
  if (report.total === 0)
    return (
      <>
        {controls}
        <div className="empty">
          // journal is empty. every forecast you run is logged here and scored once its bars have closed. this is the
          only test kronos can't have seen during pre-training.
        </div>
      </>
    )

  const sc = report.scorecard
  return (
    <>
      {controls}
      <p className="muted small">
        {report.total} forecasts logged · {report.total - report.pending} scored · {report.pending} waiting for bars to
        close
      </p>
      {sc ? (
        <>
          <ScorecardStats s={sc} />
          {sc.n < 30 && (
            <p className="muted small warn">
              only {sc.n} scored forecasts. under ~30 these numbers are mostly noise, so keep logging before drawing
              conclusions.
            </p>
          )}
        </>
      ) : (
        <div className="empty">// nothing scored yet. the first results arrive when the earliest forecast's horizon closes.</div>
      )}

      <div className="token-grid">
        <div>
          <div className="panel-title">Live vs backtest</div>
          <table className="tok-table">
            <thead>
              <tr>
                <th>series</th>
                <th>live n</th>
                <th>live hit</th>
                <th>bt hit</th>
                <th>live r²</th>
                <th>bt r²</th>
              </tr>
            </thead>
            <tbody>
              {report.groups.map((g) => {
                const gap = g.live && g.backtest ? g.live.hit_rate - g.backtest.hit_rate : null
                return (
                  <tr key={`${g.symbol}-${g.interval}-${g.model}`}>
                    <td>
                      {g.symbol} · {g.interval} · {g.model}
                    </td>
                    <td>
                      {g.live?.n ?? 0}/{g.total}
                    </td>
                    <td className={gap == null ? '' : gap < -0.1 ? 'down' : ''}>{g.live ? rate(g.live.hit_rate) : '—'}</td>
                    <td>{g.backtest ? rate(g.backtest.hit_rate) : '—'}</td>
                    <td>{g.live ? pct(g.live.r2_oos, 1) : '—'}</td>
                    <td>{g.backtest ? pct(g.backtest.r2_oos, 1) : '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <p className="muted small">
            bt = most recent backtest for the same symbol, interval and model. if live hit rate sits well below backtest
            (rust), the backtest was probably flattered by data kronos saw in pre-training.
          </p>
        </div>
        <div>
          <div className="panel-title">Calibration</div>
          {sc && sc.calibration.length > 0 ? (
            <CalibrationChart bins={sc.calibration} />
          ) : (
            <p className="muted small">appears once forecasts are scored.</p>
          )}
          <p className="muted small">
            when kronos says p(up) = x, how often did price actually go up? dots on the diagonal = honest probabilities.
            dot size = number of forecasts.
          </p>
        </div>
      </div>

      <div className="panel-title">Forecast log</div>
      <table className="tok-table">
        <thead>
          <tr>
            <th>#</th>
            <th>made</th>
            <th>from bar</th>
            <th>series</th>
            <th>h</th>
            <th>p(up)</th>
            <th>expected</th>
            <th>actual</th>
            <th>result</th>
          </tr>
        </thead>
        <tbody>
          {report.entries.map((e) => (
            <tr key={e.id}>
              <td>{e.id}</td>
              <td>{fmtDate(e.created_at)}</td>
              <td>{fmtDate(e.origin_time)}</td>
              <td>
                {e.symbol} · {e.interval} · {e.model}
              </td>
              <td>{e.horizon}</td>
              <td>{rate(e.p_up)}</td>
              <td className={e.expected_return >= 0 ? 'up' : 'down'}>{pct(e.expected_return, 2)}</td>
              <td className={e.actual_return == null ? '' : e.actual_return >= 0 ? 'up' : 'down'}>
                {e.actual_return == null ? '—' : pct(e.actual_return, 2)}
              </td>
              <td className={e.hit == null ? 'muted' : e.hit ? 'up' : 'down'}>
                {e.hit == null ? 'pending' : e.hit ? 'hit' : 'miss'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}

function ScorecardStats({ s }: { s: Scorecard }) {
  return (
    <div className="stats">
      <Stat label="scored" value={String(s.n)} />
      <Stat
        label="hit rate"
        value={rate(s.hit_rate)}
        hint={`baseline ${rate(s.baseline_hit_rate)}`}
        tone={s.hit_rate > s.baseline_hit_rate ? 'up' : 'down'}
      />
      <Stat label="r² oos" value={pct(s.r2_oos, 1)} hint="vs zero forecast" tone={s.r2_oos > 0 ? 'up' : 'down'} />
      <Stat label="coverage" value={rate(s.coverage_p10_p90)} hint="ideal ≈ 80%" />
      <Stat
        label="brier skill"
        value={pct(s.brier_skill, 1)}
        hint="p(up) vs coin flip"
        tone={s.brier_skill > 0 ? 'up' : 'down'}
      />
      <Stat
        label="luck test p"
        value={s.p_value == null ? '—' : s.p_value.toFixed(2)}
        hint={s.p_value == null ? 'needs mixed calls' : s.p_value < 0.05 ? 'beats random' : 'could be luck'}
        tone={s.p_value == null ? undefined : s.p_value < 0.05 ? 'up' : 'down'}
      />
    </div>
  )
}

function CalibrationChart({ bins }: { bins: Scorecard['calibration'] }) {
  const size = 240
  const pad = 28
  const inner = size - pad * 2
  const x = (v: number) => pad + v * inner
  const y = (v: number) => size - pad - v * inner
  const maxN = Math.max(...bins.map((b) => b.n))
  return (
    <svg className="calibration" viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Calibration of P(up)">
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={x(v)} x2={x(v)} y1={y(0)} y2={y(1)} className="cal-grid" />
          <line x1={x(0)} x2={x(1)} y1={y(v)} y2={y(v)} className="cal-grid" />
          <text x={x(v)} y={size - 8} className="cal-label" textAnchor="middle">
            {v * 100}%
          </text>
          <text x={8} y={y(v) + 3} className="cal-label">
            {v * 100}%
          </text>
        </g>
      ))}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className="cal-diag" />
      {bins.map((b) => (
        <circle key={b.lo} cx={x(b.predicted)} cy={y(b.observed)} r={3 + 7 * Math.sqrt(b.n / maxN)} className="cal-dot">
          <title>{`p(up) ${rate(b.lo)}–${rate(b.hi)}: predicted ${rate(b.predicted)}, observed ${rate(b.observed)} (n=${b.n})`}</title>
        </circle>
      ))}
    </svg>
  )
}

const FEATURE_LABELS: Record<string, string> = {
  level: 'price level in window',
  volume: 'volume',
  range: 'high–low range',
  body: 'candle body',
  return: 'bar return',
}

function TokensView({
  t,
  candles,
  overlay,
  setOverlay,
  interval,
}: {
  t: TokensResponse
  candles: Candle[]
  overlay: TokenOverlay
  setOverlay: (o: TokenOverlay) => void
  interval: Interval
}) {
  const s = t.summary
  const n = t.next
  const sem = t.semantics
  const features = Object.entries(sem.explained).sort((a, b) => b[1] - a[1])
  return (
    <>
      <div className="stats">
        <Stat label="tokens used" value={`${s.unique_s1}`} hint={`of ${t.vocab.s1} coarse · ${s.unique_s2} fine`} />
        <Stat label="perplexity" value={s.perplexity.toFixed(0)} hint={`${s.mean_surprisal_bits.toFixed(1)} bits / candle`} />
        <Stat label="next entropy" value={`${n.entropy_bits.toFixed(1)} b`} hint={`max ${n.max_entropy_bits} bits`} />
        <Stat
          label="token p(up)"
          value={n.implied_p_up == null ? '—' : rate(n.implied_p_up)}
          hint={n.implied_return == null ? '' : `next bar ${pct(n.implied_return, 2)}`}
          tone={n.implied_p_up == null ? undefined : n.implied_p_up >= 0.5 ? 'up' : 'down'}
        />
        <Stat
          label="recon error"
          value={pct(s.recon_mae_full_pct, 2).slice(1)}
          hint={`coarse only ${pct(s.recon_mae_coarse_pct, 2).slice(1)}`}
        />
        <Stat label="latency" value={`${(t.latency_ms / 1000).toFixed(1)}s`} />
      </div>

      <div className="tabs-inline segmented">
        <button className={overlay === 'surprisal' ? 'active' : ''} onClick={() => setOverlay('surprisal')}>
          surprisal
        </button>
        <button className={overlay === 'attribution' ? 'active' : ''} onClick={() => setOverlay('attribution')}>
          attribution
        </button>
      </div>
      <TokenChart result={t} candles={candles} overlay={overlay} interval={interval} />
      <p className="legend">
        <span className="sw candle-hist" /> candles <span className="sw median" /> decoded from coarse tokens only
        {overlay === 'surprisal' ? (
          <>
            <span className="sw band" /> surprisal (bits): how unexpected each candle was to Kronos; brass = top 10%
          </>
        ) : (
          <>
            <span className="sw candle-hist" /> attribution (bps): how much each block pushes the next-bar outlook up
            (sage) or down (rust), found by flattening that block
          </>
        )}
      </p>

      <div className="token-grid">
        <div>
          <div className="panel-title">Next-token candidates</div>
          <table className="tok-table">
            <thead>
              <tr>
                <th>token</th>
                <th>p</th>
                <th>seen</th>
                <th>avg ret</th>
                <th>p(up)</th>
                <th>level</th>
                <th>volume</th>
              </tr>
            </thead>
            <tbody>
              {n.top.map((c) => (
                <tr key={c.s1}>
                  <td>#{c.s1}</td>
                  <td>
                    <span className="pbar" style={{ width: `${Math.round(c.p * 100)}%` }} />
                    {rate(c.p)}
                  </td>
                  {c.prototype ? (
                    <>
                      <td>{c.prototype.count}×</td>
                      <td className={c.prototype.mean_return >= 0 ? 'up' : 'down'}>{pct(c.prototype.mean_return, 2)}</td>
                      <td>{rate(c.prototype.p_up)}</td>
                      <td>{describeZ(c.prototype.level, 'high', 'low')}</td>
                      <td>{describeZ(c.prototype.volume, 'heavy', 'light')}</td>
                    </>
                  ) : (
                    <td colSpan={5} className="muted">never seen in history</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">
            the model's distribution over the next candle's coarse token (deterministic, no sampling). columns describe
            what each token meant historically for this ticker. token p(up) above weights those by probability, which is
            a readout of the tokens and not the sampled forecast.
          </p>
        </div>
        <div>
          <div className="panel-title">What a token encodes</div>
          <div className="meters">
            {features.map(([k, v]) => (
              <div className="meter" key={k}>
                <span className="meter-label">{FEATURE_LABELS[k]}</span>
                <span className="meter-track">
                  <span className="meter-fill" style={{ width: `${v * 100}%` }} />
                  <span className="meter-chance" style={{ left: `${sem.chance * 100}%` }} />
                </span>
                <span className="meter-value">{rate(v)}</span>
              </div>
            ))}
          </div>
          <p className="muted small">
            share of each candle property explained by its coarse token, over {sem.bars} historical bars and{' '}
            {sem.tokens_used} distinct tokens. the tick marks what chance alone would give ({rate(sem.chance)}). kronos
            z-scores each window, so tokens mostly encode where a candle sits in the window and its volume, and much less
            its own shape. this is one reason it tends to forecast reversion to the window average.
          </p>
        </div>
      </div>
    </>
  )
}

function describeZ(z: number, hi: string, lo: string) {
  if (z > 0.5) return `${hi} (${z.toFixed(1)}σ)`
  if (z < -0.5) return `${lo} (${z.toFixed(1)}σ)`
  return `mid (${z.toFixed(1)}σ)`
}

function StrategyReport({ s }: { s: StrategyResult }) {
  const m = s.metrics
  const beat = m.total_return > m.bh_return
  const p = m.p_value
  return (
    <>
      <div className="stats">
        <Stat
          label="return"
          value={pct(m.total_return)}
          hint={`buy & hold ${pct(m.bh_return)}`}
          tone={beat ? 'up' : 'down'}
        />
        <Stat label="sharpe" value={m.sharpe.toFixed(2)} hint={`b&h ${m.bh_sharpe.toFixed(2)}`} tone={m.sharpe > m.bh_sharpe ? 'up' : 'down'} />
        <Stat
          label="max drawdown"
          value={pct(m.max_drawdown)}
          hint={`b&h ${pct(m.bh_max_drawdown)}`}
          tone={m.max_drawdown > m.bh_max_drawdown ? 'up' : 'down'}
        />
        <Stat label="exposure" value={rate(m.exposure)} hint={`${m.trades} trades`} />
        <Stat
          label="win rate"
          value={m.win_rate == null ? '—' : rate(m.win_rate)}
          hint={`${m.active_windows} active windows`}
        />
        <Stat
          label="luck test p"
          value={p == null ? '—' : p.toFixed(2)}
          hint={p == null ? 'needs mixed positions' : p < 0.05 ? 'timing beats random' : 'could be luck'}
          tone={p == null ? undefined : p < 0.05 ? 'up' : 'down'}
        />
      </div>
      <p className="muted small">
        luck test: the same positions reshuffled across windows 1,000 times; p = share of shuffles that did at least as
        well. below 0.05 suggests real timing skill. {m.active_windows < 10 && 'very few active windows, so treat any result as noise.'}
      </p>
    </>
  )
}
