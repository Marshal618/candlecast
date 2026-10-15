import { useEffect, useState } from 'react'
import {
  api,
  type BacktestResponse,
  type Candle,
  type ForecastResponse,
  type Health,
  type Interval,
  type ModelName,
} from './api'
import { BacktestChart, PriceChart } from './charts'
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

const pct = (x: number, digits = 1) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(digits)}%`
const rate = (x: number) => `${(x * 100).toFixed(0)}%`

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
  const [tab, setTab] = useState<'forecast' | 'backtest'>('forecast')

  const [watchlist, setWatchlist] = useState<string[]>(loadWatchlist)
  const [health, setHealth] = useState<Health | null>(null)
  const [candles, setCandles] = useState<Candle[]>([])
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null)
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
      setBacktest(await api.backtest({ ...params, horizon, windows: 20, samples: Math.min(samples, 5) }))
      setTab('backtest')
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
        <h1>Kronos Forecast</h1>
        <span className="muted">
          {health ? `device: ${health.device} · loaded: ${health.loaded_models.join(', ') || 'none'}` : 'connecting…'}
        </span>
      </header>

      <aside className="sidebar">
        <form onSubmit={submitSymbol} className="symbol-form">
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Symbol (AAPL, BTC-USD)" />
          <button type="submit">Load</button>
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

        <div className="section-title">Data</div>
        <label>
          Interval
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
          Kronos size
          <div className="segmented">
            {MODELS.map((m) => (
              <button key={m} className={m === model ? 'active' : ''} onClick={() => setModel(m)}>
                {m}
              </button>
            ))}
          </div>
        </label>
        <Slider label="Horizon (bars)" value={horizon} min={1} max={120} onChange={setHorizon} />
        <Slider label="Lookback (bars)" value={lookback} min={32} max={maxContext} onChange={setLookback} />
        <Slider label="Sample paths" value={samples} min={1} max={50} onChange={setSamples} />
        <Slider label="Temperature" value={temperature} min={0.1} max={2} step={0.1} onChange={setTemperature} />
        <Slider label="Top-p" value={topP} min={0.1} max={1} step={0.05} onChange={setTopP} />

        <div className="actions">
          <button className="primary" disabled={!!busy || candles.length === 0} onClick={runForecast}>
            Forecast
          </button>
          <button disabled={!!busy || candles.length === 0} onClick={runBacktest}>
            Backtest
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
          <span className="status">{busy ?? ''}</span>
        </div>

        {error && <div className="error">{error}</div>}

        {tab === 'forecast' ? (
          <>
            <div className="stats">
              <Stat label="Last close" value={candles[candles.length - 1]?.close.toFixed(2) ?? '—'} />
              <Stat
                label={`P(up) in ${forecast?.forecast.length ?? horizon} bars`}
                value={forecast ? rate(forecast.stats.p_up) : '—'}
                tone={forecast ? (forecast.stats.p_up >= 0.5 ? 'up' : 'down') : undefined}
              />
              <Stat
                label="Median expected move"
                value={forecast ? pct(forecast.stats.expected_return, 2) : '—'}
                tone={forecast ? (forecast.stats.expected_return >= 0 ? 'up' : 'down') : undefined}
              />
              <Stat label="p10–p90 band width" value={forecast ? pct(forecast.stats.band_width_pct).slice(1) : '—'} />
              <Stat label="Latency" value={forecast ? `${(forecast.latency_ms / 1000).toFixed(1)}s` : '—'} />
            </div>
            <PriceChart candles={candles} forecast={forecast} interval={interval} />
            <p className="legend">
              <span className="sw median" /> median of {forecast?.forecast.length ? samples : 'N'} sampled paths
              <span className="sw band" /> p10 / p90 band
            </p>
          </>
        ) : backtest ? (
          <>
            <div className="stats">
              <Stat
                label="Direction hit rate"
                value={rate(backtest.metrics.hit_rate)}
                hint={`baseline ${rate(backtest.metrics.baseline_hit_rate)}`}
                tone={backtest.metrics.hit_rate > backtest.metrics.baseline_hit_rate ? 'up' : 'down'}
              />
              <Stat
                label="Close MAE"
                value={pct(backtest.metrics.mae_pct, 2).slice(1)}
                hint={`naive ${pct(backtest.metrics.naive_mae_pct, 2).slice(1)}`}
                tone={backtest.metrics.mae_pct < backtest.metrics.naive_mae_pct ? 'up' : 'down'}
              />
              <Stat label="p10–p90 coverage" value={rate(backtest.metrics.coverage_p10_p90)} hint="ideal ≈ 80%" />
              <Stat label="Windows" value={String(backtest.metrics.windows)} />
              <Stat label="Latency" value={`${(backtest.latency_ms / 1000).toFixed(1)}s`} />
            </div>
            <BacktestChart result={backtest} interval={interval} />
            <p className="legend">
              <span className="sw actual" /> actual close <span className="sw median" /> forecast at horizon end
              <span className="sw band" /> p10 / p90 · dots below: green = direction right, red = wrong
            </p>
            <p className="muted small">
              Baseline hit rate = always guessing the majority direction for this period. Naive MAE = predicting
              no change. A model is only interesting if it beats both.
            </p>
          </>
        ) : (
          <div className="empty">Run a backtest to see how Kronos's past forecasts compared to what happened.</div>
        )}

        <p className="disclaimer">Research tool, not financial advice. Forecasts are probabilistic and often wrong.</p>
      </main>
    </div>
  )
}

function Stat({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: 'up' | 'down' }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${tone ?? ''}`}>{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  )
}

function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
}) {
  return (
    <label className="slider">
      <span>
        {label} <b>{value}</b>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}
