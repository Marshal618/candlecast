export type Interval = '5m' | '15m' | '1h' | '1d'
export type ModelName = 'mini' | 'small' | 'base'

export interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface ForecastPoint {
  time: number
  median: number
  p10: number
  p90: number
  open: number
  high: number
  low: number
  close: number
}

export interface ForecastResponse {
  symbol: string
  interval: Interval
  model: ModelName
  journal: { id: number; new: boolean }
  lookback_used: number
  last_close: number
  last_time: number
  latency_ms: number
  stats: { p_up: number; expected_return: number; band_width_pct: number }
  forecast: ForecastPoint[]
}

export interface BacktestWindow {
  time: number
  origin_time: number
  last_close: number
  pred_final: number
  p10_final: number
  p90_final: number
  actual_final: number
  p_up: number
  hit: boolean
  covered: boolean
  mae_pct: number
  naive_mae_pct: number
}

export type StrategyMode = 'long_only' | 'long_short'

export interface StrategyParams {
  mode: StrategyMode
  enter_p: number
  cost_bps: number
}

export interface StrategyResult {
  params: StrategyParams
  metrics: {
    total_return: number
    bh_return: number
    sharpe: number
    bh_sharpe: number
    max_drawdown: number
    bh_max_drawdown: number
    exposure: number
    trades: number
    active_windows: number
    win_rate: number | null
    p_value: number | null
  }
  equity: { time: number; strategy: number; buy_hold: number }[]
  positions: { time: number; position: number }[]
}

export interface BacktestResponse {
  symbol: string
  interval: Interval
  model: ModelName
  horizon: number
  latency_ms: number
  metrics: {
    windows: number
    hit_rate: number
    baseline_hit_rate: number
    coverage_p10_p90: number
    mae_pct: number
    naive_mae_pct: number
    r2_oos: number
    ridge_r2_oos: number | null
    ridge_hit_rate: number | null
  }
  windows: BacktestWindow[]
  actual: { time: number; value: number }[]
  strategy: StrategyResult
}

export interface TokenPrototype {
  count: number
  mean_return: number
  p_up: number
  level: number
  volume: number
}

export interface TokensResponse {
  symbol: string
  interval: Interval
  model: ModelName
  latency_ms: number
  lookback_used: number
  vocab: { s1: number; s2: number }
  bars: {
    time: number
    s1: number
    s2: number
    surprisal: number | null
    recon_full: number
    recon_coarse: number
  }[]
  summary: {
    unique_s1: number
    unique_s2: number
    usage_entropy_bits: number
    mean_surprisal_bits: number
    perplexity: number
    recon_mae_full_pct: number
    recon_mae_coarse_pct: number
  }
  next: {
    entropy_bits: number
    max_entropy_bits: number
    implied_return: number | null
    implied_p_up: number | null
    coverage: number
    top: { s1: number; p: number; prototype: TokenPrototype | null }[]
  }
  semantics: {
    bars: number
    tokens_used: number
    chance: number
    explained: Record<'level' | 'volume' | 'return' | 'body' | 'range', number>
  }
  attribution: { start: number; end: number; kl_bits: number; contribution: number | null }[]
}

export interface Scorecard {
  n: number
  hit_rate: number
  baseline_hit_rate: number
  r2_oos: number
  coverage_p10_p90: number
  mae_pct: number
  brier: number
  brier_skill: number
  p_value: number | null
  calibration: { lo: number; hi: number; n: number; predicted: number; observed: number }[]
}

export interface JournalEntry {
  id: number
  created_at: number
  symbol: string
  interval: Interval
  model: ModelName
  horizon: number
  origin_time: number
  last_close: number
  p_up: number
  expected_return: number
  actual_final: number | null
  actual_time: number | null
  scored_at: number | null
  actual_return?: number
  hit?: boolean
}

export interface JournalReport {
  total: number
  pending: number
  scorecard: Scorecard | null
  groups: {
    symbol: string
    interval: Interval
    model: ModelName
    total: number
    live: Scorecard | null
    backtest: {
      horizon: number
      windows: number
      hit_rate: number
      baseline_hit_rate: number
      r2_oos: number
      coverage: number
      created_at: number
    } | null
  }[]
  entries: JournalEntry[]
}

export interface Health {
  device: string
  loaded_models: string[]
  models: Record<ModelName, { max_context: number }>
}

// Same-origin by default; set VITE_API_BASE when the API is hosted elsewhere
const BASE = import.meta.env.VITE_API_BASE ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const detail = body?.detail
    throw new Error(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  health: () => request<Health>('/api/health'),
  candles: (symbol: string, interval: Interval, limit = 2100) =>
    request<{ candles: Candle[] }>(
      `/api/candles?symbol=${encodeURIComponent(symbol)}&interval=${interval}&limit=${limit}`,
    ),
  forecast: (body: Record<string, unknown>) =>
    request<ForecastResponse>('/api/forecast', { method: 'POST', body: JSON.stringify(body) }),
  backtest: (body: Record<string, unknown>) =>
    request<BacktestResponse>('/api/backtest', { method: 'POST', body: JSON.stringify(body) }),
  journal: (filters: { symbol?: string; interval?: Interval; model?: ModelName } = {}) => {
    const q = new URLSearchParams(Object.entries(filters).filter(([, v]) => v) as [string, string][])
    return request<JournalReport>(`/api/journal?${q}`)
  },
  tokens: (body: Record<string, unknown>) =>
    request<TokensResponse>('/api/tokens', { method: 'POST', body: JSON.stringify(body) }),
  strategy: (body: Record<string, unknown>, signal?: AbortSignal) =>
    request<StrategyResult>('/api/strategy', { method: 'POST', body: JSON.stringify(body), signal }),
}
