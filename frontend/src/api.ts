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

export interface BacktestResponse {
  symbol: string
  interval: Interval
  model: ModelName
  latency_ms: number
  metrics: {
    windows: number
    hit_rate: number
    baseline_hit_rate: number
    coverage_p10_p90: number
    mae_pct: number
    naive_mae_pct: number
  }
  windows: BacktestWindow[]
  actual: { time: number; value: number }[]
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
  candles: (symbol: string, interval: Interval, limit = 500) =>
    request<{ candles: Candle[] }>(
      `/api/candles?symbol=${encodeURIComponent(symbol)}&interval=${interval}&limit=${limit}`,
    ),
  forecast: (body: Record<string, unknown>) =>
    request<ForecastResponse>('/api/forecast', { method: 'POST', body: JSON.stringify(body) }),
  backtest: (body: Record<string, unknown>) =>
    request<BacktestResponse>('/api/backtest', { method: 'POST', body: JSON.stringify(body) }),
}
