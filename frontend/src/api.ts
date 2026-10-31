export type Interval = '5m' | '15m' | '1h' | '1d'
export type KronosName = 'mini' | 'small' | 'base'
// Challengers that every Kronos result can be compared against (backend/app/services/challengers.py)
export type Challenger = 'naive' | 'garch' | 'chronos'
export type ModelName = KronosName | Challenger

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
  // Conformally calibrated band, when a backtest has calibrated this series and horizon
  cal_p10?: number
  cal_p90?: number
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
  stats: { p_up: number; expected_return: number; band_width_pct: number; band_width_cal_pct?: number }
  forecast: ForecastPoint[]
  calibration?: (Omit<Calibration, 'q' | 'widening'> & { created_at: number }) | null
  calibration_mismatch?: boolean
}

export interface Calibration {
  target: number
  windows: number
  q: number[]
  oos_windows: number
  raw_coverage: number
  calibrated_coverage: number
  widening: number
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
  calibration: Calibration | null
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
  coverage_calibrated: { n: number; raw: number; calibrated: number } | null
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

export type Factor = 'momentum' | 'reversal'
// Non-price signals rebuilt point-in-time (backend services/fundamentals.py)
export type Fundamental = 'surprise' | 'revisions' | 'value' | 'size'

export interface CrossSectionRequest {
  symbols: string[]
  interval: '1d' | '1h'
  models: ModelName[]
  lookback: number
  horizon: number
  periods: number
  buckets: number
  samples: number
  cost_bps: number
  fundamentals?: Fundamental[]
}

export interface IcStats {
  mean: number
  std: number
  icir: number
  t_stat: number
  positive: number
}

export interface CrossSectionSignal {
  name: SignalName
  kind: 'model' | 'factor' | 'fundamental'
  // fundamentals only: share of (rebalance, symbol) cells with real data; the rest sit mid-ranking
  coverage?: number
  ic: IcStats
  rank_ic: IcStats
  rank_ic_residual?: IcStats
  rank_ic_series: (number | null)[]
  bucket_returns: number[]
  long_short: {
    gross_mean: number
    net_mean: number
    ann_return: number
    sharpe: number
    max_drawdown: number
    hit_rate: number
    turnover: number
    // cost per unit traded (bps) that would eat the whole gross edge; null if there's no edge
    breakeven_bps: number | null
  }
  // how similar each ranking is to the last one (high = low turnover)
  rank_autocorr: number
  // mean rank ic against returns 0, 1, 2, 3 periods ahead
  ic_decay: number[]
  equity: number[]
  corr: Partial<Record<Factor, number>>
  // only when most symbols have a GICS sector
  sector?: SectorReport
}

export interface SectorReport {
  rank_ic: IcStats // signal and returns both ranked within sectors
  long_short: CrossSectionSignal['long_short'] // bucketed on within-sector ranks
  by_sector: Record<string, IcStats & { n: number }> // sectors with at least 5 members
  exposure: Record<string, number> // the plain long-short's average net weight per sector
}

// Symbols left out of a run: `dropped` had too little history, `unavailable` had no data at all
export interface RunCoverage {
  dropped: string[]
  unavailable: string[]
}

export interface IndexUniverses {
  as_of: string
  source: string
  universes: Record<string, string[]>
  sectors: Record<string, string>
}

export interface EventStudyRequest {
  symbols: string[]
  pre: number
  post: number
  groups: number
}

export interface EventGroup {
  group: number // 0 = lowest surprise
  n: number
  surprise_median: number
  car: number[] // mean cumulative abnormal return per day in `days`, 0 on day -1
  se: number[]
  runup: number
  reaction: number
  drift: number
}

export interface SpreadStat {
  mean: number
  t_stat: number
  n: number // quarters
}

export interface EventStudyResponse extends Pick<RunCoverage, 'unavailable'> {
  pre: number
  post: number
  latency_ms: number
  symbols: string[]
  no_earnings: string[]
  events: number
  skipped: number
  first: number
  last: number
  quarters: number
  days: number[]
  groups: EventGroup[]
  spread: {
    runup: SpreadStat | null
    reaction: SpreadStat
    drift: SpreadStat
    checkpoints: (SpreadStat & { day: number })[]
  }
}

export interface ResearchProgress {
  label: string | null
  done: number
  total: number
}

export interface CrossSectionResponse extends RunCoverage {
  interval: '1d' | '1h'
  horizon: number
  buckets: number
  latency_ms: number
  symbols: string[]
  periods: number
  origin_times: number[]
  periods_per_year: number
  universe: { equity: number[]; mean: number }
  dispersion: number
  sectors: Record<string, number> | null // members per sector, when the run had sector checks
  signals: CrossSectionSignal[]
}

export type SignalName = ModelName | Factor | Fundamental

export interface VolatilityRequest {
  symbols: string[]
  interval: '1d' | '1h'
  models: ModelName[]
  lookback: number
  horizon: number
  periods: number
  samples: number
}

export interface VolForecast {
  name: ModelName | 'ewma'
  kind: 'model' | 'baseline'
  qlike: number
  qlike_cal: number
  skill: number
  dm_t: number
  mz_slope: number
  mz_r2: number
  rank_ic: IcStats
  timing_corr: number
  vol_ratio: number
  // (periods, symbols) annualized vol; calibrated is null during the warm-up periods
  raw: number[][]
  calibrated: (number | null)[][]
}

export interface VolatilityResponse extends RunCoverage {
  interval: '1d' | '1h'
  horizon: number
  latency_ms: number
  symbols: string[]
  periods: number
  scored_periods: number
  calibration_periods: number
  origin_times: number[]
  realized: number[][]
  avg_realized_vol: number
  forecasts: VolForecast[]
}

export interface Construction {
  method: 'equal' | 'inverse_vol' | 'mean_variance'
  side: 'long_short' | 'long_only'
  select: number
  risk_window: number
  vol_target: number | null
  max_gross: number
  trade_rate: number
}

export interface PortfolioRequest {
  symbols: string[]
  interval: '1d' | '1h'
  signals: { name: SignalName; weight: number }[]
  weighting: 'manual' | 'ic'
  ic_window: number
  construction: Construction
  lookback: number
  horizon: number
  periods: number
  samples: number
  cost_bps: number
}

export interface PortfolioMetrics {
  ann_return: number
  ann_vol: number
  sharpe: number
  sharpe_period: number
  max_drawdown: number
  hit_rate: number
  turnover: number
  avg_gross: number
  avg_net: number
  ex_ante_vol: number
  cost_drag: number
  bench_ann_return: number
  skew: number
  kurtosis: number
}

export interface Trial {
  id: number
  name: string | null
  created_at?: number
  config: Omit<PortfolioRequest, 'symbols'>
  metrics: PortfolioMetrics
  final_blend?: Partial<Record<SignalName, number>>
}

export interface PortfolioResponse extends RunCoverage {
  interval: '1d' | '1h'
  horizon: number
  latency_ms: number
  trial_id: number
  deflation: { trials: number; psr: number; dsr: number; luck_sharpe: number }
  study: Trial[]
  symbols: string[]
  origin_times: number[]
  periods: number
  periods_per_year: number
  signals: SignalName[]
  equity: number[]
  benchmark: number[]
  weights: number[][]
  blend: number[][]
  final_blend: Partial<Record<SignalName, number>>
  metrics: PortfolioMetrics
}

export type SimBot = 'sma_cross' | 'kronos' | 'portfolio'
export type SimPlayer = 'you' | 'buy_hold' | SimBot
export type OrderSide = 'buy' | 'sell'
export type OrderType = 'market' | 'limit' | 'stop'

export interface SimConfig {
  symbols: string[]
  interval: '1d' | '1h'
  cash: number
  length: number
  start: string
  blind: boolean
  commission_bps: number
  slippage_bps: number
  bots: SimBot[]
  kronos_model: KronosName
  recipe_id: number | null
}

export interface SimOrder {
  id: number
  symbol: string
  side: OrderSide
  type: OrderType
  qty: number
  price: number | null
  status: 'open' | 'filled' | 'cancelled' | 'rejected'
  filled_qty?: number
  fill_price?: number
  reason?: string
  placed_time: number
  filled_time: number | null
}

export interface PlayerStats {
  equity: number
  trades: number
  fees: number
  cash: number
  return: number
  sharpe: number
  max_drawdown: number
}

export interface SimState {
  id: string
  config: Omit<SimConfig, 'start' | 'bots'> & { bots: ('buy_hold' | SimBot)[] }
  step: number
  time: number
  finished: boolean
  hints: number
  blind: { base: number; bar_seconds: number } | null
  period: [number, number] | null
  quotes: { symbol: string; last: number; change: number; traded_this_bar: boolean }[]
  candles: Record<string, Candle[]>
  account: {
    cash: number
    equity: number
    realized: number
    fees: number
    trades: number
    positions: { symbol: string; qty: number; avg_cost: number; last: number; value: number; unrealized: number; weight: number }[]
  }
  orders: SimOrder[]
  curve: ({ time: number } & Partial<Record<SimPlayer, number>>)[]
  players: Partial<Record<SimPlayer, PlayerStats>>
  kronos: { p_up: number[]; weights: number[] } | null
  recipe: { id: number; name: string; blend: Partial<Record<SignalName, number>>; horizon: number } | null
  result: Partial<Record<SimPlayer, { return: number; sharpe: number; max_drawdown: number; trades: number }>> | null
}

export interface SimSummary {
  id: string
  created_at: number
  symbols: string[]
  interval: string
  blind: boolean
  length: number
  played: number
  hints: number
  finished: boolean
  result: SimState['result']
  period: [number, number] | null
}

export type SimHint = Pick<ForecastResponse, 'symbol' | 'last_close' | 'last_time' | 'stats' | 'forecast'>

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
  sim: {
    list: () => request<SimSummary[]>('/api/sim'),
    create: (cfg: SimConfig) => request<SimState>('/api/sim', { method: 'POST', body: JSON.stringify(cfg) }),
    get: (id: string) => request<SimState>(`/api/sim/${id}`),
    step: (id: string, bars: number) =>
      request<SimState>(`/api/sim/${id}/step`, { method: 'POST', body: JSON.stringify({ bars }) }),
    order: (id: string, o: { symbol: string; side: OrderSide; qty: number; type: OrderType; price: number | null }) =>
      request<SimState>(`/api/sim/${id}/orders`, { method: 'POST', body: JSON.stringify(o) }),
    cancel: (id: string, orderId: number) => request<SimState>(`/api/sim/${id}/orders/${orderId}`, { method: 'DELETE' }),
    hint: (id: string, symbol: string, horizon = 10) =>
      request<SimHint>(`/api/sim/${id}/hint`, { method: 'POST', body: JSON.stringify({ symbol, horizon }) }),
    finish: (id: string) => request<SimState>(`/api/sim/${id}/finish`, { method: 'POST' }),
  },
  portfolio: {
    run: (body: PortfolioRequest) =>
      request<PortfolioResponse>('/api/portfolio', { method: 'POST', body: JSON.stringify(body) }),
    recipes: () => request<Trial[]>('/api/portfolio/recipes'),
    save: (trial_id: number, name: string) =>
      request<Trial>('/api/portfolio/recipes', { method: 'POST', body: JSON.stringify({ trial_id, name }) }),
  },
  universes: () => request<IndexUniverses>('/api/universes'),
  progress: () => request<ResearchProgress>('/api/progress'),
  volatility: (body: VolatilityRequest) =>
    request<VolatilityResponse>('/api/volatility', { method: 'POST', body: JSON.stringify(body) }),
  events: (body: EventStudyRequest) =>
    request<EventStudyResponse>('/api/events', { method: 'POST', body: JSON.stringify(body) }),
  crosssection: (body: CrossSectionRequest) =>
    request<CrossSectionResponse>('/api/crosssection', { method: 'POST', body: JSON.stringify(body) }),
  tokens: (body: Record<string, unknown>) =>
    request<TokensResponse>('/api/tokens', { method: 'POST', body: JSON.stringify(body) }),
  strategy: (body: Record<string, unknown>, signal?: AbortSignal) =>
    request<StrategyResult>('/api/strategy', { method: 'POST', body: JSON.stringify(body), signal }),
}
