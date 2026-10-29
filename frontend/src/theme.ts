import type { Factor, Fundamental, ModelName, SignalName, SimPlayer } from './api'

export const COLORS = {
  bg: '#101418',
  grid: '#1b2127',
  text: '#7a858c',
  up: '#8fb39a',
  down: '#c0603f',
  // Projected bars use their own hue so they can't be mistaken for real history:
  // hollow = projected up, solid = projected down
  projected: '#d8b45c',
  forecast: '#d8b45c',
  band: 'rgba(216, 180, 92, 0.45)',
  calBand: '#d8b45c',
  actual: '#5fb3a1',
  benchmark: '#7a858c',
}

export const PLAYER_COLORS: Record<SimPlayer, string> = {
  you: COLORS.actual,
  buy_hold: COLORS.benchmark,
  sma_cross: '#8fa8d8',
  kronos: COLORS.projected,
  portfolio: '#c28fd6',
}

// Kronos sizes share the brass of the projected candles; each challenger gets its own hue
export const MODEL_COLORS: Record<ModelName, string> = {
  mini: COLORS.projected,
  small: COLORS.projected,
  base: COLORS.projected,
  naive: '#9aa4ab',
  garch: '#8fa8d8',
  chronos: '#c28fd6',
}

export const MODEL_LABELS: Record<ModelName, string> = {
  mini: 'kronos-mini',
  small: 'kronos-small',
  base: 'kronos-base',
  naive: 'naive (hist. avg + bootstrap)',
  garch: 'garch(1,1)-t',
  chronos: 'chronos-bolt',
}

// Price factors in the Research tab, drawn dashed so they read as yardsticks rather than models
export const FACTOR_COLORS: Record<Factor, string> = {
  momentum: '#b8b07a',
  reversal: '#7ec4cf',
}

export const FACTOR_LABELS: Record<Factor, string> = {
  momentum: 'momentum 60-5',
  reversal: 'short-term reversal',
}

// Non-price signals: muted earth tones, distinct from the model and price-factor hues
export const FUNDAMENTAL_COLORS: Record<Fundamental, string> = {
  surprise: '#d68f6a',
  revisions: '#a3c47a',
  value: '#c9a2a2',
  size: '#8ab0b8',
}

export const FUNDAMENTAL_LABELS: Record<Fundamental, string> = {
  surprise: 'earnings surprise',
  revisions: 'analyst revisions',
  value: 'earnings yield',
  size: 'small size',
}

export const FUNDAMENTAL_HINTS: Record<Fundamental, string> = {
  surprise: 'latest EPS vs consensus, if reported in the last 100 days (post-earnings drift)',
  revisions: 'upgrades + target raises minus downgrades + cuts, last 90 days',
  value: 'last 4 quarters of reported EPS / price',
  size: 'minus log market cap: smaller names score higher',
}

export const signalLabel = (s: SignalName) =>
  s in FACTOR_LABELS ? FACTOR_LABELS[s as Factor] : s in FUNDAMENTAL_LABELS ? FUNDAMENTAL_LABELS[s as Fundamental] : MODEL_LABELS[s as ModelName]
export const signalColor = (s: SignalName) =>
  s in FACTOR_COLORS ? FACTOR_COLORS[s as Factor] : s in FUNDAMENTAL_COLORS ? FUNDAMENTAL_COLORS[s as Fundamental] : MODEL_COLORS[s as ModelName]
