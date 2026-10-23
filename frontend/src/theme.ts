import type { SimPlayer } from './api'

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
  actual: '#5fb3a1',
  benchmark: '#7a858c',
}

export const PLAYER_COLORS: Record<SimPlayer, string> = {
  you: COLORS.actual,
  buy_hold: COLORS.benchmark,
  sma_cross: '#8fa8d8',
  kronos: COLORS.projected,
}
