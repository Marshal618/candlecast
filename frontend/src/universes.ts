import { useEffect, useState } from 'react'
import { api, type ResearchProgress } from './api'

// Preset universes for the Research, Volatility and Portfolio tabs
export const UNIVERSES: Record<string, string[]> = {
  sectors: ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE', 'XLC'],
  'mega caps': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'JPM', 'V', 'MA', 'UNH', 'XOM', 'LLY', 'COST', 'HD', 'PG', 'JNJ', 'WMT', 'NFLX'],
  countries: ['SPY', 'EWJ', 'EWG', 'EWU', 'EWZ', 'EWC', 'EWA', 'FXI', 'INDA', 'EWY', 'EWT'],
  crypto: ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'LINK-USD'],
}

// Measured Kronos-mini throughput on an Apple M3, in sampled bars (paths x horizon) per second:
// ~170 paths/s at a 5-bar horizon, ~47 at 20. Batch size doesn't change it.
const BARS_PER_SECOND = 900

let indexCache: Promise<{ presets: Record<string, string[]>; asOf: string | null }> | null = null

/** The small presets plus the index constituent lists the backend ships (today's members). */
export function useUniverses() {
  const [state, setState] = useState<{ presets: Record<string, string[]>; asOf: string | null }>({ presets: UNIVERSES, asOf: null })
  useEffect(() => {
    indexCache ??= api
      .universes()
      .then((u) => ({ presets: { ...UNIVERSES, ...u.universes }, asOf: u.as_of }))
      .catch(() => ({ presets: UNIVERSES, asOf: null }))
    let live = true
    indexCache.then((s) => live && setState(s))
    return () => {
      live = false
    }
  }, [])
  return state
}

/** Polls the backend's forecast counter while a run is in flight. */
export function useProgress(busy: boolean) {
  const [p, setP] = useState<ResearchProgress | null>(null)
  useEffect(() => {
    if (!busy) {
      setP(null)
      return
    }
    const id = setInterval(() => api.progress().then(setP).catch(() => {}), 1500)
    return () => clearInterval(id)
  }, [busy])
  return p
}

/** Button label while running: live forecast progress if a model is sampling, else a plain note. */
export function runningLabel(p: ResearchProgress | null, fallback: string) {
  if (p?.label && p.total) return `▶ ${p.label}: ${p.done.toLocaleString()} / ${p.total.toLocaleString()} forecasts…`
  return `▶ ${fallback}…`
}

/** Worst-case model time for a run (cached forecasts are free), or null when it's quick. */
export function estimate(forecasts: number, samples: number, horizon: number, kronosOn: boolean) {
  if (!kronosOn) return null
  const s = (forecasts * samples * horizon) / BARS_PER_SECOND
  if (s < 60) return null
  return s < 3600 ? `up to ~${Math.round(s / 60)} min of kronos, cached after` : `up to ~${(s / 3600).toFixed(1)} h of kronos, cached after`
}
