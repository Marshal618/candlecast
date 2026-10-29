import { useEffect, useMemo, useRef, useState } from 'react'
import {
  createChart,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type SeriesType,
  type UTCTimestamp,
} from 'lightweight-charts'
import type { BacktestResponse, Candle, CrossSectionResponse, ForecastResponse, Interval, SimBot, SimPlayer, SimState, StrategyResult, TokensResponse } from './api'
import { COLORS, PLAYER_COLORS, signalColor } from './theme'


const ts = (t: number) => t as UTCTimestamp

// The chart effect's cleanup may already have removed the chart (e.g. on interval change)
function removeAll(chart: IChartApi, series: ISeriesApi<SeriesType>[]) {
  try {
    for (const s of series) chart.removeSeries(s)
  } catch {
    /* chart already disposed */
  }
}

// Blind sim games label bars by number instead of date
export type TimeLabel = (t: number) => string

function useChart(interval: Interval | '1h', timeLabel?: TimeLabel) {
  const ref = useRef<HTMLDivElement>(null)
  const [chart, setChart] = useState<IChartApi | null>(null)

  useEffect(() => {
    if (!chart || !timeLabel) return
    chart.applyOptions({
      localization: { timeFormatter: (t: number) => timeLabel(t) },
      timeScale: { tickMarkFormatter: (t: number) => timeLabel(t) },
    })
  }, [chart, timeLabel])

  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, {
      autoSize: true,
      layout: {
        background: { color: COLORS.bg },
        textColor: COLORS.text,
        fontFamily: "'IBM Plex Mono', ui-monospace, Menlo, monospace",
        fontSize: 11,
      },
      grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: '#4a5560', labelBackgroundColor: '#252c33' },
        horzLine: { color: '#4a5560', labelBackgroundColor: '#252c33' },
      },
      rightPriceScale: { borderColor: COLORS.grid },
      timeScale: { borderColor: COLORS.grid, timeVisible: interval !== '1d', rightOffset: 4 },
    })
    setChart(chart)
    return () => {
      setChart(null)
      chart.remove()
    }
  }, [interval])

  return { ref, chart }
}

type ForecastLike = Pick<ForecastResponse, 'forecast' | 'last_time' | 'last_close'>

export interface Overlay {
  name: string
  color: string
  forecast: ForecastLike
}

// Stable default so the chart effect doesn't rebuild (and reset zoom) on every parent render
const NO_OVERLAYS: Overlay[] = []

export function PriceChart({
  candles,
  forecast,
  interval,
  timeLabel,
  overlays = NO_OVERLAYS,
}: {
  candles: Candle[]
  forecast: ForecastLike | null
  interval: Interval
  timeLabel?: TimeLabel
  // Challenger forecasts drawn as a median line and band in their own colour
  overlays?: Overlay[]
}) {
  const { ref, chart } = useChart(interval, timeLabel)

  useEffect(() => {
    if (!chart || candles.length === 0) return

    const candleSeries = chart.addCandlestickSeries({
      upColor: COLORS.up,
      downColor: COLORS.down,
      borderVisible: false,
      wickUpColor: COLORS.up,
      wickDownColor: COLORS.down,
    })
    candleSeries.setData(candles.map((c) => ({ ...c, time: ts(c.time) })))

    const added: ISeriesApi<SeriesType>[] = [candleSeries]
    if (forecast && forecast.forecast.length > 0) {
      // Anchor forecast lines on the last real close so they connect visually
      const anchor = { time: ts(forecast.last_time), value: forecast.last_close }
      const line = (key: 'median' | 'p10' | 'p90') => [
        anchor,
        ...forecast.forecast.map((p) => ({ time: ts(p.time), value: p[key] })),
      ]

      const projected = chart.addCandlestickSeries({
        upColor: 'rgba(0, 0, 0, 0)',
        downColor: COLORS.projected,
        borderUpColor: COLORS.projected,
        borderDownColor: COLORS.projected,
        wickUpColor: COLORS.projected,
        wickDownColor: COLORS.projected,
        lastValueVisible: false,
        priceLineVisible: false,
      })
      projected.setData(
        forecast.forecast.map((p) => ({ time: ts(p.time), open: p.open, high: p.high, low: p.low, close: p.close })),
      )

      const median = chart.addLineSeries({
        color: COLORS.forecast,
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        lastValueVisible: false,
        priceLineVisible: false,
      })
      median.setData(line('median'))
      const p90 = chart.addLineSeries({
        color: COLORS.band,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        priceLineVisible: false,
      })
      p90.setData(line('p90'))
      const p10 = chart.addLineSeries({
        color: COLORS.band,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        priceLineVisible: false,
      })
      p10.setData(line('p10'))
      added.push(projected, median, p90, p10)

      // Calibrated band: Kronos's raw band widened (or narrowed) to hit its stated 80% coverage
      if (forecast.forecast[0].cal_p10 != null) {
        for (const key of ['cal_p10', 'cal_p90'] as const) {
          const band = chart.addLineSeries({
            color: COLORS.calBand,
            lineWidth: 1,
            lastValueVisible: true,
            priceLineVisible: false,
          })
          band.setData([anchor, ...forecast.forecast.map((p) => ({ time: ts(p.time), value: p[key] as number }))])
          added.push(band)
        }
      }

      candleSeries.setMarkers([
        { time: ts(forecast.last_time), position: 'aboveBar', color: COLORS.actual, shape: 'arrowDown', text: 'NOW' },
      ])
    }

    for (const o of overlays) {
      const anchor = { time: ts(o.forecast.last_time), value: o.forecast.last_close }
      const pts = o.forecast.forecast
      const calibrated = pts[0]?.cal_p10 != null
      const series = (pick: (p: ForecastLike['forecast'][number]) => number, style: LineStyle, width: 1 | 2) => {
        const line = chart.addLineSeries({
          color: o.color,
          lineWidth: width,
          lineStyle: style,
          lastValueVisible: false,
          priceLineVisible: false,
        })
        line.setData([anchor, ...pts.map((p) => ({ time: ts(p.time), value: pick(p) }))])
        added.push(line)
      }
      series((p) => p.median, LineStyle.Solid, 2)
      series((p) => (calibrated ? (p.cal_p10 as number) : p.p10), LineStyle.Dotted, 1)
      series((p) => (calibrated ? (p.cal_p90 as number) : p.p90), LineStyle.Dotted, 1)
    }

    // Show recent history plus the full forecast
    const total = candles.length + (forecast?.forecast.length ?? 0)
    chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, total - 180), to: total + 4 })

    return () => {
      removeAll(chart, added)
    }
  }, [candles, forecast, overlays, chart])

  return <div ref={ref} className="chart" />
}

export function BacktestChart({ result, interval }: { result: BacktestResponse; interval: Interval }) {
  const { ref, chart } = useChart(interval)

  useEffect(() => {
    if (!chart) return

    const actual = chart.addLineSeries({ color: COLORS.actual, lineWidth: 2, priceLineVisible: false })
    actual.setData(result.actual.map((p) => ({ time: ts(p.time), value: p.value })))

    const points = (key: 'pred_final' | 'p10_final' | 'p90_final') =>
      result.windows.map((w) => ({ time: ts(w.time), value: w[key] }))
    const pointSeries = (color: string) =>
      chart.addLineSeries({ color, lineVisible: false, pointMarkersVisible: true, lastValueVisible: false, priceLineVisible: false })

    const pred = pointSeries(COLORS.forecast)
    pred.setData(points('pred_final'))
    const hi = pointSeries(COLORS.band)
    hi.setData(points('p90_final'))
    const lo = pointSeries(COLORS.band)
    lo.setData(points('p10_final'))

    actual.setMarkers(
      result.windows.map((w) => ({
        time: ts(w.time),
        position: 'belowBar' as const,
        color: w.hit ? COLORS.up : COLORS.down,
        shape: 'circle' as const,
        size: 0.6,
      })),
    )
    chart.timeScale().fitContent()

    return () => {
      removeAll(chart, [actual, pred, hi, lo])
    }
  }, [result, chart])

  return <div ref={ref} className="chart" />
}

export function EquityChart({ result, interval }: { result: StrategyResult; interval: Interval }) {
  const { ref, chart } = useChart(interval)

  useEffect(() => {
    if (!chart) return
    const fmt = { type: 'custom' as const, formatter: (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%` }
    const bh = chart.addLineSeries({ color: COLORS.benchmark, lineWidth: 1, priceFormat: fmt, priceLineVisible: false })
    bh.setData(result.equity.map((p) => ({ time: ts(p.time), value: p.buy_hold - 1 })))
    const strat = chart.addLineSeries({ color: COLORS.actual, lineWidth: 2, priceFormat: fmt, priceLineVisible: false })
    strat.setData(result.equity.map((p) => ({ time: ts(p.time), value: p.strategy - 1 })))
    // Mark each window's decision where it was made
    strat.setMarkers(
      result.positions
        .filter((p) => p.position !== 0)
        .map((p) => ({
          time: ts(p.time),
          position: p.position > 0 ? ('belowBar' as const) : ('aboveBar' as const),
          color: p.position > 0 ? COLORS.up : COLORS.down,
          shape: p.position > 0 ? ('arrowUp' as const) : ('arrowDown' as const),
          size: 0.6,
        })),
    )
    chart.timeScale().fitContent()
    return () => {
      removeAll(chart, [bh, strat])
    }
  }, [result, chart])

  return <div ref={ref} className="chart chart-short" />
}

export type TokenOverlay = 'surprisal' | 'attribution'

export function TokenChart({
  result,
  candles,
  overlay,
  interval,
}: {
  result: TokensResponse
  candles: Candle[]
  overlay: TokenOverlay
  interval: Interval
}) {
  const { ref, chart } = useChart(interval)

  useEffect(() => {
    if (!chart) return
    const first = result.bars[0].time
    const ctx = candles.filter((c) => c.time >= first)

    const candleSeries = chart.addCandlestickSeries({
      upColor: COLORS.up,
      downColor: COLORS.down,
      borderVisible: false,
      wickUpColor: COLORS.up,
      wickDownColor: COLORS.down,
      priceLineVisible: false,
    })
    candleSeries.setData(ctx.map((c) => ({ ...c, time: ts(c.time) })))
    candleSeries.priceScale().applyOptions({ scaleMargins: { top: 0.05, bottom: 0.3 } })

    // What the coarse tokens alone "remember" of the price
    const coarse = chart.addLineSeries({
      color: COLORS.projected,
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      priceLineVisible: false,
      lastValueVisible: false,
    })
    coarse.setData(result.bars.map((b) => ({ time: ts(b.time), value: b.recon_coarse })))

    const hist = chart.addHistogramSeries({ priceScaleId: 'overlay', priceLineVisible: false, lastValueVisible: false })
    chart.priceScale('overlay').applyOptions({ scaleMargins: { top: 0.75, bottom: 0 } })
    if (overlay === 'surprisal') {
      const vals = result.bars.map((b) => b.surprisal ?? 0)
      const hi = [...vals].sort((a, b) => a - b)[Math.floor(vals.length * 0.9)]
      hist.setData(
        result.bars.map((b, i) => ({
          time: ts(b.time),
          value: vals[i],
          // Top decile of surprise stands out
          color: vals[i] >= hi ? COLORS.projected : 'rgba(122, 133, 140, 0.45)',
        })),
      )
    } else {
      const segs = result.attribution
      hist.setData(
        result.bars.map((b) => {
          const seg = segs.find((s) => b.time >= s.start && b.time <= s.end)
          const v = seg?.contribution ?? 0
          return { time: ts(b.time), value: v * 1e4, color: v >= 0 ? COLORS.up : COLORS.down }
        }),
      )
    }
    chart.timeScale().fitContent()
    return () => {
      removeAll(chart, [candleSeries, coarse, hist])
    }
  }, [result, candles, overlay, chart])

  return <div ref={ref} className="chart" />
}


export function RaceChart({ sim, timeLabel }: { sim: SimState; timeLabel?: TimeLabel }) {
  const { ref, chart } = useChart(sim.config.interval, timeLabel)

  useEffect(() => {
    if (!chart) return
    const fmt = { type: 'custom' as const, formatter: (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%` }
    const players: SimPlayer[] = ['buy_hold', ...sim.config.bots.filter((b): b is SimBot => b !== 'buy_hold'), 'you']
    const series = players.map((p) => {
      const line = chart.addLineSeries({
        color: PLAYER_COLORS[p],
        lineWidth: p === 'you' ? 2 : 1,
        priceFormat: fmt,
        priceLineVisible: false,
      })
      line.setData(
        sim.curve
          .filter((c) => c[p] != null)
          .map((c) => ({ time: ts(c.time), value: (c[p] as number) / sim.config.cash - 1 })),
      )
      return line
    })
    chart.timeScale().fitContent()
    return () => {
      removeAll(chart, series)
    }
  }, [sim, chart])

  return <div ref={ref} className="chart chart-short" />
}

export interface Curve {
  values: number[] // growth of 1, one per period
  color: string
  width?: 1 | 2
  style?: LineStyle
}

// Cumulative-return lines over rebalance periods; each period's return lands at the next rebalance
export function CurvesChart({ interval, times, curves }: { interval: Interval | '1h'; times: number[]; curves: Curve[] }) {
  const { ref, chart } = useChart(interval)

  useEffect(() => {
    if (!chart || times.length < 2) return
    const fmt = { type: 'custom' as const, formatter: (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%` }
    const at = [...times, times.at(-1)! + (times.at(-1)! - times.at(-2)!)]
    const series = curves.map((c) => {
      const s = chart.addLineSeries({
        color: c.color, lineWidth: c.width ?? 1, lineStyle: c.style ?? LineStyle.Solid,
        priceFormat: fmt, priceLineVisible: false, lastValueVisible: false,
      })
      s.setData([0, ...c.values.map((v) => v - 1)].map((value, i) => ({ time: ts(at[i]), value })))
      return s
    })
    chart.timeScale().fitContent()
    return () => {
      removeAll(chart, series)
    }
  }, [times, curves, chart])

  return <div ref={ref} className="chart chart-short" />
}

// Cumulative net long-short return of every signal, plus the equal-weight universe for scale
export function LongShortChart({ result, selected }: { result: CrossSectionResponse; selected: string | null }) {
  const curves = useMemo<Curve[]>(
    () => [
      { values: result.universe.equity, color: COLORS.benchmark, style: LineStyle.Dotted },
      ...result.signals.map((s) => ({
        values: s.equity,
        color: signalColor(s.name),
        width: (s.name === selected ? 2 : 1) as 1 | 2,
        style: s.kind === 'factor' ? LineStyle.Dashed : s.kind === 'fundamental' ? LineStyle.SparseDotted : LineStyle.Solid,
      })),
    ],
    [result, selected],
  )
  return <CurvesChart interval={result.interval} times={result.origin_times} curves={curves} />
}

export interface VolLine {
  values: (number | null)[] // annualized vol per rebalance; null = no value (calibration warm-up)
  color: string
  width?: 1 | 2
  style?: LineStyle
}

// Forecast vs realized volatility at each rebalance, as annualized percentages
export function VolChart({ interval, times, lines }: { interval: Interval | '1h'; times: number[]; lines: VolLine[] }) {
  const { ref, chart } = useChart(interval)

  useEffect(() => {
    if (!chart || times.length < 2) return
    const fmt = { type: 'custom' as const, formatter: (v: number) => `${(v * 100).toFixed(0)}%` }
    const series = lines.map((l) => {
      const s = chart.addLineSeries({
        color: l.color, lineWidth: l.width ?? 1, lineStyle: l.style ?? LineStyle.Solid,
        priceFormat: fmt, priceLineVisible: false, lastValueVisible: false,
      })
      s.setData(l.values.map((value, i) => (value == null ? { time: ts(times[i]) } : { time: ts(times[i]), value })))
      return s
    })
    chart.timeScale().fitContent()
    return () => {
      removeAll(chart, series)
    }
  }, [times, lines, chart])

  return <div ref={ref} className="chart chart-short" />
}
