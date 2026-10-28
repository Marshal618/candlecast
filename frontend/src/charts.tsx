import { useEffect, useRef, useState } from 'react'
import {
  createChart,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type SeriesType,
  type UTCTimestamp,
} from 'lightweight-charts'
import type { BacktestResponse, Candle, ForecastResponse, Interval, SimBot, SimPlayer, SimState, StrategyResult, TokensResponse } from './api'
import { COLORS, PLAYER_COLORS } from './theme'


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

export function PriceChart({
  candles,
  forecast,
  interval,
  timeLabel,
}: {
  candles: Candle[]
  forecast: Pick<ForecastResponse, 'forecast' | 'last_time' | 'last_close'> | null
  interval: Interval
  timeLabel?: TimeLabel
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

    // Show recent history plus the full forecast
    const total = candles.length + (forecast?.forecast.length ?? 0)
    chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, total - 180), to: total + 4 })

    return () => {
      removeAll(chart, added)
    }
  }, [candles, forecast, chart])

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
