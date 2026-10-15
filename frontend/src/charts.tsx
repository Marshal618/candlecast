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
import type { BacktestResponse, Candle, ForecastResponse, Interval } from './api'

const COLORS = {
  bg: '#0d1117',
  grid: '#1c2330',
  text: '#8b949e',
  up: '#26a69a',
  down: '#ef5350',
  forecast: '#f5a524',
  band: 'rgba(245, 165, 36, 0.55)',
  actual: '#58a6ff',
}

const ts = (t: number) => t as UTCTimestamp

// The chart effect's cleanup may already have removed the chart (e.g. on interval change)
function removeAll(chart: IChartApi, series: ISeriesApi<SeriesType>[]) {
  try {
    for (const s of series) chart.removeSeries(s)
  } catch {
    /* chart already disposed */
  }
}

function useChart(interval: Interval) {
  const ref = useRef<HTMLDivElement>(null)
  const [chart, setChart] = useState<IChartApi | null>(null)

  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, {
      autoSize: true,
      layout: { background: { color: COLORS.bg }, textColor: COLORS.text },
      grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
      crosshair: { mode: CrosshairMode.Normal },
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
}: {
  candles: Candle[]
  forecast: ForecastResponse | null
  interval: Interval
}) {
  const { ref, chart } = useChart(interval)

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

      const median = chart.addLineSeries({ color: COLORS.forecast, lineWidth: 2, title: 'median' })
      median.setData(line('median'))
      const p90 = chart.addLineSeries({
        color: COLORS.band,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: 'p90',
      })
      p90.setData(line('p90'))
      const p10 = chart.addLineSeries({
        color: COLORS.band,
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        title: 'p10',
      })
      p10.setData(line('p10'))
      added.push(median, p90, p10)

      candleSeries.setMarkers([
        { time: ts(forecast.last_time), position: 'aboveBar', color: COLORS.forecast, shape: 'arrowDown', text: 'now' },
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

    const actual = chart.addLineSeries({ color: COLORS.actual, lineWidth: 2, title: 'actual' })
    actual.setData(result.actual.map((p) => ({ time: ts(p.time), value: p.value })))

    const points = (key: 'pred_final' | 'p10_final' | 'p90_final') =>
      result.windows.map((w) => ({ time: ts(w.time), value: w[key] }))
    const pointSeries = (color: string, title: string) =>
      chart.addLineSeries({ color, lineVisible: false, pointMarkersVisible: true, lastValueVisible: false, title })

    const pred = pointSeries(COLORS.forecast, 'forecast')
    pred.setData(points('pred_final'))
    const hi = pointSeries(COLORS.band, 'p90')
    hi.setData(points('p90_final'))
    const lo = pointSeries(COLORS.band, 'p10')
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
