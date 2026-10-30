export function Stat({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: 'up' | 'down' }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${tone ?? ''}`}>{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  )
}

export function Slider({
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

// Which requested symbols a run left out, and why
export function RunNote({ dropped, unavailable }: { dropped?: string[]; unavailable?: string[] }) {
  if (!dropped?.length && !unavailable?.length) return null
  const list = (xs: string[]) => (xs.length > 12 ? `${xs.slice(0, 12).join(', ')} +${xs.length - 12} more` : xs.join(', '))
  return (
    <p className="muted small">
      {!!dropped?.length && <>left out for too little history (recent listings): {list(dropped)}. </>}
      {!!unavailable?.length && <>no data from yahoo (renamed or delisted?): {list(unavailable)}.</>}
    </p>
  )
}
