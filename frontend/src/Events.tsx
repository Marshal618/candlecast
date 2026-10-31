import { useMemo, useState } from 'react'
import { api, type EventStudyResponse, type SpreadStat } from './api'
import { EventChart, type EventLine } from './charts'
import { COLORS } from './theme'
import { pct } from './format'
import { RunNote, Slider, Stat } from './ui'
import { useUniverses } from './universes'

const num = (x: number, d = 2) => (x >= 0 ? '+' : '') + x.toFixed(d)
const tone = (s: SpreadStat) => (s.t_stat >= 2 ? 'up' : s.t_stat <= -2 ? 'down' : '')
const date = (t: number) => new Date(t * 1000).toISOString().slice(0, 10)

// Biggest misses in rust, biggest beats in verdigris, the middle in slate
function groupColor(k: number, n: number) {
  const mix = (a: string, b: string, f: number) => {
    const h = (c: string, i: number) => parseInt(c.slice(1 + 2 * i, 3 + 2 * i), 16)
    return `rgb(${[0, 1, 2].map((i) => Math.round(h(a, i) + (h(b, i) - h(a, i)) * f)).join(',')})`
  }
  const f = n === 1 ? 0.5 : k / (n - 1)
  return f < 0.5 ? mix(COLORS.down, COLORS.text, f * 2) : mix(COLORS.text, COLORS.up, (f - 0.5) * 2)
}

const groupName = (k: number, n: number) => (k === 0 ? 'biggest misses' : k === n - 1 ? 'biggest beats' : `group ${k + 1}`)

export function EventsView({ onError }: { onError: (msg: string | null) => void }) {
  const { presets, asOf } = useUniverses()
  const [symbols, setSymbols] = useState('')
  const [pre, setPre] = useState(5)
  const [post, setPost] = useState(60)
  const [groups, setGroups] = useState(5)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<EventStudyResponse | null>(null)

  // Default to the S&P 500 once the index lists arrive; until then the mega caps
  const text = symbols || (presets['S&P 500'] ?? presets['mega caps']).join(', ')
  const list = [...new Set(text.split(/[\s,]+/).map((s) => s.trim().toUpperCase()).filter(Boolean))]

  const run = async () => {
    setBusy(true)
    onError(null)
    try {
      setResult(await api.events({ symbols: list, pre, post, groups }))
    } catch (e) {
      onError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const lines = useMemo<EventLine[]>(
    () =>
      result?.groups.map((g) => ({
        values: g.car,
        color: groupColor(g.group, result.groups.length),
        width: (g.group === 0 || g.group === result.groups.length - 1 ? 2 : 1) as 1 | 2,
      })) ?? [],
    [result],
  )
  const n = result?.groups.length ?? 0
  const sp = result?.spread

  return (
    <>
      <div className="sim-setup">
        <div className="panel-title">Earnings event study</div>
        <p className="muted small">
          every earnings report in the universe lined up on its own day 0 (the first close after it was public), grouped
          by how much it beat or missed consensus, ranked within its earnings season. lines show the average return beyond
          the rest of the universe. the jump at day 0 is the reaction; anything after it is drift you could have bought
          at day 0's close (post-earnings-announcement drift, Bernard &amp; Thomas 1989).
        </p>
        <div className="setup-grid">
          <label>
            stocks ({list.length}, 4–700)
            <input className="field" value={text} onChange={(e) => setSymbols(e.target.value)} spellCheck={false} />
            <div className="quick">
              {Object.entries(presets).map(([name, u]) => (
                <button key={name} onClick={() => setSymbols(u.join(', '))}>
                  {name}
                </button>
              ))}
            </div>
            {list.length > 30 && asOf && (
              <span className="muted small">
                index lists are today's members (as of {asOf}): companies that collapsed after bad reports are missing
              </span>
            )}
          </label>
          <Slider label="days before" value={pre} min={0} max={20} onChange={setPre} />
          <Slider label="days after" value={post} min={5} max={120} onChange={setPost} />
          <Slider label="surprise groups" value={groups} min={2} max={10} onChange={setGroups} />
        </div>
        <button className="primary start" disabled={busy || list.length < 4} onClick={run}>
          {busy ? '▶ loading earnings…' : '▶ Run event study'}
        </button>
        {!busy && list.length > 100 && <span className="muted small"> first run fetches every company's earnings history (a few minutes); cached after</span>}
      </div>

      {result && sp && (
        <>
          <div className="stats">
            <Stat label="reports" value={result.events.toLocaleString()} hint={`${result.symbols.length} companies, ${result.quarters} quarters`} />
            <Stat label="span" value={`${date(result.first).slice(0, 7)} → ${date(result.last).slice(0, 7)}`} />
            <Stat label="reaction gap" value={pct(sp.reaction.mean, 2)} hint={`beats − misses, day 0 · t ${num(sp.reaction.t_stat, 1)}`} tone={sp.reaction.mean > 0 ? 'up' : 'down'} />
            <Stat label={`drift gap, +${result.post}d`} value={pct(sp.drift.mean, 2)} hint={`after day 0 · t ${num(sp.drift.t_stat, 1)}`} tone={sp.drift.t_stat >= 2 ? 'up' : sp.drift.t_stat <= -2 ? 'down' : undefined} />
            <Stat label="runtime" value={`${(result.latency_ms / 1000).toFixed(0)}s`} />
          </div>
          <RunNote unavailable={result.unavailable} />
          {result.no_earnings.length > 0 && (
            <p className="muted small">
              no earnings history (funds, crypto, or yahoo has none): {result.no_earnings.slice(0, 12).join(', ')}
              {result.no_earnings.length > 12 && ` +${result.no_earnings.length - 12} more`}.
            </p>
          )}

          <div className="panel-title">Cumulative abnormal return around the report</div>
          <EventChart days={result.days} lines={lines} />
          <div className="legend">
            {result.groups.map((g) => (
              <span key={g.group}>
                <i className="dot" style={{ background: groupColor(g.group, n) }} /> {groupName(g.group, n)}
              </span>
            ))}
            <span className="muted">0 = the close before day 0</span>
          </div>

          <div className="panel-title">Groups</div>
          <table className="tok-table league research">
            <thead>
              <tr>
                <th>surprise group</th>
                <th>reports</th>
                <th>median surprise</th>
                {result.pre > 0 && <th>run-up ({result.pre}d before)</th>}
                <th>reaction (day 0)</th>
                <th>drift (+1…+{result.post})</th>
              </tr>
            </thead>
            <tbody>
              {result.groups.map((g) => (
                <tr key={g.group}>
                  <td>
                    <i className="dot" style={{ background: groupColor(g.group, n) }} /> {groupName(g.group, n)}
                  </td>
                  <td>{g.n.toLocaleString()}</td>
                  <td>{num(g.surprise_median, 1)}%</td>
                  {result.pre > 0 && <td>{pct(g.runup, 2)}</td>}
                  <td className={g.reaction > 0 ? 'up' : 'down'}>{pct(g.reaction, 2)}</td>
                  <td className={g.drift > 0 ? 'up' : 'down'}>{pct(g.drift, 2)}</td>
                </tr>
              ))}
              <tr className="me">
                <td>beats − misses</td>
                <td>{sp.drift.n} quarters</td>
                <td />
                {result.pre > 0 && sp.runup && (
                  <td className={tone(sp.runup)}>
                    {pct(sp.runup.mean, 2)} (t {num(sp.runup.t_stat, 1)})
                  </td>
                )}
                <td className={tone(sp.reaction)}>
                  {pct(sp.reaction.mean, 2)} (t {num(sp.reaction.t_stat, 1)})
                </td>
                <td className={tone(sp.drift)}>
                  {pct(sp.drift.mean, 2)} (t {num(sp.drift.t_stat, 1)})
                </td>
              </tr>
            </tbody>
          </table>

          <div className="panel-title">How fast the drift builds</div>
          <table className="tok-table league research">
            <thead>
              <tr>
                <th>held from day 0 close to</th>
                {sp.checkpoints.map((c) => (
                  <th key={c.day}>+{c.day}d</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>beats − misses</td>
                {sp.checkpoints.map((c) => (
                  <td key={c.day} className={tone(c)}>
                    {pct(c.mean, 2)} (t {num(c.t_stat, 1)})
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
          <p className="muted small">
            abnormal return = the stock's daily return minus the trimmed average of every stock in the run that day.
            the gap is averaged per quarter and the t-stat taken across quarters, since reports in the same season move
            together. green / red when |t| &gt; 2. returns are before trading costs.
          </p>
        </>
      )}
      {!result && !busy && <div className="empty">// no event study run yet. the S&amp;P 500 takes about 20s once its earnings are cached.</div>}
    </>
  )
}
