import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  api,
  type KronosName,
  type OrderSide,
  type OrderType,
  type SimBot,
  type SimConfig,
  type SimHint,
  type SimPlayer,
  type SimState,
  type SimSummary,
  type Trial,
} from './api'
import { PriceChart, RaceChart, type TimeLabel } from './charts'
import { PLAYER_COLORS } from './theme'
import { money, pct, rate } from './format'
import { Stat } from './ui'

const ACTIVE_KEY = 'kd.sim.active'
const PLAYER_NAMES: Record<SimPlayer, string> = {
  you: 'you',
  buy_hold: 'buy & hold',
  sma_cross: 'sma 20/50',
  kronos: 'kronos bot',
  portfolio: 'portfolio bot',
}
const SPEEDS = [
  { label: 'slow', ms: 1200 },
  { label: 'normal', ms: 600 },
  { label: 'fast', ms: 200 },
]

function remember(id: string | null) {
  try {
    if (id) localStorage.setItem(ACTIVE_KEY, id)
    else localStorage.removeItem(ACTIVE_KEY)
  } catch {
    /* storage unavailable; the game is still on the leaderboard */
  }
}

function recall(): string | null {
  try {
    return localStorage.getItem(ACTIVE_KEY)
  } catch {
    return null
  }
}

const fmtDay = (t: number) => new Date(t * 1000).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })

export function SimView({ watchlist, onError }: { watchlist: string[]; onError: (msg: string | null) => void }) {
  const [sim, setSim] = useState<SimState | null>(null)
  const [board, setBoard] = useState<SimSummary[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [auto, setAuto] = useState(false)
  const [speed, setSpeed] = useState(600)
  const [selected, setSelected] = useState<string | null>(null)
  const [hint, setHint] = useState<(SimHint & { step: number }) | null>(null)

  const refreshBoard = useCallback(() => {
    api.sim.list().then(setBoard).catch(() => {})
  }, [])

  const run = useCallback(
    async <T,>(label: string, fn: () => Promise<T>): Promise<T | null> => {
      setBusy(label)
      onError(null)
      try {
        return await fn()
      } catch (e) {
        onError((e as Error).message)
        setAuto(false)
        return null
      } finally {
        setBusy(null)
      }
    },
    [onError],
  )

  const show = useCallback(
    (s: SimState | null) => {
      if (!s) return
      setSim(s)
      setSelected((cur) => (cur && s.config.symbols.includes(cur) ? cur : s.config.symbols[0]))
      if (s.finished) {
        setAuto(false)
        remember(null)
        refreshBoard()
      } else remember(s.id)
    },
    [refreshBoard],
  )

  useEffect(() => {
    refreshBoard()
    const id = recall()
    if (id) api.sim.get(id).then((s) => !s.finished && show(s)).catch(() => remember(null))
  }, [refreshBoard, show])

  const step = useCallback(
    async (bars: number) => {
      if (!sim) return
      show(await run(bars > 1 ? `Advancing ${bars} bars…` : 'Next bar…', () => api.sim.step(sim.id, bars)))
    },
    [sim, run, show],
  )

  // Auto-play: one bar at a time, waiting for each response before scheduling the next
  useEffect(() => {
    if (!auto || !sim || sim.finished || busy) return
    const t = setTimeout(() => step(1), speed)
    return () => clearTimeout(t)
  }, [auto, sim, busy, speed, step])

  const blindBase = sim?.blind?.base
  const blindBar = sim?.blind?.bar_seconds
  const timeLabel = useMemo<TimeLabel | undefined>(
    () => (blindBase != null && blindBar ? (t) => `#${Math.round((t - blindBase) / blindBar)}` : undefined),
    [blindBase, blindBar],
  )

  if (!sim) {
    return (
      <>
        <SimSetup
          watchlist={watchlist}
          busy={!!busy}
          onStart={async (cfg) => show(await run('Dealing a new market…', () => api.sim.create(cfg)))}
        />
        <Leaderboard
          board={board}
          onOpen={async (id) => show(await run('Loading game…', () => api.sim.get(id)))}
        />
      </>
    )
  }

  const symbol = selected ?? sim.config.symbols[0]
  const you = sim.players.you
  const bench = sim.players.buy_hold
  const showHint = hint && hint.symbol === symbol && hint.step === sim.step ? hint : null

  return (
    <>
      <div className="stats">
        <Stat
          label={sim.blind ? 'bar' : 'date'}
          value={sim.blind ? `${sim.step} / ${sim.config.length}` : fmtDay(sim.time)}
          hint={sim.blind ? 'blind mode: dates hidden' : `bar ${sim.step} / ${sim.config.length}`}
        />
        <Stat label="equity" value={money(sim.account.equity)} hint={`cash ${money(sim.account.cash)}`} />
        <Stat label="return" value={you ? pct(you.return) : '—'} tone={you && you.return >= 0 ? 'up' : 'down'} />
        <Stat
          label="vs buy & hold"
          value={you && bench ? pct(you.return - bench.return) : '—'}
          hint={bench ? `b&h ${pct(bench.return)}` : ''}
          tone={you && bench ? (you.return >= bench.return ? 'up' : 'down') : undefined}
        />
        <Stat label="max drawdown" value={you ? pct(you.max_drawdown) : '—'} hint={`${sim.account.trades} trades`} />
        <Stat label="hints" value={String(sim.hints)} hint={`fees ${money(sim.account.fees)}`} />
      </div>

      <div className="progress">
        <span style={{ width: `${(100 * sim.step) / sim.config.length}%` }} />
      </div>

      {sim.finished ? (
        <Results sim={sim} onNew={() => setSim(null)} />
      ) : (
        <div className="sim-controls">
          <button className="primary" disabled={!!busy || auto} onClick={() => step(1)}>
            +1 bar
          </button>
          <button disabled={!!busy || auto} onClick={() => step(5)}>
            +5
          </button>
          <button disabled={!!busy || auto} onClick={() => step(20)}>
            +20
          </button>
          <button className={auto ? 'active' : ''} onClick={() => setAuto((a) => !a)}>
            {auto ? '❚❚ pause' : '▶ auto'}
          </button>
          <div className="segmented speed">
            {SPEEDS.map((s) => (
              <button key={s.ms} className={speed === s.ms ? 'active' : ''} onClick={() => setSpeed(s.ms)}>
                {s.label}
              </button>
            ))}
          </div>
          <span className="status">{busy ? `> ${busy}` : ''}</span>
          <button className="right" onClick={() => setSim(null)} title="Leave this game; resume it from the leaderboard">
            leave
          </button>
          <button
            disabled={!!busy}
            onClick={async () => {
              setAuto(false)
              show(await run('Scoring game…', () => api.sim.finish(sim.id)))
            }}
          >
            finish
          </button>
        </div>
      )}

      <div className="quote-tabs">
        {sim.quotes.map((q, i) => (
          <button key={q.symbol} className={q.symbol === symbol ? 'active' : ''} onClick={() => setSelected(q.symbol)}>
            <b>{q.symbol}</b> {money(q.last)} <span className={q.change >= 0 ? 'up' : 'down'}>{pct(q.change, 2)}</span>
            {sim.kronos && sim.config.bots.includes('kronos') && (
              <span className="muted"> · bot saw {rate(sim.kronos.p_up[i])}</span>
            )}
            {!q.traded_this_bar && <span className="muted"> · closed</span>}
          </button>
        ))}
      </div>

      <div className="sim-grid">
        <div className="sim-chart">
          <PriceChart
            candles={sim.candles[symbol] ?? []}
            forecast={showHint}
            interval={sim.config.interval}
            timeLabel={timeLabel}
          />
          {!sim.finished && (
            <div className="hint-row">
              <button
                disabled={!!busy}
                onClick={async () => {
                  const h = await run(`Asking Kronos about ${symbol}…`, () => api.sim.hint(sim.id, symbol))
                  if (h) {
                    setHint({ ...h, step: sim.step })
                    setSim((s) => (s ? { ...s, hints: s.hints + 1 } : s))
                  }
                }}
              >
                ask kronos (hint)
              </button>
              {showHint ? (
                <span>
                  next 10 bars: p(up){' '}
                  <b className={showHint.stats.p_up >= 0.5 ? 'up' : 'down'}>{rate(showHint.stats.p_up)}</b> · median{' '}
                  <b className={showHint.stats.expected_return >= 0 ? 'up' : 'down'}>
                    {pct(showHint.stats.expected_return, 2)}
                  </b>{' '}
                  · band {pct(showHint.stats.band_width_pct).slice(1)}
                </span>
              ) : (
                <span className="muted">uses only bars visible now. hints are counted on the leaderboard.</span>
              )}
            </div>
          )}
        </div>
        {!sim.finished && (
          <OrderTicket
            sim={sim}
            symbol={symbol}
            busy={!!busy}
            onSubmit={async (o) => {
              show(await run('Placing order…', () => api.sim.order(sim.id, o)))
            }}
          />
        )}
      </div>

      <div className="token-grid">
        <div>
          <div className="panel-title">Positions</div>
          {sim.account.positions.length === 0 ? (
            <p className="muted small">all cash. orders fill from the next bar's open.</p>
          ) : (
            <table className="tok-table">
              <thead>
                <tr>
                  <th>symbol</th>
                  <th>qty</th>
                  <th>avg cost</th>
                  <th>last</th>
                  <th>value</th>
                  <th>p&amp;l</th>
                  <th>weight</th>
                </tr>
              </thead>
              <tbody>
                {sim.account.positions.map((p) => (
                  <tr key={p.symbol}>
                    <td>{p.symbol}</td>
                    <td>{+p.qty.toFixed(4)}</td>
                    <td>{money(p.avg_cost)}</td>
                    <td>{money(p.last)}</td>
                    <td>{money(p.value)}</td>
                    <td className={p.unrealized >= 0 ? 'up' : 'down'}>{money(p.unrealized)}</td>
                    <td>{rate(p.weight)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="muted small">realized p&amp;l {money(sim.account.realized)}</p>
        </div>
        <div>
          <div className="panel-title">Orders</div>
          {sim.orders.length === 0 ? (
            <p className="muted small">no orders yet.</p>
          ) : (
            <table className="tok-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>order</th>
                  <th>status</th>
                  <th>fill</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {sim.orders.slice(0, 12).map((o) => (
                  <tr key={o.id}>
                    <td>{o.id}</td>
                    <td>
                      <span className={o.side === 'buy' ? 'up' : 'down'}>{o.side}</span> {+o.qty.toFixed(4)} {o.symbol}{' '}
                      {o.type !== 'market' && `${o.type} ${money(o.price ?? 0)}`}
                    </td>
                    <td className={o.status === 'rejected' ? 'down' : o.status === 'open' ? 'warn' : ''}>
                      {o.status}
                      {o.reason ? ` (${o.reason})` : ''}
                    </td>
                    <td>{o.fill_price ? money(o.fill_price) : '—'}</td>
                    <td>
                      {o.status === 'open' && !sim.finished && (
                        <button
                          className="link"
                          disabled={!!busy}
                          onClick={async () => show(await run('Cancelling…', () => api.sim.cancel(sim.id, o.id)))}
                        >
                          cancel
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="panel-title">Race</div>
      <RaceChart sim={sim} timeLabel={timeLabel} />
      <Standings sim={sim} />
    </>
  )
}

function Standings({ sim }: { sim: SimState }) {
  const rows = (Object.entries(sim.players) as [SimPlayer, NonNullable<SimState['players'][SimPlayer]>][]).sort(
    (a, b) => b[1].return - a[1].return,
  )
  return (
    <table className="tok-table standings">
      <thead>
        <tr>
          <th>rank</th>
          <th>player</th>
          <th>return</th>
          <th>sharpe</th>
          <th>max dd</th>
          <th>trades</th>
          <th>fees</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([p, s], i) => (
          <tr key={p} className={p === 'you' ? 'me' : ''}>
            <td>{i + 1}</td>
            <td>
              <span className="sw" style={{ background: PLAYER_COLORS[p], marginLeft: 0, marginRight: 6 }} />
              {PLAYER_NAMES[p]}
            </td>
            <td className={s.return >= 0 ? 'up' : 'down'}>{pct(s.return)}</td>
            <td>{s.sharpe.toFixed(2)}</td>
            <td>{pct(s.max_drawdown)}</td>
            <td>{s.trades}</td>
            <td>{money(s.fees)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Results({ sim, onNew }: { sim: SimState; onNew: () => void }) {
  const r = sim.result ?? {}
  const ranked = (Object.entries(r) as [SimPlayer, { return: number }][]).sort((a, b) => b[1].return - a[1].return)
  const place = ranked.findIndex(([p]) => p === 'you') + 1
  return (
    <div className="results">
      <div>
        <div className="results-title">
          game over · you placed {place} of {ranked.length}
        </div>
        {sim.period && (
          <div className="muted">
            {sim.blind ? 'revealed: ' : ''}you traded {sim.config.symbols.join(', ')} from {fmtDay(sim.period[0])} to{' '}
            {fmtDay(sim.period[1])} ({sim.config.interval} bars)
          </div>
        )}
      </div>
      <button className="primary" onClick={onNew}>
        new game
      </button>
    </div>
  )
}

function OrderTicket({
  sim,
  symbol,
  busy,
  onSubmit,
}: {
  sim: SimState
  symbol: string
  busy: boolean
  onSubmit: (o: { symbol: string; side: OrderSide; qty: number; type: OrderType; price: number | null }) => void
}) {
  const [side, setSide] = useState<OrderSide>('buy')
  const [type, setType] = useState<OrderType>('market')
  const [qty, setQty] = useState('')
  const [price, setPrice] = useState('')
  const last = sim.quotes.find((q) => q.symbol === symbol)?.last ?? 0
  const held = sim.account.positions.find((p) => p.symbol === symbol)?.qty ?? 0
  const costRate = (sim.config.commission_bps + sim.config.slippage_bps) / 1e4
  const refPrice = type === 'market' ? last : Number(price) || last
  const maxQty = side === 'buy' ? sim.account.cash / (refPrice * (1 + costRate)) : held
  const n = Number(qty)
  const valid = n > 0 && (type === 'market' || Number(price) > 0)

  useEffect(() => {
    setPrice(last ? last.toFixed(2) : '')
  }, [symbol, type, last])

  return (
    <div className="ticket">
      <div className="panel-title">Order · {symbol}</div>
      <div className="segmented">
        <button className={side === 'buy' ? 'active buy' : ''} onClick={() => setSide('buy')}>
          buy
        </button>
        <button className={side === 'sell' ? 'active sell' : ''} onClick={() => setSide('sell')}>
          sell
        </button>
      </div>
      <div className="segmented">
        {(['market', 'limit', 'stop'] as OrderType[]).map((t) => (
          <button key={t} className={type === t ? 'active' : ''} onClick={() => setType(t)}>
            {t}
          </button>
        ))}
      </div>
      <label>
        quantity
        <input className="field" inputMode="decimal" value={qty} onChange={(e) => setQty(e.target.value)} placeholder="0" />
      </label>
      <div className="quick">
        {[0.25, 0.5, 1].map((f) => (
          <button key={f} onClick={() => setQty(String(+(maxQty * f * 0.999).toFixed(4)))} disabled={maxQty <= 0}>
            {f * 100}%
          </button>
        ))}
      </div>
      {type !== 'market' && (
        <label>
          {type} price
          <input className="field" inputMode="decimal" value={price} onChange={(e) => setPrice(e.target.value)} />
        </label>
      )}
      <p className="muted small">
        ≈ {money(n * refPrice || 0)} · {side === 'buy' ? `cash ${money(sim.account.cash)}` : `held ${+held.toFixed(4)}`}
        <br />
        {type === 'market'
          ? 'fills at the next bar open + slippage'
          : type === 'limit'
            ? `fills when price trades ${side === 'buy' ? 'down to' : 'up to'} the limit`
            : `triggers when price ${side === 'buy' ? 'rises to' : 'falls to'} the stop, then fills like market`}
      </p>
      <button
        className="primary"
        disabled={busy || !valid}
        onClick={() => {
          onSubmit({ symbol, side, qty: n, type, price: type === 'market' ? null : Number(price) })
          setQty('')
        }}
      >
        place {side} order
      </button>
    </div>
  )
}

function SimSetup({
  watchlist,
  busy,
  onStart,
}: {
  watchlist: string[]
  busy: boolean
  onStart: (cfg: SimConfig) => void
}) {
  const [symbols, setSymbols] = useState(watchlist.slice(0, 3).join(', '))
  const [interval, setBarInterval] = useState<'1d' | '1h'>('1d')
  const [cash, setCash] = useState('100000')
  const [length, setLength] = useState(120)
  const [random, setRandom] = useState(true)
  const [start, setStart] = useState('2022-01-03')
  const [blind, setBlind] = useState(true)
  const [commission, setCommission] = useState('2')
  const [slippage, setSlippage] = useState('5')
  const [bots, setBots] = useState<SimBot[]>(['sma_cross', 'kronos'])
  const [model, setModel] = useState<KronosName>('mini')
  const [recipes, setRecipes] = useState<Trial[]>([])
  const [recipeId, setRecipeId] = useState<number | null>(null)

  useEffect(() => {
    api.portfolio
      .recipes()
      .then((r) => {
        setRecipes(r)
        setRecipeId((cur) => cur ?? r[0]?.id ?? null)
      })
      .catch(() => {})
  }, [])
  const recipe = recipes.find((r) => r.id === recipeId)

  const toggleBot = (b: SimBot) => setBots((cur) => (cur.includes(b) ? cur.filter((x) => x !== b) : [...cur, b]))

  return (
    <div className="sim-setup">
      <div className="panel-title">New game</div>
      <p className="muted small">
        replay real market history bar by bar with a pretend account. you only ever see the past; orders fill on the
        next bar with fees and slippage. beat buy &amp; hold and the bots.
      </p>
      <div className="setup-grid">
        <label>
          basket (up to 8, comma-separated)
          <input className="field" value={symbols} onChange={(e) => setSymbols(e.target.value)} spellCheck={false} />
        </label>
        <label>
          bars
          <div className="segmented">
            {(['1d', '1h'] as const).map((i) => (
              <button key={i} className={interval === i ? 'active' : ''} onClick={() => setBarInterval(i)}>
                {i}
              </button>
            ))}
          </div>
        </label>
        <label>
          starting cash
          <input className="field" inputMode="decimal" value={cash} onChange={(e) => setCash(e.target.value)} />
        </label>
        <label className="slider">
          <span>
            game length (bars) <b>{length}</b>
          </span>
          <input type="range" min={20} max={500} step={10} value={length} onChange={(e) => setLength(Number(e.target.value))} />
        </label>
        <label>
          start
          <div className="segmented">
            <button className={random ? 'active' : ''} onClick={() => setRandom(true)}>
              random
            </button>
            <button className={!random ? 'active' : ''} onClick={() => setRandom(false)}>
              pick date
            </button>
          </div>
          {!random && <input className="field" type="date" value={start} onChange={(e) => setStart(e.target.value)} />}
        </label>
        <label>
          blind mode
          <div className="segmented">
            <button className={blind ? 'active' : ''} onClick={() => setBlind(true)}>
              on
            </button>
            <button className={!blind ? 'active' : ''} onClick={() => setBlind(false)}>
              off
            </button>
          </div>
          <span className="muted small">hides dates and rebases prices to 100, so you can't trade from memory</span>
        </label>
        <label>
          commission / slippage (bps)
          <div className="pair">
            <input className="field" inputMode="decimal" value={commission} onChange={(e) => setCommission(e.target.value)} />
            <input className="field" inputMode="decimal" value={slippage} onChange={(e) => setSlippage(e.target.value)} />
          </div>
        </label>
        <label>
          opponents (buy &amp; hold always plays)
          <div className="segmented">
            <button className={bots.includes('sma_cross') ? 'active' : ''} onClick={() => toggleBot('sma_cross')}>
              sma 20/50
            </button>
            <button className={bots.includes('kronos') ? 'active' : ''} onClick={() => toggleBot('kronos')}>
              kronos bot
            </button>
            <button
              className={bots.includes('portfolio') ? 'active' : ''}
              disabled={recipes.length === 0}
              title={recipes.length === 0 ? 'save a recipe in the Portfolio tab first' : undefined}
              onClick={() => toggleBot('portfolio')}
            >
              portfolio bot
            </button>
          </div>
          <span className="muted small">kronos bot re-forecasts every 5 bars and holds what it rates ≥ 60% up</span>
        </label>
        {bots.includes('portfolio') && (
          <label>
            portfolio recipe
            <select className="field" value={recipeId ?? ''} onChange={(e) => setRecipeId(Number(e.target.value))}>
              {recipes.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} ({r.config.interval}, every {r.config.horizon} bars)
                </option>
              ))}
            </select>
            <span className="muted small">
              runs the recipe's final blend on this basket, long-only (the sim is a cash account)
              {recipe && recipe.config.interval !== interval ? `. needs ${recipe.config.interval} bars` : ''}
            </span>
          </label>
        )}
        <label>
          kronos model (bot + hints)
          <div className="segmented">
            {(['mini', 'small', 'base'] as KronosName[]).map((m) => (
              <button key={m} className={model === m ? 'active' : ''} onClick={() => setModel(m)}>
                {m}
              </button>
            ))}
          </div>
        </label>
      </div>
      <button
        className="primary start"
        disabled={busy}
        onClick={() =>
          onStart({
            symbols: symbols
              .split(/[\s,]+/)
              .map((s) => s.trim().toUpperCase())
              .filter(Boolean),
            interval,
            cash: Number(cash) || 100000,
            length,
            start: random ? 'random' : start,
            blind,
            commission_bps: Number(commission) || 0,
            slippage_bps: Number(slippage) || 0,
            bots: bots.filter((b) => b !== 'portfolio' || recipeId != null),
            kronos_model: model,
            recipe_id: bots.includes('portfolio') ? recipeId : null,
          })
        }
      >
        ▶ start game
      </button>
    </div>
  )
}

function Leaderboard({ board, onOpen }: { board: SimSummary[]; onOpen: (id: string) => void }) {
  if (board.length === 0) return null
  return (
    <>
      <div className="panel-title">Leaderboard</div>
      <table className="tok-table">
        <thead>
          <tr>
            <th>played</th>
            <th>basket</th>
            <th>period</th>
            <th>you</th>
            <th>buy &amp; hold</th>
            <th>best bot</th>
            <th>hints</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {board.map((g) => {
            const r = g.result
            const bots = r
              ? (Object.entries(r) as [SimPlayer, { return: number }][]).filter(([p]) => p !== 'you' && p !== 'buy_hold')
              : []
            const best = bots.sort((a, b) => b[1].return - a[1].return)[0]
            return (
              <tr key={g.id}>
                <td>{new Date(g.created_at * 1000).toLocaleDateString()}</td>
                <td>
                  {g.symbols.join(', ')} · {g.interval}
                  {g.blind ? ' · blind' : ''}
                </td>
                <td>{g.period ? `${fmtDay(g.period[0])} → ${fmtDay(g.period[1])}` : 'hidden'}</td>
                <td className={r?.you ? (r.you.return >= (r.buy_hold?.return ?? 0) ? 'up' : 'down') : ''}>
                  {r?.you ? pct(r.you.return) : `${g.played}/${g.length}`}
                </td>
                <td>{r?.buy_hold ? pct(r.buy_hold.return) : '—'}</td>
                <td>{best ? `${PLAYER_NAMES[best[0]]} ${pct(best[1].return)}` : '—'}</td>
                <td>{g.hints}</td>
                <td>
                  <button className="link" onClick={() => onOpen(g.id)}>
                    {g.finished ? 'view' : 'resume'}
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </>
  )
}
