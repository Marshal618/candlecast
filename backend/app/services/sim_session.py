"""Sim sessions: create a game, step it, take orders, give Kronos hints, keep a leaderboard.

Each session stores its config, a frozen market snapshot, the action log and the cursor; state
is rebuilt with `sim.simulate` on every request. Kronos bot decisions are computed once and
memoized, so replays never call the model again.
"""

import json
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from app.services import sim
from app.services.forecaster import FEATURES, summarize
from app.services.market_data import CachedMarketData, closed_bars, future_timestamps, interval_delta

WARMUP = 300  # history visible before the first playable bar (SMA 50, Kronos lookback)
KRONOS_EVERY = 5  # Kronos bot re-forecasts every N bars, horizon N
KRONOS_ENTER = 0.6  # ...and holds a symbol while P(up) >= this
KRONOS_LOOKBACK = 256
KRONOS_SAMPLES = 10
VIEW_BARS = 300
BLIND_BASE = 1_000_000_000  # synthetic epoch for blind sessions; the UI shows bar numbers, not dates

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    config TEXT NOT NULL,
    market TEXT NOT NULL,
    actions TEXT NOT NULL,
    cursor INTEGER NOT NULL,
    memo TEXT NOT NULL,
    hints INTEGER NOT NULL DEFAULT 0,
    finished_at REAL,
    result TEXT
);
"""


class SimError(Exception):
    pass


class SimStore:
    def __init__(self, path: Path, market: CachedMarketData, forecaster, now: Callable[[], float] = time.time):
        self.path, self.market, self.forecaster, self.now = path, market, forecaster, now
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- persistence -------------------------------------------------------------------------

    def _load(self, sid: str) -> dict:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise SimError(f"No sim session '{sid}'.")
        s = dict(row)
        for k in ("config", "actions", "memo"):
            s[k] = json.loads(s[k])
        s["result"] = s["result"] and json.loads(s["result"])
        s["market"] = sim.Market.from_json(json.loads(s["market"]))
        return s

    def _save(self, s: dict, **fields) -> None:
        enc = {k: json.dumps(v) if k in ("actions", "memo", "result") else v for k, v in fields.items()}
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE sessions SET {', '.join(f'{k}=?' for k in enc)} WHERE id=?", (*enc.values(), s["id"]))
        s.update(fields)

    # --- game lifecycle ----------------------------------------------------------------------

    def create(self, cfg: dict, seed: int | None = None) -> dict:
        frames, tzs = {}, {}
        for sym in dict.fromkeys(s.strip().upper() for s in cfg["symbols"]):
            df = closed_bars(self.market.get(sym, cfg["interval"]), cfg["interval"], self.now())
            frames[sym], tzs[sym] = df, str(df.index.tz or "UTC")
        full = sim.build_market(frames, cfg["interval"])
        length = cfg["length"]
        last_start = len(full.times) - length - 1
        if last_start < WARMUP:
            raise SimError(
                f"Not enough shared history: need {WARMUP + length + 1} bars, have {len(full.times)}. "
                "Shorten the game or use the daily interval."
            )
        if cfg["start"] == "random":
            s = int(np.random.default_rng(seed).integers(WARMUP, last_start + 1))
        else:
            ts = pd.Timestamp(cfg["start"], tz="UTC").timestamp()
            s = int(np.searchsorted(full.times, ts))
            if not WARMUP <= s <= last_start:
                lo, hi = (pd.Timestamp(full.times[i], unit="s").date() for i in (WARMUP, last_start))
                raise SimError(f"Start must be between {lo} and {hi} for this basket and length.")
        m = sim.slice_market(full, s - WARMUP, s + length + 1)
        if cfg["blind"]:
            m = sim.rescale_market(m, WARMUP)
        bots = ["buy_hold"] + [b for b in cfg["bots"] if b != "buy_hold"]
        config = {
            **cfg,
            "symbols": list(frames),
            "bots": bots,
            "tz": tzs,
            "start_index": WARMUP,
            "end_index": WARMUP + length,
            "real_start": int(m.times[WARMUP]),
            "real_end": int(m.times[WARMUP + length]),
        }
        sid = secrets.token_hex(4)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, created_at, config, market, actions, cursor, memo) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sid, self.now(), json.dumps(config), json.dumps(m.to_json()), "[]", WARMUP, "{}"),
            )
        s = self._load(sid)
        self._decide_kronos(s, [WARMUP])
        return self.state(sid)

    def step(self, sid: str, bars: int) -> dict:
        s = self._load(sid)
        self._require_open(s)
        new = min(s["cursor"] + bars, s["config"]["end_index"])
        self._decide_kronos(s, range(s["cursor"] + 1, new + 1))
        self._save(s, cursor=new)
        if new == s["config"]["end_index"]:
            return self.finish(sid)
        return self.state(sid)

    def order(self, sid: str, symbol: str, side: str, qty: float, kind: str, price: float | None) -> dict:
        s = self._load(sid)
        self._require_open(s)
        syms = s["config"]["symbols"]
        symbol = symbol.upper()
        if symbol not in syms:
            raise SimError(f"{symbol} isn't in this game's basket ({', '.join(syms)}).")
        if kind in ("limit", "stop") and not price:
            raise SimError(f"A {kind} order needs a price.")
        if side == "sell":
            state = self._replay(s)
            held = next((p["qty"] for p in state["account"]["positions"] if p["symbol"] == symbol), 0.0)
            pending = sum(o["qty"] for o in state["orders"] if o["status"] == "open" and o["side"] == "sell" and o["symbol"] == symbol)
            if qty > held - pending + 1e-9:
                raise SimError(f"Can only sell {max(held - pending, 0):g} {symbol} (long only, including open sell orders).")
        oid = 1 + sum(a["kind"] == "order" for a in s["actions"])
        action = {"at": s["cursor"], "kind": "order", "id": oid, "symbol_index": syms.index(symbol),
                  "side": side, "type": kind, "qty": qty, "price": price if kind != "market" else None}
        self._save(s, actions=s["actions"] + [action])
        return self.state(sid)

    def cancel(self, sid: str, order_id: int) -> dict:
        s = self._load(sid)
        self._require_open(s)
        state = self._replay(s)
        if not any(o["id"] == order_id and o["status"] == "open" for o in state["orders"]):
            raise SimError(f"Order #{order_id} isn't open.")
        self._save(s, actions=s["actions"] + [{"at": s["cursor"], "kind": "cancel", "order_id": order_id}])
        return self.state(sid)

    def finish(self, sid: str) -> dict:
        s = self._load(sid)
        if s["finished_at"] is None:
            state = self._replay(s)
            bpy = _bars_per_year(s["market"].times)
            result = {p: sim.performance(c, bpy) | {"trades": state["players"][p]["trades"]} for p, c in state["curves"].items()}
            self._save(s, finished_at=self.now(), result=result)
        return self.state(sid)

    def hint(self, sid: str, symbol: str, horizon: int) -> dict:
        """Kronos forecast for one symbol from the bars visible at the cursor."""
        s = self._load(sid)
        self._require_open(s)
        cfg, m, t = s["config"], s["market"], s["cursor"]
        if symbol.upper() not in cfg["symbols"]:
            raise SimError(f"{symbol} isn't in this game's basket.")
        i = cfg["symbols"].index(symbol.upper())
        ctx = self._context(s, i, t, KRONOS_LOOKBACK)
        if ctx is None:
            raise SimError(f"Not enough visible history for {symbol}.")
        future = future_timestamps(ctx.index, cfg["interval"], horizon)
        paths = self.forecaster.sample_paths([ctx], [future], cfg["kronos_model"], KRONOS_SAMPLES, 1.0, 0.9)
        last = float(ctx["close"].iloc[-1])
        st = summarize(paths[0], last)
        self._save(s, hints=s["hints"] + 1)
        times = self._future_times(s, t, future)
        return {
            "symbol": symbol.upper(),
            "interval": cfg["interval"],
            "model": cfg["kronos_model"],
            "last_close": last,
            "last_time": self._time(s, t),
            "stats": {"p_up": st["p_up"], "expected_return": st["expected_return"], "band_width_pct": st["band_width_pct"]},
            "forecast": [
                {"time": tm, "median": float(st["median"][k]), "p10": float(st["p10"][k]), "p90": float(st["p90"][k]),
                 "open": float(st["ohlc"][k, 0]), "high": float(st["ohlc"][k, 1]),
                 "low": float(st["ohlc"][k, 2]), "close": float(st["ohlc"][k, 3])}
                for k, tm in enumerate(times)
            ],
        }

    def leaderboard(self, limit: int = 50) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, config, cursor, hints, finished_at, result FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            cfg = json.loads(r["config"])
            revealed = r["finished_at"] is not None or not cfg["blind"]
            out.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "symbols": cfg["symbols"],
                    "interval": cfg["interval"],
                    "blind": cfg["blind"],
                    "length": cfg["length"],
                    "played": r["cursor"] - cfg["start_index"],
                    "hints": r["hints"],
                    "finished": r["finished_at"] is not None,
                    "result": r["result"] and json.loads(r["result"]),
                    "period": [cfg["real_start"], cfg["real_end"]] if revealed else None,
                }
            )
        return out

    # --- state -------------------------------------------------------------------------------

    def state(self, sid: str) -> dict:
        s = self._load(sid)
        cfg, m, t = s["config"], s["market"], s["cursor"]
        st = self._replay(s)
        start = cfg["start_index"]
        bpy = _bars_per_year(m.times)
        lo = max(0, t - VIEW_BARS + 1)
        closes = m.closes_ffill

        candles = {}
        for i, sym in enumerate(cfg["symbols"]):
            rows = [k for k in range(lo, t + 1) if not np.isnan(m.c[k, i])]
            candles[sym] = [
                {"time": self._time(s, k), "open": m.o[k, i], "high": m.h[k, i], "low": m.l[k, i], "close": m.c[k, i], "volume": m.v[k, i]}
                for k in rows
            ]
        quotes = []
        for i, sym in enumerate(cfg["symbols"]):
            valid = np.nonzero(~np.isnan(m.c[: t + 1, i]))[0]
            last = float(closes[t, i])
            prev = float(m.c[valid[-2], i]) if len(valid) > 1 else last
            quotes.append({"symbol": sym, "last": last, "change": last / prev - 1, "traded_this_bar": bool(not np.isnan(m.c[t, i]))})

        def bar_time(k):
            return None if k is None else self._time(s, k)

        orders = [
            {**{k: v for k, v in o.items() if k not in ("placed_at", "filled_at", "symbol_index")},
             "placed_time": bar_time(o["placed_at"]), "filled_time": bar_time(o.get("filled_at"))}
            for o in reversed(st["orders"])
        ]
        finished = s["finished_at"] is not None
        revealed = finished or not cfg["blind"]
        return {
            "id": s["id"],
            "config": {k: cfg[k] for k in ("symbols", "interval", "cash", "length", "blind", "commission_bps",
                                            "slippage_bps", "bots", "kronos_model")},
            "step": t - start,
            "time": self._time(s, t),
            "finished": finished,
            "hints": s["hints"],
            "blind": {"base": self._time(s, start), "bar_seconds": int(interval_delta(cfg["interval"]).total_seconds())}
            if cfg["blind"] else None,
            "period": [cfg["real_start"], cfg["real_end"]] if revealed else None,
            "quotes": quotes,
            "candles": candles,
            "account": st["account"],
            "orders": orders,
            "curve": [
                {"time": self._time(s, start + k), **{p: c[k] for p, c in st["curves"].items()}}
                for k in range(t - start + 1)
            ],
            "players": {p: {**v, **sim.performance(st["curves"][p], bpy)} for p, v in st["players"].items()},
            # The bot's readings would be free hints, so they're only shown once the game is over
            "kronos": s["memo"].get(str(max((int(k) for k in s["memo"] if int(k) <= t), default=-1))) if finished else None,
            "result": s["result"],
        }

    # --- internals ---------------------------------------------------------------------------

    def _require_open(self, s: dict) -> None:
        if s["finished_at"] is not None:
            raise SimError("This game is finished. Start a new one.")

    def _replay(self, s: dict) -> dict:
        cfg, m = s["config"], s["market"]
        start = cfg["start_index"]

        def bot_weights(bot: str, t: int):
            if bot == "buy_hold":
                return sim.buy_hold_weights(m, t, start)
            if bot == "sma_cross":
                return sim.sma_cross_weights(m, t, start)
            if bot == "kronos":
                d = s["memo"].get(str(t))
                return None if d is None else np.array(d["weights"])
            return None

        return sim.simulate(
            m, start=start, cursor=s["cursor"], cash=cfg["cash"], commission_bps=cfg["commission_bps"],
            slippage_bps=cfg["slippage_bps"], actions=s["actions"], bots=cfg["bots"], bot_weights=bot_weights,
        )

    def _time(self, s: dict, k: int) -> int:
        cfg = s["config"]
        if cfg["blind"]:
            return BLIND_BASE + (k - cfg["start_index"]) * int(interval_delta(cfg["interval"]).total_seconds())
        return int(s["market"].times[k])

    def _future_times(self, s: dict, t: int, future: pd.DatetimeIndex) -> list[int]:
        if s["config"]["blind"]:
            return [self._time(s, t + k + 1) for k in range(len(future))]
        return [int(x.timestamp()) for x in future]

    def _context(self, s: dict, i: int, t: int, lookback: int) -> pd.DataFrame | None:
        """Bars of symbol i up to and including t (only bars it actually traded), as Kronos input."""
        m = s["market"]
        rows = np.nonzero(~np.isnan(m.c[: t + 1, i]))[0][-lookback:]
        if len(rows) < 32:
            return None
        idx = pd.DatetimeIndex(pd.to_datetime(m.times[rows], unit="s"))
        if s["config"]["interval"] != "1d":
            # Intraday: restore exchange-local time so Kronos's hour/weekday features are right.
            # Daily bars stay as naive dates (the market keys them on the calendar date).
            idx = idx.tz_localize("UTC").tz_convert(s["config"]["tz"][m.symbols[i]])
        return pd.DataFrame(
            {"open": m.o[rows, i], "high": m.h[rows, i], "low": m.l[rows, i], "close": m.c[rows, i], "volume": m.v[rows, i]},
            index=idx,
        )

    def _decide_kronos(self, s: dict, bars) -> None:
        """Compute and memoize the Kronos bot's decisions for any decision bars in `bars`."""
        cfg = s["config"]
        if "kronos" not in cfg["bots"]:
            return
        todo = [t for t in bars if (t - cfg["start_index"]) % KRONOS_EVERY == 0 and str(t) not in s["memo"]]
        if not todo:
            return
        n = len(cfg["symbols"])
        jobs = [(t, i, self._context(s, i, t, KRONOS_LOOKBACK)) for t in todo for i in range(n)]
        jobs = [j for j in jobs if j[2] is not None]
        p_up = {t: [0.0] * n for t in todo}
        if jobs:
            # One batched model call; contexts must share a length, so trim to the shortest
            L = min(len(j[2]) for j in jobs)
            ctxs = [j[2].tail(L) for j in jobs]
            futures = [future_timestamps(c.index, cfg["interval"], KRONOS_EVERY) for c in ctxs]
            paths = self.forecaster.sample_paths(ctxs, futures, cfg["kronos_model"], KRONOS_SAMPLES, 1.0, 0.9)
            close = FEATURES.index("close")
            for (t, i, _), ctx, p in zip(jobs, ctxs, paths):
                p_up[t][i] = float(np.mean(p[:, -1, close] > ctx["close"].iloc[-1]))
        memo = dict(s["memo"])
        for t in todo:
            memo[str(t)] = {"p_up": p_up[t], "weights": [(1.0 / n) if p >= KRONOS_ENTER else 0.0 for p in p_up[t]]}
        self._save(s, memo=memo)


def _bars_per_year(times: np.ndarray) -> float:
    years = (times[-1] - times[0]) / (365.25 * 86400)
    return len(times) / years if years > 0 else 252.0
