"""Market replay simulator: trade a basket of real historical series bar by bar, against bots.

Game rules, chosen to behave like a plain cash brokerage account:
- You only ever see bars up to the cursor. Orders placed at the cursor are worked from the
  *next* bar onward, so nothing trades on information it couldn't have had.
- Market orders fill at the next open. Limit orders fill when the bar trades through the
  limit, at the limit or a better gap open. Stops trigger the same way and fill like market
  orders. Market and stop fills pay slippage; every fill pays commission.
- Long only. Buys need the cash at fill time (rejected otherwise); sells are capped at the
  position. Orders stay open until filled or cancelled.
- Bots follow the same rules: they decide at a bar's close and trade at the next open.

A session is stored as config + market snapshot + a log of actions, and the whole state is
rebuilt by replaying that log. Replays are deterministic, so the engine is easy to test and a
restart loses nothing.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

BOTS = ("buy_hold", "sma_cross", "kronos")
SMA_FAST, SMA_SLOW = 20, 50


@dataclass
class Market:
    symbols: list[str]
    times: np.ndarray  # (bars,) epoch seconds
    o: np.ndarray  # (bars, symbols); NaN where that symbol had no bar
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray

    @property
    def closes_ffill(self) -> np.ndarray:
        return pd.DataFrame(self.c).ffill().to_numpy()

    def to_json(self) -> dict:
        def arr(a):
            return [[None if np.isnan(x) else float(x) for x in row] for row in a]

        return {"symbols": self.symbols, "times": self.times.tolist(), **{k: arr(getattr(self, k)) for k in "ohlcv"}}

    @classmethod
    def from_json(cls, d: dict) -> "Market":
        def arr(a):
            return np.array([[np.nan if x is None else x for x in row] for row in a], dtype=float)

        return cls(d["symbols"], np.array(d["times"], dtype=np.int64), *(arr(d[k]) for k in "ohlcv"))


def _align(f: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Put one symbol's bars on a shared UTC clock.

    Daily bars key on the calendar date: stocks stamp days at local midnight (05:00 UTC for New
    York) while crypto uses UTC midnight, so raw timestamps for the same day never match.
    Intraday bars key on the UTC hour (a stock's 9:30 bar lines up with crypto's 9:00 one).
    """
    f = f.copy()
    f.index = align_keys(f.index, interval)
    return f[~f.index.duplicated(keep="last")]


def align_keys(index: pd.DatetimeIndex, interval: str) -> pd.DatetimeIndex:
    """The shared-clock timestamp of each bar (see _align)."""
    if interval == "1d":
        return pd.DatetimeIndex(index.date).tz_localize("UTC")
    return index.tz_convert("UTC").floor("h")


def build_market(frames: dict[str, pd.DataFrame], interval: str = "1d") -> Market:
    """Align several OHLCV frames on the union of their (aligned) timestamps."""
    frames = {k: _align(f, interval) for k, f in frames.items()}
    idx = pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))))
    cols = {k: [] for k in "ohlcv"}
    for f in frames.values():
        f = f.reindex(idx)
        for k, col in zip("ohlcv", ["open", "high", "low", "close", "volume"]):
            cols[k].append(f[col].to_numpy(dtype=float))
    return Market(
        list(frames),
        np.array([int(t.timestamp()) for t in idx], dtype=np.int64),
        *(np.column_stack(cols[k]) for k in "ohlcv"),
    )


def slice_market(m: Market, a: int, b: int) -> Market:
    return Market(m.symbols, m.times[a:b], m.o[a:b], m.h[a:b], m.l[a:b], m.c[a:b], m.v[a:b])


def rescale_market(m: Market, at: int) -> Market:
    """Blind mode: rebase each symbol so its close at bar `at` is 100 (hides real price levels)."""
    ref = m.closes_ffill[at]
    k = 100.0 / ref
    return Market(m.symbols, m.times, m.o * k, m.h * k, m.l * k, m.c * k, m.v)


class Account:
    def __init__(self, n: int, cash: float):
        self.cash = cash
        self.qty = np.zeros(n)
        self.cost = np.zeros(n)  # total cost basis incl. fees
        self.realized = 0.0
        self.fees = 0.0
        self.trades = 0

    def buy(self, i: int, qty: float, price: float, fee_rate: float, partial: bool) -> float:
        """Returns the quantity bought (0 if it couldn't be afforded and partial fills are off)."""
        affordable = self.cash / (price * (1 + fee_rate))
        if qty > affordable + 1e-9:
            if not partial:
                return 0.0
            qty = affordable
        if qty <= 1e-12:
            return 0.0
        notional = qty * price
        fee = notional * fee_rate
        self.cash -= notional + fee
        self.qty[i] += qty
        self.cost[i] += notional + fee
        self.fees += fee
        self.trades += 1
        return qty

    def sell(self, i: int, qty: float, price: float, fee_rate: float) -> float:
        qty = min(qty, self.qty[i])
        if qty <= 1e-12:
            return 0.0
        proceeds = qty * price
        fee = proceeds * fee_rate
        basis = self.cost[i] * qty / self.qty[i]
        self.cash += proceeds - fee
        self.realized += proceeds - fee - basis
        self.cost[i] -= basis
        self.qty[i] -= qty
        if self.qty[i] < 1e-12:
            self.qty[i], self.cost[i] = 0.0, 0.0
        self.fees += fee
        self.trades += 1
        return qty

    def equity(self, prices: np.ndarray) -> float:
        held = self.qty > 0
        return float(self.cash + np.sum(self.qty[held] * prices[held]))


def fill_price(order: dict, o: float, h: float, lo: float) -> float | None:
    """Price an open order fills at on a bar, before slippage; None if it doesn't fill."""
    if np.isnan(o):
        return None  # no trading in this symbol on this bar
    side, kind, px = order["side"], order["type"], order.get("price")
    if kind == "market":
        return o
    if kind == "limit":
        if side == "buy" and lo <= px:
            return min(o, px)
        if side == "sell" and h >= px:
            return max(o, px)
    if kind == "stop":
        if side == "buy" and h >= px:
            return max(o, px)
        if side == "sell" and lo <= px:
            return min(o, px)
    return None


# --- bots: each maps (market, bar) -> target weights, or None to keep current holdings ---------


def buy_hold_weights(m: Market, t: int, start: int) -> np.ndarray | None:
    if t != start:
        return None
    live = ~np.isnan(m.closes_ffill[t])
    return live / max(live.sum(), 1)


def sma_cross_weights(m: Market, t: int, start: int) -> np.ndarray | None:
    c = m.closes_ffill[: t + 1]
    if len(c) < SMA_SLOW:
        return np.zeros(len(m.symbols))
    fast, slow = c[-SMA_FAST:].mean(axis=0), c[-SMA_SLOW:].mean(axis=0)
    return (fast > slow) / len(m.symbols)


def simulate(
    m: Market,
    *,
    start: int,
    cursor: int,
    cash: float,
    commission_bps: float,
    slippage_bps: float,
    actions: list[dict],
    bots: list[str],
    bot_weights: Callable[[str, int], np.ndarray | None],
) -> dict:
    """Replay the session from `start` to `cursor`. `actions` are {"at", "kind", ...} entries."""
    n = len(m.symbols)
    fee, slip = commission_bps / 1e4, slippage_bps / 1e4
    closes = m.closes_ffill
    you = Account(n, cash)
    players = {"you": you, **{b: Account(n, cash) for b in bots}}
    orders: list[dict] = []
    pending_targets: dict[str, np.ndarray] = {}
    last_targets: dict[str, np.ndarray] = {}
    curves: dict[str, list[float]] = {p: [] for p in players}
    by_bar: dict[int, list[dict]] = {}
    for a in actions:
        by_bar.setdefault(a["at"], []).append(a)

    for t in range(start, cursor + 1):
        if t > start:
            # Bots rebalance at the open toward the weights they chose at the last close
            opens = np.where(np.isnan(m.o[t]), closes[t - 1], m.o[t])
            carry = {}
            for b, w in pending_targets.items():
                acct = players[b]
                equity = acct.equity(opens)
                delta = w * equity / opens - acct.qty
                for i in np.argsort(delta):  # sells first, freeing cash for buys
                    if not np.isfinite(delta[i]) or abs(delta[i]) * opens[i] < 1e-6 * equity:
                        continue
                    if np.isnan(m.o[t, i]):
                        carry[b] = w  # symbol closed this bar (e.g. stock on a weekend); retry next bar
                        continue
                    if delta[i] < 0:
                        acct.sell(i, -delta[i], opens[i] * (1 - slip), fee)
                    else:
                        acct.buy(i, delta[i], opens[i] * (1 + slip), fee, partial=True)
            pending_targets = carry

            for order in orders:
                if order["status"] != "open":
                    continue
                i = order["symbol_index"]
                px = fill_price(order, m.o[t, i], m.h[t, i], m.l[t, i])
                if px is None:
                    continue
                if order["type"] != "limit":
                    px *= (1 + slip) if order["side"] == "buy" else (1 - slip)
                if order["side"] == "buy":
                    done = you.buy(i, order["qty"], px, fee, partial=False)
                    reason = "insufficient cash"
                else:
                    done = you.sell(i, order["qty"], px, fee)
                    reason = "no position to sell"
                if done > 0:
                    order.update(status="filled", filled_qty=done, fill_price=px, filled_at=t)
                else:
                    order.update(status="rejected", reason=reason, filled_at=t)

        for p, acct in players.items():
            curves[p].append(acct.equity(closes[t]))

        for b in bots:
            w = bot_weights(b, t)
            if w is not None and (b not in last_targets or not np.allclose(w, last_targets[b])):
                pending_targets[b] = np.nan_to_num(np.asarray(w, dtype=float))
                last_targets[b] = pending_targets[b]

        for a in by_bar.get(t, []):
            if a["kind"] == "order":
                orders.append(
                    {
                        "id": a["id"],
                        "symbol_index": a["symbol_index"],
                        "symbol": m.symbols[a["symbol_index"]],
                        "side": a["side"],
                        "type": a["type"],
                        "qty": a["qty"],
                        "price": a.get("price"),
                        "placed_at": t,
                        "status": "open",
                    }
                )
            elif a["kind"] == "cancel":
                for order in orders:
                    if order["id"] == a["order_id"] and order["status"] == "open":
                        order.update(status="cancelled", filled_at=t)

    last = closes[cursor]
    positions = [
        {
            "symbol": m.symbols[i],
            "qty": float(you.qty[i]),
            "avg_cost": float(you.cost[i] / you.qty[i]),
            "last": float(last[i]),
            "value": float(you.qty[i] * last[i]),
            "unrealized": float(you.qty[i] * last[i] - you.cost[i]),
        }
        for i in range(n)
        if you.qty[i] > 0
    ]
    equity = curves["you"][-1]
    for p in positions:
        p["weight"] = p["value"] / equity if equity else 0.0
    return {
        "account": {
            "cash": you.cash,
            "equity": equity,
            "realized": you.realized,
            "fees": you.fees,
            "trades": you.trades,
            "positions": positions,
        },
        "orders": orders,
        "curves": curves,
        "players": {
            p: {"equity": curves[p][-1], "trades": a.trades, "fees": a.fees, "cash": a.cash} for p, a in players.items()
        },
    }


def performance(curve: list[float], bars_per_year: float) -> dict:
    eq = np.asarray(curve, dtype=float)
    r = eq[1:] / eq[:-1] - 1 if len(eq) > 1 else np.zeros(1)
    sd = r.std()
    peak = np.maximum.accumulate(eq)
    return {
        "return": float(eq[-1] / eq[0] - 1),
        "sharpe": float(r.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0,
        "max_drawdown": float(np.min(eq / peak - 1)),
    }
