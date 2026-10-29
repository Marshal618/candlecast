"""Non-price signals, rebuilt point-in-time so they can be backtested without look-ahead.

Price-only models see one input. Most of what separates real quant signals from them is
information that isn't in the chart. These four come from data yfinance dates event by event,
so each rebalance only sees what was public by then:

- surprise:  the latest quarterly EPS surprise (reported vs consensus, %), if reported within the
             last 100 days. Post-earnings-announcement drift: prices keep moving in the direction
             of a surprise for weeks (Ball & Brown 1968, Bernard & Thomas 1989).
- revisions: net analyst actions over the last 90 days, each upgrade or price-target raise +1,
             each downgrade or cut -1, averaged. Revisions predict returns (Womack 1996).
- value:     earnings yield, the last four reported quarters of EPS over price (Basu 1977).
- size:      minus log market cap (split-adjusted shares outstanding times price), so smaller
             names score higher, as in the size premium (Banz 1981).

Timing: earnings carry a release time, so one counts at a rebalance if it came out before that
bar's close (before-open releases count the same day, after-close ones the next). Analyst actions
and share counts carry dates without reliable time zones, so they count from the next day.

Symbols with no earnings history (ETFs, crypto, indexes) get no fundamentals; at each rebalance
they and any other gaps take that period's median score, i.e. sit in the middle of the ranking.
Known limits: EPS is GAAP-or-adjusted as Yahoo reports it, prices are dividend-adjusted (earnings
yields read slightly high further back), and Yahoo's analyst history may be incomplete for small
firms. There is no point-in-time news archive in yfinance (only the last few days), so news isn't
offered: a sentiment backtest needs a paid historical feed.
"""

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FUNDAMENTALS = ("surprise", "revisions", "value", "size")
DAY = 86400
SURPRISE_DAYS = 100
REVISION_DAYS = 90
EPS_DAYS = 400  # the four quarters summed for earnings yield must all be this recent
SHARES_DAYS = 120
TTL = DAY


@dataclass
class Fundamentals:
    """One symbol's dated events, epoch seconds ascending."""

    earn_t: np.ndarray
    eps: np.ndarray
    surprise: np.ndarray  # NaN where Yahoo has no consensus
    rate_t: np.ndarray
    rate: np.ndarray  # +1 / 0 / -1 per analyst action
    share_t: np.ndarray
    shares: np.ndarray  # split-adjusted to today's share count basis

    @classmethod
    def from_json(cls, d: dict) -> "Fundamentals":
        def cols(rows, n):
            a = np.array(rows, dtype=float).reshape(-1, n)
            return [a[:, k] for k in range(n)]

        et, eps, sur = cols(d.get("earnings", []), 3)
        rt, r = cols(d.get("ratings", []), 2)
        st, sh = cols(d.get("shares", []), 2)
        return cls(et, eps, sur, rt, r, st, sh)

    @classmethod
    def empty(cls) -> "Fundamentals":
        return cls.from_json({})


def _cutoffs(at: int, interval: str) -> tuple[int, int]:
    """(earnings, dated-event) cutoffs for a bar whose shared-clock timestamp is `at` (see sim.align_keys).

    Daily keys are UTC midnight of the trading date, so +16h (noon New York) admits before-open
    reports and excludes after-close ones. Hourly keys are the bar's start.
    """
    if interval == "1d":
        return at + 16 * 3600, at
    return at + 3600, at - DAY


def symbol_scores(f: Fundamentals, at: int, interval: str, close: float) -> dict[str, float]:
    """Each fundamental for one symbol as known at bar `at` (NaN = no data)."""
    out = dict.fromkeys(FUNDAMENTALS, np.nan)
    e_cut, d_cut = _cutoffs(at, interval)
    n = int(np.searchsorted(f.earn_t, e_cut, side="left"))
    if n == 0:
        return out  # not an operating company, or no report yet
    if e_cut - f.earn_t[n - 1] <= SURPRISE_DAYS * DAY:
        out["surprise"] = float(f.surprise[n - 1])
    if n >= 4 and e_cut - f.earn_t[n - 4] <= EPS_DAYS * DAY and close > 0:
        out["value"] = float(f.eps[n - 4 : n].sum() / close)

    lo, hi = np.searchsorted(f.rate_t, [d_cut - REVISION_DAYS * DAY, d_cut], side="left")
    if hi > lo:
        out["revisions"] = float(f.rate[lo:hi].mean())
    lo, hi = np.searchsorted(f.share_t, [d_cut - SHARES_DAYS * DAY, d_cut], side="left")
    if hi > lo and close > 0:
        # Median of recent filings: Yahoo's share series has the odd one-day glitch
        out["size"] = float(-np.log(np.median(f.shares[lo:hi]) * close))
    return out


def fundamental_scores(
    funds: list[Fundamentals], at: int, interval: str, closes: np.ndarray
) -> dict[str, np.ndarray]:
    """Each fundamental across symbols at one rebalance, NaN where unknown."""
    per = [symbol_scores(f, at, interval, float(c)) for f, c in zip(funds, closes)]
    return {k: np.array([p[k] for p in per]) for k in FUNDAMENTALS}


def fill_missing(x: np.ndarray) -> np.ndarray:
    """Gaps take their row's median (the middle of the ranking); all-missing rows are all 0."""
    x = np.array(x, dtype=float)
    rows = x if x.ndim == 2 else x[None]
    for r in rows:
        bad = ~np.isfinite(r)
        if bad.any():
            r[bad] = np.median(r[~bad]) if (~bad).any() else 0.0
    return x


class FundamentalsProvider(ABC):
    @abstractmethod
    def fetch(self, symbol: str) -> dict:
        """{"earnings": [[t, eps, surprise%]], "ratings": [[t, +-1/0]], "shares": [[t, shares]]}, t ascending."""


def _rating(action: str, target: str) -> int:
    if action == "up":
        return 1
    if action == "down":
        return -1
    return {"Raises": 1, "Lowers": -1}.get(target, 0)


class YFinanceFundamentals(FundamentalsProvider):
    def fetch(self, symbol: str) -> dict:
        import pandas as pd
        import yfinance as yf

        t = yf.Ticker(symbol)
        out = {"earnings": [], "ratings": [], "shares": []}
        # Each source fails on its own for funds and crypto (404s); treat that as "no data"
        try:
            ed = t.get_earnings_dates(limit=100)
            if ed is not None and len(ed):
                ed = ed.dropna(subset=["Reported EPS"]).sort_index()
                out["earnings"] = [
                    [int(ts.timestamp()), float(r["Reported EPS"]), float(r["Surprise(%)"]) if pd.notna(r["Surprise(%)"]) else None]
                    for ts, r in ed.iterrows()
                ]
        except Exception:
            pass
        if not out["earnings"]:
            return out  # no earnings: not a company, skip the other calls
        try:
            ud = t.upgrades_downgrades
            if ud is not None and len(ud):
                ud = ud.sort_index()
                idx = ud.index.tz_localize("UTC") if ud.index.tz is None else ud.index
                out["ratings"] = [
                    [int(ts.timestamp()), _rating(a, p)]
                    for ts, a, p in zip(idx, ud["Action"], ud["priceTargetAction"].fillna(""))
                ]
        except Exception:
            pass
        try:
            sh = t.get_shares_full(start=pd.Timestamp.now() - pd.DateOffset(years=11))
            if sh is not None and len(sh):
                sh = sh[sh > 0].sort_index()
                splits = t.splits
                # Prices are split-adjusted; bring old share counts onto the same basis
                adj = np.ones(len(sh))
                for when, ratio in splits.items():
                    if ratio > 0:
                        adj[sh.index < when] *= ratio
                out["shares"] = [[int(ts.timestamp()), float(n * a)] for ts, n, a in zip(sh.index, sh.to_numpy(float), adj)]
        except Exception:
            pass
        return out


class CachedFundamentals:
    """Per-symbol event history, refreshed once a day."""

    def __init__(self, provider: FundamentalsProvider, cache_path: Path):
        self.provider = provider
        self.cache_path = cache_path
        self._lock = threading.Lock()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(cache_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS fundamentals (symbol TEXT PRIMARY KEY, fetched_at REAL, payload TEXT)"
            )

    def get(self, symbol: str) -> Fundamentals:
        symbol = symbol.strip().upper()
        with self._lock, sqlite3.connect(self.cache_path) as conn:
            row = conn.execute("SELECT fetched_at, payload FROM fundamentals WHERE symbol=?", (symbol,)).fetchone()
        if row and time.time() - row[0] < TTL:
            return Fundamentals.from_json(json.loads(row[1]))
        try:
            payload = self.provider.fetch(symbol)
        except Exception:
            # A failed refresh keeps the stale copy rather than wiping a symbol's fundamentals
            return Fundamentals.from_json(json.loads(row[1])) if row else Fundamentals.empty()
        with self._lock, sqlite3.connect(self.cache_path) as conn:
            conn.execute("INSERT OR REPLACE INTO fundamentals VALUES (?, ?, ?)", (symbol, time.time(), json.dumps(payload)))
        return Fundamentals.from_json(payload)
