"""Forecast journal: record every live forecast, score it once its bars have happened.

Backtests replay history that Kronos may have seen during pre-training, so only forecasts
made in real time about bars that did not exist yet are a clean test. Rules that keep the
record honest:

- One entry per (symbol, interval, model, horizon, origin bar). Re-running until a forecast
  looks good does not add entries; the first one counts.
- An entry is scored only after its final bar has *closed*, so in-progress candles never
  count. Returns are measured from the close the model saw when it forecast.
- Entries cannot be edited or deleted through the API.
"""

import json
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.services.backtest import r2_oos
from app.services.market_data import CachedMarketData, MarketDataError, interval_delta

PERMUTATIONS = 1000
CALIBRATION_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    model TEXT NOT NULL,
    lookback INTEGER NOT NULL,
    horizon INTEGER NOT NULL,
    samples INTEGER NOT NULL,
    temperature REAL NOT NULL,
    top_p REAL NOT NULL,
    origin_time INTEGER NOT NULL,
    last_close REAL NOT NULL,
    p_up REAL NOT NULL,
    expected_return REAL NOT NULL,
    pred_final REAL NOT NULL,
    p10_final REAL NOT NULL,
    p90_final REAL NOT NULL,
    path TEXT NOT NULL,
    actual_final REAL,
    actual_time INTEGER,
    scored_at REAL,
    UNIQUE (symbol, interval, model, horizon, origin_time)
);
CREATE TABLE IF NOT EXISTS backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    model TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    windows INTEGER NOT NULL,
    hit_rate REAL NOT NULL,
    baseline_hit_rate REAL NOT NULL,
    r2_oos REAL NOT NULL,
    coverage REAL NOT NULL
);
"""


class Journal:
    def __init__(self, path: Path, market: CachedMarketData, now: Callable[[], float] = time.time):
        self.path = path
        self.market = market
        self.now = now
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def record_forecast(self, entry: dict) -> tuple[int, bool]:
        """Insert a forecast; returns (id, created). An existing entry for the same origin wins."""
        cols = [
            "symbol", "interval", "model", "lookback", "horizon", "samples", "temperature", "top_p",
            "origin_time", "last_close", "p_up", "expected_return", "pred_final", "p10_final", "p90_final",
        ]
        row = {k: entry[k] for k in cols}
        row["symbol"] = row["symbol"].upper()
        row["path"] = json.dumps(entry["path"])
        row["created_at"] = self.now()
        key = (row["symbol"], row["interval"], row["model"], row["horizon"], row["origin_time"])
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"INSERT OR IGNORE INTO forecasts ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                tuple(row.values()),
            )
            if cur.rowcount:
                return int(cur.lastrowid), True
            existing = conn.execute(
                "SELECT id FROM forecasts WHERE symbol=? AND interval=? AND model=? AND horizon=? AND origin_time=?",
                key,
            ).fetchone()
            return int(existing["id"]), False

    def record_backtest(self, symbol: str, interval: str, model: str, horizon: int, metrics: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO backtests (created_at, symbol, interval, model, horizon, windows, hit_rate,"
                " baseline_hit_rate, r2_oos, coverage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.now(), symbol.upper(), interval, model, horizon, metrics["windows"], metrics["hit_rate"],
                    metrics["baseline_hit_rate"], metrics["r2_oos"], metrics["coverage_p10_p90"],
                ),
            )

    def score_pending(self) -> int:
        """Score every pending entry whose horizon has fully closed. Returns how many were scored."""
        with self._lock, self._connect() as conn:
            pending = conn.execute("SELECT * FROM forecasts WHERE scored_at IS NULL").fetchall()
        scored = 0
        by_series: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for r in pending:
            by_series.setdefault((r["symbol"], r["interval"]), []).append(r)

        for (symbol, interval), rows in by_series.items():
            try:
                df = self.market.get(symbol, interval)
            except MarketDataError:
                continue  # try again next time
            stamps = df.index.asi8 // 10**9
            closes = df["close"].to_numpy()
            bar = interval_delta(interval).total_seconds()
            updates = []
            for r in rows:
                after = np.nonzero(stamps > r["origin_time"])[0]
                if len(after) < r["horizon"]:
                    continue
                final = after[r["horizon"] - 1]
                # The final bar is closed once a later bar exists or its period has elapsed
                closed = len(after) > r["horizon"] or stamps[final] + bar <= self.now()
                if closed:
                    updates.append((float(closes[final]), int(stamps[final]), self.now(), r["id"]))
            if updates:
                with self._lock, self._connect() as conn:
                    conn.executemany(
                        "UPDATE forecasts SET actual_final=?, actual_time=?, scored_at=? WHERE id=?", updates
                    )
                scored += len(updates)
        return scored

    def report(self, symbol: str | None = None, interval: str | None = None, model: str | None = None, limit: int = 100) -> dict:
        where, args = [], []
        for col, val in (("symbol", symbol and symbol.upper()), ("interval", interval), ("model", model)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock, self._connect() as conn:
            rows = [dict(r) for r in conn.execute(f"SELECT * FROM forecasts {clause} ORDER BY created_at DESC", args)]
            backtests = [dict(r) for r in conn.execute("SELECT * FROM backtests ORDER BY created_at DESC")]

        scored = [r for r in rows if r["scored_at"] is not None]
        groups = []
        for key in sorted({(r["symbol"], r["interval"], r["model"]) for r in rows}):
            members = [r for r in scored if (r["symbol"], r["interval"], r["model"]) == key]
            latest_bt = next((b for b in backtests if (b["symbol"], b["interval"], b["model"]) == key), None)
            groups.append(
                {
                    "symbol": key[0],
                    "interval": key[1],
                    "model": key[2],
                    "total": sum((r["symbol"], r["interval"], r["model"]) == key for r in rows),
                    "live": scorecard(members) if members else None,
                    "backtest": latest_bt and {k: latest_bt[k] for k in ("horizon", "windows", "hit_rate", "baseline_hit_rate", "r2_oos", "coverage", "created_at")},
                }
            )

        def public(r: dict) -> dict:
            out = {k: v for k, v in r.items() if k != "path"}
            if r["actual_final"] is not None:
                out["actual_return"] = r["actual_final"] / r["last_close"] - 1
                out["hit"] = bool(np.sign(r["pred_final"] - r["last_close"]) == np.sign(r["actual_final"] - r["last_close"]))
            return out

        return {
            "total": len(rows),
            "pending": len(rows) - len(scored),
            "scorecard": scorecard(scored) if scored else None,
            "groups": groups,
            "entries": [public(r) for r in rows[:limit]],
        }


def scorecard(rows: list[dict], seed: int = 0) -> dict:
    """Live accuracy of scored forecasts, using the same yardsticks as the backtest."""
    last = np.array([r["last_close"] for r in rows])
    actual = np.array([r["actual_final"] for r in rows]) / last - 1
    pred = np.array([r["pred_final"] for r in rows]) / last - 1
    p_up = np.array([r["p_up"] for r in rows])
    lo = np.array([r["p10_final"] for r in rows]) / last - 1
    hi = np.array([r["p90_final"] for r in rows]) / last - 1
    went_up = (actual > 0).astype(float)

    hits = np.sign(pred) == np.sign(actual)
    up_rate = went_up.mean()
    brier = float(np.mean((p_up - went_up) ** 2))

    # Luck test: shuffle which forecast made which call; how often does random assignment do as well?
    p_value = None
    calls = np.sign(pred)
    if len(np.unique(calls)) > 1:
        rng = np.random.default_rng(seed)
        shuffled = rng.permuted(np.tile(calls, (PERMUTATIONS, 1)), axis=1)
        perm_hits = (shuffled == np.sign(actual)).mean(axis=1)
        p_value = float((np.sum(perm_hits >= hits.mean()) + 1) / (PERMUTATIONS + 1))

    calibration = []
    for a, b in zip(CALIBRATION_EDGES[:-1], CALIBRATION_EDGES[1:]):
        m = (p_up >= a) & ((p_up < b) if b < 1 else (p_up <= b))
        if m.any():
            calibration.append({"lo": a, "hi": b, "n": int(m.sum()), "predicted": float(p_up[m].mean()), "observed": float(went_up[m].mean())})

    return {
        "n": len(rows),
        "hit_rate": float(hits.mean()),
        "baseline_hit_rate": float(max(up_rate, 1 - up_rate)),
        "r2_oos": r2_oos(actual, pred),
        "coverage_p10_p90": float(np.mean((actual >= lo) & (actual <= hi))),
        "mae_pct": float(np.mean(np.abs(actual - pred))),
        "brier": brier,
        # Against always saying 50%, which needs no hindsight; positive = better than a coin
        "brier_skill": 1 - brier / 0.25,
        "p_value": p_value,
        "calibration": calibration,
    }


def forecast_entry(req, last_close: float, origin_time: int, stats: dict, forecast: list[dict]) -> dict:
    """Shape a /api/forecast result into a journal row."""
    return {
        "symbol": req.symbol, "interval": req.interval, "model": req.model, "lookback": req.lookback,
        "horizon": len(forecast), "samples": req.samples, "temperature": req.temperature, "top_p": req.top_p,
        "origin_time": origin_time, "last_close": last_close, "p_up": stats["p_up"],
        "expected_return": stats["expected_return"], "pred_final": forecast[-1]["median"],
        "p10_final": forecast[-1]["p10"], "p90_final": forecast[-1]["p90"],
        "path": [{k: p[k] for k in ("time", "median", "p10", "p90")} for p in forecast],
    }

