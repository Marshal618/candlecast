"""Market data providers + a small SQLite cache.

Swap providers by implementing `MarketDataProvider.fetch` (e.g. Polygon, Alpaca) and
passing it to `CachedMarketData`.
"""

import io
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]

# interval -> (yfinance history period, cache TTL seconds)
INTERVALS: dict[str, tuple[str, int]] = {
    "5m": ("60d", 60),
    "15m": ("60d", 120),
    "1h": ("730d", 300),
    "1d": ("10y", 3600),
}


class MarketDataError(Exception):
    pass


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch(self, symbol: str, interval: str) -> pd.DataFrame:
        """Return a DataFrame indexed by tz-aware timestamps with lowercase OHLCV columns."""


class YFinanceProvider(MarketDataProvider):
    def fetch(self, symbol: str, interval: str) -> pd.DataFrame:
        import yfinance as yf

        period, _ = INTERVALS[interval]
        try:
            raw = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        except Exception as e:  # yfinance raises a zoo of exception types
            raise MarketDataError(f"Failed to fetch {symbol}: {e}") from e
        if raw is None or raw.empty:
            raise MarketDataError(f"No data returned for '{symbol}' ({interval}). Check the symbol.")

        df = raw.rename(columns=str.lower)[OHLCV].astype("float64")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df["volume"] = df["volume"].fillna(0.0)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.index.name = "timestamp"
        return df


class CachedMarketData:
    def __init__(self, provider: MarketDataProvider, cache_path: Path):
        self.provider = provider
        self.cache_path = cache_path
        self._lock = threading.Lock()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS candles ("
                " symbol TEXT, interval TEXT, fetched_at REAL, payload TEXT,"
                " PRIMARY KEY (symbol, interval))"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.cache_path)

    def get(self, symbol: str, interval: str) -> pd.DataFrame:
        if interval not in INTERVALS:
            raise MarketDataError(f"Unsupported interval '{interval}'. Use one of {list(INTERVALS)}.")
        symbol = symbol.strip().upper()
        ttl = INTERVALS[interval][1]

        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT fetched_at, payload FROM candles WHERE symbol=? AND interval=?", (symbol, interval)
            ).fetchone()
        if row and time.time() - row[0] < ttl:
            return _from_json(row[1])

        df = self.provider.fetch(symbol, interval)
        df.index.name = "timestamp"
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?)",
                (symbol, interval, time.time(), _to_json(df)),
            )
        return df


def _to_json(df: pd.DataFrame) -> str:
    tz = str(df.index.tz) if df.index.tz is not None else ""
    out = df.copy()
    out.index = out.index.tz_convert("UTC") if df.index.tz is not None else out.index
    return tz + "\n" + out.to_json(orient="split", date_unit="s")


def _from_json(payload: str) -> pd.DataFrame:
    tz, body = payload.split("\n", 1)
    df = pd.read_json(io.StringIO(body), orient="split", convert_dates=False)
    df.index = pd.to_datetime(df.index, unit="s", utc=True)
    if tz:
        df.index = df.index.tz_convert(tz)
    df.index.name = "timestamp"
    return df


def future_timestamps(index: pd.DatetimeIndex, interval: str, n: int) -> pd.DatetimeIndex:
    """Project `n` future bar timestamps, skipping times the market was never observed open.

    Learns which weekdays / times-of-day appear in history, so stocks skip nights and
    weekends while 24/7 assets (crypto) stay continuous. Holidays are not modelled.
    """
    # Calendar-day offset for daily bars keeps wall-clock time stable across DST changes
    step = pd.DateOffset(days=1) if interval == "1d" else pd.Timedelta(_interval_delta(interval))
    weekdays = set(index.weekday)
    times = set(index.time) if interval != "1d" else None
    out = []
    t = index[-1]
    # Bound the search so a weird history can't loop forever (a 5m weekend gap is ~800 steps)
    for _ in range(n * 500 + 5000):
        t = t + step
        if t.weekday() not in weekdays:
            continue
        if times is not None and t.time() not in times:
            continue
        out.append(t)
        if len(out) == n:
            break
    return pd.DatetimeIndex(out, name=index.name)


def _interval_delta(interval: str) -> str:
    return {"5m": "5min", "15m": "15min", "1h": "1h", "1d": "1D"}[interval]


def interval_delta(interval: str) -> pd.Timedelta:
    """How long one bar of `interval` lasts."""
    return pd.Timedelta(_interval_delta(interval))


def closed_bars(df: pd.DataFrame, interval: str, now: float) -> pd.DataFrame:
    """Drop the last bar if its period hasn't ended yet (providers include the in-progress candle)."""
    if len(df) and df.index[-1].timestamp() + interval_delta(interval).total_seconds() > now:
        return df.iloc[:-1]
    return df
