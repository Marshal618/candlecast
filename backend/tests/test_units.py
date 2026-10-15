"""Fast tests: no network, no model weights."""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest import run_backtest
from app.services.forecaster import FEATURES, summarize
from app.services.market_data import CachedMarketData, MarketDataProvider, future_timestamps


def _ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.linspace(100, 110, len(index))
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index
    )


def test_future_timestamps_daily_skips_weekends():
    idx = pd.bdate_range("2026-01-05", periods=30, tz="America/New_York")  # Mon..
    fut = future_timestamps(idx, "1d", 6)
    assert all(t.weekday() < 5 for t in fut)
    assert fut[0] > idx[-1]
    assert len(fut) == 6


def test_future_timestamps_intraday_respects_session_hours():
    days = pd.bdate_range("2026-01-05", periods=10)
    idx = pd.DatetimeIndex(
        [d + pd.Timedelta(hours=h, minutes=30) for d in days for h in range(9, 16)]
    ).tz_localize("America/New_York")
    fut = future_timestamps(idx, "1h", 10)
    allowed = set(idx.time)
    assert all(t.time() in allowed for t in fut)
    assert all(t.weekday() < 5 for t in fut)


def test_future_timestamps_crypto_is_continuous():
    idx = pd.date_range("2026-01-01", periods=200, freq="1h", tz="UTC")
    fut = future_timestamps(idx, "1h", 48)
    assert (fut[1:] - fut[:-1] == pd.Timedelta("1h")).all()


def test_summarize_stats():
    # 4 paths, horizon 3; finals 90, 105, 110, 120 vs last close 100
    finals = [90, 105, 110, 120]
    paths = np.zeros((4, 3, len(FEATURES)))
    for i, f in enumerate(finals):
        paths[i, :, FEATURES.index("close")] = f
    s = summarize(paths, 100.0)
    assert s["p_up"] == 0.75
    assert s["expected_return"] == pytest.approx(0.075)
    assert s["p10"][-1] <= s["median"][-1] <= s["p90"][-1]


class FakeProvider(MarketDataProvider):
    calls = 0

    def fetch(self, symbol, interval):
        FakeProvider.calls += 1
        return _ohlcv(pd.date_range("2026-01-01", periods=50, freq="1D", tz="America/New_York"))


def test_cache_roundtrip_preserves_timezone(tmp_path):
    cache = CachedMarketData(FakeProvider(), tmp_path / "c.sqlite3")
    a = cache.get("aapl", "1d")
    b = cache.get("AAPL", "1d")
    assert FakeProvider.calls == 1  # second call served from cache, symbol normalised
    assert str(b.index.tz) == "America/New_York"
    pd.testing.assert_frame_equal(a, b, check_freq=False, check_dtype=False)


class FakeForecaster:
    """Predicts 'no change' with a ±1 spread so metrics are deterministic."""

    def sample_paths(self, contexts, future_index, model, samples, temperature, top_p):
        horizon = len(future_index[0])
        out = np.zeros((len(contexts), samples, horizon, len(FEATURES)))
        for i, ctx in enumerate(contexts):
            last = ctx["close"].iloc[-1]
            for s in range(samples):
                out[i, s, :, :4] = last + (s - (samples - 1) / 2)
        return out


def test_backtest_uses_only_past_data_and_scores():
    idx = pd.date_range("2026-01-01", periods=300, freq="1D", tz="UTC")
    df = _ohlcv(idx)  # steadily rising
    r = run_backtest(
        FakeForecaster(), df, lookback=100, horizon=5, windows=10, step=5,
        model="small", samples=5, temperature=1.0, top_p=0.9,
    )
    m = r["metrics"]
    assert m["windows"] == 10
    assert m["baseline_hit_rate"] == 1.0  # always up
    for w in r["windows"]:
        assert w["origin_time"] < w["time"]
    # Flat forecast on a rising series: the model's error equals the naive baseline's
    assert m["mae_pct"] == pytest.approx(m["naive_mae_pct"])


def test_backtest_rejects_short_history():
    df = _ohlcv(pd.date_range("2026-01-01", periods=50, freq="1D", tz="UTC"))
    with pytest.raises(ValueError, match="Not enough history"):
        run_backtest(
            FakeForecaster(), df, lookback=100, horizon=5, windows=3, step=5,
            model="small", samples=2, temperature=1.0, top_p=0.9,
        )
