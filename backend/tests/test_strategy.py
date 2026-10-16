"""Strategy simulator: execution timing, costs, benchmark, and the permutation test."""

import numpy as np
import pandas as pd
import pytest

from app.services.strategy import positions_from_signal, run_strategy


def _df(close, open_=None):
    close = np.asarray(close, dtype=float)
    open_ = close.copy() if open_ is None else np.asarray(open_, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(close), freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close), "low": np.minimum(open_, close), "close": close,
         "volume": 1.0},
        index=idx,
    )


def _windows(df, origins, p_ups):
    return [{"origin_time": int(df.index[o].timestamp()), "p_up": p} for o, p in zip(origins, p_ups)]


def test_signal_mapping():
    p = np.array([0.9, 0.6, 0.5, 0.4, 0.1])
    assert positions_from_signal(p, "long_only", 0.6).tolist() == [1, 1, 0, 0, 0]
    assert positions_from_signal(p, "long_short", 0.6).tolist() == [1, 1, 0, -1, -1]


def test_always_long_without_costs_matches_buy_and_hold():
    df = _df(np.linspace(100, 150, 60))
    r = run_strategy(df, _windows(df, [9, 19, 29], [1.0, 1.0, 1.0]), horizon=10, cost_bps=0)
    m = r["metrics"]
    assert m["total_return"] == pytest.approx(m["bh_return"])
    assert m["exposure"] == 1.0
    assert m["trades"] == 1  # stays long across windows instead of churning
    assert m["p_value"] is None  # nothing to permute


def test_flat_strategy_earns_nothing():
    df = _df(np.linspace(100, 150, 60))
    r = run_strategy(df, _windows(df, [9, 19], [0.5, 0.5]), horizon=10)
    assert r["metrics"]["total_return"] == 0.0
    assert r["metrics"]["win_rate"] is None


def test_no_lookahead_signal_bar_move_is_not_captured():
    # Price jumps on the origin bar's close and is flat afterwards; a long signal formed at that
    # close can only trade from the next open, so it must earn nothing.
    close = [100.0] * 10 + [120.0] * 20
    df = _df(close)
    r = run_strategy(df, _windows(df, [10], [1.0]), horizon=10, cost_bps=0)
    assert r["metrics"]["total_return"] == pytest.approx(0.0)


def test_gap_before_entry_goes_to_previous_position():
    # Flat window, then long. The overnight gap into the long window's first bar happens before
    # the order fills at the open, so the strategy misses it.
    close = [100.0] * 15
    open_ = [100.0] * 15
    open_[6] = 110.0  # gap up at the open of bar 6 (first bar of the long window)
    close[6:] = [110.0] * 9
    df = _df(close, open_)
    # Window 0 (bars 2-5) is flat, window 1 (bars 6-9) is long; buy-and-hold is already in from bar 2
    r = run_strategy(df, _windows(df, [1, 5], [0.5, 1.0]), horizon=4, cost_bps=0)
    assert r["metrics"]["total_return"] == pytest.approx(0.0)
    assert r["metrics"]["bh_return"] == pytest.approx(0.1)


def test_costs_charged_on_entry_and_exit():
    df = _df([100.0] * 30)
    r = run_strategy(df, _windows(df, [9], [1.0]), horizon=10, cost_bps=10)
    assert r["metrics"]["total_return"] == pytest.approx((1 - 0.001) ** 2 - 1)


def test_short_profits_on_falling_series():
    df = _df(np.linspace(150, 100, 60))
    r = run_strategy(df, _windows(df, [9, 19], [0.0, 0.0]), horizon=10, mode="long_short", cost_bps=0)
    assert r["metrics"]["total_return"] > 0 > r["metrics"]["bh_return"]


def test_perfect_timing_has_low_p_value():
    # Alternating up/down regimes; a signal that's long exactly in the up regimes should beat
    # nearly every random reshuffle of the same positions.
    close, p_ups, origins, level = [100.0], [], [], 100.0
    for k in range(20):
        up = k % 2 == 0
        origins.append(len(close) - 1)
        p_ups.append(1.0 if up else 0.0)
        for _ in range(5):
            level *= 1.02 if up else 0.98
            close.append(level)
    df = _df(close)
    r = run_strategy(df, _windows(df, origins, p_ups), horizon=5, cost_bps=0)
    assert r["metrics"]["p_value"] < 0.01


def test_rejects_windows_not_in_data():
    df = _df([100.0] * 30)
    with pytest.raises(ValueError, match="Re-run"):
        run_strategy(df, [{"origin_time": 123, "p_up": 1.0}], horizon=5)
