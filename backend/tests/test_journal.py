"""Forecast journal: dedupe, scoring only closed bars, live scorecard, live vs backtest."""

import numpy as np
import pandas as pd
import pytest

from app.services.journal import Journal, scorecard


class FakeMarket:
    def __init__(self, df):
        self.df = df

    def get(self, symbol, interval):
        return self.df


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _df(n, start="2026-01-05"):
    idx = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    close = 100 + np.arange(n, dtype=float)  # +1 per bar
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)


def _entry(df, origin_idx, horizon=3, p_up=0.8, pred=1.02, **kw):
    last = float(df["close"].iloc[origin_idx])
    e = {
        "symbol": "spy", "interval": "1d", "model": "mini", "lookback": 64, "horizon": horizon, "samples": 10,
        "temperature": 1.0, "top_p": 0.9, "origin_time": int(df.index[origin_idx].timestamp()),
        "last_close": last, "p_up": p_up, "expected_return": pred - 1, "pred_final": last * pred,
        "p10_final": last * 0.97, "p90_final": last * 1.05, "path": [],
    }
    e.update(kw)
    return e


@pytest.fixture
def setup(tmp_path):
    df = _df(10)
    market = FakeMarket(df)
    clock = Clock(df.index[-1].timestamp() + 3600)  # an hour into the last daily bar
    return Journal(tmp_path / "j.sqlite3", market, clock), market, clock, df


def test_first_forecast_per_origin_wins(setup):
    j, _, _, df = setup
    a, created_a = j.record_forecast(_entry(df, 5, p_up=0.9))
    b, created_b = j.record_forecast(_entry(df, 5, p_up=0.1, temperature=1.5))
    assert created_a and not created_b and a == b
    assert j.report()["entries"][0]["p_up"] == 0.9
    # A different horizon or model is a different forecast
    _, created_c = j.record_forecast(_entry(df, 5, horizon=2))
    assert created_c


def test_scores_only_after_final_bar_closes(setup):
    j, market, clock, df = setup
    j.record_forecast(_entry(df, 6, horizon=3))  # needs bars 7, 8, 9; bar 9 is still open
    assert j.score_pending() == 0
    assert j.report()["pending"] == 1

    clock.t = df.index[-1].timestamp() + 86400  # bar 9 has now closed
    assert j.score_pending() == 1
    e = j.report()["entries"][0]
    assert e["actual_final"] == df["close"].iloc[9]
    assert e["actual_return"] == pytest.approx(df["close"].iloc[9] / df["close"].iloc[6] - 1)
    assert e["hit"] is True


def test_scores_when_a_later_bar_exists(setup):
    j, _, _, df = setup
    j.record_forecast(_entry(df, 5, horizon=3))  # bars 6-8, and bar 9 proves 8 closed
    assert j.score_pending() == 1


def test_not_scored_without_enough_bars(setup):
    j, _, clock, df = setup
    clock.t += 10 * 86400
    j.record_forecast(_entry(df, 8, horizon=5))
    assert j.score_pending() == 0


def test_report_groups_compare_live_with_backtest(setup):
    j, _, _, df = setup
    j.record_forecast(_entry(df, 3))
    j.record_forecast(_entry(df, 4))
    j.record_backtest("SPY", "1d", "mini", 3, {"windows": 20, "hit_rate": 0.7, "baseline_hit_rate": 0.6,
                                                "r2_oos": 0.02, "coverage_p10_p90": 0.8})
    j.score_pending()
    g = j.report(symbol="spy")["groups"][0]
    assert (g["symbol"], g["interval"], g["model"]) == ("SPY", "1d", "mini")
    assert g["live"]["n"] == 2 and g["live"]["hit_rate"] == 1.0
    assert g["backtest"]["hit_rate"] == 0.7


def test_scorecard_metrics_and_calibration():
    rows = []
    rng = np.random.default_rng(1)
    for i in range(200):
        p = rng.uniform()
        up = rng.uniform() < p  # perfectly calibrated forecaster
        rows.append({"last_close": 100.0, "actual_final": 101.0 if up else 99.0,
                     "pred_final": 100.5 if p >= 0.5 else 99.5, "p_up": p,
                     "p10_final": 98.0, "p90_final": 102.0})
    s = scorecard(rows)
    assert s["n"] == 200
    assert s["coverage_p10_p90"] == 1.0
    assert s["brier_skill"] > 0  # informative probabilities beat a constant 50%
    for b in s["calibration"]:
        assert abs(b["predicted"] - b["observed"]) < 0.2
    assert s["p_value"] < 0.05


def test_scorecard_coin_flip_has_no_skill():
    rng = np.random.default_rng(2)
    rows = [{"last_close": 100.0, "actual_final": 101.0 if rng.uniform() < 0.5 else 99.0,
             "pred_final": 101.0 if rng.uniform() < 0.5 else 99.0, "p_up": 0.5,
             "p10_final": 99.5, "p90_final": 100.5} for _ in range(200)]
    s = scorecard(rows)
    assert s["brier_skill"] == pytest.approx(0.0)
    assert s["p_value"] > 0.05
