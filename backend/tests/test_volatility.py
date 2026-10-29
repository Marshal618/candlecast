import numpy as np
import pandas as pd
import pytest

from app.services.volatility import CAL_WARMUP, ewma_variance, realized_variance, run_volatility

HORIZON = 5


def _universe(n_symbols=4, bars=500, seed=0):
    """Random walks whose volatility switches regime every 60 bars, so it is forecastable."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=bars, tz="America/New_York")
    frames = {}
    for k in range(n_symbols):
        sigma = np.repeat(rng.choice([0.005, 0.02, 0.04], size=bars // 60 + 1), 60)[:bars] * (1 + k / 4)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 1, bars) * sigma))
        frames[f"S{k}"] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e6, "amount": 1e8}, index=idx
        )
    return frames


class VarianceOracle:
    """Two paths whose horizon log returns have exactly the realized variance ahead, times `scale`."""

    def __init__(self, frames, scale=1.0):
        self.frames, self.scale = frames, scale

    def sample_paths(self, contexts, future_index, model, samples, temperature, top_p):
        out = np.zeros((len(contexts), samples, len(future_index[0]), 6))
        for i, c in enumerate(contexts):
            f = next(f for f in self.frames.values() if f["close"].loc[c.index[-1]] == c["close"].iloc[-1])
            p = f.index.get_loc(c.index[-1])
            r = np.diff(np.log(f["close"].iloc[p : p + HORIZON + 1].to_numpy()))
            a = np.sqrt(self.scale * (r**2).sum() / 2)  # var(+a, -a) with ddof=1 is 2a^2
            last = c["close"].iloc[-1]
            out[i, 0, :, 3], out[i, 1, :, 3] = last * np.exp(a), last * np.exp(-a)
        return out


def _run(frames, fc, **kw):
    args = dict(interval="1d", models=["mini"], lookback=64, horizon=HORIZON, periods=30, samples=2,
                temperature=1.0, top_p=0.9)
    return run_volatility(fc, frames, **{**args, **kw})


def test_realized_and_ewma_variance():
    closes = np.exp(np.cumsum(np.tile([0.01, -0.01], 50)))[:, None]
    assert realized_variance(closes, [10], 4)[0, 0] == pytest.approx(4 * 0.01**2)
    assert ewma_variance(closes[:, 0], 5) == pytest.approx(5 * 0.01**2)


def test_perfect_variance_forecast_has_zero_loss_and_beats_ewma():
    frames = _universe()
    res = _run(frames, VarianceOracle(frames))
    mini, ewma = res["forecasts"]
    assert (mini["name"], ewma["name"], ewma["kind"]) == ("mini", "ewma", "baseline")
    assert mini["qlike"] == pytest.approx(0, abs=1e-9) and mini["mz_r2"] == pytest.approx(1)
    assert mini["skill"] == pytest.approx(1) and mini["dm_t"] > 0  # EWMA losses are fat-tailed, so t stays modest
    assert mini["rank_ic"]["mean"] == pytest.approx(1) and mini["timing_corr"] == pytest.approx(1)
    assert 0 < ewma["mz_r2"] < 1  # vol regimes make the baseline useful but imperfect
    assert res["scored_periods"] == res["periods"] - CAL_WARMUP


def test_calibration_removes_a_constant_scale_error():
    frames = _universe()
    res = _run(frames, VarianceOracle(frames, scale=4.0))  # bands twice too wide
    m = res["forecasts"][0]
    assert m["vol_ratio"] == pytest.approx(0.5)
    assert m["qlike"] > 0.3 and m["qlike_cal"] == pytest.approx(0, abs=1e-9)
    assert m["mz_slope"] == pytest.approx(1) and m["mz_r2"] == pytest.approx(1)  # scale-free
    assert res["forecasts"][0]["calibrated"][0][0] is None  # warm-up periods have no calibrated value


def test_too_few_periods_is_rejected():
    frames = _universe(bars=200)
    with pytest.raises(ValueError):
        _run(frames, VarianceOracle(frames), periods=4)
