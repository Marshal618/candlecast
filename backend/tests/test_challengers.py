"""Challenger models: output shape, determinism, sanity of their distributions."""

import numpy as np
import pandas as pd
import pytest

from app.services import challengers as ch


def _ctx(n=400, drift=0.001, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)


class FakeChronos:
    """Quantiles widening linearly with the step, centred on the last close."""

    def paths(self, contexts, horizon, samples):
        out = []
        for c in contexts:
            last = c["close"].iloc[-1]
            levels = np.array(ch.CHRONOS_LEVELS)
            q = np.array([[last * (1 + (lv - 0.5) * 0.02 * (k + 1)) for lv in levels] for k in range(horizon)])
            u = np.clip((np.arange(samples) + 0.5) / samples, levels[0], levels[-1])
            out.append(np.stack([np.interp(u, levels, q[k]) for k in range(horizon)], axis=1))
        return np.stack(out)


@pytest.mark.parametrize("model", ["naive", "garch"])
def test_shape_and_ohlc_consistency(model):
    paths = ch.sample_paths([_ctx(), _ctx(seed=1)], horizon=7, model=model, samples=12, chronos=None)
    assert paths.shape == (2, 12, 7, 6)
    o, h, lo, c = (paths[..., i] for i in range(4))
    assert (h >= np.maximum(o, c) - 1e-9).all() and (lo <= np.minimum(o, c) + 1e-9).all()
    assert np.allclose(o[:, :, 0], [[_ctx()["close"].iloc[-1]], [_ctx(seed=1)["close"].iloc[-1]]])


@pytest.mark.parametrize("model", ["naive", "garch"])
def test_deterministic_per_context(model):
    a = ch.sample_paths([_ctx()], 5, model, 8, None)
    b = ch.sample_paths([_ctx()], 5, model, 8, None)
    assert np.array_equal(a, b)


def test_naive_carries_historical_drift():
    up = ch.naive_paths(_ctx(drift=0.003, vol=0.002), horizon=20, samples=200)
    down = ch.naive_paths(_ctx(drift=-0.003, vol=0.002), horizon=20, samples=200)
    assert np.median(up[:, -1]) > _ctx(drift=0.003, vol=0.002)["close"].iloc[-1]
    assert np.median(down[:, -1]) < _ctx(drift=-0.003, vol=0.002)["close"].iloc[-1]


def test_garch_bands_scale_with_volatility():
    calm = ch.garch_paths(_ctx(vol=0.005), horizon=10, samples=200)
    wild = ch.garch_paths(_ctx(vol=0.03), horizon=10, samples=200)
    spread = lambda p: np.percentile(p[:, -1], 90) / np.percentile(p[:, -1], 10)  # noqa: E731
    assert spread(wild) > spread(calm)


def test_garch_falls_back_on_degenerate_series():
    flat = _ctx()
    flat["close"] = 100.0
    paths = ch.garch_paths(flat, horizon=3, samples=4)
    assert np.isfinite(paths).all()


def test_chronos_comonotone_paths_are_ordered():
    paths = ch.sample_paths([_ctx()], horizon=5, model="chronos", samples=10, chronos=FakeChronos())
    finals = paths[0, :, -1, 3]
    assert (np.diff(finals) >= 0).all()  # path j sits at quantile level u_j throughout
    assert finals[0] < _ctx()["close"].iloc[-1] < finals[-1]


def test_unknown_challenger():
    with pytest.raises(ValueError):
        ch.sample_paths([_ctx()], 3, "prophet", 4, None)
