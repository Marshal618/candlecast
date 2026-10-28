"""Challenger forecasters to compare Kronos against.

Each returns sampled future paths shaped exactly like Kronos's (contexts, samples, horizon, 6)
in FEATURES order, so forecasts, backtests, calibration and the journal treat every model alike.

- naive:   historical average drift + block-bootstrapped past returns. The no-skill benchmark
           (Goyal & Welch 2008): a real signal has to beat it.
- garch:   GARCH(1,1) with Student-t shocks (Bollerslev 1986), simulated forward. Mean is a
           constant; its strength is volatility, i.e. band width.
- chronos: Amazon Chronos-Bolt (Ansari et al. 2024), a general time-series foundation model,
           zero-shot on closes. Bolt predicts quantiles 0.1-0.9 per step rather than samples, so
           paths are drawn comonotonically (path j sits at quantile level u_j at every step).
           The final-step distribution is exact within 0.1-0.9; tails beyond are clamped.

The challengers only model the close. Open is the previous close and high/low bracket the
two, so their "candles" are bodies without wicks; the UI compares them as lines and bands.
"""

import threading
import warnings
import zlib

import numpy as np
import pandas as pd

CHALLENGERS = ("naive", "garch", "chronos")
MAX_CONTEXT = 2048
BLOCK = 5  # bootstrap block length, keeps short-range autocorrelation and vol clustering
GARCH_MAX_OBS = 1000
CHRONOS_REPO = "amazon/chronos-bolt-small"
CHRONOS_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _seed(ctx: pd.DataFrame, model: str) -> int:
    """Deterministic per context, so backtests are reproducible."""
    key = f"{model}:{ctx.index[-1]}:{len(ctx)}:{ctx['close'].iloc[-1]:.6f}"
    return zlib.crc32(key.encode())


def _to_features(close_paths: np.ndarray, last_close: float) -> np.ndarray:
    """(samples, horizon) closes -> (samples, horizon, 6) open/high/low/close/volume/amount."""
    s, h = close_paths.shape
    opens = np.concatenate([np.full((s, 1), last_close), close_paths[:, :-1]], axis=1)
    out = np.zeros((s, h, 6))
    out[:, :, 0] = opens
    out[:, :, 1] = np.maximum(opens, close_paths)
    out[:, :, 2] = np.minimum(opens, close_paths)
    out[:, :, 3] = close_paths
    return out


def _log_returns(ctx: pd.DataFrame) -> np.ndarray:
    return np.diff(np.log(ctx["close"].to_numpy(dtype=float)))


def naive_paths(ctx: pd.DataFrame, horizon: int, samples: int) -> np.ndarray:
    rng = np.random.default_rng(_seed(ctx, "naive"))
    r = _log_returns(ctx)
    mu = r.mean()
    resid = r - mu
    starts = rng.integers(0, max(len(resid) - BLOCK, 1), size=(samples, -(-horizon // BLOCK)))
    draws = np.stack([np.concatenate([resid[s : s + BLOCK] for s in row])[:horizon] for row in starts])
    last = float(ctx["close"].iloc[-1])
    return last * np.exp(np.cumsum(mu + draws, axis=1))


def garch_paths(ctx: pd.DataFrame, horizon: int, samples: int) -> np.ndarray:
    from arch.univariate import GARCH, ConstantMean, StudentsT

    r = _log_returns(ctx)[-GARCH_MAX_OBS:] * 100  # percent returns keep the optimizer well scaled
    if len(r) < 50 or r.std() < 1e-8:
        return naive_paths(ctx, horizon, samples)  # nothing for a volatility model to fit
    last = float(ctx["close"].iloc[-1])
    model = ConstantMean(r)
    model.volatility = GARCH(p=1, q=1)
    # Simulated shocks come from the distribution's own generator; seed it for reproducibility
    model.distribution = StudentsT(seed=np.random.default_rng(_seed(ctx, "garch")))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on short or odd series
            res = model.fit(disp="off", options={"maxiter": 200})
            sims = res.forecast(horizon=horizon, method="simulation", simulations=samples, reindex=False)
        paths = sims.simulations.values[-1]  # (samples, horizon), percent returns
    except Exception:  # optimizer failure: fall back to the benchmark rather than erroring
        return naive_paths(ctx, horizon, samples)
    if not np.isfinite(paths).all():
        return naive_paths(ctx, horizon, samples)
    return last * np.exp(np.cumsum(paths / 100, axis=1))


class ChronosModel:
    """Lazily loaded Chronos-Bolt pipeline (weights download on first use, ~200MB)."""

    def __init__(self, device: str):
        # Bolt is small; CPU avoids MPS operator gaps and is fast enough
        self.device = "cuda" if device.startswith("cuda") else "cpu"
        self._pipe = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._pipe is None:
                import torch
                from chronos import BaseChronosPipeline

                kw = {"device_map": self.device, "dtype": torch.float32}
                try:
                    # Cached weights first: skips the Hub round-trip, which can stall on a slow network
                    self._pipe = BaseChronosPipeline.from_pretrained(CHRONOS_REPO, local_files_only=True, **kw)
                except Exception:
                    self._pipe = BaseChronosPipeline.from_pretrained(CHRONOS_REPO, **kw)
        return self._pipe

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def paths(self, contexts: list[pd.DataFrame], horizon: int, samples: int) -> np.ndarray:
        import torch

        pipe = self.load()
        inputs = [torch.tensor(c["close"].to_numpy(dtype=float)[-MAX_CONTEXT:], dtype=torch.float32) for c in contexts]
        q, _ = pipe.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=CHRONOS_LEVELS)
        q = q.float().cpu().numpy()  # (contexts, horizon, levels)
        q = np.sort(q, axis=-1)  # guard against tiny quantile crossings
        u = np.clip((np.arange(samples) + 0.5) / samples, CHRONOS_LEVELS[0], CHRONOS_LEVELS[-1])
        out = np.empty((len(contexts), samples, horizon))
        for i in range(len(contexts)):
            for k in range(horizon):
                out[i, :, k] = np.interp(u, CHRONOS_LEVELS, q[i, k])
        return out


def sample_paths(
    contexts: list[pd.DataFrame], horizon: int, model: str, samples: int, chronos: ChronosModel
) -> np.ndarray:
    """(contexts, samples, horizon, 6) for a challenger model."""
    lasts = [float(c["close"].iloc[-1]) for c in contexts]
    if model == "chronos":
        closes = chronos.paths(contexts, horizon, samples)
    elif model == "garch":
        closes = np.stack([garch_paths(c, horizon, samples) for c in contexts])
    elif model == "naive":
        closes = np.stack([naive_paths(c, horizon, samples) for c in contexts])
    else:
        raise ValueError(f"Unknown challenger '{model}'.")
    return np.stack([_to_features(cp, last) for cp, last in zip(closes, lasts)])
