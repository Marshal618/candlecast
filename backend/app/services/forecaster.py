"""Kronos inference wrapper.

Upstream `KronosPredictor.predict` averages its sampled paths, which hides uncertainty.
We instead batch N copies of the same series with sample_count=1, so each copy is an
independent sampled future, and compute quantile bands from those paths ourselves.
"""

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.config import BACKEND_DIR, settings

sys.path.insert(0, str(BACKEND_DIR / "vendor" / "kronos"))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

from app.services import challengers, tokens  # noqa: E402  (tokens needs the vendor path above)


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    tokenizer: str
    max_context: int


MODELS: dict[str, ModelSpec] = {
    "mini": ModelSpec("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": ModelSpec("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ModelSpec("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}

FEATURES = ["open", "high", "low", "close", "volume", "amount"]

# Every model the app can forecast with, Kronos sizes plus challengers, and its max lookback
MAX_CONTEXT: dict[str, int] = {
    **{k: v.max_context for k, v in MODELS.items()},
    **{k: challengers.MAX_CONTEXT for k in challengers.CHALLENGERS},
}


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Forecaster:
    def __init__(self, device: str = "auto", batch_size: int = 32):
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._predictors: dict[str, KronosPredictor] = {}
        self._chronos = challengers.ChronosModel(self.device)
        # Loading and torch inference are not safe to run concurrently on one device
        self._lock = threading.Lock()

    @property
    def loaded_models(self) -> list[str]:
        return list(self._predictors) + (["chronos"] if self._chronos.loaded else [])

    def _predictor(self, name: str) -> KronosPredictor:
        if name not in MODELS:
            raise ValueError(f"Unknown model '{name}'. Use one of {list(MODELS)}.")
        if name not in self._predictors:
            spec = MODELS[name]
            tokenizer = KronosTokenizer.from_pretrained(spec.tokenizer)
            model = Kronos.from_pretrained(spec.repo)
            tokenizer.eval()
            model.eval()
            self._predictors[name] = KronosPredictor(
                model, tokenizer, device=self.device, max_context=spec.max_context
            )
        return self._predictors[name]

    def load(self, name: str) -> None:
        with self._lock:
            if name == "chronos":
                self._chronos.load()
            elif name not in challengers.CHALLENGERS:
                self._predictor(name)

    def sample_paths(
        self,
        contexts: list[pd.DataFrame],
        future_index: list[pd.DatetimeIndex],
        model: str,
        samples: int,
        temperature: float,
        top_p: float,
    ) -> np.ndarray:
        """Sample `samples` independent futures for each context.

        All contexts must share length, and all future indexes must share length.
        Returns array of shape (len(contexts), samples, horizon, 6) in FEATURES order.
        """
        horizon = len(future_index[0])
        if model in challengers.CHALLENGERS:
            with self._lock:
                return challengers.sample_paths(contexts, horizon, model, samples, self._chronos)
        # Each (context, sample) pair becomes its own batch row
        jobs = [(i, ctx, fut) for i, (ctx, fut) in enumerate(zip(contexts, future_index)) for _ in range(samples)]
        out = np.empty((len(contexts), samples, horizon, len(FEATURES)), dtype=np.float64)
        filled = [0] * len(contexts)

        with self._lock:
            predictor = self._predictor(model)
            for start in range(0, len(jobs), self.batch_size):
                chunk = jobs[start : start + self.batch_size]
                preds = predictor.predict_batch(
                    [c[1] for c in chunk],
                    [pd.Series(c[1].index) for c in chunk],
                    [pd.Series(c[2]) for c in chunk],
                    pred_len=horizon,
                    T=temperature,
                    top_p=top_p,
                    sample_count=1,
                    verbose=False,
                )
                for (i, _, _), pred in zip(chunk, preds):
                    out[i, filled[i]] = pred[FEATURES].to_numpy()
                    filled[i] += 1
        return out


    def inspect(self, ctx: pd.DataFrame, history: pd.DataFrame, model: str, segments: int) -> dict:
        """Deterministic post-hoc token analysis of one context window (see services/tokens.py)."""
        with self._lock:
            return tokens.inspect(self._predictor(model), ctx, history, segments)


def projected_candles(paths: np.ndarray) -> np.ndarray:
    """Pick the sampled path whose closes sit nearest the per-bar median (the medoid).

    A per-field median of OHLC can produce impossible bars (high below close), whereas
    one real sampled path keeps each bar's open/high/low/close mutually consistent.
    Returns (horizon, 4) OHLC with high/low clamped to enclose open and close.
    """
    close = paths[:, :, FEATURES.index("close")]
    median = np.median(close, axis=0)
    medoid = int(np.argmin(np.sum((close - median) ** 2, axis=1)))
    ohlc = paths[medoid, :, :4].copy()
    ohlc[:, 1] = ohlc.max(axis=1)
    ohlc[:, 2] = ohlc.min(axis=1)
    return ohlc


def summarize(paths: np.ndarray, last_close: float) -> dict:
    """Turn sampled paths (samples, horizon, 6) into chart-ready stats."""
    close = paths[:, :, FEATURES.index("close")]
    final = close[:, -1]
    return {
        "median": np.median(close, axis=0),
        "p10": np.percentile(close, 10, axis=0),
        "p90": np.percentile(close, 90, axis=0),
        "ohlc": projected_candles(paths),
        "p_up": float(np.mean(final > last_close)),
        "expected_return": float(np.median(final) / last_close - 1.0),
        "band_width_pct": float((np.percentile(final, 90) - np.percentile(final, 10)) / last_close),
    }


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000
