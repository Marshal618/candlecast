"""Volatility test: can a model forecast how much an asset will move, rather than which way?

Direction is close to unpredictable from prices alone; the size of moves isn't. Volatility
clusters (Mandelbrot 1963, Engle 1982), which is why GARCH exists. So a forecaster can be useless
for timing and still useful for risk. This scores each model's sampled spread on that job.

Walk-forward on the same rebalance schedule as the Research tab (every `horizon` bars):
- Forecast: the variance across a model's sampled paths of the log return over the next
  `horizon` bars. It comes out of the same forecasts the Research tab ranks with, and is cached
  with them.
- Baseline: EWMA variance of past daily log returns, lambda 0.94 (RiskMetrics 1996), times the
  horizon. Cheap and hard to beat; a model that can't beat it adds nothing for risk.
- Realized: the sum of squared per-bar log returns over the next `horizon` bars (realized variance,
  Andersen & Bollerslev 1998), a far less noisy target than the squared horizon return.

Scores, pooled over every (rebalance, symbol):
- QLIKE loss, rv/f - log(rv/f) - 1 (Patton 2011): the loss that ranks variance forecasts
  correctly even with a noisy target. 0 is perfect; lower is better.
- Calibrated QLIKE: raw spreads are often off by a constant factor (Kronos's bands were too
  narrow), which raw QLIKE punishes. Each period's forecasts are first rescaled by the average
  realized/forecast ratio of *earlier* periods, the same walk-forward idea as the conformal bands.
- Skill vs EWMA: 1 - calibrated QLIKE / EWMA's. Above 0 beats the baseline. The Diebold & Mariano
  (1995) t-stat tests the per-period loss difference; > 2 means the edge is unlikely to be luck.
- Mincer & Zarnowitz (1969) regression of log realized on log forecast: R^2 is how much of the
  variation in realized vol the forecast explains (scale-free); slope 1 means the right sensitivity.
- Ranking: RankIC across symbols each rebalance (which names will be wildest) and the average
  correlation over time within each symbol (when one will get wilder).
- Vol ratio: realized vol / raw forecast vol. 1 = right size; above 1 = bands too narrow.

The first CAL_WARMUP rebalances only seed the calibration, so every model is scored on the same
later periods.
"""

import numpy as np
import pandas as pd

from app.services.crosssection import ScoreCache, _ic_stats, _ranks, _rowwise_corr, cached_stat, schedule
from app.services.forecaster import Forecaster
from app.services.strategy import bars_per_year

EWMA_LAMBDA = 0.94
EWMA_BARS = 500
CAL_WARMUP = 3
FLOOR = 1e-10  # variance floor, keeps a flat stretch (halted symbol) from producing log(0)


def ewma_variance(closes: np.ndarray, horizon: int) -> float:
    """RiskMetrics variance of the next `horizon` bars' log return from past closes."""
    r = np.diff(np.log(closes[-EWMA_BARS - 1 :]))
    if len(r) == 0:
        return FLOOR
    v = float(np.mean(r[:20] ** 2))
    for x in r:
        v = EWMA_LAMBDA * v + (1 - EWMA_LAMBDA) * x * x
    return horizon * v


def realized_variance(closes: np.ndarray, origins: list[int], horizon: int) -> np.ndarray:
    """(periods, symbols) sum of squared per-bar log returns over the `horizon` bars after each origin."""
    log_r = np.diff(np.log(closes), axis=0)  # row j = return into bar j + 1
    return np.stack([(log_r[o : o + horizon] ** 2).sum(axis=0) for o in origins])


def qlike(rv: np.ndarray, f: np.ndarray) -> np.ndarray:
    x = rv / f
    return x - np.log(x) - 1


def score_forecast(f: np.ndarray, rv: np.ndarray, base_loss: np.ndarray | None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Score a (periods, symbols) variance forecast against realized variance.

    `base_loss` is the baseline's calibrated per-period QLIKE (None when scoring the baseline).
    Returns (scores, calibrated QLIKE per scored period, calibrated forecast; NaN during warm-up).
    """
    f, rv = np.maximum(f, FLOOR), np.maximum(rv, FLOOR)
    ratio = rv / f
    T = len(f)
    scale = np.array([ratio[:t].mean() if t >= CAL_WARMUP else np.nan for t in range(T)])
    fc = f * scale[:, None]
    s = slice(CAL_WARMUP, None)
    loss_raw, loss_cal = qlike(rv[s], f[s]), qlike(rv[s], fc[s])
    per_period = loss_cal.mean(axis=1)

    x, y = np.log(f[s]).ravel(), np.log(rv[s]).ravel()
    slope, intercept = np.polyfit(x, y, 1) if x.std() > 0 else (0.0, float(y.mean()))
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if x.std() > 0 else 0.0

    ts = []
    for j in range(f.shape[1]):
        a, b = np.log(f[s, j]), np.log(rv[s, j])
        if a.std() > 0 and b.std() > 0:
            ts.append(np.corrcoef(a, b)[0, 1])

    out = {
        "qlike": float(loss_raw.mean()),
        "qlike_cal": float(loss_cal.mean()),
        "vol_ratio": float(np.sqrt(ratio[s].mean())),
        "mz_slope": float(slope),
        "mz_r2": r2,
        "rank_ic": _ic_stats(_rowwise_corr(_ranks(f[s]), _ranks(rv[s]))),
        "timing_corr": float(np.mean(ts)) if ts else 0.0,
        "skill": 0.0,
        "dm_t": 0.0,
    }
    if base_loss is not None:
        out["skill"] = float(1 - loss_cal.mean() / base_loss.mean()) if base_loss.mean() > 0 else 0.0
        d = base_loss - per_period  # positive = this forecast lost less than the baseline
        sd = d.std(ddof=1) if len(d) > 1 else 0.0
        out["dm_t"] = float(d.mean() / sd * np.sqrt(len(d))) if sd > 0 else 0.0
    return out, per_period, fc


def run_volatility(
    forecaster: Forecaster,
    frames: dict[str, pd.DataFrame],
    *,
    interval: str,
    models: list[str],
    lookback: int,
    horizon: int,
    periods: int,
    samples: int,
    temperature: float,
    top_p: float,
    cache: ScoreCache | None = None,
) -> dict:
    sc = schedule(frames, interval=interval, lookback=lookback, horizon=horizon, periods=periods)
    if len(sc.origins) <= CAL_WARMUP + 1:
        raise ValueError(f"Need more than {CAL_WARMUP + 1} rebalances: the first {CAL_WARMUP} only calibrate.")
    symbols, origins = sc.symbols, sc.origins
    closes = sc.closes(frames)
    rv = realized_variance(closes, origins, horizon)
    bpy = bars_per_year(pd.DatetimeIndex(sc.common))

    own = {s: frames[s]["close"].to_numpy(dtype=float) for s in symbols}
    forecasts = {"ewma": np.array([[ewma_variance(own[s][: sc.pos[s][o] + 1], horizon) for s in symbols] for o in origins])}
    contexts = sc.contexts(frames, lookback)
    for m in models:
        forecasts[m] = cached_stat(
            forecaster, cache, contexts, symbols, "var", interval=interval, model=m, lookback=lookback,
            horizon=horizon, samples=samples, temperature=temperature, top_p=top_p,
        ).reshape(len(origins), len(symbols))

    # Annualized vol (%-able fraction) for charts: sqrt(variance per bar * bars per year)
    ann = lambda v: np.sqrt(np.maximum(v, 0) * bpy / horizon)  # noqa: E731
    warm = lambda a: [[None if t < CAL_WARMUP else float(x) for x in row] for t, row in enumerate(a)]  # noqa: E731

    base, base_loss, base_cal = score_forecast(forecasts["ewma"], rv, None)
    out = []
    for name, f in forecasts.items():
        r, _, fc = (base, base_loss, base_cal) if name == "ewma" else score_forecast(f, rv, base_loss)
        out.append({
            "name": name,
            "kind": "baseline" if name == "ewma" else "model",
            **r,
            "raw": [[float(x) for x in row] for row in ann(f)],
            "calibrated": warm(ann(fc)),
        })
    # Models in request order, the baseline last
    out = out[1:] + out[:1]
    return {
        "symbols": symbols,
        "periods": len(origins),
        "scored_periods": len(origins) - CAL_WARMUP,
        "calibration_periods": CAL_WARMUP,
        "origin_times": [int(sc.common[o].timestamp()) for o in origins],
        "realized": [[float(x) for x in row] for row in ann(rv)],
        "avg_realized_vol": float(ann(rv)[CAL_WARMUP:].mean()),
        "forecasts": out,
    }
