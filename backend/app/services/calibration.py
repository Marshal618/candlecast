"""Conformal calibration of Kronos's p10-p90 bands.

Kronos's sampled bands tend to be too narrow: realized prices land outside p10-p90 far more
than 20% of the time. Conformalized quantile regression (CQR; Romano, Patterson & Candès 2019)
fixes that without touching the model:

    score_i = max(lo_i - y_i, y_i - hi_i)          # how far outside the band (negative = inside)
    q       = the ceil((n+1)(1-alpha))/n quantile of the scores
    band    = [lo - q, hi + q]                     # widened (or narrowed if q < 0)

Scores are computed per forecast step in return space (relative to the last close), from
walk-forward backtest windows, so a calibration transfers across price levels.

With exchangeable scores this guarantees >= 1-alpha coverage. Markets aren't exchangeable, so
`calibrate` also reports an honest *sequential* check: each window is calibrated using only
the windows before it, the way a live forecast would be.
"""

import math

import numpy as np

TARGET = 0.8  # nominal coverage of the p10-p90 band
MIN_HISTORY = 5  # windows needed before the sequential check starts scoring


def scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(lo - y, y - hi)


def conformal_quantile(s: np.ndarray, coverage: float = TARGET) -> float:
    """The k-th smallest score, k = ceil((n+1) * coverage): the finite-sample conformal quantile.

    When k > n there aren't enough windows for the guarantee; fall back to the largest score.
    """
    k = min(math.ceil((len(s) + 1) * coverage), len(s))
    return float(np.sort(s)[k - 1])


def calibrate(windows: list[dict], coverage: float = TARGET) -> dict | None:
    """`windows` in time order, each with lo_path / hi_path / actual_path (returns vs last close)."""
    if len(windows) < MIN_HISTORY + 1:
        return None
    lo = np.array([w["lo_path"] for w in windows])
    hi = np.array([w["hi_path"] for w in windows])
    y = np.array([w["actual_path"] for w in windows])
    s = scores(lo, hi, y)  # (windows, horizon)
    q = [conformal_quantile(s[:, k], coverage) for k in range(s.shape[1])]

    # Sequential, out-of-sample check on the final step
    raw, cal = [], []
    for i in range(MIN_HISTORY, len(windows)):
        qi = conformal_quantile(s[:i, -1], coverage)
        raw.append(lo[i, -1] <= y[i, -1] <= hi[i, -1])
        cal.append(lo[i, -1] - qi <= y[i, -1] <= hi[i, -1] + qi)
    return {
        "target": coverage,
        "windows": len(windows),
        "q": q,
        "oos_windows": len(raw),
        "raw_coverage": float(np.mean(raw)),
        "calibrated_coverage": float(np.mean(cal)),
        # Average widening at the final step, as a fraction of price
        "widening": float(2 * q[-1]),
    }


def apply(last_close: float, lo: np.ndarray, hi: np.ndarray, q: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated price band for a forecast. Steps beyond the calibrated horizon reuse the last q."""
    qk = np.array([q[min(k, len(q) - 1)] for k in range(len(lo))])
    lo_r, hi_r = lo / last_close - 1 - qk, hi / last_close - 1 + qk
    # A heavily negative q could cross the bands; never let lo exceed hi
    mid = (lo_r + hi_r) / 2
    lo_r, hi_r = np.minimum(lo_r, mid), np.maximum(hi_r, mid)
    return last_close * (1 + lo_r), last_close * (1 + hi_r)
