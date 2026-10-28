"""Walk-forward backtest: forecast from past windows using only data available then,
then score against what actually happened, alongside naive baselines.

Following Gu, Kelly & Xiu, accuracy is also reported as out-of-sample R^2 against a
zero-return forecast, next to a walk-forward ridge regression on standard technical features.
Return prediction R^2 is tiny even for good models; negative means worse than predicting zero.
"""

import numpy as np
import pandas as pd

from app.services import calibration
from app.services.forecaster import Forecaster, summarize


RIDGE_LAGS = (1, 5, 10, 20)
RIDGE_TRAIN_ROWS = 1000
RIDGE_ALPHA = 10.0


def ridge_features(df: pd.DataFrame) -> np.ndarray:
    """Per-bar features known at that bar's close: lagged log returns, volatility, volume z-score."""
    logc = np.log(df["close"].to_numpy())
    r1 = np.diff(logc, prepend=np.nan)
    cols = [logc - np.roll(logc, k) for k in RIDGE_LAGS]
    vol = pd.Series(r1).rolling(20).std().to_numpy()
    logv = np.log1p(df["volume"].to_numpy())
    vz = (logv - pd.Series(logv).rolling(20).mean().to_numpy()) / (pd.Series(logv).rolling(20).std().to_numpy() + 1e-9)
    x = np.column_stack(cols + [vol, vz])
    x[: max(RIDGE_LAGS) + 1] = np.nan
    return x


def ridge_forecast(df: pd.DataFrame, origin: int, horizon: int, x: np.ndarray | None = None) -> float | None:
    """Predict the simple return from bar `origin`'s close to `origin + horizon`'s close.

    Trains only on rows whose target was fully realized by `origin`, so nothing leaks.
    """
    x = ridge_features(df) if x is None else x
    logc = np.log(df["close"].to_numpy())
    rows = np.arange(max(0, origin - horizon - RIDGE_TRAIN_ROWS), origin - horizon + 1)
    rows = rows[~np.isnan(x[rows]).any(axis=1)]
    if len(rows) < 50 or np.isnan(x[origin]).any():
        return None
    X, y = x[rows], logc[rows + horizon] - logc[rows]
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    Z = (X - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + RIDGE_ALPHA * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return float(np.expm1(y.mean() + ((x[origin] - mu) / sd) @ beta))


def r2_oos(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Out-of-sample R^2 against a zero-return forecast (Gu, Kelly & Xiu 2020)."""
    denom = np.sum(actual**2)
    return float(1 - np.sum((actual - predicted) ** 2) / denom) if denom > 0 else 0.0


def run_backtest(
    forecaster: Forecaster,
    df: pd.DataFrame,
    *,
    lookback: int,
    horizon: int,
    windows: int,
    step: int,
    model: str,
    samples: int,
    temperature: float,
    top_p: float,
) -> dict:
    ends = [len(df) - horizon - k * step for k in range(windows)]
    ends = sorted(e for e in ends if e - lookback >= 0)
    if not ends:
        raise ValueError(
            f"Not enough history: need at least {lookback + horizon} bars, have {len(df)}. "
            "Reduce lookback/horizon or use a longer interval."
        )

    contexts = [df.iloc[e - lookback : e] for e in ends]
    actuals = [df.iloc[e : e + horizon] for e in ends]
    paths = forecaster.sample_paths(
        contexts, [a.index for a in actuals], model, samples, temperature, top_p
    )

    features = ridge_features(df)
    rows = []
    for e, ctx, actual, p in zip(ends, contexts, actuals, paths):
        last = float(ctx["close"].iloc[-1])
        s = summarize(p, last)
        actual_close = actual["close"].to_numpy()
        actual_final = float(actual_close[-1])
        rows.append(
            {
                "time": int(actual.index[-1].timestamp()),
                "origin_time": int(ctx.index[-1].timestamp()),
                "last_close": last,
                "pred_final": float(s["median"][-1]),
                "p10_final": float(s["p10"][-1]),
                "p90_final": float(s["p90"][-1]),
                "actual_final": actual_final,
                "p_up": s["p_up"],
                "hit": bool(np.sign(s["median"][-1] - last) == np.sign(actual_final - last)),
                "covered": bool(s["p10"][-1] <= actual_final <= s["p90"][-1]),
                "mae_pct": float(np.mean(np.abs(s["median"] - actual_close)) / last),
                "naive_mae_pct": float(np.mean(np.abs(last - actual_close)) / last),
                "ridge_return": ridge_forecast(df, e - 1, horizon, features),
                # Per-step band and outcome as returns from the last close, for conformal calibration
                "lo_path": (s["p10"] / last - 1).tolist(),
                "hi_path": (s["p90"] / last - 1).tolist(),
                "actual_path": (actual_close / last - 1).tolist(),
            }
        )

    n = len(rows)
    up_rate = sum(r["actual_final"] > r["last_close"] for r in rows) / n
    metrics = {
        "windows": n,
        "hit_rate": sum(r["hit"] for r in rows) / n,
        # Best constant-direction guess in hindsight; a model should beat this to be interesting
        "baseline_hit_rate": max(up_rate, 1 - up_rate),
        "coverage_p10_p90": sum(r["covered"] for r in rows) / n,
        "mae_pct": float(np.mean([r["mae_pct"] for r in rows])),
        "naive_mae_pct": float(np.mean([r["naive_mae_pct"] for r in rows])),
    }
    actual_ret = np.array([r["actual_final"] / r["last_close"] - 1 for r in rows])
    kronos_ret = np.array([r["pred_final"] / r["last_close"] - 1 for r in rows])
    metrics["r2_oos"] = r2_oos(actual_ret, kronos_ret)
    scored = [(a, r["ridge_return"]) for a, r in zip(actual_ret, rows) if r["ridge_return"] is not None]
    if scored:
        a, pr = np.array(scored).T
        metrics["ridge_r2_oos"] = r2_oos(a, pr)
        metrics["ridge_hit_rate"] = float(np.mean(np.sign(pr) == np.sign(a)))
    else:
        metrics["ridge_r2_oos"] = metrics["ridge_hit_rate"] = None

    first = ends[0] - lookback
    span = df.iloc[max(first, ends[0] - 50) : ends[-1] + horizon]
    actual_series = [{"time": int(t.timestamp()), "value": float(v)} for t, v in span["close"].items()]
    return {"metrics": metrics, "windows": rows, "actual": actual_series, "calibration": calibration.calibrate(rows)}
