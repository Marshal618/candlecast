"""Walk-forward backtest: forecast from past windows using only data available then,
then score against what actually happened, alongside naive baselines."""

import numpy as np
import pandas as pd

from app.services.forecaster import Forecaster, summarize


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

    rows = []
    for ctx, actual, p in zip(contexts, actuals, paths):
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

    first = ends[0] - lookback
    span = df.iloc[max(first, ends[0] - 50) : ends[-1] + horizon]
    actual_series = [{"time": int(t.timestamp()), "value": float(v)} for t, v in span["close"].items()]
    return {"metrics": metrics, "windows": rows, "actual": actual_series}
