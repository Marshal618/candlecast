"""Cross-sectional research: does a model rank a universe of assets correctly?

Time-series accuracy (the Backtest tab) asks "will this asset go up?". Quant shops mostly ask a
different, easier-to-monetize question: "which of these assets will do *better than the others*?"
A signal can be useless for timing yet still rank well, and a ranking can be traded market-neutral.

Walk-forward, at each rebalance date (every `horizon` bars, non-overlapping):
1. Every model forecasts every symbol from data up to that date's close. Its score is the median
   sampled return over the horizon, the same number the Forecast tab shows as expected return.
2. Two classic price factors are scored alongside, with no model calls:
   - momentum: the 60-bar return skipping the latest 5 bars (Jegadeesh & Titman 1993)
   - reversal: minus the return over the last `horizon` bars (Lehmann 1990, Jegadeesh 1990)
3. Each score is compared with the realized forward return across symbols:
   - IC: Pearson correlation of score and return; RankIC: Spearman (the Kronos paper's metric).
     ICIR = mean / std; t = ICIR * sqrt(periods). |t| > 2 is the usual bar for "probably real".
   - Bucket portfolios: sort symbols into `buckets` by score, equal-weight each. A real signal
     shows returns rising from the bottom bucket to the top one.
   - Long-short: long the top bucket, short the bottom, paying `cost_bps` per unit of turnover.
4. The average rank correlation between each model and the factors shows whether a model is
   finding something new or re-deriving a known effect (Kronos's per-window normalization
   predicts mean reversion, so a high correlation with reversal is expected). Residual RankIC
   regresses the factors' ranks out of the model's ranks each period and scores what remains:
   the model's skill beyond momentum and reversal.

Symbols are aligned by calendar date (daily) or UTC hour (hourly), and rebalances use dates every
symbol traded. Models forecast `horizon` of each symbol's own bars, so for crypto in a stock
universe the forecast covers slightly fewer calendar days than the return it is scored against.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from app.services.forecaster import FEATURES, Forecaster
from app.services.market_data import future_timestamps
from app.services.sim import align_keys
from app.services.strategy import _max_drawdown, _sharpe, bars_per_year

MOMENTUM_BARS = 60
MOMENTUM_SKIP = 5
FACTORS = ("momentum", "reversal")


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of a with the same row of b; NaN where either is constant."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a * b).sum(axis=1) / denom, np.nan)


def _ranks(x: np.ndarray) -> np.ndarray:
    return np.vstack([rankdata(row) for row in x])


def _ic_stats(ic: np.ndarray) -> dict:
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return {"mean": 0.0, "std": 0.0, "icir": 0.0, "t_stat": 0.0, "positive": 0.0}
    sd = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    icir = float(ic.mean() / sd) if sd > 0 else 0.0
    return {
        "mean": float(ic.mean()),
        "std": sd,
        "icir": icir,
        "t_stat": icir * float(np.sqrt(len(ic))),
        "positive": float(np.mean(ic > 0)),
    }


def bucket_weights(score: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Bucket index per cell (0 = lowest score) and long-short weights (top +1, bottom -1)."""
    n = score.shape[1]
    order = np.argsort(np.argsort(score, axis=1, kind="stable"), axis=1)
    b = order * buckets // n
    top, bottom = b == buckets - 1, b == 0
    w = top / top.sum(axis=1, keepdims=True) - bottom / bottom.sum(axis=1, keepdims=True)
    return b, w


def evaluate(score: np.ndarray, fwd: np.ndarray, buckets: int, cost_bps: float, periods_per_year: float) -> dict:
    """Score a (periods, symbols) signal against (periods, symbols) forward returns."""
    rank_ic = _rowwise_corr(_ranks(score), _ranks(fwd))
    b, w = bucket_weights(score, buckets)
    bucket_ret = np.array([[fwd[t, b[t] == k].mean() for k in range(buckets)] for t in range(len(fwd))])
    gross = (w * fwd).sum(axis=1)
    prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    turnover = np.abs(w - prev).sum(axis=1)
    turnover[-1] += np.abs(w[-1]).sum()  # close out at the end
    net = gross - turnover * cost_bps / 1e4
    equity = np.cumprod(1 + net)
    return {
        "ic": _ic_stats(_rowwise_corr(score, fwd)),
        "rank_ic": _ic_stats(rank_ic),
        # null where a period's scores were all equal (no ranking to judge)
        "rank_ic_series": [float(x) if np.isfinite(x) else None for x in rank_ic],
        "bucket_returns": [float(x) for x in bucket_ret.mean(axis=0)],
        "long_short": {
            "gross_mean": float(gross.mean()),
            "net_mean": float(net.mean()),
            "ann_return": float(equity[-1] ** (periods_per_year / len(net)) - 1),
            "sharpe": _sharpe(net, periods_per_year),
            "max_drawdown": _max_drawdown(np.concatenate([[1.0], equity])),
            "hit_rate": float(np.mean(net > 0)),
            # Share of the book traded per rebalance (1 = every position replaced), excluding entry/exit
            "turnover": float((turnover[1:-1] if len(turnover) > 2 else turnover).mean() / 4),
        },
        "equity": [float(x) for x in equity],
    }


def residual_scores(score: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    """Per period, the part of a score's ranks not explained by the factors' ranks (OLS residual)."""
    out = np.empty_like(score, dtype=float)
    for t in range(len(score)):
        X = np.column_stack([np.ones(score.shape[1])] + [rankdata(f[t]) for f in factors])
        y = rankdata(score[t])
        out[t] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    # A score that is just the factors leaves float noise; don't let its ranks pass for signal
    out[np.abs(out) < 1e-9] = 0.0
    return out


def _mean_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    c = _rowwise_corr(_ranks(a), _ranks(b))
    c = c[np.isfinite(c)]
    return float(c.mean()) if len(c) else 0.0


def run_crosssection(
    forecaster: Forecaster,
    frames: dict[str, pd.DataFrame],
    *,
    interval: str,
    models: list[str],
    lookback: int,
    horizon: int,
    periods: int,
    buckets: int,
    samples: int,
    temperature: float,
    top_p: float,
    cost_bps: float,
) -> dict:
    symbols = list(frames)
    if buckets > len(symbols):
        raise ValueError(f"{buckets} buckets need at least {buckets} symbols; got {len(symbols)}.")
    keys = {s: align_keys(f.index, interval) for s, f in frames.items()}
    common = sorted(set.intersection(*(set(k) for k in keys.values())))
    need = max(lookback, MOMENTUM_BARS + MOMENTUM_SKIP + 1, horizon + 1)

    # Position of each common date's bar in each symbol's own frame (last bar with that key)
    pos = {s: np.searchsorted(keys[s], common, side="right") - 1 for s in symbols}
    last_origin = len(common) - 1 - horizon
    origins = [o for o in range(last_origin, -1, -horizon) if all(pos[s][o] + 1 >= need for s in symbols)][:periods][::-1]
    if len(origins) < 4:
        raise ValueError(
            f"Only {len(origins)} rebalance dates have {need} bars of history for every symbol; need at least 4. "
            "Drop the newest listings or shorten the lookback."
        )

    closes = {s: frames[s]["close"].to_numpy(dtype=float) for s in symbols}

    def at(s: str, o: int, back: int = 0) -> float:
        return closes[s][pos[s][o] - back]

    fwd = np.array([[at(s, o + horizon) / at(s, o) - 1 for s in symbols] for o in origins])
    signals: dict[str, np.ndarray] = {
        "momentum": np.array(
            [[at(s, o, MOMENTUM_SKIP) / at(s, o, MOMENTUM_SKIP + MOMENTUM_BARS) - 1 for s in symbols] for o in origins]
        ),
        "reversal": np.array([[-(at(s, o) / at(s, o, horizon) - 1) for s in symbols] for o in origins]),
    }

    contexts = [frames[s].iloc[pos[s][o] + 1 - lookback : pos[s][o] + 1] for o in origins for s in symbols]
    futures = [future_timestamps(c.index, interval, horizon) for c in contexts]
    last = np.array([c["close"].iloc[-1] for c in contexts])
    for m in models:
        paths = forecaster.sample_paths(contexts, futures, m, samples, temperature, top_p)
        final = paths[:, :, -1, FEATURES.index("close")]
        signals[m] = (np.median(final, axis=1) / last - 1).reshape(len(origins), len(symbols))

    origin_times = [int(common[o].timestamp()) for o in origins]
    ppy = bars_per_year(pd.DatetimeIndex(common)) / horizon
    universe = np.cumprod(1 + fwd.mean(axis=1))
    out = []
    for name, score in signals.items():
        r = evaluate(score, fwd, buckets, cost_bps, ppy)
        r["name"] = name
        r["kind"] = "factor" if name in FACTORS else "model"
        r["corr"] = {f: _mean_rank_corr(score, signals[f]) for f in FACTORS if f != name}
        if name not in FACTORS:
            # What's left of the ranking once momentum and reversal are accounted for
            resid = residual_scores(score, [signals[f] for f in FACTORS])
            r["rank_ic_residual"] = _ic_stats(_rowwise_corr(_ranks(resid), _ranks(fwd)))
        out.append(r)
    # Models first, in request order, then the factors
    out.sort(key=lambda r: (r["kind"] == "factor", models.index(r["name"]) if r["name"] in models else FACTORS.index(r["name"])))
    return {
        "symbols": symbols,
        "periods": len(origins),
        "origin_times": origin_times,
        "periods_per_year": ppy,
        "universe": {"equity": [float(x) for x in universe], "mean": float(fwd.mean())},
        "dispersion": float(fwd.std(axis=1).mean()),
        "signals": out,
    }
