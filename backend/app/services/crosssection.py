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
   Optionally, non-price fundamentals rebuilt point-in-time (see services/fundamentals.py).
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
   the model's (or fundamental's) skill beyond momentum and reversal.

Symbols are aligned by calendar date (daily) or UTC hour (hourly), and rebalances use dates every
symbol traded. Models forecast `horizon` of each symbol's own bars, so for crypto in a stock
universe the forecast covers slightly fewer calendar days than the return it is scored against.
"""

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from app.services.forecaster import FEATURES, Forecaster
from app.services.fundamentals import FUNDAMENTALS, Fundamentals, fill_missing, fundamental_scores
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


def factor_scores(histories: list[np.ndarray], horizon: int) -> dict[str, np.ndarray]:
    """Momentum and reversal for each symbol from its own close history (last element = now).

    A history too short for a factor scores 0 (neutral) on it.
    """
    mom_need = MOMENTUM_SKIP + MOMENTUM_BARS + 1
    return {
        "momentum": np.array([h[-1 - MOMENTUM_SKIP] / h[-mom_need] - 1 if len(h) >= mom_need else 0.0 for h in histories]),
        "reversal": np.array([-(h[-1] / h[-1 - horizon] - 1) if len(h) > horizon else 0.0 for h in histories]),
    }


def model_stats(
    forecaster: Forecaster, contexts: list[pd.DataFrame], interval: str, model: str, horizon: int,
    samples: int, temperature: float, top_p: float,
) -> dict[str, np.ndarray]:
    """Two numbers per context from one set of sampled paths.

    - ret: median sampled return over the horizon (the Forecast tab's expected return)
    - var: variance across samples of the log return over the horizon, the model's volatility
           forecast (scored in services/volatility.py)
    """
    futures = [future_timestamps(c.index, interval, horizon) for c in contexts]
    paths = forecaster.sample_paths(contexts, futures, model, samples, temperature, top_p)
    last = np.array([c["close"].iloc[-1] for c in contexts])
    final = paths[:, :, -1, FEATURES.index("close")]
    log_r = np.log(np.maximum(final, 1e-12) / last[:, None])
    return {
        "ret": np.median(final, axis=1) / last - 1,
        "var": log_r.var(axis=1, ddof=1) if samples > 1 else np.zeros(len(contexts)),
    }


def model_scores(
    forecaster: Forecaster, contexts: list[pd.DataFrame], interval: str, model: str, horizon: int,
    samples: int, temperature: float, top_p: float,
) -> np.ndarray:
    """Median sampled return over the horizon for each context (the Forecast tab's expected return)."""
    return model_stats(forecaster, contexts, interval, model, horizon, samples, temperature, top_p)["ret"]


class ScoreCache:
    """Model scores per (symbol, origin bar, settings), so re-runs and longer studies only pay for new forecasts.

    Kronos is seeded per call, so a cached score can differ slightly from what a fresh batch would
    sample; the cache keeps the first one, which keeps every later run consistent with it.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)")

    def get(self, keys: list[str]) -> dict[str, float]:
        out = {}
        with self._lock, sqlite3.connect(self.path) as conn:
            for k in range(0, len(keys), 500):
                chunk = keys[k : k + 500]
                q = f"SELECT key, score FROM scores WHERE key IN ({','.join('?' * len(chunk))})"
                out.update(conn.execute(q, chunk).fetchall())
        return out

    def put(self, items: dict[str, float]) -> None:
        with self._lock, sqlite3.connect(self.path) as conn:
            conn.executemany("INSERT OR IGNORE INTO scores (key, score) VALUES (?, ?)", items.items())


def cached_stat(
    forecaster: Forecaster, cache: "ScoreCache | None", contexts: list[pd.DataFrame], symbols: list[str],
    stat: str, *, interval: str, model: str, lookback: int, horizon: int, samples: int, temperature: float, top_p: float,
) -> np.ndarray:
    """One model_stats stat per context (contexts cycle through `symbols`), forecasting only cache misses.

    Both stats of every fresh forecast are stored, so a research run warms the volatility test and
    vice versa. Return scores keep the original key format, so older caches stay valid.
    """
    def key(c, s, st):
        base = f"{s}|{interval}|{int(c.index[-1].timestamp())}|{model}|{lookback}|{horizon}|{samples}|{temperature}|{top_p}"
        return base if st == "ret" else f"{base}|{st}"

    owners = symbols * (len(contexts) // len(symbols))
    want = [key(c, s, stat) for c, s in zip(contexts, owners)]
    hit = cache.get(want) if cache else {}
    todo = [k for k, kk in enumerate(want) if kk not in hit]
    if todo:
        fresh = model_stats(forecaster, [contexts[k] for k in todo], interval, model, horizon, samples, temperature, top_p)
        new = {key(contexts[k], owners[k], st): float(v) for st, vals in fresh.items() for k, v in zip(todo, vals)}
        if cache:
            cache.put(new)
        hit = {**hit, **{k: new[k] for k in (want[j] for j in todo)}}
    return np.array([hit[k] for k in want])


@dataclass
class Schedule:
    """Walk-forward rebalance dates on the calendar every symbol shares."""

    symbols: list[str]
    common: list[pd.Timestamp]  # shared-clock bar keys (see sim.align_keys)
    pos: dict[str, np.ndarray]  # per symbol: row in its own frame of each common bar
    origins: list[int]  # rebalance rows in `common`, oldest first

    def contexts(self, frames: dict[str, pd.DataFrame], lookback: int) -> list[pd.DataFrame]:
        """The `lookback` bars up to each origin, origin-major then symbol."""
        return [frames[s].iloc[self.pos[s][o] + 1 - lookback : self.pos[s][o] + 1] for o in self.origins for s in self.symbols]

    def closes(self, frames: dict[str, pd.DataFrame]) -> np.ndarray:
        return np.column_stack([frames[s]["close"].to_numpy(dtype=float)[self.pos[s]] for s in self.symbols])


def schedule(
    frames: dict[str, pd.DataFrame], *, interval: str, lookback: int, horizon: int, periods: int, min_history: int = 0
) -> Schedule:
    symbols = list(frames)
    keys = {s: align_keys(f.index, interval) for s, f in frames.items()}
    common = sorted(set.intersection(*(set(k) for k in keys.values())))
    need = max(lookback, MOMENTUM_BARS + MOMENTUM_SKIP + 1, horizon + 1, min_history)

    # Position of each common date's bar in each symbol's own frame (last bar with that key)
    pos = {s: np.searchsorted(keys[s], common, side="right") - 1 for s in symbols}
    last_origin = len(common) - 1 - horizon
    origins = [
        o for o in range(last_origin, -1, -horizon)
        if o + 1 >= min_history and all(pos[s][o] + 1 >= need for s in symbols)
    ][:periods][::-1]
    if len(origins) < 4:
        raise ValueError(
            f"Only {len(origins)} rebalance dates have {need} bars of history for every symbol; need at least 4. "
            "Drop the newest listings, shorten the lookback, or use fewer rebalances."
        )
    return Schedule(symbols, common, pos, origins)


@dataclass
class Panel:
    """Walk-forward cross-section: scores and realized forward returns per (rebalance, symbol)."""

    symbols: list[str]
    interval: str
    horizon: int
    origin_times: list[int]
    fwd: np.ndarray  # (periods, symbols) return from each origin's close to `horizon` common bars later
    signals: dict[str, np.ndarray]  # name -> (periods, symbols): price factors, fundamentals, then models
    closes: np.ndarray  # (common bars, symbols) closes on the shared calendar, for risk models
    origins: list[int]  # row of each rebalance in `closes`
    bars_per_year: float
    coverage: dict[str, float] | None = None  # share of cells each fundamental had real data for

    @property
    def periods_per_year(self) -> float:
        return self.bars_per_year / self.horizon


def build_panel(
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
    min_history: int = 0,
    cache: ScoreCache | None = None,
    fundamentals: dict[str, Fundamentals] | None = None,
    use_fundamentals: list[str] = (),
) -> Panel:
    sc = schedule(frames, interval=interval, lookback=lookback, horizon=horizon, periods=periods, min_history=min_history)
    symbols, origins, pos = sc.symbols, sc.origins, sc.pos
    own = {s: frames[s]["close"].to_numpy(dtype=float) for s in symbols}
    closes = sc.closes(frames)
    fwd = closes[np.array(origins) + horizon] / closes[origins] - 1
    per = [factor_scores([own[s][: pos[s][o] + 1] for s in symbols], horizon) for o in origins]
    signals: dict[str, np.ndarray] = {f: np.array([p[f] for p in per]) for f in FACTORS}

    coverage = None
    if use_fundamentals:
        funds = [(fundamentals or {}).get(s) or Fundamentals.empty() for s in symbols]
        rows = [fundamental_scores(funds, int(sc.common[o].timestamp()), interval, closes[o]) for o in origins]
        coverage = {}
        for f in use_fundamentals:
            raw = np.array([r[f] for r in rows])
            coverage[f] = float(np.isfinite(raw).mean())
            signals[f] = fill_missing(raw)

    contexts = sc.contexts(frames, lookback)
    for m in models:
        signals[m] = cached_stat(
            forecaster, cache, contexts, symbols, "ret", interval=interval, model=m, lookback=lookback,
            horizon=horizon, samples=samples, temperature=temperature, top_p=top_p,
        ).reshape(len(origins), len(symbols))

    return Panel(
        symbols=symbols, interval=interval, horizon=horizon,
        origin_times=[int(sc.common[o].timestamp()) for o in origins], fwd=fwd, signals=signals,
        closes=closes, origins=origins, bars_per_year=bars_per_year(pd.DatetimeIndex(sc.common)), coverage=coverage,
    )


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
    cache: ScoreCache | None = None,
    fundamentals: dict[str, Fundamentals] | None = None,
    use_fundamentals: list[str] = (),
) -> dict:
    if buckets > len(frames):
        raise ValueError(f"{buckets} buckets need at least {buckets} symbols; got {len(frames)}.")
    p = build_panel(
        forecaster, frames, interval=interval, models=models, lookback=lookback, horizon=horizon,
        periods=periods, samples=samples, temperature=temperature, top_p=top_p, cache=cache,
        fundamentals=fundamentals, use_fundamentals=use_fundamentals,
    )
    signals, fwd = p.signals, p.fwd
    universe = np.cumprod(1 + fwd.mean(axis=1))
    out = []
    for name, score in signals.items():
        r = evaluate(score, fwd, buckets, cost_bps, p.periods_per_year)
        r["name"] = name
        r["kind"] = "factor" if name in FACTORS else "fundamental" if name in FUNDAMENTALS else "model"
        r["corr"] = {f: _mean_rank_corr(score, signals[f]) for f in FACTORS if f != name}
        if name in FUNDAMENTALS:
            r["coverage"] = p.coverage[name]
        if name not in FACTORS:
            # What's left of the ranking once momentum and reversal are accounted for
            resid = residual_scores(score, [signals[f] for f in FACTORS])
            r["rank_ic_residual"] = _ic_stats(_rowwise_corr(_ranks(resid), _ranks(fwd)))
        out.append(r)
    # Models first, in request order, then fundamentals, then the price factors
    order = [*models, *FUNDAMENTALS, *FACTORS]
    out.sort(key=lambda r: order.index(r["name"]))
    return {
        "symbols": p.symbols,
        "periods": len(p.origins),
        "origin_times": p.origin_times,
        "periods_per_year": p.periods_per_year,
        "universe": {"equity": [float(x) for x in universe], "mean": float(fwd.mean())},
        "dispersion": float(fwd.std(axis=1).mean()),
        "signals": out,
    }
