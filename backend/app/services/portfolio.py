"""Portfolio lab: turn cross-sectional signals into a portfolio you could hold, and test it honestly.

At each rebalance (walk-forward, see crosssection.build_panel):
1. Blend. Each signal is turned into cross-sectional z-scored ranks, so models and factors with
   very different scales combine fairly. Weights are fixed ("manual") or proportional to each
   signal's trailing positive RankIC over the last `ic_window` *realized* rebalances ("ic").
2. Risk. Covariance of per-bar log returns over the trailing `risk_window` bars, shrunk toward a
   scaled identity with the Ledoit & Wolf (2004) optimal intensity. Sample covariance from a few
   hundred bars is noisy enough that mean-variance weights built on it mostly fit the noise.
3. Construct.
   - equal:         the top `select` share of names long (and bottom share short), equal weight
   - inverse_vol:   same names, weighted by 1 / volatility within each leg
   - mean_variance: aim portfolio Σ⁻¹α (dollar-neutral via the constrained solution for long-short;
                    negatives clipped for long-only)
4. Size. Scale to an ex-ante annual volatility target, capped at `max_gross` (1 for long-only).
5. Trade. Move only `trade_rate` of the way from the held weights to the target (Gârleanu &
   Pedersen 2013): slower trading gives up some signal for much lower costs.

Returns per period are w · forward return minus turnover × cost (weights are not drifted within
a period). Every run is logged as a trial for its study (same universe, interval and horizon), and
the result reports the Deflated Sharpe Ratio (Bailey & López de Prado 2014): the probability that
the true Sharpe is above what the best of that many tries would show by luck alone.
"""

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew

from app.services.crosssection import Panel, _rowwise_corr
from app.services.strategy import _max_drawdown

EULER = 0.5772156649
MIN_IC_HISTORY = 3  # rebalances of realized IC before "ic" weighting replaces the manual weights


@dataclass(frozen=True)
class Construction:
    method: str = "inverse_vol"  # equal | inverse_vol | mean_variance
    side: str = "long_short"  # long_short | long_only
    select: float = 0.3  # share of the universe in each leg (equal / inverse_vol)
    risk_window: int = 120
    vol_target: float | None = 0.10  # annualized; None = fixed gross (2 long-short, 1 long-only)
    max_gross: float = 2.0
    trade_rate: float = 1.0


def zranks(x: np.ndarray) -> np.ndarray:
    """Cross-sectional ranks scaled to mean 0, std 1 (all zeros if every score ties)."""
    r = rankdata(x)
    sd = r.std()
    return (r - r.mean()) / sd if sd > 0 else np.zeros_like(r)


def ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled identity."""
    x = returns - returns.mean(axis=0)
    t, n = x.shape
    s = x.T @ x / t
    mu = np.trace(s) / n
    target = mu * np.eye(n)
    d2 = np.sum((s - target) ** 2)
    b2 = sum(np.sum((np.outer(r, r) - s) ** 2) for r in x) / t**2
    shrink = min(b2, d2) / d2 if d2 > 0 else 1.0
    return shrink * target + (1 - shrink) * s


def _legs(alpha: np.ndarray, select: float, side: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(alpha)
    k = max(1, int(round(select * n)))
    order = np.argsort(alpha, kind="stable")
    long, short = np.zeros(n, bool), np.zeros(n, bool)
    long[order[-k:]] = True
    if side == "long_short":
        short[order[: min(k, n - k)]] = True
    return long, short


def target_weights(alpha: np.ndarray, cov: np.ndarray, c: Construction, bars_per_year: float) -> np.ndarray:
    """Target weights for one rebalance from blended alpha and a covariance of per-bar returns."""
    n = len(alpha)
    vol = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    if c.method == "mean_variance":
        inv = np.linalg.pinv(cov)
        if c.side == "long_short":
            # max α'w − ½w'Σw subject to 1'w = 0
            ones = np.ones(n)
            lam = (ones @ inv @ alpha) / (ones @ inv @ ones)
            w = inv @ (alpha - lam)
        else:
            w = np.maximum(inv @ alpha, 0)
            if w.sum() <= 0:  # no name worth holding on this view: fall back to the top picks
                w = _legs(alpha, c.select, "long_only")[0] / vol
    else:
        long, short = _legs(alpha, c.select, c.side)
        scale = 1 / vol if c.method == "inverse_vol" else np.ones(n)
        w = long * scale / (long * scale).sum()
        if short.any():
            w = w - short * scale / (short * scale).sum()

    max_gross = min(c.max_gross, 1.0) if c.side == "long_only" else c.max_gross
    gross = np.abs(w).sum()
    if gross <= 0:
        return np.zeros(n)
    if c.vol_target:
        ex_ante = np.sqrt(max(w @ cov @ w, 1e-18) * bars_per_year)
        w = w * c.vol_target / ex_ante
    else:
        w = w * (2.0 if c.side == "long_short" else 1.0) / gross
    gross = np.abs(w).sum()
    return w * min(1.0, max_gross / gross) if gross > 0 else w


def blend_weights(
    ic_history: np.ndarray, manual: np.ndarray, weighting: str, ic_window: int
) -> np.ndarray:
    """Weights over signals given each signal's realized RankIC so far, shape (past periods, signals)."""
    if weighting == "ic" and len(ic_history) >= MIN_IC_HISTORY:
        ic = np.nan_to_num(ic_history[-ic_window:]).mean(axis=0)
        w = np.maximum(ic, 0)
        if w.sum() > 0:
            return w / w.sum()
    return manual / np.abs(manual).sum()


def run_portfolio(
    panel: Panel,
    blend: dict[str, float],
    *,
    weighting: str = "manual",
    ic_window: int = 12,
    construction: Construction = Construction(),
    cost_bps: float = 5.0,
) -> dict:
    names = list(blend)
    manual = np.array([blend[k] for k in names], dtype=float)
    if np.abs(manual).sum() == 0:
        raise ValueError("Give at least one signal a non-zero weight.")
    scores = np.stack([panel.signals[k] for k in names])  # (signals, periods, symbols)
    T, N = panel.fwd.shape
    # Realized RankIC of each signal per period; period t's IC is known by rebalance t + 1
    ic = np.stack([_rowwise_corr(np.vstack([rankdata(r) for r in s]), np.vstack([rankdata(r) for r in panel.fwd])) for s in scores], axis=1)
    log_r = np.diff(np.log(panel.closes), axis=0)  # row j = return into common bar j + 1

    held = np.zeros(N)
    W, BW, net, gross_ret, turnover, ex_ante = [], [], [], [], [], []
    for t in range(T):
        bw = blend_weights(ic[:t], manual, weighting, ic_window)
        alpha = sum(b * zranks(scores[k, t]) for k, b in enumerate(bw))
        o = panel.origins[t]
        window = log_r[max(0, o - construction.risk_window) : o]
        cov = ledoit_wolf(window) if len(window) >= max(10, N // 2) else np.diag(np.full(N, window.var() if len(window) else 1e-4))
        target = target_weights(alpha, cov, construction, panel.bars_per_year)
        w = held + construction.trade_rate * (target - held)
        tc = np.abs(w - held).sum()
        g = float(w @ panel.fwd[t])
        W.append(w)
        BW.append(bw)
        gross_ret.append(g)
        turnover.append(tc)
        net.append(g - tc * cost_bps / 1e4)
        ex_ante.append(float(np.sqrt(max(w @ cov @ w, 0) * panel.bars_per_year)))
        held = w
    net = np.array(net)
    net[-1] -= np.abs(held).sum() * cost_bps / 1e4  # close out at the end
    W = np.array(W)
    equity = np.cumprod(1 + net)
    ppy = panel.periods_per_year
    bench = np.cumprod(1 + panel.fwd.mean(axis=1))
    sd = net.std(ddof=1)
    sr = float(net.mean() / sd) if sd > 0 else 0.0  # per period

    return {
        "symbols": panel.symbols,
        "origin_times": panel.origin_times,
        "periods": T,
        "periods_per_year": ppy,
        "signals": names,
        "equity": equity.tolist(),
        "benchmark": bench.tolist(),
        "weights": W.tolist(),
        "blend": np.array(BW).tolist(),
        "final_blend": dict(zip(names, map(float, BW[-1]))),
        "metrics": {
            "ann_return": float(equity[-1] ** (ppy / T) - 1),
            "ann_vol": float(sd * np.sqrt(ppy)),
            "sharpe": sr * float(np.sqrt(ppy)),
            "sharpe_period": sr,
            "max_drawdown": _max_drawdown(np.concatenate([[1.0], equity])),
            "hit_rate": float(np.mean(net > 0)),
            "turnover": float(np.mean(turnover[1:]) / 2) if T > 1 else float(turnover[0] / 2),
            "avg_gross": float(np.abs(W).sum(axis=1).mean()),
            "avg_net": float(W.sum(axis=1).mean()),
            "ex_ante_vol": float(np.mean(ex_ante)),
            "cost_drag": float(np.mean(gross_ret) - net.mean()) * ppy,
            "bench_ann_return": float(bench[-1] ** (ppy / T) - 1),
            "skew": float(skew(net)) if T > 2 else 0.0,
            "kurtosis": float(kurtosis(net, fisher=False)) if T > 3 else 3.0,
        },
    }


def probabilistic_sharpe(sr: float, sr0: float, n: int, g3: float, g4: float) -> float:
    """P(true per-period Sharpe > sr0) given an observed sr over n periods (Bailey & López de Prado 2012)."""
    if n < 2:
        return 0.5
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr**2
    return float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / np.sqrt(max(denom, 1e-12))))


def expected_max_sharpe(trial_srs: list[float]) -> float:
    """Sharpe the best of N unskilled trials would reach by luck, given the trials' spread."""
    n = len(trial_srs)
    if n < 2:
        return 0.0
    sd = float(np.std(trial_srs, ddof=1))
    return sd * ((1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * np.e)))


TRIALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    study TEXT NOT NULL,
    config TEXT NOT NULL,
    sharpe_period REAL NOT NULL,
    periods INTEGER NOT NULL,
    metrics TEXT NOT NULL,
    final_blend TEXT NOT NULL,
    name TEXT
);
CREATE INDEX IF NOT EXISTS trials_study ON trials (study);
"""


def study_key(symbols: list[str], interval: str, horizon: int) -> str:
    return f"{','.join(sorted(symbols))}|{interval}|{horizon}"


class TrialLog:
    """Every portfolio run, so the Sharpe of the best one can be deflated by how many were tried.

    Saved recipes are trials given a name; the Sim's portfolio bot runs them.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(TRIALS_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, study: str, config: dict, result: dict) -> tuple[int, dict]:
        """Log a run and return its id with the study's deflation stats (including this run)."""
        m = result["metrics"]
        with self._lock, self._connect() as conn:
            # Re-running an identical config isn't a new trial
            row = conn.execute("SELECT id FROM trials WHERE study=? AND config=?", (study, json.dumps(config, sort_keys=True))).fetchone()
            if row:
                tid = row["id"]
            else:
                tid = conn.execute(
                    "INSERT INTO trials (created_at, study, config, sharpe_period, periods, metrics, final_blend) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (time.time(), study, json.dumps(config, sort_keys=True), m["sharpe_period"], result["periods"],
                     json.dumps(m), json.dumps(result["final_blend"])),
                ).lastrowid
            srs = [r[0] for r in conn.execute("SELECT sharpe_period FROM trials WHERE study=?", (study,))]
        sr0 = expected_max_sharpe(srs)
        n, g3, g4 = result["periods"], m["skew"], m["kurtosis"]
        return tid, {
            "trials": len(srs),
            "psr": probabilistic_sharpe(m["sharpe_period"], 0.0, n, g3, g4),
            "dsr": probabilistic_sharpe(m["sharpe_period"], sr0, n, g3, g4),
            "luck_sharpe": sr0 * float(np.sqrt(result["periods_per_year"])),
        }

    def save(self, trial_id: int, name: str) -> dict:
        with self._lock, self._connect() as conn:
            if conn.execute("UPDATE trials SET name=? WHERE id=?", (name, trial_id)).rowcount == 0:
                raise KeyError(trial_id)
        return self.recipe(trial_id)

    def recipe(self, trial_id: int) -> dict:
        with self._lock, self._connect() as conn:
            r = conn.execute("SELECT * FROM trials WHERE id=?", (trial_id,)).fetchone()
        if r is None:
            raise KeyError(trial_id)
        return self._row(r)

    def recipes(self) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM trials WHERE name IS NOT NULL ORDER BY created_at DESC").fetchall()
        return [self._row(r) for r in rows]

    def study(self, study: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM trials WHERE study=? ORDER BY sharpe_period DESC", (study,)).fetchall()
        return [self._row(r) for r in rows]

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        return {
            "id": r["id"],
            "created_at": r["created_at"],
            "name": r["name"],
            "config": json.loads(r["config"]),
            "metrics": json.loads(r["metrics"]),
            "final_blend": json.loads(r["final_blend"]),
        }
