"""Turn walk-forward forecasts into a trading strategy and score it honestly.

Each backtest window gives a P(up) computed from data up to its origin bar. The rule maps
that to a position (long / flat / short) held for the window's horizon bars.

Execution model (no lookahead):
- The signal is formed at the origin bar's close, but Kronos takes seconds to run, so the
  position changes at the *next* bar's open. The overnight / between-bar gap is earned by
  whatever position was held before.
- Every position change pays `cost_bps` per unit of turnover (fees + slippage), including
  the final exit.

Because forecasts are fixed once computed, re-scoring with different rules is pure numpy
and needs no model calls.
"""

import numpy as np
import pandas as pd

PERMUTATIONS = 1000


def positions_from_signal(p_up: np.ndarray, mode: str, enter_p: float) -> np.ndarray:
    long = p_up >= enter_p
    short = (p_up <= 1 - enter_p) if mode == "long_short" else np.zeros_like(long)
    return np.where(long, 1.0, np.where(short, -1.0, 0.0))


def _simulate(pos: np.ndarray, gap: np.ndarray, intra: np.ndarray, cost: float) -> np.ndarray:
    """Per-bar returns for position arrays of shape (..., bars)."""
    prev = np.concatenate([np.zeros(pos.shape[:-1] + (1,)), pos[..., :-1]], axis=-1)
    r = (1 + prev * gap) * (1 + pos * intra) - 1 - np.abs(pos - prev) * cost
    r[..., -1] -= np.abs(pos[..., -1]) * cost  # close out at the end
    return r


def _max_drawdown(equity: np.ndarray) -> float:
    return float(np.min(equity / np.maximum.accumulate(equity) - 1))


def _sharpe(r: np.ndarray, bars_per_year: float) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Inferred from history, so stocks (sessions only) and crypto (24/7) both annualize correctly."""
    years = (index[-1] - index[0]).total_seconds() / (365.25 * 86400)
    return len(index) / years if years > 0 else 252.0


def run_strategy(
    df: pd.DataFrame,
    windows: list[dict],
    horizon: int,
    *,
    mode: str = "long_only",
    enter_p: float = 0.6,
    cost_bps: float = 5.0,
    seed: int = 0,
) -> dict:
    """`windows` need `origin_time` (epoch s of the last context bar) and `p_up`."""
    stamps = df.index.asi8 // 10**9
    windows = sorted(windows, key=lambda w: w["origin_time"])
    origins = np.searchsorted(stamps, [w["origin_time"] for w in windows])
    if (origins >= len(stamps)).any() or (stamps[np.minimum(origins, len(stamps) - 1)] != [w["origin_time"] for w in windows]).any():
        raise ValueError("Backtest windows no longer match cached market data. Re-run the backtest.")
    entries = origins + 1  # first bar the position is live

    start, end = int(entries[0]), int(min(entries[-1] + horizon, len(df)))
    o = df["open"].to_numpy()[start:end]
    c = df["close"].to_numpy()[start:end]
    c_prev = df["close"].to_numpy()[start - 1 : end - 1]
    gap, intra = o / c_prev - 1, c / o - 1
    cost = cost_bps / 1e4

    # Which window governs each bar (-1 = none); later windows override overlapping ones
    owner = np.full(end - start, -1)
    for k, e in enumerate(entries):
        owner[e - start : e - start + horizon] = k

    p_up = np.array([w["p_up"] for w in windows], dtype=float)
    win_pos = positions_from_signal(p_up, mode, enter_p)
    bar_pos = np.where(owner >= 0, win_pos[owner], 0.0)

    r = _simulate(bar_pos, gap, intra, cost)
    bh = _simulate(np.ones_like(bar_pos), gap, intra, cost)
    equity, bh_equity = np.cumprod(1 + r), np.cumprod(1 + bh)
    bpy = bars_per_year(df.index)

    active = win_pos != 0
    win_returns = np.array([np.prod(1 + r[owner == k]) - 1 for k in range(len(windows))])
    prev_pos = np.concatenate([[0.0], bar_pos[:-1]])

    # Permutation test: same mix of long/flat/short decisions, randomly reassigned to windows.
    # p-value = how often random timing does at least as well as the model's timing.
    p_value = None
    if len(np.unique(win_pos)) > 1:
        rng = np.random.default_rng(seed)
        shuffled = rng.permuted(np.tile(win_pos, (PERMUTATIONS, 1)), axis=1)
        padded = np.concatenate([shuffled, np.zeros((PERMUTATIONS, 1))], axis=1)
        perm_ret = np.prod(1 + _simulate(padded[:, owner], gap, intra, cost), axis=1) - 1
        p_value = float((np.sum(perm_ret >= equity[-1] - 1) + 1) / (PERMUTATIONS + 1))

    times = [int(t.timestamp()) for t in df.index[start - 1 : end]]
    return {
        "params": {"mode": mode, "enter_p": enter_p, "cost_bps": cost_bps},
        "metrics": {
            "total_return": float(equity[-1] - 1),
            "bh_return": float(bh_equity[-1] - 1),
            "sharpe": _sharpe(r, bpy),
            "bh_sharpe": _sharpe(bh, bpy),
            "max_drawdown": _max_drawdown(np.concatenate([[1.0], equity])),
            "bh_max_drawdown": _max_drawdown(np.concatenate([[1.0], bh_equity])),
            "exposure": float(np.mean(bar_pos != 0)),
            "trades": int(np.sum((bar_pos != prev_pos) & (bar_pos != 0))),
            "active_windows": int(active.sum()),
            "win_rate": float(np.mean(win_returns[active] > 0)) if active.any() else None,
            "p_value": p_value,
        },
        "equity": [
            {"time": t, "strategy": float(s), "buy_hold": float(b)}
            for t, s, b in zip(times, np.concatenate([[1.0], equity]), np.concatenate([[1.0], bh_equity]))
        ],
        "positions": [{"time": int(windows[k]["origin_time"]), "position": float(p)} for k, p in enumerate(win_pos)],
    }
