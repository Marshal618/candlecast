"""Earnings event study: what prices do around a report, by how much it beat or missed.

The Research tab samples signals once per rebalance. Post-earnings drift is an event effect, so
this lines every report up on its own day instead (Ball & Brown 1968, Bernard & Thomas 1989):

- Day 0 is the first daily bar a report was public by: before-open reports count the same day,
  after-close ones the next (the same rule the fundamentals use, see fundamentals._cutoffs).
- Abnormal return: the stock's daily return minus the average of every symbol in the run that day
  (market-adjusted). The average trims the top and bottom 5% each day, so one bad print can't move
  it: Yahoo has the odd glitch (CHRD shows a +25,000% day after its 2020 restructuring), and
  event windows containing a daily move beyond +300% / -90% are skipped as bad data. Cumulative abnormal return (CAR) is anchored at 0 on the
  close before day 0, so CAR on day 0 is the announcement reaction, and CAR(k) - CAR(0) is the
  drift you could have captured buying at day 0's close. Run-up is the abnormal return over the
  `pre` days before day 0.
- Groups: each report is ranked by surprise among the reports of the same calendar quarter (one
  earnings season), then split into `groups`, lowest surprise first. Ranking within the season
  keeps every group spread across the whole sample.
- Significance: reports in the same season share market moves, so pooling them overstates
  confidence. The top-minus-bottom gap is averaged per quarter and the t-stat is taken across
  quarters.

Only reports with a consensus estimate and the full window of prices are used. Same survivorship
caveat as the index universes: today's members only.
"""

from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from app.services.fundamentals import Fundamentals
from app.services.sim import align_keys

CHECKPOINTS = (1, 5, 20, 60, 120)
TRIM = 0.05  # share of symbols cut from each end of a day's returns for the market average
BAD_PRINT = (-0.9, 3.0)  # daily returns outside this are data errors, not events


def _t(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return {"mean": float(x.mean()) if len(x) else 0.0, "t_stat": 0.0, "n": int(len(x))}
    return {"mean": float(x.mean()), "t_stat": float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))), "n": int(len(x))}


def abnormal_returns(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.Series], dict[str, np.ndarray]]:
    """Each symbol's daily return minus that day's trimmed mean, on its own trading days (and the raw returns)."""
    rets = {}
    for s, f in frames.items():
        c = f["close"].astype(float)
        c.index = align_keys(f.index, "1d")
        c = c[~c.index.duplicated(keep="last")]
        rets[s] = c.pct_change().iloc[1:]
    panel = pd.DataFrame(rets)
    market = pd.Series([trim_mean(row[np.isfinite(row)], TRIM) for row in panel.to_numpy()], index=panel.index)
    return {s: (r - market.loc[r.index]) for s, r in rets.items()}, {s: r.to_numpy() for s, r in rets.items()}


def run_event_study(
    frames: dict[str, pd.DataFrame], fundamentals: dict[str, Fundamentals], *, pre: int, post: int, groups: int
) -> dict:
    ar, raw_returns = abnormal_returns(frames)
    events = []  # (symbol, report time, surprise, car path)
    skipped = 0
    for s, f in fundamentals.items():
        if s not in ar or len(f.earn_t) == 0:
            continue
        r = ar[s]
        keys = (r.index.asi8 // 10**9).astype(np.int64)
        vals = r.to_numpy()
        raw = raw_returns[s]
        for t, sur in zip(f.earn_t, f.surprise):
            # First bar whose close came after the report (see fundamentals._cutoffs)
            d0 = int(np.searchsorted(keys, t - 16 * 3600, side="right"))
            if not np.isfinite(sur) or d0 - pre < 0 or d0 + post >= len(vals):
                skipped += 1
                continue
            w = vals[d0 - pre : d0 + post + 1]
            rw = raw[d0 - pre : d0 + post + 1]
            if not np.isfinite(w).all() or rw.min() < BAD_PRINT[0] or rw.max() > BAD_PRINT[1]:
                skipped += 1
                continue
            # CAR at day d < 0 is minus the returns of days d+1..-1, so CAR(-1) = 0; then days 0..post
            before = -np.concatenate([np.cumsum(w[:pre][::-1])[::-1], [0.0]])[1:]
            after = np.cumsum(w[pre:])
            events.append((s, int(t), float(sur), np.concatenate([before, after]), float(w[:pre].sum())))

    if len(events) < groups * 4:
        raise ValueError(
            f"Only {len(events)} earnings reports with a consensus and {pre}+{post} days of prices; "
            f"need at least {groups * 4}. Use stocks (not ETFs or crypto), more symbols, or a shorter window."
        )

    # Rank within each calendar quarter's reports
    quarter = [datetime.fromtimestamp(t, timezone.utc) for _, t, *_ in events]
    quarter = [d.year * 4 + (d.month - 1) // 3 for d in quarter]
    by_q = defaultdict(list)
    for i, q in enumerate(quarter):
        by_q[q].append(i)
    group = np.full(len(events), -1)
    for idx in by_q.values():
        if len(idx) < groups:
            continue  # too few reports that quarter to split
        sur = np.array([events[i][2] for i in idx])
        pct = (pd.Series(sur).rank(method="average").to_numpy() - 0.5) / len(idx)
        group[idx] = np.minimum((pct * groups).astype(int), groups - 1)

    keep = group >= 0
    cars = np.stack([e[3] for e in events])[keep]
    sur = np.array([e[2] for e in events])[keep]
    grp, qtr = group[keep], np.array(quarter)[keep]
    times = np.array([e[1] for e in events])[keep]
    runup = np.array([e[4] for e in events])[keep]
    day0 = pre
    out_groups = []
    for k in range(groups):
        m = grp == k
        c = cars[m]
        out_groups.append({
            "group": k,
            "n": int(m.sum()),
            "surprise_median": float(np.median(sur[m])),
            "car": [float(x) for x in c.mean(axis=0)],
            "se": [float(x) for x in c.std(axis=0, ddof=1) / np.sqrt(len(c))],
            "reaction": float(c[:, day0].mean()),
            "drift": float((c[:, -1] - c[:, day0]).mean()),
            "runup": float(runup[m].mean()),
        })

    def spread(values: np.ndarray) -> dict:
        """Top-minus-bottom group mean of `values`, per quarter, with a t-stat across quarters."""
        gaps = []
        for q in np.unique(qtr):
            top, bot = (qtr == q) & (grp == groups - 1), (qtr == q) & (grp == 0)
            if top.any() and bot.any():
                gaps.append(values[top].mean() - values[bot].mean())
        return _t(np.array(gaps))

    return {
        "symbols": sorted({e[0] for e in events}),
        "no_earnings": sorted(s for s in frames if s not in fundamentals or len(fundamentals[s].earn_t) == 0),
        "events": int(keep.sum()),
        "skipped": skipped + int((~keep).sum()),
        "first": int(times.min()),
        "last": int(times.max()),
        "quarters": int(len(np.unique(qtr))),
        "days": list(range(-pre, post + 1)),
        "groups": out_groups,
        "spread": {
            "reaction": spread(cars[:, day0]),
            "drift": spread(cars[:, -1] - cars[:, day0]),
            "runup": spread(runup) if pre else None,  # the `pre` days before day 0
            "checkpoints": [{"day": d, **spread(cars[:, day0 + d] - cars[:, day0])} for d in CHECKPOINTS if d <= post],
        },
    }
