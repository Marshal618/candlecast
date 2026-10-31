import numpy as np
import pandas as pd
import pytest

from app.services.events import run_event_study
from app.services.fundamentals import DAY, Fundamentals

IDX = pd.bdate_range("2021-01-04", periods=900, tz="America/New_York")


def _frames(rets: np.ndarray) -> dict[str, pd.DataFrame]:
    out = {}
    for k in range(rets.shape[1]):
        close = 100 * np.cumprod(1 + rets[:, k])
        out[f"S{k}"] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e6, "amount": 1e8}, index=IDX
        )
    return out


def _after_close(i: int) -> int:
    """Epoch seconds of 16:30 New York on trading day i."""
    return int((IDX[i].normalize() + pd.Timedelta(hours=16, minutes=30)).timestamp())


def test_after_close_report_reacts_next_day_net_of_the_market():
    n = 8
    rets = np.zeros((len(IDX), n))
    rets[301, 0] = 0.10  # S0 jumps the day after reporting after the close on day 300
    reports = {f"S{k}": [[_after_close(300 + k * 0), 1.0, float(k)]] for k in range(n)}
    # Every symbol reports that night, so each gets an event; only S0 moves
    funds = {s: Fundamentals.from_json({"earnings": e}) for s, e in reports.items()}
    res = run_event_study(_frames(rets), funds, pre=5, post=20, groups=2)
    assert res["events"] == n and res["quarters"] == 1
    # Market-adjusted: S0 beats the equal-weighted average by 10% * 7/8 on day 0; the rest lag by 10%/8
    low, high = res["groups"]
    s0 = 0.10 * 7 / 8
    assert low["reaction"] == pytest.approx((s0 - 3 * 0.10 / 8) / 4)  # S0 has the lowest surprise (0)
    assert high["reaction"] == pytest.approx(-0.10 / 8)
    assert low["car"][res["days"].index(-1)] == 0.0  # anchored on the close before day 0
    assert low["drift"] == pytest.approx(0) and low["runup"] == pytest.approx(0)


def test_drift_after_good_news_is_found_and_significant():
    rng = np.random.default_rng(0)
    n = 30
    rets = rng.normal(0, 0.01, (len(IDX), n))
    funds = {}
    for k in range(n):
        earnings = []
        for q in range(9):
            day = 40 + q * 63 + k % 10
            sur = float(rng.normal(0, 10))
            rets[day + 1 : day + 31, k] += 0.0005 * sur  # drifts in the surprise's direction for 30 days
            earnings.append([_after_close(day), 1.0, sur])
        funds[f"S{k}"] = Fundamentals.from_json({"earnings": earnings})
    funds["ETF"] = Fundamentals.empty()
    frames = _frames(rets)
    res = run_event_study(frames, funds, pre=5, post=40, groups=5)
    assert res["events"] == 9 * n
    g = res["groups"]
    assert g[-1]["drift"] > 0.05 and g[0]["drift"] < -0.05
    assert [x["n"] for x in g] == [54] * 5  # ranked within each quarter: equal groups
    assert res["spread"]["drift"]["t_stat"] > 5
    assert res["spread"]["checkpoints"][0]["day"] == 1


def test_too_few_reports_is_rejected():
    rets = np.zeros((len(IDX), 4))
    funds = {"S0": Fundamentals.from_json({"earnings": [[_after_close(300), 1.0, 1.0]]})}
    with pytest.raises(ValueError):
        run_event_study(_frames(rets), funds, pre=5, post=20, groups=5)


def test_one_bad_print_neither_moves_the_market_nor_counts_as_an_event():
    n = 40
    rets = np.zeros((len(IDX), n))
    rets[310, 0] = 250.0  # a Yahoo glitch inside S0's window
    funds = {f"S{k}": Fundamentals.from_json({"earnings": [[_after_close(300), 1.0, float(k)]]}) for k in range(n)}
    res = run_event_study(_frames(rets), funds, pre=5, post=20, groups=2)
    assert res["events"] == n - 1 and "S0" not in res["symbols"]
    assert all(g["drift"] == pytest.approx(0) for g in res["groups"])
