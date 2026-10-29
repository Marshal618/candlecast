import numpy as np
import pandas as pd
import pytest

from app.services.fundamentals import DAY, Fundamentals, fill_missing, symbol_scores
from tests.test_crosssection import HORIZON, PeekingForecaster, _run, _universe

D = int(pd.Timestamp("2024-03-01", tz="UTC").timestamp())  # a daily bar's shared-clock key (UTC midnight)


def _funds(earnings=(), ratings=(), shares=()):
    return Fundamentals.from_json({"earnings": list(earnings), "ratings": list(ratings), "shares": list(shares)})


def test_earnings_count_from_the_bar_they_were_public_by():
    before_open = D + 12 * 3600  # 07:00 New York
    after_close = D + 20 * 3600 + 1800  # 16:30 New York
    f = _funds(earnings=[[before_open - 90 * DAY, 1.0, 2.0], [after_close, 1.0, 9.0]])
    assert symbol_scores(f, D, "1d", 100)["surprise"] == 2.0  # tonight's report isn't known at today's close
    assert symbol_scores(f, D + DAY, "1d", 100)["surprise"] == 9.0
    g = _funds(earnings=[[before_open, 1.0, 5.0]])
    assert symbol_scores(g, D, "1d", 100)["surprise"] == 5.0  # this morning's is
    assert np.isnan(symbol_scores(g, D + 101 * DAY, "1d", 100)["surprise"])  # stale


def test_earnings_yield_needs_four_recent_quarters():
    q = [[D - k * 91 * DAY, 0.5 + k * 0.1, 0.0] for k in range(5, 0, -1)]  # 5 quarters, oldest first
    f = _funds(earnings=q)
    assert symbol_scores(f, D, "1d", 50)["value"] == pytest.approx(sum(r[1] for r in q[-4:]) / 50)
    assert np.isnan(symbol_scores(_funds(earnings=q[-3:]), D, "1d", 50)["value"])
    assert np.isnan(symbol_scores(f, D + 400 * DAY, "1d", 50)["value"])


def test_revisions_average_the_last_90_days_excluding_today():
    f = _funds(
        earnings=[[D - 10 * DAY, 1, 0]],
        ratings=[[D - 120 * DAY, -1], [D - 30 * DAY, 1], [D - 2 * DAY, 1], [D - 1 * DAY, -1], [D + 3600, -1]],
    )
    assert symbol_scores(f, D, "1d", 100)["revisions"] == pytest.approx(1 / 3)


def test_size_is_minus_log_market_cap_and_non_companies_score_nothing():
    f = _funds(earnings=[[D - 10 * DAY, 1, 0]], shares=[[D - 5 * DAY, 1e9], [D - 4 * DAY, 9e12], [D - 3 * DAY, 1e9]])
    assert symbol_scores(f, D, "1d", 20)["size"] == pytest.approx(-np.log(1e9 * 20))  # one-day glitch ignored
    etf = _funds(shares=[[D - 5 * DAY, 1e9]])  # shares but no earnings: a fund
    assert all(np.isnan(v) for v in symbol_scores(etf, D, "1d", 20).values())


def test_fill_missing_puts_gaps_mid_ranking():
    x = fill_missing(np.array([[1.0, np.nan, 3.0, 5.0], [np.nan] * 4]))
    assert x.tolist() == [[1.0, 3.0, 3.0, 5.0], [0.0] * 4]


def test_panel_scores_fundamentals_without_look_ahead():
    frames = _universe(n_symbols=6)
    days = list(frames["S0"].index)
    mid = days[250]
    # Every symbol reports once, mid-sample, with surprise = its index; S5 has no data (a fund)
    funds = {f"S{k}": _funds(earnings=[[int(mid.timestamp()), 1.0, float(k)]]) for k in range(5)}
    res = _run(frames, PeekingForecaster(frames), fundamentals=funds, use_fundamentals=["surprise"], periods=40)
    names = [s["name"] for s in res["signals"]]
    assert names == ["mini", "surprise", "momentum", "reversal"]
    sur = next(s for s in res["signals"] if s["name"] == "surprise")
    assert sur["kind"] == "fundamental" and 0 < sur["coverage"] < 1
    assert "rank_ic_residual" in sur

    from app.services.crosssection import build_panel

    p = build_panel(PeekingForecaster(frames), frames, interval="1d", models=[], lookback=64, horizon=HORIZON,
                    periods=40, samples=2, temperature=1.0, top_p=0.9, fundamentals=funds, use_fundamentals=["surprise"])
    for t, ts in enumerate(p.origin_times):
        row = p.signals["surprise"][t]
        if ts < mid.normalize().timestamp():
            assert (row == 0).all()  # nothing reported yet
        elif ts > mid.timestamp() + 30 * DAY and ts < mid.timestamp() + 90 * DAY:
            assert row.tolist() == [0, 1, 2, 3, 4, 2]  # S5 sits at the median


def test_sim_portfolio_bot_values_stocks_on_real_not_rebased_prices(tmp_path):
    from app.services.sim_session import SimStore
    from tests.test_portfolio import _recipe
    from tests.test_sim import FakeForecaster, FakeMarket, _cfg, _frame

    a, b = _frame(700), _frame(700)
    b[["open", "high", "low", "close"]] *= 3  # same chart, 3x the price
    frames = {"AAA": a, "BBB": b}
    t0 = int(a.index[0].normalize().tz_convert("UTC").timestamp())
    quarters = lambda eps: [[t0 + k * 91 * DAY, eps, 0.0] for k in range(8)]  # noqa: E731
    # Real earnings yields: AAA 4/~100, BBB 6/~300, so AAA is cheaper; rebased to 100 each, BBB would look cheaper
    ev = {"AAA": _funds(earnings=quarters(1.0)), "BBB": _funds(earnings=quarters(1.5))}
    now = a.index[-1].timestamp() + 5 * 86400
    store = SimStore(tmp_path / "s.sqlite3", FakeMarket(frames), FakeForecaster(), now=lambda: now,
                     recipes=lambda i: _recipe({"value": 1.0}), fundamentals=ev.get)
    st = store.create(_cfg(bots=["portfolio"], recipe_id=7, blind=True, start="2021-01-04"), seed=1)
    s = store._load(st["id"])
    w = s["memo"]["portfolio"][str(s["config"]["start_index"])]
    assert w[0] > 0 and w[1] == 0
