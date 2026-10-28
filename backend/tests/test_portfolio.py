import numpy as np
import pandas as pd
import pytest

from app.services.crosssection import Panel
from app.services.portfolio import (
    Construction,
    TrialLog,
    blend_weights,
    expected_max_sharpe,
    ledoit_wolf,
    probabilistic_sharpe,
    run_portfolio,
    target_weights,
)
from app.services.sim_session import SimStore

BPY = 252.0


def _panel(T=40, N=8, h=5, seed=0):
    """Random-walk universe with an oracle signal (true forward returns) and a pure-noise signal."""
    rng = np.random.default_rng(seed)
    bars = 200 + T * h + h
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (bars, N)) * np.linspace(0.5, 2, N), axis=0))
    origins = list(range(200, 200 + T * h, h))
    fwd = closes[np.array(origins) + h] / closes[origins] - 1
    days = pd.bdate_range("2020-01-01", periods=bars, tz="UTC")
    return Panel(
        symbols=[f"S{i}" for i in range(N)], interval="1d", horizon=h,
        origin_times=[int(days[o].timestamp()) for o in origins], fwd=fwd,
        signals={"oracle": fwd.copy(), "noise": rng.normal(size=fwd.shape), "anti": -fwd},
        closes=closes, origins=origins, bars_per_year=BPY,
    )


def test_ledoit_wolf_is_symmetric_psd_and_shrinks_toward_identity():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 10))
    cov = ledoit_wolf(x)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0  # sample cov of 30 x 10 is rank-deficient-ish; shrunk is not
    sample = np.cov(x, rowvar=False, bias=True)
    assert np.linalg.cond(cov) < np.linalg.cond(sample)


@pytest.mark.parametrize("method", ["equal", "inverse_vol", "mean_variance"])
def test_long_short_targets_are_dollar_neutral_and_hit_vol_target(method):
    rng = np.random.default_rng(1)
    cov = ledoit_wolf(rng.normal(0, 0.01, (120, 8)))
    alpha = rng.normal(size=8)
    w = target_weights(alpha, cov, Construction(method=method, vol_target=0.10, max_gross=10), BPY)
    assert w.sum() == pytest.approx(0, abs=1e-9)
    assert np.sqrt(w @ cov @ w * BPY) == pytest.approx(0.10)
    assert w[np.argmax(alpha)] > 0 and w[np.argmin(alpha)] < 0


@pytest.mark.parametrize("method", ["equal", "inverse_vol", "mean_variance"])
def test_long_only_targets_never_short_or_lever(method):
    rng = np.random.default_rng(2)
    cov = ledoit_wolf(rng.normal(0, 0.01, (120, 8)))
    w = target_weights(rng.normal(size=8), cov, Construction(method=method, side="long_only", vol_target=0.5), BPY)
    assert (w >= -1e-12).all() and w.sum() <= 1 + 1e-9


def test_gross_cap_binds_before_vol_target():
    cov = np.eye(6) * 1e-8  # almost riskless: hitting 10% vol would need huge leverage
    w = target_weights(np.arange(6.0), cov, Construction(method="equal", max_gross=1.5), BPY)
    assert np.abs(w).sum() == pytest.approx(1.5)


def test_oracle_signal_makes_money_and_inverted_noise_does_not():
    p = _panel()
    good = run_portfolio(p, {"oracle": 1.0}, cost_bps=0)
    bad = run_portfolio(p, {"anti": 1.0}, cost_bps=0)
    assert good["metrics"]["sharpe"] > 3 and good["metrics"]["hit_rate"] > 0.9
    assert bad["metrics"]["sharpe"] < 0
    # A negative weight flips a signal back
    assert run_portfolio(p, {"anti": -1.0}, cost_bps=0)["metrics"]["sharpe"] == pytest.approx(good["metrics"]["sharpe"])


def test_slower_trading_cuts_turnover_and_costs():
    p = _panel()
    fast = run_portfolio(p, {"noise": 1.0}, cost_bps=20, construction=Construction(trade_rate=1.0))
    slow = run_portfolio(p, {"noise": 1.0}, cost_bps=20, construction=Construction(trade_rate=0.25))
    assert slow["metrics"]["turnover"] < fast["metrics"]["turnover"] / 2
    assert slow["metrics"]["cost_drag"] < fast["metrics"]["cost_drag"]


def test_ic_weighting_learns_the_good_signal_from_the_past_only():
    p = _panel()
    r = run_portfolio(p, {"oracle": 1.0, "noise": 1.0}, weighting="ic", ic_window=10)
    assert r["final_blend"]["oracle"] > 0.8
    # Before enough realized IC exists, the manual weights are used
    assert r["blend"][0] == pytest.approx([0.5, 0.5])
    ic = np.array([[0.9, -0.1]] * 5)
    assert blend_weights(ic, np.array([1.0, 1.0]), "ic", 12) == pytest.approx([1.0, 0.0])


def test_deflated_sharpe_penalizes_many_trials():
    assert expected_max_sharpe([0.2]) == 0
    srs = list(np.random.default_rng(3).normal(0, 0.15, 50))
    sr0 = expected_max_sharpe(srs)
    assert sr0 > 0.2  # the best of 50 noisy tries looks good by luck
    psr = probabilistic_sharpe(0.3, 0.0, 40, 0.0, 3.0)
    dsr = probabilistic_sharpe(0.3, sr0, 40, 0.0, 3.0)
    assert psr > 0.95 and dsr < psr


def test_trial_log_counts_distinct_configs_and_saves_recipes(tmp_path):
    log = TrialLog(tmp_path / "r.sqlite3")
    p = _panel()
    res = run_portfolio(p, {"oracle": 1.0})
    cfg = {"signals": [{"name": "oracle", "weight": 1.0}], "interval": "1d", "horizon": 5}
    a, d1 = log.record("study", cfg, res)
    b, d2 = log.record("study", cfg, res)
    assert a == b and d2["trials"] == 1
    _, d3 = log.record("study", {**cfg, "x": 1}, run_portfolio(p, {"noise": 1.0}))
    assert d3["trials"] == 2
    assert log.recipes() == []
    saved = log.save(a, "oracle LS")
    assert saved["name"] == "oracle LS" and saved["final_blend"] == {"oracle": 1.0}
    assert [r["id"] for r in log.recipes()] == [a]
    with pytest.raises(KeyError):
        log.save(999, "nope")


def _recipe(blend, horizon=5, interval="1d"):
    return {
        "id": 7, "name": "test recipe", "final_blend": blend,
        "config": {"interval": interval, "horizon": horizon, "lookback": 64, "samples": 2, "temperature": 1.0, "top_p": 0.9,
                   "construction": {"method": "inverse_vol", "side": "long_short", "select": 0.5, "risk_window": 60,
                                    "vol_target": None, "max_gross": 2.0, "trade_rate": 1.0}},
    }


def test_sim_portfolio_bot_runs_recipe_long_only(tmp_path):
    from tests.test_sim import FakeForecaster, FakeMarket, _cfg, _frame

    frames = {"AAA": _frame(700), "BBB": _frame(700, drift=-0.001), "CCC": _frame(700, drift=0.0005)}
    now = frames["AAA"].index[-1].timestamp() + 5 * 86400
    store = SimStore(tmp_path / "s.sqlite3", FakeMarket(frames), FakeForecaster(), now=lambda: now,
                     recipes=lambda i: _recipe({"momentum": 1.0, "mini": 0.5}))
    st = store.create(_cfg(symbols=["aaa", "bbb", "ccc"], bots=["portfolio"], recipe_id=7), seed=1)
    assert st["config"]["bots"] == ["buy_hold", "portfolio"] and st["recipe"]["name"] == "test recipe"
    st = store.step(st["id"], 12)
    s = store._load(st["id"])
    memo = s["memo"]["portfolio"]
    start = s["config"]["start_index"]
    assert sorted(map(int, memo)) == [start, start + 5, start + 10]  # rebalances every recipe horizon
    for w in memo.values():
        assert min(w) >= 0 and sum(w) <= 1 + 1e-9  # long-only cash account
    assert np.argmax(memo[str(start)]) == 0  # the uptrending symbol ranks top on momentum
    assert "portfolio" in st["players"]


def test_sim_rejects_missing_or_mismatched_recipe(tmp_path):
    from app.services.sim_session import SimError
    from tests.test_sim import FakeForecaster, FakeMarket, _cfg, _frame

    frames = {"AAA": _frame(700), "BBB": _frame(700)}
    now = frames["AAA"].index[-1].timestamp() + 5 * 86400
    store = SimStore(tmp_path / "s.sqlite3", FakeMarket(frames), FakeForecaster(), now=lambda: now,
                     recipes=lambda i: _recipe({"reversal": 1.0}, interval="1h"))
    with pytest.raises(SimError, match="recipe"):
        store.create(_cfg(bots=["portfolio"]), seed=1)
    with pytest.raises(SimError, match="1h bars"):
        store.create(_cfg(bots=["portfolio"], recipe_id=7), seed=1)
