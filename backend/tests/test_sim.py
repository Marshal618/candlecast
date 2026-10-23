"""Market replay engine and game sessions: fills, costs, account rules, bots, no lookahead."""

import numpy as np
import pandas as pd
import pytest

from app.services import sim
from app.services.sim_session import WARMUP, SimError, SimStore


def _market(opens, highs, lows, closes, symbols=("AAA",)):
    a = lambda x: np.array(x, dtype=float).reshape(len(x), -1)  # noqa: E731
    n = len(closes)
    return sim.Market(list(symbols), np.arange(n, dtype=np.int64) * 86400, a(opens), a(highs), a(lows), a(closes), np.ones((n, len(symbols))))


def _flat(n, price=100.0):
    return _market([price] * n, [price] * n, [price] * n, [price] * n)


def _run(m, actions, cursor, cash=10_000, fee=0.0, slip=0.0, bots=(), weights=None):
    return sim.simulate(m, start=0, cursor=cursor, cash=cash, commission_bps=fee, slippage_bps=slip,
                        actions=actions, bots=list(bots), bot_weights=weights or (lambda b, t: None))


def _order(at, side="buy", qty=10, kind="market", price=None, oid=1):
    return {"at": at, "kind": "order", "id": oid, "symbol_index": 0, "side": side, "type": kind, "qty": qty, "price": price}


def test_market_order_fills_at_next_open_with_costs():
    m = _market([100, 110, 120], [100, 115, 125], [100, 105, 115], [100, 112, 122])
    r = _run(m, [_order(0)], cursor=1, fee=10, slip=10)
    o = r["orders"][0]
    assert o["status"] == "filled" and o["fill_price"] == pytest.approx(110 * 1.001)
    acct = r["account"]
    assert acct["cash"] == pytest.approx(10_000 - 10 * 110 * 1.001 * 1.001)
    assert acct["positions"][0]["value"] == pytest.approx(10 * 112)


def test_order_never_fills_on_the_bar_it_was_placed():
    m = _flat(3)
    r = _run(m, [_order(1)], cursor=1)
    assert r["orders"][0]["status"] == "open"


def test_limit_buy_fills_at_limit_or_better_gap():
    # Bar 1 trades down through 95 -> fills at 95; bar 2 gaps open below a 95 limit -> fills at the open
    m = _market([100, 98, 90], [100, 99, 91], [100, 94, 89], [100, 96, 90])
    r = _run(m, [_order(0, kind="limit", price=95, oid=1), _order(1, kind="limit", price=95, oid=2)], cursor=2)
    assert [o["fill_price"] for o in r["orders"]] == [95, 90]


def test_limit_waits_until_touched():
    m = _market([100, 100, 100], [101, 101, 101], [99, 99, 97], [100, 100, 98])
    r = _run(m, [_order(0, kind="limit", price=98)], cursor=1)
    assert r["orders"][0]["status"] == "open"
    r = _run(m, [_order(0, kind="limit", price=98)], cursor=2)
    assert r["orders"][0]["status"] == "filled" and r["orders"][0]["fill_price"] == 98


def test_stop_sell_gaps_through_and_pays_slippage():
    m = _market([100, 100, 80], [100, 100, 82], [100, 100, 78], [100, 100, 80])
    actions = [_order(0, qty=10, oid=1), _order(1, side="sell", qty=10, kind="stop", price=90, oid=2)]
    r = _run(m, actions, cursor=2, slip=10)
    stop = r["orders"][1]
    assert stop["status"] == "filled" and stop["fill_price"] == pytest.approx(80 * 0.999)


def test_buy_without_cash_is_rejected_not_partially_filled():
    m = _flat(3)
    r = _run(m, [_order(0, qty=1000)], cursor=1, cash=1000)
    assert r["orders"][0]["status"] == "rejected" and r["orders"][0]["reason"] == "insufficient cash"
    assert r["account"]["cash"] == 1000


def test_long_only_sell_capped_at_position():
    m = _flat(4)
    r = _run(m, [_order(0, qty=5, oid=1), _order(1, side="sell", qty=50, oid=2)], cursor=2)
    assert r["orders"][1]["filled_qty"] == 5
    assert r["account"]["positions"] == []


def test_cancel_stops_an_open_order():
    m = _market([100, 100, 100], [100, 100, 100], [100, 100, 90], [100, 100, 95])
    actions = [_order(0, kind="limit", price=92), {"at": 1, "kind": "cancel", "order_id": 1}]
    assert _run(m, actions, cursor=2)["orders"][0]["status"] == "cancelled"


def test_realized_pnl_and_fees():
    m = _market([100, 100, 120], [100, 100, 120], [100, 100, 120], [100, 100, 120])
    r = _run(m, [_order(0, qty=10, oid=1), _order(1, side="sell", qty=10, oid=2)], cursor=2, fee=10)
    a = r["account"]
    assert a["realized"] == pytest.approx(10 * 120 * 0.999 - 10 * 100 * 1.001)
    assert a["fees"] == pytest.approx(10 * 100 * 0.001 + 10 * 120 * 0.001)


def test_buy_hold_bot_matches_price_path():
    closes = [100, 100, 110, 121]
    m = _market(closes, closes, closes, closes)
    r = _run(m, [], cursor=3, bots=["buy_hold"], weights=lambda b, t: sim.buy_hold_weights(m, t, 0))
    eq = r["curves"]["buy_hold"]
    assert eq[-1] / eq[1] == pytest.approx(1.21)


def test_bot_trade_waits_for_a_bar_when_symbol_is_closed():
    # Second symbol has no bar at t=1 (weekend); the bot buys it at t=2 instead of skipping it
    nan = np.nan
    m = sim.Market(["CRYPTO", "STOCK"], np.arange(3, dtype=np.int64) * 86400,
                   np.array([[100, 50], [100, nan], [100, 50]], float), np.array([[100, 50], [100, nan], [100, 50]], float),
                   np.array([[100, 50], [100, nan], [100, 50]], float), np.array([[100, 50], [100, nan], [100, 50]], float),
                   np.ones((3, 2)))
    r = _run(m, [], cursor=2, bots=["buy_hold"], weights=lambda b, t: sim.buy_hold_weights(m, t, 0))
    assert r["players"]["buy_hold"]["cash"] == pytest.approx(0, abs=1e-6)


def test_sma_cross_is_flat_in_downtrend_long_in_uptrend():
    up = np.linspace(100, 200, 60)
    m = _market(up, up, up, up)
    assert sim.sma_cross_weights(m, 59, 0)[0] == 1.0
    down = up[::-1]
    m = _market(down, down, down, down)
    assert sim.sma_cross_weights(m, 59, 0)[0] == 0.0


def test_daily_bars_from_different_timezones_share_a_row_per_day():
    stock = _frame(5, tz="America/New_York")
    crypto = _frame(7, tz="UTC")
    m = sim.build_market({"STOCK": stock, "CRYPTO": crypto}, "1d")
    assert len(m.times) == 7  # one row per calendar day, not one per timestamp
    assert not np.isnan(m.c[:5, 0]).any() and np.isnan(m.c[5:, 0]).all()


def test_performance_metrics():
    p = sim.performance([100, 110, 99, 120], 252)
    assert p["return"] == pytest.approx(0.2)
    assert p["max_drawdown"] == pytest.approx(99 / 110 - 1)


# --- sessions --------------------------------------------------------------------------------


class FakeMarket:
    def __init__(self, frames):
        self.frames = frames

    def get(self, symbol, interval):
        return self.frames[symbol]


class FakeForecaster:
    """Always very bullish, so the Kronos bot goes all in."""

    calls = 0

    def sample_paths(self, contexts, future_index, model, samples, temperature, top_p):
        FakeForecaster.calls += 1
        out = np.zeros((len(contexts), samples, len(future_index[0]), 6))
        for i, c in enumerate(contexts):
            out[i, :, :, :4] = c["close"].iloc[-1] * 1.05
        return out


def _frame(n, start="2020-01-01", drift=0.001, tz="America/New_York"):
    idx = pd.date_range(start, periods=n, freq="1D", tz=tz)
    close = 100 * np.cumprod(np.full(n, 1 + drift))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)


@pytest.fixture
def store(tmp_path):
    frames = {"AAA": _frame(700), "BBB": _frame(700, drift=-0.001)}
    now = frames["AAA"].index[-1].timestamp() + 5 * 86400
    return SimStore(tmp_path / "s.sqlite3", FakeMarket(frames), FakeForecaster(), now=lambda: now)


def _cfg(**kw):
    c = dict(symbols=["aaa", "bbb"], interval="1d", cash=10_000, length=50, start="random", blind=True,
             commission_bps=0, slippage_bps=0, bots=["sma_cross", "kronos"], kronos_model="mini")
    c.update(kw)
    return c


def test_session_never_reveals_future_bars(store):
    st = store.create(_cfg(), seed=1)
    assert st["step"] == 0 and st["period"] is None  # blind: real dates hidden
    last_visible = max(c["time"] for c in st["candles"]["AAA"])
    assert last_visible == st["time"]
    assert all(q["last"] == pytest.approx(100) for q in st["quotes"])  # rebased in blind mode
    assert st["kronos"] is None  # bot readings would be free hints mid-game


def test_session_play_through(store):
    st = store.create(_cfg(blind=False, start="2021-01-04"), seed=1)
    sid = st["id"]
    st = store.order(sid, "AAA", "buy", 10, "market", None)
    assert st["orders"][0]["status"] == "open"
    st = store.step(sid, 1)
    assert st["orders"][0]["status"] == "filled" and st["account"]["positions"][0]["symbol"] == "AAA"
    with pytest.raises(SimError, match="Can only sell"):
        store.order(sid, "AAA", "sell", 11, "market", None)
    hint = store.hint(sid, "AAA", 5)
    assert hint["stats"]["p_up"] == 1.0 and len(hint["forecast"]) == 5
    assert hint["forecast"][0]["time"] > st["time"]
    st = store.step(sid, 100)  # runs past the end -> finishes
    assert st["finished"] and st["step"] == 50
    assert set(st["result"]) == {"you", "buy_hold", "sma_cross", "kronos"}
    with pytest.raises(SimError, match="finished"):
        store.step(sid, 1)
    board = store.leaderboard()
    assert board[0]["id"] == sid and board[0]["hints"] == 1 and board[0]["period"] is not None


def test_kronos_bot_decisions_are_memoized(store):
    st = store.create(_cfg(blind=False), seed=3)
    calls = FakeForecaster.calls
    store.step(st["id"], 10)
    after_step = FakeForecaster.calls
    assert after_step == calls + 1  # one batched call covers both decision bars in the step
    store.state(st["id"])
    assert FakeForecaster.calls == after_step  # replays reuse the memo
    s = store.state(st["id"])
    assert s["players"]["kronos"]["cash"] < 1  # bullish fake -> fully invested


def test_start_date_out_of_range(store):
    with pytest.raises(SimError, match="Start must be between"):
        store.create(_cfg(start="2020-01-05"))


def test_game_too_long_for_history(store):
    with pytest.raises(SimError, match="Not enough shared history"):
        store.create(_cfg(length=1000))
