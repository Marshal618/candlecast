"""End-to-end API tests against live yfinance data and real Kronos-mini weights.

Slow and needs network: run with `pytest -m integration`.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("symbol", ["AAPL", "BTC-USD"])
def test_forecast(client, symbol):
    r = client.post(
        "/api/forecast",
        json={"symbol": symbol, "interval": "1d", "model": "mini", "lookback": 256, "horizon": 10, "samples": 8},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["forecast"]) == 10
    assert body["forecast"][0]["time"] > body["last_time"]
    assert 0 <= body["stats"]["p_up"] <= 1
    for p in body["forecast"]:
        assert p["p10"] <= p["median"] <= p["p90"]


def test_backtest(client):
    r = client.post(
        "/api/backtest",
        json={"symbol": "AAPL", "interval": "1d", "model": "mini", "lookback": 256, "horizon": 5, "windows": 6, "samples": 3},
    )
    assert r.status_code == 200, r.text
    m = r.json()["metrics"]
    assert m["windows"] == 6
    assert 0 <= m["hit_rate"] <= 1


def test_unknown_symbol_is_404(client):
    r = client.get("/api/candles", params={"symbol": "NOTAREALTICKERXYZ", "interval": "1d"})
    assert r.status_code == 404


def test_lookback_over_context_is_422(client):
    r = client.post("/api/forecast", json={"symbol": "AAPL", "model": "small", "lookback": 1000})
    assert r.status_code == 422


def test_tokens_inspection(client):
    r = client.post("/api/tokens", json={"symbol": "SPY", "interval": "1d", "model": "mini", "lookback": 256, "segments": 8})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["bars"]) == 256
    assert body["bars"][0]["surprisal"] is None and body["bars"][1]["surprisal"] > 0
    n = body["next"]
    assert 0 <= n["entropy_bits"] <= n["max_entropy_bits"]
    probs = [t["p"] for t in n["top"]]
    assert probs == sorted(probs, reverse=True) and sum(probs) <= 1 + 1e-6
    assert len(body["attribution"]) == 8 and all(a["kl_bits"] >= -1e-9 for a in body["attribution"])
    # Fine tokens should reconstruct prices at least roughly as well as coarse-only
    s = body["summary"]
    assert s["recon_mae_full_pct"] <= s["recon_mae_coarse_pct"] * 1.5
    assert 0 <= body["semantics"]["explained"]["level"] <= 1


def test_backtest_reports_r2_and_ridge(client):
    r = client.post(
        "/api/backtest",
        json={"symbol": "SPY", "interval": "1d", "model": "mini", "lookback": 256, "horizon": 5, "windows": 6, "samples": 3},
    )
    m = r.json()["metrics"]
    assert m["r2_oos"] <= 1 and m["ridge_r2_oos"] <= 1
    assert 0 <= m["ridge_hit_rate"] <= 1


def test_sim_game_with_mixed_stock_and_crypto(client):
    r = client.post("/api/sim", json={"symbols": ["AAPL", "BTC-USD"], "interval": "1d", "length": 20,
                                      "bots": ["sma_cross", "kronos"], "kronos_model": "mini"})
    assert r.status_code == 200, r.text
    s = r.json()
    sid = s["id"]
    assert s["period"] is None and s["kronos"] is None
    s = client.post(f"/api/sim/{sid}/orders", json={"symbol": "BTC-USD", "side": "buy", "qty": 10, "type": "market"}).json()
    s = client.post(f"/api/sim/{sid}/step", json={"bars": 5}).json()
    assert s["orders"][0]["status"] == "filled"
    assert client.post(f"/api/sim/{sid}/hint", json={"symbol": "AAPL"}).status_code == 200
    s = client.post(f"/api/sim/{sid}/step", json={"bars": 50}).json()
    assert s["finished"] and s["period"] is not None and s["kronos"] is not None
    assert set(s["result"]) == {"you", "buy_hold", "sma_cross", "kronos"}


def test_backtest_calibration_applies_only_to_matching_forecast_settings(client):
    base = {"symbol": "SPY", "interval": "1d", "model": "mini", "lookback": 128, "horizon": 5, "samples": 4}
    bt = client.post("/api/backtest", json={**base, "windows": 12}).json()
    assert bt["calibration"]["oos_windows"] == 12 - 5
    f = client.post("/api/forecast", json=base).json()
    assert f["calibration"] is not None and "cal_p10" in f["forecast"][0]
    assert f["forecast"][-1]["cal_p10"] <= f["forecast"][-1]["cal_p90"]
    other = client.post("/api/forecast", json={**base, "samples": 8}).json()
    assert other["calibration"] is None and other["calibration_mismatch"] is True


@pytest.mark.parametrize("model", ["naive", "garch", "chronos"])
def test_challenger_forecast_and_backtest(client, model):
    base = {"symbol": "SPY", "interval": "1d", "model": model, "lookback": 256, "horizon": 5, "samples": 8}
    f = client.post("/api/forecast", json=base)
    assert f.status_code == 200, f.text
    body = f.json()
    assert len(body["forecast"]) == 5 and 0 <= body["stats"]["p_up"] <= 1
    assert all(p["p10"] <= p["median"] <= p["p90"] for p in body["forecast"])
    b = client.post("/api/backtest", json={**base, "windows": 8}).json()
    assert b["metrics"]["windows"] == 8 and b["calibration"] is not None


def test_tokens_rejects_challengers(client):
    r = client.post("/api/tokens", json={"symbol": "SPY", "interval": "1d", "model": "garch"})
    assert r.status_code == 422


def test_crosssection_ranks_a_universe(client):
    r = client.post(
        "/api/crosssection",
        json={"symbols": ["XLE", "XLF", "XLK", "XLU", "XLV", "BTC-USD"], "models": ["mini", "naive"], "periods": 6, "samples": 3},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["periods"] == 6 and len(d["symbols"]) == 6
    assert [s["name"] for s in d["signals"]] == ["mini", "naive", "momentum", "reversal"]
    for s in d["signals"]:
        assert -1 <= s["rank_ic"]["mean"] <= 1
        assert len(s["equity"]) == 6 and len(s["bucket_returns"]) == 3
    assert "rank_ic_residual" in d["signals"][0] and "rank_ic_residual" not in d["signals"][-1]


def test_portfolio_lab_logs_trials_and_saves_a_recipe(client):
    body = {
        "symbols": ["XLE", "XLF", "XLK", "XLU", "XLV", "XLP"],
        "signals": [{"name": "mini", "weight": 1}, {"name": "reversal", "weight": 1}],
        "periods": 6, "samples": 3, "construction": {"risk_window": 60},
    }
    r = client.post("/api/portfolio", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["periods"] == 6 and len(d["weights"][0]) == 6 and d["deflation"]["trials"] >= 1
    assert abs(sum(d["weights"][-1])) < 1e-6  # long-short is dollar neutral
    again = client.post("/api/portfolio", json=body).json()
    assert again["trial_id"] == d["trial_id"]  # same config isn't a new trial (and hits the score cache)
    saved = client.post("/api/portfolio/recipes", json={"trial_id": d["trial_id"], "name": "it"}).json()
    assert saved["name"] == "it"
    assert any(x["id"] == d["trial_id"] for x in client.get("/api/portfolio/recipes").json())
