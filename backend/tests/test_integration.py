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
