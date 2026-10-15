# Kronos Forecast Dashboard

A local web dashboard that forecasts the next candles for a stock or crypto ticker using
[Kronos](https://github.com/shiyu-coder/Kronos) (Shi et al., AAAI 2026), an open-source
foundation model that tokenizes OHLCV candles and predicts the next ones autoregressively,
the same way a language model predicts the next word.

- **Forecast:** median of N sampled future paths plus a p10–p90 uncertainty band, P(up), and expected move.
- **Backtest:** walk-forward forecasts from past windows scored against what actually happened,
  next to naive baselines (no-change price, majority direction).
- Data comes from Yahoo Finance via `yfinance` (free, no key, unofficial), cached in SQLite.

> Research tool, not financial advice.

## Run locally

Needs Python 3.10–3.13 and Node 18+.

```bash
./dev.sh
```

Open http://127.0.0.1:5173. The API runs on port 8010 (change with `API_PORT=9000 ./dev.sh`). The first forecast with each model size downloads its weights
from Hugging Face (mini ~20MB, small ~100MB, base ~400MB).

Or run each part on its own:

```bash
cd backend && python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8010 --reload --reload-dir app

cd frontend && npm install && npm run dev
```

## Tests

```bash
cd backend
.venv/bin/pytest                  # fast unit tests, no network
.venv/bin/pytest -m integration   # live yfinance and real Kronos-mini weights
```

## Models (measured on an Apple M3, 400-bar lookback, 20-bar horizon, 10 paths)

| Model | Params | Max lookback | Latency |
|---|---|---|---|
| mini  | 4.1M   | 2048 | ~1s  |
| small | 24.7M  | 512  | ~4s  |
| base  | 102M   | 512  | ~27s |

## Deploying later

Everything is configured via `KD_*` env vars (see `backend/.env.example`).

```bash
docker compose up --build     # single container on :8000 serving API and built frontend
```

- **The data provider is pluggable.** Implement `MarketDataProvider.fetch` in
  `backend/app/services/market_data.py` (e.g. Polygon, Alpaca) for production-grade data.
  yfinance is rate-limited and not licensed for commercial use.
- **For a GPU host,** use a CUDA base image and CUDA torch, then set `KD_DEVICE=cuda:0`.
- **If the frontend is hosted separately,** build it with `VITE_API_BASE=https://your-api` and
  add that origin to `KD_CORS_ORIGINS`.
- **Inference is serialized behind a lock (one forecast at a time per process).** For multiple
  users, put a job queue in front of the forecaster or run several workers on separate GPUs.

## Layout

```
backend/
  app/main.py                 FastAPI routes: /api/health, /api/candles, /api/forecast, /api/backtest
  app/services/market_data.py yfinance provider, SQLite cache, future-timestamp projection
  app/services/forecaster.py  Kronos loading + independent sampled paths + quantile stats
  app/services/backtest.py    walk-forward evaluation with baselines
  vendor/kronos/              upstream Kronos model code (MIT), see UPSTREAM_COMMIT
frontend/
  src/App.tsx                 dashboard UI
  src/charts.tsx              lightweight-charts candle/forecast/backtest charts
```

## Known behaviour

Kronos normalizes each input window by that window's mean and std, so after a strong trend it tends
to forecast reversion toward the window's average price. Longer lookbacks pull harder. Use the
Backtest tab to check whether that has paid off for a given ticker and interval before trusting it.
