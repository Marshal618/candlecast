# Candlecast

Research dashboard for candlestick forecasting, built on Kronos (Shi et al., AAAI 2026), an open-source foundation model that reads OHLCV candles as tokens. It projects future candles with uncertainty bands, tests them with walk-forward backtests and strategy simulations, and shows how the model tokenizes the chart.

*Independent project, not affiliated with the Kronos authors. Kronos model code is vendored under its MIT license in `backend/vendor/kronos/`.*

A local web dashboard that projects the next candles for a stock or crypto ticker using
[Kronos](https://github.com/shiyu-coder/Kronos) (Shi et al., AAAI 2026), an open-source
foundation model that tokenizes OHLCV candles and predicts the next ones autoregressively,
the same way a language model predicts the next word.

- **Projected candles:** future bars drawn as violet candles (hollow = up, solid = down) right after the
  real teal/red history, so the projection never blends into real data. They come from the *medoid*
  path, the one sampled future closest to the median, so each bar is an actual Kronos output with
  consistent open/high/low/close rather than a per-field average.
- **Forecast:** median close of N sampled future paths plus a p10–p90 uncertainty band, P(up), and expected move.
- **Backtest:** walk-forward forecasts from past windows scored against what actually happened,
  next to naive baselines (no-change price, majority direction), out-of-sample R², and a
  walk-forward ridge regression baseline.
- **Model comparison:** run the same forecast or backtest with three challengers (a naive
  benchmark, GARCH, and Amazon's Chronos-Bolt foundation model) to see whether Kronos actually
  adds anything. See [Comparing models](#comparing-models).
- **Calibrated bands:** Kronos's raw p10–p90 bands are far too narrow (20–55% real coverage in
  our tests). Backtests conformally calibrate them, and live forecasts show the corrected band.
  See [Band calibration](#band-calibration).
- **Forecast journal:** every forecast you run is logged and scored automatically once its bars
  close, giving a live track record the model cannot have seen in pre-training. See
  [Forecast journal](#forecast-journal).
- **Sim (market replay game):** trade a basket of real tickers bar by bar with a pretend account
  against buy-and-hold, an SMA crossover bot and a Kronos bot, with fees, slippage, limit and stop
  orders. Blind mode hides dates. See [Sim](#sim-market-replay-game).
- **Token inspector:** post-hoc look at how Kronos reads the chart as tokens: per-candle surprisal,
  next-token candidates and what they historically meant, reconstruction fidelity, which candle
  properties tokens actually encode, and occlusion attribution. See [Token inspector](#token-inspector).
- **Strategy test:** turns those forecasts into trades (long if P(up) ≥ threshold, optionally short
  below 1 − threshold) and reports return, Sharpe, max drawdown, exposure and win rate against
  buy-and-hold over the same bars. See [Strategy test](#strategy-test).
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
  app/main.py                 FastAPI routes: /api/health, /api/candles, /api/forecast, /api/backtest, /api/strategy, /api/tokens, /api/journal, /api/sim/*
  app/services/market_data.py yfinance provider, SQLite cache, future-timestamp projection
  app/services/forecaster.py  Kronos loading + independent sampled paths + quantile stats
  app/services/challengers.py naive / GARCH / Chronos-Bolt forecasters for comparison
  app/services/calibration.py conformal (CQR) calibration of forecast bands
  app/services/backtest.py    walk-forward evaluation with baselines
  app/services/sim.py         market replay engine: order fills, accounts, bots, performance
  app/services/sim_session.py game sessions, Kronos bot + hints, leaderboard
  app/services/journal.py     live forecast log, scoring of closed horizons, scorecard + calibration
  app/services/tokens.py      post-hoc token analysis: surprisal, next-token, semantics, attribution
  app/services/strategy.py    forecast-driven trading simulation, buy-and-hold benchmark, permutation test
  vendor/kronos/              upstream Kronos model code (MIT), see UPSTREAM_COMMIT
frontend/
  src/App.tsx                 dashboard UI
  src/Sim.tsx                 market replay game UI
  src/charts.tsx              lightweight-charts candle/forecast/backtest charts
```

## Strategy test

The Backtest tab's strategy panel answers "would trading on these forecasts have made money?"

- **No lookahead.** The signal uses data up to the window's origin close; the fill happens at the
  next bar's open, so the gap between them belongs to the previous position.
- **Costs.** `cost bps` is charged per unit of turnover on every entry, exit and flip.
- **Benchmark.** Buy-and-hold over exactly the same bars, with the same costs.
- **Luck test.** The strategy's long/flat/short decisions are reshuffled across windows 1,000 times.
  The p-value is the share of shuffles that did at least as well. Below 0.05 suggests real timing
  skill; with 20 windows, most results will not get there.
- **Re-scoring is free.** Kronos runs once per backtest. Changing the rule calls `/api/strategy`,
  which re-simulates the stored P(up) values in milliseconds without touching the model.

Simulation code: `backend/app/services/strategy.py`.

## Comparing models

A forecast only means something next to alternatives. The sidebar's **Compare** section runs the
selected challengers on exactly the same bars, lookback, horizon and sample count as Kronos:

| Model | What it is | What it tests |
|---|---|---|
| `naive` | Historical average drift + 5-bar block-bootstrapped past returns | The no-skill yardstick (Goyal & Welch 2008). A model that can't beat it has no signal. |
| `garch` | GARCH(1,1) with Student-t shocks, simulated forward (`arch` package) | The econometrics standard for volatility, i.e. band width. |
| `chronos` | Amazon Chronos-Bolt small, zero-shot on closes | A general time-series foundation model, the kind Kronos's paper claims to beat. |

**Forecasts** overlays each model's median and band on the chart with a side-by-side table.
**Backtests** gives a league table on identical walk-forward windows: hit rate vs the majority
baseline, R²_OOS vs predicting zero, MAE vs no-change, and band coverage raw → calibrated.

Every challenger produces sampled paths in Kronos's format, so it gets the full pipeline: band
calibration, the journal (live results group by model), and the scorecard. They model the close
only, so they draw lines and bands rather than candles. Chronos-Bolt outputs quantiles 0.1–0.9,
not samples; its paths are drawn at fixed quantile levels, so tails beyond p10/p90 are clamped.

The token inspector and the Sim's Kronos bot remain Kronos-only.

## Band calibration

A p10–p90 band should contain the outcome 80% of the time. Kronos's sampled bands don't: across
SPY, NVDA, BTC-USD (daily) and AAPL (hourly), 40-window backtests found raw coverage of only
**20–54%**. The model is overconfident.

Candlecast fixes this with conformalized quantile regression (CQR; Romano, Patterson & Candès 2019):

1. Each backtest window's band and outcome are expressed as returns from the last close.
2. The nonconformity score is how far the outcome landed outside the band (negative = inside).
3. The band is widened, per forecast step, by the ⌈(n+1)·0.8⌉-th smallest score.

**Honest check.** Markets aren't exchangeable, so the guarantee isn't automatic. The Backtest tab
reports a *sequential* out-of-sample test: each window is calibrated only on earlier windows. In
the runs above, coverage moved from 31–54% raw to **77–96%** calibrated, against the 80% target.
It errs on the wide side after volatile periods.

**Where it's used.** A backtest saves its calibration per symbol, interval, model and horizon, along
with the lookback and sampling settings (band width depends on them). Forecasts with exactly
those settings draw the calibrated band and log it to the journal, whose scorecard then tracks
live calibrated coverage. If the settings differ, the forecast says so and shows the raw band.

More windows give a better calibration (the "backtest windows" control, 10–40). Next step:
adaptive conformal inference (Gibbs & Candès 2021) to update the width online as live forecasts
are scored, which should cut the over-coverage after volatility drops.

## Forecast journal

Backtests replay history, and Kronos was pre-trained on 12B+ candles that very likely include
those same periods, so a backtest can be flattered by memory. Forecasts made today about bars
that don't exist yet can't be. The Journal tab keeps that record:

- **Logged on Run.** Each forecast is saved with its settings to `backend/data/journal.sqlite3`
  (kept separate from the market-data cache; `KD_JOURNAL_PATH` to move it).
- **First forecast counts.** One entry per symbol, interval, model, horizon and origin bar.
  Re-running until the forecast looks good doesn't add entries.
- **Scored only on closed bars.** An entry is scored once its final bar has finished, measured
  from the close the model saw. Opening the Journal tab (or "score now") scores whatever is due.
- **Scorecard.** Hit rate vs majority baseline, R²_OOS, p10–p90 coverage, Brier skill of P(up)
  against a constant 50%, a permutation luck test, and a calibration chart.
- **Live vs backtest.** Per symbol, interval and model, live results sit next to the latest
  backtest. A live hit rate well below the backtest suggests the backtest was inflated.

Expect at least ~30 scored forecasts per series before reading anything into the numbers.

The backtest likewise only scores against closed candles: the in-progress bar is dropped first.

## Sim: market replay game

The Sim tab replays real history as a game. Pick a basket (up to 8 tickers, stocks and crypto can
be mixed), starting cash, length and costs, then step forward bar by bar or auto-play.

**Rules (a plain cash brokerage account):**

- You only see bars up to now. The game runs on the server, so future bars never reach the browser.
- Orders placed now are worked from the next bar. Market orders fill at the next open; limits fill
  when price trades through them (at the limit, or better on a gap); stops trigger the same way
  and fill like market orders. Market and stop fills pay slippage; every fill pays commission.
- Long only. Buys without enough cash are rejected; sells are capped at what you hold.

**Opponents** play by the same rules with the same cash: buy-and-hold (equal weight, the benchmark),
an SMA 20/50 crossover, and a Kronos bot that re-forecasts every 5 bars and holds what it rates at
P(up) ≥ 60%. You can ask Kronos for a hint from the bars visible now; hints are counted. The bot's
own readings stay hidden until the game ends, since they'd be free hints.

**Blind mode** (on by default) picks a random start, hides dates and rebases every price to 100, so
you can't trade on memory of what happened next. The real period is revealed when you finish.

Finished games go on a leaderboard with your return, buy-and-hold, the best bot and hints used.
Sessions live in `backend/data/sim.sqlite3` (`KD_SIM_PATH`) as config + frozen market snapshot +
action log; every view replays the log, so games are deterministic and survive restarts.

Caveats: prices are yfinance's dividend- and split-adjusted series; there's no shorting, margin,
partial fills or market impact. Daily bars from different exchanges are aligned on calendar date,
and hourly bars on the UTC hour.

## Token inspector

Kronos is a language model over candles. Its tokenizer z-scores each window, then a BSQ quantizer
maps every bar to a coarse **s1** token and a fine **s2** token (10 bits each, so 1,024 + 1,024),
and a causal transformer predicts the next pair. The Tokens tab applies standard LLM inspection to
that, without sampling:

| Readout | What it is |
|---|---|
| tokens used | distinct s1 / s2 tokens in the window (1,024 available) |
| perplexity | 2^(mean surprisal), teacher-forced over the window |
| surprisal overlay | −log₂ p(actual s1, s2 \| past) per candle: where the chart surprised the model |
| next entropy / candidates | the model's distribution over the next bar's s1 token |
| token p(up) | candidates weighted by what each token was followed by historically (a readout, not the forecast) |
| recon error | decoding the tokens back to prices, coarse-only vs coarse+fine |
| what a token encodes | η²: share of each candle property explained by token identity, against a chance level |
| attribution overlay | flatten one block of bars at a time and see how the next-bar outlook moves |

**Finding.** On SPY and BTC-USD daily data, s1 tokens explain about 92–94% of a candle's price
level within the window and 95–98% of its volume, but only 20–32% of its own return or body
(chance is about 5–8%). So tokens describe *where* a candle sits and how heavily it traded far
more than its shape. Combined with per-window normalization, that fits the reversion-to-the-mean
behaviour described below.

Attribution flattens contiguous blocks rather than single bars, because lagged prices are highly
collinear and per-bar attributions are unstable. Flattening is an off-distribution perturbation,
so read it as "where the model is looking", not as causal proof.

Code: `backend/app/services/tokens.py`, endpoint `/api/tokens`.

## Research notes

Methods used here and where they come from:

- **Kronos** (Shi et al., arXiv 2508.02739, AAAI 2026): the tokenizer + autoregressive model this
  app runs. The paper reports gains in RankIC and volatility MAE over other time-series foundation models.
- **Out-of-sample R² and regularized baselines:** Gu, Kelly & Xiu, *Empirical Asset Pricing via
  Machine Learning* (2020), and Kelly & Xiu, *Financial Machine Learning* (2023). The backtest
  reports R²_OOS against a zero-return forecast and a walk-forward ridge regression. Monthly R²
  for good return models is around 1% or below, so tiny positive numbers can matter and negative
  ones mean worse than predicting nothing.
- **Domain fit:** *Re(Visiting) Time Series Foundation Models in Finance* (arXiv 2511.18578) finds
  off-the-shelf TSFMs weak zero-shot on financial returns and better when pre-trained on finance
  data. Kronos is finance-pretrained, but verify per ticker with the backtest.
- **Post-hoc explanation:** perturbation-based explainers for forecasting such as PAX-TS (arXiv
  2508.18982), and guidance to attribute groups of lags rather than single lags under
  multicollinearity. The occlusion attribution follows both.
- **Explainability in general (SHAP etc.)** is standard in regulated finance. Exact SHAP over 400
  bars would need thousands of model calls, so block occlusion is the practical approximation here.

## Roadmap: multimodal signals

Kronos only sees price and volume. Recent work in financial ML combines that with text (news,
social sentiment, LLM summaries) and with images of the chart itself (CNNs on candlestick plots).
Papers to read before building this (titles as collected; not yet checked):

- *LLM-Augmented Linear Transformer–CNN for Enhanced Stock Price Prediction* (2025): GPT-4o
  technical summaries embedded with FinBERT, a linear Transformer over the price sequence, and a CNN
  over candlestick chart images. Reports lower RMSE/MAPE than single-modality baselines.
- *Integrative stock price trend prediction via hierarchical LLM text processing and patch-based
  transformer with co-attention* (2025): an LLM condenses news into short semantic cues, and
  co-attention aligns them with patched price sequences.
- *Stock Price Movement Prediction Using Sentiment Analysis and Candlestick Chart Representation*
  (2021): Twitter sentiment combined with a 1D-CNN over OHLC windows.

Possible next steps for Candlecast:

1. **News/sentiment panel:** pull headlines for the ticker and score them with FinBERT, then show
   them next to the projection without feeding them to the model yet.
2. **Sentiment as a signal:** check in the backtest whether sentiment agreeing with Kronos's P(up)
   gives a better hit rate than Kronos alone.
3. **Chart-image model:** render the lookback window as a candlestick image and score it with a
   CNN as a second opinion. Report it next to Kronos, never blended in silently.

Any new signal goes through the existing walk-forward backtest against the naive baselines before
it's trusted.

## Known behaviour

Kronos normalizes each input window by that window's mean and std, so after a strong trend it tends
to forecast reversion toward the window's average price. Longer lookbacks pull harder. Use the
Backtest tab to check whether that has paid off for a given ticker and interval before trusting it.
