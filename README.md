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
- **Volatility test:** scores each model's sampled spread as a forecast of how much an asset will
  move (QLIKE loss, calibrated, Diebold-Mariano vs an EWMA baseline). See [Volatility test](#volatility-test).
- **Fundamentals:** earnings surprise, analyst revisions, earnings yield and size, rebuilt
  point-in-time so they can be backtested in Research and blended in the Portfolio lab. See
  [Non-price signals](#non-price-signals).
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
  app/main.py                 FastAPI routes: /api/health, /api/candles, /api/forecast, /api/backtest, /api/strategy, /api/tokens, /api/journal, /api/crosssection, /api/volatility, /api/universes, /api/progress, /api/portfolio, /api/sim/*
  app/services/market_data.py yfinance provider, SQLite cache, future-timestamp projection
  app/services/forecaster.py  Kronos loading + independent sampled paths + quantile stats
  app/services/challengers.py naive / GARCH / Chronos-Bolt forecasters for comparison
  app/services/calibration.py conformal (CQR) calibration of forecast bands
  app/services/backtest.py    walk-forward evaluation with baselines
  app/services/crosssection.py universe ranking research: RankIC, bucket and long-short portfolios, factors
  app/universes.json          S&P 500 / 400 / 600 constituents (refresh: scripts/update_universes.py)
  app/services/fundamentals.py point-in-time earnings surprise, analyst revisions, earnings yield, size
  app/services/volatility.py  volatility forecast test: QLIKE, calibration, Diebold-Mariano vs EWMA
  app/services/portfolio.py   portfolio lab: signal blend, Ledoit-Wolf risk, construction, deflated Sharpe
  app/services/sim.py         market replay engine: order fills, accounts, bots, performance
  app/services/sim_session.py game sessions, Kronos bot + hints, leaderboard
  app/services/journal.py     live forecast log, scoring of closed horizons, scorecard + calibration
  app/services/tokens.py      post-hoc token analysis: surprisal, next-token, semantics, attribution
  app/services/strategy.py    forecast-driven trading simulation, buy-and-hold benchmark, permutation test
  vendor/kronos/              upstream Kronos model code (MIT), see UPSTREAM_COMMIT
frontend/
  src/App.tsx                 dashboard UI
  src/Sim.tsx                 market replay game UI
  src/Research.tsx            cross-sectional research UI
  src/Volatility.tsx          volatility test UI
  src/Portfolio.tsx           portfolio lab UI
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

## Cross-sectional research

The **Research** tab asks the question quant funds actually trade on: not "will this asset go up?"
but "which of these assets will do better than the others?" A model can be useless for timing
and still rank well, and a ranking can be traded market-neutral.

Every `horizon` bars (non-overlapping), each selected model forecasts every symbol in the universe
from data up to that close, and the universe is ranked by the median sampled return. Two classic
price factors are ranked alongside with no model calls: **momentum** (60-bar return skipping the
last 5, Jegadeesh & Titman 1993) and **short-term reversal** (minus the last `horizon`-bar return,
Lehmann 1990). Each ranking is scored against realized forward returns:

- **RankIC**: Spearman correlation between ranking and returns, per rebalance (the Kronos paper's
  headline metric), with ICIR and a t-stat. |t| > 2 is the usual "probably not luck" bar.
- **Buckets**: equal-weight returns from lowest-ranked to highest. A real signal climbs.
- **Long-short**: long the top bucket, short the bottom, net of costs per unit of turnover, with
  Sharpe, drawdown and the share of the book traded each rebalance.
- **ρ reversal / ρ momentum**: how much a model's ranking is just a known factor.
- **Beyond factors**: RankIC of what's left after regressing the factors' ranks out each period.

Presets cover sector ETFs, mega caps, country ETFs and crypto. Symbols are aligned by calendar
date, and rebalances use days every symbol traded.

First results (Kronos-mini, lookback 64, 8 paths, 5 bps):

| Universe, rebalance | kronos-mini RankIC (t) | beyond factors (t) | ρ reversal | reversal RankIC (t) | best challenger |
|---|---|---|---|---|---|
| 11 sector ETFs, 5d × 24 | +0.09 (1.3) | −0.01 (−0.2) | 0.34 | +0.16 (2.1) | chronos +0.03 |
| 20 mega caps, 5d × 40 | +0.10 (2.8) | +0.07 (2.1) | 0.25 | +0.09 (2.2) | naive +0.04 |
| 20 mega caps, 20d × 30 | +0.08 (1.4) | +0.04 (0.9) | 0.34 | +0.08 (1.5) | chronos −0.01 |

Kronos-mini ranks better than every challenger in all three (RankIC +0.08 to +0.10), but much of
that is short-term reversal, which its per-window normalization builds in. Only on mega caps at
5-day rebalances does anything survive once the factors are removed (residual t = 2.1), and the
plain reversal factor does about as well for free. These are small samples (24–40 rebalances), so
treat them as a starting point, not a verdict.

## Non-price signals

Kronos only sees the chart. The Research tab can rank four signals it can't see, each rebuilt
**point-in-time** from data yfinance dates event by event, so a rebalance only uses what was public
by then:

| Signal | Definition | Source paper |
|---|---|---|
| earnings surprise | latest reported EPS vs consensus (%), if reported in the last 100 days | Bernard & Thomas 1989 (post-earnings drift) |
| analyst revisions | upgrades and target raises minus downgrades and cuts, last 90 days, averaged | Womack 1996 |
| earnings yield | last four reported quarters of EPS / price | Basu 1977 |
| small size | minus log market cap (split-adjusted shares × price) | Banz 1981 |

Earnings count from the bar they were public by (before-open reports the same day, after-close the
next). Analyst actions and share counts carry dates without time zones, so they count from the next
day. ETFs, crypto and indexes have no fundamentals; they, and any other gaps, sit at the median of
each rebalance's ranking, and the table shows each signal's coverage. They work in the Portfolio lab
blend and in Sim portfolio-bot recipes too (blind games keep the real prices for this).

**No news.** yfinance only returns the last few days of headlines, so a news or sentiment signal
can't be backtested without look-ahead. That needs a paid historical archive.

First run, 20 mega caps, all four fundamentals next to Kronos-mini (lookback 64, 8 paths, 5 buckets):

| Signal | 5d × 40 RankIC (t) | 20d × 60 RankIC (t) | ρ momentum (20d) |
|---|---|---|---|
| kronos-mini | +0.10 (2.8) | +0.08 (2.3) | −0.17 |
| earnings surprise | −0.07 (−1.8) | −0.02 (−0.4) | +0.14 |
| analyst revisions | −0.03 (−0.9) | +0.01 (0.2) | +0.52 |
| earnings yield | +0.01 (0.1) | −0.00 (−0.0) | −0.15 |
| small size | −0.02 (−0.4) | +0.01 (0.3) | −0.11 |

None of the four shows anything in mega caps, at either horizon, over five years of monthly
rebalances. That matches the literature more than it contradicts it: these anomalies are strongest
in small, thinly covered stocks and have shrunk since publication (McLean & Pontiff 2016), and 20
names is a small cross-section with little power. Analyst revisions are half momentum (ρ 0.52):
analysts chase price. Wider index universes are below.

## Large universes

Twenty names can't show a cross-sectional effect of the size these anomalies have. The universe
picker in Research, Volatility and Portfolio also lists the **S&P 500, S&P 400 and S&P 600**
(today's constituents from Wikipedia, shipped in `app/universes.json`; refresh with
`python scripts/update_universes.py`). Up to 700 symbols per run:

- Prices load 16 at a time, fundamentals 8. Symbols that fail to load are listed as
  **unavailable**, and the run goes ahead without them.
- Symbols listed too recently to cover every rebalance are **dropped**, rather than shortening the
  whole study to the youngest name's history.
- Model forecasts run and cache in chunks of 512 contexts, so an interrupted run picks up where it
  stopped. The run button shows a worst-case time, and `/api/progress` drives a live counter.
  Kronos-mini samples about 900 bars per second on an M3 (≈170 paths/s at a 5-day horizon, ≈47 at
  20 days).

**Survivorship bias.** These lists are *today's* members, so every past rebalance only ranks companies
that later survived or were promoted into the index. Names that collapsed and were removed are
missing. That biases anything that leans toward small or distressed stocks upward. The app flags
it for any universe over 30 names. Point-in-time membership needs a paid source (CRSP, Compustat,
Norgate).

Monthly rebalances (20-day horizon, 60 periods, 2022-01 to 2026-09, 5 buckets):

| Signal | S&P 500 RankIC (t) | residual t | L/S Sharpe | S&P 600 RankIC (t) | residual t | L/S Sharpe |
|---|---|---|---|---|---|---|
| kronos-mini (64, 8 paths) | +0.021 (1.14) | 1.57 | 0.41 | not run | | |
| earnings surprise | +0.020 (1.82) | **2.16** | 0.74 | +0.011 (1.53) | 1.29 | 0.36 |
| analyst revisions | +0.003 (0.18) | 1.02 | 0.04 | +0.001 (0.08) | −0.23 | −0.03 |
| earnings yield | +0.016 (0.88) | 0.72 | −0.02 | +0.017 (1.10) | 1.02 | 0.13 |
| small size | +0.009 (0.67) | 0.10 | 0.92 | **+0.073 (6.43)** | **6.63** | **3.43** |
| momentum (60-5 bars) | −0.016 (−0.72) | | −0.21 | +0.004 (0.23) | | 0.02 |
| reversal | −0.004 (−0.18) | | −0.18 | +0.007 (0.49) | | 0.12 |

The S&P 500 run covered 491 symbols (12 dropped). The S&P 600 run covered 568 symbols (34 dropped,
one unavailable). Factors and fundamentals alone take about 5 minutes. Kronos-mini on the S&P 500
took 84 minutes (29,460 forecasts × 8 paths).

What it says:

- **Earnings surprise is the one plausible signal in large caps.** After removing momentum and
  reversal its t is 2.16, and it is weaker but the same sign in small caps. That fits post-earnings
  drift. It is borderline, so treat it as worth watching rather than as a result.
- **Kronos-mini doesn't hold up at scale.** Its +0.08 RankIC on 20 mega caps becomes +0.02 (t 1.1)
  across the S&P 500, and it is mostly short-term reversal (ρ +0.45).
- **Small size in the S&P 600 is the big number, and the least trustworthy.** It is steady across
  the sample (RankIC +0.086, +0.064, +0.067 by thirds), which is what survivorship would produce: the
  smallest of today's members are exactly the ones that grew into the index or narrowly survived.
  Expect most of it to vanish with point-in-time membership.
- Revisions are still mostly momentum (ρ 0.52 in the S&P 500).

## Volatility test

The **Volatility** tab asks the question price history *can* answer: not which way, but how much.
Volatility clusters (Engle 1982), so a model can be useless for direction and still useful for
risk. Every `horizon` bars, each model's spread of sampled paths (variance of the horizon log
return) is scored as a forecast of the next `horizon` bars' realized variance (sum of squared
per-bar log returns, Andersen & Bollerslev 1998), next to a no-model **EWMA** baseline (λ = 0.94,
RiskMetrics):

- **QLIKE** loss (Patton 2011), raw and **calibrated**: each period's forecasts rescaled by the
  average realized/forecast ratio of earlier periods only, so a constant scale error isn't punished.
- **Skill vs EWMA** (1 − calibrated QLIKE ratio) and a **Diebold-Mariano** t-stat on the per-period
  loss gap.
- **Mincer-Zarnowitz** R² and slope of log realized on log forecast (scale-free).
- **Ranking**: RankIC across symbols (which names will be wildest) and correlation over time within
  each symbol (when one will get wilder).
- **Vol ratio**: realized / raw forecast vol. Above 1 means the bands are too narrow.

The forecasts come from the same sampled paths the Research tab ranks with and share its cache.

First results, 20 mega caps, 5-day horizon, 120 rebalances (117 scored), 16 paths:

| Forecast (lookback) | vol ratio | calibrated QLIKE | skill vs EWMA (DM t) | MZ R² | rank IC | timing |
|---|---|---|---|---|---|---|
| ewma baseline | 1.11× | 0.557 | — | 0.24 | 0.49 | 0.16 |
| chronos-bolt (256) | 1.33× | 0.530 | **+4.9% (+2.2)** | 0.27 | 0.51 | 0.20 |
| kronos-mini (256) | 1.90× | 0.618 | −10.8% (−2.9) | 0.20 | 0.42 | 0.15 |
| kronos-mini (64) | 2.23× | 0.663 | −18.9% (−5.9) | 0.18 | 0.40 | 0.13 |
| garch(1,1)-t (500) | 1.25× | 0.657 | −17.8% (−3.8) | 0.20 | 0.44 | 0.10 |
| garch(1,1)-t (500), 50 paths | 1.17× | 0.600 | −7.6% (−1.8) | 0.23 | 0.50 | 0.11 |

Same universe, last 40 rebalances only (37 scored), lookback 256, to see what more paths buy Kronos:

| Forecast | vol ratio | calibrated QLIKE | skill vs EWMA (DM t) | rank IC |
|---|---|---|---|---|
| ewma baseline | 1.06× | 0.442 | — | 0.46 |
| chronos-bolt | 1.27× | 0.405 | **+8.5% (+3.6)** | 0.47 |
| kronos-mini, 16 paths | 1.84× | 0.478 | −8.2% (−1.8) | 0.36 |
| kronos-mini, 50 paths | 1.75× | 0.441 | +0.2% (+0.1) | 0.41 |

What it says:

- **Kronos's raw spread is about half the real volatility** (vol ratio 1.9–2.2×, worse at longer
  horizons: 3.2× at 20 days). That's the same too-narrow bands the conformal calibration fixes.
- **Calibrated, Kronos at best ties a plain EWMA**: it ranks which stocks will be wildest
  (rank IC 0.36–0.42) but loses to EWMA with 16 paths and a short lookback, and only draws level
  with 50 paths and 256 bars (about 15 minutes of forecasts, vs instant for EWMA).
- **Chronos-Bolt is the only model that beats EWMA**, with a 256-bar context.
- **Sampling noise is a real handicap**: a variance from 16 paths has about ±37% error. GARCH's gap
  to EWMA more than halves going from 16 to 50 paths. Chronos predicts quantiles directly, so it has
  none. Use as many paths as you can afford when comparing sampled models.

## Portfolio lab

The **Portfolio** tab turns Research-tab signals into a portfolio you could hold, walk-forward:

1. **Blend.** Pick any of Kronos, the challengers, momentum, reversal and the fundamentals, each with a weight
   (negative flips a signal). Scores become z-scored cross-sectional ranks so different scales mix
   fairly. "By trailing IC" weights each signal by its positive RankIC over past rebalances only.
2. **Risk model.** Covariance of per-bar returns over a trailing window, shrunk toward a scaled
   identity with the Ledoit & Wolf (2004) optimal intensity.
3. **Construction.** Equal weight or inverse volatility on the top (and bottom) share of names, or
   mean-variance (Σ⁻¹α, dollar-neutral for long-short). Long-short or long-only.
4. **Sizing and trading.** Scaled to an annual volatility target, capped at a max gross exposure,
   and moved only part of the way toward the target each rebalance (Gârleanu & Pedersen 2013),
   which gives up some signal for much lower turnover.

**Overfitting guard.** Every distinct configuration is logged as a trial for its universe,
interval and horizon. The result shows the Deflated Sharpe Ratio (Bailey & López de Prado 2014):
the probability that the Sharpe beats what the best of that many tries would reach by luck. The
more variants you try, the higher the bar.

**Forecast cache.** Model scores are cached per symbol, rebalance date and settings
(`KD_RESEARCH_PATH`), shared with the Research tab. Only the first run pays for Kronos; changing
construction afterwards is instant, and longer studies (up to 120 rebalances) only forecast new dates.

**Into the Sim.** Save a run as a recipe, then pick "portfolio bot" when starting a Sim game. It
runs the recipe's final blend weights and construction on the game's basket, long-only (the Sim is
a cash account), re-deciding every recipe-horizon bars from visible history only.

First runs on the 20 mega caps (60 weekly rebalances, 10% vol target, 5 bps, inverse vol unless noted):

| Recipe | Ann. return | Sharpe | Max DD | Turnover | DSR (trials) |
|---|---|---|---|---|---|
| reversal, equal weight | 17.7% | 1.63 | −10.3% | 75% | 0.96 (1) |
| kronos-mini | 12.1% | 1.27 | −4.9% | 79% | 0.89 (3) |
| kronos-mini + reversal | 17.7% | 1.63 | −7.5% | 81% | 0.94 (4) |
| mini + reversal + momentum, IC-weighted | 21.8% | 2.03 | −3.5% | 72% | 0.98 (5) |
| mini + reversal, trade rate 0.5 | 8.8% | 1.47 | −2.7% | 35% | 0.89 (7) |

60 rebalances is a little over a year, all in one market regime, so these numbers say the
machinery works, not that the strategy does. IC weighting ended with about 0.44 Kronos-mini,
0.56 reversal and no momentum.

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
   them next to the projection without feeding them to the model yet. Backtesting it needs a
   historical news archive; yfinance only has the last few days (see [Non-price signals](#non-price-signals)).
2. **Sentiment as a signal:** check in the backtest whether sentiment agreeing with Kronos's P(up)
   gives a better hit rate than Kronos alone.
3. **Chart-image model:** render the lookback window as a candlestick image and score it with a
   CNN as a second opinion. Report it next to Kronos, never blended in silently.

Any new signal goes through the existing walk-forward backtest against the naive baselines before
it's trusted.

## Known behaviour

Kronos normalizes each input window by that window's mean and std, so after a strong trend it tends
to forecast reversion toward the window's average price. Longer lookbacks pull harder: in a
40-window comparison on SPY, NVDA and BTC-USD, Kronos-mini's R²_OOS was −23% at 64 bars, −41% at 128
and −190% at 400, which is why the default lookback is 64. Use the Backtest tab to check whether
that has paid off for a given ticker and interval before trusting it.

Kronos sampling is seeded (`SEED` in `forecaster.py`), so the same inputs give the same paths and
backtests reproduce. Before seeding, two identical 8-path research runs gave RankIC 0.036 and 0.081.
