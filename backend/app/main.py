import asyncio
import json
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import (
    BacktestRequest,
    CrossSectionRequest,
    EventStudyRequest,
    ForecastRequest,
    Interval,
    ModelName,
    PortfolioRequest,
    RecipeSave,
    SimCreate,
    SimHint,
    SimOrder,
    SimStep,
    StrategyRequest,
    TokensRequest,
    VolatilityRequest,
)
from app.services import calibration
from app.services.backtest import run_backtest
from app.services.crosssection import FACTORS, ScoreCache, build_panel, progress, run_crosssection
from app.services.fundamentals import FUNDAMENTALS, CachedFundamentals, YFinanceFundamentals
from app.services.volatility import run_volatility
from app.services.events import run_event_study
from app.services.portfolio import Construction, TrialLog, run_portfolio, study_key
from app.services.journal import Journal, forecast_entry
from app.services.sim_session import SimError, SimStore
from app.services.forecaster import MAX_CONTEXT, Forecaster, Timer, summarize
from app.services.strategy import run_strategy
from app.services.market_data import (
    CachedMarketData,
    MarketDataError,
    YFinanceProvider,
    closed_bars,
    future_timestamps,
)

market = CachedMarketData(YFinanceProvider(), settings.cache_path)
forecaster = Forecaster(settings.device, settings.inference_batch_size)
journal = Journal(settings.journal_path, market)
scores = ScoreCache(settings.research_path)
trials = TrialLog(settings.research_path)
funds = CachedFundamentals(YFinanceFundamentals(), settings.cache_path)
sims = SimStore(settings.sim_path, market, forecaster, recipes=trials.recipe, fundamentals=funds.get)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in filter(None, (m.strip() for m in settings.preload_models.split(","))):
        await run_in_threadpool(forecaster.load, name)
    yield


app = FastAPI(title="Candlecast", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _candles(df) -> list[dict]:
    return [
        {"time": int(t.timestamp()), "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
        for t, r in zip(df.index, df.itertuples(index=False))
    ]


async def _load(symbol: str, interval: str):
    try:
        return await run_in_threadpool(market.get, symbol, interval)
    except MarketDataError as e:
        raise HTTPException(404, str(e)) from e


def _check_lookback(model: str, lookback: int, available: int) -> int:
    max_ctx = MAX_CONTEXT[model]
    if lookback > max_ctx:
        raise HTTPException(422, f"lookback {lookback} exceeds {model}'s max context of {max_ctx}.")
    return min(lookback, available)


def _band_settings(req) -> dict:
    """Settings that change how wide Kronos's sampled bands come out."""
    return {"lookback": req.lookback, "samples": req.samples, "temperature": req.temperature, "top_p": req.top_p}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": forecaster.device,
        "loaded_models": forecaster.loaded_models,
        "models": {k: {"max_context": v} for k, v in MAX_CONTEXT.items()},
        "default_model": settings.default_model,
    }


@app.get("/api/candles")
async def candles(symbol: str = Query(..., max_length=20), interval: Interval = "1d", limit: int = Query(500, ge=1, le=5000)):
    df = await _load(symbol, interval)
    return {"symbol": symbol.upper(), "interval": interval, "candles": _candles(df.tail(limit))}


@app.post("/api/forecast")
async def forecast(req: ForecastRequest):
    df = await _load(req.symbol, req.interval)
    lookback = _check_lookback(req.model, req.lookback, len(df))
    if lookback < 32:
        raise HTTPException(422, f"Only {len(df)} bars of history available; need at least 32.")
    ctx = df.tail(lookback)
    future = future_timestamps(ctx.index, req.interval, req.horizon)

    with Timer() as t:
        paths = await run_in_threadpool(
            forecaster.sample_paths, [ctx], [future], req.model, req.samples, req.temperature, req.top_p
        )
    last_close = float(ctx["close"].iloc[-1])
    s = summarize(paths[0], last_close)
    times = [int(ts.timestamp()) for ts in future]

    forecast_points = [
        {
            "time": tm,
            "median": float(s["median"][i]),
            "p10": float(s["p10"][i]),
            "p90": float(s["p90"][i]),
            "open": float(s["ohlc"][i, 0]),
            "high": float(s["ohlc"][i, 1]),
            "low": float(s["ohlc"][i, 2]),
            "close": float(s["ohlc"][i, 3]),
        }
        for i, tm in enumerate(times)
    ]
    stats = {"p_up": s["p_up"], "expected_return": s["expected_return"], "band_width_pct": s["band_width_pct"]}
    # Conformally widen the band if a backtest has calibrated this series at this horizon
    cal = await run_in_threadpool(journal.get_calibration, req.symbol, req.interval, req.model, len(times))
    # Band width depends on the sampling settings, so a calibration only transfers to identical ones
    settings_now = _band_settings(req)
    cal_mismatch = cal is not None and cal.get("settings") != settings_now
    if cal_mismatch:
        cal = None
    if cal:
        lo, hi = calibration.apply(last_close, s["p10"], s["p90"], cal["q"])
        for p, a, b in zip(forecast_points, lo, hi):
            p["cal_p10"], p["cal_p90"] = float(a), float(b)
        stats["band_width_cal_pct"] = float((hi[-1] - lo[-1]) / last_close)
    origin_time = int(ctx.index[-1].timestamp())
    journal_id, created = await run_in_threadpool(
        journal.record_forecast, forecast_entry(req, last_close, origin_time, stats, forecast_points)
    )

    return {
        "symbol": req.symbol.upper(),
        "interval": req.interval,
        "model": req.model,
        "journal": {"id": journal_id, "new": created},
        "calibration": cal and {k: cal[k] for k in ("target", "windows", "oos_windows", "raw_coverage", "calibrated_coverage", "created_at")},
        "calibration_mismatch": cal_mismatch,
        "lookback_used": lookback,
        "last_close": last_close,
        "last_time": origin_time,
        "latency_ms": round(t.ms),
        "stats": stats,
        "forecast": forecast_points,
    }


@app.post("/api/backtest")
async def backtest(req: BacktestRequest):
    # Score only against finished candles; today's in-progress bar would leak a moving target
    df = closed_bars(await _load(req.symbol, req.interval), req.interval, time.time())
    _check_lookback(req.model, req.lookback, len(df))
    with Timer() as t:
        try:
            result = await run_in_threadpool(
                lambda: run_backtest(
                    forecaster,
                    df,
                    lookback=req.lookback,
                    horizon=req.horizon,
                    windows=req.windows,
                    step=req.step or req.horizon,
                    model=req.model,
                    samples=req.samples,
                    temperature=req.temperature,
                    top_p=req.top_p,
                )
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    strategy = run_strategy(df, result["windows"], req.horizon, **req.strategy.model_dump())
    await run_in_threadpool(journal.record_backtest, req.symbol, req.interval, req.model, req.horizon, result["metrics"])
    if result["calibration"]:
        result["calibration"]["settings"] = _band_settings(req)
        await run_in_threadpool(
            journal.record_calibration, req.symbol, req.interval, req.model, req.horizon, result["calibration"]
        )
    return {
        "symbol": req.symbol.upper(),
        "interval": req.interval,
        "model": req.model,
        "horizon": req.horizon,
        "latency_ms": round(t.ms),
        **result,
        "strategy": strategy,
    }


@app.post("/api/strategy")
async def strategy(req: StrategyRequest):
    df = await _load(req.symbol, req.interval)
    try:
        return run_strategy(df, [w.model_dump() for w in req.windows], req.horizon, **req.strategy.model_dump())
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/tokens")
async def tokens(req: TokensRequest):
    df = await _load(req.symbol, req.interval)
    lookback = _check_lookback(req.model, req.lookback, len(df))
    if lookback < 32:
        raise HTTPException(422, f"Only {len(df)} bars of history available; need at least 32.")
    with Timer() as t:
        result = await run_in_threadpool(forecaster.inspect, df.tail(lookback), df, req.model, req.segments)
    return {"symbol": req.symbol.upper(), "interval": req.interval, "model": req.model, "latency_ms": round(t.ms), **result}


FETCH_CONCURRENCY = 16  # Yahoo rate-limits bursts; index-sized universes fetch this many at a time
UNIVERSES = json.loads((Path(__file__).parent / "universes.json").read_text())


async def _gather_limited(fn, items, limit: int = FETCH_CONCURRENCY) -> list:
    sem = asyncio.Semaphore(limit)

    async def one(x):
        async with sem:
            try:
                return await run_in_threadpool(fn, x)
            except Exception as e:  # reported per symbol by the caller
                return e

    return await asyncio.gather(*(one(x) for x in items))


async def _universe(
    symbols: list[str], interval: str, models: list[str], lookback: int, min_symbols: int = 4
) -> tuple[dict, list[str]]:
    """Closed bars for each distinct symbol that loads, plus the ones that didn't.

    Every model is checked against the lookback first. Symbols Yahoo has no data for are skipped
    (an index list always has a few renamed or delisted tickers), as long as enough remain.
    """
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if len(symbols) < min_symbols:
        raise HTTPException(422, f"Need at least {min_symbols} distinct symbols to rank.")
    for m in models:
        _check_lookback(m, lookback, lookback)
    now = time.time()
    loaded = await _gather_limited(lambda s: market.get(s, interval), symbols)
    frames = {s: closed_bars(df, interval, now) for s, df in zip(symbols, loaded) if not isinstance(df, Exception)}
    failed = [s for s, df in zip(symbols, loaded) if isinstance(df, Exception)]
    if len(frames) < min_symbols:
        first = next(e for e in loaded if isinstance(e, Exception))
        raise HTTPException(404, f"Only {len(frames)} of {len(symbols)} symbols loaded. {first}")
    return frames, failed


async def _fundamentals(symbols, names) -> dict:
    """Point-in-time fundamentals per symbol, fetched only when some are asked for."""
    if not names:
        return {}
    loaded = await _gather_limited(funds.get, symbols, limit=8)
    return {s: f for s, f in zip(symbols, loaded) if not isinstance(f, Exception)}


@app.get("/api/universes")
def universes():
    """Index constituent lists (today's members, see scripts/update_universes.py)."""
    return UNIVERSES


@app.get("/api/progress")
def research_progress():
    """Model forecasts done / needed for the research run in flight."""
    return progress.snapshot()


@app.post("/api/crosssection")
async def crosssection(req: CrossSectionRequest):
    frames, failed = await _universe(req.symbols, req.interval, req.models, req.lookback)
    use = list(dict.fromkeys(req.fundamentals))
    fund = await _fundamentals(list(frames), use)
    with Timer() as t:
        try:
            result = await run_in_threadpool(
                lambda: run_crosssection(
                    forecaster,
                    frames,
                    interval=req.interval,
                    models=list(dict.fromkeys(req.models)),
                    lookback=req.lookback,
                    horizon=req.horizon,
                    periods=req.periods,
                    buckets=req.buckets,
                    samples=req.samples,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    cost_bps=req.cost_bps,
                    cache=scores,
                    fundamentals=fund,
                    use_fundamentals=use,
                    sectors=UNIVERSES.get("sectors"),
                )
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    return {"interval": req.interval, "horizon": req.horizon, "buckets": req.buckets, "latency_ms": round(t.ms),
            **result, "unavailable": failed}


@app.post("/api/events")
async def events(req: EventStudyRequest):
    frames, failed = await _universe(req.symbols, "1d", [], 32)
    fund = await _fundamentals(list(frames), ["surprise"])
    with Timer() as t:
        try:
            result = await run_in_threadpool(
                lambda: run_event_study(frames, fund, pre=req.pre, post=req.post, groups=req.groups)
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    return {"pre": req.pre, "post": req.post, "latency_ms": round(t.ms), **result, "unavailable": failed}


@app.post("/api/portfolio")
async def portfolio(req: PortfolioRequest):
    names = list(dict.fromkeys(s.name for s in req.signals))
    models = [n for n in names if n not in FACTORS and n not in FUNDAMENTALS]
    use = [n for n in names if n in FUNDAMENTALS]
    frames, failed = await _universe(req.symbols, req.interval, models, req.lookback)
    fund = await _fundamentals(list(frames), use)
    c = req.construction
    with Timer() as t:
        try:
            panel = await run_in_threadpool(
                lambda: build_panel(
                    forecaster, frames, interval=req.interval, models=models, lookback=req.lookback,
                    horizon=req.horizon, periods=req.periods, samples=req.samples, temperature=req.temperature,
                    top_p=req.top_p, min_history=c.risk_window + 1, cache=scores,
                    fundamentals=fund, use_fundamentals=use,
                )
            )
            result = run_portfolio(
                panel,
                {s.name: s.weight for s in req.signals},
                weighting=req.weighting,
                ic_window=req.ic_window,
                construction=Construction(**c.model_dump()),
                cost_bps=req.cost_bps,
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    # Everything that shapes the result except the universe/interval/horizon, which define the study
    config = req.model_dump(exclude={"symbols", "interval", "horizon"})
    config["signals"] = [s for s in config["signals"] if s["name"] in names]
    study = study_key(panel.symbols, req.interval, req.horizon)
    trial_id, deflation = await run_in_threadpool(trials.record, study, {**config, "horizon": req.horizon, "interval": req.interval}, result)
    leaders = await run_in_threadpool(trials.study, study)
    return {
        "interval": req.interval,
        "horizon": req.horizon,
        "latency_ms": round(t.ms),
        "trial_id": trial_id,
        "deflation": deflation,
        "study": [{"id": r["id"], "name": r["name"], "config": r["config"], "metrics": r["metrics"]} for r in leaders[:10]],
        **result,
        "dropped": list(panel.dropped),
        "unavailable": failed,
    }


@app.post("/api/volatility")
async def volatility(req: VolatilityRequest):
    models = list(dict.fromkeys(req.models))
    frames, failed = await _universe(req.symbols, req.interval, models, req.lookback, min_symbols=1)
    with Timer() as t:
        try:
            result = await run_in_threadpool(
                lambda: run_volatility(
                    forecaster, frames, interval=req.interval, models=models, lookback=req.lookback,
                    horizon=req.horizon, periods=req.periods, samples=req.samples,
                    temperature=req.temperature, top_p=req.top_p, cache=scores,
                )
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    return {"interval": req.interval, "horizon": req.horizon, "latency_ms": round(t.ms), **result, "unavailable": failed}


@app.get("/api/portfolio/recipes")
async def recipes():
    return await run_in_threadpool(trials.recipes)


@app.post("/api/portfolio/recipes")
async def save_recipe(req: RecipeSave):
    try:
        return await run_in_threadpool(trials.save, req.trial_id, req.name)
    except KeyError as e:
        raise HTTPException(404, f"No trial #{req.trial_id}.") from e


@app.get("/api/journal")
async def journal_report(
    symbol: str | None = Query(None, max_length=20),
    interval: Interval | None = None,
    model: ModelName | None = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """Score any forecasts whose bars have closed, then report the live track record."""
    await run_in_threadpool(journal.score_pending)
    return await run_in_threadpool(journal.report, symbol, interval, model, limit)


async def _sim(fn, *args):
    try:
        return await run_in_threadpool(fn, *args)
    except SimError as e:
        raise HTTPException(422, str(e)) from e
    except MarketDataError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/sim")
async def sim_leaderboard():
    return await run_in_threadpool(sims.leaderboard)


@app.post("/api/sim")
async def sim_create(req: SimCreate):
    return await _sim(sims.create, req.model_dump())


@app.get("/api/sim/{sid}")
async def sim_state(sid: str):
    return await _sim(sims.state, sid)


@app.post("/api/sim/{sid}/step")
async def sim_step(sid: str, req: SimStep):
    return await _sim(sims.step, sid, req.bars)


@app.post("/api/sim/{sid}/orders")
async def sim_order(sid: str, req: SimOrder):
    return await _sim(sims.order, sid, req.symbol, req.side, req.qty, req.type, req.price)


@app.delete("/api/sim/{sid}/orders/{order_id}")
async def sim_cancel(sid: str, order_id: int):
    return await _sim(sims.cancel, sid, order_id)


@app.post("/api/sim/{sid}/hint")
async def sim_hint(sid: str, req: SimHint):
    return await _sim(sims.hint, sid, req.symbol, req.horizon)


@app.post("/api/sim/{sid}/finish")
async def sim_finish(sid: str):
    return await _sim(sims.finish, sid)


# In production, serve the built frontend from the same origin as the API
if settings.frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
