import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import BacktestRequest, ForecastRequest, Interval, ModelName, SimCreate, SimHint, SimOrder, SimStep, StrategyRequest, TokensRequest
from app.services import calibration
from app.services.backtest import run_backtest
from app.services.journal import Journal, forecast_entry
from app.services.sim_session import SimError, SimStore
from app.services.forecaster import MODELS, Forecaster, Timer, summarize
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
sims = SimStore(settings.sim_path, market, forecaster)


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
    max_ctx = MODELS[model].max_context
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
        "models": {k: {"max_context": v.max_context} for k, v in MODELS.items()},
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
