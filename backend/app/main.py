from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import BacktestRequest, ForecastRequest, Interval
from app.services.backtest import run_backtest
from app.services.forecaster import MODELS, Forecaster, Timer, summarize
from app.services.market_data import CachedMarketData, MarketDataError, YFinanceProvider, future_timestamps

market = CachedMarketData(YFinanceProvider(), settings.cache_path)
forecaster = Forecaster(settings.device, settings.inference_batch_size)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in filter(None, (m.strip() for m in settings.preload_models.split(","))):
        await run_in_threadpool(forecaster.load, name)
    yield


app = FastAPI(title="Kronos Forecast Dashboard", lifespan=lifespan)
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

    return {
        "symbol": req.symbol.upper(),
        "interval": req.interval,
        "model": req.model,
        "lookback_used": lookback,
        "last_close": last_close,
        "last_time": int(ctx.index[-1].timestamp()),
        "latency_ms": round(t.ms),
        "stats": {
            "p_up": s["p_up"],
            "expected_return": s["expected_return"],
            "band_width_pct": s["band_width_pct"],
        },
        "forecast": [
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
        ],
    }


@app.post("/api/backtest")
async def backtest(req: BacktestRequest):
    df = await _load(req.symbol, req.interval)
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
    return {"symbol": req.symbol.upper(), "interval": req.interval, "model": req.model, "latency_ms": round(t.ms), **result}


# In production, serve the built frontend from the same origin as the API
if settings.frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
