from typing import Literal

from pydantic import BaseModel, Field

Interval = Literal["5m", "15m", "1h", "1d"]
KronosName = Literal["mini", "small", "base"]
# Kronos sizes plus challengers to compare it against (see services/challengers.py)
ModelName = Literal["mini", "small", "base", "naive", "garch", "chronos"]


class SamplingParams(BaseModel):
    model: ModelName = "small"
    samples: int = Field(10, ge=1, le=50)
    temperature: float = Field(1.0, gt=0, le=2.0)
    top_p: float = Field(0.9, gt=0, le=1.0)


class ForecastRequest(SamplingParams):
    symbol: str = Field(..., min_length=1, max_length=20)
    interval: Interval = "1d"
    lookback: int = Field(400, ge=32, le=2048)
    horizon: int = Field(20, ge=1, le=120)


class StrategyParams(BaseModel):
    mode: Literal["long_only", "long_short"] = "long_only"
    # Go long when P(up) >= enter_p; short when P(up) <= 1 - enter_p (long_short only)
    enter_p: float = Field(0.6, ge=0.5, le=1.0)
    # Fees + slippage per unit of turnover, in basis points
    cost_bps: float = Field(5.0, ge=0, le=100)


class BacktestRequest(SamplingParams):
    symbol: str = Field(..., min_length=1, max_length=20)
    interval: Interval = "1d"
    lookback: int = Field(400, ge=32, le=2048)
    horizon: int = Field(10, ge=1, le=120)
    windows: int = Field(20, ge=1, le=40)
    # Bars between window origins; defaults to horizon (non-overlapping)
    step: int | None = Field(None, ge=1)
    samples: int = Field(5, ge=1, le=50)
    strategy: StrategyParams = StrategyParams()


class StrategyWindow(BaseModel):
    origin_time: int
    p_up: float = Field(..., ge=0, le=1)


class StrategyRequest(BaseModel):
    """Re-score already-computed backtest forecasts under different rules (no model calls)."""

    symbol: str = Field(..., min_length=1, max_length=20)
    interval: Interval = "1d"
    horizon: int = Field(..., ge=1, le=120)
    windows: list[StrategyWindow] = Field(..., min_length=1, max_length=40)
    strategy: StrategyParams = StrategyParams()


class TokensRequest(BaseModel):
    model: KronosName = "small"
    symbol: str = Field(..., min_length=1, max_length=20)
    interval: Interval = "1d"
    lookback: int = Field(400, ge=32, le=2048)
    # Contiguous blocks for occlusion attribution
    segments: int = Field(16, ge=2, le=64)


class SimCreate(BaseModel):
    symbols: list[str] = Field(..., min_length=1, max_length=8)
    interval: Literal["1d", "1h"] = "1d"
    cash: float = Field(100_000, ge=1_000, le=10_000_000)
    length: int = Field(250, ge=20, le=1000)
    # "random" or an ISO date for the first playable bar
    start: str = "random"
    # Hide dates and rebase prices to 100 so you can't trade from memory of what happened
    blind: bool = True
    commission_bps: float = Field(2.0, ge=0, le=100)
    slippage_bps: float = Field(5.0, ge=0, le=100)
    bots: list[Literal["sma_cross", "kronos"]] = ["sma_cross"]
    kronos_model: KronosName = "mini"


class SimOrder(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    side: Literal["buy", "sell"]
    qty: float = Field(..., gt=0)
    type: Literal["market", "limit", "stop"] = "market"
    price: float | None = Field(None, gt=0)


class SimStep(BaseModel):
    bars: int = Field(1, ge=1, le=250)


class SimHint(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    horizon: int = Field(10, ge=1, le=60)
