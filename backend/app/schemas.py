from typing import Literal

from pydantic import BaseModel, Field

Interval = Literal["5m", "15m", "1h", "1d"]
ModelName = Literal["mini", "small", "base"]


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
    model: ModelName = "small"
    symbol: str = Field(..., min_length=1, max_length=20)
    interval: Interval = "1d"
    lookback: int = Field(400, ge=32, le=2048)
    # Contiguous blocks for occlusion attribution
    segments: int = Field(16, ge=2, le=64)
