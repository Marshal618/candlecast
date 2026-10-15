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


class BacktestRequest(SamplingParams):
    symbol: str = Field(..., min_length=1, max_length=20)
    interval: Interval = "1d"
    lookback: int = Field(400, ge=32, le=2048)
    horizon: int = Field(10, ge=1, le=120)
    windows: int = Field(20, ge=1, le=40)
    # Bars between window origins; defaults to horizon (non-overlapping)
    step: int | None = Field(None, ge=1)
    samples: int = Field(5, ge=1, le=50)
