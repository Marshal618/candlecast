from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All runtime config comes from env vars / .env so the same code runs locally or deployed."""

    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_prefix="KD_", extra="ignore")

    # "auto" picks cuda > mps > cpu
    device: str = "auto"
    default_model: str = "small"
    # Comma-separated models to load at startup (empty = lazy load on first request)
    preload_models: str = ""

    cache_path: Path = BACKEND_DIR / "data" / "cache.sqlite3"
    # Static frontend build to serve from FastAPI (for single-container deploys)
    frontend_dist: Path = BACKEND_DIR.parent / "frontend" / "dist"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Sequences per Kronos forward batch; lower this if you run out of memory
    inference_batch_size: int = 32


settings = Settings()
