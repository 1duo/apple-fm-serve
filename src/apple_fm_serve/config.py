from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APPLE_FM_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    model_id: str = "apple.fm.system"
    api_key: str | None = None
    max_concurrency: int = Field(default=4, ge=1, le=128)
    estimate_usage: bool = True
    request_timeout_s: float = Field(default=120.0, gt=0.0, le=600.0)
