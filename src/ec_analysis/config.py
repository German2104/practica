"""Application settings loaded from environment variables."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EC_", env_file=".env", extra="ignore")

    dataset_path: str = "data/portfolio_data.csv"
    target_metric: str = "npv"
    jobs_directory: str = "data/jobs"
    host: str = "127.0.0.1"
    port: int = 8000
    llm_model: str = ""
    openrouter_api_key: SecretStr | None = None
    llm_timeout_seconds: float = 60
