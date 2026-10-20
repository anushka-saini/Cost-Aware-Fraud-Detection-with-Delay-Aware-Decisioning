"""Service configuration, from environment variables (prefix ``FRAUD_``) or ``.env``."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from fraud.config import PROJECT_ROOT

DEV_SECRET = "dev-only-secret-change-me"


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    # PostgreSQL in docker-compose; a local SQLite file otherwise.
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'api' / 'fraud.db').as_posix()}"

    model_name: str = "fraud_scorer"
    model_version: str | None = None

    jwt_secret: str = DEV_SECRET
    jwt_algorithm: str = "HS256"
    token_minutes: int = 480

    # Demo accounts created on first start if the users table is empty.
    admin_username: str = "admin"
    admin_password: str = "admin123"
    analyst_username: str = "analyst"
    analyst_password: str = "analyst123"

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8501"
    rate_limit_per_minute: int = 1200          # per client address; 0 disables

    label_maturity_days: float = 2
    max_label_age_days: float | None = None    # overrides the bundle's fallback limit when set

    replay_path: str = str(PROJECT_ROOT / "api" / "data" / "replay.csv")
    replay_history_path: str = str(PROJECT_ROOT / "api" / "data" / "replay_history.csv")
    simulator_dir: str = str(PROJECT_ROOT / "results" / "simulator")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_service_settings() -> ServiceSettings:
    return ServiceSettings()
