"""Service configuration, from environment variables (prefix ``FRAUD_``) or ``.env``."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from fraud.config import PROJECT_ROOT

DEV_SECRET = "dev-only-secret-change-me"
DEMO_ADMIN_PASSWORD = "admin123"
DEMO_ANALYST_PASSWORD = "analyst123"


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    # "development" or "production". In production the service refuses to
    # start with unsafe settings (see settings_problems below).
    environment: str = "development"

    # Public demo: the login page shows the demo accounts, and admin actions
    # that would change the system for every visitor (model rollback) are off.
    demo_mode: bool = False

    # With several workers the database is prepared once, before they start
    # (python -m fraud.service.setup), so each worker must skip it.
    setup_on_start: bool = True

    # PostgreSQL in docker-compose; a local SQLite file otherwise.
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'api' / 'fraud.db').as_posix()}"

    model_name: str = "fraud_scorer"
    model_version: str | None = None

    jwt_secret: str = DEV_SECRET
    jwt_algorithm: str = "HS256"
    token_minutes: int = 480

    # Demo accounts created on first start if the users table is empty.
    admin_username: str = "admin"
    admin_password: str = DEMO_ADMIN_PASSWORD
    analyst_username: str = "analyst"
    analyst_password: str = DEMO_ANALYST_PASSWORD

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8501"
    rate_limit_per_minute: int = 1200          # per client address and worker; 0 disables
    login_attempts_per_minute: int = 20        # per client address and worker; 0 disables
    max_request_bytes: int = 1_000_000         # larger request bodies are refused (413)

    label_maturity_days: float = 2
    max_label_age_days: float | None = None    # overrides the bundle's fallback limit when set

    replay_path: str = str(PROJECT_ROOT / "api" / "data" / "replay.csv")
    replay_history_path: str = str(PROJECT_ROOT / "api" / "data" / "replay_history.csv")
    simulator_dir: str = str(PROJECT_ROOT / "results" / "simulator")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def settings_problems(settings: ServiceSettings) -> list[str]:
    """Settings that are fine on a laptop but unsafe on a public server."""
    problems = []
    if settings.jwt_secret == DEV_SECRET or len(settings.jwt_secret) < 32:
        problems.append("FRAUD_JWT_SECRET must be set to a random value of at least 32 characters")
    demo_passwords = (settings.admin_password == DEMO_ADMIN_PASSWORD
                      or settings.analyst_password == DEMO_ANALYST_PASSWORD)
    if demo_passwords and not settings.demo_mode:
        problems.append("the demo passwords are in use: set FRAUD_ADMIN_PASSWORD and FRAUD_ANALYST_PASSWORD, "
                        "or FRAUD_DEMO_MODE=true for a public demo")
    if "*" in settings.cors_origin_list:
        problems.append("FRAUD_CORS_ORIGINS must list the dashboard's address, not '*'")
    return problems


def check_settings(settings: ServiceSettings) -> list[str]:
    """Stop a production start with unsafe settings; elsewhere just return the warnings."""
    problems = settings_problems(settings)
    if problems and settings.environment == "production":
        raise RuntimeError("Unsafe settings for production: " + "; ".join(problems))
    return problems


@lru_cache
def get_service_settings() -> ServiceSettings:
    return ServiceSettings()
