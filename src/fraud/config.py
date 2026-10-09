"""Project configuration.

Every path is relative to the project root unless overridden by an environment
variable (prefix ``FRAUD_``) or a ``.env`` file. Nothing in the code base
should contain an absolute path.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The repository root. When the package is installed somewhere else (a Docker
# image), FRAUD_PROJECT_ROOT says where the data, models and results live.
PROJECT_ROOT = Path(os.environ.get("FRAUD_PROJECT_ROOT") or Path(__file__).resolve().parents[2])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    project_root: Path = PROJECT_ROOT
    raw_path: Path | None = None            # PaySim CSV
    features_path: Path | None = None       # cached feature frame (parquet)
    model_dir: Path | None = None
    results_dir: Path | None = None

    seed: int = 42
    n_bootstrap: int = 200

    # Cost assumptions (illustrative, not real business figures).
    cost_false_block: float = 5.0
    cost_review: float = 1.0

    def _resolve(self, value: Path | None, default: str) -> Path:
        path = value if value is not None else Path(default)
        return path if path.is_absolute() else self.project_root / path

    @property
    def raw(self) -> Path:
        return self._resolve(self.raw_path, "data/raw/transaction_data.csv")

    @property
    def features(self) -> Path:
        return self._resolve(self.features_path, "data/processed/paysim_features.parquet")

    @property
    def models(self) -> Path:
        return self._resolve(self.model_dir, "models")

    @property
    def results(self) -> Path:
        return self._resolve(self.results_dir, "results")


@lru_cache
def get_settings() -> Settings:
    return Settings()
