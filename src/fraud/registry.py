"""Versioned model bundles.

A bundle is everything needed to serve or re-evaluate a model: the fitted
model, the exact feature list, the calibrator, the thresholds, the metrics it
was promoted on and where it came from. Stored as
``<model_dir>/registry/<name>/<version>/bundle.joblib`` with a readable
``meta.json`` next to it.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm
import sklearn
import torch

from fraud.config import get_settings


def _registry_root() -> Path:
    return get_settings().models / "registry"


def git_hash() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=get_settings().project_root,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def library_versions() -> dict:
    return {"sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__, "torch": torch.__version__}


def list_versions(name: str) -> list[str]:
    root = _registry_root() / name
    if not root.is_dir():
        return []
    return sorted((p.name for p in root.iterdir() if (p / "bundle.joblib").is_file()),
                  key=lambda v: int(v.lstrip("v")))


def save_bundle(name: str, *, model, feature_columns, calibrator=None, thresholds=None, metrics=None,
                train_days=None, protocol=None, extra=None) -> Path:
    """Write a new version of ``name`` and return its directory."""
    versions = list_versions(name)
    version = f"v{int(versions[-1].lstrip('v')) + 1 if versions else 1}"
    target = _registry_root() / name / version
    target.mkdir(parents=True, exist_ok=False)
    meta = {
        "name": name, "version": version,
        "model_class": type(model).__name__,
        "feature_columns": list(feature_columns),
        "calibration": getattr(calibrator, "method", None),
        "thresholds": thresholds, "metrics": metrics,
        "train_days": list(train_days) if train_days is not None else None,
        "protocol": protocol,
        "git_hash": git_hash(), "versions": library_versions(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    joblib.dump({"model": model, "calibrator": calibrator, **meta}, target / "bundle.joblib")
    readable = {k: v for k, v in meta.items() if k not in ("autoencoder", "reference_row")}
    (target / "meta.json").write_text(json.dumps(readable, indent=2, default=str), encoding="utf-8")
    return target


def load_bundle(name: str, version: str | None = None) -> dict:
    """Load a version of ``name`` (the latest when ``version`` is None)."""
    versions = list_versions(name)
    if not versions:
        raise FileNotFoundError(f"No registered model named {name!r} under {_registry_root()}")
    version = version or versions[-1]
    if version not in versions:
        raise FileNotFoundError(f"{name} has no version {version}; available: {versions}")
    return joblib.load(_registry_root() / name / version / "bundle.joblib")
