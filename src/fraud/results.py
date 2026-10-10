"""One place for every reported number.

``record`` merges a block into ``results/metrics.json`` together with the
script that produced it, so any figure in the dashboard or report can be
traced back to code.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fraud.config import get_settings
from fraud.registry import git_hash


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def metrics_path() -> Path:
    return get_settings().results / "metrics.json"


def load_metrics() -> dict:
    path = metrics_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record(section: str, payload: dict, source: str) -> dict:
    """Replace ``section`` in metrics.json with ``payload`` and note its source."""
    path = metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    metrics[section] = {
        "_source": source,
        "_generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "_git_hash": git_hash(),
        **_jsonable(payload),
    }
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics[section]
