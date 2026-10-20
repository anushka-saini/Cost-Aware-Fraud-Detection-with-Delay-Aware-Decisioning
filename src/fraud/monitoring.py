"""Drift monitoring: PSI per feature and per score, and operating-rate bands.

``build_report`` compares one period of traffic with the training reference.
Reports are appended to a JSON file by ``DriftReportStore`` and served by
``GET /monitoring/drift``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.evaluation import psi
from fraud.features import BINARY_FEATURES

PSI_WARNING, PSI_ALERT = 0.10, 0.25
# Operating rates may move this far (as a multiple of the expected rate) before an alert is raised.
RATE_BAND = (0.5, 2.0)
OVERRIDE_RATE_BAND = (0.0, 0.30)


def psi_level(value: float) -> str:
    if value > PSI_ALERT:
        return "alert"
    if value >= PSI_WARNING:
        return "warning"
    return "ok"


def binary_psi(reference, current) -> float:
    """PSI of a 0/1 feature: two bins, the share of ones and the share of zeros."""
    p, q = float(np.mean(reference)), float(np.mean(current))
    p, q = np.clip([p, 1 - p], 1e-6, None), np.clip([q, 1 - q], 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame, columns) -> list[dict]:
    rows = []
    for column in columns:
        ref, cur = reference[column].to_numpy(dtype=np.float64), current[column].to_numpy(dtype=np.float64)
        value = binary_psi(ref, cur) if column in BINARY_FEATURES else psi(ref, cur)
        rows.append({"name": column, "kind": "feature", "psi": value, "level": psi_level(value)})
    return rows


def rate_check(name: str, observed: float, expected: float | None, band=RATE_BAND, absolute: bool = False) -> dict:
    """Is an operating rate inside its expected band?

    Relative band (default): expected * band. Absolute band: the band itself.
    """
    if absolute:
        low, high = band
    elif expected is None:
        return {"name": name, "observed": observed, "expected": None, "level": "unknown"}
    else:
        low, high = expected * band[0], expected * band[1]
    return {"name": name, "observed": float(observed), "expected": expected, "band_low": float(low),
            "band_high": float(high), "level": "ok" if low <= observed <= high else "alert"}


def build_report(*, period: str, days, reference: pd.DataFrame, current: pd.DataFrame, feature_columns,
                 reference_scores: dict, current_scores: dict, review_rate: float | None = None,
                 expected_review_rate: float | None = None, block_rate: float | None = None,
                 expected_block_rate: float | None = None, override_rate: float | None = None,
                 model: dict | None = None) -> dict:
    """One drift report: a period of traffic against the training reference.

    ``*_scores`` map a score name (e.g. "model_score", "anomaly_score") to an array.
    """
    psi_rows = feature_psi(reference, current, feature_columns)
    for name, ref in reference_scores.items():
        value = psi(ref, current_scores[name])
        psi_rows.append({"name": name, "kind": "score", "psi": value, "level": psi_level(value)})

    rates = []
    if review_rate is not None:
        rates.append(rate_check("review_rate", review_rate, expected_review_rate))
    if block_rate is not None:
        rates.append(rate_check("block_rate", block_rate, expected_block_rate))
    if override_rate is not None:
        rates.append(rate_check("override_rate", override_rate, None, OVERRIDE_RATE_BAND, absolute=True))

    alerts = [{"type": "psi", "name": r["name"], "level": r["level"], "value": r["psi"]}
              for r in psi_rows if r["level"] != "ok"]
    alerts += [{"type": "rate", "name": r["name"], "level": r["level"], "value": r["observed"]}
               for r in rates if r["level"] == "alert"]
    worst = "alert" if any(a["level"] == "alert" for a in alerts) else "warning" if alerts else "ok"
    return {
        "period": period, "days": list(days), "reference": "train (days 5-7)",
        "n_rows": int(len(current)), "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model, "status": worst,
        "thresholds": {"psi_warning": PSI_WARNING, "psi_alert": PSI_ALERT, "rate_band": list(RATE_BAND),
                       "override_rate_band": list(OVERRIDE_RATE_BAND)},
        "psi": psi_rows, "rates": rates, "alerts": alerts,
    }


class DriftReportStore:
    """Drift reports in a JSON file (one list, newest last)."""

    def __init__(self, path):
        self.path = Path(path)

    def all(self) -> list[dict]:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def add(self, report: dict) -> dict:
        reports = [r for r in self.all() if r["period"] != report["period"]]
        reports.append(report)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        return report

    def latest(self) -> dict | None:
        reports = self.all()
        return reports[-1] if reports else None
