"""First-start data: recipient history for the demo replay and the drift reports."""

import csv
import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fraud.config import get_settings
from fraud.service.db import DriftReport, Transaction
from fraud.service.settings import get_service_settings


def seed_history(db: Session) -> int:
    """Load the replay recipients' earlier transactions into the feature store (once)."""
    if db.scalar(select(func.count()).select_from(Transaction)):
        return 0
    path = Path(get_service_settings().replay_history_path)
    if not path.is_file():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{"name_dest": r["nameDest"], "step": int(r["step"]), "amount": float(r["amount"]), "source": "history"}
                for r in csv.DictReader(f)]
    db.bulk_insert_mappings(Transaction, rows)
    db.commit()
    return len(rows)


def import_drift_reports(db: Session, path=None) -> int:
    """Copy drift reports written by scripts/run_monitoring.py into the database,
    replacing any stored report for the same period."""
    path = Path(path or get_settings().results / "monitoring" / "drift_reports.json")
    if not path.is_file():
        return 0
    reports = json.loads(path.read_text(encoding="utf-8"))
    stored = {r.period: r for r in db.scalars(select(DriftReport))}
    changed = 0
    for report in reports:
        existing = stored.get(report["period"])
        if existing is None:
            db.add(DriftReport(period=report["period"], status=report["status"], report=report))
            changed += 1
        elif existing.report.get("created_at") != report.get("created_at"):
            existing.status, existing.report = report["status"], report
            changed += 1
    db.commit()
    return changed
