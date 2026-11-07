"""Public demo mode: what visitors may not do, and the daily reset.

In demo mode anyone can sign in with the demo accounts. Two things keep that safe:

1. Admin actions that would change production for everyone are refused
   (``refuse_in_demo``): switching the serving model. Threshold proposals are
   still accepted, because a proposal is only written to the audit log and
   never changes the thresholds.

2. The demo data is put back to a fresh start once a day
   (``FRAUD_DEMO_RESET_HOURS``, 24 by default). Visitors' work is removed:
   replayed and scored transactions, assessments, cases, notes, decisions,
   feedback labels and the audit log. The recipient history is seeded again.
   Users, model versions and drift reports stay.

The reset is checked on incoming requests, at most every few minutes per
worker. The time of the last reset is the newest ``demo.reset`` entry in the
audit log. On PostgreSQL a lock makes sure only one worker resets.
"""

import threading
import time
from datetime import timedelta

from fastapi import Depends, HTTPException
from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from fraud.service.db import (
    AnalystCase,
    Assessment,
    AuditLog,
    CaseNote,
    Decision,
    FeedbackLabel,
    Transaction,
    get_db,
    now,
)
from fraud.service.seed import seed_history
from fraud.service.settings import get_service_settings

RESET_ACTION = "demo.reset"
CHECK_EVERY_SECONDS = 300
LOCK_KEY = 4_271_903          # any fixed number; names the PostgreSQL lock

_lock = threading.Lock()
_checked_at = 0.0


def refuse_in_demo(what: str):
    """Stop an action that would change production for every visitor."""
    if get_service_settings().demo_mode:
        raise HTTPException(403, f"{what} is turned off in the public demo")


def last_reset(db: Session):
    return db.scalar(select(func.max(AuditLog.at)).where(AuditLog.action == RESET_ACTION))


def reset_schedule(db: Session) -> dict | None:
    """When the demo data was last reset and when it will be next (for /status)."""
    settings = get_service_settings()
    if not settings.demo_mode or settings.demo_reset_hours <= 0:
        return None
    last = last_reset(db)
    return {"every_hours": settings.demo_reset_hours,
            "last_reset_at": last.isoformat() if last else None,
            "next_reset_at": (last + timedelta(hours=settings.demo_reset_hours)).isoformat() if last else None}


def clear_demo_activity(db: Session) -> dict:
    """Remove everything visitors have done; reseed the recipient history. Does not commit."""
    removed = {}
    for table in (CaseNote, Decision, AnalystCase, Assessment, FeedbackLabel, Transaction, AuditLog):
        removed[table.__tablename__] = db.execute(delete(table)).rowcount
    db.flush()
    return removed


def reset_if_due(db: Session, current_time=None) -> bool:
    """Reset the demo data if the last reset is older than the reset interval.

    The first time it runs there is no earlier reset, so it only starts the
    clock. Returns True when the data was reset.
    """
    settings = get_service_settings()
    if not settings.demo_mode or settings.demo_reset_hours <= 0:
        return False
    current_time = current_time or now()
    with _lock:
        if db.get_bind().dialect.name == "postgresql":
            # Held until commit or rollback, so two workers cannot reset at once.
            db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": LOCK_KEY})
        last = last_reset(db)
        if last is not None and current_time - last < timedelta(hours=settings.demo_reset_hours):
            db.rollback()
            return False
        removed = clear_demo_activity(db) if last is not None else {}
        db.add(AuditLog(at=current_time, actor="system", role="system", action=RESET_ACTION,
                        entity_type="database", entity_id="demo",
                        detail={"removed": removed, "every_hours": settings.demo_reset_hours,
                                "first_start": last is None}))
        db.commit()
    if removed:
        seed_history(db)          # only seeds an empty feature store, so it runs after the commit
    return last is not None


def reset_demo_if_due(db: Session = Depends(get_db)):
    """Request dependency: check for a due reset, at most every few minutes per worker."""
    global _checked_at
    if not get_service_settings().demo_mode:
        return
    if time.monotonic() - _checked_at < CHECK_EVERY_SECONDS:
        return
    _checked_at = time.monotonic()
    reset_if_due(db)
