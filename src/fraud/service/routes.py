"""HTTP endpoints."""

import csv
import io
import json
from datetime import UTC
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from fraud.config import PROJECT_ROOT
from fraud.features import TRANSACTION_TYPES, hour_sin_cos, to_model_frame
from fraud.labels import SOURCE_PRIORITY
from fraud.registry import list_versions
from fraud.service import schemas
from fraud.service.core import (
    SqlHistoryStore,
    assessment_response,
    audit,
    get_scorer,
    last_day,
    score_features,
    score_raw,
    state,
    sync_model_versions,
    tier_counts,
)
from fraud.service.db import (
    AnalystCase,
    Assessment,
    AuditLog,
    CaseNote,
    Decision,
    DriftReport,
    FeedbackLabel,
    ModelVersion,
    Transaction,
    User,
    get_db,
    now,
)
from fraud.service.security import admin_only, any_user, create_token, verify_password
from fraud.service.settings import get_service_settings

router = APIRouter()

VALIDATION_METRICS_PATH = PROJECT_ROOT / "api" / "validation_metrics.json"
DEMO_SAMPLE_PATH = PROJECT_ROOT / "demo_sample.csv"
CASE_SORTS = {
    "risk": AnalystCase.fraud_probability.desc(),
    "amount": AnalystCase.amount.desc(),
    "age": AnalystCase.opened_at.asc(),
    "newest": AnalystCase.opened_at.desc(),
}
DECISION_STATUS = {"approve": "approved", "decline": "declined", "escalate": "escalated"}


# ---------------------------------------------------------------------------
# Health, auth
# ---------------------------------------------------------------------------

@router.get("/health", tags=["system"])
def health():
    scorer = get_scorer()
    return {"status": "ok", "model": {"name": scorer.name, "version": scorer.version}}


@router.post("/auth/login", tags=["auth"])
def login(body: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == body.username))
    if user is None or not verify_password(body.password, user.password_hash):
        audit(db, body.username, "unknown", "auth.login_failed", "user", body.username)
        db.commit()
        raise HTTPException(401, "Incorrect username or password")
    audit(db, user.username, user.role, "auth.login", "user", user.username)
    db.commit()
    return {"access_token": create_token(user.username, user.role), "token_type": "bearer",
            "username": user.username, "role": user.role}


@router.get("/auth/me", tags=["auth"])
def me(user: User = Depends(any_user)):
    return {"username": user.username, "role": user.role}


# ---------------------------------------------------------------------------
# Scoring (called by the payment system; open, like the original API)
# ---------------------------------------------------------------------------

@router.post("/score", tags=["scoring"])
def score(txn: schemas.FeatureTransaction, db: Session = Depends(get_db)):
    try:
        return assessment_response(score_features(db, txn.model_dump()))
    except KeyError as exc:
        raise HTTPException(422, f"Missing feature: {exc}") from exc


@router.post("/score/raw", tags=["scoring"])
def score_raw_endpoint(txn: schemas.RawTransaction, db: Session = Depends(get_db)):
    """Score a raw transaction. Recipient-history features are computed here
    from the transactions this service has already stored, then the
    transaction itself is added to that history."""
    assessment = score_raw(db, txn.model_dump())
    return assessment_response(assessment, {"features": assessment.features,
                                            "transaction_id": assessment.transaction_id})


@router.post("/transactions", tags=["scoring"])
def record_transaction(record: schemas.HistoryRecord, db: Session = Depends(get_db)):
    """Add a past transaction to the recipient history without scoring it."""
    db.add(Transaction(step=record.step, amount=record.amount, name_dest=record.nameDest, source="history"))
    db.commit()
    return {"status": "recorded"}


@router.post("/explain", tags=["scoring"])
def explain(txn: schemas.FeatureTransaction):
    """Signed SHAP contribution of every feature. Creates no assessment."""
    scorer = get_scorer()
    row_dict = txn.model_dump()
    hour = row_dict["step"] if row_dict["step"] is not None else now().hour
    row_dict["hour_sin"], row_dict["hour_cos"] = (float(v) for v in hour_sin_cos(hour))
    row = to_model_frame([row_dict], scorer.base_columns)
    scored = scorer.score_frame(row).iloc[0]
    return {"fraud_probability": float(scored["fraud_probability"]), "raw_score": float(scored["raw_score"]),
            **scorer.explain(row)}


@router.post("/confirm-decision", tags=["scoring"])
def confirm_decision(entry: schemas.ConfirmDecision, db: Session = Depends(get_db)):
    """The customer's own choice after a warning: proceed anyway, or cancel."""
    assessment = db.get(Assessment, entry.assessment_id)
    if assessment is None:
        return {"error": "Unknown or expired assessment_id"}
    already = db.scalar(select(Decision).where(Decision.assessment_id == assessment.id, Decision.actor_role == "customer"))
    if already is not None:
        return {"error": "A decision is already recorded for this assessment_id"}
    action = "proceed" if entry.user_proceeded else "cancel"
    db.add(Decision(assessment_id=assessment.id, actor="customer", actor_role="customer", action=action))
    audit(db, "customer", "customer", f"customer.{action}", "assessment", assessment.id,
          {"tier": assessment.tier, "amount": assessment.amount})
    db.commit()
    return {"status": "logged"}


@router.get("/decision-log", tags=["scoring"])
def decision_log(db: Session = Depends(get_db)):
    """Customer proceed / cancel choices (the original decision log)."""
    rows = db.execute(
        select(Decision, Assessment).join(Assessment, Decision.assessment_id == Assessment.id)
        .where(Decision.actor_role == "customer").order_by(Decision.id)).all()
    return {"exists": bool(rows), "rows": [
        {"timestamp": d.created_at.isoformat(), "assessment_id": a.id, "tier": a.tier,
         "fraud_probability": a.fraud_probability, "user_proceeded": d.action == "proceed", "amount": a.amount}
        for d, a in rows]}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@router.get("/model-info", tags=["models"])
def model_info():
    scorer = get_scorer()
    first = scorer.models[0]
    params = first.get_params() if hasattr(first, "get_params") else {}
    shown = ["n_estimators", "max_depth", "min_samples_leaf", "class_weight", "num_leaves", "min_child_samples",
             "scale_pos_weight"]
    bundle = scorer.bundle
    return {**scorer.info(), "params": {k: params[k] for k in shown if k in params},
            "metrics": bundle.get("metrics"), "train_days": bundle.get("train_days"), "protocol": bundle.get("protocol"),
            "git_hash": bundle.get("git_hash"), "created_at": bundle.get("created_at"),
            "transaction_types": TRANSACTION_TYPES}


@router.get("/models", tags=["models"])
def models(db: Session = Depends(get_db), user: User = Depends(any_user)):
    sync_model_versions(db)
    rows = db.scalars(select(ModelVersion).where(ModelVersion.name == get_service_settings().model_name)).all()
    rows = sorted(rows, key=lambda m: int(m.version.lstrip("v")))
    return {"active": get_scorer().version, "versions": [
        {"name": m.name, "version": m.version, "is_active": m.is_active,
         "activated_at": m.activated_at.isoformat() if m.activated_at else None, "activated_by": m.activated_by,
         **m.meta} for m in rows]}


@router.post("/models/rollback", tags=["models"])
def rollback(body: schemas.RollbackRequest, db: Session = Depends(get_db), user: User = Depends(admin_only)):
    """Switch the serving model to another registered version (default: the one before the active one)."""
    versions = list_versions(get_service_settings().model_name)
    current = get_scorer().version
    target = body.version
    if target is None:
        position = versions.index(current)
        if position == 0:
            raise HTTPException(409, "There is no earlier version to roll back to")
        target = versions[position - 1]
    if target not in versions:
        raise HTTPException(404, f"Unknown version {target!r}; available: {versions}")
    if target == current:
        raise HTTPException(409, f"Version {target} is already active")
    state.load(target)
    sync_model_versions(db, activated_by=user.username)
    audit(db, user.username, user.role, "model.activate", "model_version", target,
          {"from": current, "to": target, "reason": body.reason})
    db.commit()
    return {"status": "activated", "from": current, "active": target}


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------

def _case_row(case: AnalystCase) -> dict:
    a, t = case.assessment, case.assessment.transaction
    return {
        "id": case.id, "status": case.status, "amount": case.amount, "fraud_probability": case.fraud_probability,
        "anomaly_score": a.anomaly_score, "tier": a.tier, "fallback_applied": a.fallback_applied,
        "opened_at": case.opened_at.isoformat(), "closed_at": case.closed_at.isoformat() if case.closed_at else None,
        "decided_by": case.decided_by, "assessment_id": a.id, "reasons": a.reasons,
        "type": t.type if t else None, "step": t.step if t else None,
        "name_orig": t.name_orig if t else None, "name_dest": t.name_dest if t else None,
        "age_seconds": (_aware(case.closed_at or now()) - _aware(case.opened_at)).total_seconds(),
    }


def _aware(dt):
    """SQLite returns naive datetimes; they were stored as UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@router.get("/cases", tags=["cases"])
def list_cases(status: str | None = Query(default="open", description="open, approved, declined, escalated or all"),
               sort: str = Query(default="risk", description="risk, amount, age or newest"),
               min_amount: float | None = None, type: str | None = None, q: str | None = None,
               limit: int = Query(default=50, ge=1, le=500), offset: int = Query(default=0, ge=0),
               db: Session = Depends(get_db), user: User = Depends(any_user)):
    if sort not in CASE_SORTS:
        raise HTTPException(422, f"sort must be one of {sorted(CASE_SORTS)}")
    query = select(AnalystCase).join(Assessment).outerjoin(Transaction, Assessment.transaction_id == Transaction.id)
    if status and status != "all":
        query = query.where(AnalystCase.status == status)
    if min_amount is not None:
        query = query.where(AnalystCase.amount >= min_amount)
    if type:
        query = query.where(Transaction.type == type)
    if q:
        like = f"%{q}%"
        query = query.where(or_(Transaction.name_dest.like(like), Transaction.name_orig.like(like),
                                AnalystCase.assessment_id.like(like)))
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    cases = db.scalars(query.order_by(CASE_SORTS[sort], AnalystCase.id).limit(limit).offset(offset)).all()
    counts = dict(db.execute(select(AnalystCase.status, func.count()).group_by(AnalystCase.status)).all())
    return {"total": total, "limit": limit, "offset": offset, "counts_by_status": counts,
            "cases": [_case_row(c) for c in cases]}


@router.get("/cases/{case_id}", tags=["cases"])
def case_detail(case_id: int, db: Session = Depends(get_db), user: User = Depends(any_user)):
    case = db.get(AnalystCase, case_id)
    if case is None:
        raise HTTPException(404, "Case not found")
    a, t = case.assessment, case.assessment.transaction
    scorer = get_scorer()

    explanation = None
    try:
        explanation = scorer.explain(to_model_frame([a.features], scorer.base_columns))
    except Exception:       # features stored by an older model version may not match the active one
        explanation = None

    history, similar = [], []
    if t is not None:
        history = [{"transaction_id": h.id, "step": h.step, "type": h.type, "amount": h.amount, "name_orig": h.name_orig}
                   for h in SqlHistoryStore(db).recent(t.name_dest, limit=20, before_id=t.id)]
        low, high = case.amount * 0.5, case.amount * 2
        closed = db.scalars(
            select(AnalystCase).join(Assessment).join(Transaction, Assessment.transaction_id == Transaction.id)
            .where(AnalystCase.id != case.id, AnalystCase.status != "open", Transaction.type == t.type,
                   AnalystCase.amount.between(low, high))
            .order_by(func.abs(AnalystCase.fraud_probability - case.fraud_probability)).limit(5)).all()
        similar = [{"id": c.id, "status": c.status, "amount": c.amount, "fraud_probability": c.fraud_probability,
                    "decided_by": c.decided_by} for c in closed]

    decisions = db.scalars(select(Decision).where(Decision.assessment_id == a.id).order_by(Decision.id)).all()
    trail = db.scalars(select(AuditLog).where(
        or_((AuditLog.entity_type == "case") & (AuditLog.entity_id == str(case.id)),
            (AuditLog.entity_type == "assessment") & (AuditLog.entity_id == a.id))).order_by(AuditLog.id)).all()
    return {
        **_case_row(case),
        "transaction": None if t is None else {
            "id": t.id, "step": t.step, "type": t.type, "amount": t.amount, "name_orig": t.name_orig,
            "name_dest": t.name_dest, "old_balance_orig": t.old_balance_orig, "new_balance_orig": t.new_balance_orig,
            "old_balance_dest": t.old_balance_dest, "new_balance_dest": t.new_balance_dest},
        "scores": {"fraud_probability": a.fraud_probability, "raw_score": a.raw_score, "anomaly_score": a.anomaly_score,
                   "t_review": a.t_review, "t_block": a.t_block, "supervised_tier": a.supervised_tier,
                   "fallback_applied": a.fallback_applied, "model_version": a.model_version,
                   "anomaly_review_threshold": scorer.staleness.get("anomaly_review_threshold")},
        "features": a.features, "explanation": explanation,
        "recipient_history": history, "similar_cases": similar,
        "notes": [{"author": n.author, "note": n.note, "created_at": n.created_at.isoformat()} for n in case.notes],
        "decisions": [{"actor": d.actor, "role": d.actor_role, "action": d.action, "note": d.note,
                       "created_at": d.created_at.isoformat()} for d in decisions],
        "audit_trail": [_audit_row(e) for e in trail],
    }


def _decide(db: Session, case: AnalystCase, action: str, note: str, user: User):
    if case.status not in ("open", "escalated"):
        raise HTTPException(409, f"Case {case.id} is already {case.status}")
    if case.status == "escalated" and user.role != "admin":
        raise HTTPException(403, "An escalated case can only be resolved by an admin")
    if case.status == "escalated" and action == "escalate":
        raise HTTPException(409, f"Case {case.id} is already escalated")
    a = case.assessment
    case.status = DECISION_STATUS[action]
    case.decided_by = user.username
    case.closed_at = None if action == "escalate" else now()
    db.add(Decision(assessment_id=a.id, actor=user.username, actor_role=user.role, action=action, note=note))
    db.add(CaseNote(case_id=case.id, author=user.username, note=f"[{action}] {note}"))
    label = None
    if action in ("approve", "decline") and a.transaction is not None:
        # The analyst's verdict becomes a training label once it has matured.
        label = FeedbackLabel(transaction_id=str(a.transaction_id), is_fraud=int(action == "decline"), source="analyst",
                              transaction_step=a.transaction.step, decided_step=a.transaction.step, note=note)
        db.add(label)
    audit(db, user.username, user.role, f"case.{action}", "case", case.id,
          {"note": note, "assessment_id": a.id, "amount": case.amount, "fraud_probability": case.fraud_probability,
           "label_recorded": label is not None})


@router.post("/cases/{case_id}/decision", tags=["cases"])
def decide_case(case_id: int, body: schemas.CaseDecision, db: Session = Depends(get_db),
                user: User = Depends(any_user)):
    case = db.get(AnalystCase, case_id)
    if case is None:
        raise HTTPException(404, "Case not found")
    _decide(db, case, body.action, body.note, user)
    db.commit()
    return {"status": case.status, "case_id": case.id, "decided_by": user.username}


@router.post("/cases/bulk-decision", tags=["cases"])
def decide_cases(body: schemas.BulkCaseDecision, db: Session = Depends(get_db), user: User = Depends(any_user)):
    done, failed = [], []
    for case_id in body.case_ids:
        case = db.get(AnalystCase, case_id)
        if case is None:
            failed.append({"case_id": case_id, "error": "not found"})
            continue
        try:
            _decide(db, case, body.action, body.note, user)
            done.append(case_id)
        except HTTPException as exc:
            failed.append({"case_id": case_id, "error": exc.detail})
    db.commit()
    return {"decided": done, "failed": failed}


@router.post("/cases/{case_id}/notes", tags=["cases"])
def add_note(case_id: int, body: schemas.CaseNoteIn, db: Session = Depends(get_db), user: User = Depends(any_user)):
    case = db.get(AnalystCase, case_id)
    if case is None:
        raise HTTPException(404, "Case not found")
    db.add(CaseNote(case_id=case.id, author=user.username, note=body.note))
    audit(db, user.username, user.role, "case.note", "case", case.id, {"note": body.note})
    db.commit()
    return {"status": "added"}


# ---------------------------------------------------------------------------
# Feedback labels
# ---------------------------------------------------------------------------

def _usable_from(label: FeedbackLabel, maturity_days: float) -> int:
    return max(label.decided_step, int(label.transaction_step + maturity_days * 24))


@router.post("/labels", tags=["labels"])
def add_label(entry: schemas.FeedbackLabelIn, db: Session = Depends(get_db), user: User = Depends(any_user)):
    """Store an outcome with both timestamps. Retraining only uses it once it has matured."""
    if entry.decided_step < entry.transaction_step:
        raise HTTPException(422, "decided_step is before transaction_step")
    label = FeedbackLabel(transaction_id=entry.transaction_id, is_fraud=int(entry.is_fraud), source=entry.source,
                          transaction_step=entry.transaction_step, decided_step=entry.decided_step, note=entry.note)
    db.add(label)
    audit(db, user.username, user.role, "label.add", "transaction", entry.transaction_id,
          {"is_fraud": entry.is_fraud, "source": entry.source})
    db.commit()
    maturity = get_service_settings().label_maturity_days
    return {"status": "stored", "label_maturity_days": maturity, "usable_for_training_from_step": _usable_from(label, maturity)}


@router.get("/labels", tags=["labels"])
def list_labels(as_of_step: int | None = None, db: Session = Depends(get_db), user: User = Depends(any_user)):
    maturity = get_service_settings().label_maturity_days
    rows = []
    for label in db.scalars(select(FeedbackLabel).order_by(FeedbackLabel.id)):
        usable = _usable_from(label, maturity)
        rows.append({"transaction_id": label.transaction_id, "is_fraud": bool(label.is_fraud), "source": label.source,
                     "source_priority": SOURCE_PRIORITY[label.source], "transaction_step": label.transaction_step,
                     "decided_step": label.decided_step, "note": label.note, "usable_for_training_from_step": usable,
                     "mature": None if as_of_step is None else usable <= as_of_step})
    return {"label_maturity_days": maturity, "as_of_step": as_of_step, "rows": rows}


# ---------------------------------------------------------------------------
# Monitoring, simulator, overview
# ---------------------------------------------------------------------------

@router.get("/monitoring/drift", tags=["monitoring"])
def monitoring_drift(db: Session = Depends(get_db), user: User = Depends(any_user)):
    """Drift reports written by scripts/run_monitoring.py, plus live operating rates."""
    reports = [r.report for r in db.scalars(select(DriftReport).order_by(DriftReport.id))]
    statuses = [r["status"] for r in reports]
    overall = "alert" if "alert" in statuses else "warning" if "warning" in statuses else "ok" if statuses else "no_reports"
    return {"status": overall, "reports": reports, "live": _live_rates(db)}


def _live_rates(db: Session) -> dict:
    counts = tier_counts(db)
    total = sum(counts.values())
    flagged = db.scalar(select(func.count()).select_from(Decision).where(Decision.actor_role == "customer")) or 0
    proceeded = db.scalar(select(func.count()).select_from(Decision)
                          .where(Decision.actor_role == "customer", Decision.action == "proceed")) or 0
    return {"assessments": total, "tier_counts": counts,
            "review_rate": counts["REVIEW"] / total if total else None,
            "block_rate": counts["BLOCK"] / total if total else None,
            "customer_decisions": flagged, "override_rate": proceeded / flagged if flagged else None}


@router.get("/simulator/scores", tags=["simulator"])
def simulator_scores(split: str = Query(default="eval_2", pattern="^(eval_2|held_out)$"),
                     user: User = Depends(any_user)):
    """Precomputed scores for the threshold simulator: every fraud case
    exactly, and legitimate transactions as a fine histogram of scores."""
    path = Path(get_service_settings().simulator_dir) / f"{split}.json"
    if not path.is_file():
        raise HTTPException(404, f"No simulator data for {split}. Run scripts/build_simulator_data.py")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/thresholds/proposals", tags=["simulator"])
def propose_thresholds(body: schemas.ThresholdProposal, db: Session = Depends(get_db),
                       user: User = Depends(admin_only)):
    """Record a proposed threshold change. This never changes the serving
    thresholds: a proposal is an audit-log entry for a later reviewed release."""
    if body.t_review > body.t_block:
        raise HTTPException(422, "t_review must not exceed t_block")
    scorer = get_scorer()
    audit(db, user.username, user.role, "thresholds.propose", "model_version", scorer.version,
          {**body.model_dump(), "current": {"t_review": scorer.t_review, "t_block": scorer.t_block}})
    db.commit()
    return {"status": "proposal_recorded", "applied": False}


@router.get("/overview", tags=["monitoring"])
def overview(db: Session = Depends(get_db), user: User = Depends(any_user)):
    scorer = get_scorer()
    today = tier_counts(db, since=last_day())
    total_today = sum(today.values())
    open_cases = db.scalar(select(func.count()).select_from(AnalystCase).where(AnalystCase.status == "open")) or 0
    escalated = db.scalar(select(func.count()).select_from(AnalystCase).where(AnalystCase.status == "escalated")) or 0
    oldest = db.scalar(select(func.min(AnalystCase.opened_at)).where(AnalystCase.status == "open"))
    live = _live_rates(db)
    reports = [r.report for r in db.scalars(select(DriftReport).order_by(DriftReport.id))]
    statuses = [r["status"] for r in reports]
    drift = "alert" if "alert" in statuses else "warning" if "warning" in statuses else "ok" if statuses else "no_reports"

    # Tier mix over time, in buckets of 500 assessments (wall-clock buckets are meaningless for a replay).
    ids = db.execute(select(Assessment.tier).order_by(Assessment.created_at, Assessment.id)).scalars().all()
    bucket, mix = max(1, len(ids) // 24) if ids else 1, []
    for start in range(0, len(ids), bucket):
        chunk = ids[start:start + bucket]
        mix.append({"from": start + 1, "to": start + len(chunk),
                    **{tier: chunk.count(tier) for tier in ("ALLOW", "REVIEW", "BLOCK")}})

    alerts = [{"source": "drift", "period": r["period"], **a} for r in reports for a in r["alerts"]]
    notable = ["assessment.block", "case.escalate", "model.activate", "thresholds.propose"]
    recent = db.scalars(select(AuditLog).where(AuditLog.action.in_(notable)).order_by(AuditLog.id.desc()).limit(10)).all()
    return {
        "kpis": {
            "scored_last_24h": total_today, "scored_total": live["assessments"],
            "block_rate": today["BLOCK"] / total_today if total_today else None,
            "review_rate": today["REVIEW"] / total_today if total_today else None,
            "review_queue": open_cases, "escalated": escalated,
            "oldest_open_case_seconds": (now() - _aware(oldest)).total_seconds() if oldest else None,
            "override_rate": live["override_rate"],
            "active_model": {"name": scorer.name, "version": scorer.version}, "drift_status": drift,
        },
        "tier_mix": mix, "drift_alerts": alerts, "recent_events": [_audit_row(e) for e in recent],
    }


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def _audit_row(e: AuditLog) -> dict:
    return {"id": e.id, "at": e.at.isoformat(), "actor": e.actor, "role": e.role, "action": e.action,
            "entity_type": e.entity_type, "entity_id": e.entity_id, "detail": e.detail, "request_id": e.request_id}


def _audit_query(actor, action, entity_type, q):
    query = select(AuditLog)
    if actor:
        query = query.where(AuditLog.actor == actor)
    if action:
        query = query.where(AuditLog.action.like(f"{action}%"))
    if entity_type:
        query = query.where(AuditLog.entity_type == entity_type)
    if q:
        like = f"%{q}%"
        query = query.where(or_(AuditLog.entity_id.like(like), AuditLog.actor.like(like), AuditLog.action.like(like)))
    return query


@router.get("/audit", tags=["audit"])
def audit_log(actor: str | None = None, action: str | None = None, entity_type: str | None = None,
              q: str | None = None, limit: int = Query(default=100, ge=1, le=1000), offset: int = Query(default=0, ge=0),
              db: Session = Depends(get_db), user: User = Depends(any_user)):
    query = _audit_query(actor, action, entity_type, q)
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    rows = db.scalars(query.order_by(AuditLog.id.desc()).limit(limit).offset(offset)).all()
    actions = [r[0] for r in db.execute(select(AuditLog.action).distinct().order_by(AuditLog.action))]
    return {"total": total, "limit": limit, "offset": offset, "actions": actions, "rows": [_audit_row(e) for e in rows]}


@router.get("/audit/export.csv", tags=["audit"])
def audit_export(actor: str | None = None, action: str | None = None, entity_type: str | None = None,
                 q: str | None = None, db: Session = Depends(get_db), user: User = Depends(any_user)):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "at", "actor", "role", "action", "entity_type", "entity_id", "detail", "request_id"])
    for e in db.scalars(_audit_query(actor, action, entity_type, q).order_by(AuditLog.id)):
        writer.writerow([e.id, e.at.isoformat(), e.actor, e.role, e.action, e.entity_type, e.entity_id,
                         json.dumps(e.detail), e.request_id])
    buffer.seek(0)
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=audit_log.csv"})


# ---------------------------------------------------------------------------
# Static project results (used by the validation and comparison pages)
# ---------------------------------------------------------------------------

@router.get("/validation-metrics", tags=["results"])
def validation_metrics():
    """Figures from the original notebooks (kept for the previous dashboard)."""
    return json.loads(VALIDATION_METRICS_PATH.read_text(encoding="utf-8"))


@router.get("/results/metrics", tags=["results"])
def results_metrics(sections: str | None = Query(default=None, description="Comma-separated section names")):
    """results/metrics.json: every reported number with the script that produced it."""
    from fraud.results import load_metrics
    metrics = load_metrics()
    if sections:
        wanted = [s.strip() for s in sections.split(",")]
        return {k: metrics[k] for k in wanted if k in metrics}
    return {"sections": sorted(metrics)}


@router.get("/results/summary", tags=["results"])
def results_summary(user: User = Depends(any_user)):
    """Old-versus-honest model comparison and the supporting experiments, for the dashboard."""
    from fraud.reporting import build_summary
    family = (get_scorer().bundle.get("protocol") or {}).get("family", "lightgbm")
    return build_summary(deployed_family=family)


@router.get("/demo-sample/random", tags=["results"])
def demo_sample_random():
    """One random held-out transaction from demo_sample.csv with its label."""
    if not DEMO_SAMPLE_PATH.is_file():
        return {"error": "demo_sample.csv not found"}
    row = pd.read_csv(DEMO_SAMPLE_PATH).sample(1)
    record = json.loads(row.to_json(orient="records"))[0]
    actual_fraud = int(record.pop("actual_fraud"))
    return {"transaction": record, "actual_fraud": actual_fraud}
