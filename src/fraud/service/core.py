"""Shared service state and the operations several routers need:
the active scorer, the SQL feature store, scoring-and-recording, and the audit log."""

import uuid
from contextvars import ContextVar
from datetime import timedelta

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fraud.features import compute_online_features, hour_sin_cos, to_model_frame
from fraud.registry import list_versions, load_bundle
from fraud.service.db import AnalystCase, Assessment, AuditLog, ModelVersion, Transaction, now
from fraud.service.settings import get_service_settings
from fraud.serving import Scorer

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


# ---------------------------------------------------------------------------
# Active model
# ---------------------------------------------------------------------------

class ModelState:
    """The scorer currently serving. Swapped atomically on activate / rollback."""

    def __init__(self):
        self.scorer: Scorer | None = None

    def load(self, version: str | None = None) -> Scorer:
        settings = get_service_settings()
        scorer = Scorer(load_bundle(settings.model_name, version))
        if settings.max_label_age_days is not None:
            scorer.staleness["max_label_age_days"] = settings.max_label_age_days
        self.scorer = scorer
        return scorer


state = ModelState()


def get_scorer() -> Scorer:
    if state.scorer is None:
        state.load(get_service_settings().model_version)
    return state.scorer


def sync_model_versions(db: Session, activated_by: str = "system"):
    """Mirror the on-disk registry into the model_versions table and make sure
    exactly one version is marked active (the one that is loaded)."""
    settings = get_service_settings()
    scorer = get_scorer()
    known = {m.version: m for m in db.scalars(select(ModelVersion).where(ModelVersion.name == settings.model_name))}
    for version in list_versions(settings.model_name):
        if version not in known:
            bundle = load_bundle(settings.model_name, version)
            meta = {k: bundle.get(k) for k in ("metrics", "thresholds", "train_days", "protocol", "git_hash",
                                               "created_at", "calibration", "staleness", "feature_columns")}
            known[version] = ModelVersion(name=settings.model_name, version=version, meta=meta)
            db.add(known[version])
    active = [m for m in known.values() if m.is_active]
    if len(active) != 1 or active[0].version != scorer.version:
        for m in known.values():
            m.is_active = m.version == scorer.version
            if m.is_active:
                m.activated_at, m.activated_by = now(), activated_by
    db.commit()


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------

class SqlHistoryStore:
    """Recipient history from the transactions table (index on name_dest, step)."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, destination: str, before_id: int | None = None):
        query = select(Transaction.step, Transaction.amount).where(Transaction.name_dest == destination)
        if before_id is not None:
            query = query.where(Transaction.id < before_id)
        rows = self.db.execute(query.order_by(Transaction.id)).all()
        return [r[0] for r in rows], [r[1] for r in rows]

    def recent(self, destination: str, limit: int = 20, before_id: int | None = None) -> list[Transaction]:
        query = select(Transaction).where(Transaction.name_dest == destination)
        if before_id is not None:
            query = query.where(Transaction.id < before_id)
        return list(self.db.scalars(query.order_by(Transaction.id.desc()).limit(limit)))


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit(db: Session, actor: str, role: str, action: str, entity_type: str, entity_id, detail: dict | None = None):
    db.add(AuditLog(actor=actor, role=role, action=action, entity_type=entity_type, entity_id=str(entity_id),
                    detail=detail or {}, request_id=request_id_var.get()))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _record(db: Session, scorer: Scorer, row: pd.DataFrame, scored, *, amount: float, features: dict,
            transaction: Transaction | None, actor: str) -> Assessment:
    tier = scored["tier"]
    reasons = scorer.top_reasons(row) if tier != "ALLOW" else []
    if scored["fallback_applied"]:
        reasons = ["this transaction looking unlike ordinary traffic (the fraud model's labels are out of date, "
                   "so the anomaly detector is being relied on)"]
    assessment = Assessment(
        id=f"a-{uuid.uuid4().hex[:20]}", transaction_id=transaction.id if transaction else None,
        model_name=scorer.name, model_version=scorer.version, amount=float(amount),
        raw_score=float(scored["raw_score"]), fraud_probability=float(scored["fraud_probability"]),
        anomaly_score=float(scored["anomaly_score"]) if "anomaly_score" in scored else None,
        tier=tier, supervised_tier=scored["supervised_tier"], fallback_applied=bool(scored["fallback_applied"]),
        t_review=scorer.t_review, t_block=scorer.t_block, reasons=reasons,
        features={k: (v.item() if hasattr(v, "item") else v) for k, v in features.items()},
    )
    assessment.model_stale = bool(scored["model_stale"])      # not stored: derivable from the model version
    db.add(assessment)
    if tier == "REVIEW":
        db.flush()
        db.add(AnalystCase(assessment_id=assessment.id, amount=float(amount),
                           fraud_probability=assessment.fraud_probability))
    if tier != "ALLOW":
        audit(db, actor, "system", f"assessment.{tier.lower()}", "assessment", assessment.id,
              {"fraud_probability": assessment.fraud_probability, "amount": float(amount),
               "model_version": scorer.version, "fallback_applied": assessment.fallback_applied})
    db.commit()
    return assessment


def score_raw(db: Session, txn: dict, *, source: str = "scored", true_label: int | None = None,
              actor: str = "scoring-api") -> Assessment:
    """Score a raw transaction: look up the recipient's history, compute the
    features, score, then store the transaction so it becomes history."""
    scorer = get_scorer()
    steps, amounts = SqlHistoryStore(db).get(txn["nameDest"])
    features = compute_online_features(
        history_steps=steps, history_amounts=amounts, step=txn["step"], type=txn["type"], amount=txn["amount"],
        old_balance_origin=txn["oldbalanceOrg"], new_balance_origin=txn["newbalanceOrig"],
        old_balance_destination=txn["oldbalanceDest"], new_balance_destination=txn["newbalanceDest"],
    )
    row = to_model_frame([features], scorer.base_columns)
    scored = scorer.score_frame(row, current_day=int(txn["step"]) // 24).iloc[0]
    transaction = Transaction(
        step=txn["step"], type=txn["type"], amount=txn["amount"], name_orig=txn.get("nameOrig"),
        name_dest=txn["nameDest"], old_balance_orig=txn["oldbalanceOrg"], new_balance_orig=txn["newbalanceOrig"],
        old_balance_dest=txn["oldbalanceDest"], new_balance_dest=txn["newbalanceDest"],
        source=source, true_label=true_label,
    )
    db.add(transaction)
    db.flush()
    return _record(db, scorer, row, scored, amount=txn["amount"], features=features, transaction=transaction,
                   actor=actor)


def score_features(db: Session, payload: dict, actor: str = "scoring-api") -> Assessment:
    """Score pre-computed features (the original request format)."""
    scorer = get_scorer()
    row_dict = dict(payload)
    step = row_dict.get("step")
    hour_source = step if step is not None else now().hour
    row_dict["hour_sin"], row_dict["hour_cos"] = (float(v) for v in hour_sin_cos(hour_source))
    row = to_model_frame([row_dict], scorer.base_columns)
    scored = scorer.score_frame(row, current_day=step // 24 if step is not None else None).iloc[0]
    features = {c: row_dict[c] for c in scorer.base_columns}
    return _record(db, scorer, row, scored, amount=payload["amount"], features=features, transaction=None, actor=actor)


def assessment_response(a: Assessment, extra: dict | None = None) -> dict:
    body = {
        "assessment_id": a.id, "fraud_probability": a.fraud_probability, "raw_score": a.raw_score,
        "tier": a.tier, "reasons": a.reasons,
        "thresholds_used": {"t_review": a.t_review, "t_block": a.t_block},
        "model": {"name": a.model_name, "version": a.model_version},
    }
    if a.anomaly_score is not None:
        body.update(anomaly_score=a.anomaly_score, supervised_tier=a.supervised_tier,
                    model_stale=bool(getattr(a, "model_stale", a.fallback_applied)),
                    fallback_applied=a.fallback_applied)
    return {**body, **(extra or {})}


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

def tier_counts(db: Session, since=None) -> dict:
    query = select(Assessment.tier, func.count()).group_by(Assessment.tier)
    if since is not None:
        query = query.where(Assessment.created_at >= since)
    counts = dict(db.execute(query).all())
    return {tier: int(counts.get(tier, 0)) for tier in ("ALLOW", "REVIEW", "BLOCK")}


def last_day():
    return now() - timedelta(days=1)
