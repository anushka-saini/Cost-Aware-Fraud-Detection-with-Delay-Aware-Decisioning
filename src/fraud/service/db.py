"""Database engine, session and tables (SQLAlchemy 2).

The schema is owned by Alembic (``alembic upgrade head``). For local
development and tests ``init_db`` creates any missing tables directly.
"""

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator

from fraud.service.settings import get_service_settings


def now() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """A timezone-aware UTC datetime on every backend. SQLite stores
    datetimes without a zone, so the zone is put back when rows are read."""
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_result_value(self, value, dialect):
        return value.replace(tzinfo=UTC) if value is not None and value.tzinfo is None else value


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(16))               # "analyst" or "admin"
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now)


class Transaction(Base):
    """Every transaction the service has seen. Doubles as the feature store:
    recipient-history features are computed from the rows for ``name_dest``."""
    __tablename__ = "transactions"
    id: Mapped[int] = mapped_column(primary_key=True)
    step: Mapped[int] = mapped_column(Integer)
    type: Mapped[str | None] = mapped_column(String(16))
    amount: Mapped[float] = mapped_column(Float)
    name_orig: Mapped[str | None] = mapped_column(String(64))
    name_dest: Mapped[str] = mapped_column(String(64))
    old_balance_orig: Mapped[float | None] = mapped_column(Float)
    new_balance_orig: Mapped[float | None] = mapped_column(Float)
    old_balance_dest: Mapped[float | None] = mapped_column(Float)
    new_balance_dest: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(16), default="scored")   # "scored", "history" or "stream"
    true_label: Mapped[int | None] = mapped_column(Integer)             # known only for replayed demo data
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now)

    __table_args__ = (Index("ix_transactions_dest_step", "name_dest", "step"),)


class Assessment(Base):
    """One scoring result."""
    __tablename__ = "assessments"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    transaction_id: Mapped[int | None] = mapped_column(ForeignKey("transactions.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(16))
    amount: Mapped[float] = mapped_column(Float)
    raw_score: Mapped[float] = mapped_column(Float)
    fraud_probability: Mapped[float] = mapped_column(Float)
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    tier: Mapped[str] = mapped_column(String(8), index=True)
    supervised_tier: Mapped[str | None] = mapped_column(String(8))
    fallback_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    t_review: Mapped[float] = mapped_column(Float)
    t_block: Mapped[float] = mapped_column(Float)
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now, index=True)

    transaction: Mapped[Transaction | None] = relationship()


class Decision(Base):
    """What a person did with an assessment: the customer's proceed / cancel,
    or an analyst's approve / decline / escalate."""
    __tablename__ = "decisions"
    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), index=True)
    actor: Mapped[str] = mapped_column(String(64))
    actor_role: Mapped[str] = mapped_column(String(16))          # "customer", "analyst" or "admin"
    action: Mapped[str] = mapped_column(String(16))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now)


class AnalystCase(Base):
    """A REVIEW-tier assessment waiting for, or resolved by, an analyst."""
    __tablename__ = "analyst_cases"
    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[str] = mapped_column(ForeignKey("assessments.id"), unique=True)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)   # open / approved / declined / escalated
    amount: Mapped[float] = mapped_column(Float)
    fraud_probability: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now, index=True)
    closed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    decided_by: Mapped[str | None] = mapped_column(String(64))

    assessment: Mapped[Assessment] = relationship()
    notes: Mapped[list["CaseNote"]] = relationship(back_populates="case", order_by="CaseNote.created_at")


class CaseNote(Base):
    __tablename__ = "case_notes"
    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("analyst_cases.id"), index=True)
    author: Mapped[str] = mapped_column(String(64))
    note: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now)

    case: Mapped[AnalystCase] = relationship(back_populates="notes")


class AuditLog(Base):
    """Append-only record of every state-changing action."""
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime, default=now, index=True)
    actor: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(48), index=True)
    entity_type: Mapped[str] = mapped_column(String(32))
    entity_id: Mapped[str] = mapped_column(String(64))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    request_id: Mapped[str | None] = mapped_column(String(40))


class ModelVersion(Base):
    """Which registered bundle is serving, and the history of switches."""
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(16))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    activated_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    activated_by: Mapped[str | None] = mapped_column(String(64))
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (Index("ix_model_versions_name_version", "name", "version", unique=True),)


class DriftReport(Base):
    __tablename__ = "drift_reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    period: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now)
    report: Mapped[dict] = mapped_column(JSON)


class FeedbackLabel(Base):
    """An outcome learned after scoring, with when it happened and when it became known."""
    __tablename__ = "feedback_labels"
    id: Mapped[int] = mapped_column(primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(64), index=True)
    is_fraud: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(16))              # analyst / customer / chargeback
    transaction_step: Mapped[int] = mapped_column(Integer)
    decided_step: Mapped[int] = mapped_column(Integer)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now)


# ---------------------------------------------------------------------------
# Engine and sessions
# ---------------------------------------------------------------------------

def make_engine(url: str | None = None):
    url = url or get_service_settings().database_url
    if url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool          # one shared in-memory database
        return create_engine(url, **kwargs)
    return create_engine(url, pool_pre_ping=True)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def configure(url: str):
    """Point the module at a different database (used by tests)."""
    global engine
    engine = make_engine(url)
    SessionLocal.configure(bind=engine)
    return engine


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
