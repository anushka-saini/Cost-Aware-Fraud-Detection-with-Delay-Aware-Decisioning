"""Request bodies."""

from typing import Literal

from pydantic import BaseModel, Field

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class FeatureTransaction(BaseModel):
    """Pre-computed features (the original request format, still accepted).

    Prefer /score/raw, where the service derives the recipient-history
    features itself. ``step`` is the hour index of the transaction; when it is
    missing the current hour is used.
    """
    amount: float
    origin_balance_error: float
    destination_balance_error: float
    destination_balance_is_zero: int = 0
    destination_transactions_last_24h: float = 0
    destination_transactions_last_7d: float = 0
    destination_avg_previous_amount: float = 0
    destination_amount_deviation: float = 0
    destination_is_first_transaction: int = 0
    step: int | None = Field(default=None, ge=0)
    # Whole-hour totals: no longer model inputs (they leaked future
    # transactions). Accepted so existing clients keep working, then ignored.
    total_transactions: float | None = None
    total_transaction_amount: float | None = None
    avg_transaction_amount: float | None = None
    type_CASH_IN: bool = False
    type_CASH_OUT: bool = False
    type_DEBIT: bool = False
    type_PAYMENT: bool = False
    type_TRANSFER: bool = False


class RawTransaction(BaseModel):
    """A transaction as the payment system sees it, before feature engineering."""
    step: int = Field(ge=0, description="Hour index of the transaction")
    type: TransactionType
    amount: float = Field(ge=0)
    nameDest: str = Field(min_length=1, max_length=64)
    nameOrig: str | None = Field(default=None, max_length=64)
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


class HistoryRecord(BaseModel):
    nameDest: str = Field(min_length=1, max_length=64)
    step: int = Field(ge=0)
    amount: float = Field(ge=0)


class ConfirmDecision(BaseModel):
    assessment_id: str
    user_proceeded: bool


class CaseDecision(BaseModel):
    action: Literal["approve", "decline", "escalate"]
    note: str = Field(min_length=3, max_length=2000, description="Why: mandatory for every decision")


class BulkCaseDecision(CaseDecision):
    case_ids: list[int] = Field(min_length=1, max_length=200)


class CaseNoteIn(BaseModel):
    note: str = Field(min_length=1, max_length=2000)


class FeedbackLabelIn(BaseModel):
    """A fraud / not-fraud outcome learned after the transaction was scored."""
    transaction_id: str = Field(min_length=1, max_length=64)
    is_fraud: bool
    source: Literal["analyst", "customer", "chargeback"] = "analyst"
    transaction_step: int = Field(ge=0)
    decided_step: int = Field(ge=0)
    note: str = Field(default="", max_length=2000)


class ThresholdProposal(BaseModel):
    t_review: float = Field(ge=0, le=1)
    t_block: float = Field(ge=0, le=1)
    cost_false_block: float = Field(gt=0)
    cost_review: float = Field(gt=0)
    split: Literal["eval_2", "held_out"]
    projected: dict = Field(default_factory=dict, description="The simulator's figures for these thresholds")
    rationale: str = Field(min_length=3, max_length=2000)


class RollbackRequest(BaseModel):
    version: str | None = Field(default=None, description="Version to activate; defaults to the previous one")
    reason: str = Field(min_length=3, max_length=2000)
