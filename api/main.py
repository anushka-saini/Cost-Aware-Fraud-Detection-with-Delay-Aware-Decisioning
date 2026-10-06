# Fraud Detection API
# Loads the trained model, scores incoming transactions, converts the fraud
# probability into an ALLOW / REVIEW / BLOCK decision, explains WHY using
# per-transaction SHAP values, and logs the user's final decision when they
# override a REVIEW/BLOCK recommendation (a real feedback-loop pattern used
# by production fraud systems for monitoring and future retraining).

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import csv
import os
from datetime import datetime, timezone

app = FastAPI(title="Fraud Detection API")

model = joblib.load("rf_model.pkl")
explainer = shap.TreeExplainer(model)

T_REVIEW = 0.32
T_BLOCK = 0.85

DEFAULT_TOTAL_TRANSACTIONS = 31900
DEFAULT_TOTAL_TRANSACTION_AMOUNT = 5233469000
DEFAULT_AVG_TRANSACTION_AMOUNT = 164058.6

FEATURE_COLUMNS = [
    'amount', 'destination_transactions_last_24h', 'destination_transactions_last_7d',
    'destination_avg_previous_amount', 'destination_amount_deviation',
    'destination_is_first_transaction', 'origin_balance_error', 'destination_balance_error',
    'destination_balance_is_zero', 'total_transactions', 'total_transaction_amount',
    'avg_transaction_amount', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
    'type_PAYMENT', 'type_TRANSFER'
]

# Human-readable explanations for each feature, shown to the end user instead
# of raw column names. Keep these short — they appear directly in the UI.
FEATURE_EXPLANATIONS = {
    'amount': "the transaction amount",
    'destination_transactions_last_24h': "how often this recipient has received money recently",
    'destination_transactions_last_7d': "this recipient's weekly transaction activity",
    'destination_avg_previous_amount': "how this amount compares to what this recipient usually receives",
    'destination_amount_deviation': "this amount being unusual for this recipient's history",
    'destination_is_first_transaction': "this being the first transaction ever to this recipient",
    'origin_balance_error': "a mismatch in your account's balance after this transaction",
    'destination_balance_error': "a mismatch in the recipient's balance after this transaction",
    'destination_balance_is_zero': "the recipient's account balance being zero",
    'total_transactions': "overall system transaction volume at this time",
    'total_transaction_amount': "overall system transaction value at this time",
    'avg_transaction_amount': "typical transaction size across the system right now",
    'type_CASH_IN': "this being a cash-in transaction",
    'type_CASH_OUT': "this being a cash-out transaction",
    'type_DEBIT': "this being a debit transaction",
    'type_PAYMENT': "this being a payment",
    'type_TRANSFER': "this being a transfer",
}

DECISION_LOG_PATH = "decision_log.csv"


class Transaction(BaseModel):
    amount: float
    origin_balance_error: float
    destination_balance_error: float
    destination_balance_is_zero: int = 0
    destination_transactions_last_24h: float = 0
    destination_transactions_last_7d: float = 0
    destination_avg_previous_amount: float = 0
    destination_amount_deviation: float = 0
    destination_is_first_transaction: int = 0
    total_transactions: float = DEFAULT_TOTAL_TRANSACTIONS
    total_transaction_amount: float = DEFAULT_TOTAL_TRANSACTION_AMOUNT
    avg_transaction_amount: float = DEFAULT_AVG_TRANSACTION_AMOUNT
    type_CASH_IN: bool = False
    type_CASH_OUT: bool = False
    type_DEBIT: bool = False
    type_PAYMENT: bool = False
    type_TRANSFER: bool = False


class DecisionLogEntry(BaseModel):
    assessment_id: str          # pass back the id returned by /score
    user_proceeded: bool        # True = user clicked "Proceed Anyway", False = "Cancel"


# In-memory store mapping assessment_id -> the original assessment, so /confirm
# can log the full context without the client having to resend every field.
# NOTE: this resets if the API restarts — fine for a demo, not for production.
# (a production deployment would persist this in a database keyed by a real
# transaction ID, not in process memory)
_pending_assessments = {}


def explain_prediction(row_df):
    """Return the top 3 factors pushing this specific prediction toward fraud,
    as human-readable strings, using this one transaction's own SHAP values."""
    shap_values = explainer.shap_values(row_df)
    # TreeExplainer on a binary LightGBM classifier returns values for the
    # positive (fraud) class directly for a single-output model.
    values = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    # A Random Forest returns one column per class, shape (n_features, 2) —
    # keep only the positive (fraud) class column.
    if getattr(values, "ndim", 1) == 2:
        values = values[:, 1]

    contributions = list(zip(FEATURE_COLUMNS, values))
    # Only features pushing TOWARD fraud (positive SHAP value) are useful
    # as "reasons this was flagged" — negative ones reduced the risk score.
    risk_factors = [(f, v) for f, v in contributions if v > 0]
    risk_factors.sort(key=lambda x: x[1], reverse=True)

    top_reasons = [FEATURE_EXPLANATIONS.get(f, f) for f, _ in risk_factors[:3]]
    return top_reasons


@app.post("/score")
def score_transaction(txn: Transaction):
    row = pd.DataFrame([txn.dict()])
    row = row[FEATURE_COLUMNS]

    prob = model.predict_proba(row)[:, 1][0]

    if prob >= T_BLOCK:
        tier = "BLOCK"
    elif prob >= T_REVIEW:
        tier = "REVIEW"
    else:
        tier = "ALLOW"

    reasons = explain_prediction(row) if tier != "ALLOW" else []

    assessment_id = f"a{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    _pending_assessments[assessment_id] = {
        "txn": txn.dict(),
        "fraud_probability": float(prob),
        "tier": tier,
    }

    return {
        "assessment_id": assessment_id,
        "fraud_probability": float(prob),
        "tier": tier,
        "reasons": reasons,
        "thresholds_used": {"t_review": T_REVIEW, "t_block": T_BLOCK},
    }


@app.post("/confirm-decision")
def confirm_decision(entry: DecisionLogEntry):
    """Logs what the user actually chose to do after seeing the ALLOW/REVIEW/BLOCK
    recommendation. This is the real feedback-loop data a production system would
    use to monitor how often users override warnings, and to retrain on confirmed
    outcomes over time."""
    assessment = _pending_assessments.get(entry.assessment_id)
    if assessment is None:
        return {"error": "Unknown or expired assessment_id"}

    log_exists = os.path.isfile(DECISION_LOG_PATH)
    with open(DECISION_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow([
                "timestamp", "assessment_id", "tier", "fraud_probability",
                "user_proceeded", "amount",
            ])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            entry.assessment_id,
            assessment["tier"],
            assessment["fraud_probability"],
            entry.user_proceeded,
            assessment["txn"]["amount"],
        ])

    del _pending_assessments[entry.assessment_id]
    return {"status": "logged"}


@app.get("/health")
def health():
    return {"status": "ok"}


# Read-only endpoints for the analytics dashboard (dashboard/). Kept in a
# separate module so the scoring and logging endpoints above stay as they are.
from dashboard_routes import build_router

app.include_router(build_router(
    model=model,
    explainer=explainer,
    transaction_model=Transaction,
    feature_columns=FEATURE_COLUMNS,
    feature_explanations=FEATURE_EXPLANATIONS,
    decision_log_path=DECISION_LOG_PATH,
    thresholds={"t_review": T_REVIEW, "t_block": T_BLOCK},
))
