# Fraud Detection API
# Loads the trained model, scores incoming transactions,
# and converts the fraud probability into an ALLOW / REVIEW / BLOCK decision.

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Fraud Detection API")

model = joblib.load("fraud_model.pkl")

T_REVIEW = 0.0001
T_BLOCK = 0.90

# TODO: replace these three with the actual mean/median values from your
# training data — run this in a notebook against X_train (days 5-7):
#   X_train[["total_transactions", "total_transaction_amount", "avg_transaction_amount"]].describe()
# These are step-level (hourly) system-wide aggregates, not per-transaction —
# a live API can't know "total transactions this hour" for an in-progress hour,
# so we use a fixed typical value as a stand-in. State this assumption if asked.
DEFAULT_TOTAL_TRANSACTIONS = 1000        # <-- placeholder, replace with real mean
DEFAULT_TOTAL_TRANSACTION_AMOUNT = 5_000_000  # <-- placeholder, replace with real mean
DEFAULT_AVG_TRANSACTION_AMOUNT = 5000     # <-- placeholder, replace with real mean

# Must match training's feature_columns EXACTLY — see 01_data_audit.ipynb cell 25
FEATURE_COLUMNS = [
    'amount', 'destination_transactions_last_24h', 'destination_transactions_last_7d',
    'destination_avg_previous_amount', 'destination_amount_deviation',
    'destination_is_first_transaction', 'origin_balance_error', 'destination_balance_error',
    'destination_balance_is_zero', 'total_transactions', 'total_transaction_amount',
    'avg_transaction_amount', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
    'type_PAYMENT', 'type_TRANSFER'
]


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
    # System-level features — optional, default to typical training-data values
    # since a live caller usually won't know current system-wide throughput.
    total_transactions: float = DEFAULT_TOTAL_TRANSACTIONS
    total_transaction_amount: float = DEFAULT_TOTAL_TRANSACTION_AMOUNT
    avg_transaction_amount: float = DEFAULT_AVG_TRANSACTION_AMOUNT
    type_CASH_IN: bool = False
    type_CASH_OUT: bool = False
    type_DEBIT: bool = False
    type_PAYMENT: bool = False
    type_TRANSFER: bool = False


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

    return {
        "fraud_probability": float(prob),
        "tier": tier,
        "thresholds_used": {"t_review": T_REVIEW, "t_block": T_BLOCK}
    }


@app.get("/health")
def health():
    return {"status": "ok"}