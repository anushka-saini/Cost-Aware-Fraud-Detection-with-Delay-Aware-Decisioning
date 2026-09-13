from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Fraud Detection API")

model = joblib.load(r"C:\Users\Anushka\OneDrive\Desktop\fraud-detection-project\models\fraud_model.pkl")

T_REVIEW = 0.0001
T_BLOCK = 0.90

FEATURE_COLUMNS = [
    'amount', 'destination_transactions_last_24h', 'destination_transactions_last_7d',
    'destination_avg_previous_amount', 'destination_amount_deviation',
    'destination_is_first_transaction', 'origin_balance_error', 'destination_balance_error',
    'destination_balance_is_zero', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
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