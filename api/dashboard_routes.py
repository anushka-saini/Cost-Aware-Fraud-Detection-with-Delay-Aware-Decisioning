# Read-only endpoints used by the analytics dashboard (dashboard/).
# Nothing here writes to disk or changes API state: /score and
# /confirm-decision in main.py remain the only endpoints that do.

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter

API_DIR = Path(__file__).resolve().parent
VALIDATION_METRICS_PATH = API_DIR / "validation_metrics.json"
DEMO_SAMPLE_PATH = API_DIR.parent / "demo_sample.csv"


def _fraud_class_values(shap_values):
    """SHAP values for the fraud class of a single row, as a 1-D array.
    Handles both output layouts: (1, n_features) for a single-output model
    and (1, n_features, 2) for a model with one column per class."""
    values = np.asarray(shap_values[1] if isinstance(shap_values, list) else shap_values)[0]
    return values[:, 1] if values.ndim == 2 else values


def _fraud_class_base_value(explainer):
    base = np.atleast_1d(explainer.expected_value)
    return float(base[1] if len(base) > 1 else base[0])


def build_router(model, explainer, transaction_model, feature_columns,
                 feature_explanations, decision_log_path, thresholds):
    router = APIRouter()
    Transaction = transaction_model

    @router.get("/model-info")
    def model_info():
        """Which model the API has loaded, read from the model object itself."""
        params = model.get_params()
        shown = ["n_estimators", "max_depth", "min_samples_leaf", "class_weight",
                 "num_leaves", "min_child_samples", "scale_pos_weight"]
        return {
            "model_class": type(model).__name__,
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
            "thresholds": thresholds,
            "params": {k: params[k] for k in shown if k in params},
        }

    @router.post("/explain")
    def explain(txn: Transaction):
        """Signed SHAP contribution of every feature for one transaction.
        Does not create an assessment and is never logged."""
        row = pd.DataFrame([txn.dict()])[feature_columns]
        prob = float(model.predict_proba(row)[:, 1][0])
        values = _fraud_class_values(explainer.shap_values(row))
        base_value = _fraud_class_base_value(explainer)

        # For the Random Forest, base value + contributions equals the fraud
        # probability. For a boosted model the values are in log-odds instead,
        # so report which one applies rather than assume.
        additive = abs(base_value + float(values.sum()) - prob) < 1e-6

        contributions = [
            {
                "feature": f,
                "label": feature_explanations.get(f, f),
                "feature_value": row.iloc[0][f].item(),
                "shap_value": float(v),
            }
            for f, v in zip(feature_columns, values)
        ]
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        return {
            "fraud_probability": prob,
            "base_value": base_value,
            "units": "probability" if additive else "log-odds",
            "contributions": contributions,
        }

    @router.get("/decision-log")
    def decision_log():
        """Every row of the decision log written by /confirm-decision."""
        if not os.path.isfile(decision_log_path):
            return {"exists": False, "rows": []}
        with open(decision_log_path, newline="") as f:
            rows = [
                {
                    "timestamp": r["timestamp"],
                    "assessment_id": r["assessment_id"],
                    "tier": r["tier"],
                    "fraud_probability": float(r["fraud_probability"]),
                    "user_proceeded": r["user_proceeded"] == "True",
                    "amount": float(r["amount"]),
                }
                for r in csv.DictReader(f)
            ]
        return {"exists": True, "rows": rows}

    @router.get("/validation-metrics")
    def validation_metrics():
        """Figures transcribed from the notebooks, each with its source cell."""
        with open(VALIDATION_METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)

    @router.get("/demo-sample/random")
    def demo_sample_random():
        """One random held-out transaction from demo_sample.csv with its label."""
        if not DEMO_SAMPLE_PATH.is_file():
            return {"error": "demo_sample.csv not found"}
        row = pd.read_csv(DEMO_SAMPLE_PATH).sample(1)
        record = json.loads(row.to_json(orient="records"))[0]
        actual_fraud = int(record.pop("actual_fraud"))
        return {"transaction": record, "actual_fraud": actual_fraud}

    return router
