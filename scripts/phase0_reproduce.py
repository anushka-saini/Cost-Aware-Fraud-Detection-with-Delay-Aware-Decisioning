"""Phase 0: confirm the refactored package reproduces the original models.

Retrains LightGBM, Random Forest and the logistic baseline through
fraud.models on the original 17 features, and compares eval_1 PR-AUC with
(a) the figures recorded in the notebooks and (b) the saved model files
scored on the same rows. The held-out set is not read.

    python scripts/phase0_reproduce.py
"""

import joblib

from fraud.config import get_settings
from fraud.data import get_split, load_dense
from fraud.evaluation import pr_auc
from fraud.features import FEATURES_V1
from fraud.models import make_lgbm, make_logreg, make_rf, predict_fraud_proba
from fraud.results import record

SOURCE = "scripts/phase0_reproduce.py"
TOLERANCE = 1e-3
# eval_1 figures printed by notebooks/02_modeling.ipynb (cells 11 and 13).
NOTEBOOK_EVAL1 = {"logreg": 0.5676, "lightgbm": 0.7113}


def main():
    settings = get_settings()
    frame = load_dense(FEATURES_V1)
    X_train, y_train = get_split(frame, "train", FEATURES_V1)
    X_eval1, y_eval1 = get_split(frame, "eval_1", FEATURES_V1)
    print(f"train {len(X_train):,} rows / {int(y_train.sum())} fraud; "
          f"eval_1 {len(X_eval1):,} rows / {int(y_eval1.sum())} fraud")

    saved = {
        "lightgbm": joblib.load(settings.models / "fraud_lgbm_days5-7.joblib")["model"],
        "random_forest": joblib.load(settings.models / "rf_model.pkl"),
    }
    builders = {
        "logreg": lambda: make_logreg(settings.seed),
        "lightgbm": lambda: make_lgbm(settings.seed, scale_pos_weight=20),
        "random_forest": lambda: make_rf(settings.seed, fraud_weight=500),
    }

    rows, ok = {}, True
    for name, build in builders.items():
        model = build().fit(X_train, y_train)
        new = pr_auc(y_eval1, predict_fraud_proba(model, X_eval1))
        row = {"retrained_eval1_pr_auc": new}
        if name in NOTEBOOK_EVAL1:
            row["notebook_eval1_pr_auc"] = NOTEBOOK_EVAL1[name]
            row["diff_vs_notebook"] = new - NOTEBOOK_EVAL1[name]
            ok &= abs(row["diff_vs_notebook"]) <= TOLERANCE
        if name in saved:
            old = pr_auc(y_eval1, predict_fraud_proba(saved[name], X_eval1))
            row["saved_model_eval1_pr_auc"] = old
            row["diff_vs_saved_model"] = new - old
            ok &= abs(row["diff_vs_saved_model"]) <= TOLERANCE
        rows[name] = row
        print(f"  {name:14s} " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()))

    print("\nReproduced within tolerance." if ok else "\nNOT reproduced: stop and investigate.")
    record("phase0_reproduction", {
        "split": "eval_1 (days 8-10)", "features": "v1 (17 original features)",
        "tolerance": TOLERANCE, "reproduced": bool(ok), "models": rows,
    }, SOURCE)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
