"""Delay-aware retraining job with a champion/challenger gate.

    python scripts/retrain.py --as-of-day 14 --maturity-days 2
    python scripts/retrain.py --as-of-day 14 --maturity-days 2 --promote

Run before day ``--as-of-day`` starts. Only labels that have matured are used:
with a maturity of D days, the last usable day is as_of_day - 1 - D. Feedback
labels from the service database (analyst decisions in the review queue,
POST /labels) override the stored label for their transaction once they too
have matured.

The labelled days are split in time: the most recent VALIDATION_DAYS are held
back, the challenger is trained on the rest and then calibrated and
thresholded on the held-back days. The challenger replaces the serving model
only if its PR-AUC on those days is significantly higher (paired bootstrap).
Every run appends its decision to results/retraining_log.json; a new model
version is registered only with --promote.
"""

import argparse
import json
import os
from datetime import UTC, datetime

import numpy as np

from fraud.config import get_settings
from fraud.data import DENSE_DAYS, TARGET, get_days, load_dense
from fraud.evaluation import paired_bootstrap_difference, pr_auc
from fraud.labels import Label, LabelStore, last_fully_labelled_day, usable_labels
from fraud.models import make_lgbm
from fraud.protocol import FINAL_SEEDS, calibrate_and_threshold
from fraud.registry import load_bundle, save_bundle
from fraud.results import _jsonable
from fraud.serving import Scorer

VALIDATION_DAYS = 3
MODEL_NAME = os.environ.get("FRAUD_MODEL_NAME", "fraud_scorer")


def database_labels() -> list[Label]:
    """Feedback labels recorded by the service (analyst decisions, POST /labels)."""
    from sqlalchemy import select

    from fraud.service import db as database
    database.init_db()
    with database.SessionLocal() as session:
        return [Label(r.transaction_id, r.is_fraud, r.source, r.transaction_step, r.decided_step, r.note)
                for r in session.scalars(select(database.FeedbackLabel))]


def apply_feedback(frame, labels, as_of_step: int, maturity_days: float) -> dict:
    """Overwrite stored labels with matured feedback labels.

    A feedback label's transaction_id is matched against the row index of the
    feature table. Labels for transactions that only exist in the service's
    own database (live traffic) are counted but cannot be joined here.
    """
    usable = usable_labels(labels, as_of_step, maturity_days)
    changed = 0
    for transaction_id, label in usable.items():
        try:
            index = int(transaction_id)
        except ValueError:
            continue
        if index in frame.index and frame.at[index, TARGET] != label.is_fraud:
            frame.at[index, TARGET] = label.is_fraud
            changed += 1
    return {"matured_feedback_labels": len(usable), "labels_changed": changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of-day", type=int, required=True)
    parser.add_argument("--maturity-days", type=float, default=2)
    parser.add_argument("--promote", action="store_true", help="register the challenger if it wins")
    parser.add_argument("--label-store", default=None,
                        help="CSV of feedback labels; default is the service database's feedback_labels table")
    args = parser.parse_args()

    settings = get_settings()
    champion_bundle = load_bundle(MODEL_NAME)
    champion = Scorer(champion_bundle)
    features = [c for c in champion.feature_columns]
    weight = next(iter(champion_bundle["protocol"]["weight"].values()))

    last_day = last_fully_labelled_day(args.as_of_day, args.maturity_days)
    first_day = DENSE_DAYS[0]
    train_days = list(range(first_day, last_day - VALIDATION_DAYS + 1))
    validation_days = list(range(last_day - VALIDATION_DAYS + 1, last_day + 1))
    if len(train_days) < 2:
        raise SystemExit(f"Not enough matured labels: labelled days end at {last_day}.")

    frame = load_dense(champion.base_columns)
    labels = LabelStore(args.label_store).all() if args.label_store else database_labels()
    feedback = apply_feedback(frame, labels, as_of_step=args.as_of_day * 24, maturity_days=args.maturity_days)

    X_train, y_train = get_days(frame, train_days, features)
    X_val, y_val = get_days(frame, validation_days, features)
    models = [make_lgbm(seed, scale_pos_weight=weight).fit(X_train, y_train) for seed in FINAL_SEEDS]
    challenger_raw = np.mean([m.predict_proba(X_val)[:, 1] for m in models], axis=0)
    champion_raw = champion.raw_score(X_val[champion.base_columns])

    test = paired_bootstrap_difference(y_val.to_numpy(), challenger_raw, champion_raw)
    wins = test["ci_low"] > 0
    val_days_array = frame.loc[X_val.index, "day_bucket"].to_numpy()
    chosen = calibrate_and_threshold(challenger_raw, y_val.to_numpy(), X_val["amount"].to_numpy(), val_days_array)
    thresholds = chosen["thresholds_eval_2"]["robust"]

    entry = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "as_of_day": args.as_of_day, "label_maturity_days": args.maturity_days,
        "last_fully_labelled_day": last_day, "train_days": train_days, "validation_days": validation_days,
        "feedback": feedback, "n_train_rows": int(len(X_train)), "n_train_fraud": int(y_train.sum()),
        "champion": {"name": champion.name, "version": champion.version,
                     "trained_through_day": max(champion_bundle["train_days"]),
                     "validation_pr_auc": pr_auc(y_val, champion_raw)},
        "challenger": {"validation_pr_auc": pr_auc(y_val, challenger_raw),
                       "calibration": chosen["calibration"]["method"],
                       "t_review": thresholds["t_review"], "t_block": thresholds["t_block"]},
        "pr_auc_difference": test["difference"], "ci_low": test["ci_low"], "ci_high": test["ci_high"],
        "decision": "promote" if wins else "keep",
        "reason": ("challenger significantly better on the most recent labelled days" if wins
                   else "challenger not significantly better on the most recent labelled days"),
        "registered_version": None,
    }
    if wins and args.promote:
        extra = {k: champion_bundle[k] for k in ("autoencoder", "autoencoder_columns", "reference_row") if k in champion_bundle}
        if "staleness" in champion_bundle:
            extra["staleness"] = {**champion_bundle["staleness"], "labels_through_day": max(train_days)}
        target = save_bundle(
            MODEL_NAME, model=models, feature_columns=features, calibrator=chosen["calibrator"],
            thresholds={"t_review": thresholds["t_review"], "t_block": thresholds["t_block"]},
            metrics={"validation_pr_auc": entry["challenger"]["validation_pr_auc"], "validation_days": validation_days},
            train_days=train_days,
            protocol={**champion_bundle["protocol"], "run_tag": f"retrain_as_of_day_{args.as_of_day}",
                      "calibration_and_thresholds_chosen_on": f"days {validation_days[0]}-{validation_days[-1]}"},
            extra=extra,
        )
        entry["registered_version"] = target.name

    log_path = settings.results / "retraining_log.json"
    history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    history.append(_jsonable(entry))
    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(entry), indent=2))


if __name__ == "__main__":
    main()
