import numpy as np
import pandas as pd

from fraud.serving import ANOMALY_COLUMN, Scorer

COLUMNS = ["amount", "x"]


class FixedModel:
    """Scores a row by its ``x`` column."""

    def predict_proba(self, X):
        p = X["x"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


class FixedDetector:
    """Anomaly score = the amount."""

    def anomaly_score(self, X):
        return X["amount"].to_numpy(dtype=float)


def bundle(**extra):
    return {"model": [FixedModel(), FixedModel()], "feature_columns": COLUMNS, "calibrator": None,
            "thresholds": {"t_review": 0.3, "t_block": 0.8}, "name": "test", "version": "v1", **extra}


FRAME = pd.DataFrame({"amount": [10.0, 500.0, 500.0, 500.0], "x": [0.1, 0.1, 0.5, 0.9]})
STALENESS = {"labels_through_day": 7, "max_label_age_days": 5, "anomaly_review_threshold": 100.0}


def test_tiers_follow_the_thresholds():
    scored = Scorer(bundle()).score_frame(FRAME)
    assert scored["tier"].tolist() == ["ALLOW", "ALLOW", "REVIEW", "BLOCK"]
    assert not scored["fallback_applied"].any() and not scored["model_stale"].any()


def test_anomaly_score_is_reported_alongside_the_supervised_score():
    scored = Scorer(bundle(autoencoder=FixedDetector(), autoencoder_columns=COLUMNS)).score_frame(FRAME)
    assert scored[ANOMALY_COLUMN].tolist() == [10.0, 500.0, 500.0, 500.0]
    assert scored["raw_score"].tolist() == [0.1, 0.1, 0.5, 0.9]


def test_fallback_does_nothing_while_labels_are_fresh():
    scorer = Scorer(bundle(autoencoder=FixedDetector(), autoencoder_columns=COLUMNS, staleness=STALENESS))
    assert not scorer.is_stale(current_day=12)        # 5 days after the last labelled day: at the limit
    scored = scorer.score_frame(FRAME, current_day=12)
    assert scored["tier"].tolist() == ["ALLOW", "ALLOW", "REVIEW", "BLOCK"]
    assert not scored["fallback_applied"].any()
    # No date supplied: never treated as stale.
    assert not scorer.score_frame(FRAME)["model_stale"].any()


def test_fallback_escalates_only_allowed_anomalous_rows_when_stale():
    scorer = Scorer(bundle(autoencoder=FixedDetector(), autoencoder_columns=COLUMNS, staleness=STALENESS))
    assert scorer.is_stale(current_day=13)
    scored = scorer.score_frame(FRAME, current_day=13)
    # Row 0: allowed and ordinary -> stays. Row 1: allowed but anomalous -> review.
    # Rows 2 and 3 were already flagged by the supervised model and are untouched.
    assert scored["tier"].tolist() == ["ALLOW", "REVIEW", "REVIEW", "BLOCK"]
    assert scored["supervised_tier"].tolist() == ["ALLOW", "ALLOW", "REVIEW", "BLOCK"]
    assert scored["fallback_applied"].tolist() == [False, True, False, False]


def test_fallback_never_downgrades_a_block():
    scorer = Scorer(bundle(autoencoder=FixedDetector(), autoencoder_columns=COLUMNS, staleness=STALENESS))
    scored = scorer.score_frame(pd.DataFrame({"amount": [1.0], "x": [0.95]}), current_day=99)
    assert scored["tier"].tolist() == ["BLOCK"]


def test_hybrid_bundle_derives_the_anomaly_feature_itself():
    hybrid = bundle(feature_columns=[*COLUMNS, ANOMALY_COLUMN], autoencoder=FixedDetector(), autoencoder_columns=COLUMNS)
    scorer = Scorer(hybrid)
    assert scorer.base_columns == COLUMNS                 # callers never supply the anomaly score
    assert list(scorer.model_frame(FRAME).columns) == [*COLUMNS, ANOMALY_COLUMN]
