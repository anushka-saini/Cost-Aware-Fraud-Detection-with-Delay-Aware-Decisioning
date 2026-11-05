"""Save the V2 autoencoder as a monitoring-only novelty detector.

In V2 the autoencoder sat in the scoring path: when labels were stale it could
send a transaction to REVIEW. As a feature it did not help (hybrid vs LightGBM
was not significant) and on its own it found little fraud (PR-AUC 0.2091 on
held-out, unstable across seeds). In V3 it no longer touches decisions. It is
used only by the monitoring job, which tracks the share of transactions that
look very unusual and alerts when that share rises.

The novelty threshold is the 99th percentile of anomaly scores on unlabelled
eval_2 traffic (days 11-13), so about 1% of normal traffic is expected above it.
No labels are used.

    python scripts/register_novelty_detector.py
"""

import joblib
import numpy as np

from fraud.config import get_settings
from fraud.data import get_split, load_dense
from fraud.registry import save_bundle

SEED = 42
QUANTILE = 0.99


def main():
    settings = get_settings()
    saved = joblib.load(settings.models / "runs" / "phase4" / "autoencoder.joblib")
    columns = saved["feature_columns"]
    detector = saved["autoencoders"][SEED]

    X_eval_2, _ = get_split(load_dense(columns), "eval_2", columns)
    threshold = float(np.quantile(detector.anomaly_score(X_eval_2), QUANTILE))
    target = save_bundle(
        "novelty_detector", model=detector, feature_columns=columns, calibrator=None,
        thresholds={"novelty": threshold}, metrics={"expected_novelty_share": 1 - QUANTILE},
        protocol={"source_run": "phase4", "seed": SEED, "threshold": f"{QUANTILE:.0%} quantile of eval_2 anomaly scores",
                  "labels_used": False, "used_for": "monitoring only; never changes a decision"},
    )
    print("registered", target, "threshold", threshold)


if __name__ == "__main__":
    main()
