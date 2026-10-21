"""Precompute the scores the dashboard's threshold simulator works on.

For eval_2 (where the thresholds were chosen) and the held-out period, the
deployed model's calibrated probability is stored for every fraud exactly
(with its amount) and for legitimate transactions as a histogram over 4,000
log-spaced score bins. From that the browser can recompute cost, frauds let
through, review workload and block precision / recall for any pair of
thresholds and any cost assumptions, without a server round trip.

    python scripts/build_simulator_data.py
"""

import json
import os

import numpy as np

from fraud.config import get_settings
from fraud.data import SPLITS, get_split, load_dense
from fraud.evaluation import cost_report
from fraud.registry import load_bundle
from fraud.serving import Scorer

N_BINS = 4000
FLOOR = 1e-9
DEPLOYED_KEYS = ("total_cost", "fraud_allowed", "fraud_allowed_amount", "legit_reviewed", "legit_reviewed_pct",
                 "legit_blocked", "block_precision", "block_recall")


def main():
    settings = get_settings()
    scorer = Scorer(load_bundle(os.environ.get("FRAUD_MODEL_NAME", "fraud_scorer")))
    frame = load_dense(scorer.base_columns)
    out_dir = settings.results / "simulator"
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = np.r_[0.0, np.geomspace(FLOOR, 1.0, N_BINS)]          # bin i covers [edges[i], edges[i+1])

    for split in ("eval_2", "held_out"):
        X, y = get_split(frame, split, scorer.base_columns)
        y, amounts = y.to_numpy(), X["amount"].to_numpy()
        probs = scorer.probability(X)
        fraud = y == 1
        order = np.argsort(probs[fraud])
        bins = np.clip(np.searchsorted(edges, probs[~fraud], side="right") - 1, 0, N_BINS - 1)
        counts = np.bincount(bins, minlength=N_BINS)
        occupied = np.flatnonzero(counts)
        deployed = cost_report(y, probs, amounts, scorer.t_review, scorer.t_block)
        payload = {
            "split": split, "days": SPLITS[split],
            "model": {"name": scorer.name, "version": scorer.version},
            "n_rows": int(len(y)), "n_fraud": int(fraud.sum()), "n_legit": int((~fraud).sum()),
            "deployed": {"t_review": scorer.t_review, "t_block": scorer.t_block,
                         "cost_false_block": settings.cost_false_block, "cost_review": settings.cost_review,
                         **{k: deployed[k] for k in DEPLOYED_KEYS}},
            # Every fraud case: [calibrated probability, amount], ascending by probability.
            "fraud": [[float(p), float(a)] for p, a in zip(probs[fraud][order], amounts[fraud][order], strict=True)],
            # Legitimate transactions: [lower edge of the score bin, count], ascending, empty bins omitted.
            "legit_histogram": [[float(edges[i]), int(counts[i])] for i in occupied],
            "note": "Legitimate counts are exact per bin. A threshold inside a bin is treated as the lower edge of that "
                    "bin, so legitimate-side figures can differ from the exact ones by at most the count in one bin.",
        }
        (out_dir / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")
        print(f"{split}: {payload['n_fraud']} fraud, {len(occupied)} occupied legit bins, "
              f"deployed cost {deployed['total_cost']:,.0f}")


if __name__ == "__main__":
    main()
