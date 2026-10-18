"""Phase 3: how much does the model depend on PaySim's balance bookkeeping?

Compares the leak-free LightGBM ensemble with and without the three balance
features (both already evaluated once on held-out by scripts/run_protocol.py)
and writes results/phase3/summary.md.

    python scripts/phase3_summary.py
"""

import json

import numpy as np

from fraud.config import get_settings
from fraud.evaluation import paired_bootstrap_difference
from fraud.features import BALANCE
from fraud.results import record


def main():
    settings = get_settings()
    final = {tag: json.loads((settings.results / tag / "final.json").read_text(encoding="utf-8"))["families"]["lightgbm"]
             for tag in ("phase2", "phase3")}
    scores = {tag: np.load(settings.models / "runs" / tag / "held_out_scores.npz") for tag in ("phase2", "phase3")}
    y = scores["phase2"]["y"]
    difference = paired_bootstrap_difference(y, scores["phase3"]["lightgbm__ensemble"], scores["phase2"]["lightgbm__ensemble"])

    def row(tag):
        e = final[tag]["ensemble"]
        p = e["operating_points"]["robust"]
        return {"pr_auc": e["pr_auc"], "ci_low": e["ci_low"], "ci_high": e["ci_high"],
                "seed_mean": final[tag]["seed_summary"]["pr_auc"]["mean"], "seed_std": final[tag]["seed_summary"]["pr_auc"]["std"],
                **{k: p[k] for k in ("total_cost", "fraud_allowed", "legit_reviewed_pct", "legit_blocked",
                                     "block_precision", "block_recall", "block_f1")}}

    with_balance, without_balance = row("phase2"), row("phase3")
    payload = {"model": "LightGBM 5-seed ensemble, leak-free features", "removed_features": BALANCE,
               "with_balance_features": with_balance, "without_balance_features": without_balance,
               "pr_auc_difference_without_minus_with": difference}
    record("phase3_balance_ablation", payload, "scripts/phase3_summary.py (from scripts/run_protocol.py --tag phase2 / phase3)")

    def line(name, r):
        return (f"| {name} | {r['pr_auc']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['seed_mean']:.4f} ± {r['seed_std']:.4f} "
                f"| {r['total_cost']:,.0f} | {r['fraud_allowed']} | {100 * r['legit_reviewed_pct']:.2f}% | {r['legit_blocked']} "
                f"| {r['block_precision']:.4f} | {r['block_recall']:.4f} |")

    text = "\n".join([
        "# Simulator-artefact check: the model without the balance features", "",
        "Same protocol, same data, same model family (LightGBM 5-seed ensemble on the leak-free features). "
        f"The only change is removing `{'`, `'.join(BALANCE)}`.", "",
        "| Features | Held-out PR-AUC [95% CI] | 5 seeds | Total cost | Fraud let through | Legit reviewed | Legit blocked | Block precision | Block recall |",
        "|---|---|---|---|---|---|---|---|---|",
        line("With balance features", with_balance),
        line("Without balance features", without_balance), "",
        f"Paired bootstrap, PR-AUC without minus with: {difference['difference']:+.4f} "
        f"(95% CI {difference['ci_low']:+.4f} to {difference['ci_high']:+.4f}), "
        f"{'significant' if difference['significant'] else 'not significant'}.", "",
        "## What this says", "",
        f"The model depends heavily on the balance features: without them held-out PR-AUC falls from "
        f"{with_balance['pr_auc']:.4f} to {without_balance['pr_auc']:.4f} and the share of fraud that can be blocked "
        f"automatically falls from {100 * with_balance['block_recall']:.1f}% to {100 * without_balance['block_recall']:.1f}%. "
        f"It still lets no held-out fraud through, but only by sending {100 * without_balance['legit_reviewed_pct']:.1f}% of "
        f"legitimate transactions to review instead of {100 * with_balance['legit_reviewed_pct']:.1f}%.", "",
        "The balance features compare an account's balance before and after a transaction. In PaySim that relationship is "
        "produced by the simulator's own bookkeeping, and fraud rows break it in a regular way, so part of what the model "
        "has learned is a property of the simulator. A real payment system would not necessarily expose post-transaction "
        "balances at scoring time, and real fraud would not necessarily leave this trace. The headline figures should "
        "therefore be read as an upper bound for this dataset, not as an estimate of performance on real payments.", "",
        "The features have not been removed from the deployed model; that decision is left open.", "",
    ])
    (settings.results / "phase3" / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
