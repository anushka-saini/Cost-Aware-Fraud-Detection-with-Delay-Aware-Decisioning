"""Phase 4: label-free autoencoder, on its own and as a feature of the supervised model.

    python scripts/phase4_autoencoder.py --stage select
    python scripts/phase4_autoencoder.py --stage final

select (train / eval_1 / eval_2 only):
  * trains the autoencoder on every training-window transaction, without labels,
    early-stopped on eval_1 reconstruction loss, for 5 seeds
  * reports how well reconstruction error alone ranks fraud on eval_1 / eval_2
  * builds the hybrid: LightGBM on the leak-free features plus the anomaly
    score, chosen and thresholded with the standard protocol

final (held-out, once):
  * autoencoder alone, hybrid, paired comparison with the supervised model,
    and whether the autoencoder catches the frauds the supervised model
    scores lowest
"""

import argparse
import json
import time

import joblib
import numpy as np

from fraud.autoencoder import AE_PARAMS, TabularAutoencoder
from fraud.config import get_settings
from fraud.data import SPLITS, get_split, load_dense
from fraud.evaluation import paired_bootstrap_difference, pr_auc, pr_auc_with_ci, precision_at_k
from fraud.features import FEATURES_V2
from fraud.protocol import FAMILIES, FINAL_SEEDS, evaluate_family, select_family
from fraud.results import _jsonable, record
from fraud.serving import ANOMALY_COLUMN

SOURCE = "scripts/phase4_autoencoder.py"
TAG = "phase4"
HYBRID_FEATURES = [*FEATURES_V2, ANOMALY_COLUMN]
SUPERVISED_TAG, SUPERVISED_FAMILY = "phase2", "lightgbm"
LOWEST_SCORED = 41          # the 5% of held-out frauds the supervised model is least sure about


def log(message):
    print(message, flush=True)


def strip(obj, drop=("models", "calibrator", "_raw")):
    if isinstance(obj, dict):
        return {str(k): strip(v, drop) for k, v in obj.items() if k not in drop}
    if isinstance(obj, (list, tuple)):
        return [strip(v, drop) for v in obj]
    return obj


def ranking_report(y, score):
    return {**pr_auc_with_ci(y, score), "precision_at_1pct": precision_at_k(y, score, 0.01),
            "precision_at_0.1pct": precision_at_k(y, score, 0.001),
            "share_of_fraud_in_top_1pct": float(np.asarray(y)[np.argsort(-score)[: len(score) // 100]].sum() / np.sum(y)),
            "share_of_fraud_in_top_5pct": float(np.asarray(y)[np.argsort(-score)[: len(score) // 20]].sum() / np.sum(y))}


def stage_select(run_dir, out_dir):
    frame = load_dense(FEATURES_V2)
    frame = frame[frame["day_bucket"] < min(SPLITS["held_out"])].copy()
    train, eval_1, eval_2 = (get_split(frame, name, FEATURES_V2) for name in ("train", "eval_1", "eval_2"))

    log("[autoencoder] training without labels")
    autoencoders, per_seed = {}, {}
    for seed in FINAL_SEEDS:
        start = time.time()
        ae = TabularAutoencoder(seed=seed).fit(train[0], eval_1[0])   # no y anywhere
        autoencoders[seed] = ae
        per_seed[seed] = {
            "best_epoch": ae.best_epoch_, "eval_1_reconstruction_loss": ae.validation_loss_,
            "n_train_rows": ae.n_train_rows_used_,
            "eval_1_pr_auc": pr_auc(eval_1[1], ae.anomaly_score(eval_1[0])),
            "eval_2_pr_auc": pr_auc(eval_2[1], ae.anomaly_score(eval_2[0])),
        }
        log(f"  seed {seed}: epoch {ae.best_epoch_}, eval_1 loss {ae.validation_loss_:.5f}, "
            f"eval_1 PR-AUC {per_seed[seed]['eval_1_pr_auc']:.4f}, eval_2 PR-AUC {per_seed[seed]['eval_2_pr_auc']:.4f} "
            f"[{time.time() - start:.0f}s]")

    ae = autoencoders[FINAL_SEEDS[0]]
    joblib.dump({"autoencoders": autoencoders, "feature_columns": FEATURES_V2}, run_dir / "autoencoder.joblib")
    standalone = {
        "config": {**AE_PARAMS, "input_features": FEATURES_V2, "trained_on": "train window (days 5-7), all rows, no labels",
                   "early_stopping_on": "eval_1 reconstruction loss (no labels)",
                   "legit_subsampling": "none: all 1,311,225 training rows used"},
        "per_seed": per_seed,
        "eval_1_seed_42": ranking_report(eval_1[1], ae.anomaly_score(eval_1[0])),
        "eval_2_seed_42": ranking_report(eval_2[1], ae.anomaly_score(eval_2[0])),
        "base_rate_eval_1": float(eval_1[1].mean()),
    }
    log(f"  seed 42 alone: eval_1 PR-AUC {standalone['eval_1_seed_42']['pr_auc']:.4f} "
        f"(base rate {standalone['base_rate_eval_1']:.5f}), "
        f"fraud in top 1% = {standalone['eval_1_seed_42']['share_of_fraud_in_top_1pct']:.3f}")

    log("\n[hybrid] LightGBM on leak-free features + anomaly score")
    frame[ANOMALY_COLUMN] = ae.anomaly_score(frame[FEATURES_V2])
    h_train, h_eval_1, h_eval_2 = (get_split(frame, name, HYBRID_FEATURES) for name in ("train", "eval_1", "eval_2"))
    eval_2_days = frame.loc[h_eval_2[0].index, "day_bucket"].to_numpy()
    hybrid = select_family(FAMILIES["lightgbm"], h_train, h_eval_1, h_eval_2, eval_2_days, log=log)
    hybrid["label"] = "Hybrid: LightGBM + anomaly score"
    joblib.dump(hybrid, run_dir / "hybrid.joblib", compress=3)

    payload = {"autoencoder": standalone, "hybrid": strip(hybrid)}
    (out_dir / "selection.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    record(f"{TAG}_selection", payload, f"{SOURCE} --stage select")


def stage_final(run_dir, out_dir, force):
    path = out_dir / "final.json"
    if path.exists() and not force:
        raise SystemExit(f"{path} already exists: held-out has been used for this phase. Use --force to overwrite.")
    settings = get_settings()
    frame = load_dense(FEATURES_V2)
    held = frame[frame["day_bucket"].isin(SPLITS["held_out"])].copy()
    y = held["isFraud"].to_numpy()
    amounts = held["amount"].to_numpy()

    saved = joblib.load(run_dir / "autoencoder.joblib")
    scores = {seed: ae.anomaly_score(held[FEATURES_V2]) for seed, ae in saved["autoencoders"].items()}
    anomaly = scores[FINAL_SEEDS[0]]
    seed_pr_auc = [pr_auc(y, s) for s in scores.values()]
    autoencoder = {
        "seed_42": ranking_report(y, anomaly),
        "seeds": {"values": seed_pr_auc, "mean": float(np.mean(seed_pr_auc)), "std": float(np.std(seed_pr_auc))},
        "five_seed_mean_score": ranking_report(y, np.mean(list(scores.values()), axis=0)),
        "base_rate": float(y.mean()),
    }
    log(f"[autoencoder] held-out PR-AUC {autoencoder['seed_42']['pr_auc']:.4f} "
        f"[{autoencoder['seed_42']['ci_low']:.4f}, {autoencoder['seed_42']['ci_high']:.4f}]; "
        f"5 seeds {autoencoder['seeds']['mean']:.4f} +/- {autoencoder['seeds']['std']:.4f}")

    held[ANOMALY_COLUMN] = anomaly
    hybrid = evaluate_family(joblib.load(run_dir / "hybrid.joblib"), (held[HYBRID_FEATURES], held["isFraud"]))
    point = hybrid["ensemble"]["operating_points"]["robust"]
    log(f"[hybrid] ensemble PR-AUC {hybrid['ensemble']['pr_auc']:.4f} "
        f"[{hybrid['ensemble']['ci_low']:.4f}, {hybrid['ensemble']['ci_high']:.4f}], cost {point['total_cost']:,.0f}, "
        f"fraud allowed {point['fraud_allowed']}, legit reviewed {point['legit_reviewed_pct']:.2%}")

    # Supervised-only reference: the Phase 2 LightGBM ensemble, scored on the same rows.
    reference = np.load(settings.models / "runs" / SUPERVISED_TAG / "held_out_scores.npz")
    assert np.array_equal(reference["y"], y)
    supervised = reference[f"{SUPERVISED_FAMILY}__ensemble"]
    comparison = paired_bootstrap_difference(y, hybrid["_raw"]["ensemble"], supervised)
    log(f"[hybrid - supervised] PR-AUC difference {comparison['difference']:+.4f} "
        f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}], significant: {comparison['significant']}")

    # Does the label-free detector see the frauds the supervised model is least sure about?
    fraud_idx = np.flatnonzero(y == 1)
    lowest = fraud_idx[np.argsort(supervised[fraud_idx])[:LOWEST_SCORED]]
    percentile = (np.searchsorted(np.sort(anomaly), anomaly[lowest], side="right") / len(anomaly))
    all_fraud_percentile = np.searchsorted(np.sort(anomaly), anomaly[fraud_idx], side="right") / len(anomaly)
    blind_spot = {
        "definition": f"the {LOWEST_SCORED} held-out frauds (5%) with the lowest supervised ensemble score",
        "n": int(len(lowest)), "amount": float(amounts[lowest].sum()),
        "anomaly_percentile_median": float(np.median(percentile)),
        "in_top_1pct_most_anomalous": int((percentile >= 0.99).sum()),
        "in_top_5pct_most_anomalous": int((percentile >= 0.95).sum()),
        "in_top_10pct_most_anomalous": int((percentile >= 0.90).sum()),
        "all_fraud_anomaly_percentile_median": float(np.median(all_fraud_percentile)),
        "all_fraud_in_top_5pct_most_anomalous": int((all_fraud_percentile >= 0.95).sum()),
        "hybrid_score_higher_than_supervised": int((hybrid["_raw"]["ensemble"][lowest] > supervised[lowest]).sum()),
    }
    log(f"[blind spot] of the {len(lowest)} frauds the supervised model scores lowest, "
        f"{blind_spot['in_top_5pct_most_anomalous']} are in the top 5% most anomalous "
        f"(median anomaly percentile {blind_spot['anomaly_percentile_median']:.3f})")

    payload = {
        "split": "held_out (days 14-16)", "n_rows": int(len(y)), "n_fraud": int(y.sum()),
        "autoencoder": autoencoder, "hybrid": strip(hybrid),
        "hybrid_minus_supervised": {**comparison, "supervised": f"{SUPERVISED_TAG}/{SUPERVISED_FAMILY} ensemble",
                                    "supervised_pr_auc": pr_auc(y, supervised)},
        "supervised_blind_spot": blind_spot,
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    record(f"{TAG}_final", payload, f"{SOURCE} --stage final")
    np.savez_compressed(run_dir / "held_out_scores.npz", y=y, amount=amounts, anomaly=anomaly,
                        hybrid__ensemble=hybrid["_raw"]["ensemble"], hybrid__ensemble_probs=hybrid["_raw"]["ensemble_probs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["select", "final"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    run_dir, out_dir = settings.models / "runs" / TAG, settings.results / TAG
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "select":
        stage_select(run_dir, out_dir)
    else:
        stage_final(run_dir, out_dir, args.force)


if __name__ == "__main__":
    main()
