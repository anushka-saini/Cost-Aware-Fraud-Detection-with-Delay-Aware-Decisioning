"""Run the evaluation protocol (fraud.protocol) for a feature set.

Two stages, deliberately separate:

    python scripts/run_protocol.py --tag phase1 --features v1 --stage select
    python scripts/run_protocol.py --tag phase1 --features v1 --stage final

``select`` uses train, eval_1 and eval_2 only: it drops the held-out days
before doing anything, chooses weights, calibration and thresholds, and saves
the fitted models. ``final`` loads those models and scores the held-out days
once. It refuses to run a second time for the same tag unless --force is given.
"""

import argparse
import json
import time

import joblib
import numpy as np

from fraud.config import get_settings
from fraud.data import SPLITS, get_split, load_dense
from fraud.features import FEATURE_SETS
from fraud.protocol import (
    FAMILIES,
    FINAL_SEEDS,
    ROBUSTNESS_DELTAS,
    SELECTION_SEEDS,
    THRESHOLD_ODDS_MARGIN,
    blind_spot_overlap,
    compare_families,
    evaluate_family,
    select_family,
)
from fraud.results import _jsonable, record

SOURCE = "scripts/run_protocol.py"


def strip(obj, drop=("models", "calibrator", "_raw")):
    """Copy of a result tree without fitted objects or raw score arrays."""
    if isinstance(obj, dict):
        return {str(k): strip(v, drop) for k, v in obj.items() if k not in drop}
    if isinstance(obj, (list, tuple)):
        return [strip(v, drop) for v in obj]
    return obj


def log(message):
    print(message, flush=True)


def stage_select(args, features, run_dir, out_dir):
    frame = load_dense(features)
    frame = frame[frame["day_bucket"] < min(SPLITS["held_out"])]   # held-out rows never reach this stage
    train, eval_1, eval_2 = (get_split(frame, name, features) for name in ("train", "eval_1", "eval_2"))
    eval_2_days = frame.loc[eval_2[0].index, "day_bucket"].to_numpy()
    for name, (X, y) in (("train", train), ("eval_1", eval_1), ("eval_2", eval_2)):
        log(f"{name:7s} {len(X):>10,} rows  {int(y.sum()):>4} fraud")

    summary = {}
    for name in args.families:
        start = time.time()
        log(f"\n[{name}]")
        selection = select_family(FAMILIES[name], train, eval_1, eval_2, eval_2_days, log=log)
        joblib.dump(selection, run_dir / f"{name}.joblib", compress=3)
        summary[name] = strip(selection)
        log(f"  done in {time.time() - start:.0f}s")

    path = out_dir / "selection.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(_jsonable(summary))
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    record(f"{args.tag}_selection", {
        "features": args.features, "n_features": len(features),
        "protocol": {"weights_chosen_on": "eval_1 (days 8-10), mean PR-AUC over selection seeds",
                     "calibration_and_thresholds_chosen_on": "eval_2 (days 11-13)",
                     "selection_seeds": SELECTION_SEEDS, "final_seeds": FINAL_SEEDS,
                     "threshold_rule": ("lowest worst-case cost when each threshold's odds are scaled by up to "
                                        f"{THRESHOLD_ODDS_MARGIN}x either way")},
        "families": existing,
    }, f"{SOURCE} --tag {args.tag} --features {args.features} --stage select")


def stage_final(args, features, run_dir, out_dir):
    path = out_dir / "final.json"
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists: the held-out set has been used for tag {args.tag!r}. "
                         "Use a new tag for a new model, or --force to overwrite.")
    frame = load_dense(features)
    held_out = get_split(frame, "held_out", features)
    y = held_out[1].to_numpy()
    log(f"held_out {len(y):,} rows  {int(y.sum())} fraud")

    results = {}
    for name in args.families:
        selection = joblib.load(run_dir / f"{name}.joblib")
        results[name] = evaluate_family(selection, held_out)
        r = results[name]
        point = r["ensemble"]["operating_points"]["robust"]
        log(f"[{name}] seeds PR-AUC {r['seed_summary']['pr_auc']['mean']:.4f} +/- {r['seed_summary']['pr_auc']['std']:.4f} | "
            f"ensemble PR-AUC {r['ensemble']['pr_auc']:.4f} [{r['ensemble']['ci_low']:.4f}, {r['ensemble']['ci_high']:.4f}] | "
            f"cost {point['total_cost']:,.0f}, fraud allowed {point['fraud_allowed']}, "
            f"legit reviewed {point['legit_reviewed_pct']:.2%}, legit blocked {point['legit_blocked']}")

    payload = {
        "features": args.features, "n_features": len(features),
        "split": "held_out (days 14-16)", "n_rows": int(len(y)), "n_fraud": int(y.sum()),
        "threshold_odds_margin": THRESHOLD_ODDS_MARGIN, "robustness_deltas": list(ROBUSTNESS_DELTAS),
        "families": strip(results),
        "paired_bootstrap_ensemble": compare_families(results, y, "ensemble"),
        "paired_bootstrap_seed_42": compare_families(results, y, "seed_42"),
        "blind_spot_overlap_ensemble": blind_spot_overlap(results, y),
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    record(f"{args.tag}_final", payload,
           f"{SOURCE} --tag {args.tag} --features {args.features} --stage final")
    np.savez_compressed(run_dir / "held_out_scores.npz", y=y, amount=held_out[0]["amount"].to_numpy(),
                        **{f"{n}__{k}": v for n, r in results.items() for k, v in r["_raw"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features", default="v1", choices=sorted(FEATURE_SETS))
    parser.add_argument("--stage", required=True, choices=["select", "final"])
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    features = FEATURE_SETS[args.features]
    run_dir = settings.models / "runs" / args.tag
    out_dir = settings.results / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (stage_select if args.stage == "select" else stage_final)(args, features, run_dir, out_dir)


if __name__ == "__main__":
    main()
