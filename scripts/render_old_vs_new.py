"""Write results/old_vs_new.md from the protocol results.

Old figures are the ones reported in the README and notebooks 02, 05 and 07.
New figures are read from results/<tag>/final.json and selection.json; nothing
is typed in by hand.

    python scripts/render_old_vs_new.py --tags phase1
"""

import argparse
import json

from fraud.config import get_settings
from fraud.reporting import OLD, OLD_FLAWS

TAG_TITLES = {
    "phase1": "Same 17 features, honest protocol",
    "phase2": "Leak-free features (hourly totals replaced by hour of day), honest protocol",
}


def money(x):
    return f"{x:,.0f}"


def pct(x):
    return f"{100 * x:.2f}%"


def g(x):
    return f"{x:.4g}"


def section(tag, final, selection):
    lines = [f"## {TAG_TITLES.get(tag, tag)}", ""]
    lines += [f"Source: `results/{tag}/final.json` (held-out days 14-16, {final['n_rows']:,} rows, "
              f"{final['n_fraud']} fraud), written by `scripts/run_protocol.py --tag {tag}`.", ""]

    lines += ["### Held-out PR-AUC", "",
              "| Model | Old | Old 95% CI | New: 5-seed mean ± std | New: seed 42 [95% CI] | New: 5-seed ensemble [95% CI] |",
              "|---|---|---|---|---|---|"]
    for name, fam in final["families"].items():
        old = OLD.get(name)
        s, e, s42 = fam["seed_summary"]["pr_auc"], fam["ensemble"], fam["seeds"]["42"]
        lines.append(
            f"| {fam['label']} | {old['pr_auc']:.4f} | {old['ci'][0]:.4f} to {old['ci'][1]:.4f} "
            if old else f"| {fam['label']} | not reported | not reported ")
        lines[-1] += (f"| {s['mean']:.4f} ± {s['std']:.4f} | {s42['pr_auc']:.4f} [{s42['ci_low']:.4f}, {s42['ci_high']:.4f}] "
                      f"| {e['pr_auc']:.4f} [{e['ci_low']:.4f}, {e['ci_high']:.4f}] |")
    lines.append("")

    lines += ["### What was chosen, and where", "",
              "| Model | Old class weight | New class weight (chosen on eval_1) | Calibration (chosen on eval_2) | Old flaw |",
              "|---|---|---|---|---|"]
    for name, fam in final["families"].items():
        old = OLD.get(name)
        lines.append(f"| {fam['label']} | {old['weight'] if old else 'balanced'} | {fam['weight']} "
                     f"| {fam['ensemble']['calibration_method']} | {OLD_FLAWS.get(name, 'not previously evaluated on held-out')} |")
    lines.append("")

    lines += ["### Operating point on held-out (thresholds fixed on eval_2, 5-seed ensemble)", "",
              "| Model | | Review ≥ | Block ≥ | Total cost | Fraud let through | Legit reviewed | Legit blocked | Block precision | Block recall | Block F1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, fam in final["families"].items():
        old = OLD.get(name)
        if old:
            lines.append(f"| {fam['label']} | old | {g(old['t_review'])} | {g(old['t_block'])} | {money(old['total_cost'])} "
                         f"| {old['fraud_allowed']} | {pct(old['legit_reviewed_pct'])} | {old['legit_blocked']} "
                         f"| {old['block_precision']:.4f} | {old['block_recall']:.4f} | {old['block_f1']:.4f} |")
        p = fam["ensemble"]["operating_points"]["robust"]
        lines.append(f"| {fam['label']} | **new** | {g(p['t_review'])} | {g(p['t_block'])} | {money(p['total_cost'])} "
                     f"| {p['fraud_allowed']} | {pct(p['legit_reviewed_pct'])} | {p['legit_blocked']} "
                     f"| {p['block_precision']:.4f} | {p['block_recall']:.4f} | {p['block_f1']:.4f} |")
    lines += ["", "Old thresholds are on raw model scores; new thresholds are on calibrated probabilities, "
              "so the two threshold columns are not on the same scale.", ""]

    lines += ["### Seed spread at the deployed thresholds (each seed calibrated and thresholded on eval_2)", "",
              "| Model | Total cost: mean ± std | min to max | Fraud let through: mean (min to max) | Legit reviewed: mean |",
              "|---|---|---|---|---|"]
    for fam in final["families"].values():
        c, f, r = (fam["seed_summary"][k] for k in ("total_cost", "fraud_allowed", "legit_reviewed_pct"))
        lines.append(f"| {fam['label']} | {money(c['mean'])} ± {money(c['std'])} | {money(c['min'])} to {money(c['max'])} "
                     f"| {f['mean']:.1f} ({f['min']:.0f} to {f['max']:.0f}) | {pct(r['mean'])} |")
    lines.append("")

    lines += ["### Threshold robustness on held-out (ensemble; cost change when one threshold moves)", "",
              "| Model | Threshold | −0.05 | −0.02 | +0.02 | +0.05 | odds ÷2 | odds ×2 |", "|---|---|---|---|---|---|---|---|"]
    for fam in final["families"].values():
        rows = fam["ensemble"]["operating_points"]["robust"]["robustness"]
        for which in ("t_review", "t_block"):
            cells = []
            for kind, amount in (("absolute", -0.05), ("absolute", -0.02), ("absolute", 0.02), ("absolute", 0.05),
                                 ("odds_factor", 0.5), ("odds_factor", 2.0)):
                hit = [r for r in rows if r["moved"] == which and r["kind"] == kind and abs(r["amount"] - amount) < 1e-9]
                cells.append(f"{hit[0]['cost_change']:+,.0f}" if hit else "n/a")
            lines.append(f"| {fam['label']} | {which.replace('t_', '')} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["### Is one model better than another? (paired bootstrap on held-out PR-AUC, ensembles)", "",
              "| A | B | PR-AUC(A) − PR-AUC(B) | 95% CI of the difference | Significant? |", "|---|---|---|---|---|"]
    labels = {n: f["label"] for n, f in final["families"].items()}
    for row in final["paired_bootstrap_ensemble"]:
        lines.append(f"| {labels[row['a']]} | {labels[row['b']]} | {row['difference']:+.4f} "
                     f"| {row['ci_low']:+.4f} to {row['ci_high']:+.4f} | {'yes' if row['significant'] else 'no'} |")
    lines.append("")

    lines += ["### Blind spot (real fraud given a raw score below 0.1%)", "",
              "| Model | Held-out frauds below 0.1% | Their total amount | Of those, allowed at the deployed threshold |",
              "|---|---|---|---|"]
    for fam in final["families"].values():
        b = fam["ensemble"]["blind_spot"]
        lines.append(f"| {fam['label']} (ensemble) | {b['n']} of {final['n_fraud']} | {money(b['amount'])} "
                     f"| {b['n_allowed_at_deployed_threshold']} |")
    lines += ["", "Old figure: LightGBM (weight 20) scored 37 of 822 held-out frauds below 0.1%, worth 48,707,865.", ""]

    sweep_lines = ["### Class-weight sweeps on eval_1 (mean PR-AUC over 3 seeds)", ""]
    for fam in selection.values():
        cells = ", ".join(f"{r['weight']} → {r['mean']:.4f}" for r in fam["weight_sweep"])
        sweep_lines.append(f"- **{fam['label']}**: {cells}. Chosen: {fam['weight']}.")
    return lines + sweep_lines + [""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=["phase1"])
    args = parser.parse_args()
    results = get_settings().results

    out = ["# Old numbers against honest numbers", "",
           "Every \"new\" figure below is read from a results file written by code in this repository. "
           "\"Old\" figures are the ones previously reported in the README and notebooks 02, 05 and 07.", "",
           "**What changed in the protocol.** Previously the class weights of LightGBM and Random Forest and the "
           "REVIEW / BLOCK thresholds of all models were chosen by looking at the held-out set (days 14-16). Now class "
           "weights are chosen on eval_1 (days 8-10), calibration and thresholds on eval_2 (days 11-13), and the "
           "held-out set is scored once. Results are reported over 5 training seeds and for the 5-seed "
           "probability-averaged ensemble.", ""]
    for tag in args.tags:
        final = json.loads((results / tag / "final.json").read_text(encoding="utf-8"))
        selection = json.loads((results / tag / "selection.json").read_text(encoding="utf-8"))
        out += section(tag, final, selection)
    extra = results / "old_vs_new_notes.md"
    if extra.exists():
        out += [extra.read_text(encoding="utf-8").rstrip(), ""]
    (results / "old_vs_new.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote", results / "old_vs_new.md")


if __name__ == "__main__":
    main()
