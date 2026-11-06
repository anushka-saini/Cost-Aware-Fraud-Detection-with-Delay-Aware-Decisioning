"""Copy the figures V1 reported into results/metrics.json.

V1's numbers were never produced by a script in this repository: they come
from the first README and notebooks 05 and 07. They are flawed (class
weights and thresholds were chosen on the held-out days), but they are part
of the project's history, and the dashboard shows V1 -> V2 -> V3. Recording
them here means every number on the dashboard is read from metrics.json.

    python scripts/record_v1_reported.py
"""

from fraud.reporting import OLD, OLD_DEPLOYED, OLD_FLAWS
from fraud.results import record

SOURCE = "scripts/record_v1_reported.py (figures as first reported: V1 README, notebooks 05 and 07)"


def main():
    models = {}
    for key, figures in OLD.items():
        models[key] = {**figures, "ci_low": figures["ci"][0], "ci_high": figures["ci"][1], "flaw": OLD_FLAWS[key]}
        del models[key]["ci"]
    record("v1_reported", {
        "split": "held_out (days 14-16)",
        "deployed": OLD_DEPLOYED,
        "warning": "class weights and thresholds were chosen by looking at the held-out days, so these "
                   "figures are optimistic and are not comparable with V2 or V3",
        "models": models,
    }, SOURCE)
    print("recorded v1_reported")


if __name__ == "__main__":
    main()
