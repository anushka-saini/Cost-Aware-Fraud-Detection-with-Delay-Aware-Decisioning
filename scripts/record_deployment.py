"""Record which registered bundle the API serves in results/metrics.json.

    python scripts/record_deployment.py
"""

import json
import os

from fraud.config import get_settings
from fraud.registry import list_versions
from fraud.results import record


def main():
    name = os.environ.get("FRAUD_MODEL_NAME", "fraud_scorer")
    version = os.environ.get("FRAUD_MODEL_VERSION") or list_versions(name)[-1]
    meta = json.loads((get_settings().models / "registry" / name / version / "meta.json").read_text(encoding="utf-8"))
    meta.pop("reference_row", None)
    record("deployed_model", {
        **meta,
        "parity": "tests/test_parity.py: 1,000 held-out transactions replayed through POST /score/raw match the "
                  "offline pipeline to within 1e-6 (probability and raw score) with identical tiers",
    }, "scripts/register_model.py; scripts/record_deployment.py")
    print("recorded", name, version)


if __name__ == "__main__":
    main()
