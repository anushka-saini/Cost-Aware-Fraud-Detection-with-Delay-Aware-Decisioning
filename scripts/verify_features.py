"""Check that fraud.features reproduces the cached feature table from the raw CSV.

Rebuilds every v1 feature from data/raw/transaction_data.csv and compares it,
row by row, with data/processed/paysim_features.parquet (the table every
earlier notebook trained on). Writes the outcome to results/metrics.json.

    python scripts/verify_features.py
"""

import numpy as np
import pandas as pd

from fraud.config import get_settings
from fraud.data import load_raw
from fraud.features import FEATURES_V1, build_feature_frame
from fraud.results import record

SOURCE = "scripts/verify_features.py"


def main():
    settings = get_settings()
    raw = load_raw()
    print(f"Raw: {len(raw):,} rows, {int(raw['isFraud'].sum()):,} fraud")
    built = build_feature_frame(raw)
    del raw

    report, all_ok = {}, True
    for column in [*FEATURES_V1, "day_bucket", "isFraud", "step"]:
        cached = pd.read_parquet(settings.features, columns=[column])[column].to_numpy()
        fresh = built[column].to_numpy()
        both = np.asarray(cached, dtype=np.float64), np.asarray(fresh, dtype=np.float64)
        match = np.isclose(both[0], both[1], rtol=1e-9, atol=1e-6)
        share = float(match.mean())
        max_diff = float(np.abs(both[0] - both[1]).max())
        report[column] = {"share_matching": share, "max_abs_difference": max_diff}
        all_ok &= share == 1.0
        print(f"  {column:36s} match = {share:.6f}   max |diff| = {max_diff:.3g}")

    print("\nAll columns reproduce the cache exactly." if all_ok else "\nMISMATCH: see above.")
    record("feature_verification", {
        "n_rows": int(len(built)), "all_match": bool(all_ok), "columns": report,
        "note": "fraud.features.build_feature_frame on the raw CSV vs the cached parquet used by notebooks 02-07",
    }, SOURCE)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
