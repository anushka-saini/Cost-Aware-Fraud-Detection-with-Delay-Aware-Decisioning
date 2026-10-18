"""Regenerate demo_sample.csv: 500 held-out transactions (100 fraud, 400 legitimate).

Each row carries the raw transaction fields (for POST /score/raw), the
engineered features (for POST /score) and the true label.

    python scripts/make_demo_sample.py
"""

import pandas as pd

from fraud.config import get_settings
from fraud.data import SPLITS, load_feature_frame
from fraud.features import FEATURES_V2, TRANSACTION_TYPES

RAW = ["step", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def main():
    frame = load_feature_frame([*RAW, *FEATURES_V2, "day_bucket", "isFraud"])
    held = frame[frame["day_bucket"].isin(SPLITS["held_out"])]
    sample = pd.concat([held[held["isFraud"] == 1].sample(100, random_state=1),
                        held[held["isFraud"] == 0].sample(400, random_state=1)]).reset_index(drop=True)
    sample.insert(1, "type", [next(t for t in TRANSACTION_TYPES if row[f"type_{t}"]) for _, row in sample.iterrows()])
    sample = sample.rename(columns={"isFraud": "actual_fraud"}).drop(columns="day_bucket")
    path = get_settings().project_root / "demo_sample.csv"
    sample.to_csv(path, index=False)
    print(path, sample.shape, sample["actual_fraud"].value_counts().to_dict())


if __name__ == "__main__":
    main()
