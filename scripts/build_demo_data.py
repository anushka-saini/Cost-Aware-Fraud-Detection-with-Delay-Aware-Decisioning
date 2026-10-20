"""Build the data the live demo replays: api/data/replay.csv and replay_history.csv.

replay.csv is a chronological slice of real held-out transactions that the
WS /stream endpoint scores one by one. Fraud is oversampled on purpose (the
true rate, 0.07%, would show about one fraud in 1,400 rows): the file holds a
random sample of legitimate held-out transactions, a random sample of held-out
frauds, and every other held-out transaction to the same recipients up to the
last sampled row, so each recipient's history is complete.

replay_history.csv is everything those recipients received before the
held-out period. The service loads it into its feature store on first start,
so the features it computes during the replay equal the offline ones.

    python scripts/build_demo_data.py
"""

import numpy as np
import pandas as pd

from fraud.config import get_settings
from fraud.data import SPLITS, load_feature_frame
from fraud.features import TRANSACTION_TYPES, TYPE_COLUMNS

N_LEGIT, N_FRAUD, SEED = 2500, 60, 7
RAW = ["step", "nameOrig", "nameDest", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def main():
    out_dir = get_settings().project_root / "api" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_feature_frame([*RAW, *TYPE_COLUMNS, "day_bucket", "isFraud"])
    held = frame[frame["day_bucket"].isin(SPLITS["held_out"])]
    sample = pd.concat([held[held["isFraud"] == 0].sample(N_LEGIT, random_state=SEED),
                        held[held["isFraud"] == 1].sample(N_FRAUD, random_state=SEED)]).sort_index()

    related = frame[frame["nameDest"].isin(set(sample["nameDest"])) & (frame.index <= sample.index.max())]
    replay = related[related.index >= held.index.min()].copy()
    history = related[related.index < held.index.min()]

    type_index = np.argmax(replay[TYPE_COLUMNS].to_numpy(), axis=1)
    replay["type"] = [TRANSACTION_TYPES[i] for i in type_index]
    replay = replay.rename(columns={"isFraud": "actual_fraud"})[["step", "type", *RAW[1:], "actual_fraud"]]
    replay.to_csv(out_dir / "replay.csv", index=False)
    history[["nameDest", "step", "amount"]].to_csv(out_dir / "replay_history.csv", index=False)

    print(f"replay.csv: {len(replay):,} transactions, {int(replay['actual_fraud'].sum())} fraud "
          f"({replay['actual_fraud'].mean():.2%}; true held-out rate {held['isFraud'].mean():.3%}), "
          f"steps {replay['step'].min()}-{replay['step'].max()}")
    print(f"replay_history.csv: {len(history):,} earlier transactions to {replay['nameDest'].nunique():,} recipients")


if __name__ == "__main__":
    main()
