"""Loading PaySim, the dense window and the time-ordered splits."""

from collections.abc import Sequence

import pandas as pd

from fraud.config import get_settings
from fraud.features import (
    LEGACY_HOURLY,
    RECIPIENT_PROFILE,
    TIME_OF_DAY,
    build_feature_frame,
    recipient_profile_features,
    time_features,
)

TARGET = "isFraud"

# Days 5-16 are the only stretch with steady daily volume (see 01_data_audit).
DENSE_DAYS = (5, 16)
SPLITS = {
    "train": [5, 6, 7],
    "eval_1": [8, 9, 10],      # model selection: weights, hyperparameters, early stopping
    "eval_2": [11, 12, 13],    # calibration and thresholds
    "held_out": [14, 15, 16],  # touched once per final model
}

RAW_DTYPES = {
    "step": "int64", "type": "string", "amount": "float64",
    "nameOrig": "string", "oldbalanceOrg": "float64", "newbalanceOrig": "float64",
    "nameDest": "string", "oldbalanceDest": "float64", "newbalanceDest": "float64",
    "isFraud": "int64", "isFlaggedFraud": "int64",
}


def load_raw(path=None) -> pd.DataFrame:
    """The raw PaySim CSV, in its original (chronological) row order."""
    return pd.read_csv(path or get_settings().raw, dtype=RAW_DTYPES)


def load_feature_frame(columns: Sequence[str] | None = None, rebuild: bool = False) -> pd.DataFrame:
    """The engineered feature table for all 6.36M transactions.

    Reads the parquet cache when it exists, otherwise builds it from the raw
    CSV with ``fraud.features``. The cache is never modified in place: columns
    it lacks (the hour-of-day features) are added in memory.
    ``scripts/verify_features.py`` checks the cache against a fresh build.
    """
    settings = get_settings()
    if rebuild or not settings.features.exists():
        frame = build_feature_frame(load_raw())
        return frame if columns is None else frame[list(columns)]

    import pyarrow.parquet as pq

    cached = set(pq.read_schema(settings.features).names)
    wanted = list(cached) if columns is None else list(columns)
    need_time = columns is None or any(c in TIME_OF_DAY and c not in cached for c in wanted)
    need_profile = columns is None or any(c in RECIPIENT_PROFILE and c not in cached for c in wanted)

    read = [c for c in wanted if c in cached]
    helpers = ["step"] + (["nameDest", "type_TRANSFER"] if need_profile else [])
    read += [c for c in helpers if c not in read]
    frame = pd.read_parquet(settings.features, columns=read)

    # Columns the cache does not store are computed here (the cache file is never changed).
    if need_time:
        frame = frame.join(time_features(frame))
    if need_profile:
        frame = frame.join(recipient_profile_features(frame))
    return frame if columns is None else frame[wanted]


def dense_window(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["day_bucket"].between(*DENSE_DAYS)]


def load_dense(feature_columns: Sequence[str], extra: Sequence[str] = ()) -> pd.DataFrame:
    """Dense-window rows with the given features plus the split/label columns."""
    needed = list(dict.fromkeys([*feature_columns, *extra, "step", "day_bucket", TARGET]))
    return dense_window(load_feature_frame(needed)).copy()


def get_days(frame: pd.DataFrame, days: Sequence[int], feature_columns: Sequence[str]):
    subset = frame[frame["day_bucket"].isin(list(days))]
    return subset[list(feature_columns)], subset[TARGET]


def get_split(frame: pd.DataFrame, name: str, feature_columns: Sequence[str]):
    """(X, y) for one of the named splits in ``SPLITS``."""
    return get_days(frame, SPLITS[name], feature_columns)


__all__ = [
    "DENSE_DAYS", "LEGACY_HOURLY", "SPLITS", "TARGET", "dense_window", "get_days", "get_split",
    "load_dense", "load_feature_frame", "load_raw",
]
