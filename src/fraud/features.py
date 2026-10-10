"""Feature engineering, as pure functions.

There are two entry points and they share the same formulas:

* ``build_feature_frame`` computes features for a whole table of transactions
  (training and evaluation).
* ``compute_online_features`` computes them for one incoming transaction from
  the recipient's stored history (serving).

``tests/test_features.py`` checks that the two agree row for row.

Time semantics, inherited from the original pipeline and kept so results stay
comparable: PaySim's ``step`` is an hour index. "Previous" transactions for the
average/first-transaction features are all earlier rows to the same recipient,
including earlier rows in the same hour. The 24h / 7d counts cover the
preceding 24 / 168 whole hours and exclude the current hour.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TYPE_COLUMNS = [f"type_{t}" for t in TRANSACTION_TYPES]

DESTINATION_HISTORY = [
    "destination_transactions_last_24h",
    "destination_transactions_last_7d",
    "destination_avg_previous_amount",
    "destination_amount_deviation",
    "destination_is_first_transaction",
]
BALANCE = ["origin_balance_error", "destination_balance_error", "destination_balance_is_zero"]
# Totals over the whole hour: they include transactions that happen after the
# one being scored. Kept only to reproduce the original (v1) models.
LEGACY_HOURLY = ["total_transactions", "total_transaction_amount", "avg_transaction_amount"]
TIME_OF_DAY = ["hour_sin", "hour_cos"]

# v1: the original 17 features, in the order the saved models were trained on.
FEATURES_V1 = ["amount", *DESTINATION_HISTORY, *BALANCE, *LEGACY_HOURLY, *TYPE_COLUMNS]
# v2: leak-free. The hourly totals are replaced by the hour of day.
FEATURES_V2 = ["amount", *DESTINATION_HISTORY, *BALANCE, *TIME_OF_DAY, *TYPE_COLUMNS]
# v2 without the balance features (simulator-artefact check).
FEATURES_V2_NO_BALANCE = [f for f in FEATURES_V2 if f not in BALANCE]

FEATURE_SETS = {"v1": FEATURES_V1, "v2": FEATURES_V2, "v2_no_balance": FEATURES_V2_NO_BALANCE}

BINARY_FEATURES = {"destination_is_first_transaction", "destination_balance_is_zero", *TYPE_COLUMNS}

WINDOW_24H = 24
WINDOW_7D = 168


# ---------------------------------------------------------------------------
# Row-wise formulas (shared by the batch and online paths)
# ---------------------------------------------------------------------------

def origin_balance_error(old_balance, amount, new_balance):
    return old_balance - amount - new_balance


def destination_balance_error(old_balance, amount, new_balance):
    return old_balance + amount - new_balance


def destination_balance_is_zero(old_balance, new_balance):
    return (old_balance == 0) & (new_balance == 0)


def hour_of_day(step):
    return step % 24


def hour_sin_cos(step):
    angle = 2 * np.pi * hour_of_day(step) / 24
    return np.sin(angle), np.cos(angle)


def day_bucket(step, first_step: int = 1):
    return (step - first_step) // 24


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------

def balance_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "origin_balance_error": origin_balance_error(df["oldbalanceOrg"], df["amount"], df["newbalanceOrig"]),
        "destination_balance_error": destination_balance_error(
            df["oldbalanceDest"], df["amount"], df["newbalanceDest"]),
        "destination_balance_is_zero": destination_balance_is_zero(
            df["oldbalanceDest"], df["newbalanceDest"]).astype("int64"),
    }, index=df.index)


def time_features(df: pd.DataFrame) -> pd.DataFrame:
    sin, cos = hour_sin_cos(df["step"].to_numpy())
    return pd.DataFrame({"hour_sin": sin, "hour_cos": cos}, index=df.index)


def type_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"type_{t}": (df["type"] == t).to_numpy() for t in TRANSACTION_TYPES}, index=df.index)


def destination_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recipient-history features for every row, using only earlier transactions.

    ``df`` needs ``nameDest``, ``step`` and ``amount`` and must be in
    chronological row order (as the raw PaySim file is). The result is aligned
    to ``df.index``; nothing is joined by position on a re-sorted copy.
    """
    n = len(df)
    dest_codes, _ = pd.factorize(df["nameDest"], sort=False)
    step = df["step"].to_numpy(dtype=np.int64)
    amount = df["amount"].to_numpy(dtype=np.float64)

    # Stable sort by (recipient, step): ties keep the original row order.
    order = np.lexsort((np.arange(n), step, dest_codes))
    d, s, a = dest_codes[order], step[order], amount[order]

    new_group = np.r_[True, d[1:] != d[:-1]]
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(n), 0))
    prior_count = np.arange(n) - group_start

    # Running sum restarted per recipient (a single global cumsum loses precision).
    prior_sum = pd.Series(a).groupby(d, sort=False).cumsum().to_numpy() - a
    avg_previous = np.divide(prior_sum, prior_count, out=np.zeros(n), where=prior_count > 0)

    # Counts over [step - window, step): one sorted composite key, two searches.
    span = int(s.max()) + WINDOW_7D + 2
    key = d.astype(np.int64) * span + s + WINDOW_7D
    upper = np.searchsorted(key, key, side="left")

    def window_count(window):
        return upper - np.searchsorted(key, key - window, side="left")

    out = np.empty((n, 5))
    out[order, 0] = window_count(WINDOW_24H)
    out[order, 1] = window_count(WINDOW_7D)
    out[order, 2] = avg_previous
    out[order, 3] = a - avg_previous
    out[order, 4] = prior_count == 0
    result = pd.DataFrame(out, columns=DESTINATION_HISTORY, index=df.index)
    result["destination_is_first_transaction"] = result["destination_is_first_transaction"].astype("int64")
    return result


def legacy_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Whole-hour totals (leaky: see LEGACY_HOURLY). Only for reproducing v1."""
    grouped = df.groupby("step")["amount"]
    return pd.DataFrame({
        "total_transactions": grouped.transform("count"),
        "total_transaction_amount": grouped.transform("sum"),
        "avg_transaction_amount": grouped.transform("mean"),
    }, index=df.index)


def build_feature_frame(raw: pd.DataFrame, include_legacy_hourly: bool = True) -> pd.DataFrame:
    """All engineered features for a raw PaySim table, plus the columns needed
    for splitting and evaluation (``step``, ``day_bucket``, ``isFraud``, names)."""
    parts = [
        raw[["step", "amount", "nameOrig", "nameDest", "isFraud"]],
        pd.DataFrame({"day_bucket": day_bucket(raw["step"], int(raw["step"].min()))}, index=raw.index),
        destination_history_features(raw),
        balance_features(raw),
        time_features(raw),
        type_one_hot(raw),
    ]
    if include_legacy_hourly:
        parts.append(legacy_hourly_features(raw))
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Online path
# ---------------------------------------------------------------------------

def compute_online_features(
    *,
    step: int,
    type: str,
    amount: float,
    old_balance_origin: float,
    new_balance_origin: float,
    old_balance_destination: float,
    new_balance_destination: float,
    history_steps: Sequence[int] = (),
    history_amounts: Sequence[float] = (),
) -> dict:
    """Features for one incoming transaction.

    ``history_steps`` / ``history_amounts`` are the recipient's earlier
    transactions (any order). Returns every v2 feature keyed by name.
    """
    if type not in TRANSACTION_TYPES:
        raise ValueError(f"Unknown transaction type: {type!r}")
    steps = np.asarray(history_steps, dtype=np.int64)
    amounts = np.asarray(history_amounts, dtype=np.float64)
    if steps.shape != amounts.shape:
        raise ValueError("history_steps and history_amounts must have the same length")

    avg_previous = float(amounts.mean()) if len(amounts) else 0.0
    sin, cos = hour_sin_cos(step)
    features = {
        "amount": float(amount),
        "destination_transactions_last_24h": float(((steps >= step - WINDOW_24H) & (steps < step)).sum()),
        "destination_transactions_last_7d": float(((steps >= step - WINDOW_7D) & (steps < step)).sum()),
        "destination_avg_previous_amount": avg_previous,
        "destination_amount_deviation": float(amount) - avg_previous,
        "destination_is_first_transaction": int(len(amounts) == 0),
        "origin_balance_error": float(origin_balance_error(old_balance_origin, amount, new_balance_origin)),
        "destination_balance_error": float(
            destination_balance_error(old_balance_destination, amount, new_balance_destination)),
        "destination_balance_is_zero": int(
            destination_balance_is_zero(old_balance_destination, new_balance_destination)),
        "hour_sin": float(sin),
        "hour_cos": float(cos),
    }
    features.update({f"type_{t}": type == t for t in TRANSACTION_TYPES})
    return features


def to_model_frame(rows: Sequence[dict], feature_columns: Sequence[str]) -> pd.DataFrame:
    """Rows of feature dicts -> a frame with exactly the model's columns, in order."""
    missing = [c for c in feature_columns if any(c not in r for r in rows)]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    return pd.DataFrame(rows)[list(feature_columns)]
