"""Turning a model bundle into decisions.

``Scorer`` is the single scoring path. The evaluation scripts and the API both
call it, so a probability computed offline and one returned by the service
come from the same code: features in the bundle's column order -> mean raw
score of the bundle's models -> calibrator -> tier.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fraud.evaluation import assign_tier
from fraud.features import compute_online_features, to_model_frame
from fraud.models import predict_fraud_proba

ANOMALY_COLUMN = "anomaly_score"

FEATURE_EXPLANATIONS = {
    "amount": "the transaction amount",
    "destination_transactions_last_24h": "how often this recipient has received money recently",
    "destination_transactions_last_7d": "this recipient's weekly transaction activity",
    "destination_avg_previous_amount": "how this amount compares to what this recipient usually receives",
    "destination_amount_deviation": "this amount being unusual for this recipient's history",
    "destination_is_first_transaction": "this being the first transaction ever to this recipient",
    "origin_balance_error": "a mismatch in the sender's balance after this transaction",
    "destination_balance_error": "a mismatch in the recipient's balance after this transaction",
    "destination_balance_is_zero": "the recipient's account balance being zero",
    "hour_sin": "the time of day",
    "hour_cos": "the time of day",
    "type_CASH_IN": "this being a cash-in transaction",
    "type_CASH_OUT": "this being a cash-out transaction",
    "type_DEBIT": "this being a debit transaction",
    "type_PAYMENT": "this being a payment",
    "type_TRANSFER": "this being a transfer",
    ANOMALY_COLUMN: "this transaction looking unlike ordinary traffic",
}


# How a yes/no feature reads when it is false.
NEGATED_EXPLANATIONS = {
    "destination_is_first_transaction": "this recipient having received money before",
    "destination_balance_is_zero": "the recipient's account balance not being zero",
    "type_CASH_IN": "this not being a cash-in transaction",
    "type_CASH_OUT": "this not being a cash-out transaction",
    "type_DEBIT": "this not being a debit transaction",
    "type_PAYMENT": "this not being a payment",
    "type_TRANSFER": "this not being a transfer",
}


class Scorer:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        model = bundle["model"]
        self.models = list(model) if isinstance(model, (list, tuple)) else [model]
        self.calibrator = bundle.get("calibrator")
        self.feature_columns: list[str] = list(bundle["feature_columns"])
        self.autoencoder = bundle.get("autoencoder")
        self.autoencoder_columns = list(bundle.get("autoencoder_columns") or [])
        thresholds = bundle.get("thresholds") or {}
        self.t_review, self.t_block = thresholds.get("t_review"), thresholds.get("t_block")
        self.name, self.version = bundle.get("name"), bundle.get("version")
        self.staleness = dict(bundle.get("staleness") or {})
        self._explainers = None

    # --- features -----------------------------------------------------------

    @property
    def base_columns(self) -> list[str]:
        """Columns a caller must supply (everything except derived scores)."""
        wanted = [c for c in self.feature_columns if c != ANOMALY_COLUMN]
        return list(dict.fromkeys([*wanted, *self.autoencoder_columns]))

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray | None:
        if self.autoencoder is None:
            return None
        return self.autoencoder.anomaly_score(X[self.autoencoder_columns])

    def model_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if ANOMALY_COLUMN in self.feature_columns and ANOMALY_COLUMN not in X.columns:
            X = X.assign(**{ANOMALY_COLUMN: self.anomaly_score(X)})
        return X[self.feature_columns]

    # --- scores -------------------------------------------------------------

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        frame = self.model_frame(X)
        return np.mean([predict_fraud_proba(m, frame) for m in self.models], axis=0)

    def probability(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.raw_score(X)
        return raw if self.calibrator is None else self.calibrator.transform(raw)

    def tier(self, probability: float) -> str:
        return assign_tier(probability, self.t_review, self.t_block)

    def is_stale(self, current_day: int | None) -> bool:
        """True when the newest fraud labels the supervised model was trained
        on are older than the configured label-maturity age."""
        through, max_age = self.staleness.get("labels_through_day"), self.staleness.get("max_label_age_days")
        if current_day is None or through is None or max_age is None:
            return False
        return current_day - through > max_age

    def score_frame(self, X: pd.DataFrame, current_day: int | None = None) -> pd.DataFrame:
        """Raw score, calibrated probability, tier and (if the bundle has a
        detector) anomaly score for each row.

        Fallback rule: when the supervised model is stale, a transaction it
        would ALLOW is sent to REVIEW if the label-free anomaly score is above
        its threshold. The supervised tier is kept in ``supervised_tier``.
        """
        frame = self.model_frame(X)
        raw = np.mean([predict_fraud_proba(m, frame) for m in self.models], axis=0)
        prob = raw if self.calibrator is None else self.calibrator.transform(raw)
        out = pd.DataFrame({"raw_score": raw, "fraud_probability": prob}, index=X.index)
        out["tier"] = [self.tier(p) for p in prob]
        if ANOMALY_COLUMN in frame.columns:
            out[ANOMALY_COLUMN] = frame[ANOMALY_COLUMN].to_numpy()
        elif self.autoencoder is not None:
            out[ANOMALY_COLUMN] = self.anomaly_score(X)

        stale = self.is_stale(current_day)
        out["supervised_tier"] = out["tier"]
        out["model_stale"] = stale
        out["fallback_applied"] = False
        threshold = self.staleness.get("anomaly_review_threshold")
        if stale and threshold is not None and ANOMALY_COLUMN in out.columns:
            escalate = (out["tier"] == "ALLOW") & (out[ANOMALY_COLUMN] >= threshold)
            out.loc[escalate, "tier"] = "REVIEW"
            out["fallback_applied"] = escalate
        return out

    def score_transaction(self, history_steps: Sequence[int] = (), history_amounts: Sequence[float] = (),
                          **transaction) -> dict:
        """Score one raw transaction given the recipient's stored history."""
        features = compute_online_features(history_steps=history_steps, history_amounts=history_amounts,
                                           **transaction)
        row = to_model_frame([features], self.base_columns)
        scored = self.score_frame(row, current_day=int(transaction["step"]) // 24).iloc[0]
        return {"features": features, **scored.to_dict()}

    # --- explanations -------------------------------------------------------

    def _tree_explainers(self):
        if self._explainers is None:
            import shap
            self._explainers = [shap.TreeExplainer(m) for m in self.models]
        return self._explainers

    def explain(self, X_row: pd.DataFrame) -> dict:
        """Signed contribution of each input for one transaction.

        Tree models: SHAP values averaged over the bundle's models (log-odds
        for boosted trees, probability for a random forest). Other models:
        the change in raw score when a feature is reset to its typical value.

        Two things are tidied for the reader. The sine and cosine of the hour
        are one input, so their contributions are added and shown as "the time
        of day". A yes/no feature that is false is described as its absence
        ("this not being a payment"), since that is what the model reacted to.
        """
        frame = self.model_frame(X_row)
        try:
            values, base, units = self._shap(frame)
        except Exception:  # not a tree model
            values, base, units = self._occlusion(frame)
        row = frame.iloc[0]
        contributions, hour_effect = [], 0.0
        for feature, value in zip(self.feature_columns, values, strict=True):
            if feature in ("hour_sin", "hour_cos"):
                hour_effect += float(value)
                continue
            feature_value = row[feature].item()
            absent = feature in NEGATED_EXPLANATIONS and not feature_value
            label = NEGATED_EXPLANATIONS[feature] if absent else FEATURE_EXPLANATIONS.get(feature, feature)
            contributions.append({"feature": feature, "label": label, "feature_value": feature_value,
                                  "shap_value": float(value)})
        if "hour_sin" in row.index and "hour_cos" in row.index:
            hour = round(float(np.degrees(np.arctan2(row["hour_sin"], row["hour_cos"])) % 360 / 15)) % 24
            contributions.append({"feature": "hour_of_day", "label": FEATURE_EXPLANATIONS["hour_sin"],
                                  "feature_value": hour, "shap_value": hour_effect})
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        return {"base_value": float(base), "units": units, "contributions": contributions}

    def _shap(self, frame):
        all_values, bases = [], []
        for explainer in self._tree_explainers():
            sv = explainer.shap_values(frame)
            sv = np.asarray(sv[1] if isinstance(sv, list) else sv)[0]
            if sv.ndim == 2:          # random forest: one column per class
                sv = sv[:, 1]
            base = np.atleast_1d(explainer.expected_value)
            all_values.append(sv)
            bases.append(float(base[1] if len(base) > 1 else base[0]))
        values, base = np.mean(all_values, axis=0), float(np.mean(bases))
        raw = float(self.raw_score(frame)[0])
        units = "probability" if abs(base + values.sum() - raw) < 1e-6 else "log-odds"
        return values, base, units

    def _occlusion(self, frame):
        reference = self.bundle.get("reference_row")
        if reference is None:
            raise RuntimeError("This bundle has no reference_row, so non-tree models cannot be explained.")
        raw = float(np.mean([predict_fraud_proba(m, frame) for m in self.models]))
        values = []
        for column in self.feature_columns:
            changed = frame.copy()
            changed[column] = reference[column]
            values.append(raw - float(np.mean([predict_fraud_proba(m, changed) for m in self.models])))
        return np.array(values), raw - float(np.sum(values)), "probability"

    def top_reasons(self, X_row: pd.DataFrame, n: int = 3) -> list[str]:
        """Plain-English reasons: the inputs pushing this score up the most.

        "This not being a payment" style reasons are left out: they are true
        of the model but say nothing a reader does not get from the
        transaction's actual type.
        """
        reasons = []
        for c in sorted(self.explain(X_row)["contributions"], key=lambda c: c["shap_value"], reverse=True):
            absent_type = c["feature"].startswith("type_") and not c["feature_value"]
            if c["shap_value"] > 0 and not absent_type and c["label"] not in reasons:
                reasons.append(c["label"])
        return reasons[:n]

    def info(self) -> dict:
        return {
            "name": self.name, "version": self.version,
            "model_class": type(self.models[0]).__name__, "n_models": len(self.models),
            "n_features": len(self.feature_columns), "feature_columns": self.feature_columns,
            "calibration": getattr(self.calibrator, "method", None),
            "thresholds": {"t_review": self.t_review, "t_block": self.t_block},
            "has_anomaly_detector": self.autoencoder is not None,
            "staleness": self.staleness or None,
        }
