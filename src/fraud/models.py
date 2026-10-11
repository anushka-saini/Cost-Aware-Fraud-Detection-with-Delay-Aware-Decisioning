"""Model definitions. Every model exposes ``fit`` and ``predict_proba``."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from fraud.features import BINARY_FEATURES

# Trees are kept shallow on purpose: the training window has 778 fraud rows.
LGBM_PARAMS = dict(n_estimators=200, num_leaves=15, max_depth=5, min_child_samples=100,
                   reg_alpha=1.0, reg_lambda=1.0, verbose=-1)
RF_PARAMS = dict(n_estimators=200, max_depth=5, min_samples_leaf=100, n_jobs=-1)
MLP_PARAMS = dict(hidden=(64, 32, 16), dropout=0.2, learning_rate=1e-3, weight_decay=1e-5,
                  batch_size=4096, max_epochs=30, patience=5)


def make_logreg(seed: int = 42):
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)


def make_lgbm(seed: int = 42, scale_pos_weight: float = 20, **overrides):
    return lgb.LGBMClassifier(**{**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight,
                                 "random_state": seed, **overrides})


def make_rf(seed: int = 42, fraud_weight: float = 500, **overrides):
    return RandomForestClassifier(**{**RF_PARAMS, "class_weight": {0: 1, 1: fraud_weight},
                                     "random_state": seed, **overrides})


def predict_fraud_proba(model, X) -> np.ndarray:
    """Fraud-class probability as a 1-D array, for any model in this module.

    LightGBM models are called through their booster on a plain array in the
    model's own feature order. The result is identical to ``predict_proba``;
    it skips the scikit-learn wrapper, which dominates the cost of scoring a
    single transaction.
    """
    booster = getattr(model, "booster_", None)
    if booster is not None and hasattr(X, "columns"):
        return np.asarray(booster.predict(X[booster.feature_name()].to_numpy(dtype=np.float64)))
    return np.asarray(model.predict_proba(X))[:, 1]


# ---------------------------------------------------------------------------
# Neural network input scaling
# ---------------------------------------------------------------------------

class SignedLogScaler:
    """Signed log1p then standardise the continuous columns, using training
    statistics only. Binary and one-hot columns pass through as 0/1."""

    def fit(self, X: pd.DataFrame):
        self.columns_ = list(X.columns)
        self.continuous_ = [c for c in self.columns_ if c not in BINARY_FEATURES]
        self.binary_ = [c for c in self.columns_ if c in BINARY_FEATURES]
        logged = self._log(X)
        self.mean_ = logged.mean(axis=0)
        self.std_ = logged.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def _log(self, X):
        values = X[self.continuous_].to_numpy(dtype=np.float64)
        return np.sign(values) * np.log1p(np.abs(values))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        continuous = (self._log(X) - self.mean_) / self.std_
        binary = X[self.binary_].to_numpy(dtype=np.float64)
        return np.hstack([continuous, binary]).astype(np.float32)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def _mlp_net(n_inputs, hidden, dropout):
    layers, size = [], n_inputs
    for width in hidden:
        layers += [nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)]
        size = width
    layers.append(nn.Linear(size, 1))
    return nn.Sequential(*layers)


class MLPClassifier:
    """Feed-forward network with a weighted loss and early stopping on a
    validation set's PR-AUC. The best epoch's weights are kept."""

    def __init__(self, pos_weight: float = 20, seed: int = 42, **params):
        self.pos_weight, self.seed = pos_weight, seed
        self.params = {**MLP_PARAMS, **params}

    def fit(self, X, y, X_val, y_val, verbose: bool = False):
        p = self.params
        torch.manual_seed(self.seed)
        self.scaler_ = SignedLogScaler().fit(X)
        T, T_val = torch.from_numpy(self.scaler_.transform(X)), torch.from_numpy(self.scaler_.transform(X_val))
        t_y = torch.tensor(np.asarray(y), dtype=torch.float32)
        y_val = np.asarray(y_val)

        self.net_ = _mlp_net(T.shape[1], p["hidden"], p["dropout"])
        optimiser = torch.optim.Adam(self.net_.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(self.pos_weight)))
        generator = torch.Generator().manual_seed(self.seed)

        best_score, best_state, stale = -1.0, None, 0
        self.history_ = []
        for epoch in range(1, p["max_epochs"] + 1):
            self.net_.train()
            order = torch.randperm(len(T), generator=generator)
            for i in range(0, len(order), p["batch_size"]):
                idx = order[i:i + p["batch_size"]]
                optimiser.zero_grad()
                loss_fn(self.net_(T[idx]).squeeze(1), t_y[idx]).backward()
                optimiser.step()
            score = float(average_precision_score(y_val, self._predict_tensor(T_val)))
            self.history_.append(score)
            if verbose:
                print(f"    epoch {epoch:2d}  validation PR-AUC = {score:.4f}")
            if score > best_score:
                best_score, stale, self.best_epoch_ = score, 0, epoch
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                stale += 1
                if stale >= p["patience"]:
                    break
        self.net_.load_state_dict(best_state)
        self.validation_pr_auc_ = best_score
        return self

    def _predict_tensor(self, T, batch=262144):
        self.net_.eval()
        with torch.no_grad():
            return torch.cat([torch.sigmoid(self.net_(T[i:i + batch]).squeeze(1))
                              for i in range(0, len(T), batch)]).numpy().astype(np.float64)

    def predict_proba(self, X):
        p = self._predict_tensor(torch.from_numpy(self.scaler_.transform(X[self.scaler_.columns_])))
        return np.column_stack([1 - p, p])


class AveragingEnsemble:
    """Mean of the fraud probabilities of several fitted models."""

    def __init__(self, models):
        self.models = list(models)

    def predict_proba(self, X):
        p = np.mean([predict_fraud_proba(m, X) for m in self.models], axis=0)
        return np.column_stack([1 - p, p])
