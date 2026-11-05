"""Version 3 model search, using days 5-13 only.

How the days are used:

* Hyperparameters: two time-ordered folds inside days 5-10
  (train 5-7 -> validate 8-10, and train 5-8 -> validate 9-10).
  The score of a setting is its mean PR-AUC over the two folds.
* Final models: refitted on the chosen training window with 5 seeds.
* Days 11-13: calibration, thresholds and choosing between models.
* Days 14-16 (held-out) are never loaded here.
"""

import numpy as np
import optuna

from fraud.data import get_days, load_dense
from fraud.evaluation import pr_auc
from fraud.models import (
    ScaledLogisticRegression,
    make_catboost,
    make_lgbm,
    make_rf,
    make_xgb,
    predict_fraud_proba,
)
from fraud.nets import NeuralNetClassifier

FOLDS = [([5, 6, 7], [8, 9, 10]), ([5, 6, 7, 8], [9, 10])]
VALIDATION_DAYS = [11, 12, 13]
FINAL_SEEDS = [42, 1, 2, 3, 4]
FIRST_HELD_OUT_DAY = 14

NEURAL = {"mlp", "resnet", "ft_transformer", "tabm"}
LABELS = {
    "logreg": "Logistic regression", "random_forest": "Random Forest", "lightgbm": "LightGBM",
    "catboost": "CatBoost", "xgboost": "XGBoost", "mlp": "MLP", "resnet": "ResNet-MLP",
    "ft_transformer": "FT-Transformer", "tabm": "TabM",
}

# Optuna trials per family. Lower for the slow networks: on this laptop's CPU one
# FT-Transformer epoch takes about 90 s and one TabM epoch about 3 minutes.
# ResNet was cut from 30 to 10 trials on 5 Nov to keep model work inside its
# time budget (it had finished 5 trials by then).
TRIALS = {"logreg": 20, "random_forest": 30, "lightgbm": 60, "catboost": 60, "xgboost": 60,
          "mlp": 30, "resnet": 10, "ft_transformer": 10, "tabm": 10}

# Networks train on all fraud rows plus this share of legitimate rows (CPU time).
# Their early-stopping checks use 25% of the validation fold's legitimate rows.
NET_LEGIT_SHARE = {"mlp": 0.2, "resnet": 0.2, "ft_transformer": 0.05, "tabm": 0.05}
NET_VALIDATION_LEGIT_SHARE = 0.25


def load_search_frame(features):
    """Days 5-13 with the given features. The held-out days are dropped first."""
    frame = load_dense(features)
    return frame[frame["day_bucket"] < FIRST_HELD_OUT_DAY]


def amount_weights(X, y):
    """Sample weights: 1 for legitimate rows; for fraud rows log(1 + amount),
    scaled so the average fraud weight is 1. A missed fraud costs its amount,
    so bigger frauds count a bit more."""
    weights = np.ones(len(y))
    is_fraud = np.asarray(y) == 1
    log_amount = np.log1p(X["amount"].to_numpy()[is_fraud])
    weights[is_fraud] = log_amount / log_amount.mean()
    return weights


def keep_some_legit(y, share, seed):
    """Boolean mask: every fraud row and a random ``share`` of legitimate rows."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    return (y == 1) | (rng.random(len(y)) < share)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def suggest_params(family: str, trial: optuna.Trial) -> dict:
    s = trial
    if family == "logreg":
        return {"C": s.suggest_float("C", 1e-3, 10, log=True),
                "fraud_weight": s.suggest_float("fraud_weight", 1, 1000, log=True)}
    if family == "random_forest":
        return {"n_estimators": s.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth": s.suggest_int("max_depth", 4, 16),
                "min_samples_leaf": s.suggest_int("min_samples_leaf", 10, 500, log=True),
                "max_features": s.suggest_float("max_features", 0.3, 1.0),
                "fraud_weight": s.suggest_float("fraud_weight", 1, 1000, log=True)}
    if family == "lightgbm":
        return {"n_estimators": s.suggest_int("n_estimators", 100, 800, step=50),
                "learning_rate": s.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": s.suggest_int("num_leaves", 7, 63),
                "max_depth": s.suggest_int("max_depth", 3, 8),
                "min_child_samples": s.suggest_int("min_child_samples", 20, 1000, log=True),
                "subsample": s.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
                "colsample_bytree": s.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": s.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "reg_lambda": s.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "scale_pos_weight": s.suggest_float("scale_pos_weight", 1, 100, log=True)}
    if family == "xgboost":
        return {"n_estimators": s.suggest_int("n_estimators", 100, 800, step=50),
                "learning_rate": s.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": s.suggest_int("max_depth", 3, 8),
                "min_child_weight": s.suggest_float("min_child_weight", 1, 100, log=True),
                "subsample": s.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": s.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": s.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "reg_lambda": s.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "scale_pos_weight": s.suggest_float("scale_pos_weight", 1, 100, log=True)}
    if family == "catboost":
        return {"iterations": s.suggest_int("iterations", 100, 600, step=50),
                "learning_rate": s.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": s.suggest_int("depth", 4, 8),
                "l2_leaf_reg": s.suggest_float("l2_leaf_reg", 1, 30, log=True),
                "rsm": s.suggest_float("rsm", 0.5, 1.0),
                "scale_pos_weight": s.suggest_float("scale_pos_weight", 1, 100, log=True)}

    # Networks: shared training settings plus the architecture's own.
    params = {"learning_rate": s.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
              "weight_decay": s.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
              "pos_weight": s.suggest_float("pos_weight", 1, 100, log=True),
              "dropout": s.suggest_float("dropout", 0.0, 0.3)}
    if family == "mlp":
        params |= {"width": s.suggest_categorical("width", [64, 128, 256]), "n_layers": s.suggest_int("n_layers", 2, 4)}
    elif family == "resnet":
        params |= {"width": s.suggest_categorical("width", [64, 128, 256]), "n_blocks": s.suggest_int("n_blocks", 1, 3),
                   "hidden_factor": s.suggest_categorical("hidden_factor", [1, 2])}
    elif family == "ft_transformer":
        params |= {"d_token": s.suggest_categorical("d_token", [16, 32, 64]), "n_layers": s.suggest_int("n_layers", 1, 3)}
    elif family == "tabm":
        params |= {"width": s.suggest_categorical("width", [64, 128, 256]), "n_blocks": s.suggest_int("n_blocks", 1, 3)}
    return params


def build_model(family: str, params: dict, seed: int):
    params = dict(params)
    if family == "logreg":
        return ScaledLogisticRegression(seed=seed, **params)
    if family == "random_forest":
        return make_rf(seed, fraud_weight=params.pop("fraud_weight"), **params)
    if family == "lightgbm":
        return make_lgbm(seed, **params)
    if family == "xgboost":
        return make_xgb(seed, **params)
    if family == "catboost":
        return make_catboost(seed, **params)
    if family in NEURAL:
        return NeuralNetClassifier(family, seed=seed, **params)
    raise ValueError(f"Unknown family {family!r}")


def fit_model(family, params, seed, X, y, weights=None, X_val=None, y_val=None):
    """Fit one model. Networks see all fraud rows and a share of the legitimate
    ones; with validation data they stop early (see ``best_epoch_``)."""
    model = build_model(family, params, seed)
    if family not in NEURAL:
        return model.fit(X, y, sample_weight=weights)

    keep = keep_some_legit(y, NET_LEGIT_SHARE[family], seed)
    X, y = X[keep], np.asarray(y)[keep]
    weights = None if weights is None else np.asarray(weights)[keep]
    if X_val is not None:
        check = keep_some_legit(y_val, NET_VALIDATION_LEGIT_SHARE, seed)
        X_val, y_val = X_val[check], np.asarray(y_val)[check]
    return model.fit(X, y, X_val, y_val, sample_weight=weights)


# ---------------------------------------------------------------------------
# The search itself
# ---------------------------------------------------------------------------

def run_search(family, frame, features, use_amount_weights, n_trials, storage, log=print):
    """Optuna (TPE sampler) on the two time folds. Returns the finished study.

    A trial that is clearly worse than the median after the first fold is
    stopped early (Optuna's MedianPruner) to save time.
    """
    folds = []
    for train_days, val_days in FOLDS:
        X, y = get_days(frame, train_days, features)
        X_val, y_val = get_days(frame, val_days, features)
        weights = amount_weights(X, y) if use_amount_weights else None
        folds.append((X, y, weights, X_val, y_val))

    def objective(trial):
        params = suggest_params(family, trial)
        scores, epochs = [], []
        for i, (X, y, weights, X_val, y_val) in enumerate(folds):
            model = fit_model(family, params, 42, X, y, weights, X_val, y_val)
            scores.append(pr_auc(y_val, predict_fraud_proba(model, X_val)))
            if family in NEURAL:
                epochs.append(model.best_epoch_)
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_pr_auc", scores)
        if epochs:
            trial.set_user_attr("best_epochs", epochs)
        log(f"  trial {trial.number:3d}: mean fold PR-AUC {np.mean(scores):.4f}  {scores}")
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name=family, storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    return study


def final_params(family, study) -> dict:
    """The best trial's settings, ready for the final refit. Networks are
    refitted for the average of their best epochs on the two folds."""
    best = study.best_trial
    params = suggest_params(family, optuna.trial.FixedTrial(best.params))
    if family in NEURAL:
        params["epochs"] = max(1, int(round(np.mean(best.user_attrs["best_epochs"]))))
    return params
