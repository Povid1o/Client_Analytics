"""CatBoost target-transform wrappers.

Target is heavily right-skewed (see EDA section 2), so the model is fit on
log1p(target) with MAE loss (log-MAE approximates WMAE reasonably well in
this space) and `sample_weight=w` is passed straight through to CatBoost.

``PowerTargetCatBoost`` is the stronger alternative found in the second
iteration.  Weighted RMSE after a mild power transform retains much more
information about the upper tail than log-MAE while remaining substantially
more stable than RMSE in the raw target scale.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.config import RANDOM_SEED

DEFAULT_PARAMS = dict(
    loss_function="MAE",
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    random_seed=RANDOM_SEED,
    verbose=False,
    early_stopping_rounds=50,
)

POWER_TARGET_DEFAULT_PARAMS = dict(
    loss_function="RMSE",
    iterations=1500,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=5,
    random_seed=RANDOM_SEED,
    verbose=False,
    early_stopping_rounds=100,
    allow_writing_files=False,
)


def prepare_cat_features(df, cat_cols):
    """Fill missing categorical values and cast to str for CatBoost."""
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("NaT", "missing").fillna("missing")
    return df


class LogTargetCatBoost:
    """CatBoostRegressor fit/predicted in log1p(target) space."""

    def __init__(self, cat_features=None, params=None):
        self.cat_features = list(cat_features or [])
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model = CatBoostRegressor(**self.params)

    def fit(self, X_train, y_train, sample_weight=None, eval_set=None):
        y_train_log = np.log1p(y_train)
        train_pool = Pool(X_train, label=y_train_log, weight=sample_weight, cat_features=self.cat_features)

        eval_pool = None
        if eval_set is not None:
            X_val, y_val, w_val = eval_set
            eval_pool = Pool(
                X_val, label=np.log1p(y_val), weight=w_val, cat_features=self.cat_features
            )

        self.model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
        return self

    def predict(self, X):
        pred_log = self.model.predict(X)
        pred = np.expm1(pred_log)
        return np.clip(pred, 0, None)

    def get_feature_importance(self):
        importances = self.model.get_feature_importance()
        names = self.model.feature_names_
        return pd.Series(importances, index=names).sort_values(ascending=False)


def power_transform_target(target, power):
    """Apply ``target ** power`` with validation suitable for income data."""
    target = np.asarray(target, dtype=float)
    if not 0 < power <= 1:
        raise ValueError("power must be in the interval (0, 1]")
    if np.any(~np.isfinite(target)) or np.any(target < 0):
        raise ValueError("target must contain finite non-negative values")
    return np.power(target, power)


def inverse_power_target(transformed, power):
    """Invert :func:`power_transform_target`, clipping numerical negatives."""
    transformed = np.asarray(transformed, dtype=float)
    if not 0 < power <= 1:
        raise ValueError("power must be in the interval (0, 1]")
    return np.power(np.clip(transformed, 0, None), 1.0 / power)


class PowerTargetCatBoost:
    """CatBoost RMSE model trained on a power-transformed target.

    A power of ``0.25`` is the best single model in local CV.  A blend with
    a ``0.5`` (square-root) model reduces model variance further.
    """

    def __init__(self, target_power=0.25, cat_features=None, params=None):
        if not 0 < target_power <= 1:
            raise ValueError("target_power must be in the interval (0, 1]")
        self.target_power = float(target_power)
        self.cat_features = list(cat_features or [])
        self.params = {**POWER_TARGET_DEFAULT_PARAMS, **(params or {})}
        self.model = CatBoostRegressor(**self.params)

    def fit(self, X_train, y_train, sample_weight=None, eval_set=None):
        train_pool = Pool(
            X_train,
            label=power_transform_target(y_train, self.target_power),
            weight=sample_weight,
            cat_features=self.cat_features,
        )

        eval_pool = None
        if eval_set is not None:
            X_val, y_val, w_val = eval_set
            eval_pool = Pool(
                X_val,
                label=power_transform_target(y_val, self.target_power),
                weight=w_val,
                cat_features=self.cat_features,
            )

        self.model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
        return self

    def predict(self, X):
        return inverse_power_target(self.model.predict(X), self.target_power)

    def get_feature_importance(self):
        importances = self.model.get_feature_importance()
        names = self.model.feature_names_
        return pd.Series(importances, index=names).sort_values(ascending=False)
