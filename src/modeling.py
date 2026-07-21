"""Thin CatBoost wrapper: log1p-target training with expm1 inverse at predict time.

Target is heavily right-skewed (see EDA section 2), so the model is fit on
log1p(target) with MAE loss (log-MAE approximates WMAE reasonably well in
this space) and `sample_weight=w` is passed straight through to CatBoost.
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
