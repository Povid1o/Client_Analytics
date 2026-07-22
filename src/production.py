"""Shared, validated building blocks for the two production champions.

This module deliberately contains no experiment selection and no entry point.
The exact full and compact ensembles are assembled explicitly in
``train_full_champion.py`` and ``train_compact_champion.py``.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import (
    CATEGORICAL_COLS,
    DATE_COL,
    NON_FEATURE_COLS,
    RANDOM_SEED,
)
from src.feature_engineering import add_feature_groups
from src.modeling import PowerTargetCatBoost, prepare_cat_features
from src.postprocessing import blend_salary_signal


TAIL_THRESHOLDS = (150_000.0, 300_000.0, 500_000.0)
PRODUCTION_FEATURE_GROUPS = ("anchors", "scale", "flows", "trends", "log_rank")


def build_catboost_matrix(frame):
    """Prepare the raw/preprocessed feature matrix used by CatBoost blocks."""
    features = [column for column in frame.columns if column not in NON_FEATURE_COLS]
    matrix = frame[features].copy()
    matrix[DATE_COL] = matrix[DATE_COL].astype(str)
    return prepare_cat_features(matrix, CATEGORICAL_COLS)


def build_engineered_lgbm_matrices(train, inference):
    """Create aligned numeric matrices with the validated feature families."""
    combined = pd.concat([train, inference], ignore_index=True, sort=False)
    combined = add_feature_groups(combined, PRODUCTION_FEATURE_GROUPS)
    features = [column for column in combined.columns if column not in NON_FEATURE_COLS]
    matrix = combined[features].copy()
    for column in CATEGORICAL_COLS:
        if column in matrix:
            values = matrix[column].astype(str).replace(
                {"nan": "missing", "NaT": "missing"}
            )
            matrix[column] = pd.Categorical(values).codes.astype("int32")
    return matrix.iloc[: len(train)].copy(), matrix.iloc[len(train) :].copy()


def calibrate(prediction, frame, target_range):
    """Apply the validated salary-anchor blend and legal target clipping."""
    prediction = blend_salary_signal(
        prediction,
        pd.to_numeric(frame["salary_6to12m_avg"], errors="coerce").to_numpy(),
        alpha=0.6,
    )
    return np.clip(prediction, *target_range)


def fit_base_model(features, target, weights, iterations):
    model = PowerTargetCatBoost(
        target_power=0.25,
        cat_features=CATEGORICAL_COLS,
        params={
            "iterations": iterations,
            "random_seed": RANDOM_SEED + 10,
            "early_stopping_rounds": None,
        },
    )
    model.fit(features, target, weights)
    return model


def fit_tail_classifier(features, target, weights, threshold, iterations, seed):
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=iterations,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=10,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        Pool(
            features,
            label=(target >= threshold).astype(int),
            weight=weights,
            cat_features=CATEGORICAL_COLS,
        )
    )
    return model


def fit_specialist(
    features,
    target,
    weights,
    selected,
    iterations,
    seed,
    l2_leaf_reg=20,
):
    model = PowerTargetCatBoost(
        target_power=0.25,
        cat_features=CATEGORICAL_COLS,
        params={
            "iterations": iterations,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": l2_leaf_reg,
            "random_seed": seed,
            "early_stopping_rounds": None,
        },
    )
    indices = np.flatnonzero(selected)
    model.fit(features.iloc[indices], target[indices], weights[indices])
    return model


def classifier_params(iterations, seed):
    """LightGBM parameters for every node of the hierarchical router."""
    return {
        "objective": "binary",
        "n_estimators": iterations,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_child_samples": 80,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 12.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def band_expert_params(iterations, seed):
    """LightGBM parameters for the eight full-champion band experts."""
    return {
        "objective": "regression",
        "n_estimators": iterations,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 15.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
