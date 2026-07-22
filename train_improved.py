"""Train the improved power-target ensemble and create a submission.

Examples:
    python3 train_improved.py
    python3 train_improved.py --cv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    CSV_READ_KWARGS,
    DATE_COL,
    ID_COL,
    NON_FEATURE_COLS,
    OUTPUTS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    WEIGHT_COL,
)
from src.metrics import bootstrap_wmae_ci, wmae
from src.feature_engineering import GROUP_BUILDERS, add_feature_groups
from src.modeling import PowerTargetCatBoost, prepare_cat_features
from src.postprocessing import blend_salary_signal
from src.preprocessing import preprocess
from src.validation import get_folds

MODEL_SPECS = (
    # (target power, ensemble weight, seed offset)
    (0.25, 0.60, 0),
    (0.50, 0.40, 1),
)
SALARY_COL = "salary_6to12m_avg"
SALARY_BLEND_ALPHA = 0.60


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv",
        action="store_true",
        help="run the expensive 5-fold validation before final training",
    )
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument(
        "--feature-groups",
        default="",
        help=(
            "optional comma-separated input-only feature groups, for example "
            "log_rank or anchors,flows"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "submission_improved.csv",
    )
    return parser.parse_args()


def build_feature_matrix(df, feature_cols):
    """Create an identical CatBoost matrix for train, validation, and test."""
    X = df[feature_cols].copy()
    X[DATE_COL] = X[DATE_COL].astype(str)
    return prepare_cat_features(X, CATEGORICAL_COLS)


def model_params(iterations, seed, use_early_stopping):
    return {
        "iterations": iterations,
        "random_seed": seed,
        "early_stopping_rounds": 100 if use_early_stopping else None,
    }


def fit_predict_ensemble(X_train, y_train, weights, X_pred, iterations, eval_set=None, seed=RANDOM_SEED):
    """Fit the two complementary power models and return their weighted blend."""
    prediction = np.zeros(len(X_pred), dtype=float)
    best_iterations = []
    for power, blend_weight, seed_offset in MODEL_SPECS:
        model = PowerTargetCatBoost(
            target_power=power,
            cat_features=CATEGORICAL_COLS,
            params=model_params(iterations, seed + seed_offset, eval_set is not None),
        )
        model.fit(X_train, y_train, sample_weight=weights, eval_set=eval_set)
        prediction += blend_weight * model.predict(X_pred)
        best_iterations.append(model.model.get_best_iteration())
    return prediction, best_iterations


def calibrate_predictions(predictions, frame, target_range):
    """Apply feature-only calibration and safe target-support clipping."""
    calibrated = blend_salary_signal(
        predictions,
        pd.to_numeric(frame[SALARY_COL], errors="coerce").to_numpy(),
        alpha=SALARY_BLEND_ALPHA,
    )
    return np.clip(calibrated, target_range[0], target_range[1])


def run_cv(train, X, y, weights, iterations):
    folds = get_folds(y)
    oof_raw = np.full(len(train), np.nan)

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        eval_set = (X.iloc[val_idx], y[val_idx], weights[val_idx])
        fold_pred, best_iterations = fit_predict_ensemble(
            X.iloc[train_idx],
            y[train_idx],
            weights[train_idx],
            X.iloc[val_idx],
            iterations,
            eval_set=eval_set,
            seed=RANDOM_SEED + 10 * fold_i,
        )
        oof_raw[val_idx] = fold_pred
        fold_calibrated = calibrate_predictions(
            fold_pred, train.iloc[val_idx], (y.min(), y.max())
        )
        print(
            f"fold {fold_i}: WMAE={wmae(y[val_idx], fold_calibrated, weights[val_idx]):,.0f} "
            f"best_iterations={best_iterations}"
        )

    oof = calibrate_predictions(oof_raw, train, (y.min(), y.max()))
    score = wmae(y, oof, weights)
    ci = bootstrap_wmae_ci(y, oof, weights, n_boot=1000, seed=RANDOM_SEED)
    print(f"pooled OOF WMAE: {score:,.0f}")
    print(f"bootstrap 95% CI: [{ci['ci_low']:,.0f}, {ci['ci_high']:,.0f}]")


def main():
    args = parse_args()
    train_raw = pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS, low_memory=False)
    test_raw = pd.read_csv(TEST_PATH, **CSV_READ_KWARGS, low_memory=False)
    train, _ = preprocess(train_raw, is_train=True)
    test, _ = preprocess(test_raw, is_train=False)

    feature_groups = [group for group in args.feature_groups.split(",") if group]
    unknown_groups = sorted(set(feature_groups) - set(GROUP_BUILDERS))
    if unknown_groups:
        raise ValueError(f"unknown feature groups: {unknown_groups}")
    if feature_groups:
        train = add_feature_groups(train, feature_groups)
        test = add_feature_groups(test, feature_groups)

    feature_cols = [c for c in train.columns if c not in NON_FEATURE_COLS]
    X_train = build_feature_matrix(train, feature_cols)
    X_test = build_feature_matrix(test, feature_cols)
    y = train[TARGET_COL].to_numpy(dtype=float)
    weights = train[WEIGHT_COL].to_numpy(dtype=float)
    target_range = (float(y.min()), float(y.max()))

    if args.cv:
        run_cv(train, X_train, y, weights, args.iterations)

    test_pred, _ = fit_predict_ensemble(
        X_train,
        y,
        weights,
        X_test,
        args.iterations,
        eval_set=None,
    )
    test_pred = calibrate_predictions(test_pred, test, target_range)

    submission = pd.DataFrame({ID_COL: test[ID_COL], "predict": test_pred})
    assert len(submission) == len(test)
    assert submission["predict"].notna().all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, sep=";", decimal=",")
    print(f"saved {args.output} ({len(submission):,} rows)")


if __name__ == "__main__":
    main()
