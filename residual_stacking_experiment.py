"""Strict nested-OOF experiment for a second-stage residual CatBoost.

The residual model never sees in-sample base predictions. For one outer
validation split, base predictions for the outer-train rows are produced by
an inner CV, while the outer-validation prediction comes from a base model
fit on the complete outer-train partition.
"""
import argparse
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.config import (
    CATEGORICAL_COLS,
    CSV_READ_KWARGS,
    DATE_COL,
    ID_COL,
    NON_FEATURE_COLS,
    PARTIAL_OUTPUTS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TRAIN_PATH,
    WEIGHT_COL,
)
from src.feature_engineering import add_feature_groups
from src.metrics import wmae, weighted_median
from src.modeling import PowerTargetCatBoost, prepare_cat_features
from src.postprocessing import blend_salary_signal
from src.preprocessing import preprocess
from src.validation import get_folds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--time-holdout", action="store_true")
    parser.add_argument("--base-iterations", type=int, default=1200)
    parser.add_argument("--meta-iterations", type=int, default=600)
    parser.add_argument(
        "--output",
        default=PARTIAL_OUTPUTS_DIR / "residual_stacking_outer_predictions.csv",
    )
    return parser.parse_args()


def build_matrix(frame):
    features = [column for column in frame.columns if column not in NON_FEATURE_COLS]
    matrix = frame[features].copy()
    matrix[DATE_COL] = matrix[DATE_COL].astype(str)
    return prepare_cat_features(matrix, CATEGORICAL_COLS)


def calibrate(prediction, frame, target_range):
    prediction = blend_salary_signal(
        prediction,
        pd.to_numeric(frame["salary_6to12m_avg"], errors="coerce"),
        alpha=0.6,
    )
    return np.clip(prediction, *target_range)


def fit_base(X_train, y_train, w_train, iterations, eval_set=None, seed=RANDOM_SEED):
    model = PowerTargetCatBoost(
        target_power=0.25,
        cat_features=CATEGORICAL_COLS,
        params={
            "iterations": iterations,
            "random_seed": seed,
            "early_stopping_rounds": 100 if eval_set is not None else None,
        },
    )
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=eval_set)
    return model


def create_nested_base_predictions(frame, X, y, weights, outer_train, outer_val, iterations):
    train_frame = frame.iloc[outer_train].reset_index(drop=True)
    X_outer_train = X.iloc[outer_train].reset_index(drop=True)
    y_outer = y[outer_train]
    w_outer = weights[outer_train]
    inner_folds = get_folds(y_outer, n_folds=4, seed=RANDOM_SEED + 101)
    inner_oof = np.full(len(outer_train), np.nan)
    best_iterations = []

    for inner_i, (inner_train, inner_val) in enumerate(inner_folds):
        model = fit_base(
            X_outer_train.iloc[inner_train],
            y_outer[inner_train],
            w_outer[inner_train],
            iterations,
            eval_set=(
                X_outer_train.iloc[inner_val],
                y_outer[inner_val],
                w_outer[inner_val],
            ),
            seed=RANDOM_SEED + inner_i,
        )
        prediction = model.predict(X_outer_train.iloc[inner_val])
        inner_oof[inner_val] = calibrate(
            prediction,
            train_frame.iloc[inner_val],
            (float(y.min()), float(y.max())),
        )
        best_iterations.append(model.model.get_best_iteration() + 1)
        print(
            f"inner {inner_i}: WMAE={wmae(y_outer[inner_val], inner_oof[inner_val], w_outer[inner_val]):,.0f} "
            f"iterations={best_iterations[-1]}",
            flush=True,
        )

    final_iterations = max(100, int(np.mean(best_iterations)))
    outer_model = fit_base(
        X_outer_train,
        y_outer,
        w_outer,
        final_iterations,
        eval_set=None,
        seed=RANDOM_SEED + 10,
    )
    outer_prediction = calibrate(
        outer_model.predict(X.iloc[outer_val]),
        frame.iloc[outer_val],
        (float(y.min()), float(y.max())),
    )
    return inner_oof, outer_prediction, final_iterations


def build_meta_matrix(frame, base_prediction, full_features):
    if full_features:
        engineered = add_feature_groups(frame, ["anchors", "flows", "trends"])
        matrix = build_matrix(engineered)
        cat_columns = list(CATEGORICAL_COLS)
    else:
        numeric_columns = [
            "salary_6to12m_avg",
            "incomeValue",
            "first_salary_income",
            "dp_ils_avg_salary_1y",
            "dp_payoutincomedata_payout_avg_6_month",
            "turn_cur_cr_avg_act_v2",
            "turn_cur_db_avg_act_v2",
            "hdb_bki_total_max_limit",
            "age",
        ]
        matrix = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        cat_columns = []
    matrix = matrix.copy()
    matrix["base_prediction"] = np.asarray(base_prediction, dtype=float)
    matrix["base_distance_from_weight_pivot"] = np.abs(matrix["base_prediction"] - 84017.08)
    return matrix, cat_columns


def fit_residual_model(X, residual, weights, cat_columns, iterations, depth, seed):
    model = CatBoostRegressor(
        loss_function="MAE",
        iterations=iterations,
        learning_rate=0.03,
        depth=depth,
        l2_leaf_reg=10,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(X, label=residual, weight=weights, cat_features=cat_columns))
    return model


def print_error_segments(frame, y, prediction, weights):
    diagnostics = pd.DataFrame(
        {
            "target": y,
            "prediction": prediction,
            "w": weights,
            "month": frame[DATE_COL].astype(str).to_numpy(),
            "salary_present": frame["salary_6to12m_avg"].notna().to_numpy(),
        }
    )
    diagnostics["target_decile"] = pd.qcut(diagnostics["target"], 10, labels=False)
    denominator = diagnostics["w"].sum()
    print("\nOuter-validation WMAE contribution by target decile:")
    for decile, group in diagnostics.groupby("target_decile"):
        contribution = np.sum(group["w"] * np.abs(group["target"] - group["prediction"])) / denominator
        signed_residual = weighted_median(
            (group["target"] - group["prediction"]).to_numpy(), group["w"].to_numpy()
        )
        print(
            f"  D{decile}: contribution={contribution:,.0f}, "
            f"weighted_median_residual={signed_residual:,.0f}"
        )


def main():
    args = parse_args()
    started = time.time()
    raw = pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS, low_memory=False)
    frame, _ = preprocess(raw, is_train=True)
    X = build_matrix(frame)
    y = frame[TARGET_COL].to_numpy(dtype=float)
    weights = frame[WEIGHT_COL].to_numpy(dtype=float)

    if args.time_holdout:
        cutoff = frame[DATE_COL].max()
        outer_train = np.flatnonzero((frame[DATE_COL] < cutoff).to_numpy())
        outer_val = np.flatnonzero((frame[DATE_COL] == cutoff).to_numpy())
    else:
        outer_train, outer_val = get_folds(y)[args.outer_fold]

    inner_prediction, outer_prediction, final_iterations = create_nested_base_predictions(
        frame, X, y, weights, outer_train, outer_val, args.base_iterations
    )
    base_score = wmae(y[outer_val], outer_prediction, weights[outer_val])
    residual = y[outer_train] - inner_prediction
    print(f"\nbase outer WMAE: {base_score:,.0f}; refit iterations={final_iterations}")
    print_error_segments(
        frame.iloc[outer_val], y[outer_val], outer_prediction, weights[outer_val]
    )

    rows = []
    for full_features, depth in [(False, 4), (True, 6)]:
        train_meta, cat_columns = build_meta_matrix(
            frame.iloc[outer_train].reset_index(drop=True), inner_prediction, full_features
        )
        val_meta, _ = build_meta_matrix(
            frame.iloc[outer_val].reset_index(drop=True), outer_prediction, full_features
        )
        model = fit_residual_model(
            train_meta,
            residual,
            weights[outer_train],
            cat_columns,
            args.meta_iterations,
            depth,
            seed=RANDOM_SEED + depth,
        )
        correction = model.predict(val_meta)
        label = "full" if full_features else "compact"
        for shrinkage in [0.25, 0.5, 0.75, 1.0]:
            corrected = np.clip(
                outer_prediction + shrinkage * correction,
                y.min(),
                y.max(),
            )
            score = wmae(y[outer_val], corrected, weights[outer_val])
            rows.append(
                {
                    "meta_features": label,
                    "shrinkage": shrinkage,
                    "wmae": score,
                    "delta": score - base_score,
                }
            )
            print(
                f"residual={label:<7} shrink={shrinkage:.2f}: "
                f"WMAE={score:,.0f} delta={score-base_score:+,.0f}"
            )

    result = pd.DataFrame(rows).sort_values("wmae")
    print("\nBest residual configurations:")
    print(result.head(8).to_string(index=False))
    output = pd.DataFrame(
        {
            ID_COL: frame.iloc[outer_val][ID_COL].to_numpy(),
            TARGET_COL: y[outer_val],
            WEIGHT_COL: weights[outer_val],
            "base_prediction": outer_prediction,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"saved {args.output}; total time={time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
