"""Fast, reproducible ablation study for feature-engineering hypotheses.

The default run evaluates every group on the first fixed CV fold. Promising
groups can then be passed together through ``--groups anchors,flows,trends``.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    CSV_READ_KWARGS,
    DATE_COL,
    ID_COL,
    PARTIAL_OUTPUTS_DIR,
    RANDOM_SEED,
    REGION_COL,
    TARGET_COL,
    TRAIN_PATH,
    WEIGHT_COL,
)
from src.feature_engineering import GROUP_BUILDERS, add_feature_groups
from src.metrics import wmae
from src.modeling import PowerTargetCatBoost, prepare_cat_features
from src.postprocessing import blend_salary_signal
from src.preprocessing import preprocess
from src.region_encoding import (
    apply_smoothed_region_stats,
    compute_smoothed_region_stats,
    crossfit_smoothed_region_encoding,
)
from src.validation import get_folds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--time-holdout",
        action="store_true",
        help="use the final train month instead of a random stratified fold",
    )
    parser.add_argument(
        "--groups",
        help="comma-separated combined group; default evaluates all groups separately",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="with --groups, run the baseline immediately before the combined group",
    )
    parser.add_argument(
        "--output",
        default=PARTIAL_OUTPUTS_DIR / "feature_hypothesis_ablation.csv",
        type=str,
    )
    return parser.parse_args()


def add_nested_region_feature(frame, y, weights, train_idx, val_idx):
    """Build an inner-OOF train feature and outer-fold validation feature."""
    train_part = frame.iloc[train_idx].reset_index(drop=True)
    inner_folds = get_folds(y[train_idx], n_folds=4, seed=RANDOM_SEED + 101)
    train_encoding = crossfit_smoothed_region_encoding(
        train_part,
        y[train_idx],
        weights[train_idx],
        inner_folds,
        smoothing=60,
    )
    stats, fallback = compute_smoothed_region_stats(
        frame.iloc[train_idx][REGION_COL].to_numpy(),
        y[train_idx],
        weights[train_idx],
        smoothing=60,
    )
    val_encoding = apply_smoothed_region_stats(
        frame.iloc[val_idx][REGION_COL].to_numpy(), stats, fallback
    )
    return train_encoding, val_encoding


def build_fold_matrices(frame, groups, y, weights, train_idx, val_idx):
    use_region_target = "region_bayes" in groups
    ordinary_groups = [group for group in groups if group != "region_bayes"]
    engineered = add_feature_groups(frame, ordinary_groups)
    features = [column for column in engineered.columns if column not in {ID_COL, TARGET_COL, WEIGHT_COL}]

    if use_region_target:
        features.remove(REGION_COL)
    X = engineered[features].copy()
    X[DATE_COL] = X[DATE_COL].astype(str)
    cat_columns = [column for column in CATEGORICAL_COLS if column in X.columns]
    X = prepare_cat_features(X, cat_columns)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)

    if use_region_target:
        train_encoding, val_encoding = add_nested_region_feature(
            frame, y, weights, train_idx, val_idx
        )
        X_train["fe_region_income_bayes"] = train_encoding
        X_val["fe_region_income_bayes"] = val_encoding
    return X_train, X_val, cat_columns


def evaluate(frame, y, weights, train_idx, val_idx, groups, iterations):
    X_train, X_val, cat_columns = build_fold_matrices(
        frame, groups, y, weights, train_idx, val_idx
    )
    started = time.time()
    model = PowerTargetCatBoost(
        target_power=0.25,
        cat_features=cat_columns,
        params={
            "iterations": iterations,
            "random_seed": RANDOM_SEED,
            "early_stopping_rounds": 100,
        },
    )
    model.fit(
        X_train,
        y[train_idx],
        sample_weight=weights[train_idx],
        eval_set=(X_val, y[val_idx], weights[val_idx]),
    )
    prediction = model.predict(X_val)
    prediction = blend_salary_signal(
        prediction,
        pd.to_numeric(frame.iloc[val_idx]["salary_6to12m_avg"], errors="coerce"),
        alpha=0.6,
    )
    prediction = np.clip(prediction, y.min(), y.max())
    return {
        "groups": "+".join(groups) if groups else "baseline",
        "wmae": wmae(y[val_idx], prediction, weights[val_idx]),
        "n_features": X_train.shape[1],
        "best_iteration": model.model.get_best_iteration(),
        "seconds": time.time() - started,
    }


def main():
    args = parse_args()
    raw = pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS, low_memory=False)
    frame, _ = preprocess(raw, is_train=True)
    y = frame[TARGET_COL].to_numpy(dtype=float)
    weights = frame[WEIGHT_COL].to_numpy(dtype=float)
    if args.time_holdout:
        cutoff = frame[DATE_COL].max()
        train_idx = np.flatnonzero((frame[DATE_COL] < cutoff).to_numpy())
        val_idx = np.flatnonzero((frame[DATE_COL] == cutoff).to_numpy())
    else:
        folds = get_folds(y)
        train_idx, val_idx = folds[args.fold]

    if args.groups:
        experiments = ([[]] if args.compare else []) + [args.groups.split(",")]
    else:
        experiments = [[]] + [[group] for group in GROUP_BUILDERS] + [["region_bayes"]]

    rows = []
    for groups in experiments:
        row = evaluate(frame, y, weights, train_idx, val_idx, groups, args.iterations)
        rows.append(row)
        print(
            f"{row['groups']:<25} WMAE={row['wmae']:>10,.0f} "
            f"features={row['n_features']:>3} iter={row['best_iteration']:>4} "
            f"time={row['seconds']:.1f}s",
            flush=True,
        )

    result = pd.DataFrame(rows)
    baseline = result.loc[result["groups"] == "baseline", "wmae"]
    if len(baseline):
        result["delta_vs_baseline"] = result["wmae"] - baseline.iloc[0]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
