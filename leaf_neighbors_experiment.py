"""Leakage-safe local estimators in CatBoost leaf space.

A compact CatBoost is fit on the outer-train partition only. Validation rows
retrieve train neighbours that repeatedly land in the same leaves across a
subsample of trees. Targets are aggregated with a similarity-weighted median.
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
    NON_FEATURE_COLS,
    PARTIAL_OUTPUTS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TRAIN_PATH,
    WEIGHT_COL,
)
from src.metrics import weighted_median, wmae
from src.modeling import PowerTargetCatBoost, prepare_cat_features
from src.preprocessing import preprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-predictions",
        default=PARTIAL_OUTPUTS_DIR / "residual_stacking_outer_predictions.csv",
    )
    parser.add_argument("--embedding-trees", type=int, default=300)
    parser.add_argument("--selected-trees", type=int, default=24)
    parser.add_argument("--bucket-cap", type=int, default=150)
    parser.add_argument(
        "--output",
        default=PARTIAL_OUTPUTS_DIR / "leaf_neighbors_results.csv",
    )
    return parser.parse_args()


def build_matrix(frame):
    features = [column for column in frame.columns if column not in NON_FEATURE_COLS]
    matrix = frame[features].copy()
    matrix[DATE_COL] = matrix[DATE_COL].astype(str)
    return prepare_cat_features(matrix, CATEGORICAL_COLS)


def make_leaf_buckets(train_leaves, tree_indices, cap, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    buckets = []
    leaf_stats = []
    for tree in tree_indices:
        tree_buckets = {}
        values = train_leaves[:, tree]
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1, len(order)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            leaf = int(sorted_values[start])
            indices = order[start:end]
            if len(indices) > cap:
                indices = rng.choice(indices, size=cap, replace=False)
            tree_buckets[leaf] = indices
        buckets.append(tree_buckets)
        leaf_stats.append({})
    return buckets, leaf_stats


def fill_leaf_stats(leaf_stats, buckets, target, weights):
    for tree_stats, tree_buckets in zip(leaf_stats, buckets):
        for leaf, indices in tree_buckets.items():
            tree_stats[leaf] = weighted_median(target[indices], weights[indices])


def predict_neighbours(
    val_leaves,
    tree_indices,
    buckets,
    leaf_stats,
    target,
    weights,
    neighbour_counts=(25, 50, 100, 200),
):
    n_val = len(val_leaves)
    neighbour_predictions = {
        (count, power): np.full(n_val, np.nan)
        for count in neighbour_counts
        for power in (1.0, 2.0)
    }
    leaf_median_prediction = np.full(n_val, np.nan)
    neighbour_spread = np.full(n_val, np.nan)
    neighbour_max_similarity = np.full(n_val, np.nan)

    for row in range(n_val):
        candidate_parts = []
        per_tree_predictions = []
        for local_tree, tree in enumerate(tree_indices):
            leaf = int(val_leaves[row, tree])
            indices = buckets[local_tree].get(leaf)
            if indices is not None:
                candidate_parts.append(indices)
                per_tree_predictions.append(leaf_stats[local_tree][leaf])
        if not candidate_parts:
            continue

        candidates, similarity = np.unique(
            np.concatenate(candidate_parts), return_counts=True
        )
        order = np.argsort(similarity)[::-1]
        candidates = candidates[order]
        similarity = similarity[order].astype(float)
        neighbour_max_similarity[row] = similarity[0] / len(tree_indices)
        leaf_median_prediction[row] = np.median(per_tree_predictions)

        top_for_spread = candidates[: min(100, len(candidates))]
        neighbour_spread[row] = np.subtract(*np.percentile(target[top_for_spread], [75, 25]))
        for count in neighbour_counts:
            take = min(count, len(candidates))
            selected = candidates[:take]
            selected_similarity = similarity[:take]
            for power in (1.0, 2.0):
                local_weights = weights[selected] * np.power(selected_similarity, power)
                neighbour_predictions[(count, power)][row] = weighted_median(
                    target[selected], local_weights
                )

        if row and row % 2000 == 0:
            print(f"processed neighbours for {row:,}/{n_val:,} validation rows", flush=True)
    return (
        neighbour_predictions,
        leaf_median_prediction,
        neighbour_spread,
        neighbour_max_similarity,
    )


def best_blend(base, local, target, weights):
    valid = np.isfinite(local)
    fallback_local = np.where(valid, local, base)
    rows = []
    for alpha in np.arange(0, 1.01, 0.05):
        prediction = (1 - alpha) * base + alpha * fallback_local
        score = wmae(target, np.clip(prediction, 20_000, 1_500_000), weights)
        rows.append((score, alpha))
    return min(rows)


def main():
    args = parse_args()
    started = time.time()
    raw = pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS, low_memory=False)
    frame, _ = preprocess(raw, is_train=True)
    outer = pd.read_csv(args.base_predictions)
    validation_ids = set(outer[ID_COL])
    train_mask = ~frame[ID_COL].isin(validation_ids).to_numpy()
    val_mask = ~train_mask
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    y = frame[TARGET_COL].to_numpy(dtype=float)
    weights = frame[WEIGHT_COL].to_numpy(dtype=float)
    X = build_matrix(frame)
    base = outer.set_index(ID_COL).loc[frame.iloc[val_idx][ID_COL], "base_prediction"].to_numpy()
    base_score = wmae(y[val_idx], base, weights[val_idx])

    embedding_model = PowerTargetCatBoost(
        target_power=0.25,
        cat_features=CATEGORICAL_COLS,
        params={
            "iterations": args.embedding_trees,
            "learning_rate": 0.08,
            "depth": 8,
            "l2_leaf_reg": 10,
            "random_seed": RANDOM_SEED + 301,
            "early_stopping_rounds": None,
        },
    )
    embedding_model.fit(X.iloc[train_idx], y[train_idx], weights[train_idx])
    train_leaves = embedding_model.model.calc_leaf_indexes(X.iloc[train_idx])
    val_leaves = embedding_model.model.calc_leaf_indexes(X.iloc[val_idx])
    tree_indices = np.linspace(
        args.embedding_trees // 3,
        args.embedding_trees - 1,
        args.selected_trees,
        dtype=int,
    )
    buckets, leaf_stats = make_leaf_buckets(
        train_leaves, tree_indices, args.bucket_cap
    )
    fill_leaf_stats(
        leaf_stats,
        buckets,
        y[train_idx],
        weights[train_idx],
    )
    (
        neighbours,
        leaf_median,
        neighbour_spread,
        neighbour_max_similarity,
    ) = predict_neighbours(
        val_leaves,
        tree_indices,
        buckets,
        leaf_stats,
        y[train_idx],
        weights[train_idx],
    )

    rows = []
    candidates = {"leaf_median": leaf_median}
    candidates.update(
        {
            f"knn_k{count}_p{int(power)}": prediction
            for (count, power), prediction in neighbours.items()
        }
    )
    prediction_output = {
        ID_COL: frame.iloc[val_idx][ID_COL].to_numpy(),
        TARGET_COL: y[val_idx],
        WEIGHT_COL: weights[val_idx],
        "base_prediction": base,
        "neighbour_iqr": neighbour_spread,
        "max_leaf_similarity": neighbour_max_similarity,
    }
    for name, prediction in candidates.items():
        local_score = wmae(
            y[val_idx][np.isfinite(prediction)],
            prediction[np.isfinite(prediction)],
            weights[val_idx][np.isfinite(prediction)],
        )
        blend_score, alpha = best_blend(base, prediction, y[val_idx], weights[val_idx])
        rows.append(
            {
                "estimator": name,
                "coverage": float(np.isfinite(prediction).mean()),
                "local_wmae": local_score,
                "best_alpha": alpha,
                "blend_wmae": blend_score,
                "delta": blend_score - base_score,
            }
        )
        prediction_output[name] = prediction

    result = pd.DataFrame(rows).sort_values("blend_wmae")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    prediction_path = output_path.with_name(f"{output_path.stem}_predictions.csv")
    pd.DataFrame(prediction_output).to_csv(prediction_path, index=False)
    print(f"base WMAE={base_score:,.0f}")
    print(result.to_string(index=False))
    print(f"saved {args.output} and {prediction_path}; time={time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
