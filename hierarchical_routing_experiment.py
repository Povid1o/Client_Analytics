"""Audit unsupervised clusters and evaluate local hierarchical income routing."""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from cost_sensitive_ordinal_experiment import build_matrix, evaluate_probability_set
from src.config import (
    CSV_READ_KWARGS, ID_COL, PARTIAL_OUTPUTS_DIR, RANDOM_SEED, TARGET_COL,
    TRAIN_PATH, WEIGHT_COL,
)
from src.metrics import weighted_median, wmae
from src.hierarchical_routing import TREE_NODES, hierarchy_to_class_probabilities
from src.ordinal_routing import income_band
from src.preprocessing import preprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-predictions", type=Path,
        default=PARTIAL_OUTPUTS_DIR / "ordinal_cumulative_results_predictions.csv",
    )
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--cluster-components", type=int, default=30)
    parser.add_argument(
        "--output", type=Path,
        default=PARTIAL_OUTPUTS_DIR / "hierarchical_routing_results.csv",
    )
    parser.add_argument(
        "--skip-clusters", action="store_true",
        help="Skip the unsupervised clustering diagnostic.",
    )
    parser.add_argument(
        "--weighted-only", action="store_true",
        help="Train only the sample-weighted hierarchy.",
    )
    return parser.parse_args()


def classifier_params(iterations, seed):
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


def cluster_audit(X_train, X_val, y_train, y_val, w_train, w_val, components):
    numeric_train = X_train.select_dtypes(include=[np.number])
    numeric_val = X_val[numeric_train.columns]
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    train_imputed = imputer.fit_transform(numeric_train)
    val_imputed = imputer.transform(numeric_val)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    val_scaled = scaler.transform(val_imputed)
    pca = PCA(
        n_components=min(components, train_scaled.shape[1]),
        svd_solver="randomized", random_state=RANDOM_SEED + 1300,
    )
    train_embedding = pca.fit_transform(train_scaled)
    val_embedding = pca.transform(val_scaled)
    train_band = income_band(y_train)
    val_band = income_band(y_val)
    rows = []
    best = None
    for clusters in (8, 16, 32):
        model = MiniBatchKMeans(
            n_clusters=clusters, batch_size=4096, n_init=10,
            random_state=RANDOM_SEED + 1310 + clusters,
        )
        train_cluster = model.fit_predict(train_embedding)
        val_cluster = model.predict(val_embedding)
        band_map = {}
        target_map = {}
        entropy = []
        for cluster in range(clusters):
            selected = train_cluster == cluster
            class_weights = np.bincount(
                train_band[selected], weights=w_train[selected], minlength=8
            )
            band_map[cluster] = int(np.argmax(class_weights))
            target_map[cluster] = weighted_median(y_train[selected], w_train[selected])
            probability = class_weights / max(class_weights.sum(), 1e-12)
            entropy.append(-np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0))))
        predicted_band = np.asarray([band_map[value] for value in val_cluster])
        predicted_target = np.asarray([target_map[value] for value in val_cluster])
        accuracy = np.average(predicted_band == val_band, weights=w_val)
        within_one = np.average(np.abs(predicted_band - val_band) <= 1, weights=w_val)
        score = wmae(y_val, predicted_target, w_val)
        row = {
            "clusters": clusters,
            "explained_variance_30pc": float(pca.explained_variance_ratio_.sum()),
            "weighted_band_accuracy": accuracy,
            "weighted_within_one": within_one,
            "unweighted_NMI": normalized_mutual_info_score(val_band, val_cluster),
            "mean_cluster_target_entropy": float(np.mean(entropy)),
            "cluster_median_WMAE": score,
        }
        rows.append(row)
        if best is None or accuracy > best[0]:
            best = (accuracy, train_embedding, val_embedding, model, train_cluster, val_cluster)
        print(
            f"KMeans k={clusters}: accuracy={accuracy:.4f}, within1={within_one:.4f}, "
            f"NMI={row['unweighted_NMI']:.4f}, WMAE={score:,.0f}", flush=True,
        )
    return pd.DataFrame(rows), best


def fit_hierarchy(X_train, labels, weights, X_val, iterations, balanced):
    node_probabilities = {}
    for node_index, (name, lower, upper, threshold) in enumerate(TREE_NODES):
        selected = (labels >= lower) & (labels <= upper)
        node_label = (labels[selected] >= threshold).astype(int)
        node_weight = weights[selected].copy()
        if balanced:
            negative_weight = node_weight[node_label == 0].sum()
            positive_weight = node_weight[node_label == 1].sum()
            node_weight[node_label == 0] *= (negative_weight + positive_weight) / (2 * negative_weight)
            node_weight[node_label == 1] *= (negative_weight + positive_weight) / (2 * positive_weight)
        model = lgb.LGBMClassifier(
            **classifier_params(
                iterations, RANDOM_SEED + 1400 + node_index + (100 if balanced else 0)
            )
        )
        model.fit(X_train.loc[selected], node_label, sample_weight=node_weight)
        node_probabilities[name] = model.predict_proba(X_val)[:, 1]
        print(
            f"hierarchy {'balanced' if balanced else 'weighted'}: {name} "
            f"rows={selected.sum():,}", flush=True,
        )

    return hierarchy_to_class_probabilities(node_probabilities)


def main():
    args = parse_args()
    raw = pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS, low_memory=False)
    frame, _ = preprocess(raw, is_train=True)
    reference = pd.read_csv(args.reference_predictions).set_index(ID_COL)
    validation_ids = set(reference.index)
    train_mask = ~frame[ID_COL].isin(validation_ids).to_numpy()
    val_mask = ~train_mask
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    X = build_matrix(frame)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y = frame[TARGET_COL].to_numpy(dtype=float)
    weights = frame[WEIGHT_COL].to_numpy(dtype=float)
    labels = income_band(y)
    ordered = reference.loc[frame.iloc[val_idx][ID_COL]]
    base = ordered["base"].to_numpy(dtype=float)
    experts = ordered[[f"expert_band_{i}" for i in range(8)]].to_numpy(dtype=float)
    baseline_probability = ordered[
        [f"probability_band_{i}" for i in range(8)]
    ].to_numpy(dtype=float)
    base_score = wmae(y[val_idx], base, weights[val_idx])
    print(f"base WMAE={base_score:,.0f}; cluster audit", flush=True)

    if not args.skip_clusters:
        cluster_result, _ = cluster_audit(
            X_train, X_val, y[train_idx], y[val_idx],
            weights[train_idx], weights[val_idx], args.cluster_components,
        )
        cluster_path = args.output.with_name(f"{args.output.stem}_clusters.csv")
        cluster_result.to_csv(cluster_path, index=False)

    probability_sets = {"baseline": baseline_probability}
    hierarchy_specs = [(False, "hierarchical_weighted")]
    if not args.weighted_only:
        hierarchy_specs.append((True, "hierarchical_balanced"))
    for balanced, name in hierarchy_specs:
        probability = fit_hierarchy(
            X_train, labels[train_idx], weights[train_idx], X_val,
            args.iterations, balanced,
        )
        probability_sets[name] = probability
        for alpha in (0.25, 0.5, 0.75):
            probability_sets[f"blend_{name}_{alpha:.2f}"] = (
                (1 - alpha) * baseline_probability + alpha * probability
            )

    results = []
    diagnostics = []
    output = {
        ID_COL: frame.iloc[val_idx][ID_COL].to_numpy(), TARGET_COL: y[val_idx],
        WEIGHT_COL: weights[val_idx], "base": base,
    }
    for name, probability in probability_sets.items():
        predicted = probability.argmax(axis=1)
        diagnostics.append(
            {
                "probability_set": name,
                "weighted_accuracy": np.average(
                    predicted == labels[val_idx], weights=weights[val_idx]
                ),
                "weighted_within_one": np.average(
                    np.abs(predicted - labels[val_idx]) <= 1, weights=weights[val_idx]
                ),
                "weighted_class_mae": np.average(
                    np.abs(predicted - labels[val_idx]), weights=weights[val_idx]
                ),
            }
        )
        grid, best, prediction = evaluate_probability_set(
            name, probability, base, experts, y[val_idx], weights[val_idx]
        )
        results.append(grid)
        output[f"prediction_{name}"] = prediction
        for band in range(8):
            output[f"{name}_probability_{band}"] = probability[:, band]
        print(
            f"{name}: best={best[0]:,.0f} ({best[0]-base_score:+,.0f}); "
            f"T={best[1]}, mode={best[2]}, gamma={best[3]}, strength={best[4]:.1f}",
            flush=True,
        )
    result = pd.concat(results, ignore_index=True).sort_values("wmae")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    pd.DataFrame(diagnostics).to_csv(
        args.output.with_name(f"{args.output.stem}_classifiers.csv"), index=False
    )
    pd.DataFrame(output).to_csv(
        args.output.with_name(f"{args.output.stem}_predictions.csv"), index=False
    )
    print("\nClassifier diagnostics:")
    print(pd.DataFrame(diagnostics).sort_values("weighted_accuracy", ascending=False).to_string(index=False))
    print("\nTop routing configurations:")
    print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
