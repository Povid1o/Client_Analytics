"""Evaluate cost-sensitive cumulative boundaries using fixed OOF experts."""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLS, CSV_READ_KWARGS, ID_COL, NON_FEATURE_COLS,
    PARTIAL_OUTPUTS_DIR, RANDOM_SEED, TARGET_COL, TRAIN_PATH, WEIGHT_COL,
)
from src.cost_sensitive import boundary_cost_weights
from src.feature_engineering import add_feature_groups
from src.metrics import wmae
from src.ordinal_routing import (
    DEFAULT_BAND_EDGES, cumulative_to_class_probabilities, income_band,
    route_predictions, temperature_scale,
)
from src.preprocessing import preprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-predictions", type=Path,
        default=PARTIAL_OUTPUTS_DIR / "ordinal_cumulative_results_predictions.csv",
    )
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1.5])
    parser.add_argument(
        "--output", type=Path,
        default=PARTIAL_OUTPUTS_DIR / "cost_sensitive_ordinal_results.csv",
    )
    return parser.parse_args()


def build_matrix(frame):
    engineered = add_feature_groups(
        frame, ["anchors", "scale", "flows", "trends", "log_rank"]
    )
    columns = [column for column in engineered.columns if column not in NON_FEATURE_COLS]
    matrix = engineered[columns].copy()
    for column in CATEGORICAL_COLS:
        if column in matrix:
            values = matrix[column].astype(str).replace({"nan": "missing", "NaT": "missing"})
            matrix[column] = pd.Categorical(values).codes.astype("int32")
    return matrix


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


def fit_boundaries(X_train, labels, y, weights, X_val, iterations, strength):
    cumulative = []
    for threshold in range(1, len(DEFAULT_BAND_EDGES) - 1):
        boundary = DEFAULT_BAND_EDGES[threshold]
        sample_weight = boundary_cost_weights(y, weights, boundary, strength)
        model = lgb.LGBMClassifier(
            **classifier_params(
                iterations, RANDOM_SEED + 1000 + threshold + int(strength * 100)
            )
        )
        model.fit(
            X_train, (labels >= threshold).astype(int), sample_weight=sample_weight
        )
        cumulative.append(model.predict_proba(X_val)[:, 1])
        print(
            f"strength={strength:g}: boundary {int(boundary/1000)}k "
            f"({threshold}/7)", flush=True,
        )
    return cumulative_to_class_probabilities(np.column_stack(cumulative))


def evaluate_probability_set(name, probabilities, base, experts, y, weights):
    base_score = wmae(y, base, weights)
    rows = []
    best_prediction = None
    best_key = None
    for temperature in (0.35, 0.5, 0.75, 1.0, 1.25):
        scaled = temperature_scale(probabilities, temperature)
        confidence = scaled.max(axis=1)
        for mode in ("soft", "median"):
            routed = route_predictions(base, scaled, experts, mode=mode)
            for gamma in (0.0, 0.5, 1.0, 2.0):
                gate = np.ones(len(base)) if gamma == 0 else confidence ** gamma
                for strength in np.arange(0.5, 1.21, 0.1):
                    prediction = base + strength * gate * (routed - base)
                    score = wmae(y, prediction, weights)
                    row = {
                        "probability_set": name,
                        "temperature": temperature,
                        "mode": mode,
                        "confidence_gamma": gamma,
                        "route_strength": strength,
                        "wmae": score,
                        "delta": score - base_score,
                    }
                    rows.append(row)
                    key = (score, temperature, mode, gamma, strength)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_prediction = prediction
    return pd.DataFrame(rows), best_key, best_prediction


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
    print(f"base WMAE={base_score:,.0f}; train={len(train_idx):,}; val={len(val_idx):,}")

    probability_sets = {"baseline": baseline_probability}
    for strength in args.strengths:
        cost_probability = fit_boundaries(
            X.iloc[train_idx], labels[train_idx], y[train_idx], weights[train_idx],
            X.iloc[val_idx], args.iterations, strength,
        )
        cost_name = f"cost_{strength:g}"
        probability_sets[cost_name] = cost_probability
        for alpha in (0.25, 0.5, 0.75):
            probability_sets[f"blend_{strength:g}_{alpha:.2f}"] = (
                (1 - alpha) * baseline_probability + alpha * cost_probability
            )

    results = []
    best_predictions = {"base": base}
    diagnostics = []
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
        best_predictions[name] = prediction
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
    output = {
        ID_COL: frame.iloc[val_idx][ID_COL].to_numpy(),
        TARGET_COL: y[val_idx], WEIGHT_COL: weights[val_idx],
        **best_predictions,
    }
    for name, probability in probability_sets.items():
        for band in range(8):
            output[f"{name}_probability_{band}"] = probability[:, band]
    pd.DataFrame(output).to_csv(
        args.output.with_name(f"{args.output.stem}_predictions.csv"), index=False
    )
    print("\nTop configurations:")
    print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
