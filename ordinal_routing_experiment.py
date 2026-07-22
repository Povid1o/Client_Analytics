"""Evaluate an ordered income-band classifier and band specialists.

The validation IDs come from a precomputed strict outer base prediction file.
All classifiers and experts are fitted only on the complementary rows.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix

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
from src.metrics import wmae
from src.feature_engineering import add_feature_groups
from src.modeling import inverse_power_target, power_transform_target
from src.ordinal_routing import (
    DEFAULT_BAND_EDGES,
    cumulative_to_class_probabilities,
    income_band,
    project_to_bands,
    route_predictions,
    temperature_scale,
)
from src.postprocessing import blend_salary_signal
from src.preprocessing import preprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=PARTIAL_OUTPUTS_DIR / "residual_stacking_outer_predictions.csv",
    )
    parser.add_argument("--classifier-iterations", type=int, default=400)
    parser.add_argument("--expert-iterations", type=int, default=350)
    parser.add_argument("--cumulative", action="store_true")
    parser.add_argument("--engineered", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=PARTIAL_OUTPUTS_DIR / "ordinal_routing_results.csv"
    )
    return parser.parse_args()


def build_matrix(frame):
    columns = [column for column in frame.columns if column not in NON_FEATURE_COLS]
    matrix = frame[columns].copy()
    for column in CATEGORICAL_COLS:
        if column in matrix:
            values = matrix[column].astype(str).replace({"nan": "missing", "NaT": "missing"})
            matrix[column] = pd.Categorical(values).codes.astype("int32")
    return matrix


def calibrate(prediction, frame, target_range):
    prediction = blend_salary_signal(
        prediction,
        pd.to_numeric(frame["salary_6to12m_avg"], errors="coerce").to_numpy(),
        alpha=0.6,
    )
    return np.clip(prediction, *target_range)


def fit_classifier(X, labels, weights, iterations, weighted, seed):
    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=iterations,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=12.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    sample_weight = weights if weighted else np.ones_like(weights)
    model.fit(X, labels, sample_weight=sample_weight)
    return model


def fit_cumulative_classifiers(X_train, labels, weights, X_val, iterations, n_bands):
    probabilities = []
    for threshold in range(1, n_bands):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=iterations,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=12.0,
            random_state=RANDOM_SEED + 740 + threshold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X_train, (labels >= threshold).astype(int), sample_weight=weights)
        probabilities.append(model.predict_proba(X_val)[:, 1])
    return cumulative_to_class_probabilities(np.column_stack(probabilities))


def fit_expert(X, y, weights, selected, iterations, seed):
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=iterations,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=15.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    rows = np.flatnonzero(selected)
    model.fit(
        X.iloc[rows], power_transform_target(y[rows], 0.25),
        sample_weight=weights[rows],
    )
    return model, len(rows)


def expanded_band_mask(labels, band, n_bands):
    """Train a specialist on its band and immediate neighbours."""
    lower = max(0, band - 1)
    upper = min(n_bands - 1, band + 1)
    return (labels >= lower) & (labels <= upper)


def candidate_grid(base, probabilities, candidates, y, weights, prefix):
    rows = []
    predictions = {}
    base_score = wmae(y, base, weights)
    for temperature in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        scaled = temperature_scale(probabilities, temperature)
        confidence = scaled.max(axis=1)
        for mode in ("soft", "median"):
            routed = route_predictions(base, scaled, candidates, mode=mode)
            for gamma in (0.0, 0.5, 1.0, 2.0):
                gate = np.ones_like(confidence) if gamma == 0 else confidence ** gamma
                for strength in np.arange(0.1, 1.01, 0.1):
                    prediction = base + strength * gate * (routed - base)
                    score = wmae(y, prediction, weights)
                    rows.append(
                        {
                            "family": prefix,
                            "temperature": temperature,
                            "mode": mode,
                            "confidence_gamma": gamma,
                            "strength": strength,
                            "wmae": score,
                            "delta": score - base_score,
                        }
                    )
                    key = (score, temperature, mode, gamma, strength)
                    if not predictions or key < min(predictions):
                        predictions = {key: prediction}
    best_key = min(predictions)
    return pd.DataFrame(rows), best_key, predictions[best_key]


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
    target_range = (float(y.min()), float(y.max()))
    labels = income_band(y)
    n_bands = len(DEFAULT_BAND_EDGES) - 1
    model_frame = frame
    if args.engineered:
        model_frame = add_feature_groups(
            frame, ["anchors", "scale", "flows", "trends", "log_rank"]
        )
    X = build_matrix(model_frame)
    base = (
        outer.set_index(ID_COL)
        .loc[frame.iloc[val_idx][ID_COL], "base_prediction"]
        .to_numpy(dtype=float)
    )
    base_score = wmae(y[val_idx], base, weights[val_idx])
    print(f"base WMAE={base_score:,.0f}; train={len(train_idx):,}; val={len(val_idx):,}")

    probability_sets = {}
    classifier_rows = []
    classifier_specs = [(True, "wmae_weighted", RANDOM_SEED + 702)]
    if not args.cumulative:
        classifier_specs.insert(0, (False, "uniform", RANDOM_SEED + 701))
    for weighted, name, seed in classifier_specs:
        model = fit_classifier(
            X.iloc[train_idx], labels[train_idx], weights[train_idx],
            args.classifier_iterations, weighted, seed,
        )
        probability = model.predict_proba(X.iloc[val_idx])
        probability_sets[name] = probability
        predicted = probability.argmax(axis=1)
        exact = np.average(predicted == labels[val_idx], weights=weights[val_idx])
        ordinal_mae = np.average(
            np.abs(predicted - labels[val_idx]), weights=weights[val_idx]
        )
        within_one = np.average(
            np.abs(predicted - labels[val_idx]) <= 1, weights=weights[val_idx]
        )
        classifier_rows.append(
            {"classifier": name, "weighted_accuracy": exact,
             "weighted_within_one": within_one, "weighted_class_mae": ordinal_mae}
        )
        print(
            f"{name}: accuracy={exact:.4f}, within1={within_one:.4f}, "
            f"class_MAE={ordinal_mae:.4f}", flush=True,
        )

    if args.cumulative:
        cumulative = fit_cumulative_classifiers(
            X.iloc[train_idx], labels[train_idx], weights[train_idx],
            X.iloc[val_idx], args.classifier_iterations, n_bands,
        )
        probability_sets["cumulative_weighted"] = cumulative
        predicted = cumulative.argmax(axis=1)
        print(
            "cumulative_weighted: "
            f"accuracy={np.average(predicted == labels[val_idx], weights=weights[val_idx]):.4f}, "
            f"within1={np.average(np.abs(predicted-labels[val_idx]) <= 1, weights=weights[val_idx]):.4f}, "
            f"class_MAE={np.average(np.abs(predicted-labels[val_idx]), weights=weights[val_idx]):.4f}",
            flush=True,
        )
        for alpha in (0.25, 0.5, 0.75):
            probability_sets[f"cumulative_blend_{alpha:.2f}"] = (
                (1 - alpha) * probability_sets["wmae_weighted"] + alpha * cumulative
            )
    else:
        for alpha in (0.25, 0.5, 0.75):
            probability_sets[f"blend_{alpha:.2f}"] = (
                (1 - alpha) * probability_sets["uniform"]
                + alpha * probability_sets["wmae_weighted"]
            )

    projected = project_to_bands(base)
    expert_candidates = np.empty((len(val_idx), n_bands), dtype=float)
    expert_counts = []
    for band in range(n_bands):
        selected = expanded_band_mask(labels[train_idx], band, n_bands)
        expert, count = fit_expert(
            X.iloc[train_idx].reset_index(drop=True), y[train_idx], weights[train_idx],
            selected, args.expert_iterations, RANDOM_SEED + 720 + band,
        )
        prediction = calibrate(
            inverse_power_target(expert.predict(X.iloc[val_idx]), 0.25),
            frame.iloc[val_idx], target_range,
        )
        expert_candidates[:, band] = np.clip(
            prediction, DEFAULT_BAND_EDGES[band], DEFAULT_BAND_EDGES[band + 1]
        )
        expert_counts.append(count)
        print(f"expert band {band}: train_rows={count:,}", flush=True)

    all_rows = []
    best_predictions = {"base": base}
    for probability_name, probability in probability_sets.items():
        for candidate_name, candidates in (
            ("projection", projected), ("expert", expert_candidates),
        ):
            family = f"{probability_name}_{candidate_name}"
            grid, best_key, prediction = candidate_grid(
                base, probability, candidates, y[val_idx], weights[val_idx], family
            )
            all_rows.append(grid)
            best_predictions[family] = prediction
            print(
                f"{family}: best={best_key[0]:,.0f} ({best_key[0]-base_score:+,.0f}), "
                f"T={best_key[1]}, mode={best_key[2]}, gamma={best_key[3]}, "
                f"strength={best_key[4]:.1f}", flush=True,
            )

    result = pd.concat(all_rows, ignore_index=True).sort_values("wmae")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    prediction_path = args.output.with_name(f"{args.output.stem}_predictions.csv")
    primary_name = "cumulative_weighted" if args.cumulative else "blend_0.50"
    probability_columns = {
        f"probability_band_{i}": values
        for i, values in enumerate(probability_sets[primary_name].T)
    }
    expert_columns = {
        f"expert_band_{i}": values for i, values in enumerate(expert_candidates.T)
    }
    pd.DataFrame(
        {
            ID_COL: frame.iloc[val_idx][ID_COL].to_numpy(),
            TARGET_COL: y[val_idx], WEIGHT_COL: weights[val_idx],
            **best_predictions, **probability_columns, **expert_columns,
        }
    ).to_csv(prediction_path, index=False)
    diagnostics_path = args.output.with_name(f"{args.output.stem}_classifiers.csv")
    pd.DataFrame(classifier_rows).to_csv(diagnostics_path, index=False)
    matrix_path = args.output.with_name(f"{args.output.stem}_confusion.csv")
    confusion = confusion_matrix(
        labels[val_idx], probability_sets[primary_name].argmax(axis=1),
        labels=np.arange(n_bands), sample_weight=weights[val_idx], normalize="true",
    )
    pd.DataFrame(confusion).to_csv(matrix_path, index=False)
    print("\nTop configurations:")
    print(result.head(15).to_string(index=False))
    print(f"saved {args.output}; time={time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
