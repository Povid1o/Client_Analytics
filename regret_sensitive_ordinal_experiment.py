"""Nested-OOF downstream-regret weighting for cumulative boundaries."""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from cost_sensitive_ordinal_experiment import (
    build_matrix, classifier_params, evaluate_probability_set,
)
from src.config import (
    CSV_READ_KWARGS, ID_COL, PARTIAL_OUTPUTS_DIR, RANDOM_SEED, TARGET_COL,
    TRAIN_PATH, WEIGHT_COL,
)
from src.cost_sensitive import boundary_regret_weights
from src.metrics import wmae
from src.modeling import inverse_power_target, power_transform_target
from src.ordinal_routing import (
    DEFAULT_BAND_EDGES, cumulative_to_class_probabilities, income_band,
)
from src.postprocessing import blend_salary_signal
from src.preprocessing import preprocess
from src.validation import get_folds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-predictions", type=Path,
        default=PARTIAL_OUTPUTS_DIR / "ordinal_cumulative_results_predictions.csv",
    )
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--expert-iterations", type=int, default=250)
    parser.add_argument("--classifier-iterations", type=int, default=400)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument(
        "--output", type=Path,
        default=PARTIAL_OUTPUTS_DIR / "regret_sensitive_ordinal_results.csv",
    )
    return parser.parse_args()


def expert_params(iterations, seed):
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


def expanded_mask(labels, band, n_bands):
    return (labels >= max(0, band - 1)) & (labels <= min(n_bands - 1, band + 1))


def create_oof_experts(X, frame, y, weights, labels, n_folds, iterations):
    n_bands = len(DEFAULT_BAND_EDGES) - 1
    prediction = np.full((len(y), n_bands), np.nan)
    folds = get_folds(y, n_folds=n_folds, seed=RANDOM_SEED + 1100)
    for fold, (train_idx, val_idx) in enumerate(folds):
        for band in range(n_bands):
            selected_local = expanded_mask(labels[train_idx], band, n_bands)
            selected = train_idx[selected_local]
            model = lgb.LGBMRegressor(
                **expert_params(iterations, RANDOM_SEED + 1110 + fold * n_bands + band)
            )
            model.fit(
                X.iloc[selected], power_transform_target(y[selected], 0.25),
                sample_weight=weights[selected],
            )
            band_prediction = inverse_power_target(model.predict(X.iloc[val_idx]), 0.25)
            band_prediction = blend_salary_signal(
                band_prediction,
                pd.to_numeric(
                    frame.iloc[val_idx]["salary_6to12m_avg"], errors="coerce"
                ).to_numpy(),
                alpha=0.6,
            )
            prediction[val_idx, band] = np.clip(
                band_prediction, DEFAULT_BAND_EDGES[band], DEFAULT_BAND_EDGES[band + 1]
            )
        print(f"inner expert fold {fold + 1}/{n_folds} complete", flush=True)
    if not np.isfinite(prediction).all():
        raise RuntimeError("nested OOF expert matrix is incomplete")
    return prediction


def fit_regret_boundaries(
    X_train, labels, y, weights, oof_experts, X_val, iterations, strength
):
    cumulative = []
    diagnostics = []
    for threshold in range(1, len(DEFAULT_BAND_EDGES) - 1):
        sample_weight, regret = boundary_regret_weights(
            y, weights, labels, oof_experts, threshold, strength
        )
        model = lgb.LGBMClassifier(
            **classifier_params(
                iterations, RANDOM_SEED + 1200 + threshold + int(strength * 100)
            )
        )
        model.fit(
            X_train, (labels >= threshold).astype(int), sample_weight=sample_weight
        )
        cumulative.append(model.predict_proba(X_val)[:, 1])
        diagnostics.append(
            {
                "strength": strength,
                "boundary": DEFAULT_BAND_EDGES[threshold],
                "positive_regret_rate": float(np.mean(regret > 0)),
                "median_positive_regret": float(np.median(regret[regret > 0])),
                "p95_regret": float(np.quantile(regret, 0.95)),
            }
        )
        print(
            f"strength={strength:g}: regret boundary "
            f"{int(DEFAULT_BAND_EDGES[threshold]/1000)}k ({threshold}/7)",
            flush=True,
        )
    return cumulative_to_class_probabilities(np.column_stack(cumulative)), diagnostics


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
    outer_frame = frame.iloc[train_idx].reset_index(drop=True)
    X = build_matrix(frame)
    X_outer = X.iloc[train_idx].reset_index(drop=True)
    y = frame[TARGET_COL].to_numpy(dtype=float)
    weights = frame[WEIGHT_COL].to_numpy(dtype=float)
    labels = income_band(y)
    y_outer = y[train_idx]
    w_outer = weights[train_idx]
    labels_outer = labels[train_idx]
    ordered = reference.loc[frame.iloc[val_idx][ID_COL]]
    base = ordered["base"].to_numpy(dtype=float)
    experts = ordered[[f"expert_band_{i}" for i in range(8)]].to_numpy(dtype=float)
    baseline_probability = ordered[
        [f"probability_band_{i}" for i in range(8)]
    ].to_numpy(dtype=float)
    base_score = wmae(y[val_idx], base, weights[val_idx])
    print(f"base WMAE={base_score:,.0f}; creating nested OOF experts", flush=True)

    oof_experts = create_oof_experts(
        X_outer, outer_frame, y_outer, w_outer, labels_outer,
        args.inner_folds, args.expert_iterations,
    )
    np.save(args.output.with_name(f"{args.output.stem}_oof_experts.npy"), oof_experts)

    probability_sets = {"baseline": baseline_probability}
    diagnostic_rows = []
    for strength in args.strengths:
        probability, diagnostics = fit_regret_boundaries(
            X_outer, labels_outer, y_outer, w_outer, oof_experts,
            X.iloc[val_idx], args.classifier_iterations, strength,
        )
        name = f"regret_{strength:g}"
        probability_sets[name] = probability
        diagnostic_rows.extend(diagnostics)
        for alpha in (0.25, 0.5, 0.75):
            probability_sets[f"blend_{strength:g}_{alpha:.2f}"] = (
                (1 - alpha) * baseline_probability + alpha * probability
            )

    results = []
    output = {
        ID_COL: frame.iloc[val_idx][ID_COL].to_numpy(),
        TARGET_COL: y[val_idx], WEIGHT_COL: weights[val_idx], "base": base,
    }
    for name, probability in probability_sets.items():
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
    pd.DataFrame(output).to_csv(
        args.output.with_name(f"{args.output.stem}_predictions.csv"), index=False
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        args.output.with_name(f"{args.output.stem}_regret_diagnostics.csv"), index=False
    )
    print("\nTop configurations:")
    print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
