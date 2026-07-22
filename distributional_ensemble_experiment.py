"""Conditional quantile ensemble with a probabilistic tail gate."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

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
from src.modeling import inverse_power_target, power_transform_target, prepare_cat_features
from src.postprocessing import blend_salary_signal
from src.preprocessing import preprocess


QUANTILE_LEVELS = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-predictions",
        default=PARTIAL_OUTPUTS_DIR / "residual_stacking_outer_predictions.csv",
    )
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument(
        "--output",
        default=PARTIAL_OUTPUTS_DIR / "distributional_ensemble_results.csv",
    )
    return parser.parse_args()


def build_matrix(frame):
    features = [column for column in frame.columns if column not in NON_FEATURE_COLS]
    matrix = frame[features].copy()
    matrix[DATE_COL] = matrix[DATE_COL].astype(str)
    return prepare_cat_features(matrix, CATEGORICAL_COLS)


def fit_quantile_model(X, y, weights, iterations):
    alpha = ",".join(str(value) for value in QUANTILE_LEVELS)
    model = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={alpha}",
        iterations=iterations,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=10,
        random_seed=RANDOM_SEED + 401,
        verbose=False,
        allow_writing_files=False,
    )
    transformed = power_transform_target(y, 0.25)
    model.fit(Pool(X, label=transformed, weight=weights, cat_features=CATEGORICAL_COLS))
    return model


def fit_tail_classifier(X, y, weights, threshold=150_000):
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=10,
        random_seed=RANDOM_SEED + 402,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        Pool(
            X,
            label=(y >= threshold).astype(int),
            weight=weights,
            cat_features=CATEGORICAL_COLS,
        )
    )
    return model


def calibrate_quantiles(prediction, salary, target_range):
    prediction = inverse_power_target(prediction, 0.25)
    prediction = np.sort(prediction, axis=1)
    calibrated = np.empty_like(prediction)
    for column in range(prediction.shape[1]):
        calibrated[:, column] = blend_salary_signal(
            prediction[:, column], salary, alpha=0.6
        )
    return np.clip(calibrated, *target_range)


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

    quantile_model = fit_quantile_model(
        X.iloc[train_idx], y[train_idx], weights[train_idx], args.iterations
    )
    raw_quantiles = quantile_model.predict(X.iloc[val_idx])
    quantiles = calibrate_quantiles(
        raw_quantiles,
        frame.iloc[val_idx]["salary_6to12m_avg"].to_numpy(),
        (float(y.min()), float(y.max())),
    )
    classifier = fit_tail_classifier(
        X.iloc[train_idx], y[train_idx], weights[train_idx]
    )
    tail_probability = classifier.predict_proba(X.iloc[val_idx])[:, 1]

    rows = []
    candidate_predictions = {"base": base}
    for index, level in enumerate(QUANTILE_LEVELS):
        prediction = quantiles[:, index]
        name = f"q{int(level * 100):02d}"
        candidate_predictions[name] = prediction
        direct_score = wmae(y[val_idx], prediction, weights[val_idx])
        best = (base_score, 0.0)
        for alpha in np.arange(0, 1.01, 0.05):
            blended = (1 - alpha) * base + alpha * prediction
            score = wmae(y[val_idx], blended, weights[val_idx])
            if score < best[0]:
                best = (score, alpha)
        rows.append(
            {
                "variant": name,
                "direct_wmae": direct_score,
                "blend_wmae": best[0],
                "delta": best[0] - base_score,
                "alpha": best[1],
                "gamma": np.nan,
                "strength": np.nan,
            }
        )

    for index, level in enumerate(QUANTILE_LEVELS[2:], start=2):
        prediction = quantiles[:, index]
        best = (base_score, None, None)
        for gamma in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            gate = np.power(tail_probability, gamma)
            for strength in np.arange(0.1, 2.01, 0.1):
                gated = base + strength * gate * (prediction - base)
                score = wmae(
                    y[val_idx], np.clip(gated, y.min(), y.max()), weights[val_idx]
                )
                if score < best[0]:
                    best = (score, gamma, strength)
        rows.append(
            {
                "variant": f"tail_gated_q{int(level * 100):02d}",
                "direct_wmae": wmae(y[val_idx], prediction, weights[val_idx]),
                "blend_wmae": best[0],
                "delta": best[0] - base_score,
                "alpha": np.nan,
                "gamma": best[1],
                "strength": best[2],
            }
        )

    oracle_candidates = np.column_stack([base, quantiles])
    oracle_index = np.argmin(np.abs(oracle_candidates - y[val_idx, None]), axis=1)
    oracle_prediction = oracle_candidates[np.arange(len(val_idx)), oracle_index]
    oracle_score = wmae(y[val_idx], oracle_prediction, weights[val_idx])
    rows.append(
        {
            "variant": "oracle_best_quantile",
            "direct_wmae": oracle_score,
            "blend_wmae": oracle_score,
            "delta": oracle_score - base_score,
            "alpha": np.nan,
            "gamma": np.nan,
            "strength": np.nan,
        }
    )

    result = pd.DataFrame(rows).sort_values("blend_wmae")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    prediction_path = output_path.with_name(f"{output_path.stem}_predictions.csv")
    pd.DataFrame(
        {
            ID_COL: frame.iloc[val_idx][ID_COL].to_numpy(),
            TARGET_COL: y[val_idx],
            WEIGHT_COL: weights[val_idx],
            "tail_probability_150k": tail_probability,
            **candidate_predictions,
        }
    ).to_csv(prediction_path, index=False)
    print(f"base WMAE={base_score:,.0f}")
    print(result.to_string(index=False))
    print(f"saved {args.output} and {prediction_path}; time={time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
