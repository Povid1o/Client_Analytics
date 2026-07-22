"""Independent LightGBM base/tail models for diversity against CatBoost."""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

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
from src.modeling import inverse_power_target, power_transform_target
from src.postprocessing import blend_salary_signal
from src.preprocessing import preprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-predictions",
        default=PARTIAL_OUTPUTS_DIR / "residual_stacking_outer_predictions.csv",
    )
    parser.add_argument("--threshold", type=float, default=150_000)
    parser.add_argument("--estimators", type=int, default=1200)
    parser.add_argument(
        "--output",
        default=PARTIAL_OUTPUTS_DIR / "lightgbm_diversity_results.csv",
    )
    return parser.parse_args()


def build_matrix(frame):
    features = [column for column in frame.columns if column not in NON_FEATURE_COLS]
    matrix = frame[features].copy()
    for column in CATEGORICAL_COLS:
        if column in matrix:
            values = matrix[column].astype(str).replace({"nan": "missing", "NaT": "missing"})
            matrix[column] = pd.Categorical(values).codes.astype("int32")
    return matrix


def common_params(estimators, seed):
    return {
        "n_estimators": estimators,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def calibrate(prediction, salary, target_range):
    prediction = blend_salary_signal(prediction, salary, alpha=0.6)
    return np.clip(prediction, *target_range)


def best_plain_blend(base, alternative, y, weights):
    best = (wmae(y, base, weights), 0.0)
    for alpha in np.arange(0, 1.01, 0.05):
        prediction = (1 - alpha) * base + alpha * alternative
        score = wmae(y, prediction, weights)
        if score < best[0]:
            best = (score, alpha)
    return best


def best_tail_gate(base, expert, probability, y, weights):
    best = (wmae(y, base, weights), None, None)
    for gamma in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        gate = np.power(probability, gamma)
        for strength in np.arange(0.1, 1.51, 0.1):
            prediction = base + strength * gate * (expert - base)
            score = wmae(y, np.clip(prediction, 20_000, 1_500_000), weights)
            if score < best[0]:
                best = (score, gamma, strength)
    return best


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
    X = build_matrix(frame)
    y = frame[TARGET_COL].to_numpy(dtype=float)
    weights = frame[WEIGHT_COL].to_numpy(dtype=float)
    salary = frame.iloc[val_idx]["salary_6to12m_avg"].to_numpy()
    target_range = (float(y.min()), float(y.max()))
    base = outer.set_index(ID_COL).loc[frame.iloc[val_idx][ID_COL], "base_prediction"].to_numpy()
    base_score = wmae(y[val_idx], base, weights[val_idx])

    classifier = lgb.LGBMClassifier(
        objective="binary", **common_params(args.estimators, RANDOM_SEED + 601)
    )
    train_label = (y[train_idx] >= args.threshold).astype(int)
    val_label = (y[val_idx] >= args.threshold).astype(int)
    classifier.fit(X.iloc[train_idx], train_label, sample_weight=weights[train_idx])
    tail_probability = classifier.predict_proba(X.iloc[val_idx])[:, 1]
    auc = roc_auc_score(val_label, tail_probability, sample_weight=weights[val_idx])
    average_precision = average_precision_score(
        val_label, tail_probability, sample_weight=weights[val_idx]
    )

    base_model = lgb.LGBMRegressor(
        objective="regression", **common_params(args.estimators, RANDOM_SEED + 602)
    )
    base_model.fit(
        X.iloc[train_idx],
        power_transform_target(y[train_idx], 0.25),
        sample_weight=weights[train_idx],
    )
    lgbm_base = calibrate(
        inverse_power_target(base_model.predict(X.iloc[val_idx]), 0.25),
        salary,
        target_range,
    )
    base_blend_score, base_alpha = best_plain_blend(
        base, lgbm_base, y[val_idx], weights[val_idx]
    )

    tail_train = train_idx[y[train_idx] >= args.threshold]
    tail_model = lgb.LGBMRegressor(
        objective="regression", **common_params(args.estimators, RANDOM_SEED + 603)
    )
    tail_model.fit(
        X.iloc[tail_train],
        power_transform_target(y[tail_train], 0.25),
        sample_weight=weights[tail_train],
    )
    tail_expert = calibrate(
        inverse_power_target(tail_model.predict(X.iloc[val_idx]), 0.25),
        salary,
        target_range,
    )
    gate_score, gamma, strength = best_tail_gate(
        base, tail_expert, tail_probability, y[val_idx], weights[val_idx]
    )

    result = pd.DataFrame(
        [
            {
                "base_wmae": base_score,
                "classifier_weighted_auc": auc,
                "classifier_weighted_ap": average_precision,
                "lgbm_base_wmae": wmae(y[val_idx], lgbm_base, weights[val_idx]),
                "base_blend_alpha": base_alpha,
                "base_blend_wmae": base_blend_score,
                "base_blend_delta": base_blend_score - base_score,
                "tail_gate_gamma": gamma,
                "tail_gate_strength": strength,
                "tail_gate_wmae": gate_score,
                "tail_gate_delta": gate_score - base_score,
            }
        ]
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    prediction_path = output_path.with_name(f"{output_path.stem}_predictions.csv")
    pd.DataFrame(
        {
            ID_COL: frame.iloc[val_idx][ID_COL].to_numpy(),
            TARGET_COL: y[val_idx],
            WEIGHT_COL: weights[val_idx],
            "base": base,
            "lgbm_base": lgbm_base,
            "tail_probability_150k_lgbm": tail_probability,
            "tail_expert_150k_lgbm": tail_expert,
        }
    ).to_csv(prediction_path, index=False)
    print(result.to_string(index=False))
    print(f"saved {args.output} and {prediction_path}; time={time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
