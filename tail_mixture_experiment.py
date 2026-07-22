"""Evaluate probabilistic tail gates and tail-specialist regressors.

The script reuses strict outer-fold base predictions produced by
``residual_stacking_experiment.py``. Classifiers and specialists train only
on the complementary outer-train partition.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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
from src.modeling import PowerTargetCatBoost, prepare_cat_features
from src.postprocessing import blend_salary_signal
from src.preprocessing import preprocess


TAIL_THRESHOLDS = (150_000.0, 300_000.0, 500_000.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-predictions",
        default=PARTIAL_OUTPUTS_DIR / "residual_stacking_outer_predictions.csv",
    )
    parser.add_argument("--classifier-iterations", type=int, default=1000)
    parser.add_argument("--expert-iterations", type=int, default=900)
    parser.add_argument(
        "--output",
        default=PARTIAL_OUTPUTS_DIR / "tail_mixture_results.csv",
    )
    return parser.parse_args()


def build_matrix(frame):
    feature_columns = [column for column in frame.columns if column not in NON_FEATURE_COLS]
    matrix = frame[feature_columns].copy()
    matrix[DATE_COL] = matrix[DATE_COL].astype(str)
    return prepare_cat_features(matrix, CATEGORICAL_COLS)


def fit_classifier(X, label, weights, iterations, seed):
    model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=iterations,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=10,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(X, label=label, weight=weights, cat_features=CATEGORICAL_COLS))
    return model


def fit_tail_expert(X, y, weights, threshold, iterations, seed):
    selected = y >= threshold
    model = PowerTargetCatBoost(
        target_power=0.25,
        cat_features=CATEGORICAL_COLS,
        params={
            "iterations": iterations,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": 20,
            "random_seed": seed,
            "early_stopping_rounds": None,
        },
    )
    model.fit(X.iloc[np.flatnonzero(selected)], y[selected], weights[selected])
    return model, int(selected.sum())


def evaluate_gate(base, expert, probability, y, weights):
    rows = []
    for gamma in (0.5, 1.0, 1.5, 2.0, 3.0):
        gate = np.power(np.clip(probability, 0, 1), gamma)
        for strength in np.arange(0.1, 1.51, 0.1):
            prediction = base + strength * gate * (expert - base)
            score = wmae(y, np.clip(prediction, 20_000, 1_500_000), weights)
            rows.append((score, gamma, strength))
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

    outer_by_id = outer.set_index(ID_COL)
    ordered_outer = outer_by_id.loc[frame.iloc[val_idx][ID_COL]]
    base = ordered_outer["base_prediction"].to_numpy(dtype=float)
    y = frame[TARGET_COL].to_numpy(dtype=float)
    weights = frame[WEIGHT_COL].to_numpy(dtype=float)
    X = build_matrix(frame)
    base_score = wmae(y[val_idx], base, weights[val_idx])
    print(f"base WMAE={base_score:,.0f}")

    results = []
    prediction_columns = {"base_prediction": base}
    for threshold_i, threshold in enumerate(TAIL_THRESHOLDS):
        label_train = (y[train_idx] >= threshold).astype(int)
        label_val = (y[val_idx] >= threshold).astype(int)
        classifier = fit_classifier(
            X.iloc[train_idx],
            label_train,
            weights[train_idx],
            args.classifier_iterations,
            seed=RANDOM_SEED + threshold_i,
        )
        probability = classifier.predict_proba(X.iloc[val_idx])[:, 1]
        expert_model, expert_rows = fit_tail_expert(
            X.iloc[train_idx].reset_index(drop=True),
            y[train_idx],
            weights[train_idx],
            threshold,
            args.expert_iterations,
            seed=RANDOM_SEED + 10 + threshold_i,
        )
        expert_prediction = expert_model.predict(X.iloc[val_idx])
        expert_prediction = blend_salary_signal(
            expert_prediction,
            frame.iloc[val_idx]["salary_6to12m_avg"].to_numpy(),
            alpha=0.6,
        )
        expert_prediction = np.clip(expert_prediction, y.min(), y.max())

        auc = roc_auc_score(label_val, probability, sample_weight=weights[val_idx])
        average_precision = average_precision_score(
            label_val, probability, sample_weight=weights[val_idx]
        )
        gate_score, gamma, strength = evaluate_gate(
            base, expert_prediction, probability, y[val_idx], weights[val_idx]
        )
        oracle_hard = np.where(label_val.astype(bool), expert_prediction, base)
        oracle_hard_score = wmae(y[val_idx], oracle_hard, weights[val_idx])
        oracle_best = np.where(
            np.abs(y[val_idx] - expert_prediction) < np.abs(y[val_idx] - base),
            expert_prediction,
            base,
        )
        oracle_best_score = wmae(y[val_idx], oracle_best, weights[val_idx])
        key = int(threshold / 1000)
        prediction_columns[f"tail_probability_{key}k"] = probability
        prediction_columns[f"tail_expert_{key}k"] = expert_prediction
        results.append(
            {
                "threshold": threshold,
                "positive_train_rows": int(label_train.sum()),
                "expert_train_rows": expert_rows,
                "weighted_auc": auc,
                "weighted_average_precision": average_precision,
                "gate_gamma": gamma,
                "gate_strength": strength,
                "base_wmae": base_score,
                "soft_gate_wmae": gate_score,
                "soft_gate_delta": gate_score - base_score,
                "oracle_hard_wmae": oracle_hard_score,
                "oracle_hard_delta": oracle_hard_score - base_score,
                "oracle_best_wmae": oracle_best_score,
                "oracle_best_delta": oracle_best_score - base_score,
            }
        )
        print(
            f"threshold={key}k rows={expert_rows:,} AUC={auc:.4f} AP={average_precision:.4f} | "
            f"soft={gate_score:,.0f} ({gate_score-base_score:+,.0f}) "
            f"oracle-hard={oracle_hard_score:,.0f} ({oracle_hard_score-base_score:+,.0f}) "
            f"oracle-best={oracle_best_score:,.0f} ({oracle_best_score-base_score:+,.0f}) "
            f"gamma={gamma} strength={strength:.1f}",
            flush=True,
        )

    result = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    prediction_frame = pd.DataFrame(
        {
            ID_COL: frame.iloc[val_idx][ID_COL].to_numpy(),
            TARGET_COL: y[val_idx],
            WEIGHT_COL: weights[val_idx],
            **prediction_columns,
        }
    )
    output_path = Path(args.output)
    prediction_path = output_path.with_name(f"{output_path.stem}_predictions.csv")
    prediction_frame.to_csv(prediction_path, index=False)
    print(result.to_string(index=False))
    print(f"saved {args.output} and {prediction_path}; time={time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
