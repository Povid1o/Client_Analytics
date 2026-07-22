"""Train the validated tail/diversity/source ensemble on all train rows."""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import (
    CATEGORICAL_COLS,
    CSV_READ_KWARGS,
    DATE_COL,
    ID_COL,
    NON_FEATURE_COLS,
    OUTPUTS_DIR,
    PARTIAL_OUTPUTS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    WEIGHT_COL,
)
from src.feature_engineering import INCOME_ANCHORS
from src.modeling import (
    PowerTargetCatBoost,
    inverse_power_target,
    power_transform_target,
    prepare_cat_features,
)
from src.postprocessing import blend_salary_signal
from src.preprocessing import preprocess


TAIL_THRESHOLDS = (150_000.0, 300_000.0, 500_000.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-iterations", type=int, default=1150)
    parser.add_argument("--classifier-iterations", type=int, default=1000)
    parser.add_argument("--expert-iterations", type=int, default=900)
    parser.add_argument("--lgbm-estimators", type=int, default=1200)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "submission_advanced_tail.csv",
    )
    return parser.parse_args()


def build_catboost_matrix(frame):
    features = [column for column in frame.columns if column not in NON_FEATURE_COLS]
    matrix = frame[features].copy()
    matrix[DATE_COL] = matrix[DATE_COL].astype(str)
    return prepare_cat_features(matrix, CATEGORICAL_COLS)


def build_lgbm_matrices(train, test):
    features = [column for column in train.columns if column not in NON_FEATURE_COLS]
    combined = pd.concat([train[features], test[features]], ignore_index=True)
    for column in CATEGORICAL_COLS:
        if column in combined:
            values = combined[column].astype(str).replace({"nan": "missing", "NaT": "missing"})
            combined[column] = pd.Categorical(values).codes.astype("int32")
    return combined.iloc[: len(train)].copy(), combined.iloc[len(train) :].copy()


def calibrate(prediction, frame, target_range):
    prediction = blend_salary_signal(
        prediction,
        pd.to_numeric(frame["salary_6to12m_avg"], errors="coerce").to_numpy(),
        alpha=0.6,
    )
    return np.clip(prediction, *target_range)


def fit_base_model(X, y, weights, iterations):
    model = PowerTargetCatBoost(
        target_power=0.25,
        cat_features=CATEGORICAL_COLS,
        params={
            "iterations": iterations,
            "random_seed": RANDOM_SEED + 10,
            "early_stopping_rounds": None,
        },
    )
    model.fit(X, y, weights)
    return model


def fit_tail_classifier(X, y, weights, threshold, iterations, seed):
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
    model.fit(
        Pool(
            X,
            label=(y >= threshold).astype(int),
            weight=weights,
            cat_features=CATEGORICAL_COLS,
        )
    )
    return model


def fit_specialist(X, y, weights, selected, iterations, seed, l2_leaf_reg=20):
    model = PowerTargetCatBoost(
        target_power=0.25,
        cat_features=CATEGORICAL_COLS,
        params={
            "iterations": iterations,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": l2_leaf_reg,
            "random_seed": seed,
            "early_stopping_rounds": None,
        },
    )
    indices = np.flatnonzero(selected)
    model.fit(X.iloc[indices], y[indices], weights[indices])
    return model


def lgbm_params(estimators):
    return {
        "objective": "regression",
        "n_estimators": estimators,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "random_state": RANDOM_SEED + 602,
        "n_jobs": -1,
        "verbosity": -1,
    }


def main():
    args = parse_args()
    train_raw = pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS, low_memory=False)
    test_raw = pd.read_csv(TEST_PATH, **CSV_READ_KWARGS, low_memory=False)
    train, _ = preprocess(train_raw, is_train=True)
    test, _ = preprocess(test_raw, is_train=False)
    y = train[TARGET_COL].to_numpy(dtype=float)
    weights = train[WEIGHT_COL].to_numpy(dtype=float)
    target_range = (float(y.min()), float(y.max()))
    X_cat_train = build_catboost_matrix(train)
    X_cat_test = build_catboost_matrix(test)

    base_model = fit_base_model(X_cat_train, y, weights, args.base_iterations)
    base = calibrate(base_model.predict(X_cat_test), test, target_range)
    print("trained base CatBoost", flush=True)

    tail_probabilities = []
    tail_experts = []
    for index, threshold in enumerate(TAIL_THRESHOLDS):
        classifier = fit_tail_classifier(
            X_cat_train,
            y,
            weights,
            threshold,
            args.classifier_iterations,
            seed=RANDOM_SEED + index,
        )
        probability = classifier.predict_proba(X_cat_test)[:, 1]
        expert = fit_specialist(
            X_cat_train,
            y,
            weights,
            selected=y >= threshold,
            iterations=args.expert_iterations,
            seed=RANDOM_SEED + 10 + index,
        )
        expert_prediction = calibrate(expert.predict(X_cat_test), test, target_range)
        tail_probabilities.append(probability)
        tail_experts.append(expert_prediction)
        print(
            f"trained tail {int(threshold/1000)}k: "
            f"train_rows={(y >= threshold).sum():,}, mean_probability={probability.mean():.4f}",
            flush=True,
        )

    p150, p300, p500 = tail_probabilities
    e150, e300, e500 = tail_experts
    tail_correction = (
        0.8 * np.power(p150, 2.5) * (e150 - base)
        + 0.4 * np.power(p300, 2.5) * (e300 - e150)
        + 0.4 * np.power(p500, 2.5) * (e500 - e300)
    )

    X_lgb_train, X_lgb_test = build_lgbm_matrices(train, test)
    lgbm_model = lgb.LGBMRegressor(**lgbm_params(args.lgbm_estimators))
    lgbm_model.fit(
        X_lgb_train,
        power_transform_target(y, 0.25),
        sample_weight=weights,
    )
    lgbm_prediction = calibrate(
        inverse_power_target(lgbm_model.predict(X_lgb_test), 0.25),
        test,
        target_range,
    )
    print("trained LightGBM diversity model", flush=True)

    anchor_count_train = train[INCOME_ANCHORS].notna().sum(axis=1).to_numpy()
    anchor_count_test = test[INCOME_ANCHORS].notna().sum(axis=1).to_numpy()
    source_expert = fit_specialist(
        X_cat_train,
        y,
        weights,
        selected=anchor_count_train >= 3,
        iterations=1000,
        seed=RANDOM_SEED + 503,
        l2_leaf_reg=15,
    )
    source_prediction = calibrate(source_expert.predict(X_cat_test), test, target_range)
    source_gate = (anchor_count_test >= 3).astype(float)
    source_correction = source_gate * (source_prediction - base)
    print(
        f"trained multi-anchor expert: train_rows={(anchor_count_train >= 3).sum():,}, "
        f"test_rows={(anchor_count_test >= 3).sum():,}",
        flush=True,
    )

    prediction = (
        base
        + tail_correction
        + 0.5 * (lgbm_prediction - base)
        + 0.5 * source_correction
    )
    prediction = np.clip(prediction, *target_range)
    submission = pd.DataFrame({ID_COL: test[ID_COL], "predict": prediction})
    assert submission["predict"].notna().all()
    assert np.isfinite(submission["predict"]).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, sep=";", decimal=",")

    component_path = PARTIAL_OUTPUTS_DIR / f"{args.output.stem}_components.csv"
    component_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            ID_COL: test[ID_COL],
            "base": base,
            "tail_correction": tail_correction,
            "lgbm_prediction": lgbm_prediction,
            "source_gate": source_gate,
            "source_prediction": source_prediction,
            "predict": prediction,
        }
    ).to_csv(component_path, index=False)
    print(f"saved {args.output} and {component_path}")


if __name__ == "__main__":
    main()
