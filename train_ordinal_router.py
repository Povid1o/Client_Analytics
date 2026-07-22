"""Train the validated hierarchical ordinal income router on all rows."""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    CSV_READ_KWARGS,
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
from src.feature_engineering import INCOME_ANCHORS, add_feature_groups
from src.modeling import inverse_power_target, power_transform_target
from src.hierarchical_routing import TREE_NODES, hierarchy_to_class_probabilities
from src.ordinal_routing import (
    DEFAULT_BAND_EDGES,
    income_band,
    route_predictions,
    temperature_scale,
)
from src.preprocessing import preprocess
from train_advanced_tail import (
    TAIL_THRESHOLDS,
    build_catboost_matrix,
    calibrate,
    fit_base_model,
    fit_specialist,
    fit_tail_classifier,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-iterations", type=int, default=1150)
    parser.add_argument("--tail-classifier-iterations", type=int, default=1000)
    parser.add_argument("--tail-expert-iterations", type=int, default=900)
    parser.add_argument("--ordinal-iterations", type=int, default=400)
    parser.add_argument("--band-expert-iterations", type=int, default=350)
    parser.add_argument(
        "--output", type=Path,
        default=OUTPUTS_DIR / "submission_hierarchical_router.csv",
    )
    return parser.parse_args()


def build_engineered_lgbm_matrices(train, test):
    combined = pd.concat([train, test], ignore_index=True, sort=False)
    combined = add_feature_groups(
        combined, ["anchors", "scale", "flows", "trends", "log_rank"]
    )
    features = [column for column in combined.columns if column not in NON_FEATURE_COLS]
    matrix = combined[features].copy()
    for column in CATEGORICAL_COLS:
        if column in matrix:
            values = matrix[column].astype(str).replace({"nan": "missing", "NaT": "missing"})
            matrix[column] = pd.Categorical(values).codes.astype("int32")
    return matrix.iloc[: len(train)].copy(), matrix.iloc[len(train):].copy()


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


def main():
    args = parse_args()
    train_raw = pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS, low_memory=False)
    test_raw = pd.read_csv(TEST_PATH, **CSV_READ_KWARGS, low_memory=False)
    train, _ = preprocess(train_raw, is_train=True)
    test, _ = preprocess(test_raw, is_train=False)
    y = train[TARGET_COL].to_numpy(dtype=float)
    weights = train[WEIGHT_COL].to_numpy(dtype=float)
    target_range = (float(y.min()), float(y.max()))
    labels = income_band(y)
    n_bands = len(DEFAULT_BAND_EDGES) - 1

    X_cat_train = build_catboost_matrix(train)
    X_cat_test = build_catboost_matrix(test)
    base_model = fit_base_model(X_cat_train, y, weights, args.base_iterations)
    base = calibrate(base_model.predict(X_cat_test), test, target_range)
    print("trained base CatBoost", flush=True)

    tail_probabilities = []
    tail_experts = []
    for index, threshold in enumerate(TAIL_THRESHOLDS):
        classifier = fit_tail_classifier(
            X_cat_train, y, weights, threshold, args.tail_classifier_iterations,
            seed=RANDOM_SEED + index,
        )
        tail_probabilities.append(classifier.predict_proba(X_cat_test)[:, 1])
        expert = fit_specialist(
            X_cat_train, y, weights, y >= threshold, args.tail_expert_iterations,
            seed=RANDOM_SEED + 10 + index,
        )
        tail_experts.append(calibrate(expert.predict(X_cat_test), test, target_range))
        print(f"trained tail block {int(threshold / 1000)}k", flush=True)
    p150, p300, p500 = tail_probabilities
    e150, e300, e500 = tail_experts
    tail_correction = (
        0.8 * p150 ** 2.5 * (e150 - base)
        + 0.4 * p300 ** 2.5 * (e300 - e150)
        + 0.4 * p500 ** 2.5 * (e500 - e300)
    )

    anchor_count_train = train[INCOME_ANCHORS].notna().sum(axis=1).to_numpy()
    anchor_count_test = test[INCOME_ANCHORS].notna().sum(axis=1).to_numpy()
    source_model = fit_specialist(
        X_cat_train, y, weights, anchor_count_train >= 3, 1000,
        seed=RANDOM_SEED + 503, l2_leaf_reg=15,
    )
    source_prediction = calibrate(source_model.predict(X_cat_test), test, target_range)
    source_correction = (anchor_count_test >= 3) * (source_prediction - base)
    print("trained multi-anchor source expert", flush=True)

    X_lgb_train, X_lgb_test = build_engineered_lgbm_matrices(train, test)
    node_probabilities = {}
    for node_index, (name, lower, upper, threshold) in enumerate(TREE_NODES):
        selected = (labels >= lower) & (labels <= upper)
        model = lgb.LGBMClassifier(
            **classifier_params(args.ordinal_iterations, RANDOM_SEED + 1400 + node_index)
        )
        model.fit(
            X_lgb_train.loc[selected],
            (labels[selected] >= threshold).astype(int),
            sample_weight=weights[selected],
        )
        node_probabilities[name] = model.predict_proba(X_lgb_test)[:, 1]
        print(
            f"trained hierarchical node {node_index + 1}/{len(TREE_NODES)}: {name}",
            flush=True,
        )
    probabilities = hierarchy_to_class_probabilities(node_probabilities)

    expert_candidates = np.empty((len(test), n_bands), dtype=float)
    for band in range(n_bands):
        lower = max(0, band - 1)
        upper = min(n_bands - 1, band + 1)
        selected = (labels >= lower) & (labels <= upper)
        model = lgb.LGBMRegressor(
            **expert_params(args.band_expert_iterations, RANDOM_SEED + 720 + band)
        )
        model.fit(
            X_lgb_train.iloc[np.flatnonzero(selected)],
            power_transform_target(y[selected], 0.25),
            sample_weight=weights[selected],
        )
        prediction = inverse_power_target(model.predict(X_lgb_test), 0.25)
        prediction = calibrate(prediction, test, target_range)
        expert_candidates[:, band] = np.clip(
            prediction, DEFAULT_BAND_EDGES[band], DEFAULT_BAND_EDGES[band + 1]
        )
        print(f"trained band expert {band + 1}/{n_bands}", flush=True)

    scaled_probability = temperature_scale(probabilities, 0.5)
    routed = route_predictions(base, scaled_probability, expert_candidates, mode="soft")
    ordinal_confidence = scaled_probability.max(axis=1)
    ordinal_correction = ordinal_confidence * (routed - base)

    # Common coefficients selected by minimising the worse result across the
    # random outer fold and the June temporal holdout.
    prediction = (
        base
        + 0.25 * tail_correction
        + 0.25 * source_correction
        + 0.75 * ordinal_correction
    )
    prediction = np.clip(prediction, *target_range)
    submission = pd.DataFrame({ID_COL: test[ID_COL], "predict": prediction})
    assert submission["predict"].notna().all()
    assert np.isfinite(submission["predict"]).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, sep=";", decimal=",")

    components_path = PARTIAL_OUTPUTS_DIR / f"{args.output.stem}_components.csv"
    components_path.parent.mkdir(parents=True, exist_ok=True)
    components = {
        ID_COL: test[ID_COL],
        "base": base,
        "tail_correction": tail_correction,
        "source_correction": source_correction,
        "ordinal_confidence": ordinal_confidence,
        "ordinal_routed_prediction": routed,
        "ordinal_correction": ordinal_correction,
        "predict": prediction,
    }
    components.update(
        {f"probability_band_{band}": probabilities[:, band] for band in range(n_bands)}
    )
    pd.DataFrame(components).to_csv(components_path, index=False)
    print(f"saved {args.output} and {components_path}")


if __name__ == "__main__":
    main()
