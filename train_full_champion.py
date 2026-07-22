"""Train the validated 26-model full champion and create its submission."""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    CSV_READ_KWARGS,
    ID_COL,
    OUTPUTS_DIR,
    PARTIAL_OUTPUTS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    WEIGHT_COL,
)
from src.feature_engineering import INCOME_ANCHORS
from src.hierarchical_routing import TREE_NODES, hierarchy_to_class_probabilities
from src.modeling import inverse_power_target, power_transform_target
from src.ordinal_routing import (
    DEFAULT_BAND_EDGES,
    adaptive_temperature_scale,
    income_band,
    route_predictions,
)
from src.preprocessing import preprocess
from src.production import (
    TAIL_THRESHOLDS,
    band_expert_params,
    build_catboost_matrix,
    build_engineered_lgbm_matrices,
    calibrate,
    classifier_params,
    fit_base_model,
    fit_specialist,
    fit_tail_classifier,
)
from src.quantile_head import fit_predict_weighted_lognormal
from src.wmae_distribution import recover_weight_rule


DISTRIBUTION_BLEND = 0.20


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-iterations", type=int, default=1150)
    parser.add_argument("--tail-classifier-iterations", type=int, default=1000)
    parser.add_argument("--tail-expert-iterations", type=int, default=900)
    parser.add_argument("--router-iterations", type=int, default=400)
    parser.add_argument("--band-expert-iterations", type=int, default=350)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "submission_full_champion.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_raw = pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS, low_memory=False)
    test_raw = pd.read_csv(TEST_PATH, **CSV_READ_KWARGS, low_memory=False)
    train, _ = preprocess(train_raw, is_train=True)
    test, _ = preprocess(test_raw, is_train=False)
    target = train[TARGET_COL].to_numpy(dtype=float)
    weights = train[WEIGHT_COL].to_numpy(dtype=float)
    target_range = (float(target.min()), float(target.max()))
    labels = income_band(target)
    n_bands = len(DEFAULT_BAND_EDGES) - 1

    cat_train = build_catboost_matrix(train)
    cat_test = build_catboost_matrix(test)
    base_model = fit_base_model(cat_train, target, weights, args.base_iterations)
    base = calibrate(base_model.predict(cat_test), test, target_range)
    print("trained base CatBoost (1/26)", flush=True)

    tail_probabilities = []
    tail_experts = []
    for index, threshold in enumerate(TAIL_THRESHOLDS):
        classifier = fit_tail_classifier(
            cat_train,
            target,
            weights,
            threshold,
            args.tail_classifier_iterations,
            seed=RANDOM_SEED + index,
        )
        tail_probabilities.append(classifier.predict_proba(cat_test)[:, 1])
        expert = fit_specialist(
            cat_train,
            target,
            weights,
            target >= threshold,
            args.tail_expert_iterations,
            seed=RANDOM_SEED + 10 + index,
        )
        tail_experts.append(calibrate(expert.predict(cat_test), test, target_range))
        print(f"trained tail classifier/expert {int(threshold / 1000)}k", flush=True)

    p150, p300, p500 = tail_probabilities
    e150, e300, e500 = tail_experts
    tail_correction = (
        0.8 * p150**2.5 * (e150 - base)
        + 0.4 * p300**2.5 * (e300 - e150)
        + 0.4 * p500**2.5 * (e500 - e300)
    )

    anchor_count_train = train[INCOME_ANCHORS].notna().sum(axis=1).to_numpy()
    anchor_count_test = test[INCOME_ANCHORS].notna().sum(axis=1).to_numpy()
    source_model = fit_specialist(
        cat_train,
        target,
        weights,
        anchor_count_train >= 3,
        1000,
        seed=RANDOM_SEED + 503,
        l2_leaf_reg=15,
    )
    source_prediction = calibrate(source_model.predict(cat_test), test, target_range)
    source_correction = (anchor_count_test >= 3) * (source_prediction - base)
    print("trained multi-anchor source expert (8/26)", flush=True)

    lgb_train, lgb_test = build_engineered_lgbm_matrices(train, test)
    node_probabilities = {}
    for node_index, (name, lower, upper, threshold) in enumerate(TREE_NODES):
        selected = (labels >= lower) & (labels <= upper)
        model = lgb.LGBMClassifier(
            **classifier_params(
                args.router_iterations, RANDOM_SEED + 1400 + node_index
            )
        )
        model.fit(
            lgb_train.loc[selected],
            (labels[selected] >= threshold).astype(int),
            sample_weight=weights[selected],
        )
        node_probabilities[name] = model.predict_proba(lgb_test)[:, 1]
        print(f"trained hierarchical node {node_index + 1}/7", flush=True)
    probabilities = hierarchy_to_class_probabilities(node_probabilities)

    candidates = np.empty((len(test), n_bands), dtype=float)
    for band in range(n_bands):
        selected = (labels >= max(0, band - 1)) & (
            labels <= min(n_bands - 1, band + 1)
        )
        model = lgb.LGBMRegressor(
            **band_expert_params(
                args.band_expert_iterations, RANDOM_SEED + 720 + band
            )
        )
        model.fit(
            lgb_train.iloc[np.flatnonzero(selected)],
            power_transform_target(target[selected], 0.25),
            sample_weight=weights[selected],
        )
        prediction = inverse_power_target(model.predict(lgb_test), 0.25)
        prediction = calibrate(prediction, test, target_range)
        candidates[:, band] = np.clip(
            prediction, DEFAULT_BAND_EDGES[band], DEFAULT_BAND_EDGES[band + 1]
        )
        print(f"trained band expert {band + 1}/8", flush=True)

    scaled = adaptive_temperature_scale(
        probabilities, default_temperature=0.5, overrides={0: 0.3, 4: 0.3}
    )
    routed = route_predictions(base, scaled, candidates, mode="soft")
    ordinal_confidence = scaled.max(axis=1)
    ordinal_correction = ordinal_confidence * (routed - base)
    hierarchical_prediction = np.clip(
        base
        + 0.25 * tail_correction
        + 0.25 * source_correction
        + 0.75 * ordinal_correction,
        *target_range,
    )

    weight_rule = recover_weight_rule(target, weights)
    distribution_prediction, _ = fit_predict_weighted_lognormal(
        lgb_train, target, lgb_test, weight_rule
    )
    prediction = (
        (1.0 - DISTRIBUTION_BLEND) * hierarchical_prediction
        + DISTRIBUTION_BLEND * distribution_prediction
    )

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
        "hierarchical_prediction": hierarchical_prediction,
        "weighted_lognormal": distribution_prediction,
        "predict": prediction,
    }
    components.update(
        {f"probability_band_{band}": probabilities[:, band] for band in range(n_bands)}
    )
    components.update(
        {f"candidate_band_{band}": candidates[:, band] for band in range(n_bands)}
    )
    pd.DataFrame(components).to_csv(components_path, index=False)
    print(f"saved {args.output} and {components_path}", flush=True)


if __name__ == "__main__":
    main()
