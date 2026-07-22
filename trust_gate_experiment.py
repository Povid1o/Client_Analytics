"""Train a cost-sensitive gate that can abstain from harmful ordinal routes."""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import (
    CSV_READ_KWARGS, DATE_COL, ID_COL, PARTIAL_OUTPUTS_DIR, RANDOM_SEED,
    TARGET_COL, TRAIN_PATH, WEIGHT_COL,
)
from src.metrics import wmae
from src.ordinal_routing import income_band, temperature_scale
from src.preprocessing import preprocess
from src.trust_gate import apply_abstention_policy, build_trust_features


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=650)
    parser.add_argument(
        "--output", type=Path, default=PARTIAL_OUTPUTS_DIR / "trust_gate_results.csv"
    )
    return parser.parse_args()


def auxiliary_paths(kind):
    if kind == "random":
        return (
            PARTIAL_OUTPUTS_DIR / "tail_mixture_results_predictions.csv",
            PARTIAL_OUTPUTS_DIR / "source_experts_results_predictions.csv",
        )
    return (
        PARTIAL_OUTPUTS_DIR / "tail_mixture_outer_predictions.csv",
        PARTIAL_OUTPUTS_DIR / "source_experts_time_results_predictions.csv",
    )


def load_components(kind, raw_by_id):
    ordinal_path = PARTIAL_OUTPUTS_DIR / (
        "ordinal_cumulative_results_predictions.csv"
        if kind == "random" else "ordinal_cumulative_time_results_predictions.csv"
    )
    ordinal = pd.read_csv(ordinal_path).set_index(ID_COL)
    tail_path, source_path = auxiliary_paths(kind)
    tail = pd.read_csv(tail_path).set_index(ID_COL).loc[ordinal.index]
    source = pd.read_csv(source_path).set_index(ID_COL).loc[ordinal.index]
    base = ordinal["base"].to_numpy(dtype=float)
    tail_correction = (
        0.8 * tail["tail_probability_150k"].to_numpy() ** 2.5
        * (tail["tail_expert_150k"].to_numpy() - base)
        + 0.4 * tail["tail_probability_300k"].to_numpy() ** 2.5
        * (tail["tail_expert_300k"].to_numpy() - tail["tail_expert_150k"].to_numpy())
        + 0.4 * tail["tail_probability_500k"].to_numpy() ** 2.5
        * (tail["tail_expert_500k"].to_numpy() - tail["tail_expert_300k"].to_numpy())
    )
    source_correction = (
        source["gate_multi_anchor_ge3"].to_numpy()
        * (source["expert_multi_anchor_ge3"].to_numpy() - base)
    )
    components = ordinal.copy()
    components["tail_correction"] = tail_correction
    components["source_correction"] = source_correction
    components["ordinal_routed_prediction"] = (
        base + ordinal["ordinal_correction"].to_numpy()
        if "ordinal_correction" in ordinal else np.nan
    )
    # Reconstruct the fixed cumulative-only correction from stored candidates.
    from src.ordinal_routing import route_predictions, temperature_scale
    probabilities = ordinal[[f"probability_band_{i}" for i in range(8)]].to_numpy()
    experts = ordinal[[f"expert_band_{i}" for i in range(8)]].to_numpy()
    scaled = temperature_scale(probabilities, 0.5)
    routed = route_predictions(base, scaled, experts, mode="soft")
    correction = scaled.max(axis=1) * (routed - base)
    components["ordinal_routed_prediction"] = routed
    components["ordinal_correction"] = correction
    raw = raw_by_id.loc[components.index].copy()
    components = components.reset_index()
    raw = raw.reset_index()
    nonordinal = base + 0.25 * tail_correction + 0.25 * source_correction
    current = nonordinal + 0.75 * correction
    target = components[TARGET_COL].to_numpy(dtype=float)
    utility = np.abs(target - nonordinal) - np.abs(target - current)
    components["nonordinal_prediction"] = nonordinal
    components["current_prediction"] = current
    components["utility"] = utility
    components["harm"] = (utility < 0).astype(int)
    stored_probability = components[[f"probability_band_{i}" for i in range(8)]].to_numpy()
    predicted_band = temperature_scale(stored_probability, 0.5).argmax(axis=1)
    true_band = income_band(target)
    components["far_misroute"] = (np.abs(predicted_band - true_band) >= 2).astype(int)
    return components, raw


def model_params(iterations, seed):
    return {
        "objective": "binary",
        "n_estimators": iterations,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 2.0,
        "reg_lambda": 20.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def cost_weights(components):
    weights = (
        components[WEIGHT_COL].to_numpy(dtype=float)
        * np.abs(components["utility"].to_numpy(dtype=float))
    )
    cap = np.quantile(weights[weights > 0], 0.995)
    weights = np.clip(weights, 1e-6, cap)
    return weights / np.mean(weights)


def routing_weights(components):
    weights = components[WEIGHT_COL].to_numpy(dtype=float)
    return weights / np.mean(weights)


def fit_oof(features, label, sample_weight, iterations):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + 810)
    prediction = np.full(len(features), np.nan)
    fold_index = np.full(len(features), -1, dtype=int)
    for fold, (train_idx, val_idx) in enumerate(folds.split(features, label)):
        model = lgb.LGBMClassifier(**model_params(iterations, RANDOM_SEED + 820 + fold))
        model.fit(features.iloc[train_idx], label[train_idx], sample_weight=sample_weight[train_idx])
        prediction[val_idx] = model.predict_proba(features.iloc[val_idx])[:, 1]
        fold_index[val_idx] = fold
    return prediction, fold_index


def fit_oof_alpha(features, target, sample_weight, iterations):
    bins = pd.qcut(target, 10, labels=False, duplicates="drop")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + 910)
    prediction = np.full(len(features), np.nan)
    fold_index = np.full(len(features), -1, dtype=int)
    for fold, (train_idx, val_idx) in enumerate(folds.split(features, bins)):
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=iterations,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=20.0,
            random_state=RANDOM_SEED + 920 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(features.iloc[train_idx], target[train_idx], sample_weight=sample_weight[train_idx])
        prediction[val_idx] = model.predict(features.iloc[val_idx])
        fold_index[val_idx] = fold
    return np.clip(prediction, 0.0, 1.0), fold_index


def policy_grid(components, probability, fold_index):
    y = components[TARGET_COL].to_numpy(dtype=float)
    w = components[WEIGHT_COL].to_numpy(dtype=float)
    nonordinal = components["nonordinal_prediction"].to_numpy(dtype=float)
    correction = components["ordinal_correction"].to_numpy(dtype=float)
    base_score = wmae(y, components["current_prediction"].to_numpy(), w)
    rows = []
    policies = []
    for threshold in np.arange(0.1, 0.91, 0.05):
        for retain in (0.0, 0.25, 0.5, 0.75):
            policies.append({"kind": "hard", "threshold": threshold, "retain": retain})
    for gamma in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        policies.append({"kind": "soft", "gamma": gamma})
    for policy in policies:
        prediction = apply_abstention_policy(nonordinal, correction, probability, policy)
        fold_deltas = []
        for fold in range(5):
            selected = fold_index == fold
            fold_deltas.append(
                wmae(y[selected], prediction[selected], w[selected])
                - wmae(y[selected], components.loc[selected, "current_prediction"], w[selected])
            )
        score = wmae(y, prediction, w)
        rows.append(
            {
                **policy,
                "wmae": score,
                "delta": score - base_score,
                "worst_fold_delta": max(fold_deltas),
                "improved_folds": int(np.sum(np.asarray(fold_deltas) < 0)),
            }
        )
    result = pd.DataFrame(rows)
    eligible = result[result["improved_folds"] >= 4]
    if len(eligible) == 0:
        eligible = result
    best = eligible.sort_values(["worst_fold_delta", "wmae"]).iloc[0].to_dict()
    policy = {key: best[key] for key in ("kind", "threshold", "retain", "gamma") if key in best and pd.notna(best[key])}
    return result.sort_values(["worst_fold_delta", "wmae"]), policy


def main():
    args = parse_args()
    raw, _ = preprocess(
        pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS, low_memory=False), is_train=True
    )
    raw_by_id = raw.set_index(ID_COL)
    random_components, random_raw = load_components("random", raw_by_id)
    time_components, time_raw = load_components("time", raw_by_id)
    cutoff = raw[DATE_COL].max()
    meta_train_mask = random_raw[DATE_COL].to_numpy() < cutoff
    random_components = random_components.loc[meta_train_mask].reset_index(drop=True)
    random_raw = random_raw.loc[meta_train_mask].reset_index(drop=True)

    summaries = []
    prediction_output = time_components[[ID_COL, TARGET_COL, WEIGHT_COL]].copy()
    for objective, label_column, weight_builder in (
        ("harm", "harm", cost_weights),
        ("far", "far_misroute", routing_weights),
    ):
        for include_raw, feature_name in ((False, "compact"), (True, "full_numeric")):
            name = f"{objective}_{feature_name}"
            train_features = build_trust_features(random_components, random_raw, include_raw)
            time_features = build_trust_features(time_components, time_raw, include_raw)
            label = random_components[label_column].to_numpy(dtype=int)
            sample_weight = weight_builder(random_components)
            oof_probability, fold_index = fit_oof(
                train_features, label, sample_weight, args.iterations
            )
            auc = roc_auc_score(label, oof_probability, sample_weight=sample_weight)
            ap = average_precision_score(label, oof_probability, sample_weight=sample_weight)
            policies, policy = policy_grid(random_components, oof_probability, fold_index)
            policies.insert(0, "feature_set", feature_name)
            policies.insert(0, "objective", objective)
            summaries.append(policies)

            final_model = lgb.LGBMClassifier(**model_params(args.iterations, RANDOM_SEED + 899))
            final_model.fit(train_features, label, sample_weight=sample_weight)
            time_probability = final_model.predict_proba(time_features)[:, 1]
            time_prediction = apply_abstention_policy(
                time_components["nonordinal_prediction"],
                time_components["ordinal_correction"],
                time_probability,
                policy,
            )
            time_y = time_components[TARGET_COL].to_numpy(dtype=float)
            time_w = time_components[WEIGHT_COL].to_numpy(dtype=float)
            current_score = wmae(time_y, time_components["current_prediction"], time_w)
            gated_score = wmae(time_y, time_prediction, time_w)
            selected = policies.iloc[0]
            print(
                f"{name}: weighted-AUC={auc:.4f} AP={ap:.4f}; "
                f"meta-OOF delta={selected['delta']:+,.0f}, folds={int(selected['improved_folds'])}/5; "
                f"time delta={gated_score-current_score:+,.0f}; policy={policy}",
                flush=True,
            )
            prediction_output[f"probability_{name}"] = time_probability
            prediction_output[f"prediction_{name}"] = time_prediction

    # Continuous trust: predict the best point on the segment between the
    # non-ordinal prediction and the current ordinal prediction.
    for include_raw, feature_name in ((False, "compact"), (True, "full_numeric")):
        train_features = build_trust_features(random_components, random_raw, include_raw)
        time_features = build_trust_features(time_components, time_raw, include_raw)
        y = random_components[TARGET_COL].to_numpy(dtype=float)
        nonordinal = random_components["nonordinal_prediction"].to_numpy(dtype=float)
        step = 0.75 * random_components["ordinal_correction"].to_numpy(dtype=float)
        alpha_target = np.ones(len(step), dtype=float)
        valid = np.abs(step) > 1e-6
        alpha_target[valid] = np.clip((y[valid] - nonordinal[valid]) / step[valid], 0.0, 1.0)
        alpha_weight = random_components[WEIGHT_COL].to_numpy(dtype=float) * np.abs(step)
        cap = np.quantile(alpha_weight[alpha_weight > 0], 0.995)
        alpha_weight = np.clip(alpha_weight, 1e-6, cap)
        alpha_weight /= alpha_weight.mean()
        oof_alpha, fold_index = fit_oof_alpha(
            train_features, alpha_target, alpha_weight, args.iterations
        )
        current_score = wmae(y, random_components["current_prediction"], random_components[WEIGHT_COL])
        candidates = []
        for shrinkage in np.arange(0.0, 1.01, 0.1):
            factor = 1.0 - shrinkage * (1.0 - oof_alpha)
            prediction = nonordinal + factor * step
            fold_deltas = []
            for fold in range(5):
                selected = fold_index == fold
                fold_deltas.append(
                    wmae(y[selected], prediction[selected], random_components.loc[selected, WEIGHT_COL])
                    - wmae(
                        y[selected], random_components.loc[selected, "current_prediction"],
                        random_components.loc[selected, WEIGHT_COL],
                    )
                )
            score = wmae(y, prediction, random_components[WEIGHT_COL])
            candidates.append((max(fold_deltas), score, shrinkage, fold_deltas))
        eligible = [row for row in candidates if sum(delta < 0 for delta in row[3]) >= 4]
        best = min(eligible or candidates)

        final_model = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=args.iterations,
            learning_rate=0.025, num_leaves=31, min_child_samples=100,
            subsample=0.85, colsample_bytree=0.8, reg_alpha=2.0, reg_lambda=20.0,
            random_state=RANDOM_SEED + 999, n_jobs=-1, verbosity=-1,
        )
        final_model.fit(train_features, alpha_target, sample_weight=alpha_weight)
        time_alpha = np.clip(final_model.predict(time_features), 0.0, 1.0)
        time_factor = 1.0 - best[2] * (1.0 - time_alpha)
        time_nonordinal = time_components["nonordinal_prediction"].to_numpy(dtype=float)
        time_step = 0.75 * time_components["ordinal_correction"].to_numpy(dtype=float)
        time_prediction = time_nonordinal + time_factor * time_step
        time_y = time_components[TARGET_COL].to_numpy(dtype=float)
        time_w = time_components[WEIGHT_COL].to_numpy(dtype=float)
        time_current = wmae(time_y, time_components["current_prediction"], time_w)
        time_score = wmae(time_y, time_prediction, time_w)
        print(
            f"alpha_{feature_name}: meta-OOF delta={best[1]-current_score:+,.0f}, "
            f"folds={sum(delta < 0 for delta in best[3])}/5; "
            f"time delta={time_score-time_current:+,.0f}; shrinkage={best[2]:.1f}",
            flush=True,
        )
        prediction_output[f"alpha_{feature_name}"] = time_alpha
        prediction_output[f"prediction_alpha_{feature_name}"] = time_prediction

    result = pd.concat(summaries, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    prediction_path = args.output.with_name(f"{args.output.stem}_time_predictions.csv")
    prediction_output.to_csv(prediction_path, index=False)
    print(f"saved {args.output} and {prediction_path}")


if __name__ == "__main__":
    main()
