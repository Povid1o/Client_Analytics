"""Build tabular inputs for the income-model error audit workbook."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CSV_READ_KWARGS, DATE_COL, ID_COL, PARTIAL_OUTPUTS_DIR, TRAIN_PATH
from src.feature_engineering import INCOME_ANCHORS
from src.metrics import weighted_median, wmae
from src.ordinal_routing import DEFAULT_BAND_EDGES, income_band, route_predictions, temperature_scale
from src.preprocessing import preprocess


AUDIT_DIR = PARTIAL_OUTPUTS_DIR / "error_audit_20260722"
BAND_NAMES = [
    "20–50k", "50–75k", "75–100k", "100–150k",
    "150–250k", "250–400k", "400–700k", "700k+",
]


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(values * weights) / np.sum(weights))


def load_validation(kind, raw_by_id):
    if kind == "random":
        ordinal_path = PARTIAL_OUTPUTS_DIR / "ordinal_cumulative_results_predictions.csv"
        tail_path = PARTIAL_OUTPUTS_DIR / "tail_mixture_results_predictions.csv"
        source_path = PARTIAL_OUTPUTS_DIR / "source_experts_results_predictions.csv"
    else:
        ordinal_path = PARTIAL_OUTPUTS_DIR / "ordinal_cumulative_time_results_predictions.csv"
        tail_path = PARTIAL_OUTPUTS_DIR / "tail_mixture_outer_predictions.csv"
        source_path = PARTIAL_OUTPUTS_DIR / "source_experts_time_results_predictions.csv"
    ordinal = pd.read_csv(ordinal_path).set_index(ID_COL)
    tail = pd.read_csv(tail_path).set_index(ID_COL).loc[ordinal.index]
    source = pd.read_csv(source_path).set_index(ID_COL).loc[ordinal.index]
    raw = raw_by_id.loc[ordinal.index].copy()

    y = ordinal["target"].to_numpy(dtype=float)
    weights = ordinal["w"].to_numpy(dtype=float)
    base = ordinal["base"].to_numpy(dtype=float)
    probabilities = ordinal[[f"probability_band_{i}" for i in range(8)]].to_numpy(dtype=float)
    experts = ordinal[[f"expert_band_{i}" for i in range(8)]].to_numpy(dtype=float)
    scaled = temperature_scale(probabilities, 0.5)
    routed = route_predictions(base, scaled, experts, mode="soft")
    confidence = scaled.max(axis=1)
    ordinal_correction = confidence * (routed - base)
    ordinal_prediction = base + ordinal_correction
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
    final = base + 0.25 * tail_correction + 0.25 * source_correction + 0.75 * ordinal_correction
    final = np.clip(final, DEFAULT_BAND_EDGES[0], DEFAULT_BAND_EDGES[-1])

    result = pd.DataFrame(
        {
            "split": kind,
            ID_COL: ordinal.index.to_numpy(),
            "target": y,
            "w": weights,
            "base_prediction": base,
            "ordinal_prediction": ordinal_prediction,
            "final_prediction": final,
            "ordinal_routed_prediction": routed,
            "ordinal_correction": ordinal_correction,
            "tail_correction": tail_correction,
            "source_correction": source_correction,
            "confidence": confidence,
            "entropy": -np.sum(scaled * np.log(np.clip(scaled, 1e-12, 1.0)), axis=1),
            "posterior_spread": np.sqrt(
                np.sum(
                    scaled
                    * (
                        np.arange(8)[None, :]
                        - np.sum(scaled * np.arange(8)[None, :], axis=1)[:, None]
                    ) ** 2,
                    axis=1,
                )
            ),
        }
    )
    for band in range(8):
        result[f"p_band_{band}"] = scaled[:, band]
    result["true_band"] = income_band(y)
    result["predicted_band"] = scaled.argmax(axis=1)
    result["true_band_name"] = [BAND_NAMES[index] for index in result["true_band"]]
    result["predicted_band_name"] = [BAND_NAMES[index] for index in result["predicted_band"]]
    result["band_distance"] = np.abs(result["true_band"] - result["predicted_band"])
    result["signed_error"] = result["final_prediction"] - result["target"]
    result["abs_error"] = result["signed_error"].abs()
    result["weighted_abs_loss"] = result["w"] * result["abs_error"]
    result["base_abs_error"] = np.abs(result["base_prediction"] - result["target"])
    result["ordinal_abs_error"] = np.abs(result["ordinal_prediction"] - result["target"])
    result["improvement_vs_base"] = result["base_abs_error"] - result["abs_error"]

    raw = raw.reset_index()
    for column in [DATE_COL, "gender", "adminarea", "age"] + INCOME_ANCHORS:
        if column in raw:
            result[column] = raw[column].to_numpy()
    anchors = raw[[column for column in INCOME_ANCHORS if column in raw]].apply(
        pd.to_numeric, errors="coerce"
    )
    result["anchor_count"] = anchors.notna().sum(axis=1).to_numpy()
    result["anchor_median"] = anchors.median(axis=1).to_numpy()
    result["anchor_std"] = anchors.std(axis=1, ddof=0).to_numpy()
    result["salary_present"] = raw["salary_6to12m_avg"].notna().to_numpy()
    result["source_quality"] = pd.cut(
        result["anchor_count"], [-1, 1, 2, 8], labels=["0–1 anchors", "2 anchors", "3+ anchors"]
    ).astype(str)
    result["route_error_bucket"] = pd.cut(
        result["band_distance"], [-1, 0, 1, 2, 99],
        labels=["correct band", "off by 1", "off by 2", "off by 3+"],
    ).astype(str)
    result["diagnostic_flag"] = np.select(
        [
            result["band_distance"] >= 2,
            result["anchor_count"] <= 1,
            result["confidence"] < 0.5,
            result["signed_error"] < -100_000,
            result["signed_error"] > 100_000,
        ],
        [
            "far route error", "sparse income sources", "low-confidence boundary",
            "large underprediction", "large overprediction",
        ],
        default="within ordinary regime",
    )
    return result


def model_metrics(data, prediction_column):
    y = data["target"].to_numpy(dtype=float)
    p = data[prediction_column].to_numpy(dtype=float)
    w = data["w"].to_numpy(dtype=float)
    residual = p - y
    weighted_target_mean = weighted_mean(y, w)
    sse = np.sum(w * (y - p) ** 2)
    sst = np.sum(w * (y - weighted_target_mean) ** 2)
    x_mean = weighted_mean(p, w)
    covariance = np.sum(w * (p - x_mean) * (y - weighted_target_mean)) / np.sum(w)
    variance = np.sum(w * (p - x_mean) ** 2) / np.sum(w)
    slope = covariance / variance if variance > 0 else np.nan
    intercept = weighted_target_mean - slope * x_mean
    return {
        "WMAE": wmae(y, p, w),
        "MAE": float(np.mean(np.abs(residual))),
        "weighted_RMSE": float(np.sqrt(np.sum(w * residual ** 2) / np.sum(w))),
        "weighted_mean_bias_pred_minus_target": weighted_mean(residual, w),
        "weighted_median_bias": float(weighted_median(residual, w)),
        "weighted_underprediction_rate": weighted_mean(p < y, w),
        "weighted_MAPE": weighted_mean(np.abs(residual) / y, w),
        "weighted_R2": float(1 - sse / sst),
        "calibration_slope_y_on_prediction": float(slope),
        "calibration_intercept": float(intercept),
    }


def overall_metrics(data):
    rows = []
    for split, group in data.groupby("split"):
        for model, column in (
            ("Base CatBoost", "base_prediction"),
            ("Ordinal only", "ordinal_prediction"),
            ("Final minimax ensemble", "final_prediction"),
        ):
            rows.append({"split": split, "model": model, **model_metrics(group, column)})
    return pd.DataFrame(rows)


def group_metrics(data, grouping, label):
    rows = []
    for (split, key), group in data.groupby(["split", grouping], observed=True):
        y = group["target"].to_numpy(dtype=float)
        final = group["final_prediction"].to_numpy(dtype=float)
        base = group["base_prediction"].to_numpy(dtype=float)
        w = group["w"].to_numpy(dtype=float)
        split_weight = data.loc[data["split"] == split, "w"].sum()
        residual = final - y
        rows.append(
            {
                "split": split,
                "group_type": label,
                "group": str(key),
                "rows": len(group),
                "weight_share": w.sum() / split_weight,
                "target_weighted_mean": weighted_mean(y, w),
                "prediction_weighted_mean": weighted_mean(final, w),
                "weighted_mean_bias_pred_minus_target": weighted_mean(residual, w),
                "weighted_median_bias": weighted_median(residual, w),
                "local_WMAE": wmae(y, final, w),
                "base_local_WMAE": wmae(y, base, w),
                "delta_vs_base": wmae(y, final, w) - wmae(y, base, w),
                "global_WMAE_contribution": np.sum(w * np.abs(residual)) / split_weight,
                "weighted_underprediction_rate": weighted_mean(final < y, w),
                "weighted_error_over_50k_rate": weighted_mean(np.abs(residual) > 50_000, w),
                "weighted_error_over_100k_rate": weighted_mean(np.abs(residual) > 100_000, w),
                "mean_confidence": weighted_mean(group["confidence"], w),
                "band_accuracy": weighted_mean(group["band_distance"] == 0, w),
                "within_one_band": weighted_mean(group["band_distance"] <= 1, w),
                "mean_anchor_count": weighted_mean(group["anchor_count"], w),
            }
        )
    return pd.DataFrame(rows)


def confusion(data):
    rows = []
    for split, group in data.groupby("split"):
        for true_band in range(8):
            selected = group[group["true_band"] == true_band]
            denominator = selected["w"].sum()
            for predicted_band in range(8):
                cell = selected[selected["predicted_band"] == predicted_band]
                rows.append(
                    {
                        "split": split,
                        "true_band": BAND_NAMES[true_band],
                        "predicted_band": BAND_NAMES[predicted_band],
                        "weighted_row_share": cell["w"].sum() / denominator if denominator else 0.0,
                        "rows": len(cell),
                    }
                )
    return pd.DataFrame(rows)


def representative_sample(data, seed):
    selected = []
    rng = np.random.default_rng(seed)
    for (_, _), group in data.groupby(["split", "true_band"], observed=True):
        if len(group) <= 200:
            selected.append(group)
            continue
        ranks = group["abs_error"].rank(method="first", pct=True)
        quintile = np.minimum((ranks * 5).astype(int), 4)
        pieces = []
        for value in range(5):
            candidates = group.loc[quintile == value]
            take = min(40, len(candidates))
            pieces.append(candidates.iloc[rng.choice(len(candidates), size=take, replace=False)])
        sample = pd.concat(pieces)
        if len(sample) < 200:
            remaining = group.drop(sample.index)
            extra = remaining.iloc[rng.choice(len(remaining), size=200 - len(sample), replace=False)]
            sample = pd.concat([sample, extra])
        selected.append(sample)
    result = pd.concat(selected).sort_values(["split", "true_band", "abs_error"])
    result["error_quintile_within_band"] = (
        result.groupby(["split", "true_band"])["abs_error"]
        .rank(method="average", pct=True)
        .mul(5).clip(upper=4.999).astype(int).add(1)
    )
    return result


def to_json_records(frame, path):
    cleaned = frame.copy()
    for column in cleaned.columns:
        if pd.api.types.is_datetime64_any_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].dt.strftime("%Y-%m-%d")
    cleaned = cleaned.replace({np.nan: None, np.inf: None, -np.inf: None})
    path.write_text(json.dumps(cleaned.to_dict("records"), ensure_ascii=False), encoding="utf-8")


def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    raw, _ = preprocess(
        pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS, low_memory=False), is_train=True
    )
    raw_by_id = raw.set_index(ID_COL)
    data = pd.concat(
        [load_validation("random", raw_by_id), load_validation("temporal", raw_by_id)],
        ignore_index=True,
    )
    metrics = overall_metrics(data)
    groups = pd.concat(
        [
            group_metrics(data, "true_band_name", "true income band"),
            group_metrics(data, "predicted_band_name", "predicted income band"),
            group_metrics(data, "route_error_bucket", "route error distance"),
            group_metrics(data, "source_quality", "income source quality"),
            group_metrics(data, "salary_present", "salary source present"),
            group_metrics(data, "diagnostic_flag", "diagnostic flag"),
        ],
        ignore_index=True,
    )
    samples = representative_sample(data, seed=42)
    worst = (
        data.sort_values(["split", "true_band", "weighted_abs_loss"], ascending=[True, True, False])
        .groupby(["split", "true_band"], observed=True)
        .head(50)
    )
    detail_columns = [
        "split", ID_COL, DATE_COL, "true_band_name", "predicted_band_name",
        "band_distance", "target", "base_prediction", "ordinal_prediction",
        "final_prediction", "signed_error", "abs_error", "w", "weighted_abs_loss",
        "improvement_vs_base", "confidence", "entropy", "posterior_spread",
        "ordinal_correction", "tail_correction", "source_correction",
        "anchor_count", "anchor_median", "anchor_std", "source_quality",
        "salary_present", "gender", "age", "adminarea", "diagnostic_flag",
        "salary_6to12m_avg", "incomeValue", "dp_ils_avg_salary_1y",
        "dp_payoutincomedata_payout_avg_3_month",
        "dp_payoutincomedata_payout_avg_6_month",
        "dp_payoutincomedata_payout_avg_prev_year",
    ] + [f"p_band_{band}" for band in range(8)]
    detail_columns = [column for column in detail_columns if column in samples]
    to_json_records(metrics, AUDIT_DIR / "overall_metrics.json")
    to_json_records(groups, AUDIT_DIR / "group_metrics.json")
    to_json_records(confusion(data), AUDIT_DIR / "confusion.json")
    to_json_records(samples[detail_columns], AUDIT_DIR / "samples.json")
    to_json_records(worst[detail_columns], AUDIT_DIR / "worst_cases.json")
    print(
        f"saved audit data to {AUDIT_DIR}; samples={len(samples):,}, "
        f"worst={len(worst):,}"
    )


if __name__ == "__main__":
    main()
