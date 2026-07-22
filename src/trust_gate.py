"""Feature construction and policies for the ordinal correction trust gate."""

import numpy as np
import pandas as pd

from src.feature_engineering import INCOME_ANCHORS
from src.ordinal_routing import DEFAULT_BAND_EDGES, income_band, temperature_scale


def _safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def build_trust_features(components, raw_frame, include_raw_numeric=False):
    """Build target-free diagnostics of whether the ordinal route is reliable."""
    components = components.reset_index(drop=True)
    raw_frame = raw_frame.reset_index(drop=True)
    probability_columns = [f"probability_band_{i}" for i in range(8)]
    probabilities = components[probability_columns].to_numpy(dtype=float)
    scaled = temperature_scale(probabilities, 0.5)
    class_index = np.arange(scaled.shape[1], dtype=float)
    expected_band = scaled @ class_index
    spread = np.sqrt(np.sum(scaled * (class_index[None, :] - expected_band[:, None]) ** 2, axis=1))
    ordered = np.sort(scaled, axis=1)
    hard_band = scaled.argmax(axis=1)
    confidence = scaled.max(axis=1)
    entropy = -np.sum(scaled * np.log(np.clip(scaled, 1e-12, 1.0)), axis=1)

    base = components["base"].to_numpy(dtype=float)
    routed = components["ordinal_routed_prediction"].to_numpy(dtype=float)
    correction = components["ordinal_correction"].to_numpy(dtype=float)
    base_band = income_band(base)

    existing_anchors = [column for column in INCOME_ANCHORS if column in raw_frame]
    anchors = raw_frame[existing_anchors].apply(pd.to_numeric, errors="coerce")
    anchor_count = anchors.notna().sum(axis=1).to_numpy(dtype=float)
    anchor_median = anchors.median(axis=1).to_numpy(dtype=float)
    anchor_mean = anchors.mean(axis=1).to_numpy(dtype=float)
    anchor_std = anchors.std(axis=1, ddof=0).to_numpy(dtype=float)
    anchor_fallback = np.where(np.isfinite(anchor_median), anchor_median, base)
    anchor_band = income_band(anchor_fallback)

    features = pd.DataFrame(
        {
            "trust_confidence": confidence,
            "trust_entropy": entropy,
            "trust_margin": ordered[:, -1] - ordered[:, -2],
            "trust_posterior_spread": spread,
            "trust_expected_band": expected_band,
            "trust_hard_band": hard_band,
            "trust_base_band": base_band,
            "trust_hard_base_distance": np.abs(hard_band - base_band),
            "trust_expected_base_distance": np.abs(expected_band - base_band),
            "trust_base": base,
            "trust_routed": routed,
            "trust_correction": correction,
            "trust_abs_correction": np.abs(correction),
            "trust_relative_correction": _safe_divide(correction, base),
            "trust_anchor_count": anchor_count,
            "trust_anchor_median": anchor_median,
            "trust_anchor_mean": anchor_mean,
            "trust_anchor_std": anchor_std,
            "trust_anchor_cv": _safe_divide(anchor_std, anchor_mean),
            "trust_hard_anchor_distance": np.abs(hard_band - anchor_band),
            "trust_base_anchor_distance": np.abs(base_band - anchor_band),
            "trust_base_minus_anchor": base - anchor_median,
            "trust_routed_minus_anchor": routed - anchor_median,
            "trust_tail_correction": components["tail_correction"].to_numpy(dtype=float),
            "trust_source_correction": components["source_correction"].to_numpy(dtype=float),
        }
    )
    for index, column in enumerate(probability_columns):
        features[f"trust_probability_{index}"] = probabilities[:, index]
        features[f"trust_scaled_probability_{index}"] = scaled[:, index]

    if include_raw_numeric:
        numeric = raw_frame.select_dtypes(include=[np.number]).copy()
        excluded = {"id", "target", "w"}
        numeric = numeric[[column for column in numeric.columns if column not in excluded]]
        numeric = numeric.add_prefix("raw_")
        features = pd.concat([features, numeric.reset_index(drop=True)], axis=1)
    return features.replace([np.inf, -np.inf], np.nan)


def apply_abstention_policy(nonordinal, ordinal_correction, harm_probability, policy):
    """Apply a fixed hard or soft trust policy to the ordinal correction."""
    nonordinal = np.asarray(nonordinal, dtype=float)
    correction = np.asarray(ordinal_correction, dtype=float)
    probability = np.clip(np.asarray(harm_probability, dtype=float), 0.0, 1.0)
    if policy["kind"] == "hard":
        factor = np.where(probability >= policy["threshold"], policy["retain"], 1.0)
    elif policy["kind"] == "soft":
        factor = np.power(1.0 - probability, policy["gamma"])
    else:
        raise ValueError("unknown trust policy")
    return nonordinal + 0.75 * factor * correction

