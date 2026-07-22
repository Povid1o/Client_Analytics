"""Utilities for routing a regression prediction through ordered income bands."""

import numpy as np


DEFAULT_BAND_EDGES = np.asarray(
    [20_000.0, 50_000.0, 75_000.0, 100_000.0, 150_000.0,
     250_000.0, 400_000.0, 700_000.0, 1_500_000.0]
)


def income_band(target, edges=DEFAULT_BAND_EDGES):
    """Return zero-based ordered band labels for numeric targets."""
    target = np.asarray(target, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be a strictly increasing one-dimensional array")
    return np.clip(np.searchsorted(edges[1:-1], target, side="right"), 0, len(edges) - 2)


def normalize_probabilities(probabilities):
    """Clip numeric noise and ensure every row is a probability distribution."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    result = np.clip(probabilities, 0.0, None)
    denominator = result.sum(axis=1, keepdims=True)
    empty = denominator[:, 0] <= 0
    if np.any(empty):
        result[empty] = 1.0 / result.shape[1]
        denominator = result.sum(axis=1, keepdims=True)
    return result / denominator


def temperature_scale(probabilities, temperature):
    """Sharpen (T<1) or flatten (T>1) multiclass probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = normalize_probabilities(probabilities)
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_probabilities(np.exp(logits))


def adaptive_temperature_scale(probabilities, default_temperature, overrides):
    """Apply class-specific temperatures based on the unscaled top class."""
    probabilities = normalize_probabilities(probabilities)
    result = temperature_scale(probabilities, default_temperature)
    top_class = probabilities.argmax(axis=1)
    for band, temperature in overrides.items():
        selected = top_class == int(band)
        if np.any(selected):
            result[selected] = temperature_scale(probabilities[selected], temperature)
    return result


def cumulative_to_class_probabilities(probability_at_least):
    """Convert P(class >= k), k=1..K-1, into a monotone K-class posterior."""
    probability_at_least = np.asarray(probability_at_least, dtype=float)
    if probability_at_least.ndim != 2:
        raise ValueError("probability_at_least must be a two-dimensional array")
    cumulative = np.clip(probability_at_least, 0.0, 1.0)
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    result = np.column_stack(
        [1.0 - cumulative[:, 0], cumulative[:, :-1] - cumulative[:, 1:], cumulative[:, -1]]
    )
    return normalize_probabilities(result)


def project_to_bands(base_prediction, edges=DEFAULT_BAND_EDGES):
    """Project each base prediction into every possible target band."""
    base_prediction = np.asarray(base_prediction, dtype=float)
    edges = np.asarray(edges, dtype=float)
    lower = edges[:-1]
    upper = edges[1:]
    return np.minimum(np.maximum(base_prediction[:, None], lower[None, :]), upper[None, :])


def posterior_median_band(probabilities):
    """Return the first ordered class whose cumulative probability reaches 0.5."""
    probabilities = normalize_probabilities(probabilities)
    return np.argmax(np.cumsum(probabilities, axis=1) >= 0.5, axis=1)


def route_predictions(base_prediction, probabilities, candidates, mode="soft"):
    """Route candidate predictions using soft or posterior-median assignment."""
    base_prediction = np.asarray(base_prediction, dtype=float)
    probabilities = normalize_probabilities(probabilities)
    candidates = np.asarray(candidates, dtype=float)
    if candidates.shape != probabilities.shape:
        raise ValueError("candidates and probabilities must have identical shapes")
    if len(base_prediction) != len(probabilities):
        raise ValueError("base_prediction length does not match probabilities")
    if mode == "soft":
        return np.sum(probabilities * candidates, axis=1)
    if mode == "median":
        selected = posterior_median_band(probabilities)
        return candidates[np.arange(len(selected)), selected]
    raise ValueError("mode must be 'soft' or 'median'")
