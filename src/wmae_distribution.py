"""Distribution utilities for the target-dependent WMAE weight function."""

import numpy as np
from scipy.special import ndtr


def recover_weight_rule(target, weights):
    """Recover w(y)=min(abs(y/center-1), cap) and verify it exactly."""
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    cap = float(np.nanmax(weights))
    unsaturated_right = (
        np.isfinite(target) & np.isfinite(weights) & (weights < cap - 1e-10)
        & (target > np.nanmedian(target))
    )
    if not unsaturated_right.any():
        raise ValueError("cannot recover the WMAE center")
    center = float(np.nanmedian(target[unsaturated_right] / (1.0 + weights[unsaturated_right])))
    reconstructed = np.minimum(np.abs(target / center - 1.0), cap)
    max_error = float(np.nanmax(np.abs(reconstructed - weights)))
    if max_error > 1e-8:
        raise ValueError(f"weight rule does not fit observed weights: max error {max_error:g}")
    return {
        "center": center,
        "cap": cap,
        "saturation": center * (1.0 + cap),
        "max_error": max_error,
    }


def _probability(value, mu, sigma):
    value = np.maximum(np.asarray(value, dtype=float), 1e-12)
    return ndtr((np.log(value) - mu) / sigma)


def _first_moment(value, mu, sigma):
    value = np.maximum(np.asarray(value, dtype=float), 1e-12)
    return np.exp(mu + 0.5 * sigma**2) * ndtr(
        (np.log(value) - mu - sigma**2) / sigma
    )


def _interval_mass(lower, upper, mu, sigma, rule, regime):
    probability = _probability(upper, mu, sigma) - _probability(lower, mu, sigma)
    if regime == "high":
        return rule["cap"] * probability
    moment = _first_moment(upper, mu, sigma) - _first_moment(lower, mu, sigma)
    if regime == "low":
        return probability - moment / rule["center"]
    if regime == "middle":
        return moment / rule["center"] - probability
    raise ValueError(f"unknown weight regime: {regime}")


def _mass_to(value, mu, sigma, rule):
    value = np.maximum(np.asarray(value, dtype=float), 0.0)
    result = np.zeros_like(mu, dtype=float)
    low_upper = np.minimum(value, rule["center"])
    low = low_upper > 0
    result[low] += _interval_mass(
        0.0, low_upper[low], mu[low], sigma[low], rule, "low"
    )
    middle_upper = np.minimum(value, rule["saturation"])
    middle = middle_upper > rule["center"]
    result[middle] += _interval_mass(
        rule["center"], middle_upper[middle], mu[middle], sigma[middle],
        rule, "middle",
    )
    high = value > rule["saturation"]
    result[high] += _interval_mass(
        rule["saturation"], value[high], mu[high], sigma[high], rule, "high"
    )
    return result


def exact_weighted_lognormal_median(mu, sigma, rule):
    """Return the median after weighting a LogNormal(mu, sigma) by w(y)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.clip(np.asarray(sigma, dtype=float), 0.08, 2.0)
    total = (
        _interval_mass(0.0, rule["center"], mu, sigma, rule, "low")
        + _interval_mass(
            rule["center"], rule["saturation"], mu, sigma, rule, "middle"
        )
        + rule["cap"] * (1.0 - _probability(rule["saturation"], mu, sigma))
    )
    target_mass = 0.5 * total
    lower = np.zeros(len(mu), dtype=float)
    upper = np.exp(np.minimum(mu + 10.0 * sigma, 700.0))
    for _ in range(52):
        middle = 0.5 * (lower + upper)
        move_right = _mass_to(middle, mu, sigma, rule) < target_mass
        lower = np.where(move_right, middle, lower)
        upper = np.where(move_right, upper, middle)
    return 0.5 * (lower + upper)


def log_quantiles_to_weighted_lognormal(log_q20, log_q50, log_q80, rule):
    """Fit a row-wise lognormal from q20/q50/q80 and return WMAE optimum."""
    z80 = 0.8416212335729143
    mu = np.asarray(log_q50, dtype=float)
    sigma = (np.asarray(log_q80) - np.asarray(log_q20)) / (2.0 * z80)
    return exact_weighted_lognormal_median(mu, sigma, rule)
