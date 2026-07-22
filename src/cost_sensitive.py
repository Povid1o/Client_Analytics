"""Cost-sensitive weights for cumulative income-boundary classifiers."""

import numpy as np


def boundary_cost_weights(target, base_weight, boundary, strength=1.0):
    """Upweight expensive mistakes far away from an ordinal boundary.

    Distance is measured on a log scale so the rule is symmetric for
    multiplicative income differences. It is capped to prevent a few extreme
    rows from dominating an entire binary classifier.
    """
    target = np.asarray(target, dtype=float)
    base_weight = np.asarray(base_weight, dtype=float)
    if boundary <= 0 or strength < 0:
        raise ValueError("boundary must be positive and strength non-negative")
    safe_target = np.clip(target, 1e-6, None)
    severity = np.abs(np.log(safe_target / float(boundary))) / np.log(1.5)
    severity = np.clip(severity, 0.0, 3.0)
    result = base_weight * (1.0 + float(strength) * severity)
    return result / np.mean(result)


def boundary_regret_weights(
    target, base_weight, true_band, expert_predictions, boundary_index, strength=1.0
):
    """Weight a boundary by OOF regret from routing to its wrong side."""
    target = np.asarray(target, dtype=float)
    base_weight = np.asarray(base_weight, dtype=float)
    true_band = np.asarray(true_band, dtype=int)
    experts = np.asarray(expert_predictions, dtype=float)
    if experts.ndim != 2 or len(experts) != len(target):
        raise ValueError("expert_predictions must be rows by bands")
    if not 1 <= boundary_index < experts.shape[1]:
        raise ValueError("boundary_index must separate two existing bands")
    rows = np.arange(len(target))
    correct = experts[rows, true_band]
    wrong_band = np.where(true_band >= boundary_index, boundary_index - 1, boundary_index)
    wrong = experts[rows, wrong_band]
    regret = np.maximum(0.0, np.abs(target - wrong) - np.abs(target - correct))
    positive = regret > 0
    if np.any(positive):
        scale = np.median(regret[positive])
    else:
        scale = 1.0
    severity = np.clip(regret / max(scale, 1e-6), 0.0, 4.0)
    result = base_weight * (1.0 + float(strength) * severity)
    return result / np.mean(result), regret
