"""Competition metric (WMAE) and the weighted median used to approximate it."""
import numpy as np


def wmae(y_true, y_pred, weights=None):
    """Weighted Mean Absolute Error: sum(w*|y-y_hat|) / sum(w).

    With weights=None this reduces to the ordinary MAE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)
    if weights is None:
        return abs_err.mean()
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights * abs_err) / np.sum(weights))


def weighted_median(values, weights):
    """Weighted median: the value at which the cumulative weight crosses 50%.

    The WMAE-optimal point estimate for a group is the weighted median of its
    targets (weighted absolute error is minimized by the weighted median, the
    same way MAE is minimized by the plain median).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~np.isnan(values)
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cum_weights = np.cumsum(weights)
    total_weight = cum_weights[-1]
    if total_weight <= 0:
        return float(np.median(values))

    cutoff = total_weight / 2.0
    idx = np.searchsorted(cum_weights, cutoff)
    idx = min(idx, len(values) - 1)
    return float(values[idx])
