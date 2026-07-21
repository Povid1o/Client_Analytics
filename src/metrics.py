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


def bootstrap_wmae_ci(y_true, y_pred, weights=None, n_boot=2000, seed=42, ci=0.95):
    """Percentile-method bootstrap CI for WMAE, computed on the pooled OOF predictions.

    Resamples rows (with replacement) `n_boot` times and recomputes `wmae`
    each time, rather than relying on the mean/std of a handful of per-fold
    scores - with only `N_FOLDS` folds, that naive std is a std over a
    sample of size `N_FOLDS` and tends to overstate uncertainty (see
    `02_baseline_pipeline.ipynb`, section 4, for the naive-vs-bootstrap
    comparison on this dataset).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    rng = np.random.default_rng(seed)
    boot_scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_scores[b] = wmae(y_true[idx], y_pred[idx], weights[idx])

    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.percentile(boot_scores, [100 * alpha, 100 * (1 - alpha)])
    return {
        "mean": float(boot_scores.mean()),
        "std": float(boot_scores.std()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }
