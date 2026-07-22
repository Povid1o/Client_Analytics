"""OOF weighted-median region encoding.

`adminarea` is a legitimate in-dataset feature, but turning it into a
per-region target statistic leaks the target unless it's computed
out-of-fold. Section 0.6 showed empirically that a *weighted median* (using
the same competition weights `w`) beats mean/weighted-mean as the
region-level summary for approximating the WMAE-optimal prediction, so that
is the only statistic used here.
"""
import numpy as np
import pandas as pd

from src.config import MIN_GROUP_COUNT, REGION_COL
from src.metrics import weighted_median
from src.validation import make_oof_array


def compute_region_stats(regions, target, weights, min_count=MIN_GROUP_COUNT):
    """Weighted-median target per region (only for regions with count >= min_count).

    Returns (stats_dict, global_fallback) where global_fallback is the
    weighted median over the whole input, used for regions absent or too
    small.
    """
    frame = pd.DataFrame({"region": regions, "target": target, "w": weights})
    global_fallback = weighted_median(frame["target"].to_numpy(), frame["w"].to_numpy())

    stats = {}
    for region, group in frame.groupby("region"):
        if len(group) >= min_count:
            stats[region] = weighted_median(group["target"].to_numpy(), group["w"].to_numpy())
    return stats, global_fallback


def apply_region_stats(regions, stats, global_fallback):
    """Map each region to its stat, falling back to the global stat."""
    return np.array([stats.get(r, global_fallback) for r in regions], dtype=float)


def fit_region_encoding(train_fold_df, target, weights, region_col=REGION_COL, min_count=MIN_GROUP_COUNT):
    """Fit region stats on one fold's training rows only."""
    return compute_region_stats(train_fold_df[region_col].to_numpy(), np.asarray(target), np.asarray(weights), min_count)


def oof_region_encoding(df, target, weights, folds, region_col=REGION_COL, min_count=MIN_GROUP_COUNT):
    """Out-of-fold region encoding for the training set.

    For each fold, stats are fit on the other folds only and applied to the
    held-out fold, so no row's own target/weight ever contributes to its own
    encoding.
    """
    target = np.asarray(target)
    weights = np.asarray(weights)
    oof = make_oof_array(len(df))

    for train_idx, val_idx in folds:
        stats, fallback = fit_region_encoding(
            df.iloc[train_idx], target[train_idx], weights[train_idx], region_col, min_count
        )
        oof[val_idx] = apply_region_stats(df[region_col].to_numpy()[val_idx], stats, fallback)

    return oof


def fit_full_region_encoding(train_df, target, weights, region_col=REGION_COL, min_count=MIN_GROUP_COUNT):
    """Fit region stats on the entire train set, for encoding the test set."""
    return fit_region_encoding(train_df, target, weights, region_col, min_count)


def apply_region_encoding(df, stats, global_fallback, region_col=REGION_COL):
    """Apply previously-fit region stats to any dataframe (e.g. test)."""
    return apply_region_stats(df[region_col].to_numpy(), stats, global_fallback)


def compute_smoothed_region_stats(regions, target, weights, smoothing=60.0):
    """Bayesian-shrunk weighted-median target statistic per region.

    The shrinkage formula is ``(n * region_median + smoothing * global) /
    (n + smoothing)``. Unlike the old hard ``min_count`` cutoff, this keeps
    small regions while continuously pulling their noisy estimates towards
    the global weighted median.
    """
    frame = pd.DataFrame({"region": regions, "target": target, "w": weights})
    global_stat = weighted_median(frame["target"].to_numpy(), frame["w"].to_numpy())
    stats = {}
    for region, group in frame.groupby("region", dropna=False):
        region_stat = weighted_median(group["target"].to_numpy(), group["w"].to_numpy())
        count = len(group)
        stats[region] = (count * region_stat + smoothing * global_stat) / (count + smoothing)
    return stats, global_stat


def apply_smoothed_region_stats(regions, stats, global_fallback):
    """Apply smoothed stats, including a stable key for missing regions."""
    return np.asarray([stats.get(region, global_fallback) for region in regions], dtype=float)


def crossfit_smoothed_region_encoding(
    df,
    target,
    weights,
    folds,
    region_col=REGION_COL,
    smoothing=60.0,
):
    """Create target-safe OOF smoothed region statistics."""
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    regions = df[region_col].to_numpy()
    result = make_oof_array(len(df))
    for train_idx, val_idx in folds:
        stats, fallback = compute_smoothed_region_stats(
            regions[train_idx], target[train_idx], weights[train_idx], smoothing=smoothing
        )
        result[val_idx] = apply_smoothed_region_stats(regions[val_idx], stats, fallback)
    return result
