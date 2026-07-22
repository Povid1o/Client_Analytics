"""Training utilities for one band-conditioned regression expert."""

import numpy as np
import pandas as pd

from src.ordinal_routing import DEFAULT_BAND_EDGES


CONTEXT_PREFIX = "fe_requested_band_"


def leaky_clip(values, lower, upper, leak=0.0):
    """Project to an interval while retaining a fraction of outside distance."""
    if not 0.0 <= leak <= 1.0:
        raise ValueError("leak must be between zero and one")
    values = np.asarray(values, dtype=float)
    clipped = np.clip(values, lower, upper)
    return clipped + float(leak) * (values - clipped)


def add_band_context(matrix, band, edges=DEFAULT_BAND_EDGES):
    """Attach a numeric query describing the income band to every row."""
    edges = np.asarray(edges, dtype=float)
    if band < 0 or band >= len(edges) - 1:
        raise ValueError("band is outside the supplied edges")
    result = matrix.copy()
    lower = edges[band]
    upper = edges[band + 1]
    center = np.sqrt(lower * upper)
    result[f"{CONTEXT_PREFIX}id"] = np.int16(band)
    result[f"{CONTEXT_PREFIX}lower_log1p"] = np.log1p(lower)
    result[f"{CONTEXT_PREFIX}upper_log1p"] = np.log1p(upper)
    result[f"{CONTEXT_PREFIX}center_log1p"] = np.log1p(center)
    result[f"{CONTEXT_PREFIX}width_log1p"] = np.log1p(upper - lower)
    for candidate in range(len(edges) - 1):
        result[f"{CONTEXT_PREFIX}is_{candidate}"] = np.int8(candidate == band)
    if "fe_income_anchor_median" in result:
        anchor = pd.to_numeric(result["fe_income_anchor_median"], errors="coerce")
        result[f"{CONTEXT_PREFIX}anchor_log_distance"] = (
            np.sign(anchor - center)
            * np.log1p(np.abs(anchor - center))
        )
        result[f"{CONTEXT_PREFIX}anchor_log_ratio"] = (
            np.log1p(np.clip(anchor, 0, None)) - np.log1p(center)
        )
    return result


def expand_band_training_rows(
    matrix, labels, target, weights, *, radius=1, edges=DEFAULT_BAND_EDGES
):
    """Replicate rows for the requested bands whose training windows contain them.

    This reproduces the old specialist rule (true band plus immediate
    neighbours), while fitting all requested bands with one shared model.
    """
    labels = np.asarray(labels, dtype=int)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(matrix) != len(labels) or len(labels) != len(target) or len(target) != len(weights):
        raise ValueError("matrix, labels, target and weights must have equal length")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    n_bands = len(edges) - 1
    frames = []
    expanded_target = []
    expanded_weights = []
    source_rows = []
    for band in range(n_bands):
        selected = np.flatnonzero(np.abs(labels - band) <= radius)
        frames.append(add_band_context(matrix.iloc[selected], band, edges=edges))
        expanded_target.append(target[selected])
        expanded_weights.append(weights[selected])
        source_rows.append(selected)
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(expanded_target),
        np.concatenate(expanded_weights),
        np.concatenate(source_rows),
    )


def build_band_query_matrix(matrix, band, edges=DEFAULT_BAND_EDGES):
    """Build inference rows for one requested band."""
    return add_band_context(matrix.reset_index(drop=True), band, edges=edges)
