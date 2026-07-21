"""Shared K-Fold splitter so the model CV and the region encoding use
identical folds (required for the region encoding's leakage-safety guarantee).
"""
import numpy as np
from sklearn.model_selection import KFold

from src.config import N_FOLDS, RANDOM_SEED


def get_folds(n_rows, n_folds=N_FOLDS, seed=RANDOM_SEED):
    """Return a list of (train_idx, val_idx) arrays for `n_rows` samples."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_rows)))


def make_oof_array(n_rows, fill_value=np.nan):
    """Allocate an out-of-fold prediction/encoding array."""
    return np.full(n_rows, fill_value, dtype=float)
