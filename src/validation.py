"""Shared Stratified K-Fold splitter so the model CV and the region encoding
use identical folds (required for the region encoding's leakage-safety
guarantee).

Stratified by target decile bins rather than plain shuffling: a plain
`KFold` shuffle can, by chance, put an uneven share of the (rare, high-`w`)
top-decile clients in one fold, which is exactly the tail WMAE is most
sensitive to. Stratifying on `pd.qcut(target, 10)` keeps the target
distribution - and therefore the tail - balanced across folds, which is why
fold-to-fold variance drops noticeably after this change. The validation
protocol is documented in `notebooks/EDA.ipynb`.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import N_FOLDS, RANDOM_SEED


def get_folds(target, n_folds=N_FOLDS, seed=RANDOM_SEED, n_bins=10):
    """Return (train_idx, val_idx) folds stratified by target decile bins.

    `target` must be the array of continuous target values for every row
    (not just its length) since the stratification bins are derived from it.
    """
    target = np.asarray(target, dtype=float)
    bins = pd.qcut(target, n_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(target)), bins))


def make_oof_array(n_rows, fill_value=np.nan):
    """Allocate an out-of-fold prediction/encoding array."""
    return np.full(n_rows, fill_value, dtype=float)
