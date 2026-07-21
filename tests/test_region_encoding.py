import numpy as np
import pandas as pd
import pytest

from src.region_encoding import (
    apply_region_stats,
    compute_region_stats,
    oof_region_encoding,
)
from src.validation import get_folds


def _make_synthetic_df(n_per_region=50, n_regions=4, seed=0):
    rng = np.random.default_rng(seed)
    regions = np.repeat([f"region_{i}" for i in range(n_regions)], n_per_region)
    n = len(regions)
    target = rng.normal(loc=100, scale=10, size=n) + (
        np.repeat(np.arange(n_regions) * 20, n_per_region)
    )
    weights = rng.uniform(0.1, 2.0, size=n)
    df = pd.DataFrame({"adminarea": regions})
    return df, target, weights


def test_min_count_fallback_used_for_small_regions():
    df, target, weights = _make_synthetic_df(n_per_region=50, n_regions=4)
    # shrink one region below min_count
    small_region_mask = df["adminarea"] == "region_0"
    keep_idx = np.where(small_region_mask)[0][:5]
    drop_idx = np.where(small_region_mask)[0][5:]
    mask = np.ones(len(df), dtype=bool)
    mask[drop_idx] = False

    df_small = df[mask].reset_index(drop=True)
    target_small = target[mask]
    weights_small = weights[mask]

    stats, fallback = compute_region_stats(
        df_small["adminarea"].to_numpy(), target_small, weights_small, min_count=20
    )
    assert "region_0" not in stats
    assert "region_1" in stats


def test_oof_encoding_does_not_use_row_own_target():
    """Leakage test: perturbing one row's target must not change its own OOF encoding."""
    df, target, weights = _make_synthetic_df(n_per_region=50, n_regions=4, seed=1)
    folds = get_folds(target, n_folds=5, seed=42)

    oof_original = oof_region_encoding(df, target, weights, folds, min_count=20)

    target_perturbed = target.copy()
    perturb_row = 0
    target_perturbed[perturb_row] = target_perturbed[perturb_row] + 1_000_000

    oof_perturbed = oof_region_encoding(df, target_perturbed, weights, folds, min_count=20)

    assert oof_original[perturb_row] == pytest.approx(oof_perturbed[perturb_row])

    # every other row in the same fold as perturb_row should also be unaffected,
    # since the perturbed row is held out (not part of any training fold used
    # to compute stats for its own fold) - but rows in OTHER folds whose
    # training split includes perturb_row COULD change.
    fold_of_perturbed = None
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        if perturb_row in val_idx:
            fold_of_perturbed = fold_i
            break
    train_idx, val_idx = folds[fold_of_perturbed]
    # other rows held out in the SAME fold as perturb_row never train on it either
    same_fold_other_rows = [r for r in val_idx if r != perturb_row]
    for r in same_fold_other_rows[:10]:
        assert oof_original[r] == pytest.approx(oof_perturbed[r])


def test_apply_region_stats_uses_fallback_for_unseen_region():
    stats = {"region_a": 100.0}
    fallback = 50.0
    result = apply_region_stats(["region_a", "region_unseen"], stats, fallback)
    assert result[0] == pytest.approx(100.0)
    assert result[1] == pytest.approx(50.0)
