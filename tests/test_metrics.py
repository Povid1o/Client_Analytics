import numpy as np
import pytest

from src.metrics import wmae, weighted_median


def test_wmae_unweighted_is_plain_mae():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 4]
    # abs errors: 0, 0, 1 -> mean = 1/3
    assert wmae(y_true, y_pred) == pytest.approx(1 / 3)


def test_wmae_weighted_manual():
    y_true = [10, 20]
    y_pred = [12, 18]
    weights = [1, 3]
    # abs errors: 2, 2 -> (1*2 + 3*2) / (1+3) = 8/4 = 2.0
    assert wmae(y_true, y_pred, weights) == pytest.approx(2.0)


def test_wmae_weighted_asymmetric_manual():
    y_true = [0, 0, 0]
    y_pred = [10, 0, 0]
    weights = [1, 1, 2]
    # abs errors: 10, 0, 0 -> (1*10 + 1*0 + 2*0) / 4 = 2.5
    assert wmae(y_true, y_pred, weights) == pytest.approx(2.5)


def test_weighted_median_equal_weights_matches_implementation():
    values = [1, 2, 3, 4]
    weights = [1, 1, 1, 1]
    # cumulative weights [1,2,3,4], total=4, cutoff=2 -> first value whose
    # cumulative weight reaches 2 is 2 (lower weighted median convention)
    assert weighted_median(values, weights) == pytest.approx(2.0)


def test_weighted_median_skewed_weight_pulls_result():
    values = [1, 2, 3]
    weights = [1, 1, 10]
    # cumulative weights [1,2,12], total=12, cutoff=6 -> value 3 dominates
    assert weighted_median(values, weights) == pytest.approx(3.0)


def test_weighted_median_ignores_nan_values():
    values = [np.nan, 5, 10]
    weights = [100, 1, 1]
    # the huge weight on NaN must be dropped, leaving values [5, 10] weights [1, 1]
    # cumulative [1, 2], total=2, cutoff=1 -> idx 0 -> 5
    assert weighted_median(values, weights) == pytest.approx(5.0)


def test_weighted_median_empty_returns_nan():
    assert np.isnan(weighted_median([], []))
