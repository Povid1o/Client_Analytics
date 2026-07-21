import numpy as np
import pytest

from src.metrics import wmae, weighted_median, bootstrap_wmae_ci


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


def test_bootstrap_wmae_ci_is_deterministic_and_well_ordered():
    rng = np.random.default_rng(0)
    y_true = rng.normal(100, 10, size=500)
    y_pred = y_true + rng.normal(0, 5, size=500)
    weights = rng.uniform(0.1, 2.0, size=500)

    result_a = bootstrap_wmae_ci(y_true, y_pred, weights, n_boot=300, seed=42)
    result_b = bootstrap_wmae_ci(y_true, y_pred, weights, n_boot=300, seed=42)

    assert result_a == result_b  # same seed -> reproducible
    assert result_a["ci_low"] <= result_a["mean"] <= result_a["ci_high"]
    assert result_a["std"] > 0


def test_bootstrap_wmae_ci_coverage_sanity():
    """Simple coverage sanity check (not a full coverage study): repeatedly
    draw small samples from a distribution with a known population WMAE, and
    check that the 95% bootstrap CI contains the true value roughly 95% of
    the time (loose tolerance, since the check itself uses a small number of
    replicates and is meant to catch gross miscalibration, not to certify
    exact coverage).
    """
    sigma = 5.0
    true_wmae = sigma * np.sqrt(2 / np.pi)  # E|error| for error ~ N(0, sigma), weights independent

    rng = np.random.default_rng(123)
    n_replicates = 150
    n_samples = 300
    n_boot = 200

    covered = 0
    for i in range(n_replicates):
        errors = rng.normal(0, sigma, size=n_samples)
        weights = rng.uniform(0.5, 1.5, size=n_samples)
        y_true = rng.normal(0, 1, size=n_samples)
        y_pred = y_true - errors

        result = bootstrap_wmae_ci(y_true, y_pred, weights, n_boot=n_boot, seed=i)
        if result["ci_low"] <= true_wmae <= result["ci_high"]:
            covered += 1

    coverage_rate = covered / n_replicates
    assert coverage_rate >= 0.80, f"coverage rate {coverage_rate:.2f} is too far below the nominal 95%"
