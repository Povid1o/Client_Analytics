import numpy as np

from src.wmae_distribution import (
    exact_weighted_lognormal_median,
    recover_weight_rule,
)


def test_recover_weight_rule_reconstructs_weights():
    target = np.linspace(20_000.0, 500_000.0, 500)
    center = 84_000.0
    cap = 2.5
    weights = np.minimum(np.abs(target / center - 1.0), cap)

    rule = recover_weight_rule(target, weights)

    assert np.isclose(rule["center"], center)
    assert np.isclose(rule["cap"], cap)
    assert rule["max_error"] < 1e-10


def test_weighted_lognormal_median_is_finite_and_positive():
    rule = {"center": 84_000.0, "cap": 2.5, "saturation": 294_000.0}
    result = exact_weighted_lognormal_median(
        np.log([50_000.0, 200_000.0]), np.asarray([0.3, 0.8]), rule
    )

    assert np.isfinite(result).all()
    assert (result > 0).all()
