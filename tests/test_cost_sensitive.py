import numpy as np
import pytest

from src.cost_sensitive import boundary_cost_weights, boundary_regret_weights


def test_boundary_weights_increase_with_log_distance():
    target = np.asarray([100.0, 150.0, 225.0, 337.5])
    weights = boundary_cost_weights(target, np.ones(4), boundary=150.0, strength=1.0)
    assert weights[1] < weights[0]
    assert weights[0] == pytest.approx(weights[2])
    assert weights[2] < weights[3]
    assert weights.mean() == pytest.approx(1.0)


def test_zero_strength_preserves_relative_base_weights():
    base = np.asarray([1.0, 2.0, 4.0])
    result = boundary_cost_weights([50.0, 100.0, 200.0], base, 100.0, strength=0.0)
    np.testing.assert_allclose(result / result[0], base / base[0])


def test_regret_weights_focus_on_rows_hurt_by_wrong_specialist():
    target = np.asarray([40.0, 80.0])
    true_band = np.asarray([0, 1])
    experts = np.asarray([[42.0, 75.0], [45.0, 79.0]])
    weights, regret = boundary_regret_weights(
        target, np.ones(2), true_band, experts, boundary_index=1, strength=1.0
    )
    np.testing.assert_allclose(regret, [33.0, 34.0])
    assert weights[1] > weights[0]
