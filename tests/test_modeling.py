import numpy as np
import pytest

from src.modeling import inverse_power_target, power_transform_target


@pytest.mark.parametrize("power", [0.25, 0.5, 1.0])
def test_power_target_round_trip(power):
    target = np.array([0.0, 20_000.0, 84_017.0, 1_500_000.0])
    restored = inverse_power_target(power_transform_target(target, power), power)
    np.testing.assert_allclose(restored, target, rtol=1e-12, atol=1e-8)


def test_power_target_rejects_invalid_values():
    with pytest.raises(ValueError):
        power_transform_target([-1.0, 2.0], 0.25)
    with pytest.raises(ValueError):
        power_transform_target([1.0, 2.0], 0.0)
