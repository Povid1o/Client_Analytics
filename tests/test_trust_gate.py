import numpy as np

from src.trust_gate import apply_abstention_policy


def test_hard_abstention_retains_only_high_risk_rows():
    prediction = apply_abstention_policy(
        np.asarray([100.0, 100.0]),
        np.asarray([20.0, 20.0]),
        np.asarray([0.2, 0.8]),
        {"kind": "hard", "threshold": 0.5, "retain": 0.0},
    )
    np.testing.assert_allclose(prediction, [115.0, 100.0])


def test_soft_abstention_is_monotone_in_harm_probability():
    prediction = apply_abstention_policy(
        np.full(3, 100.0),
        np.full(3, 20.0),
        np.asarray([0.0, 0.5, 1.0]),
        {"kind": "soft", "gamma": 1.0},
    )
    np.testing.assert_allclose(prediction, [115.0, 107.5, 100.0])

