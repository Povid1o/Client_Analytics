import numpy as np
import pandas as pd
import pytest

from src.shared_band_expert import (
    add_band_context,
    expand_band_training_rows,
    leaky_clip,
)


def test_expand_band_training_rows_uses_true_band_and_neighbours():
    matrix = pd.DataFrame({"feature": [10.0, 20.0, 30.0]})
    labels = np.array([0, 1, 7])
    target = np.array([30_000.0, 60_000.0, 900_000.0])
    weights = np.array([1.0, 2.0, 3.0])

    expanded, expanded_target, expanded_weights, source_rows = (
        expand_band_training_rows(matrix, labels, target, weights)
    )

    assert len(expanded) == 7
    np.testing.assert_array_equal(source_rows, [0, 1, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(
        expanded["fe_requested_band_id"].to_numpy(), [0, 0, 1, 1, 2, 6, 7]
    )
    np.testing.assert_array_equal(expanded_target, target[source_rows])
    np.testing.assert_array_equal(expanded_weights, weights[source_rows])


def test_band_context_contains_ordered_finite_boundaries():
    result = add_band_context(pd.DataFrame({"feature": [1.0]}), 4)

    assert result.loc[0, "fe_requested_band_id"] == 4
    assert result.loc[0, "fe_requested_band_lower_log1p"] < result.loc[
        0, "fe_requested_band_upper_log1p"
    ]
    assert np.isfinite(result.filter(like="fe_requested_band_").to_numpy()).all()


def test_band_context_rejects_invalid_band():
    with pytest.raises(ValueError, match="outside"):
        add_band_context(pd.DataFrame({"feature": [1.0]}), 8)


def test_leaky_clip_interpolates_between_hard_clip_and_raw_values():
    values = np.asarray([5.0, 15.0, 25.0])

    assert np.allclose(leaky_clip(values, 10.0, 20.0, 0.0), [10.0, 15.0, 20.0])
    assert np.allclose(leaky_clip(values, 10.0, 20.0, 0.2), [9.0, 15.0, 21.0])
    assert np.allclose(leaky_clip(values, 10.0, 20.0, 1.0), values)
