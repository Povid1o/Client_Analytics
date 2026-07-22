import numpy as np

from src.ordinal_routing import (
    cumulative_to_class_probabilities,
    income_band,
    normalize_probabilities,
    posterior_median_band,
    project_to_bands,
    route_predictions,
    temperature_scale,
)


def test_income_band_obeys_boundaries():
    edges = np.asarray([20.0, 50.0, 100.0, 200.0])
    values = np.asarray([20.0, 49.9, 50.0, 100.0, 200.0])
    np.testing.assert_array_equal(income_band(values, edges), [0, 0, 1, 2, 2])


def test_temperature_scale_preserves_rows_and_sharpens():
    probabilities = np.asarray([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    sharpened = temperature_scale(probabilities, 0.5)
    np.testing.assert_allclose(sharpened.sum(axis=1), 1.0)
    assert sharpened[0, 0] > probabilities[0, 0]
    assert sharpened[1, 2] > probabilities[1, 2]


def test_projection_and_routing():
    edges = np.asarray([20.0, 50.0, 100.0])
    base = np.asarray([40.0, 120.0])
    candidates = project_to_bands(base, edges)
    np.testing.assert_allclose(candidates, [[40.0, 50.0], [50.0, 100.0]])
    probabilities = normalize_probabilities([[0.75, 0.25], [0.2, 0.8]])
    np.testing.assert_allclose(
        route_predictions(base, probabilities, candidates, mode="soft"),
        [42.5, 90.0],
    )
    np.testing.assert_array_equal(posterior_median_band(probabilities), [0, 1])
    np.testing.assert_allclose(
        route_predictions(base, probabilities, candidates, mode="median"),
        [40.0, 100.0],
    )


def test_cumulative_probabilities_are_made_monotone():
    cumulative = np.asarray([[0.8, 0.9, 0.2], [0.6, 0.4, 0.1]])
    classes = cumulative_to_class_probabilities(cumulative)
    np.testing.assert_allclose(classes.sum(axis=1), 1.0)
    assert np.all(classes >= 0)
    np.testing.assert_allclose(classes[0], [0.2, 0.0, 0.6, 0.2])
