import numpy as np

from src.hierarchical_routing import TREE_NODES, hierarchy_to_class_probabilities


def node_values(value, rows=3):
    return {name: np.full(rows, value, dtype=float) for name, *_ in TREE_NODES}


def test_hierarchy_extreme_paths_reach_first_and_last_band():
    first = hierarchy_to_class_probabilities(node_values(0.0, rows=1))
    last = hierarchy_to_class_probabilities(node_values(1.0, rows=1))

    np.testing.assert_allclose(first, [[1, 0, 0, 0, 0, 0, 0, 0]])
    np.testing.assert_allclose(last, [[0, 0, 0, 0, 0, 0, 0, 1]])


def test_hierarchy_probabilities_are_normalized_and_nonnegative():
    probabilities = hierarchy_to_class_probabilities(node_values(0.5))

    assert probabilities.shape == (3, 8)
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        probabilities[0], [0.5, 0.125, 0.0625, 0.0625, 0.125, 0.0625, 0.03125, 0.03125]
    )
