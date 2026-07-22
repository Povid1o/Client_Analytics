"""Utilities for a local hierarchical classifier over ordered income bands."""

import numpy as np


# Each node is trained only inside its own contiguous subtree.  This avoids
# asking every classifier to solve a global boundary problem dominated by the
# first income band.
TREE_NODES = [
    # name, eligible lower/upper bands, positive threshold
    ("root_ge1", 0, 7, 1),
    ("ge4_within_1_7", 1, 7, 4),
    ("ge2_within_1_3", 1, 3, 2),
    ("ge3_within_2_3", 2, 3, 3),
    ("ge5_within_4_7", 4, 7, 5),
    ("ge6_within_5_7", 5, 7, 6),
    ("ge7_within_6_7", 6, 7, 7),
]


def hierarchy_to_class_probabilities(node_probabilities):
    """Convert seven conditional tree probabilities into eight leaf masses."""
    p_root = np.asarray(node_probabilities["root_ge1"], dtype=float)
    p_ge4 = np.asarray(node_probabilities["ge4_within_1_7"], dtype=float)
    p_ge2 = np.asarray(node_probabilities["ge2_within_1_3"], dtype=float)
    p_ge3 = np.asarray(node_probabilities["ge3_within_2_3"], dtype=float)
    p_ge5 = np.asarray(node_probabilities["ge5_within_4_7"], dtype=float)
    p_ge6 = np.asarray(node_probabilities["ge6_within_5_7"], dtype=float)
    p_ge7 = np.asarray(node_probabilities["ge7_within_6_7"], dtype=float)

    low_mass = p_root * (1 - p_ge4)
    high_mass = p_root * p_ge4
    mass_23 = low_mass * p_ge2
    mass_567 = high_mass * p_ge5
    mass_67 = mass_567 * p_ge6
    probabilities = np.column_stack(
        [
            1 - p_root,
            low_mass * (1 - p_ge2),
            mass_23 * (1 - p_ge3),
            mass_23 * p_ge3,
            high_mass * (1 - p_ge5),
            mass_567 * (1 - p_ge6),
            mass_67 * (1 - p_ge7),
            mass_67 * p_ge7,
        ]
    )
    return probabilities / probabilities.sum(axis=1, keepdims=True)
