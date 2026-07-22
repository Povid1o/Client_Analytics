"""Leakage-free prediction calibration using raw test-time features only."""
import numpy as np


def blend_salary_signal(predictions, salary, alpha=0.6):
    """Blend predictions with the high-quality 6-12 month salary estimate.

    ``salary_6to12m_avg`` is present for roughly 19% of rows and is unusually
    close to the target there. Missing, infinite, and negative values are
    ignored. The operation never mutates the input prediction array.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in the interval [0, 1]")

    predictions = np.asarray(predictions, dtype=float)
    salary = np.asarray(salary, dtype=float)
    if predictions.shape != salary.shape:
        raise ValueError("predictions and salary must have the same shape")

    result = predictions.copy()
    usable = np.isfinite(salary) & (salary >= 0)
    result[usable] = (1.0 - alpha) * result[usable] + alpha * salary[usable]
    return result
