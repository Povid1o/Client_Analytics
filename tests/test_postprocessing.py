import numpy as np
import pytest

from src.postprocessing import blend_salary_signal


def test_salary_blend_only_changes_rows_with_usable_salary():
    predictions = np.array([100.0, 200.0, 300.0, 400.0])
    salary = np.array([120.0, np.nan, -1.0, np.inf])

    result = blend_salary_signal(predictions, salary, alpha=0.5)

    np.testing.assert_allclose(result, [110.0, 200.0, 300.0, 400.0])
    np.testing.assert_allclose(predictions, [100.0, 200.0, 300.0, 400.0])


def test_salary_blend_validates_inputs():
    with pytest.raises(ValueError):
        blend_salary_signal([1.0], [2.0], alpha=1.1)
    with pytest.raises(ValueError):
        blend_salary_signal([1.0, 2.0], [2.0], alpha=0.5)
