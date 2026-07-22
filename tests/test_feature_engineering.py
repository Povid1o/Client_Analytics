import numpy as np
import pandas as pd

from src.feature_engineering import (
    add_anchor_agreement,
    add_flow_balance,
    add_missing_flags,
    safe_ratio,
)


def test_safe_ratio_handles_zero_and_extreme_denominators():
    result = safe_ratio([10.0, 10.0, np.nan, 1e9], [2.0, 0.0, 2.0, 1.0])
    assert result[0] == 5.0
    assert np.isnan(result[1])
    assert np.isnan(result[2])
    assert np.isnan(result[3])


def test_anchor_agreement_uses_available_non_negative_values():
    frame = pd.DataFrame(
        {
            "salary_6to12m_avg": [100.0, np.nan, -1.0],
            "incomeValue": [110.0, 50.0, 60.0],
            "dp_ils_avg_salary_1y": [90.0, 70.0, np.nan],
        }
    )
    result = add_anchor_agreement(frame)
    np.testing.assert_allclose(result["fe_income_anchor_count"], [3, 2, 1])
    np.testing.assert_allclose(result["fe_income_anchor_mean"], [100, 60, 60])
    np.testing.assert_allclose(result["fe_income_anchor_range_rel"], [0.2, 1 / 3, 0])


def test_flow_balance_is_scale_invariant():
    frame = pd.DataFrame(
        {
            "turn_cur_cr_avg_v2": [100.0, 1000.0],
            "turn_cur_db_avg_v2": [50.0, 500.0],
        }
    )
    result = add_flow_balance(frame)
    np.testing.assert_allclose(result["fe_cur_avg12_cr_to_db"], [2.0, 2.0])
    np.testing.assert_allclose(result["fe_cur_avg12_balance"], [1 / 3, 1 / 3])


def test_missing_flags_keep_missing_distinct_from_zero():
    frame = pd.DataFrame({"x": [np.nan, 0.0, 1.0]})
    result = add_missing_flags(frame, columns=["x"])
    np.testing.assert_array_equal(result["fe_missing_x"], [1, 0, 0])
