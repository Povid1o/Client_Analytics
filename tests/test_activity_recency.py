import numpy as np
import pandas as pd

from src.feature_engineering import add_activity_recency


def test_activity_recency_clips_old_and_future_dates_and_flags_missing():
    frame = pd.DataFrame(
        {
            "dt": ["2024-06-30"] * 4,
            "period_last_act_ad": ["2024-06-01", "2020-01-01", "2024-07-01", None],
        }
    )
    result = add_activity_recency(frame)

    assert result["fe_days_since_last_activity"].iloc[:3].tolist() == [29.0, 365.0, 0.0]
    assert np.isnan(result["fe_days_since_last_activity"].iloc[3])
    assert result["fe_missing_last_activity"].tolist() == [0, 0, 0, 1]
