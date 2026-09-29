"""Tests for Calendar and Business Event Feature Engineering."""

from datetime import date
import numpy as np
import pandas as pd
import pytest

from commerce_ai.analytics.calendar import (
    BusinessEvent,
    build_calendar_features,
    build_event_features,
    get_standard_ecommerce_events,
)


@pytest.fixture
def sample_date_dataframe():
    """Generates dates spanning key calendar boundaries:
    - 2024-01-01: Monday, New Year, Month start, Quarter start
    - 2024-01-31: Wednesday, Month end
    - 2024-02-03: Saturday, Weekend
    - 2024-03-31: Sunday, Month end, Quarter end
    """
    dates = ["2024-01-01", "2024-01-31", "2024-02-03", "2024-03-31"]
    return pd.DataFrame({"date": pd.to_datetime(dates)})


class TestCalendarFeatures:
    def test_calendar_temporal_flags(self, sample_date_dataframe):
        features = build_calendar_features(sample_date_dataframe, date_col="date")

        # 2024-01-01
        assert features.loc[0, "year"] == 2024
        assert features.loc[0, "month"] == 1
        assert features.loc[0, "quarter"] == 1
        assert features.loc[0, "day_of_week"] == 0  # Monday
        assert features.loc[0, "is_weekend"] == 0
        assert features.loc[0, "is_month_start"] == 1
        assert features.loc[0, "is_quarter_start"] == 1

        # 2024-01-31
        assert features.loc[1, "is_month_end"] == 1
        assert features.loc[1, "is_quarter_end"] == 0

        # 2024-02-03 (Saturday)
        assert features.loc[2, "day_of_week"] == 5
        assert features.loc[2, "is_weekend"] == 1

        # 2024-03-31
        assert features.loc[3, "is_month_end"] == 1
        assert features.loc[3, "is_quarter_end"] == 1

    def test_cyclical_trigonometric_invariants(self, sample_date_dataframe):
        features = build_calendar_features(sample_date_dataframe, date_col="date")

        # sin^2 + cos^2 == 1.0 for all rows
        dow_trig = features["sin_day_of_week"] ** 2 + features["cos_day_of_week"] ** 2
        month_trig = features["sin_month"] ** 2 + features["cos_month"] ** 2

        np.testing.assert_allclose(dow_trig, 1.0, atol=1e-5)
        np.testing.assert_allclose(month_trig, 1.0, atol=1e-5)

    def test_business_event_matching(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-11-29", "2024-11-15"])})
        events = [
            BusinessEvent(date(2024, 11, 29), "Black Friday", "Mega Sale", 3.5),
        ]
        result = build_event_features(df, events=events, date_col="date")

        # Black Friday row
        assert result.loc[0, "is_event"] == 1
        assert result.loc[0, "event_name"] == "Black Friday"
        assert result.loc[0, "event_impact_factor"] == 3.5

        # Normal day row
        assert result.loc[1, "is_event"] == 0
        assert result.loc[1, "event_impact_factor"] == 1.0
