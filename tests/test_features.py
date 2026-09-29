"""Tests for Feature Engineering and Data Leakage Protection."""

import numpy as np
import pandas as pd
import pytest

from commerce_ai.analytics.features import (
    build_lag_features,
    build_rolling_features,
    build_trend_features,
    calculate_intermittency_metrics,
    build_demand_features,
)


@pytest.fixture
def sequence_demand():
    """Deterministic demand sequence: [10, 20, 30, 40, 50, 60, 70]."""
    dates = pd.date_range("2025-01-01", periods=7, freq="D")
    return pd.DataFrame({
        "date": dates,
        "sku_id": ["SKU_1"] * 7,
        "warehouse_id": ["WH_1"] * 7,
        "units_sold": [10, 20, 30, 40, 50, 60, 70],
        "available_qty": [100] * 7,
        "revenue": [100.0 * i for i in range(1, 8)],
    })


class TestFeatureEngineering:
    def test_lag_feature_correctness(self, sequence_demand):
        df = build_lag_features(sequence_demand, lags=[1, 2], group_cols=["sku_id", "warehouse_id"])

        # Day 1 (idx 0): Lag 1 is NaN
        assert pd.isna(df.loc[0, "demand_lag_1"])
        # Day 2 (idx 1): Lag 1 is Day 1 demand (10)
        assert df.loc[1, "demand_lag_1"] == 10
        # Day 3 (idx 2): Lag 1 is Day 2 (20), Lag 2 is Day 1 (10)
        assert df.loc[2, "demand_lag_1"] == 20
        assert df.loc[2, "demand_lag_2"] == 10

    def test_rolling_feature_causality_and_leakage_protection(self, sequence_demand):
        """CRITICAL: Rolling features at day T must NOT include demand from day T."""
        df = build_rolling_features(sequence_demand, windows=[3], group_cols=["sku_id", "warehouse_id"])

        # Day 1: shifted demand is NaN -> rolling mean is NaN
        assert pd.isna(df.loc[0, "demand_rolling_mean_3"])
        # Day 2: rolling mean over 1 available historical day (Day 1: 10) -> 10.0
        assert df.loc[1, "demand_rolling_mean_3"] == 10.0
        # Day 4: rolling mean over Days 1, 2, 3 (10, 20, 30) -> mean is 20.0 (Day 4's own demand of 40 is NOT included!)
        assert df.loc[3, "demand_rolling_mean_3"] == 20.0

    def test_strict_anti_leakage_invariance(self, sequence_demand):
        """Demonstrate that modifying future demand does NOT change past historical features."""
        df_original = build_rolling_features(sequence_demand, windows=[3])

        # Mutate day 7 from 70 to 99,999
        tampered_demand = sequence_demand.copy()
        tampered_demand.loc[6, "units_sold"] = 99999
        df_tampered = build_rolling_features(tampered_demand, windows=[3])

        # Days 1 through 6 MUST be strictly identical
        pd.testing.assert_series_equal(
            df_original.loc[:5, "demand_rolling_mean_3"],
            df_tampered.loc[:5, "demand_rolling_mean_3"],
        )

    def test_trend_features_safe_division(self):
        df = pd.DataFrame({
            "demand_rolling_mean_7": [10.0, 0.0, 50.0],
            "demand_rolling_mean_30": [20.0, 0.0, 0.0],  # Zero denominators
            "demand_lag_1": [15, 0, 10],
            "demand_lag_7": [10, 0, 0],
        })
        trends = build_trend_features(df)

        assert trends.loc[0, "trend_7_vs_30"] == 0.5
        # Zero denominator should produce NaN, NEVER inf
        assert not np.isinf(trends.loc[1, "trend_7_vs_30"]).any()
        assert not np.isinf(trends.loc[2, "trend_7_vs_30"]).any()

    def test_intermittency_metrics(self):
        df = pd.DataFrame({
            "sku_id": ["REGULAR", "REGULAR", "REGULAR", "INTERMITTENT", "INTERMITTENT", "INTERMITTENT", "DORMANT", "DORMANT", "DORMANT"],
            "units_sold": [10, 20, 15, 0, 10, 0, 0, 0, 0],
        })
        intermittency = calculate_intermittency_metrics(df)

        reg_row = intermittency[intermittency["sku_id"] == "REGULAR"].iloc[0]
        assert reg_row["intermittency_class"] == "Regular"
        assert reg_row["demand_occurrence_rate"] == 1.0

        int_row = intermittency[intermittency["sku_id"] == "INTERMITTENT"].iloc[0]
        assert int_row["intermittency_class"] == "Intermittent"
        assert int_row["demand_occurrence_rate"] == round(1 / 3, 4)

        dorm_row = intermittency[intermittency["sku_id"] == "DORMANT"].iloc[0]
        assert dorm_row["intermittency_class"] == "Highly Intermittent"
        assert dorm_row["demand_occurrence_rate"] == 0.0

    def test_master_build_demand_features_pipeline(self, sequence_demand):
        features = build_demand_features(sequence_demand)

        expected_columns = [
            "date", "sku_id", "warehouse_id", "units_sold", "revenue",
            "is_stockout", "forecast_training_eligible",
            "abc_class", "xyz_class", "abc_xyz_class",
            "demand_lag_1", "demand_rolling_mean_7",
            "year", "month", "day_of_week", "is_weekend",
        ]
        for col in expected_columns:
            assert col in features.columns, f"Missing expected column {col}"
        assert len(features) == len(sequence_demand)
