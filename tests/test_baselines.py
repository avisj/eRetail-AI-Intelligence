"""Unit Tests for Heuristic and Statistical Baseline Forecasting Models."""

import pytest
import numpy as np
import pandas as pd
from commerce_ai.forecasting.baselines import (
    NaiveModel,
    SeasonalNaiveModel,
    MovingAverageModel,
)


class TestBaselineModels:
    @pytest.fixture
    def linear_series(self):
        dates = pd.date_range("2023-01-01", periods=30)
        return pd.DataFrame({
            "date": dates,
            "sku_id": "SKU_001",
            "warehouse_id": "WH_01",
            "units_sold": [float(i) for i in range(1, 31)],
            "forecast_training_eligible": True,
        })

    @pytest.fixture
    def seasonal_series(self):
        # 4 weeks of pattern [10, 20, 30, 40, 50, 60, 70]
        pattern = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        units = pattern * 4
        dates = pd.date_range("2023-01-01", periods=len(units))
        return pd.DataFrame({
            "date": dates,
            "sku_id": "SKU_SEASONAL",
            "warehouse_id": "WH_01",
            "units_sold": units,
            "forecast_training_eligible": True,
        })

    def test_naive_model_forward_projection(self, linear_series):
        model = NaiveModel()
        model.fit(linear_series)
        output = model.predict(horizon=5)

        assert len(output.forecast_units) == 5
        # Last observed value in linear_series is 30.0
        assert np.allclose(output.forecast_units, 30.0)
        assert output.forecast_dates[0] == "2023-01-31"

    def test_naive_model_masks_stockout(self, linear_series):
        # Mark last 2 days as ineligible (e.g. stockout)
        series = linear_series.copy()
        series.loc[28:29, "forecast_training_eligible"] = False

        model = NaiveModel()
        model.fit(series)
        output = model.predict(horizon=3)

        # Should take index 27 (value 28.0)
        assert np.allclose(output.forecast_units, 28.0)

    def test_seasonal_naive_model(self, seasonal_series):
        model = SeasonalNaiveModel(season_length=7)
        model.fit(seasonal_series)
        output = model.predict(horizon=14)

        expected = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0] * 2
        assert len(output.forecast_units) == 14
        assert np.allclose(output.forecast_units, expected)

    def test_moving_average_model(self, linear_series):
        # Last 7 days: 24, 25, 26, 27, 28, 29, 30 -> mean is 27.0
        model = MovingAverageModel(window=7)
        model.fit(linear_series)
        output = model.predict(horizon=5)

        assert len(output.forecast_units) == 5
        assert np.allclose(output.forecast_units, 27.0)

    def test_baseline_non_negative_guarantee(self):
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "sku_id": "SKU_NEG",
            "warehouse_id": "WH_01",
            "units_sold": [-5.0, -10.0, -2.0, -1.0, -4.0],
        })
        model = NaiveModel()
        model.fit(df)
        output = model.predict(horizon=3)
        assert (output.forecast_units >= 0.0).all()
