"""Unit Tests for LightGBM Demand Forecasting Model."""

import pytest
import numpy as np
import pandas as pd
from commerce_ai.forecasting.lightgbm_model import LightGBMForecastModel


class TestLightGBMModel:
    @pytest.fixture
    def demand_feature_series(self):
        np.random.seed(42)
        n = 80
        dates = pd.date_range("2023-01-01", periods=n)
        units = np.random.poisson(lam=12.0, size=n).astype(float)
        
        df = pd.DataFrame({
            "date": dates,
            "sku_id": "SKU_LGB",
            "warehouse_id": "WH_01",
            "units_sold": units,
            "forecast_training_eligible": True,
            "day_of_week": dates.dayofweek,
            "month": dates.month,
            "demand_lag_1": np.roll(units, 1),
            "demand_lag_7": np.roll(units, 7),
            "demand_rolling_mean_7": pd.Series(units).rolling(7, min_periods=1).mean().values,
        })
        return df

    def test_lightgbm_fit_and_recursive_predict(self, demand_feature_series):
        model = LightGBMForecastModel()
        model.fit(demand_feature_series)
        assert model.is_fitted
        assert model.regressor is not None

        out = model.predict(horizon=7, confidence_level=0.95)
        assert len(out.forecast_units) == 7
        assert (out.forecast_units >= 0.0).all()
        assert out.lower_bound is not None
        assert out.upper_bound is not None
        assert (out.lower_bound <= out.upper_bound).all()
        assert out.metadata.status == "SUCCESS"

    def test_lightgbm_direct_predict_with_future_features(self, demand_feature_series):
        train_df = demand_feature_series.iloc[:60].copy()
        test_df = demand_feature_series.iloc[60:].copy()

        model = LightGBMForecastModel()
        model.fit(train_df)
        
        out = model.predict(horizon=len(test_df), future_features=test_df)
        assert len(out.forecast_units) == len(test_df)
        assert (out.forecast_units >= 0.0).all()

    def test_lightgbm_insufficient_history_fallback(self):
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3),
            "sku_id": "SKU_TINY",
            "warehouse_id": "WH_01",
            "units_sold": [5.0, 5.0, 5.0],
        })
        model = LightGBMForecastModel()
        model.fit(df)
        assert model.is_fitted
        assert model._metadata.status == "INSUFFICIENT_HISTORY"

        out = model.predict(horizon=4)
        assert len(out.forecast_units) == 4
        assert np.allclose(out.forecast_units, 5.0)
