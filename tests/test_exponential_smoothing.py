"""Unit Tests for Exponential Smoothing Models."""

import pytest
import numpy as np
import pandas as pd
from commerce_ai.forecasting.exponential_smoothing import ExponentialSmoothingModel


class TestExponentialSmoothing:
    @pytest.fixture
    def trend_seasonal_series(self):
        # 100 days of upward trend with weekly seasonality
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        trend = 0.2 * t
        seasonal = 5.0 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 1.0, n)
        units = np.maximum(0.0, 10.0 + trend + seasonal + noise)
        dates = pd.date_range("2023-01-01", periods=n)
        return pd.DataFrame({
            "date": dates,
            "sku_id": "SKU_ETS",
            "warehouse_id": "WH_01",
            "units_sold": units,
            "forecast_training_eligible": True,
        })

    def test_exponential_smoothing_fit_predict(self, trend_seasonal_series):
        model = ExponentialSmoothingModel(seasonal_periods=7)
        model.fit(trend_seasonal_series)
        assert model.is_fitted

        output = model.predict(horizon=14, confidence_level=0.95)
        assert len(output.forecast_units) == 14
        assert (output.forecast_units >= 0.0).all()
        assert output.lower_bound is not None
        assert output.upper_bound is not None
        assert (output.lower_bound <= output.upper_bound).all()
        assert output.metadata.status == "SUCCESS"

    def test_exponential_smoothing_fallback_on_constant_series(self):
        # Constant zero/low series can cause statsmodels non-convergence
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            "sku_id": "SKU_CONST",
            "warehouse_id": "WH_01",
            "units_sold": [0.0] * 10,
        })
        model = ExponentialSmoothingModel()
        model.fit(df)
        out = model.predict(horizon=5)
        assert len(out.forecast_units) == 5
        assert np.allclose(out.forecast_units, 0.0)
