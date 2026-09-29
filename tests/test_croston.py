"""Unit Tests for Croston and SBA Intermittent Demand Forecasting Models."""

import pytest
import numpy as np
import pandas as pd
from commerce_ai.forecasting.croston import CrostonModel


class TestCrostonModel:
    @pytest.fixture
    def intermittent_series(self):
        # 60 days, mostly zeros with occasional bursts of demand
        np.random.seed(42)
        n = 60
        units = np.zeros(n)
        demand_days = [5, 12, 23, 29, 38, 45, 54]
        for day in demand_days:
            units[day] = np.random.randint(5, 20)

        dates = pd.date_range("2023-01-01", periods=n)
        return pd.DataFrame({
            "date": dates,
            "sku_id": "SKU_INTERMITTENT",
            "warehouse_id": "WH_01",
            "units_sold": units,
            "forecast_training_eligible": True,
        })

    def test_croston_classic_and_sba_variants(self, intermittent_series):
        model_classic = CrostonModel(variant="classic", alpha=0.1)
        model_classic.fit(intermittent_series)
        out_classic = model_classic.predict(horizon=10)

        model_sba = CrostonModel(variant="sba", alpha=0.1)
        model_sba.fit(intermittent_series)
        out_sba = model_sba.predict(horizon=10)

        assert len(out_classic.forecast_units) == 10
        assert len(out_sba.forecast_units) == 10
        assert (out_classic.forecast_units > 0.0).all()
        assert (out_sba.forecast_units > 0.0).all()

        # SBA applies deflator factor (1 - alpha/2), so SBA forecast <= Classic forecast
        assert (out_sba.forecast_units <= out_classic.forecast_units).all()

    def test_croston_all_zeros_series(self):
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=20),
            "sku_id": "SKU_ALL_ZERO",
            "warehouse_id": "WH_01",
            "units_sold": [0.0] * 20,
        })
        model = CrostonModel()
        model.fit(df)
        out = model.predict(horizon=5)
        assert np.allclose(out.forecast_units, 0.0)
