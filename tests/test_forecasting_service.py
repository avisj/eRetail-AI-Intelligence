"""Unit Tests for ForecastService Single-Series and Batch Pipelines."""

import pytest
import numpy as np
import pandas as pd
from commerce_ai.forecasting.service import ForecastService
from commerce_ai.forecasting.selection import ModelSelectionPolicy


class TestForecastService:
    @pytest.fixture
    def single_series(self):
        dates = pd.date_range("2023-01-01", periods=60)
        return pd.DataFrame({
            "date": dates,
            "sku_id": "SKU_001",
            "warehouse_id": "WH_01",
            "units_sold": [float(i % 10 + 5) for i in range(60)],
            "forecast_training_eligible": True,
        })

    @pytest.fixture
    def multi_series(self):
        dates = pd.date_range("2023-01-01", periods=40)
        rows = []
        for d in dates:
            for s, wh, seg in [("SKU_A", "WH_01", "smooth"), ("SKU_B", "WH_01", "intermittent")]:
                rows.append({
                    "date": d,
                    "sku_id": s,
                    "warehouse_id": wh,
                    "units_sold": 10.0 if seg == "smooth" else (5.0 if d.day % 4 == 0 else 0.0),
                    "intermittency_class": seg,
                    "forecast_training_eligible": True,
                })
        return pd.DataFrame(rows)

    def test_forecast_series_default_and_explicit_model(self, single_series):
        service = ForecastService()

        # Explicit model: Moving Average
        out_ma = service.forecast_series(single_series, horizon=7, model_name="Moving Average (7d)")
        assert len(out_ma.forecast_units) == 7
        assert out_ma.model_name == "Moving Average (7d)"
        assert (out_ma.forecast_units >= 0.0).all()

        # Explicit model: Seasonal Naive
        out_sn = service.forecast_series(single_series, horizon=7, model_name="Seasonal Naive")
        assert len(out_sn.forecast_units) == 7
        assert out_sn.model_name == "Seasonal Naive"

    def test_forecast_batch_with_policy(self, multi_series):
        service = ForecastService()
        policy = ModelSelectionPolicy(
            global_champion="Naive",
            segment_champions={"smooth": "Moving Average (7d)", "intermittent": "Croston"},
        )

        batch_df = service.forecast_batch(
            dataset=multi_series,
            horizon=5,
            policy=policy,
        )

        assert not batch_df.empty
        # 2 series * 5 horizon = 10 rows
        assert len(batch_df) == 10
        sku_a_models = batch_df[batch_df["sku_id"] == "SKU_A"]["model_name"].unique()
        sku_b_models = batch_df[batch_df["sku_id"] == "SKU_B"]["model_name"].unique()

        assert "Moving Average (7d)" in sku_a_models
        assert "Croston" in sku_b_models
        assert (batch_df["forecast_units"] >= 0.0).all()

    def test_forecast_series_input_validation(self):
        service = ForecastService()
        with pytest.raises(ValueError, match="Input history DataFrame is empty"):
            service.forecast_series(pd.DataFrame())

        with pytest.raises(ValueError, match="Target column 'units_sold' not found"):
            service.forecast_series(pd.DataFrame({"date": ["2023-01-01"]}))
