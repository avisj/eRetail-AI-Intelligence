"""Unit Tests for Forecasting Base Interfaces and Contracts."""

import pytest
import numpy as np
import pandas as pd
from commerce_ai.forecasting.base import (
    ForecastModel,
    ForecastRecord,
    ForecastMetadata,
    ForecastOutput,
)


class DummyModel(ForecastModel):
    name = "Dummy"

    def fit(self, history, target_col="units_sold", date_col="date", metadata=None):
        self._extract_entity_ids(history)
        self.is_fitted = True
        self.last_date = pd.to_datetime(history[date_col].iloc[-1])
        return self

    def predict(self, horizon, future_features=None, confidence_level=None):
        dates = self._build_future_dates(self.last_date, horizon)
        preds = np.full(horizon, 5.0)
        return ForecastOutput(
            sku_id=self._sku_id,
            warehouse_id=self._warehouse_id,
            forecast_dates=dates,
            forecast_units=preds,
            model_name=self.name,
        )


class TestForecastingBaseContracts:
    def test_forecast_record_serialization(self):
        rec = ForecastRecord(
            forecast_date="2023-01-01",
            sku_id="SKU_001",
            warehouse_id="WH_01",
            forecast_units=15.5,
            model_name="TestModel",
            model_version="1.0.0",
            prediction_run_id="run_123",
            forecast_generated_at="2023-01-01T00:00:00Z",
            lower_bound=10.0,
            upper_bound=20.0,
            confidence_level=0.95,
        )
        d = rec.to_dict()
        assert d["forecast_units"] == 15.5
        assert d["sku_id"] == "SKU_001"
        assert d["lower_bound"] == 10.0
        assert d["confidence_level"] == 0.95

    def test_forecast_output_to_dataframe(self):
        dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
        units = np.array([10.0, 20.0, 30.0])
        lowers = np.array([8.0, 18.0, 28.0])
        uppers = np.array([12.0, 22.0, 32.0])

        output = ForecastOutput(
            sku_id="SKU_TEST",
            warehouse_id="WH_TEST",
            forecast_dates=dates,
            forecast_units=units,
            model_name="MockModel",
            lower_bound=lowers,
            upper_bound=uppers,
            confidence_level=0.9,
        )

        df = output.to_dataframe()
        assert len(df) == 3
        assert list(df.columns) == [
            "forecast_date", "sku_id", "warehouse_id", "forecast_units",
            "model_name", "model_version", "prediction_run_id", "forecast_generated_at",
            "lower_bound", "upper_bound", "confidence_level",
        ]
        assert df["forecast_units"].tolist() == [10.0, 20.0, 30.0]
        assert df["lower_bound"].tolist() == [8.0, 18.0, 28.0]

    def test_forecast_metadata_defaults(self):
        meta = ForecastMetadata(
            model_name="Naive",
            model_version="1.0.0",
            status="SUCCESS",
        )
        assert meta.status == "SUCCESS"
        assert meta.runtime_seconds == 0.0
        assert meta.error_code is None

    def test_custom_model_lifecycle(self):
        model = DummyModel()
        assert not model.is_fitted
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            "sku_id": "SKU_A",
            "warehouse_id": "WH_01",
            "units_sold": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        model.fit(df)
        assert model.is_fitted
        assert model._sku_id == "SKU_A"
        assert model._warehouse_id == "WH_01"

        out = model.predict(horizon=5)
        assert len(out.forecast_dates) == 5
        assert out.forecast_dates[0] == "2023-01-11"
        assert (out.forecast_units == 5.0).all()
