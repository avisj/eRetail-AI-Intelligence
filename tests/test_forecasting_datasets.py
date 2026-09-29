"""Unit Tests for Forecasting Datasets and Rolling Origin Splits."""

import pytest
import pandas as pd
import numpy as np
from commerce_ai.forecasting.datasets import (
    build_forecast_dataset,
    time_series_train_test_split,
    generate_rolling_origin_folds,
)
from commerce_ai.forecasting.config import BacktestConfig


class TestForecastingDatasets:
    @pytest.fixture
    def sample_daily_demand(self):
        dates = pd.date_range("2023-01-01", periods=100)
        records = []
        for d in dates:
            for s in ["SKU_1", "SKU_2"]:
                records.append({
                    "date": d,
                    "sku_id": s,
                    "warehouse_id": "WH_1",
                    "units_sold": 10.0,
                    "available_qty": 50.0,
                    "is_stockout": False,
                })
        return pd.DataFrame(records)

    def test_build_forecast_dataset(self, sample_daily_demand):
        ds = build_forecast_dataset(sample_daily_demand)
        assert "units_sold" in ds.columns
        assert "forecast_training_eligible" in ds.columns
        assert len(ds) == 200
        assert (ds["units_sold"] >= 0).all()

    def test_time_series_train_test_split(self, sample_daily_demand):
        train, test = time_series_train_test_split(
            sample_daily_demand,
            test_days=14,
            date_col="date",
        )
        assert len(train) + len(test) == len(sample_daily_demand)
        max_train_date = pd.to_datetime(train["date"]).max()
        min_test_date = pd.to_datetime(test["date"]).min()
        assert max_train_date < min_test_date
        assert (pd.to_datetime(test["date"]).max() - min_test_date).days + 1 == 14

    def test_generate_rolling_origin_folds(self, sample_daily_demand):
        config = BacktestConfig(
            horizon=7,
            folds=3,
            step_size=7,
            min_history=30,
        )
        folds = generate_rolling_origin_folds(sample_daily_demand, config)
        assert len(folds) == 3
        for f in folds:
            assert "train_df" in f
            assert "test_df" in f
            assert "cutoff_date" in f
            train_max = pd.to_datetime(f["train_df"]["date"]).max()
            test_min = pd.to_datetime(f["test_df"]["date"]).min()
            assert train_max < test_min

    def test_generate_rolling_origin_folds_insufficient_history(self, sample_daily_demand):
        config = BacktestConfig(
            horizon=30,
            folds=5,
            step_size=30,
            min_history=60,
        )
        with pytest.raises(ValueError, match="Insufficient historical span"):
            generate_rolling_origin_folds(sample_daily_demand, config)
