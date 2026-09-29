"""Unit Tests for Rolling Origin Time-Series Backtesting."""

import pytest
import numpy as np
import pandas as pd
from commerce_ai.forecasting.backtesting import RollingOriginBacktester
from commerce_ai.forecasting.config import BacktestConfig
from commerce_ai.forecasting.baselines import NaiveModel, MovingAverageModel
from commerce_ai.forecasting.base import ForecastModel, ForecastOutput


class FailingModel(ForecastModel):
    name = "AlwaysFails"

    def fit(self, history, target_col="units_sold", date_col="date", metadata=None):
        raise RuntimeError("Simulated model training explosion!")

    def predict(self, horizon, future_features=None, confidence_level=None):
        return ForecastOutput(
            sku_id="X", warehouse_id="Y", forecast_dates=[], forecast_units=np.zeros(horizon), model_name=self.name
        )


class TestBacktesting:
    @pytest.fixture
    def multi_series_df(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=90)
        rows = []
        for d in dates:
            for s in ["SKU_1", "SKU_2"]:
                rows.append({
                    "date": d,
                    "sku_id": s,
                    "warehouse_id": "WH_1",
                    "units_sold": float(np.random.poisson(lam=15)),
                    "forecast_training_eligible": True,
                })
        return pd.DataFrame(rows)

    def test_backtester_multi_fold_execution(self, multi_series_df):
        config = BacktestConfig(horizon=7, folds=2, step_size=7, min_history=30)
        backtester = RollingOriginBacktester(
            config=config,
            models={
                "Naive": NaiveModel,
                "MA7": lambda: MovingAverageModel(window=7),
            },
        )

        res = backtester.run(
            multi_series_df,
            series_keys=[("SKU_1", "WH_1")],
        )

        assert not res.predictions_df.empty
        assert res.total_failures == 0
        assert res.failure_rate == 0.0
        assert set(res.predictions_df["model_name"]) == {"Naive", "MA7"}
        assert set(res.predictions_df["fold_index"]) == {1, 2}
        # 2 folds * 7 horizon = 14 predictions per model -> 28 total
        assert len(res.predictions_df) == 28

    def test_backtester_error_isolation(self, multi_series_df):
        config = BacktestConfig(horizon=7, folds=2, step_size=7, min_history=30)
        backtester = RollingOriginBacktester(
            config=config,
            models={
                "Naive": NaiveModel,
                "Failing": FailingModel,
            },
        )

        res = backtester.run(multi_series_df, series_keys=[("SKU_1", "WH_1")])

        # Naive succeeded, Failing recorded in diagnostics
        assert len(res.predictions_df[res.predictions_df["model_name"] == "Naive"]) == 14
        assert res.total_failures > 0
        failing_diag = res.diagnostics_df[res.diagnostics_df["model_name"] == "Failing"]
        assert (failing_diag["status"] == "FAILED").all()
        assert "Simulated model training explosion!" in failing_diag["error_message"].iloc[0]
