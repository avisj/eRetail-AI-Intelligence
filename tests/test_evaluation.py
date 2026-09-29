"""Unit Tests for Hierarchical Forecast Evaluation and Benchmarking."""

import pytest
import numpy as np
import pandas as pd
from commerce_ai.forecasting.evaluation import ForecastEvaluator


class TestForecastEvaluation:
    @pytest.fixture
    def mock_predictions(self):
        return pd.DataFrame({
            "model_name": ["ModelA"] * 4 + ["ModelB"] * 4,
            "sku_id": ["SKU_1", "SKU_1", "SKU_2", "SKU_2"] * 2,
            "warehouse_id": ["WH_1"] * 8,
            "actual_units": [10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0],
            "forecast_units": [12.0, 18.0, 33.0, 37.0, 15.0, 25.0, 35.0, 45.0],
            "lower_bound": [8.0, 15.0, 25.0, 30.0, 10.0, 20.0, 30.0, 40.0],
            "upper_bound": [15.0, 25.0, 38.0, 42.0, 20.0, 30.0, 40.0, 50.0],
        })

    def test_evaluate_slice_metrics(self):
        evaluator = ForecastEvaluator()
        actual = [10.0, 20.0, 30.0]
        pred = [12.0, 18.0, 30.0]  # errors: 2, 2, 0 -> MAE = 4/3 = 1.3333
        metrics = evaluator.evaluate_slice(actual, pred)

        assert abs(metrics["mae"] - 1.3333) < 1e-3
        assert metrics["total_actual"] == 60.0
        assert metrics["total_forecast"] == 60.0
        assert metrics["bias"] == 0.0  # (60 - 60) / 60

    def test_evaluate_predictions_grouped(self, mock_predictions):
        evaluator = ForecastEvaluator()
        res = evaluator.evaluate_predictions(mock_predictions, group_by=["model_name"])
        assert len(res) == 2
        assert "wape" in res.columns
        # ModelA has total abs error |2| + |2| + |3| + |3| = 10 / 100 = 10%
        # ModelB has total abs error |5| + |5| + |5| + |5| = 20 / 100 = 20%
        model_a = res[res["model_name"] == "ModelA"].iloc[0]
        model_b = res[res["model_name"] == "ModelB"].iloc[0]
        assert model_a["wape"] == 10.0
        assert model_b["wape"] == 20.0

    def test_generate_benchmark_table(self, mock_predictions):
        evaluator = ForecastEvaluator()
        diag_df = pd.DataFrame([
            {"model_name": "ModelA", "status": "SUCCESS", "runtime_seconds": 0.5},
            {"model_name": "ModelB", "status": "SUCCESS", "runtime_seconds": 1.2},
            {"model_name": "ModelC", "status": "TIMESFM_UNAVAILABLE", "runtime_seconds": 0.01},
        ])
        bench = evaluator.generate_benchmark_table(mock_predictions, diag_df)

        assert len(bench) == 3
        assert list(bench.columns) == [
            "Model", "MAE", "RMSE", "WAPE (%)", "Bias (%)", "Runtime (s)", "Failures", "Status"
        ]
        # ModelA should be first (WAPE 10.0 < 20.0)
        assert bench.iloc[0]["Model"] == "ModelA"
        assert bench.iloc[0]["WAPE (%)"] == 10.0
        # ModelC should have UNAVAILABLE status
        model_c_row = bench[bench["Model"] == "ModelC"].iloc[0]
        assert model_c_row["Status"] == "UNAVAILABLE"
