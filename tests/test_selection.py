"""Unit Tests for Model Selection and Policy Routing Engine."""

import pytest
import pandas as pd
from commerce_ai.forecasting.selection import (
    ModelSelector,
    SelectionCriteria,
    ModelSelectionPolicy,
)


class TestModelSelection:
    @pytest.fixture
    def benchmark_table(self):
        return pd.DataFrame([
            {
                "Model": "LightGBM",
                "MAE": 2.1,
                "RMSE": 3.0,
                "WAPE (%)": 12.5,
                "Bias (%)": 0.8,
                "Runtime (s)": 2.4,
                "Failures": 0,
                "Status": "SUCCESS",
            },
            {
                "Model": "Seasonal Naive",
                "MAE": 2.3,
                "RMSE": 3.2,
                "WAPE (%)": 13.0,
                "Bias (%)": -0.2,
                "Runtime (s)": 0.05,
                "Failures": 0,
                "Status": "SUCCESS",
            },
            {
                "Model": "Moving Average (7d)",
                "MAE": 2.8,
                "RMSE": 3.8,
                "WAPE (%)": 18.0,
                "Bias (%)": -1.5,
                "Runtime (s)": 0.02,
                "Failures": 0,
                "Status": "SUCCESS",
            },
            {
                "Model": "TimesFM",
                "MAE": None,
                "RMSE": None,
                "WAPE (%)": None,
                "Bias (%)": None,
                "Runtime (s)": 0.0,
                "Failures": 1,
                "Status": "UNAVAILABLE",
            },
        ])

    def test_select_best_model_without_parsimony(self, benchmark_table):
        # Strict lowest WAPE
        selector = ModelSelector(SelectionCriteria(prefer_simpler_baseline_if_within_pct=0.0))
        best = selector.select_best_model(benchmark_table)
        assert best == "LightGBM"

    def test_select_best_model_with_parsimony(self, benchmark_table):
        # Seasonal Naive is 13.0%, within 2.0% of LightGBM (12.5%) -> parsimony picks Seasonal Naive
        selector = ModelSelector(SelectionCriteria(prefer_simpler_baseline_if_within_pct=2.0))
        best = selector.select_best_model(benchmark_table)
        assert best == "Seasonal Naive"

    def test_segment_routing_policy(self, benchmark_table):
        seg_eval = pd.DataFrame([
            {"intermittency_class": "smooth", "model_name": "LightGBM", "wape": 10.0},
            {"intermittency_class": "smooth", "model_name": "Croston", "wape": 25.0},
            {"intermittency_class": "intermittent", "model_name": "LightGBM", "wape": 35.0},
            {"intermittency_class": "intermittent", "model_name": "Croston", "wape": 18.0},
        ])

        selector = ModelSelector()
        policy = selector.build_segment_policy(
            global_benchmark=benchmark_table,
            segment_eval_df=seg_eval,
            segment_col="intermittency_class",
        )

        assert policy.get_model_for_entity(segment="smooth") in ["LightGBM", "Seasonal Naive"]
        assert policy.get_model_for_entity(segment="intermittent") == "Croston"
        # Unknown segment falls back to global champion
        assert policy.get_model_for_entity(segment="unknown") == policy.global_champion
