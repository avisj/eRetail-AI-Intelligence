"""Tests for Forecast Error Metrics."""

import numpy as np
import pytest

from commerce_ai.analytics.metrics import (
    mae,
    rmse,
    mape,
    wape,
    bias,
    evaluate_forecast,
)


class TestForecastMetrics:
    def test_mae_and_rmse_known_values(self):
        y_true = [10.0, 20.0, 30.0]
        y_pred = [12.0, 18.0, 34.0]
        # Errors: |2|, |-2|, |4| -> mean abs = (2 + 2 + 4) / 3 = 8 / 3 = 2.6667
        # Squared: 4 + 4 + 16 = 24 -> mean sq = 8 -> sqrt(8) = 2.8284

        assert np.isclose(mae(y_true, y_pred), 8.0 / 3.0)
        assert np.isclose(rmse(y_true, y_pred), np.sqrt(8.0))

    def test_wape_calculation(self):
        y_true = [100.0, 200.0]
        y_pred = [110.0, 180.0]
        # Sum actual = 300
        # Sum abs errors = |10| + |-20| = 30
        # WAPE = (30 / 300) * 100 = 10.0%
        assert np.isclose(wape(y_true, y_pred), 10.0)

    def test_wape_zero_actuals_safe_handling(self):
        y_true = [0.0, 0.0]
        y_pred = [0.0, 0.0]
        assert wape(y_true, y_pred) == 0.0

    def test_mape_zero_handling(self):
        # Time series with zero demand points (common in intermittent retail)
        y_true = [100.0, 0.0, 50.0]
        y_pred = [110.0, 5.0, 45.0]

        # With ignore_zero_actuals=True, only 100.0 (10% err) and 50.0 (10% err) are evaluated -> mean = 10.0%
        result = mape(y_true, y_pred, ignore_zero_actuals=True)
        assert np.isclose(result, 10.0)

    def test_bias_direction(self):
        y_true = [100.0, 100.0]
        # Systemic over-forecast: sum(y_pred - y_true) = 20 -> +10.0%
        y_over = [110.0, 110.0]
        assert np.isclose(bias(y_true, y_over), 10.0)

        # Systemic under-forecast: sum(y_pred - y_true) = -30 -> -15.0%
        y_under = [85.0, 85.0]
        assert np.isclose(bias(y_true, y_under), -15.0)

    def test_evaluate_forecast_bundle(self):
        y_true = [50.0, 100.0, 150.0]
        y_pred = [55.0, 90.0, 160.0]
        res = evaluate_forecast(y_true, y_pred)

        assert "mae" in res
        assert "rmse" in res
        assert "mape" in res
        assert "wape" in res
        assert "bias" in res
        assert all(isinstance(v, (int, float)) for v in res.values())
