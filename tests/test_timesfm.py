"""Unit Tests for TimesFM Foundation Model Adapter."""

import pytest
import numpy as np
import pandas as pd
from commerce_ai.forecasting.timesfm import TimesFMForecastModel


class MockTimesFMPredictor:
    """Mock engine simulating zero-shot TimesFM inference for test environments."""

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        # Returns simple persistence + trend heuristic
        base = float(context[-1]) if len(context) > 0 else 10.0
        return np.full(horizon, base)


class TestTimesFMAdapter:
    @pytest.fixture
    def sample_history(self):
        dates = pd.date_range("2023-01-01", periods=50)
        return pd.DataFrame({
            "date": dates,
            "sku_id": "SKU_TFM",
            "warehouse_id": "WH_01",
            "units_sold": [float(i) for i in range(50)],
            "forecast_training_eligible": True,
        })

    def test_timesfm_graceful_handling_when_native_lib_unavailable(self, sample_history):
        # Without mock_predictor, tests environment detection
        model = TimesFMForecastModel()
        model.fit(sample_history)

        out = model.predict(horizon=7)
        assert len(out.forecast_units) == 7

        if not model.is_available:
            # On macOS Darwin where paxml is unavailable
            assert out.metadata.status == "TIMESFM_UNAVAILABLE"
            assert out.metadata.error_code == "TIMESFM_DEPENDENCY_MISSING"
            assert np.allclose(out.forecast_units, 0.0)

    def test_timesfm_lifecycle_with_mock_predictor(self, sample_history):
        mock_engine = MockTimesFMPredictor()
        model = TimesFMForecastModel(mock_predictor=mock_engine)
        assert model.is_available

        model.fit(sample_history)
        assert model.is_fitted

        out = model.predict(horizon=10)
        assert len(out.forecast_units) == 10
        # Mock engine repeats context[-1] which is 49.0
        assert np.allclose(out.forecast_units, 49.0)
        assert out.metadata.status == "SUCCESS"
        assert (out.forecast_units >= 0.0).all()
