"""TimesFM Foundation Model Forecasting Adapter.

Provides integration with Google TimesFM (Time Series Foundation Model)
while safely handling system environment constraints (e.g. Darwin/macOS paxml limitations)
with standardized status reporting and mock testability.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from commerce_ai.forecasting.base import (
    ForecastModel,
    ForecastOutput,
    ForecastMetadata,
)


class TimesFMForecastModel(ForecastModel):
    """Adapter for Google TimesFM time-series foundation model.

    Designed to follow the universal ForecastModel interface.
    Gracefully detects availability of `timesfm` package and underlying dependencies
    (such as paxml/jax). If unavailable in the current runtime environment (e.g. macOS Darwin),
    it reports TIMESFM_UNAVAILABLE status without corrupting pipeline execution.
    """

    name: str = "TimesFM"
    version: str = "1.0.0"

    def __init__(
        self,
        context_len: int = 512,
        horizon_len: int = 30,
        backend: str = "cpu",
        mock_predictor: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.backend = backend
        self.mock_predictor = mock_predictor
        self._history_values: Optional[np.ndarray] = None
        self._last_date: Optional[pd.Timestamp] = None
        self._is_available: bool = False
        self._timesfm_instance = None
        self._init_error: Optional[str] = None

        self._check_availability()

    def _check_availability(self) -> None:
        """Verify whether timesfm library can be imported and initialized."""
        if self.mock_predictor is not None:
            self._is_available = True
            return

        try:
            import timesfm  # noqa: F401
            self._is_available = True
        except ImportError as e:
            self._is_available = False
            self._init_error = (
                f"Google TimesFM is unavailable in this environment: {str(e)}. "
                "TimesFM requires Linux x86_64 TPU/GPU with paxml."
            )
        except Exception as e:
            self._is_available = False
            self._init_error = f"Error initializing TimesFM: {str(e)}"

    @property
    def is_available(self) -> bool:
        """Return True if TimesFM is operational in this environment."""
        return self._is_available

    def fit(
        self,
        history: pd.DataFrame,
        target_col: str = "units_sold",
        date_col: str = "date",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TimesFMForecastModel":
        """Prepares history context for zero-shot TimesFM inference."""
        start_time = time.perf_counter()
        self._extract_entity_ids(history)

        clean_history = history.sort_values(date_col).copy()
        excluded_count = 0
        if "forecast_training_eligible" in clean_history.columns:
            ineligible_mask = ~clean_history["forecast_training_eligible"].astype(bool)
            excluded_count = int(ineligible_mask.sum())
            clean_history = clean_history[~ineligible_mask].reset_index(drop=True)

        if not self._is_available and self.mock_predictor is None:
            self.is_fitted = True
            self._last_date = pd.to_datetime(clean_history[date_col].iloc[-1]) if not clean_history.empty else pd.Timestamp.now()
            self._metadata = ForecastMetadata(
                model_name=self.name,
                model_version=self.version,
                training_observations=len(clean_history),
                excluded_constrained_observations=excluded_count,
                runtime_seconds=time.perf_counter() - start_time,
                status="TIMESFM_UNAVAILABLE",
                error_code="TIMESFM_DEPENDENCY_MISSING",
                error_message=self._init_error,
            )
            return self

        # Extract continuous demand series
        if clean_history.empty or target_col not in clean_history.columns:
            self._history_values = np.zeros(0)
            self._last_date = pd.Timestamp.now()
        else:
            self._history_values = clean_history[target_col].astype(float).values
            self._last_date = pd.to_datetime(clean_history[date_col].iloc[-1])

        self.is_fitted = True
        self._metadata = ForecastMetadata(
            model_name=self.name,
            model_version=self.version,
            training_observations=len(clean_history),
            excluded_constrained_observations=excluded_count,
            runtime_seconds=time.perf_counter() - start_time,
            status="SUCCESS",
        )
        return self

    def predict(
        self,
        horizon: int,
        future_features: Optional[pd.DataFrame] = None,
        confidence_level: Optional[float] = None,
    ) -> ForecastOutput:
        """Produce forecast using zero-shot foundation model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")

        future_dates = self._build_future_dates(self._last_date, horizon)

        if not self._is_available and self.mock_predictor is None:
            # Fallback zero-array with TIMESFM_UNAVAILABLE metadata
            return ForecastOutput(
                sku_id=self._sku_id,
                warehouse_id=self._warehouse_id,
                forecast_dates=future_dates,
                forecast_units=np.zeros(horizon, dtype=float),
                model_name=self.name,
                model_version=self.version,
                metadata=self._metadata,
            )

        # If mock predictor supplied for testing
        if self.mock_predictor is not None:
            raw_forecasts = self.mock_predictor.predict(
                context=self._history_values,
                horizon=horizon,
            )
            forecasts = np.maximum(0.0, np.array(raw_forecasts, dtype=float))
        else:
            # Native timesfm execution
            context = self._history_values[-self.context_len:] if len(self._history_values) > self.context_len else self._history_values
            # timesfm.forecast accepts list of 1D arrays
            point_forecast, _ = self._timesfm_instance.forecast(
                [context],
                freq=[0],  # 0 indicates daily high-frequency
            )
            forecasts = np.maximum(0.0, point_forecast[0][:horizon])

        return ForecastOutput(
            sku_id=self._sku_id,
            warehouse_id=self._warehouse_id,
            forecast_dates=future_dates,
            forecast_units=forecasts,
            model_name=self.name,
            model_version=self.version,
            metadata=self._metadata,
        )
