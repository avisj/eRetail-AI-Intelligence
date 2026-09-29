"""Heuristic and Statistical Baseline Forecasting Models.

Implements standard benchmark models:
- NaiveModel: Last observed value projected forward
- SeasonalNaiveModel: Repeating seasonal cycle (default: 7-day weekly seasonality)
- MovingAverageModel: Rolling mean of recent eligible historical demand (7, 14, 30 days)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from commerce_ai.forecasting.base import ForecastModel, ForecastOutput, ForecastMetadata


class NaiveModel(ForecastModel):
    """Last-Value Naive Baseline: forecast[t+h] = y_last."""

    name: str = "Naive"
    version: str = "1.0.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_value: float = 0.0
        self.last_date: Optional[pd.Timestamp] = None

    def fit(
        self,
        history: pd.DataFrame,
        target_col: str = "units_sold",
        date_col: str = "date",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "NaiveModel":
        start_time = time.perf_counter()
        self._extract_entity_ids(history)

        clean_history = history.sort_values(date_col).copy()
        if "forecast_training_eligible" in clean_history.columns:
            eligible = clean_history[clean_history["forecast_training_eligible"]]
            if not eligible.empty:
                clean_history = eligible

        if clean_history.empty or target_col not in clean_history.columns:
            self.last_value = 0.0
            self.last_date = pd.Timestamp.now()
        else:
            self.last_value = float(clean_history[target_col].iloc[-1])
            self.last_date = pd.to_datetime(clean_history[date_col].iloc[-1])

        self.last_value = max(0.0, self.last_value)
        self.is_fitted = True

        self._metadata = ForecastMetadata(
            model_name=self.name,
            model_version=self.version,
            training_observations=len(history),
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")

        future_dates = self._build_future_dates(self.last_date, horizon)
        forecast_array = np.full(horizon, self.last_value, dtype=float)

        return ForecastOutput(
            sku_id=self._sku_id,
            warehouse_id=self._warehouse_id,
            forecast_dates=future_dates,
            forecast_units=forecast_array,
            model_name=self.name,
            model_version=self.version,
            confidence_level=confidence_level,
            metadata=self._metadata,
        )


class SeasonalNaiveModel(ForecastModel):
    """Seasonal Naive Baseline: forecast[t+h] = y[t + h - seasonal_period]."""

    name: str = "Seasonal Naive"
    version: str = "1.0.0"

    def __init__(self, seasonal_period: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.seasonal_period = seasonal_period
        self.name = kwargs.get("name", "Seasonal Naive")
        self.seasonal_pattern: np.ndarray = np.array([])
        self.last_date: Optional[pd.Timestamp] = None

    def fit(
        self,
        history: pd.DataFrame,
        target_col: str = "units_sold",
        date_col: str = "date",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SeasonalNaiveModel":
        start_time = time.perf_counter()
        self._extract_entity_ids(history)

        clean_history = history.sort_values(date_col).copy()
        if clean_history.empty:
            self.seasonal_pattern = np.zeros(self.seasonal_period)
            self.last_date = pd.Timestamp.now()
        else:
            self.last_date = pd.to_datetime(clean_history[date_col].iloc[-1])
            values = clean_history[target_col].values
            if len(values) >= self.seasonal_period:
                self.seasonal_pattern = values[-self.seasonal_period:].astype(float)
            else:
                # If history is shorter than period, repeat available values
                reps = int(np.ceil(self.seasonal_period / len(values)))
                self.seasonal_pattern = np.tile(values, reps)[:self.seasonal_period].astype(float)

        # Non-negative clipping
        self.seasonal_pattern = np.maximum(0.0, self.seasonal_pattern)
        self.is_fitted = True

        self._metadata = ForecastMetadata(
            model_name=self.name,
            model_version=self.version,
            training_observations=len(history),
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")

        future_dates = self._build_future_dates(self.last_date, horizon)
        reps = int(np.ceil(horizon / self.seasonal_period))
        forecast_array = np.tile(self.seasonal_pattern, reps)[:horizon]

        return ForecastOutput(
            sku_id=self._sku_id,
            warehouse_id=self._warehouse_id,
            forecast_dates=future_dates,
            forecast_units=forecast_array,
            model_name=self.name,
            model_version=self.version,
            confidence_level=confidence_level,
            metadata=self._metadata,
        )


class MovingAverageModel(ForecastModel):
    """Moving Average Baseline: forecast = mean(last window eligible observations)."""

    name: str = "MovingAverage"
    version: str = "1.0.0"

    def __init__(self, window: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.name = kwargs.get("name", f"Moving Average ({window}d)")
        self.mean_value: float = 0.0
        self.last_date: Optional[pd.Timestamp] = None

    def fit(
        self,
        history: pd.DataFrame,
        target_col: str = "units_sold",
        date_col: str = "date",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MovingAverageModel":
        start_time = time.perf_counter()
        self._extract_entity_ids(history)

        clean_history = history.sort_values(date_col).copy()
        if "forecast_training_eligible" in clean_history.columns:
            eligible = clean_history[clean_history["forecast_training_eligible"]]
            if not eligible.empty:
                clean_history = eligible

        if clean_history.empty:
            self.mean_value = 0.0
            self.last_date = pd.Timestamp.now()
        else:
            self.last_date = pd.to_datetime(clean_history[date_col].iloc[-1])
            recent_vals = clean_history[target_col].tail(self.window).values
            self.mean_value = float(np.mean(recent_vals)) if len(recent_vals) > 0 else 0.0

        self.mean_value = max(0.0, self.mean_value)
        self.is_fitted = True

        self._metadata = ForecastMetadata(
            model_name=self.name,
            model_version=self.version,
            training_observations=len(history),
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")

        future_dates = self._build_future_dates(self.last_date, horizon)
        forecast_array = np.full(horizon, self.mean_value, dtype=float)

        return ForecastOutput(
            sku_id=self._sku_id,
            warehouse_id=self._warehouse_id,
            forecast_dates=future_dates,
            forecast_units=forecast_array,
            model_name=self.name,
            model_version=self.version,
            confidence_level=confidence_level,
            metadata=self._metadata,
        )
