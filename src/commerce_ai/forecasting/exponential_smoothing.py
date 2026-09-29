"""Exponential Smoothing Forecasting Model.

Uses Holt-Winters / Exponential Smoothing for trend and seasonal time series,
with robust fallback and failure containment for short or erratic histories.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as StatsmodelsExpSmoothing
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from commerce_ai.forecasting.base import ForecastModel, ForecastOutput, ForecastMetadata


class ExponentialSmoothingModel(ForecastModel):
    """Exponential Smoothing model with Holt-Winters trend/seasonal capabilities."""

    name: str = "ExponentialSmoothing"
    version: str = "1.0.0"

    def __init__(
        self,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: int = 7,
        damped_trend: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self._fitted_model: Optional[Any] = None
        self.last_date: Optional[pd.Timestamp] = None
        self._fallback_mean: float = 0.0

    def fit(
        self,
        history: pd.DataFrame,
        target_col: str = "units_sold",
        date_col: str = "date",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ExponentialSmoothingModel":
        start_time = time.perf_counter()
        self._extract_entity_ids(history)

        clean_history = history.sort_values(date_col).copy()
        if "forecast_training_eligible" in clean_history.columns:
            eligible = clean_history[clean_history["forecast_training_eligible"]]
            if not eligible.empty:
                clean_history = eligible

        if clean_history.empty or target_col not in clean_history.columns:
            self.last_date = pd.Timestamp.now()
            self._fallback_mean = 0.0
            self.is_fitted = True
            self._metadata = ForecastMetadata(
                model_name=self.name,
                model_version=self.version,
                training_observations=0,
                runtime_seconds=time.perf_counter() - start_time,
                status="INSUFFICIENT_HISTORY",
                error_message="Empty history provided to model.",
            )
            return self

        self.last_date = pd.to_datetime(clean_history[date_col].iloc[-1])
        y = clean_history[target_col].values.astype(float)
        self._fallback_mean = max(0.0, float(np.mean(y[-14:] if len(y) >= 14 else y)))

        if not STATSMODELS_AVAILABLE:
            self.is_fitted = True
            self._metadata = ForecastMetadata(
                model_name=self.name,
                model_version=self.version,
                training_observations=len(y),
                runtime_seconds=time.perf_counter() - start_time,
                status="FAILED",
                error_message="statsmodels library is not installed.",
            )
            return self

        status = "SUCCESS"
        error_msg = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Need at least 2 * seasonal_periods for seasonal model
                if self.seasonal and len(y) >= 2 * self.seasonal_periods:
                    model = StatsmodelsExpSmoothing(
                        y,
                        trend=self.trend,
                        seasonal=self.seasonal,
                        seasonal_periods=self.seasonal_periods,
                        damped_trend=self.damped_trend,
                        initialization_method="estimated",
                    )
                    self._fitted_model = model.fit()
                elif len(y) >= 10:
                    # Simple exponential smoothing with level
                    model = SimpleExpSmoothing(y, initialization_method="estimated")
                    self._fitted_model = model.fit()
                else:
                    status = "FALLBACK_TO_MEAN"
                    error_msg = f"Series too short ({len(y)} observations) for optimization."
            except Exception as e:
                # Contain fit exception gracefully
                self._fitted_model = None
                status = "FALLBACK_TO_MEAN"
                error_msg = f"ExponentialSmoothing optimization failed: {e}"

        self.is_fitted = True
        self._metadata = ForecastMetadata(
            model_name=self.name,
            model_version=self.version,
            training_observations=len(y),
            runtime_seconds=time.perf_counter() - start_time,
            status=status,
            error_message=error_msg,
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

        if self._fitted_model is not None:
            try:
                forecast_array = np.asarray(self._fitted_model.forecast(horizon), dtype=float)
            except Exception:
                forecast_array = np.full(horizon, self._fallback_mean, dtype=float)
        else:
            forecast_array = np.full(horizon, self._fallback_mean, dtype=float)

        # Demand must be non-negative
        forecast_array = np.maximum(0.0, np.nan_to_num(forecast_array, nan=self._fallback_mean))

        lower_bound = None
        upper_bound = None
        if confidence_level is not None:
            from scipy import stats
            residuals = getattr(self._fitted_model, "resid", None)
            res_std = float(np.std(residuals)) if residuals is not None and len(residuals) > 0 else 1.0
            alpha = 1.0 - float(confidence_level)
            z_score = float(stats.norm.ppf(1.0 - alpha / 2.0))
            lower_bound = np.maximum(0.0, forecast_array - z_score * res_std)
            upper_bound = np.maximum(0.0, forecast_array + z_score * res_std)

        return ForecastOutput(
            sku_id=self._sku_id,
            warehouse_id=self._warehouse_id,
            forecast_dates=future_dates,
            forecast_units=forecast_array,
            model_name=self.name,
            model_version=self.version,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            metadata=self._metadata,
        )
