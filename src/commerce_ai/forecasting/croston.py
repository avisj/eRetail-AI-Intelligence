"""Croston Method for Intermittent Demand Forecasting.

Implements the classical Croston (1972) decomposition and the Syntetos-Boylan
Approximation (SBA, 2005) for intermittent, lumpy demand series.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from commerce_ai.forecasting.base import ForecastModel, ForecastOutput, ForecastMetadata


class CrostonModel(ForecastModel):
    """Croston's Method for intermittent demand forecasting.

    Separates demand into two components:
    1. Demand Magnitude (z): Exponential smoothing of non-zero transaction sizes.
    2. Inter-arrival Interval (p): Exponential smoothing of time between non-zero transactions.

    Forecast rate:
        Standard: y_hat = z / p
        SBA Variant: y_hat = (1 - alpha / 2) * (z / p)
    """

    name: str = "Croston"
    version: str = "1.0.0"

    def __init__(
        self,
        alpha: float = 0.1,
        variant: str = "sba",  # "croston" or "sba" (Syntetos-Boylan Approximation)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.variant = variant.lower()
        self.name = kwargs.get("name", "Croston" if self.variant == "croston" else "Croston")
        self.demand_rate: float = 0.0
        self.last_date: Optional[pd.Timestamp] = None

    def fit(
        self,
        history: pd.DataFrame,
        target_col: str = "units_sold",
        date_col: str = "date",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CrostonModel":
        start_time = time.perf_counter()
        self._extract_entity_ids(history)

        clean_history = history.sort_values(date_col).copy()
        if "forecast_training_eligible" in clean_history.columns:
            eligible = clean_history[clean_history["forecast_training_eligible"]]
            if not eligible.empty:
                clean_history = eligible

        if clean_history.empty or target_col not in clean_history.columns:
            self.demand_rate = 0.0
            self.last_date = pd.Timestamp.now()
            self.is_fitted = True
            self._metadata = ForecastMetadata(
                model_name=self.name,
                model_version=self.version,
                training_observations=0,
                runtime_seconds=time.perf_counter() - start_time,
                status="INSUFFICIENT_HISTORY",
                error_message="Empty history provided to Croston model.",
            )
            return self

        self.last_date = pd.to_datetime(clean_history[date_col].iloc[-1])
        y = clean_history[target_col].values.astype(float)

        non_zero_indices = np.where(y > 0)[0]

        if len(non_zero_indices) == 0:
            # Completely dormant series: 0 demand everywhere
            self.demand_rate = 0.0
        elif len(non_zero_indices) == 1:
            # Single non-zero point: rate is magnitude / total_days
            self.demand_rate = float(y[non_zero_indices[0]] / len(y))
        else:
            # Initialize z (magnitude) and p (interval) from first non-zero demand
            first_idx = non_zero_indices[0]
            z = float(y[first_idx])
            p = float(max(1.0, first_idx + 1))
            q = 1.0  # Periods counter since last non-zero demand

            # Iterate through history
            for t in range(first_idx + 1, len(y)):
                if y[t] > 0:
                    z = self.alpha * float(y[t]) + (1.0 - self.alpha) * z
                    p = self.alpha * q + (1.0 - self.alpha) * p
                    q = 1.0
                else:
                    q += 1.0

            # Calculate forecast rate
            if p > 0:
                raw_rate = z / p
                if self.variant == "sba":
                    # Syntetos-Boylan de-biasing adjustment
                    self.demand_rate = float((1.0 - self.alpha / 2.0) * raw_rate)
                else:
                    self.demand_rate = float(raw_rate)
            else:
                self.demand_rate = float(z)

        self.demand_rate = max(0.0, self.demand_rate)
        self.is_fitted = True

        self._metadata = ForecastMetadata(
            model_name=f"{self.name}_{self.variant.upper()}",
            model_version=self.version,
            training_observations=len(y),
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
        forecast_array = np.full(horizon, self.demand_rate, dtype=float)

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
