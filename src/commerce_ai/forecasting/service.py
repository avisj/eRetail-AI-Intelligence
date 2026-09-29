"""Unified Forecasting Service for Single-Series and Batch Operational Workflows.

Provides high-level APIs for generating out-of-sample demand forecasts with
automatic model routing, stockout protection, input validation, and audit tracking.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from commerce_ai.forecasting.base import (
    ForecastModel,
    ForecastOutput,
    ForecastRecord,
    ForecastMetadata,
)
from commerce_ai.forecasting.config import ForecastConfig
from commerce_ai.forecasting.baselines import (
    NaiveModel,
    SeasonalNaiveModel,
    MovingAverageModel,
)
from commerce_ai.forecasting.exponential_smoothing import ExponentialSmoothingModel
from commerce_ai.forecasting.croston import CrostonModel
from commerce_ai.forecasting.lightgbm_model import LightGBMForecastModel
from commerce_ai.forecasting.timesfm import TimesFMForecastModel
from commerce_ai.forecasting.selection import ModelSelectionPolicy


class ForecastService:
    """High-level forecasting service coordinating inference across models."""

    def __init__(
        self,
        default_model: str = "LightGBM",
        config: Optional[ForecastConfig] = None,
        custom_models: Optional[Dict[str, Union[ForecastModel, Callable[[], ForecastModel]]]] = None,
    ):
        self.default_model_name = default_model
        self.config = config or ForecastConfig()
        self._model_registry: Dict[str, Union[ForecastModel, Callable[[], ForecastModel]]] = {
            "Naive": lambda: NaiveModel(),
            "Seasonal Naive": lambda: SeasonalNaiveModel(season_length=7),
            "Moving Average (7d)": lambda: MovingAverageModel(window=7),
            "Moving Average (14d)": lambda: MovingAverageModel(window=14),
            "Moving Average (30d)": lambda: MovingAverageModel(window=30),
            "Exponential Smoothing": lambda: ExponentialSmoothingModel(),
            "Croston": lambda: CrostonModel(variant="sba"),
            "LightGBM": lambda: LightGBMForecastModel(),
            "TimesFM": lambda: TimesFMForecastModel(),
        }
        if custom_models:
            self._model_registry.update(custom_models)

    def register_model(
        self,
        name: str,
        factory: Union[ForecastModel, Callable[[], ForecastModel]],
    ) -> None:
        """Register a new forecast model in the service."""
        self._model_registry[name] = factory

    def _get_model_instance(self, model_name: str) -> ForecastModel:
        """Instantiate a model by name from registry."""
        if model_name not in self._model_registry:
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {list(self._model_registry.keys())}"
            )
        entry = self._model_registry[model_name]
        if callable(entry) and not isinstance(entry, ForecastModel):
            return entry()
        cls = entry.__class__
        return cls(**getattr(entry, "params", {}))

    def forecast_series(
        self,
        history: pd.DataFrame,
        horizon: Optional[int] = None,
        model_name: Optional[str] = None,
        future_features: Optional[pd.DataFrame] = None,
        confidence_level: Optional[float] = None,
        target_col: str = "units_sold",
        date_col: str = "date",
    ) -> ForecastOutput:
        """Generate forecasts for an individual SKU-warehouse series.

        Args:
            history: Historical DataFrame for a single series.
            horizon: Forecast horizon in days (defaults to config.horizon).
            model_name: Optional specific model name; defaults to default_model_name.
            future_features: Optional future covariate DataFrame.
            confidence_level: Optional prediction interval confidence (e.g. 0.95).
            target_col: Name of demand target column.
            date_col: Name of date column.

        Returns:
            ForecastOutput with predictions, intervals, and metadata.
        """
        start_time = time.perf_counter()
        h = horizon or self.config.horizon
        chosen_model_name = model_name or self.default_model_name
        conf = confidence_level if confidence_level is not None else self.config.confidence_level

        # Input validation
        if history.empty:
            raise ValueError("Input history DataFrame is empty.")
        if target_col not in history.columns:
            raise ValueError(f"Target column '{target_col}' not found in history DataFrame.")
        if date_col not in history.columns:
            raise ValueError(f"Date column '{date_col}' not found in history DataFrame.")

        model = self._get_model_instance(chosen_model_name)
        model.fit(
            history=history,
            target_col=target_col,
            date_col=date_col,
        )

        output = model.predict(
            horizon=h,
            future_features=future_features,
            confidence_level=conf,
        )

        # Enforce non-negative demand contract
        output.forecast_units = np.maximum(0.0, output.forecast_units)
        if output.lower_bound is not None:
            output.lower_bound = np.maximum(0.0, output.lower_bound)
        if output.upper_bound is not None:
            output.upper_bound = np.maximum(output.lower_bound if output.lower_bound is not None else 0.0, output.upper_bound)

        # Ensure metadata audit
        if output.metadata is None:
            output.metadata = ForecastMetadata(
                model_name=output.model_name,
                model_version=output.model_version,
                training_observations=len(history),
                runtime_seconds=time.perf_counter() - start_time,
                status="SUCCESS",
            )

        return output

    def forecast_batch(
        self,
        dataset: pd.DataFrame,
        horizon: Optional[int] = None,
        policy: Optional[ModelSelectionPolicy] = None,
        model_name: Optional[str] = None,
        target_col: str = "units_sold",
        date_col: str = "date",
        series_cols: Optional[List[str]] = None,
        confidence_level: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run batch forecasting across multiple time-series entities.

        Args:
            dataset: Multi-series DataFrame containing historical records.
            horizon: Number of days forward to forecast.
            policy: Optional ModelSelectionPolicy routing series to champion models.
            model_name: Optional fixed model name override for all series.
            target_col: Demand volume column.
            date_col: Date column.
            series_cols: Columns defining entity key (default: ['sku_id', 'warehouse_id']).
            confidence_level: Optional interval confidence (e.g. 0.95).

        Returns:
            pd.DataFrame containing flat standardized ForecastRecords.
        """
        h = horizon or self.config.horizon
        groups = series_cols or ["sku_id", "warehouse_id"]

        all_records: List[ForecastRecord] = []
        run_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        grouped = dataset.groupby(groups)
        for name, series_df in grouped:
            if not isinstance(name, tuple):
                name = (name,)
            sku_id = str(name[0])
            wh_id = str(name[1]) if len(name) > 1 else "WH_01"

            # Route model
            if model_name:
                selected_model = model_name
            elif policy:
                segment = series_df["intermittency_class"].iloc[-1] if "intermittency_class" in series_df.columns else None
                selected_model = policy.get_model_for_entity(sku_id=sku_id, segment=segment)
            else:
                selected_model = self.default_model_name

            try:
                out = self.forecast_series(
                    history=series_df,
                    horizon=h,
                    model_name=selected_model,
                    confidence_level=confidence_level,
                    target_col=target_col,
                    date_col=date_col,
                )
                out.prediction_run_id = run_id
                all_records.extend(out.to_records())
            except Exception as e:
                # Fallback to simple baseline on error to ensure batch reliability
                fallback_model = NaiveModel()
                fallback_model.fit(series_df, target_col=target_col, date_col=date_col)
                out = fallback_model.predict(horizon=h)
                out.prediction_run_id = run_id
                if out.metadata:
                    out.metadata.status = "FALLBACK"
                    out.metadata.error_message = str(e)
                all_records.extend(out.to_records())

        return pd.DataFrame([r.to_dict() for r in all_records])
