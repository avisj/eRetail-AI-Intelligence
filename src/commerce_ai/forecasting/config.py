"""Configuration Models for Forecasting and Backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ForecastConfig:
    """Configuration for demand forecasting models and dataset construction."""

    horizon: int = 30  # Forecast horizon in days
    min_history: int = 60  # Minimum historical days required to attempt training
    target_column: str = "units_sold"  # Target column to forecast
    date_column: str = "date"  # Temporal column
    entity_columns: List[str] = field(default_factory=lambda: ["sku_id", "warehouse_id"])
    clip_negative: bool = True  # Prevent negative forecasted demand
    filter_stockouts: bool = True  # Fit only on forecast_training_eligible == True observations
    confidence_level: Optional[float] = 0.95  # Confidence level for intervals
    random_seed: int = 42  # Deterministic seed


@dataclass
class BacktestConfig:
    """Configuration for time-series rolling-origin backtesting."""

    horizon: int = 30  # Forecast evaluation horizon per fold
    folds: int = 3  # Number of rolling-origin folds
    min_history: int = 90  # Initial historical window for Fold 1
    step_size: Optional[int] = None  # Step forward per fold (defaults to horizon for non-overlapping folds)
    target_column: str = "units_sold"
    date_column: str = "date"
    entity_columns: List[str] = field(default_factory=lambda: ["sku_id", "warehouse_id"])
    filter_stockouts_for_training: bool = True  # Fit models on clean data
    evaluate_on_unconstrained_only: bool = True  # Evaluate models on unconstrained actuals
