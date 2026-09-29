"""Forecasting Engine Base Interface and Data Contracts.

Defines the universal forecasting interface (ForecastModel) and standardized
output contracts (ForecastOutput, ForecastRecord, ForecastMetadata).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class ForecastRecord:
    """Standardized single-date forecast output record."""

    forecast_date: str
    sku_id: str
    warehouse_id: str
    forecast_units: float
    model_name: str
    model_version: str
    prediction_run_id: str
    forecast_generated_at: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    confidence_level: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast_date": self.forecast_date,
            "sku_id": self.sku_id,
            "warehouse_id": self.warehouse_id,
            "forecast_units": self.forecast_units,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction_run_id": self.prediction_run_id,
            "forecast_generated_at": self.forecast_generated_at,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_level": self.confidence_level,
        }


@dataclass
class ForecastMetadata:
    """Audit metadata tracking training, execution, and data provenance."""

    model_name: str
    model_version: str
    training_start_date: Optional[str] = None
    training_end_date: Optional[str] = None
    forecast_start_date: Optional[str] = None
    forecast_end_date: Optional[str] = None
    horizon: int = 30
    training_observations: int = 0
    excluded_constrained_observations: int = 0
    runtime_seconds: float = 0.0
    status: str = "SUCCESS"  # SUCCESS, FAILED, INSUFFICIENT_HISTORY, TIMESFM_UNAVAILABLE
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "training_start_date": self.training_start_date,
            "training_end_date": self.training_end_date,
            "forecast_start_date": self.forecast_start_date,
            "forecast_end_date": self.forecast_end_date,
            "horizon": self.horizon,
            "training_observations": self.training_observations,
            "excluded_constrained_observations": self.excluded_constrained_observations,
            "runtime_seconds": self.runtime_seconds,
            "status": self.status,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass
class ForecastOutput:
    """Standardized batch forecast result for a time-series entity."""

    sku_id: str
    warehouse_id: str
    forecast_dates: List[str]
    forecast_units: np.ndarray
    model_name: str
    model_version: str = "1.0.0"
    prediction_run_id: str = field(default_factory=lambda: f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    forecast_generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    confidence_level: Optional[float] = None
    metadata: Optional[ForecastMetadata] = None

    def to_records(self) -> List[ForecastRecord]:
        """Convert output into a list of standardized ForecastRecords."""
        records = []
        for i, dt in enumerate(self.forecast_dates):
            low = float(self.lower_bound[i]) if self.lower_bound is not None else None
            high = float(self.upper_bound[i]) if self.upper_bound is not None else None
            records.append(
                ForecastRecord(
                    forecast_date=str(dt),
                    sku_id=self.sku_id,
                    warehouse_id=self.warehouse_id,
                    forecast_units=float(self.forecast_units[i]),
                    model_name=self.model_name,
                    model_version=self.model_version,
                    prediction_run_id=self.prediction_run_id,
                    forecast_generated_at=self.forecast_generated_at,
                    lower_bound=low,
                    upper_bound=high,
                    confidence_level=self.confidence_level,
                )
            )
        return records

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast output to a flat pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.to_records()])


class ForecastModel(ABC):
    """Abstract Base Class for all forecasting models.

    Ensures unified lifecycle:
        model.fit(history, metadata)
        output = model.predict(horizon, future_features)
    """

    name: str = "BaseModel"
    version: str = "1.0.0"

    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted: bool = False
        self._last_history: Optional[pd.DataFrame] = None
        self._sku_id: str = "UNKNOWN"
        self._warehouse_id: str = "UNKNOWN"
        self._metadata: Optional[ForecastMetadata] = None

    @abstractmethod
    def fit(
        self,
        history: pd.DataFrame,
        target_col: str = "units_sold",
        date_col: str = "date",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ForecastModel":
        """Fit model to historical data."""
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int,
        future_features: Optional[pd.DataFrame] = None,
        confidence_level: Optional[float] = None,
    ) -> ForecastOutput:
        """Generate out-of-sample forecast for the specified horizon."""
        pass

    def _extract_entity_ids(self, history: pd.DataFrame) -> None:
        """Helper to extract SKU and Warehouse ID from history."""
        if "sku_id" in history.columns and not history["sku_id"].empty:
            self._sku_id = str(history["sku_id"].iloc[0])
        if "warehouse_id" in history.columns and not history["warehouse_id"].empty:
            self._warehouse_id = str(history["warehouse_id"].iloc[0])

    def _build_future_dates(self, last_date: pd.Timestamp, horizon: int) -> List[str]:
        """Construct continuous daily dates starting the day after last_date."""
        start = last_date + pd.Timedelta(days=1)
        future_range = pd.date_range(start=start, periods=horizon, freq="D")
        return [d.strftime("%Y-%m-%d") for d in future_range]
