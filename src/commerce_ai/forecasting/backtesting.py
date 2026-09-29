"""Time-Series Backtesting Framework with Rolling-Origin Cross Validation.

Executes leakage-free out-of-sample backtests across multiple temporal folds,
models, and SKU-warehouse series with granular error isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.forecasting.base import ForecastModel
from commerce_ai.forecasting.config import BacktestConfig
from commerce_ai.forecasting.datasets import generate_rolling_origin_folds


@dataclass
class BacktestResult:
    """Standardized output container for rolling-origin backtesting."""

    predictions_df: pd.DataFrame
    diagnostics_df: pd.DataFrame
    config: BacktestConfig
    models_evaluated: List[str]

    @property
    def total_evaluations(self) -> int:
        return len(self.predictions_df)

    @property
    def total_failures(self) -> int:
        if self.diagnostics_df.empty:
            return 0
        return int((self.diagnostics_df["status"] != "SUCCESS").sum())

    @property
    def failure_rate(self) -> float:
        if self.diagnostics_df.empty:
            return 0.0
        return float(self.total_failures / max(1, len(self.diagnostics_df)))


class RollingOriginBacktester:
    """Orchestrates rolling-origin time-series cross validation across models.

    Key Features:
        - Strict chronological partition: zero lookahead bias or future leakage.
        - Series-level error containment: a single failure does not crash the run.
        - Fine-grained timing and execution diagnostics.
        - Universal support for all models conforming to ForecastModel.
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        models: Optional[Dict[str, Union[ForecastModel, Callable[[], ForecastModel]]]] = None,
    ):
        self.config = config or BacktestConfig()
        self.models = models or {}

    def add_model(
        self,
        name: str,
        model_factory: Union[ForecastModel, Callable[[], ForecastModel]],
    ) -> None:
        """Register a model instance or factory function."""
        self.models[name] = model_factory

    def _instantiate_model(
        self,
        model_obj: Union[ForecastModel, Callable[[], ForecastModel]],
    ) -> ForecastModel:
        """Helper to create a fresh model instance for each fold/series."""
        if callable(model_obj) and not isinstance(model_obj, ForecastModel):
            return model_obj()
        # If it's already an instance, instantiate a fresh copy of its class with same kwargs
        cls = model_obj.__class__
        return cls(**getattr(model_obj, "params", {}))

    def run(
        self,
        data: pd.DataFrame,
        series_keys: Optional[List[Tuple[str, str]]] = None,
        target_col: str = "units_sold",
        date_col: str = "date",
    ) -> BacktestResult:
        """Run backtesting across all folds, series, and models.

        Args:
            data: Unified demand or feature dataset.
            series_keys: Optional list of (sku_id, warehouse_id) pairs to evaluate.
                         If None, discovers unique series in `data`.
            target_col: Column name representing demand volume.
            date_col: Date column name.

        Returns:
            BacktestResult with predictions and diagnostics DataFrames.
        """
        if not self.models:
            raise ValueError("No models registered in backtester.")

        # Ensure datetime sorting
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Discover series if not specified
        if series_keys is None:
            if "sku_id" in df.columns and "warehouse_id" in df.columns:
                series_keys = (
                    df[["sku_id", "warehouse_id"]]
                    .drop_duplicates()
                    .to_records(index=False)
                    .tolist()
                )
            else:
                series_keys = [("ALL", "ALL")]

        # Generate folds
        folds = generate_rolling_origin_folds(df, self.config)

        all_preds: List[Dict[str, Any]] = []
        diagnostics: List[Dict[str, Any]] = []

        for fold in folds:
            fold_idx = fold["fold_index"]
            cutoff_date = fold["cutoff_date"]
            train_df = fold["train_df"]
            test_df = fold["test_df"]

            for sku_id, wh_id in series_keys:
                # Filter train and test slices
                if sku_id != "ALL" and wh_id != "ALL":
                    s_train = train_df[
                        (train_df["sku_id"] == sku_id) & (train_df["warehouse_id"] == wh_id)
                    ].copy()
                    s_test = test_df[
                        (test_df["sku_id"] == sku_id) & (test_df["warehouse_id"] == wh_id)
                    ].copy()
                else:
                    s_train = train_df.copy()
                    s_test = test_df.copy()

                if s_train.empty:
                    continue

                for model_name, model_obj in self.models.items():
                    start_time = time.perf_counter()
                    try:
                        fresh_model = self._instantiate_model(model_obj)
                        fresh_model.fit(
                            s_train,
                            target_col=target_col,
                            date_col=date_col,
                        )
                        output = fresh_model.predict(
                            horizon=self.config.horizon,
                            future_features=s_test,
                        )
                        runtime = time.perf_counter() - start_time

                        # Check model status
                        model_status = "SUCCESS"
                        if output.metadata and output.metadata.status != "SUCCESS":
                            model_status = output.metadata.status

                        diagnostics.append({
                            "fold_index": fold_idx,
                            "cutoff_date": cutoff_date,
                            "model_name": model_name,
                            "sku_id": sku_id,
                            "warehouse_id": wh_id,
                            "status": model_status,
                            "runtime_seconds": runtime,
                            "error_message": output.metadata.error_message if output.metadata else None,
                        })

                        # Format prediction records matched with ground truth test actuals
                        pred_df = output.to_dataframe()
                        pred_df["date"] = pd.to_datetime(pred_df["forecast_date"])
                        
                        # Merge with actuals
                        merged = pd.merge(
                            pred_df,
                            s_test[[date_col, target_col]],
                            left_on="date",
                            right_on=date_col,
                            how="left",
                        )
                        merged[target_col] = merged[target_col].fillna(0.0)

                        for _, row in merged.iterrows():
                            all_preds.append({
                                "fold_index": fold_idx,
                                "cutoff_date": cutoff_date,
                                "date": row["forecast_date"],
                                "sku_id": sku_id,
                                "warehouse_id": wh_id,
                                "actual_units": float(row[target_col]),
                                "forecast_units": float(row["forecast_units"]),
                                "lower_bound": row.get("lower_bound"),
                                "upper_bound": row.get("upper_bound"),
                                "model_name": model_name,
                                "runtime_seconds": runtime / max(1, self.config.horizon),
                            })

                    except Exception as exc:
                        runtime = time.perf_counter() - start_time
                        diagnostics.append({
                            "fold_index": fold_idx,
                            "cutoff_date": cutoff_date,
                            "model_name": model_name,
                            "sku_id": sku_id,
                            "warehouse_id": wh_id,
                            "status": "FAILED",
                            "runtime_seconds": runtime,
                            "error_message": str(exc),
                        })

        predictions_df = pd.DataFrame(all_preds)
        diagnostics_df = pd.DataFrame(diagnostics)

        return BacktestResult(
            predictions_df=predictions_df,
            diagnostics_df=diagnostics_df,
            config=self.config,
            models_evaluated=list(self.models.keys()),
        )
