"""Forecasting & Accuracy Query Tools (Phase 7A).

Provides read-only query tools for Dashboards and eRetail Copilot:
- get_forecast: Forecast point estimates and prediction intervals over planning horizon
- get_forecast_accuracy: Standard error metrics (WAPE, MAE, RMSE, MAPE)
- get_forecast_bias: Directional tracking (over- vs under-forecasting bias)
- get_forecast_summary: Portfolio forecast health and model provenance
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.analytics.metrics import evaluate_forecast
from commerce_ai.forecasting.base import ForecastOutput
from commerce_ai.forecasting.service import ForecastService
from commerce_ai.query_layer.base import (
    build_empty_response,
    build_metadata,
    build_unavailable_response,
)
from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.schemas import (
    CalculationStatus,
    EvidenceReference,
    MetricResult,
    QueryDomain,
    QueryResponse,
    TableResult,
    TimeSeriesPoint,
    TimeSeriesResult,
)


def get_forecast(
    context: QueryContext,
    forecast_df: Optional[pd.DataFrame] = None,
    forecast_output: Optional[ForecastOutput] = None,
) -> QueryResponse:
    """Retrieve forward-looking demand forecast observations.
    
    If forecast outputs are not available, returns structured UNAVAILABLE status
    without fabricating synthetic numbers.
    """
    t0 = time.perf_counter()
    tool_name = "get_forecast"

    # If single ForecastOutput object is provided
    if forecast_output is not None:
        if hasattr(forecast_output, "to_records"):
            records = forecast_output.to_records()
        else:
            records = getattr(forecast_output, "records", [])

        default_sku = getattr(forecast_output, "sku_id", "UNKNOWN")
        points = [
            TimeSeriesPoint(
                date=getattr(rec, "forecast_date", getattr(rec, "date", "")),
                value=round(float(getattr(rec, "forecast_units", getattr(rec, "predicted_mean", 0.0))), 2),
                metric_name="forecast_demand",
                currency=None,
                dimensions={"sku_id": getattr(rec, "sku_id", default_sku)},
            )
            for rec in records
        ]
        model_name = getattr(forecast_output, "model_name", None)
        if not model_name and hasattr(forecast_output, "metadata") and forecast_output.metadata:
            model_name = getattr(forecast_output.metadata, "model_name", "ForecastEngine")
        model_name = model_name or "ForecastEngine"

        meta = build_metadata(
            tool_name=tool_name,
            domain=QueryDomain.FORECASTING,
            context=context,
            start_time=t0,
            confidence_provenance="MODEL_BASED",
            source_engine=f"commerce_ai.forecasting.{model_name}",
        )
        ts_res = TimeSeriesResult(
            metric_name="forecast_demand",
            display_name=f"Demand Forecast ({model_name})",
            time_grain=context.time_grain,
            points=points,
            unit="units",
            currency=None,
            metadata=meta,
        )
        return QueryResponse(
            query_id=meta.query_id,
            tool_name=tool_name,
            domain=meta.domain,
            status=CalculationStatus.SUCCESS,
            metadata=meta,
            time_series=ts_res,
        )

    # If forecast DataFrame is provided
    if forecast_df is not None and not forecast_df.empty:
        df = forecast_df.copy()
        if context.sku_ids and "sku_id" in df.columns:
            df = df[df["sku_id"].astype(str).isin(context.sku_ids)]
        if context.warehouse_ids and "warehouse_id" in df.columns:
            df = df[df["warehouse_id"].astype(str).isin(context.warehouse_ids)]

        if df.empty:
            return build_empty_response(
                tool_name=tool_name,
                domain=QueryDomain.FORECASTING,
                context=context,
                start_time=t0,
            )

        date_col = "date" if "date" in df.columns else "forecast_date"
        pred_col = "predicted_mean" if "predicted_mean" in df.columns else "forecast"

        if date_col in df.columns and pred_col in df.columns:
            points = [
                TimeSeriesPoint(
                    date=str(row[date_col])[:10],
                    value=round(float(row[pred_col]), 2),
                    metric_name="forecast_demand",
                    currency=None,
                    dimensions={"sku_id": str(row.get("sku_id", ""))},
                )
                for _, row in df.iterrows()
            ]
            meta = build_metadata(
                tool_name=tool_name,
                domain=QueryDomain.FORECASTING,
                context=context,
                start_time=t0,
                confidence_provenance="MODEL_BASED",
                source_engine="commerce_ai.forecasting",
            )
            ts_res = TimeSeriesResult(
                metric_name="forecast_demand",
                display_name="Projected Demand Forecast",
                time_grain=context.time_grain,
                points=points,
                unit="units",
                currency=None,
                metadata=meta,
            )
            return QueryResponse(
                query_id=meta.query_id,
                tool_name=tool_name,
                domain=meta.domain,
                status=CalculationStatus.SUCCESS,
                metadata=meta,
                time_series=ts_res,
            )

    return build_unavailable_response(
        tool_name=tool_name,
        domain=QueryDomain.FORECASTING,
        context=context,
        start_time=t0,
        reason="Demand forecast outputs are unavailable for the requested parameters. Forecasting models must be trained or backtested before querying.",
        source_engine="commerce_ai.forecasting",
    )


def get_forecast_accuracy(
    context: QueryContext,
    actual_series: Optional[Sequence[float]] = None,
    forecast_series: Optional[Sequence[float]] = None,
    evaluation_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Query standard statistical accuracy metrics (WAPE, MAE, RMSE, MAPE)."""
    t0 = time.perf_counter()
    tool_name = "get_forecast_accuracy"

    if evaluation_df is not None and not evaluation_df.empty:
        # If pre-computed evaluation table is passed
        rows = evaluation_df.to_dict(orient="records")
        meta = build_metadata(
            tool_name=tool_name,
            domain=QueryDomain.FORECASTING,
            context=context,
            start_time=t0,
            confidence_provenance="MODEL_BASED",
            source_engine="commerce_ai.forecasting.evaluation",
        )
        table = TableResult(
            columns=list(evaluation_df.columns),
            rows=rows,
            total_rows=len(rows),
            metadata=meta,
        )
        return QueryResponse(
            query_id=meta.query_id,
            tool_name=tool_name,
            domain=meta.domain,
            status=CalculationStatus.SUCCESS,
            metadata=meta,
            table=table,
        )

    if actual_series is not None and forecast_series is not None:
        metrics = evaluate_forecast(actual_series, forecast_series)
        meta = build_metadata(
            tool_name=tool_name,
            domain=QueryDomain.FORECASTING,
            context=context,
            start_time=t0,
            confidence_provenance="MODEL_BASED",
            source_engine="commerce_ai.analytics.metrics",
        )
        evidence = [
            EvidenceReference(
                source_engine="commerce_ai.analytics.metrics",
                source_type="forecast_evaluation",
                source_id="accuracy_batch",
                metric="wape",
                value=metrics["wape"],
                as_of_date=context.iso_as_of_date,
            )
        ]
        kpis = [
            MetricResult(
                metric_name="wape",
                display_name="Weighted Absolute Percentage Error (WAPE)",
                value=round(metrics["wape"], 4),
                unit="ratio",
                source="ForecastMetrics",
                evidence=evidence,
            ),
            MetricResult(
                metric_name="mae",
                display_name="Mean Absolute Error (MAE)",
                value=round(metrics["mae"], 2),
                unit="units",
                source="ForecastMetrics",
                evidence=evidence,
            ),
            MetricResult(
                metric_name="rmse",
                display_name="Root Mean Squared Error (RMSE)",
                value=round(metrics["rmse"], 2),
                unit="units",
                source="ForecastMetrics",
                evidence=evidence,
            ),
            MetricResult(
                metric_name="mape",
                display_name="Mean Absolute Percentage Error (MAPE)",
                value=round(metrics["mape"], 4),
                unit="ratio",
                source="ForecastMetrics",
                evidence=evidence,
            ),
        ]
        return QueryResponse(
            query_id=meta.query_id,
            tool_name=tool_name,
            domain=meta.domain,
            status=CalculationStatus.SUCCESS,
            metadata=meta,
            metrics=kpis,
        )

    return build_unavailable_response(
        tool_name=tool_name,
        domain=QueryDomain.FORECASTING,
        context=context,
        start_time=t0,
        reason="Forecast accuracy evaluation data is unavailable without paired actual vs predicted observations.",
    )


def get_forecast_bias(
    context: QueryContext,
    actual_series: Optional[Sequence[float]] = None,
    forecast_series: Optional[Sequence[float]] = None,
) -> QueryResponse:
    """Query directional forecast bias (tracking signal / tendency)."""
    t0 = time.perf_counter()
    tool_name = "get_forecast_bias"

    if actual_series is not None and forecast_series is not None:
        metrics = evaluate_forecast(actual_series, forecast_series)
        bias_val = metrics["bias"]
        meta = build_metadata(
            tool_name=tool_name,
            domain=QueryDomain.FORECASTING,
            context=context,
            start_time=t0,
            confidence_provenance="MODEL_BASED",
            source_engine="commerce_ai.analytics.metrics",
        )
        direction = "BALANCED"
        if bias_val > 0.05:
            direction = "OVER_FORECASTING"
        elif bias_val < -0.05:
            direction = "UNDER_FORECASTING"

        kpi = MetricResult(
            metric_name="forecast_bias",
            display_name="Forecast Bias",
            value=round(bias_val, 4),
            unit="ratio",
            source="ForecastMetrics",
            status=direction,
        )
        return QueryResponse(
            query_id=meta.query_id,
            tool_name=tool_name,
            domain=meta.domain,
            status=CalculationStatus.SUCCESS,
            metadata=meta,
            metrics=[kpi],
        )

    return build_unavailable_response(
        tool_name=tool_name,
        domain=QueryDomain.FORECASTING,
        context=context,
        start_time=t0,
        reason="Forecast bias data is unavailable. Paired actual and forecast time series are required.",
    )


def get_forecast_summary(
    context: QueryContext,
    forecast_output: Optional[ForecastOutput] = None,
    forecast_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Summarize overall forecasting model configuration and performance."""
    t0 = time.perf_counter()
    tool_name = "get_forecast_summary"

    if forecast_output is not None:
        meta = build_metadata(
            tool_name=tool_name,
            domain=QueryDomain.FORECASTING,
            context=context,
            start_time=t0,
            confidence_provenance="MODEL_BASED",
            source_engine="commerce_ai.forecasting",
        )
        metrics = [
            MetricResult(
                metric_name="forecast_horizon",
                display_name="Forecast Horizon",
                value=len(forecast_output.records),
                unit="periods",
                source="ForecastService",
            ),
            MetricResult(
                metric_name="selected_model",
                display_name="Active Forecasting Model",
                value=forecast_output.metadata.model_name,
                unit="model",
                source="ForecastService",
            ),
        ]
        return QueryResponse(
            query_id=meta.query_id,
            tool_name=tool_name,
            domain=meta.domain,
            status=CalculationStatus.SUCCESS,
            metadata=meta,
            metrics=metrics,
        )

    return build_unavailable_response(
        tool_name=tool_name,
        domain=QueryDomain.FORECASTING,
        context=context,
        start_time=t0,
        reason="Forecast model summary is unavailable. No active forecast model output was provided.",
    )
