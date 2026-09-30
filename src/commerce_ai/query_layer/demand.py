"""Demand Intelligence & Segmentation Query Tools (Phase 7A).

Provides read-only query tools for Dashboards and eRetail Copilot:
- get_demand_summary: Overall demand volume, daily velocity, active catalog breadth
- get_demand_trend: Demand time series across daily, weekly, or monthly grain
- get_demand_profile: SKU-level demand intermittency, ADI, CV2, and demand categorization
- get_abc_xyz_distribution: ABC-XYZ 9-box matrix distribution and revenue/volatility breakdown
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.analytics.abc_xyz import (
    ABCConfig,
    XYZConfig,
    calculate_abc_xyz_matrix,
)
from commerce_ai.analytics.features import calculate_intermittency_metrics
from commerce_ai.query_layer.base import (
    build_currency_inconsistency_response,
    build_empty_response,
    build_metadata,
)
from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.filters import apply_context_filters
from commerce_ai.query_layer.schemas import (
    BreakdownItem,
    BreakdownResult,
    CalculationStatus,
    EvidenceReference,
    MetricResult,
    QueryDomain,
    QueryResponse,
    TableResult,
    TimeSeriesPoint,
    TimeSeriesResult,
)


def get_demand_summary(
    context: QueryContext,
    sales_df: pd.DataFrame,
) -> QueryResponse:
    """Calculate aggregate demand volume, daily demand velocity, and active SKU breadth."""
    t0 = time.perf_counter()
    tool_name = "get_demand_summary"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.DEMAND,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty or "quantity" not in filtered.columns:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.DEMAND,
            context=context,
            start_time=t0,
            currency=curr,
        )

    qty_series = pd.to_numeric(filtered["quantity"], errors="coerce").fillna(0)
    total_demand_units = int(qty_series.sum())
    active_skus = int(filtered["sku_id"].nunique()) if "sku_id" in filtered.columns else 0

    # Calculate days in evaluation window
    if "date" in filtered.columns:
        d_min = pd.to_datetime(filtered["date"].min())
        d_max = pd.to_datetime(filtered["date"].max())
        day_count = max(1, (d_max - d_min).days + 1)
    else:
        day_count = 1

    avg_daily_demand = round(total_demand_units / day_count, 2)

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.DEMAND,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.analytics.demand",
    )

    evidence = [
        EvidenceReference(
            source_engine="commerce_ai.analytics.demand",
            source_type="sales_transactions",
            source_id=f"demand_batch_{len(filtered)}",
            metric="total_demand_units",
            value=total_demand_units,
            as_of_date=context.iso_as_of_date,
            notes=f"Aggregated across {active_skus} active SKUs over {day_count} calendar days",
        )
    ]

    metrics = [
        MetricResult(
            metric_name="total_demand_units",
            display_name="Total Demand Volume",
            value=total_demand_units,
            unit="units",
            currency=None,
            source="DemandAnalytics",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="average_daily_demand",
            display_name="Average Daily Demand Rate",
            value=avg_daily_demand,
            unit="units/day",
            currency=None,
            source="DemandAnalytics",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="active_sku_count",
            display_name="Active SKUs with Demand",
            value=active_skus,
            unit="SKUs",
            currency=None,
            source="DemandAnalytics",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="calendar_days_observed",
            display_name="Observed Calendar Days",
            value=day_count,
            unit="days",
            currency=None,
            source="DemandAnalytics",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
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


def get_demand_trend(
    context: QueryContext,
    sales_df: pd.DataFrame,
) -> QueryResponse:
    """Generate time-series observations of customer demand volume."""
    t0 = time.perf_counter()
    tool_name = "get_demand_trend"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.DEMAND,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty or "date" not in filtered.columns or "quantity" not in filtered.columns:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.DEMAND,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = filtered.copy()
    df["qty"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["dt"])

    grain = context.time_grain.lower()
    if grain == "monthly":
        period_series = df["dt"].dt.to_period("M").astype(str)
    elif grain == "weekly":
        period_series = df["dt"].dt.to_period("W").astype(str)
    else:
        grain = "daily"
        period_series = df["dt"].dt.strftime("%Y-%m-%d")

    df["period_bucket"] = period_series
    grouped = df.groupby("period_bucket")["qty"].sum().reset_index().sort_values("period_bucket")

    points = [
        TimeSeriesPoint(
            date=str(row["period_bucket"]),
            value=int(row["qty"]),
            metric_name="demand_units",
            currency=None,
        )
        for _, row in grouped.iterrows()
    ]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.DEMAND,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.analytics.demand",
    )

    ts_res = TimeSeriesResult(
        metric_name="demand_units",
        display_name=f"Customer Demand Volume ({grain.capitalize()})",
        time_grain=grain,
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


def get_demand_profile(
    context: QueryContext,
    sales_df: pd.DataFrame,
) -> QueryResponse:
    """Generate tabular demand intermittency profiles (ADI, CV2, intermittency classification)."""
    t0 = time.perf_counter()
    tool_name = "get_demand_profile"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.DEMAND,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty or "sku_id" not in filtered.columns:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.DEMAND,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = filtered.copy()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    metrics_df = calculate_intermittency_metrics(df, target_col="quantity", group_col="sku_id")
    qty_sums = df.groupby("sku_id")["quantity"].sum().to_dict()

    profiles = []
    for _, row in metrics_df.iterrows():
        s_id = str(row["sku_id"])
        profiles.append({
            "sku_id": s_id,
            "demand_occurrence_rate": float(row["demand_occurrence_rate"]),
            "average_nonzero_demand": float(row["average_nonzero_demand"]),
            "intermittency_class": str(row["intermittency_class"]),
            "total_demand_units": int(qty_sums.get(s_id, 0)),
            "active_days_count": int(row["non_zero_demand_days"]),
        })

    profiles.sort(key=lambda x: x["sku_id"])

    # Pagination
    if context.offset is not None:
        profiles = profiles[context.offset:]
    if context.limit is not None:
        profiles = profiles[:context.limit]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.DEMAND,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.analytics.features",
    )

    table = TableResult(
        columns=[
            "sku_id", "demand_occurrence_rate", "average_nonzero_demand",
            "intermittency_class", "total_demand_units", "active_days_count"
        ],
        column_types={
            "sku_id": "string",
            "demand_occurrence_rate": "float",
            "average_nonzero_demand": "float",
            "intermittency_class": "string",
            "total_demand_units": "integer",
            "active_days_count": "integer",
        },
        rows=profiles,
        total_rows=len(profiles),
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


def get_abc_xyz_distribution(
    context: QueryContext,
    sales_df: pd.DataFrame,
) -> QueryResponse:
    """Query ABC-XYZ 9-box matrix distribution across product catalog."""
    t0 = time.perf_counter()
    tool_name = "get_abc_xyz_distribution"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.DEMAND,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty or "sku_id" not in filtered.columns:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.DEMAND,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = filtered.copy()
    qty = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    price = pd.to_numeric(df.get("unit_price", 0.0), errors="coerce").fillna(0.0)
    df["revenue"] = qty * price
    df["units_sold"] = qty

    # Calculate ABC-XYZ matrix
    sku_class_df, summary_df, _ = calculate_abc_xyz_matrix(df)

    items = []
    total_skus = len(sku_class_df)
    for _, row in summary_df.iterrows():
        box = str(row["abc_xyz_class"])
        cnt = int(row["sku_count"])
        pct = (cnt / total_skus) * 100.0 if total_skus > 0 else 0.0
        items.append(
            BreakdownItem(
                dimension_name="abc_xyz_box",
                dimension_value=box,
                metric_value=cnt,
                metric_name="sku_count",
                percentage_of_total=round(pct, 2),
                currency=None,
                additional_metrics={
                    "total_revenue": round(float(row.get("total_revenue", 0.0)), 2),
                    "avg_demand": round(float(row.get("avg_demand", 0.0)), 2),
                    "avg_cv": round(float(row.get("avg_cv", 0.0)), 2) if pd.notna(row.get("avg_cv")) else None,
                },
            )
        )

    items.sort(key=lambda x: x.dimension_value)

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.DEMAND,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.analytics.abc_xyz",
    )

    breakdown = BreakdownResult(
        metric_name="sku_count",
        display_name="ABC-XYZ 9-Box Matrix Segmentation",
        dimension_name="abc_xyz_box",
        items=items,
        total_value=total_skus,
        unit="SKUs",
        currency=None,
        metadata=meta,
    )

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        breakdown=breakdown,
    )
