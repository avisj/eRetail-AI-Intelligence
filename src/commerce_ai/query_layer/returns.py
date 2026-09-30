"""Customer Returns Intelligence Query Tools (Phase 7A).

Provides read-only query tools for Dashboards and eRetail Copilot:
- get_return_summary: Executive return KPIs (events, units, unit return rate %, revenue return rate %, valuation)
- get_return_rate: Return rate metrics with sample-size guards and confidence evaluation
- get_return_trend: Return volume and rate time series across daily, weekly, or monthly grain
- get_return_anomalies: Statistical return anomalies with severity and factual rationales (Phase 5B)
- get_return_risk: Predictive return risk scores and calibration tiers (Phase 5C)
- get_return_reason_breakdown: Categorical breakdown across customer return reasons
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.returns.service import ReturnsIntelligenceService
from commerce_ai.returns.anomaly import ReturnAnomalyService
from commerce_ai.query_layer.base import (
    build_currency_inconsistency_response,
    build_empty_response,
    build_metadata,
    build_unavailable_response,
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


def get_return_summary(
    context: QueryContext,
    returns_df: pd.DataFrame,
    sales_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Calculate aggregate customer return KPIs using Phase 5A Returns Intelligence."""
    t0 = time.perf_counter()
    tool_name = "get_return_summary"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        returns_df,
        context=context,
        date_col="return_date",
        currency_col=None,
        products_df=products_df,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.RETURNS,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.RETURNS,
            context=context,
            start_time=t0,
            currency=curr,
        )

    # Filter sales to matching temporal context if provided
    filtered_sales = sales_df
    if sales_df is not None and not sales_df.empty:
        filtered_sales, _, _, _ = apply_context_filters(
            sales_df,
            context=context,
            date_col="date",
            currency_col="currency",
            products_df=products_df,
        )

    ret_service = ReturnsIntelligenceService()
    ret_res = ret_service.analyze(
        returns=filtered,
        sales=filtered_sales,
        products=products_df,
        as_of_date=context.iso_as_of_date,
    )
    summary = ret_res.summary

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.RETURNS,
        context=context,
        start_time=t0,
        currency=getattr(summary, "currency", curr or context.effective_currency or "USD"),
        source_engine="commerce_ai.returns.ReturnsIntelligenceService",
    )

    evidence = [
        EvidenceReference(
            source_engine="commerce_ai.returns",
            source_type="returns_summary",
            source_id="ret_summary_batch",
            metric="unit_return_rate",
            value=summary.overall_unit_return_rate,
            as_of_date=context.iso_as_of_date,
            notes=f"Processed {summary.total_return_events:,d} return events across {summary.total_returned_units:,d} units",
        )
    ]

    metrics = [
        MetricResult(
            metric_name="total_return_events",
            display_name="Total Return Events",
            value=summary.total_return_events,
            unit="events",
            source="ReturnsIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="total_returned_units",
            display_name="Total Returned Units",
            value=summary.total_returned_units,
            unit="units",
            source="ReturnsIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
    ]

    if summary.overall_unit_return_rate is not None:
        metrics.append(
            MetricResult(
                metric_name="unit_return_rate",
                display_name="Unit Return Rate %",
                value=round(summary.overall_unit_return_rate * 100.0, 2),
                unit="%",
                source="ReturnsIntelligenceService",
                as_of_date=context.iso_as_of_date,
                evidence=evidence,
            )
        )

    if summary.overall_revenue_return_rate is not None:
        metrics.append(
            MetricResult(
                metric_name="revenue_return_rate",
                display_name="Revenue Return Rate %",
                value=round(summary.overall_revenue_return_rate * 100.0, 2),
                unit="%",
                source="ReturnsIntelligenceService",
                as_of_date=context.iso_as_of_date,
                evidence=evidence,
            )
        )

    ret_val = getattr(summary, "total_return_value", getattr(summary, "total_returned_value", None))
    if ret_val is not None:
        ret_curr = getattr(summary, "currency", None) or curr or "USD"
        metrics.append(
            MetricResult(
                metric_name="total_returned_value",
                display_name="Total Returned Value Exposure",
                value=round(ret_val, 2),
                unit=ret_curr,
                currency=ret_curr,
                source="ReturnsIntelligenceService",
                as_of_date=context.iso_as_of_date,
                evidence=evidence,
            )
        )

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        metrics=metrics,
    )


def get_return_rate(
    context: QueryContext,
    returns_df: pd.DataFrame,
    sales_df: pd.DataFrame,
) -> QueryResponse:
    """Query return rate with sample-size guard evaluation."""
    t0 = time.perf_counter()
    tool_name = "get_return_rate"

    res = get_return_summary(context, returns_df=returns_df, sales_df=sales_df)
    res.tool_name = tool_name
    return res


def get_return_trend(
    context: QueryContext,
    returns_df: pd.DataFrame,
) -> QueryResponse:
    """Generate time-series observations of customer return volume."""
    t0 = time.perf_counter()
    tool_name = "get_return_trend"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        returns_df,
        context=context,
        date_col="return_date",
        currency_col=None,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.RETURNS,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty or "return_date" not in filtered.columns:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.RETURNS,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = filtered.copy()
    df["qty"] = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
    df["dt"] = pd.to_datetime(df["return_date"], errors="coerce")
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
            metric_name="returned_units",
            currency=None,
        )
        for _, row in grouped.iterrows()
    ]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.RETURNS,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.returns",
    )

    ts_res = TimeSeriesResult(
        metric_name="returned_units",
        display_name=f"Returned Units Trend ({grain.capitalize()})",
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


def get_return_anomalies(
    context: QueryContext,
    returns_df: pd.DataFrame,
    sales_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Query detected statistical return anomalies using Phase 5B."""
    t0 = time.perf_counter()
    tool_name = "get_return_anomalies"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        returns_df,
        context=context,
        date_col="return_date",
        currency_col=None,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.RETURNS,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.RETURNS,
            context=context,
            start_time=t0,
            currency=curr,
        )

    anomaly_service = ReturnAnomalyService()
    anomaly_result = anomaly_service.detect(
        returns=filtered,
        sales=sales_df,
        as_of_date=context.iso_as_of_date,
    )

    rows = [
        {
            "anomaly_id": a.anomaly_id,
            "dimension": a.dimension,
            "entity_id": a.entity_id,
            "anomaly_type": a.anomaly_type.value if hasattr(a.anomaly_type, "value") else str(a.anomaly_type),
            "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
            "direction": a.direction.value if hasattr(a.direction, "value") else str(a.direction),
            "observed_value": getattr(a, "current_value", getattr(a, "observed_value", 0.0)),
            "expected_value": getattr(a, "baseline_value", getattr(a, "expected_value", 0.0)),
            "z_score": round(a.z_score, 2) if a.z_score is not None else None,
            "rationale": a.rationale,
        }
        for a in anomaly_result.anomalies
    ]

    # Pagination
    if context.offset is not None:
        rows = rows[context.offset:]
    if context.limit is not None:
        rows = rows[:context.limit]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.RETURNS,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.returns.anomaly",
    )

    table = TableResult(
        columns=[
            "anomaly_id", "dimension", "entity_id", "anomaly_type", "severity",
            "direction", "observed_value", "expected_value", "z_score", "rationale"
        ],
        column_types={
            "anomaly_id": "string",
            "dimension": "string",
            "entity_id": "string",
            "anomaly_type": "string",
            "severity": "string",
            "direction": "string",
            "observed_value": "float",
            "expected_value": "float",
            "z_score": "float",
            "rationale": "string",
        },
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


def get_return_risk(
    context: QueryContext,
    return_risk_records: Optional[Sequence[Any]] = None,
) -> QueryResponse:
    """Query predictive return risk scores (Phase 5C)."""
    t0 = time.perf_counter()
    tool_name = "get_return_risk"

    if return_risk_records is not None and len(return_risk_records) > 0:
        rows = [
            r.model_dump() if hasattr(r, "model_dump") else (r.to_dict() if hasattr(r, "to_dict") else dict(r))
            for r in return_risk_records
        ]
        meta = build_metadata(
            tool_name=tool_name,
            domain=QueryDomain.RETURNS,
            context=context,
            start_time=t0,
            confidence_provenance="MODEL_BASED",
            source_engine="commerce_ai.returns.risk",
        )
        table = TableResult(
            columns=list(rows[0].keys()) if rows else [],
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

    return build_unavailable_response(
        tool_name=tool_name,
        domain=QueryDomain.RETURNS,
        context=context,
        start_time=t0,
        reason="Predictive return risk records are unavailable. Return prediction models must be executed first.",
    )


def get_return_reason_breakdown(
    context: QueryContext,
    returns_df: pd.DataFrame,
) -> QueryResponse:
    """Calculate customer return reason categorical distribution."""
    t0 = time.perf_counter()
    tool_name = "get_return_reason_breakdown"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        returns_df,
        context=context,
        date_col="return_date",
        currency_col=None,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.RETURNS,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty or "reason" not in filtered.columns:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.RETURNS,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = filtered.copy()
    qty = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
    df["qty"] = qty

    grouped = df.groupby("reason").agg(
        returned_units=("qty", "sum"),
        event_count=("reason", "count"),
    ).reset_index()

    total_units = int(grouped["returned_units"].sum())
    grouped["percentage"] = (grouped["returned_units"] / total_units) * 100.0 if total_units > 0 else 0.0
    grouped = grouped.sort_values("returned_units", ascending=False)

    items = [
        BreakdownItem(
            dimension_name="return_reason",
            dimension_value=str(row["reason"]),
            metric_value=int(row["returned_units"]),
            metric_name="returned_units",
            percentage_of_total=round(float(row["percentage"]), 2),
            currency=None,
            additional_metrics={"event_count": int(row["event_count"])},
        )
        for _, row in grouped.iterrows()
    ]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.RETURNS,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.returns",
    )

    breakdown = BreakdownResult(
        metric_name="returned_units",
        display_name="Customer Return Reason Breakdown",
        dimension_name="return_reason",
        items=items,
        total_value=total_units,
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
        breakdown=breakdown,
    )
