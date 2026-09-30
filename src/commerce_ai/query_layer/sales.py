"""Sales & Commercial Revenue Query Tools (Phase 7A).

Provides read-only query tools for Dashboards and eRetail Copilot:
- get_sales_summary: Executive KPI cards (Gross Revenue, Net Revenue, Units Sold, Orders, AOV)
- get_sales_trend: Time series trends across daily, weekly, or monthly grain
- get_sales_by_sku: Tabular / dimensional breakdown across individual SKUs
- get_sales_by_channel: Breakdown across commercial channels
- get_sales_by_warehouse: Breakdown across distribution centers
- get_sales_by_category: Commercial breakdown by product category
- get_sales_by_brand: Commercial breakdown by product brand
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

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


def _extract_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _extract_discount(df: pd.DataFrame) -> pd.Series:
    return _extract_series(df, "discount", 0.0)


def get_sales_summary(
    context: QueryContext,
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Calculate aggregate commercial sales KPI metrics."""
    t0 = time.perf_counter()
    tool_name = "get_sales_summary"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
        products_df=products_df,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.SALES,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.SALES,
            context=context,
            start_time=t0,
            currency=curr,
        )

    # Calculate vectorized sales aggregates
    qty = _extract_series(filtered, "quantity", 0.0)
    price = _extract_series(filtered, "unit_price", 0.0)
    disc = _extract_discount(filtered)

    gross_rev = float((qty * price).sum())
    total_disc = float(disc.sum())
    net_rev = float(gross_rev - total_disc)
    total_units = int(qty.sum())
    
    order_col = "order_id" if "order_id" in filtered.columns else "sale_id"
    order_count = int(filtered[order_col].nunique()) if order_col in filtered.columns else len(filtered)
    
    aov = round(net_rev / order_count, 2) if order_count > 0 else 0.0
    asp = round(gross_rev / total_units, 2) if total_units > 0 else 0.0

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.SALES,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.sales",
    )

    sample_id = str(filtered["sale_id"].iloc[0]) if "sale_id" in filtered.columns else "N/A"
    evidence = [
        EvidenceReference(
            source_engine="commerce_ai.sales",
            source_type="transaction_dataset",
            source_id=f"sales_batch_{len(filtered)}",
            metric="net_revenue",
            value=round(net_rev, 2),
            currency=curr,
            as_of_date=context.iso_as_of_date,
            notes=f"Calculated from {len(filtered):,d} transaction rows. Sample sale_id: {sample_id}",
        )
    ]

    metrics = [
        MetricResult(
            metric_name="gross_revenue",
            display_name="Gross Revenue",
            value=round(gross_rev, 2),
            unit=curr or "USD",
            currency=curr,
            source="commerce_ai.sales",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="total_discount",
            display_name="Total Discount",
            value=round(total_disc, 2),
            unit=curr or "USD",
            currency=curr,
            source="commerce_ai.sales",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="net_revenue",
            display_name="Net Revenue",
            value=round(net_rev, 2),
            unit=curr or "USD",
            currency=curr,
            source="commerce_ai.sales",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="units_sold",
            display_name="Units Sold",
            value=total_units,
            unit="units",
            currency=None,
            source="commerce_ai.sales",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="order_count",
            display_name="Order Count",
            value=order_count,
            unit="orders",
            currency=None,
            source="commerce_ai.sales",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="average_order_value",
            display_name="Average Order Value (AOV)",
            value=aov,
            unit=curr or "USD",
            currency=curr,
            source="commerce_ai.sales",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="average_selling_price",
            display_name="Average Selling Price (ASP)",
            value=asp,
            unit=curr or "USD",
            currency=curr,
            source="commerce_ai.sales",
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


def get_sales_trend(
    context: QueryContext,
    sales_df: pd.DataFrame,
) -> QueryResponse:
    """Generate time-series observations of net revenue and volume."""
    t0 = time.perf_counter()
    tool_name = "get_sales_trend"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.SALES,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty or "date" not in filtered.columns:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.SALES,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = filtered.copy()
    qty = _extract_series(df, "quantity", 0.0)
    price = _extract_series(df, "unit_price", 0.0)
    disc = _extract_discount(df)
    df["net_revenue"] = (qty * price) - disc
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
    grouped = df.groupby("period_bucket")["net_revenue"].sum().reset_index()
    grouped = grouped.sort_values("period_bucket")

    points = [
        TimeSeriesPoint(
            date=str(row["period_bucket"]),
            value=round(float(row["net_revenue"]), 2),
            metric_name="net_revenue",
            currency=curr,
        )
        for _, row in grouped.iterrows()
    ]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.SALES,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.sales",
    )

    ts_res = TimeSeriesResult(
        metric_name="net_revenue",
        display_name=f"Net Revenue Trend ({grain.capitalize()})",
        time_grain=grain,
        points=points,
        unit=curr or "USD",
        currency=curr,
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


def get_sales_by_sku(
    context: QueryContext,
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Generate tabular sales performance breakdown across SKUs (neutral, no ranking)."""
    t0 = time.perf_counter()
    tool_name = "get_sales_by_sku"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
        products_df=products_df,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.SALES,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty or "sku_id" not in filtered.columns:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.SALES,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = filtered.copy()
    qty = _extract_series(df, "quantity", 0.0)
    price = _extract_series(df, "unit_price", 0.0)
    disc = _extract_discount(df)
    df["gross_revenue"] = qty * price
    df["discount_amt"] = disc
    df["net_revenue"] = df["gross_revenue"] - df["discount_amt"]
    df["qty"] = qty

    agg = df.groupby("sku_id").agg(
        units_sold=("qty", "sum"),
        gross_revenue=("gross_revenue", "sum"),
        total_discount=("discount_amt", "sum"),
        net_revenue=("net_revenue", "sum"),
        order_count=("order_id", "nunique") if "order_id" in df.columns else ("sale_id", "count"),
    ).reset_index()

    total_net = agg["net_revenue"].sum()
    agg["share_of_revenue"] = np.where(
        total_net > 0,
        (agg["net_revenue"] / total_net) * 100.0,
        0.0,
    )

    # Sort deterministically by sku_id (no winner ranking)
    agg = agg.sort_values("sku_id")

    # Pagination if requested
    if context.offset is not None:
        agg = agg.iloc[context.offset:]
    if context.limit is not None:
        agg = agg.iloc[:context.limit]

    rows = agg.to_dict(orient="records")
    for r in rows:
        r["units_sold"] = int(r["units_sold"])
        r["gross_revenue"] = round(float(r["gross_revenue"]), 2)
        r["total_discount"] = round(float(r["total_discount"]), 2)
        r["net_revenue"] = round(float(r["net_revenue"]), 2)
        r["order_count"] = int(r["order_count"])
        r["share_of_revenue"] = round(float(r["share_of_revenue"]), 2)
        r["currency"] = curr

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.SALES,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.sales",
    )

    table = TableResult(
        columns=["sku_id", "units_sold", "gross_revenue", "total_discount", "net_revenue", "order_count", "share_of_revenue", "currency"],
        column_types={
            "sku_id": "string",
            "units_sold": "integer",
            "gross_revenue": "float",
            "total_discount": "float",
            "net_revenue": "float",
            "order_count": "integer",
            "share_of_revenue": "float",
            "currency": "string",
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


def _get_dimensional_breakdown(
    dimension_col: str,
    display_title: str,
    context: QueryContext,
    sales_df: pd.DataFrame,
    tool_name: str,
    joined_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Helper to compute deterministic dimensional breakdowns for channels, warehouses, categories, brands."""
    t0 = time.perf_counter()
    source_df = joined_df if joined_df is not None else sales_df

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        source_df,
        context=context,
        date_col="date" if "date" in source_df.columns else None,
        currency_col="currency",
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.SALES,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty or dimension_col not in filtered.columns:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.SALES,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = filtered.copy()
    qty = _extract_series(df, "quantity", 0.0)
    price = _extract_series(df, "unit_price", 0.0)
    disc = _extract_discount(df)
    df["net_revenue"] = (qty * price) - disc
    df["qty"] = qty

    grouped = df.groupby(dimension_col).agg(
        net_revenue=("net_revenue", "sum"),
        units_sold=("qty", "sum"),
    ).reset_index()

    total_net = float(grouped["net_revenue"].sum())
    grouped["percentage"] = np.where(
        total_net > 0,
        (grouped["net_revenue"] / total_net) * 100.0,
        0.0,
    )

    # Sort deterministically by dimension value
    grouped = grouped.sort_values(dimension_col)

    items = [
        BreakdownItem(
            dimension_name=dimension_col,
            dimension_value=str(row[dimension_col]),
            metric_value=round(float(row["net_revenue"]), 2),
            metric_name="net_revenue",
            percentage_of_total=round(float(row["percentage"]), 2),
            currency=curr,
            additional_metrics={"units_sold": int(row["units_sold"])},
        )
        for _, row in grouped.iterrows()
    ]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.SALES,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.sales",
    )

    breakdown = BreakdownResult(
        metric_name="net_revenue",
        display_name=display_title,
        dimension_name=dimension_col,
        items=items,
        total_value=round(total_net, 2),
        unit=curr or "USD",
        currency=curr,
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


def get_sales_by_channel(
    context: QueryContext,
    sales_df: pd.DataFrame,
) -> QueryResponse:
    """Calculate commercial revenue breakdown by sales channel."""
    return _get_dimensional_breakdown(
        dimension_col="channel_id",
        display_title="Net Revenue by Channel",
        context=context,
        sales_df=sales_df,
        tool_name="get_sales_by_channel",
    )


def get_sales_by_warehouse(
    context: QueryContext,
    sales_df: pd.DataFrame,
) -> QueryResponse:
    """Calculate commercial revenue breakdown by fulfillment warehouse."""
    return _get_dimensional_breakdown(
        dimension_col="warehouse_id",
        display_title="Net Revenue by Warehouse",
        context=context,
        sales_df=sales_df,
        tool_name="get_sales_by_warehouse",
    )


def get_sales_by_category(
    context: QueryContext,
    sales_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> QueryResponse:
    """Calculate commercial revenue breakdown by product category."""
    cat_col = "category_id" if "category_id" in products_df.columns else "category"
    joined = sales_df.merge(products_df[["sku_id", cat_col]], on="sku_id", how="left")
    return _get_dimensional_breakdown(
        dimension_col=cat_col,
        display_title="Net Revenue by Category",
        context=context,
        sales_df=sales_df,
        tool_name="get_sales_by_category",
        joined_df=joined,
    )


def get_sales_by_brand(
    context: QueryContext,
    sales_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> QueryResponse:
    """Calculate commercial revenue breakdown by product brand."""
    joined = sales_df.merge(products_df[["sku_id", "brand"]], on="sku_id", how="left")
    return _get_dimensional_breakdown(
        dimension_col="brand",
        display_title="Net Revenue by Brand",
        context=context,
        sales_df=sales_df,
        tool_name="get_sales_by_brand",
        joined_df=joined,
    )
