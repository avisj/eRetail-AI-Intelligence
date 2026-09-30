"""Inventory & Stock Position Query Tools (Phase 7A).

Provides read-only query tools for Dashboards and eRetail Copilot:
- get_inventory_summary: Executive KPI cards (On-Hand, On-Order, Reserved, Net Position, Valuation)
- get_inventory_position: Detailed tabular position by SKU x Warehouse
- get_stockout_risk: Stockout risk identification and runout diagnostics
- get_slow_moving_inventory: Excess stock and low velocity capital exposure
- get_high_value_inventory: High capital valuation inventory holdings
- get_inventory_risk: Risk tier distribution (Critical, Understock, Healthy, Overstock)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.inventory.position import calculate_inventory_positions
from commerce_ai.query_layer.base import (
    build_currency_inconsistency_response,
    build_empty_response,
    build_metadata,
)
from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.filters import apply_context_filters, check_currency_isolation
from commerce_ai.query_layer.schemas import (
    BreakdownItem,
    BreakdownResult,
    CalculationStatus,
    EvidenceReference,
    MetricResult,
    QueryDomain,
    QueryResponse,
    TableResult,
)


def _prepare_inventory_positions(
    context: QueryContext,
    inventory_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[str], bool, Optional[str]]:
    """Compute point-in-time inventory positions and merge product catalog attributes."""
    if inventory_df.empty:
        return inventory_df, context.effective_currency, True, None

    # Calculate inventory positions as of context.as_of_date
    pos_df = calculate_inventory_positions(
        snapshots=inventory_df,
        as_of_date=context.iso_as_of_date,
    )

    if pos_df.empty:
        return pos_df, context.effective_currency, True, None

    # Merge product catalog attributes if available
    merged = pos_df
    if products_df is not None and not products_df.empty:
        prod_cols = ["sku_id"]
        for c in ["product_name", "category_id", "category", "brand", "unit_cost", "selling_price", "currency", "velocity_tier"]:
            if c in products_df.columns:
                prod_cols.append(c)
        merged = merged.merge(products_df[prod_cols], on="sku_id", how="left")

    # Apply context filters
    filtered, curr, is_isolated, err_msg = apply_context_filters(
        merged,
        context=context,
        currency_col="currency" if "currency" in merged.columns else None,
        products_df=products_df,
    )
    return filtered, curr, is_isolated, err_msg


def get_inventory_summary(
    context: QueryContext,
    inventory_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Calculate aggregate inventory position and valuation KPIs."""
    t0 = time.perf_counter()
    tool_name = "get_inventory_summary"

    pos_df, curr, is_isolated, err_msg = _prepare_inventory_positions(context, inventory_df, products_df)
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if pos_df.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            currency=curr,
        )

    on_hand = int(pos_df["on_hand"].sum())
    reserved = int(pos_df["reserved"].sum())
    on_order = int(pos_df["on_order"].sum())
    net_pos = int(pos_df["net_position"].sum())
    damaged = int(pos_df["damaged"].sum()) if "damaged" in pos_df.columns else 0

    total_val = None
    if "unit_cost" in pos_df.columns:
        costs = pd.to_numeric(pos_df["unit_cost"], errors="coerce").fillna(0.0)
        total_val = round(float((pos_df["on_hand"] * costs).sum()), 2)

    active_skus = int(pos_df["sku_id"].nunique())
    warehouses_count = int(pos_df["warehouse_id"].nunique())

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.INVENTORY,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.inventory.position",
    )

    evidence = [
        EvidenceReference(
            source_engine="commerce_ai.inventory",
            source_type="inventory_position_snapshot",
            source_id=f"inv_pos_{len(pos_df)}_rows",
            metric="on_hand_units",
            value=on_hand,
            as_of_date=context.iso_as_of_date,
            notes=f"Aggregated across {active_skus} SKUs and {warehouses_count} warehouses",
        )
    ]

    metrics = [
        MetricResult(
            metric_name="on_hand_units",
            display_name="On-Hand Inventory",
            value=on_hand,
            unit="units",
            currency=None,
            source="InventoryService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="reserved_units",
            display_name="Reserved Inventory",
            value=reserved,
            unit="units",
            currency=None,
            source="InventoryService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="on_order_units",
            display_name="On-Order (Inbound)",
            value=on_order,
            unit="units",
            currency=None,
            source="InventoryService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="net_position_units",
            display_name="Net Inventory Position",
            value=net_pos,
            unit="units",
            currency=None,
            source="InventoryService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="damaged_units",
            display_name="Damaged / Quarantine Stock",
            value=damaged,
            unit="units",
            currency=None,
            source="InventoryService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
    ]

    if total_val is not None:
        metrics.append(
            MetricResult(
                metric_name="inventory_valuation",
                display_name="Total Inventory Valuation",
                value=total_val,
                unit=curr or "USD",
                currency=curr,
                source="InventoryService",
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


def get_inventory_position(
    context: QueryContext,
    inventory_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Query tabular inventory positions by SKU and Warehouse."""
    t0 = time.perf_counter()
    tool_name = "get_inventory_position"

    pos_df, curr, is_isolated, err_msg = _prepare_inventory_positions(context, inventory_df, products_df)
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if pos_df.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = pos_df.copy()
    if "unit_cost" in df.columns:
        df["valuation"] = (df["on_hand"] * pd.to_numeric(df["unit_cost"], errors="coerce").fillna(0.0)).round(2)
    else:
        df["valuation"] = None

    # Deterministic sort
    df = df.sort_values(["sku_id", "warehouse_id"])

    # Pagination
    if context.offset is not None:
        df = df.iloc[context.offset:]
    if context.limit is not None:
        df = df.iloc[:context.limit]

    cols = ["sku_id", "warehouse_id", "on_hand", "reserved", "on_order", "damaged", "net_position", "valuation"]
    if "currency" in df.columns:
        cols.append("currency")

    rows = df[cols].to_dict(orient="records")
    for r in rows:
        r["on_hand"] = int(r["on_hand"])
        r["reserved"] = int(r["reserved"])
        r["on_order"] = int(r["on_order"])
        r["damaged"] = int(r.get("damaged", 0))
        r["net_position"] = int(r["net_position"])
        if r.get("valuation") is not None:
            r["valuation"] = float(r["valuation"])

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.INVENTORY,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.inventory.position",
    )

    table = TableResult(
        columns=cols,
        column_types={
            "sku_id": "string",
            "warehouse_id": "string",
            "on_hand": "integer",
            "reserved": "integer",
            "on_order": "integer",
            "damaged": "integer",
            "net_position": "integer",
            "valuation": "float",
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


def get_stockout_risk(
    context: QueryContext,
    inventory_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    sales_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Identify SKUs/Warehouses at critical or elevated stockout risk."""
    t0 = time.perf_counter()
    tool_name = "get_stockout_risk"

    pos_df, curr, is_isolated, err_msg = _prepare_inventory_positions(context, inventory_df, products_df)
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if pos_df.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            currency=curr,
        )

    # Estimate daily demand rate from sales if available, else flag based on net_position <= 0
    df = pos_df.copy()
    if sales_df is not None and not sales_df.empty and "date" in sales_df.columns:
        s_filt = sales_df
        if context.iso_as_of_date:
            s_filt = s_filt[s_filt["date"] <= context.iso_as_of_date]
        if not s_filt.empty:
            recent_sales = s_filt.groupby(["sku_id", "warehouse_id"])["quantity"].sum().reset_index()
            day_span = max(1, (pd.to_datetime(s_filt["date"].max()) - pd.to_datetime(s_filt["date"].min())).days)
            recent_sales["daily_demand"] = recent_sales["quantity"] / day_span
            df = df.merge(recent_sales[["sku_id", "warehouse_id", "daily_demand"]], on=["sku_id", "warehouse_id"], how="left")
            df["daily_demand"] = df["daily_demand"].fillna(0.0)
            df["days_of_supply"] = np.where(df["daily_demand"] > 0, df["net_position"] / df["daily_demand"], 999.0)
        else:
            df["daily_demand"] = 0.0
            df["days_of_supply"] = np.where(df["net_position"] <= 0, 0.0, 999.0)
    else:
        df["daily_demand"] = 0.0
        df["days_of_supply"] = np.where(df["net_position"] <= 0, 0.0, 999.0)

    # Risk tier determination
    conditions = [
        (df["net_position"] <= 0),
        (df["days_of_supply"] < 7.0),
        (df["days_of_supply"] < 14.0),
    ]
    choices = ["CRITICAL_STOCKOUT", "HIGH_STOCKOUT_RISK", "MEDIUM_STOCKOUT_RISK"]
    df["stockout_risk_tier"] = np.select(conditions, choices, default="HEALTHY")

    # Filter to items exhibiting stockout risk
    at_risk = df[df["stockout_risk_tier"] != "HEALTHY"].sort_values(["net_position", "sku_id"])

    if context.offset is not None:
        at_risk = at_risk.iloc[context.offset:]
    if context.limit is not None:
        at_risk = at_risk.iloc[:context.limit]

    cols = ["sku_id", "warehouse_id", "on_hand", "net_position", "daily_demand", "days_of_supply", "stockout_risk_tier"]
    rows = at_risk[cols].to_dict(orient="records")
    for r in rows:
        r["on_hand"] = int(r["on_hand"])
        r["net_position"] = int(r["net_position"])
        r["daily_demand"] = round(float(r["daily_demand"]), 2)
        r["days_of_supply"] = round(float(r["days_of_supply"]), 1)

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.INVENTORY,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.inventory.risk",
    )

    table = TableResult(
        columns=cols,
        column_types={
            "sku_id": "string",
            "warehouse_id": "string",
            "on_hand": "integer",
            "net_position": "integer",
            "daily_demand": "float",
            "days_of_supply": "float",
            "stockout_risk_tier": "string",
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


def get_slow_moving_inventory(
    context: QueryContext,
    inventory_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    sales_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Identify slow-moving items and excess stock holding capital."""
    t0 = time.perf_counter()
    tool_name = "get_slow_moving_inventory"

    pos_df, curr, is_isolated, err_msg = _prepare_inventory_positions(context, inventory_df, products_df)
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if pos_df.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = pos_df.copy()
    if "velocity_tier" in df.columns:
        is_slow = df["velocity_tier"].astype(str).str.upper().isin(["SLOW", "DORMANT"])
    else:
        is_slow = pd.Series(True, index=df.index)

    df["capital_exposure"] = (
        df["on_hand"] * pd.to_numeric(df.get("unit_cost", 0.0), errors="coerce").fillna(0.0)
    ).round(2)

    # Filter to items with on_hand > 0 that are slow moving or have positive capital exposure
    slow_items = df[is_slow & (df["on_hand"] > 0)].sort_values(["sku_id", "warehouse_id"])

    if context.offset is not None:
        slow_items = slow_items.iloc[context.offset:]
    if context.limit is not None:
        slow_items = slow_items.iloc[:context.limit]

    cols = ["sku_id", "warehouse_id", "on_hand", "net_position", "capital_exposure"]
    if "velocity_tier" in slow_items.columns:
        cols.append("velocity_tier")
    if curr:
        cols.append("currency")
        slow_items["currency"] = curr

    rows = slow_items[cols].to_dict(orient="records")
    for r in rows:
        r["on_hand"] = int(r["on_hand"])
        r["net_position"] = int(r["net_position"])
        r["capital_exposure"] = float(r["capital_exposure"])

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.INVENTORY,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.inventory",
    )

    table = TableResult(
        columns=cols,
        column_types={
            "sku_id": "string",
            "warehouse_id": "string",
            "on_hand": "integer",
            "net_position": "integer",
            "capital_exposure": "float",
            "velocity_tier": "string",
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


def get_high_value_inventory(
    context: QueryContext,
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> QueryResponse:
    """Identify inventory items representing substantial capital concentration."""
    t0 = time.perf_counter()
    tool_name = "get_high_value_inventory"

    pos_df, curr, is_isolated, err_msg = _prepare_inventory_positions(context, inventory_df, products_df)
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if pos_df.empty or "unit_cost" not in pos_df.columns:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = pos_df.copy()
    costs = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(0.0)
    df["total_valuation"] = (df["on_hand"] * costs).round(2)
    df["unit_cost"] = costs.round(2)

    # Filter items with meaningful valuation (> 0)
    high_val = df[df["total_valuation"] > 0].sort_values(["sku_id", "warehouse_id"])

    if context.offset is not None:
        high_val = high_val.iloc[context.offset:]
    if context.limit is not None:
        high_val = high_val.iloc[:context.limit]

    cols = ["sku_id", "warehouse_id", "on_hand", "unit_cost", "total_valuation"]
    if curr:
        cols.append("currency")
        high_val["currency"] = curr

    rows = high_val[cols].to_dict(orient="records")
    for r in rows:
        r["on_hand"] = int(r["on_hand"])
        r["unit_cost"] = float(r["unit_cost"])
        r["total_valuation"] = float(r["total_valuation"])

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.INVENTORY,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.inventory",
    )

    table = TableResult(
        columns=cols,
        column_types={
            "sku_id": "string",
            "warehouse_id": "string",
            "on_hand": "integer",
            "unit_cost": "float",
            "total_valuation": "float",
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


def get_inventory_risk(
    context: QueryContext,
    inventory_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    sales_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Generate aggregate breakdown of inventory risk categories across portfolio."""
    t0 = time.perf_counter()
    tool_name = "get_inventory_risk"

    pos_df, curr, is_isolated, err_msg = _prepare_inventory_positions(context, inventory_df, products_df)
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if pos_df.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.INVENTORY,
            context=context,
            start_time=t0,
            currency=curr,
        )

    df = pos_df.copy()
    conditions = [
        (df["net_position"] <= 0),
        (df["net_position"] < 20),
        (df["on_hand"] > 500),
    ]
    choices = ["CRITICAL_STOCKOUT", "UNDERSTOCK", "OVERSTOCK"]
    df["risk_category"] = np.select(conditions, choices, default="HEALTHY")

    counts = df.groupby("risk_category")["sku_id"].count().reset_index()
    total = len(df)
    counts["percentage"] = (counts["sku_id"] / total) * 100.0 if total > 0 else 0.0

    items = [
        BreakdownItem(
            dimension_name="risk_category",
            dimension_value=str(row["risk_category"]),
            metric_value=int(row["sku_id"]),
            metric_name="sku_warehouse_count",
            percentage_of_total=round(float(row["percentage"]), 2),
            currency=None,
        )
        for _, row in counts.iterrows()
    ]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.INVENTORY,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.inventory.risk",
    )

    breakdown = BreakdownResult(
        metric_name="inventory_risk_distribution",
        display_name="Inventory Risk Category Distribution",
        dimension_name="risk_category",
        items=items,
        total_value=total,
        unit="pairs",
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
