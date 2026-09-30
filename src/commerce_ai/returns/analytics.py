"""Deterministic Returns Analytics and Quality Audit Engine (Phase 5A).

Implements core returns intelligence logic:
- Data quality validation and audit
- Dimensional aggregations (SKU, Channel, Warehouse, SKU×Channel, SKU×Warehouse)
- Unit, order, and revenue return rate calculations
- Return reason categorizations and time series bucketing
- Sample size reliability guards and deterministic investigation flags
- Anti-leakage as_of_date chronological filtering
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.returns.schemas import (
    InvestigationFlag,
    ReturnMetricRecord,
    ReturnReasonBreakdown,
    ReturnsConfig,
    ReturnsDataQualityReport,
    ReturnTimeSeriesPoint,
)


def filter_by_as_of_date(
    df: pd.DataFrame,
    date_col: str,
    as_of_date: Optional[Union[str, date, datetime]],
) -> pd.DataFrame:
    """Filter records to ensure zero leakage past the evaluation as_of_date."""
    if df is None or df.empty or as_of_date is None or date_col not in df.columns:
        return df if df is not None else pd.DataFrame()

    if isinstance(as_of_date, str):
        target_date = datetime.fromisoformat(as_of_date).date()
    elif isinstance(as_of_date, datetime):
        target_date = as_of_date.date()
    else:
        target_date = as_of_date

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce").dt.date
    valid_mask = parsed_dates.notna() & (parsed_dates <= target_date)
    return df[valid_mask].copy()


def assess_returns_data_quality(
    returns_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    warehouses_df: Optional[pd.DataFrame] = None,
    channels_df: Optional[pd.DataFrame] = None,
    known_reasons: Optional[List[str]] = None,
) -> ReturnsDataQualityReport:
    """Audit returns dataset for missing values, invalid domains, duplicates, and FK referential integrity."""
    if returns_df is None or returns_df.empty:
        return ReturnsDataQualityReport(
            total_records=0,
            is_clean=True,
            issues=[],
        )

    total_records = len(returns_df)
    issues: List[Dict[str, Any]] = []

    # 1. Missing SKU
    missing_sku_mask = returns_df["sku_id"].isna() | (returns_df["sku_id"].astype(str).str.strip() == "")
    missing_sku_count = int(missing_sku_mask.sum())
    if missing_sku_count > 0:
        issues.append({"field": "sku_id", "issue": "Missing or blank SKU", "count": missing_sku_count})

    # 2. Missing Date
    missing_date_mask = returns_df["return_date"].isna()
    missing_date_count = int(missing_date_mask.sum())
    if missing_date_count > 0:
        issues.append({"field": "return_date", "issue": "Missing or null return date", "count": missing_date_count})

    # 3. Missing Order ID
    missing_order_mask = returns_df["order_id"].isna() | (returns_df["order_id"].astype(str).str.strip() == "")
    missing_order_count = int(missing_order_mask.sum())
    if missing_order_count > 0:
        issues.append({"field": "order_id", "issue": "Missing or blank order ID", "count": missing_order_count})

    # 4. Invalid Quantity (<= 0)
    qty_numeric = pd.to_numeric(returns_df["quantity"], errors="coerce")
    invalid_qty_mask = qty_numeric.isna() | (qty_numeric <= 0)
    invalid_qty_count = int(invalid_qty_mask.sum())
    if invalid_qty_count > 0:
        issues.append({"field": "quantity", "issue": "Non-positive or invalid quantity", "count": invalid_qty_count})

    # 5. Duplicate return_id
    duplicate_count = 0
    if "return_id" in returns_df.columns:
        valid_return_ids = returns_df["return_id"].dropna().astype(str)
        duplicate_count = int(valid_return_ids.duplicated().sum())
        if duplicate_count > 0:
            issues.append({"field": "return_id", "issue": "Duplicate return_id detected", "count": duplicate_count})

    # 6. Referential Integrity (SKU)
    unknown_sku_count = 0
    if products_df is not None and not products_df.empty and "sku_id" in products_df.columns:
        valid_skus = set(products_df["sku_id"].dropna().astype(str))
        ret_skus = returns_df["sku_id"].dropna().astype(str)
        unknown_sku_mask = ~ret_skus.isin(valid_skus)
        unknown_sku_count = int(unknown_sku_mask.sum())
        if unknown_sku_count > 0:
            issues.append({"field": "sku_id", "issue": "Unknown SKU not found in products master", "count": unknown_sku_count})

    # 7. Referential Integrity (Warehouse)
    unknown_wh_count = 0
    if warehouses_df is not None and not warehouses_df.empty and "warehouse_id" in warehouses_df.columns:
        valid_whs = set(warehouses_df["warehouse_id"].dropna().astype(str))
        ret_whs = returns_df["warehouse_id"].dropna().astype(str)
        unknown_wh_mask = ~ret_whs.isin(valid_whs)
        unknown_wh_count = int(unknown_wh_mask.sum())
        if unknown_wh_count > 0:
            issues.append({"field": "warehouse_id", "issue": "Unknown warehouse not found in warehouses master", "count": unknown_wh_count})

    # 8. Referential Integrity (Channel)
    unknown_ch_count = 0
    if channels_df is not None and not channels_df.empty and "channel_id" in channels_df.columns:
        valid_chs = set(channels_df["channel_id"].dropna().astype(str))
        ret_chs = returns_df["channel_id"].dropna().astype(str)
        unknown_ch_mask = ~ret_chs.isin(valid_chs)
        unknown_ch_count = int(unknown_ch_mask.sum())
        if unknown_ch_count > 0:
            issues.append({"field": "channel_id", "issue": "Unknown channel not found in channels master", "count": unknown_ch_count})

    # 9. Return Reasons
    unknown_reason_count = 0
    if known_reasons and "reason" in returns_df.columns:
        valid_reasons = set(known_reasons)
        ret_reasons = returns_df["reason"].dropna().astype(str)
        unknown_reason_mask = ~ret_reasons.isin(valid_reasons)
        unknown_reason_count = int(unknown_reason_mask.sum())
        if unknown_reason_count > 0:
            issues.append({"field": "reason", "issue": "Unknown reason category", "count": unknown_reason_count})

    is_clean = len(issues) == 0

    return ReturnsDataQualityReport(
        total_records=total_records,
        missing_sku_count=missing_sku_count,
        missing_date_count=missing_date_count,
        missing_order_id_count=missing_order_count,
        invalid_quantity_count=invalid_qty_count,
        unknown_sku_count=unknown_sku_count,
        unknown_warehouse_count=unknown_wh_count,
        unknown_channel_count=unknown_ch_count,
        duplicate_record_count=duplicate_count,
        unknown_reason_count=unknown_reason_count,
        issues=issues,
        is_clean=is_clean,
    )


def enrich_returns_with_monetary_value(
    returns_df: pd.DataFrame,
    sales_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Attach estimated monetary unit price and total return valuation to return records."""
    if returns_df.empty:
        df = returns_df.copy()
        df["estimated_unit_price"] = pd.Series(dtype=float)
        df["estimated_return_value"] = pd.Series(dtype=float)
        return df

    df = returns_df.copy()
    df["estimated_unit_price"] = np.nan

    # 1. First priority: Exact match from sales order lines (order_id, sku_id)
    if sales_df is not None and not sales_df.empty:
        s_df = sales_df.copy()
        if "order_id" in s_df.columns and "sku_id" in s_df.columns:
            # Calculate realized unit price: revenue / quantity if revenue exists, else unit_price
            if "revenue" in s_df.columns and "quantity" in s_df.columns:
                s_df["calc_unit_price"] = np.where(
                    s_df["quantity"] > 0,
                    s_df["revenue"] / s_df["quantity"],
                    s_df.get("unit_price", np.nan),
                )
            else:
                s_df["calc_unit_price"] = s_df.get("unit_price", np.nan)

            price_lookup = s_df.groupby(["order_id", "sku_id"])["calc_unit_price"].mean().to_dict()

            def get_order_price(row):
                key = (str(row.get("order_id")), str(row.get("sku_id")))
                return price_lookup.get(key, np.nan)

            df["estimated_unit_price"] = df.apply(get_order_price, axis=1)

    # 2. Second priority: Fallback to product catalog master selling price
    if products_df is not None and not products_df.empty and "sku_id" in products_df.columns:
        cat_price_col = "selling_price" if "selling_price" in products_df.columns else (
            "unit_cost" if "unit_cost" in products_df.columns else None
        )
        if cat_price_col:
            cat_price_map = products_df.set_index("sku_id")[cat_price_col].dropna().to_dict()
            missing_price_mask = df["estimated_unit_price"].isna()
            df.loc[missing_price_mask, "estimated_unit_price"] = df.loc[missing_price_mask, "sku_id"].astype(str).map(cat_price_map)

    # Calculate line return value: quantity * estimated_unit_price
    df["estimated_return_value"] = np.where(
        df["estimated_unit_price"].notna(),
        df["quantity"] * df["estimated_unit_price"],
        np.nan,
    )
    return df


def compute_reason_breakdowns(
    returns_df: pd.DataFrame,
    total_returned_units: int,
    total_return_events: int,
) -> List[ReturnReasonBreakdown]:
    """Compute reason distribution and share of returns."""
    if returns_df.empty or "reason" not in returns_df.columns:
        return []

    breakdowns: List[ReturnReasonBreakdown] = []
    reason_groups = returns_df.groupby("reason")

    for reason_val, group in reason_groups:
        reason_str = str(reason_val)
        ret_units = int(group["quantity"].sum())
        ret_count = len(group)

        unit_share = (ret_units / total_returned_units * 100.0) if total_returned_units > 0 else 0.0
        event_share = (ret_count / total_return_events * 100.0) if total_return_events > 0 else 0.0

        val_sum = group["estimated_return_value"].sum(min_count=1) if "estimated_return_value" in group.columns else np.nan
        est_val = round(float(val_sum), 2) if pd.notna(val_sum) else None

        breakdowns.append(
            ReturnReasonBreakdown(
                reason=reason_str,
                returned_units=ret_units,
                return_count=ret_count,
                percentage_of_units=round(unit_share, 2),
                percentage_of_returns=round(event_share, 2),
                estimated_return_value=est_val,
            )
        )

    breakdowns.sort(key=lambda b: -b.returned_units)
    return breakdowns


def compute_time_series_points(
    returns_df: pd.DataFrame,
    sales_df: Optional[pd.DataFrame] = None,
    freq: str = "M",
) -> List[ReturnTimeSeriesPoint]:
    """Aggregate returns and sales into periodic time-series buckets (daily, weekly, monthly)."""
    # Build timeline series
    ret_dates = pd.to_datetime(returns_df["return_date"], errors="coerce") if not returns_df.empty else pd.Series(dtype="datetime64[ns]")
    sale_dates = pd.to_datetime(sales_df["date"], errors="coerce") if (sales_df is not None and not sales_df.empty) else pd.Series(dtype="datetime64[ns]")

    combined_dates = pd.concat([ret_dates, sale_dates]).dropna()
    if combined_dates.empty:
        return []

    # Format period buckets
    def format_period(dt_val: pd.Timestamp) -> str:
        if freq == "D":
            return dt_val.strftime("%Y-%m-%d")
        elif freq == "W":
            return f"{dt_val.year}-W{dt_val.isocalendar().week:02d}"
        else:  # Monthly
            return dt_val.strftime("%Y-%m")

    r_copy = returns_df.copy()
    if not r_copy.empty:
        r_copy["period"] = pd.to_datetime(r_copy["return_date"]).apply(format_period)
        r_grouped = r_copy.groupby("period").agg(
            returned_units=("quantity", "sum"),
            return_count=("quantity", "count"),
            return_value=("estimated_return_value", lambda x: x.sum(min_count=1)),
        ).to_dict(orient="index")
    else:
        r_grouped = {}

    s_grouped = {}
    if sales_df is not None and not sales_df.empty:
        s_copy = sales_df.copy()
        s_copy["period"] = pd.to_datetime(s_copy["date"]).apply(format_period)
        revenue_col = "revenue" if "revenue" in s_copy.columns else ("unit_price" if "unit_price" in s_copy.columns else None)
        if revenue_col:
            s_grouped = s_copy.groupby("period").agg(
                sold_units=("quantity", "sum"),
                sales_value=(revenue_col, lambda x: x.sum(min_count=1)),
            ).to_dict(orient="index")
        else:
            s_grouped = s_copy.groupby("period").agg(
                sold_units=("quantity", "sum"),
            ).to_dict(orient="index")
            for k in s_grouped:
                s_grouped[k]["sales_value"] = np.nan

    all_periods = sorted(set(list(r_grouped.keys()) + list(s_grouped.keys())))
    points: List[ReturnTimeSeriesPoint] = []

    for p in all_periods:
        r_info = r_grouped.get(p, {})
        s_info = s_grouped.get(p, {})

        ret_u = int(r_info.get("returned_units", 0))
        ret_c = int(r_info.get("return_count", 0))
        ret_v = float(r_info.get("return_value", np.nan))
        est_ret_v = round(ret_v, 2) if pd.notna(ret_v) else None

        sold_u = int(s_info.get("sold_units", 0))
        sales_v = float(s_info.get("sales_value", np.nan))
        est_sales_v = round(sales_v, 2) if pd.notna(sales_v) else None

        unit_rate = round(ret_u / sold_u, 4) if sold_u > 0 else None

        points.append(
            ReturnTimeSeriesPoint(
                period=p,
                sold_units=sold_u,
                returned_units=ret_u,
                return_rate=unit_rate,
                return_count=ret_c,
                sales_value=est_sales_v,
                return_value=est_ret_v,
            )
        )

    return points


def compute_dimensional_metrics(
    dimension_name: str,
    group_cols: List[str],
    returns_df: pd.DataFrame,
    sales_df: Optional[pd.DataFrame] = None,
    previous_period_returns_df: Optional[pd.DataFrame] = None,
    previous_period_sales_df: Optional[pd.DataFrame] = None,
    config: Optional[ReturnsConfig] = None,
) -> List[ReturnMetricRecord]:
    """Aggregate returns and sales across specified dimension keys (SKU, Channel, Warehouse, etc.)."""
    cfg = config or ReturnsConfig()

    # 1. Aggregate Sales by dimension
    sales_agg: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    if sales_df is not None and not sales_df.empty:
        has_all_cols = all(c in sales_df.columns for c in group_cols)
        if has_all_cols:
            s_df = sales_df.copy()
            rev_col = "revenue" if "revenue" in s_df.columns else ("unit_price" if "unit_price" in s_df.columns else None)
            for key_tuple, group in s_df.groupby(group_cols):
                normalized_key = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
                str_key = tuple(str(k) for k in normalized_key)
                sold_qty = int(group["quantity"].sum())
                rev_sum = group[rev_col].sum(min_count=1) if rev_col and rev_col in group.columns else np.nan
                orders = set(group["order_id"].dropna().astype(str)) if "order_id" in group.columns else set()

                sales_agg[str_key] = {
                    "sold_units": sold_qty,
                    "sales_value": float(rev_sum) if pd.notna(rev_sum) else None,
                    "order_count": len(orders),
                    "orders": orders,
                }

    # 2. Aggregate Returns by dimension
    returns_agg: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    if not returns_df.empty:
        has_all_cols = all(c in returns_df.columns for c in group_cols)
        if has_all_cols:
            r_df = returns_df.copy()
            for key_tuple, group in r_df.groupby(group_cols):
                normalized_key = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
                str_key = tuple(str(k) for k in normalized_key)
                ret_qty = int(group["quantity"].sum())
                ret_cnt = len(group)
                val_sum = group["estimated_return_value"].sum(min_count=1) if "estimated_return_value" in group.columns else np.nan
                ret_orders = set(group["order_id"].dropna().astype(str)) if "order_id" in group.columns else set()

                # Top return reason
                top_reason = None
                if "reason" in group.columns and not group["reason"].dropna().empty:
                    top_reason = str(group["reason"].mode().iloc[0])

                returns_agg[str_key] = {
                    "returned_units": ret_qty,
                    "return_count": ret_cnt,
                    "return_value": float(val_sum) if pd.notna(val_sum) else None,
                    "top_return_reason": top_reason,
                    "return_orders": ret_orders,
                }

    # 3. Calculate previous period rates for trend comparison if available
    prev_rates: Dict[Tuple[str, ...], float] = {}
    if (
        previous_period_returns_df is not None
        and not previous_period_returns_df.empty
        and previous_period_sales_df is not None
        and not previous_period_sales_df.empty
    ):
        prev_s = previous_period_sales_df.groupby(group_cols)["quantity"].sum().to_dict()
        prev_r = previous_period_returns_df.groupby(group_cols)["quantity"].sum().to_dict()
        for k_raw, s_qty in prev_s.items():
            norm_k = k_raw if isinstance(k_raw, tuple) else (k_raw,)
            str_k = tuple(str(x) for x in norm_k)
            r_qty = prev_r.get(k_raw, 0)
            if s_qty > 0:
                prev_rates[str_k] = r_qty / s_qty

    # 4. Formulate composite records across all observed dimension keys
    all_keys = sorted(set(list(sales_agg.keys()) + list(returns_agg.keys())))
    records: List[ReturnMetricRecord] = []

    for k in all_keys:
        s_data = sales_agg.get(k, {})
        r_data = returns_agg.get(k, {})

        sold_u = int(s_data.get("sold_units", 0))
        sales_val = s_data.get("sales_value")
        order_cnt = int(s_data.get("order_count", 0))
        sales_order_set = s_data.get("orders", set())

        ret_u = int(r_data.get("returned_units", 0))
        ret_cnt = int(r_data.get("return_count", 0))
        ret_val = r_data.get("return_value")
        top_reason = r_data.get("top_return_reason")
        ret_order_set = r_data.get("return_orders", set())

        # Unit Return Rate: returned_units / sold_units
        if sold_u > 0:
            unit_ret_rate = round(ret_u / sold_u, 4)
        elif sold_u == 0 and ret_u == 0:
            unit_ret_rate = 0.0
        else:
            unit_ret_rate = None

        # Order Return Rate: distinct returned orders / distinct sales orders
        if order_cnt > 0:
            matched_orders = len(ret_order_set & sales_order_set) if sales_order_set else len(ret_order_set)
            order_ret_rate = round(matched_orders / order_cnt, 4)
        else:
            order_ret_rate = None

        # Revenue Return Rate: return_value / sales_value
        if sales_val is not None and sales_val > 0 and ret_val is not None:
            rev_ret_rate = round(ret_val / sales_val, 4)
        else:
            rev_ret_rate = None

        # Sample size guard
        is_sufficient = sold_u >= cfg.min_sold_units_threshold

        # Investigation flags
        flags: List[str] = []
        if not is_sufficient and sold_u > 0:
            flags.append(InvestigationFlag.INSUFFICIENT_SAMPLE.value)

        if sold_u == 0 and ret_u > 0:
            flags.append(InvestigationFlag.ZERO_SALES_RECORDED.value)

        if is_sufficient and unit_ret_rate is not None and unit_ret_rate >= cfg.high_return_rate_threshold:
            flags.append(InvestigationFlag.HIGH_RETURN_RATE.value)

        if ret_u >= cfg.high_return_volume_threshold:
            flags.append(InvestigationFlag.HIGH_RETURN_VOLUME.value)

        if sold_u > 0 and ret_u == 0:
            flags.append(InvestigationFlag.NO_RETURN_DATA.value)

        # Trend & Rate Change vs Previous Period
        rate_change: Optional[float] = None
        trend_status: Optional[str] = None
        if k in prev_rates and unit_ret_rate is not None:
            rate_change = round(unit_ret_rate - prev_rates[k], 4)
            if rate_change >= cfg.increasing_rate_delta_threshold:
                flags.append(InvestigationFlag.RETURN_RATE_INCREASING.value)
                trend_status = "INCREASING"
            elif rate_change <= -cfg.increasing_rate_delta_threshold:
                trend_status = "DECREASING"
            else:
                trend_status = "STABLE"
        else:
            trend_status = "INSUFFICIENT_DATA"

        # Format dimension identifiers
        dim_key = ":".join(k)
        sku_val = k[0] if "sku_id" in group_cols else None
        ch_val = k[group_cols.index("channel_id")] if "channel_id" in group_cols else None
        wh_val = k[group_cols.index("warehouse_id")] if "warehouse_id" in group_cols else None

        records.append(
            ReturnMetricRecord(
                dimension=dimension_name,
                key=dim_key,
                sku_id=sku_val,
                channel_id=ch_val,
                warehouse_id=wh_val,
                sold_units=sold_u,
                returned_units=ret_u,
                return_rate=unit_ret_rate,
                return_count=ret_cnt,
                order_count=order_cnt,
                order_return_rate=order_ret_rate,
                sales_value=round(sales_val, 2) if sales_val is not None else None,
                return_value=round(ret_val, 2) if ret_val is not None else None,
                revenue_return_rate=rev_ret_rate,
                top_return_reason=top_reason,
                is_sufficient_sample=is_sufficient,
                investigation_flags=flags,
                return_rate_change=rate_change,
                trend=trend_status,
            )
        )

    # Sort descending by returned_units
    records.sort(key=lambda r: -r.returned_units)
    return records
