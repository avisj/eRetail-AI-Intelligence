"""Filtering utilities and point-in-time / currency guards for Query Layer (Phase 7A).

Enforces:
1. Point-in-time chronological filtering: record_date <= as_of_date (no future data leakage)
2. Date-range bounding: start_date <= record_date <= effective_end_date
3. Multi-currency isolation: prevents mixed currency aggregations without FX conversions
4. Dimensional filtering across SKUs, Warehouses, Channels, Categories, Brands, Suppliers
5. Analytical segment filtering across Velocity tiers, ABC, and XYZ classes
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import pandas as pd
import numpy as np

from commerce_ai.query_layer.context import QueryContext


def apply_point_in_time_filter(
    df: pd.DataFrame,
    date_col: str,
    as_of_date: Optional[str],
) -> pd.DataFrame:
    """Strictly filter DataFrame so no records after as_of_date are included."""
    if df.empty or as_of_date is None or date_col not in df.columns:
        return df

    # Convert to string format if series is datetime
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        cutoff = pd.to_datetime(as_of_date)
        return df[df[date_col] <= cutoff].copy()
    else:
        # String comparison works for ISO YYYY-MM-DD
        cutoff_str = str(as_of_date)[:10]
        dates = df[date_col].astype(str).str[:10]
        return df[dates <= cutoff_str].copy()


def apply_date_range_filter(
    df: pd.DataFrame,
    date_col: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    """Apply point-in-time and start/end temporal boundaries."""
    if df.empty or date_col not in df.columns:
        return df

    filtered = df

    # 1. Point-in-time boundary (Anti-leakage)
    if as_of_date is not None:
        filtered = apply_point_in_time_filter(filtered, date_col, as_of_date)
        if filtered.empty:
            return filtered

    # Determine effective upper bound
    effective_end = None
    if as_of_date is not None and end_date is not None:
        effective_end = min(str(as_of_date)[:10], str(end_date)[:10])
    elif as_of_date is not None:
        effective_end = str(as_of_date)[:10]
    elif end_date is not None:
        effective_end = str(end_date)[:10]

    # Apply date boundaries
    if pd.api.types.is_datetime64_any_dtype(filtered[date_col]):
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            filtered = filtered[filtered[date_col] >= start_dt]
        if effective_end is not None:
            end_dt = pd.to_datetime(effective_end)
            filtered = filtered[filtered[date_col] <= end_dt]
    else:
        date_series = filtered[date_col].astype(str).str[:10]
        if start_date is not None:
            filtered = filtered[date_series >= str(start_date)[:10]]
        if effective_end is not None:
            date_series = filtered[date_col].astype(str).str[:10]
            filtered = filtered[date_series <= str(effective_end)[:10]]

    return filtered.copy()


def check_currency_isolation(
    df: pd.DataFrame,
    currency_col: str = "currency",
    requested_currency: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str], bool, Optional[str]]:
    """Validate currency isolation without performing arbitrary FX conversions.
    
    Returns:
        (filtered_df, effective_currency, is_isolated, error_message)
    """
    if df.empty:
        return df, requested_currency, True, None

    if currency_col not in df.columns:
        # No currency column, assume single or N/A
        return df, requested_currency, True, None

    # Standardize currency column strings
    curr_series = df[currency_col].dropna().astype(str).str.strip().str.upper()

    if requested_currency:
        req = requested_currency.strip().upper()
        matching = df[curr_series == req]
        return matching.copy(), req, True, None

    unique_currencies = sorted(curr_series.unique().tolist())
    if len(unique_currencies) > 1:
        msg = (
            f"Multi-currency dataset detected with currencies: {unique_currencies}. "
            f"Queries require an explicit currency filter to ensure currency isolation. "
            f"Cross-currency aggregation and FX conversion are strictly prohibited in Phase 7A."
        )
        return df, None, False, msg

    isolated_currency = unique_currencies[0] if unique_currencies else None
    return df, isolated_currency, True, None


def apply_context_filters(
    df: pd.DataFrame,
    context: QueryContext,
    date_col: Optional[str] = None,
    currency_col: Optional[str] = "currency",
    sku_col: str = "sku_id",
    warehouse_col: str = "warehouse_id",
    channel_col: str = "channel_id",
    products_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[str], bool, Optional[str]]:
    """Apply all relevant context filters (temporal, currency, dimensional) to DataFrame.
    
    Returns:
        (filtered_df, effective_currency, is_isolated, error_message)
    """
    if df.empty:
        return df, context.effective_currency, True, None

    filtered = df

    # 1. Temporal filter
    if date_col and date_col in filtered.columns:
        filtered = apply_date_range_filter(
            filtered,
            date_col=date_col,
            start_date=context.iso_start_date,
            end_date=context.iso_end_date,
            as_of_date=context.iso_as_of_date,
        )
        if filtered.empty:
            return filtered, context.effective_currency, True, None

    # 2. Currency Isolation check
    filtered, effective_curr, is_isolated, err = check_currency_isolation(
        filtered,
        currency_col=currency_col or "currency",
        requested_currency=context.effective_currency,
    )
    if not is_isolated:
        return filtered, None, False, err

    # 3. Direct Dimension Filters
    if context.sku_ids and sku_col in filtered.columns:
        filtered = filtered[filtered[sku_col].astype(str).isin(context.sku_ids)]

    if context.warehouse_ids and warehouse_col in filtered.columns:
        filtered = filtered[filtered[warehouse_col].astype(str).isin(context.warehouse_ids)]

    if context.channel_ids and channel_col in filtered.columns:
        filtered = filtered[filtered[channel_col].astype(str).isin(context.channel_ids)]

    # 4. Product-level Attribute Filters (category_id, brand, supplier_id)
    product_filters_active = (
        context.category_ids or context.brands or context.supplier_ids
    )
    if product_filters_active:
        # Check if columns are already in df
        has_cat = "category_id" in filtered.columns or "category" in filtered.columns
        has_brand = "brand" in filtered.columns
        has_supp = "supplier_id" in filtered.columns

        cat_col = "category_id" if "category_id" in filtered.columns else "category"

        # If columns not in filtered df but products_df is provided and sku_col exists, filter SKUs
        if (not (has_cat and has_brand and has_supp)) and products_df is not None and sku_col in filtered.columns:
            prod_filtered = products_df
            if context.category_ids:
                p_cat_col = "category_id" if "category_id" in prod_filtered.columns else "category"
                if p_cat_col in prod_filtered.columns:
                    prod_filtered = prod_filtered[prod_filtered[p_cat_col].astype(str).isin(context.category_ids)]
            if context.brands and "brand" in prod_filtered.columns:
                prod_filtered = prod_filtered[prod_filtered["brand"].astype(str).isin(context.brands)]
            if context.supplier_ids and "supplier_id" in prod_filtered.columns:
                prod_filtered = prod_filtered[prod_filtered["supplier_id"].astype(str).isin(context.supplier_ids)]

            allowed_skus = set(prod_filtered["sku_id"].astype(str).unique())
            filtered = filtered[filtered[sku_col].astype(str).isin(allowed_skus)]
        else:
            if context.category_ids and cat_col in filtered.columns:
                filtered = filtered[filtered[cat_col].astype(str).isin(context.category_ids)]
            if context.brands and "brand" in filtered.columns:
                filtered = filtered[filtered["brand"].astype(str).isin(context.brands)]
            if context.supplier_ids and "supplier_id" in filtered.columns:
                filtered = filtered[filtered["supplier_id"].astype(str).isin(context.supplier_ids)]

    return filtered.copy(), effective_curr, True, None
