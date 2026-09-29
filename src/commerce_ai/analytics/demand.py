"""Demand Reconstruction Engine.

Constructs regularized daily SKU-Warehouse demand time series from sales and inventory data.
Guarantees continuous date grids where non-selling days are explicitly represented
as zero demand rather than dropped rows.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional, Union
import numpy as np
import pandas as pd


def build_daily_demand(
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    products: Optional[pd.DataFrame] = None,
    warehouses: Optional[pd.DataFrame] = None,
    start_date: Optional[Union[str, date, datetime]] = None,
    end_date: Optional[Union[str, date, datetime]] = None,
    skus: Optional[List[str]] = None,
    warehouse_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a complete, regularized daily SKU-warehouse demand dataset.

    Args:
        sales: Transactional sales DataFrame (must contain 'date', 'sku_id', 'warehouse_id', 'quantity', 'revenue').
        inventory: Inventory snapshot DataFrame (must contain 'snapshot_date', 'sku_id', 'warehouse_id', 'available_qty').
        products: Optional products DataFrame (used to infer active SKU list if provided).
        warehouses: Optional warehouses DataFrame (used to infer active warehouse list).
        start_date: Optional lower date bound (inclusive).
        end_date: Optional upper date bound (inclusive).
        skus: Optional subset of SKUs to restrict to.
        warehouse_ids: Optional subset of warehouses to restrict to.

    Returns:
        pd.DataFrame: Continuous daily time series at the (date, sku_id, warehouse_id) grain.
    """
    sales_df = sales.copy()
    inv_df = inventory.copy()

    # Standardize dates
    sales_df["date"] = pd.to_datetime(sales_df["date"]).dt.normalize()
    inv_df["snapshot_date"] = pd.to_datetime(inv_df["snapshot_date"]).dt.normalize()

    # Determine date bounds
    min_date = pd.to_datetime(start_date) if start_date is not None else min(sales_df["date"].min(), inv_df["snapshot_date"].min())
    max_date = pd.to_datetime(end_date) if end_date is not None else max(sales_df["date"].max(), inv_df["snapshot_date"].max())
    min_date = min_date.normalize()
    max_date = max_date.normalize()

    # Filter to date range
    sales_df = sales_df[(sales_df["date"] >= min_date) & (sales_df["date"] <= max_date)]
    inv_df = inv_df[(inv_df["snapshot_date"] >= min_date) & (inv_df["snapshot_date"] <= max_date)]

    # Determine SKU universe
    if skus is not None:
        target_skus = [str(s) for s in skus]
    elif products is not None and "sku_id" in products.columns:
        target_skus = products["sku_id"].astype(str).unique().tolist()
    else:
        target_skus = sorted(list(set(sales_df["sku_id"].astype(str)).union(set(inv_df["sku_id"].astype(str)))))

    # Determine Warehouse universe
    if warehouse_ids is not None:
        target_whs = [str(w) for w in warehouse_ids]
    elif warehouses is not None and "warehouse_id" in warehouses.columns:
        target_whs = warehouses["warehouse_id"].astype(str).unique().tolist()
    else:
        target_whs = sorted(list(set(sales_df["warehouse_id"].astype(str)).union(set(inv_df["warehouse_id"].astype(str)))))

    # Filter dataframes to target SKUs and Warehouses
    sales_df = sales_df[sales_df["sku_id"].isin(target_skus) & sales_df["warehouse_id"].isin(target_whs)]
    inv_df = inv_df[inv_df["sku_id"].isin(target_skus) & inv_df["warehouse_id"].isin(target_whs)]

    # Aggregate sales by (date, sku_id, warehouse_id)
    daily_sales = (
        sales_df.groupby(["date", "sku_id", "warehouse_id"], as_index=False)
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("revenue", "sum"),
        )
    )

    # Aggregate/clean inventory snapshots (in case of duplicates, keep last)
    inv_cols = ["available_qty", "reserved_qty", "in_transit_qty", "damaged_qty"]
    for col in inv_cols:
        if col not in inv_df.columns:
            inv_df[col] = 0

    inv_agg = (
        inv_df.groupby(["snapshot_date", "sku_id", "warehouse_id"], as_index=False)[inv_cols]
        .last()
        .rename(columns={"snapshot_date": "date"})
    )

    # Construct the full Cartesian grid: Dates x SKUs x Warehouses
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D", name="date")
    grid_index = pd.MultiIndex.from_product(
        [all_dates, target_skus, target_whs],
        names=["date", "sku_id", "warehouse_id"],
    )
    full_grid = pd.DataFrame(index=grid_index).reset_index()

    # Merge daily sales onto grid
    merged = pd.merge(
        full_grid,
        daily_sales,
        on=["date", "sku_id", "warehouse_id"],
        how="left",
    )
    merged["units_sold"] = merged["units_sold"].fillna(0).astype(int)
    merged["revenue"] = merged["revenue"].fillna(0.0).astype(float)

    # Merge inventory onto grid
    merged = pd.merge(
        merged,
        inv_agg,
        on=["date", "sku_id", "warehouse_id"],
        how="left",
    )

    # Sort deterministically for forward-filling
    merged = merged.sort_values(["sku_id", "warehouse_id", "date"]).reset_index(drop=True)

    # Forward fill inventory states across dates within each (sku_id, warehouse_id)
    # (Since inventory is typically snapshotted periodically, forward fill represents current on-hand position)
    merged[inv_cols] = (
        merged.groupby(["sku_id", "warehouse_id"])[inv_cols]
        .ffill()
        .bfill()
        .fillna(0)
        .astype(int)
    )

    # Final sorting by date, sku, warehouse
    merged = merged.sort_values(["date", "sku_id", "warehouse_id"]).reset_index(drop=True)

    return merged
