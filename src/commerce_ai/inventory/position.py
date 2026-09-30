"""Inventory Position Accounting Engine.

Calculates on-hand, on-order (inbound pipeline), reserved, and net inventory positions
per (SKU, warehouse) without double-counting delivered or cancelled purchase orders.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from commerce_ai.inventory.schemas import InventoryPositionRecord


OPEN_PO_STATUSES = {"PENDING", "IN_TRANSIT", "ORDERED", "OPEN", "PLACED"}
CLOSED_PO_STATUSES = {"DELIVERED", "CANCELLED", "COMPLETED", "RECEIVED"}


def calculate_inventory_positions(
    snapshots: pd.DataFrame,
    purchases: Optional[pd.DataFrame] = None,
    as_of_date: Optional[Union[str, date, datetime]] = None,
    use_po_for_on_order: bool = True,
) -> pd.DataFrame:
    """Calculate the net inventory position across SKU and warehouse combinations.

    Net Inventory Position Formula:
        Net Position = On-Hand (available_qty) + On-Order (in_transit / open POs) - Reserved (reserved_qty)

    Args:
        snapshots: Inventory snapshot DataFrame. Expected columns:
            ['sku_id', 'warehouse_id', 'available_qty'] and optionally
            ['snapshot_date', 'reserved_qty', 'in_transit_qty', 'damaged_qty'].
        purchases: Optional purchase orders DataFrame. Expected columns:
            ['sku_id', 'warehouse_id', 'quantity', 'status'] and optionally
            ['order_date', 'actual_delivery_date'].
        as_of_date: Optional cutoff date for evaluation. If provided, filters snapshots
            and purchases up to this date.
        use_po_for_on_order: If True and purchases DataFrame is supplied, calculates
            on-order directly from open POs rather than snapshot in_transit_qty,
            ensuring strict non-duplication of delivered or cancelled orders.

    Returns:
        pd.DataFrame: Augmented DataFrame containing:
            ['sku_id', 'warehouse_id', 'on_hand', 'reserved', 'on_order',
             'damaged', 'net_position', 'total_physical', 'as_of_date']
    """
    output_cols = [
        "sku_id",
        "warehouse_id",
        "on_hand",
        "reserved",
        "on_order",
        "damaged",
        "net_position",
        "total_physical",
        "as_of_date",
    ]

    if snapshots is None or snapshots.empty:
        return pd.DataFrame(columns=output_cols)

    inv_df = snapshots.copy()

    # Standardize column names
    if "sku_id" not in inv_df.columns or "warehouse_id" not in inv_df.columns:
        raise ValueError("Inventory snapshot DataFrame must contain 'sku_id' and 'warehouse_id'")
    if "available_qty" not in inv_df.columns:
        raise ValueError("Inventory snapshot DataFrame must contain 'available_qty'")

    # Ensure optional columns exist with safe defaults
    for col in ["reserved_qty", "in_transit_qty", "damaged_qty"]:
        if col not in inv_df.columns:
            inv_df[col] = 0

    # Parse and filter dates if provided
    cutoff_dt = pd.to_datetime(as_of_date).normalize() if as_of_date is not None else None

    if "snapshot_date" in inv_df.columns:
        inv_df["snapshot_date"] = pd.to_datetime(inv_df["snapshot_date"]).dt.normalize()
        if cutoff_dt is not None:
            inv_df = inv_df[inv_df["snapshot_date"] <= cutoff_dt]
        if inv_df.empty:
            return pd.DataFrame(columns=output_cols)
        # Select latest snapshot per (sku_id, warehouse_id)
        inv_df = (
            inv_df.sort_values("snapshot_date")
            .groupby(["sku_id", "warehouse_id"], as_index=False)
            .last()
        )
    else:
        # If no snapshot_date column, aggregate/deduplicate taking last
        inv_df = inv_df.groupby(["sku_id", "warehouse_id"], as_index=False).last()

    # Extract base quantities
    inv_df["on_hand"] = np.maximum(0, inv_df["available_qty"].fillna(0).astype(int))
    inv_df["reserved"] = np.maximum(0, inv_df["reserved_qty"].fillna(0).astype(int))
    inv_df["damaged"] = np.maximum(0, inv_df["damaged_qty"].fillna(0).astype(int))
    snapshot_in_transit = np.maximum(0, inv_df["in_transit_qty"].fillna(0).astype(int))

    # Calculate on-order quantity
    if use_po_for_on_order and purchases is not None and not purchases.empty:
        po_df = purchases.copy()
        req_po_cols = {"sku_id", "warehouse_id", "quantity", "status"}
        if req_po_cols.issubset(po_df.columns):
            po_df["status_norm"] = po_df["status"].astype(str).str.strip().str.upper()

            # Date filtering for POs
            if cutoff_dt is not None:
                if "order_date" in po_df.columns:
                    po_df["order_date_dt"] = pd.to_datetime(po_df["order_date"]).dt.normalize()
                    po_df = po_df[po_df["order_date_dt"] <= cutoff_dt]

                if "actual_delivery_date" in po_df.columns:
                    # If actual_delivery_date is on or before cutoff, order was already delivered
                    po_df["actual_delivery_dt"] = pd.to_datetime(po_df["actual_delivery_date"]).dt.normalize()
                    already_delivered = po_df["actual_delivery_dt"].notna() & (po_df["actual_delivery_dt"] <= cutoff_dt)
                    po_df = po_df[~already_delivered]

            # Filter for open PO statuses strictly (exclude DELIVERED, CANCELLED)
            open_pos = po_df[po_df["status_norm"].isin(OPEN_PO_STATUSES)]

            po_agg = (
                open_pos.groupby(["sku_id", "warehouse_id"], as_index=False)["quantity"]
                .sum()
                .rename(columns={"quantity": "po_on_order"})
            )

            merged = pd.merge(inv_df, po_agg, on=["sku_id", "warehouse_id"], how="left")
            merged["on_order"] = merged["po_on_order"].fillna(0).astype(int)
            inv_df = merged
        else:
            inv_df["on_order"] = snapshot_in_transit
    else:
        inv_df["on_order"] = snapshot_in_transit

    # Net inventory position: on_hand + on_order - reserved
    inv_df["net_position"] = inv_df["on_hand"] + inv_df["on_order"] - inv_df["reserved"]
    # Total physical inventory present in facility
    inv_df["total_physical"] = inv_df["on_hand"] + inv_df["reserved"] + inv_df["damaged"]
    inv_df["as_of_date"] = cutoff_dt.strftime("%Y-%m-%d") if cutoff_dt is not None else None

    result = inv_df[output_cols].sort_values(["sku_id", "warehouse_id"]).reset_index(drop=True)
    return result


def extract_position_records(df: pd.DataFrame) -> List[InventoryPositionRecord]:
    """Convert an inventory positions DataFrame to a list of validated InventoryPositionRecord models."""
    records = []
    for _, row in df.iterrows():
        records.append(
            InventoryPositionRecord(
                sku_id=str(row["sku_id"]),
                warehouse_id=str(row["warehouse_id"]),
                on_hand=int(row["on_hand"]),
                reserved=int(row["reserved"]),
                on_order=int(row["on_order"]),
                damaged=int(row["damaged"]),
                as_of_date=row["as_of_date"] if pd.notna(row.get("as_of_date")) else None,
            )
        )
    return records
