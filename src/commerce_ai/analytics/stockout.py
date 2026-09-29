"""Stockout Detection and Demand Masking Engine.

Identifies periods where on-hand inventory constrained actual sales, distinguishes
observed demand from true unconstrained demand, and flags clean observations
eligible for time-series forecasting training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class StockoutConfig:
    """Configurable thresholds for stockout detection and demand masking."""

    stockout_threshold: int = 0  # Available units <= this value is a stockout
    low_stock_threshold: int = 5  # Available units <= this value is low stock
    mask_low_stock_exhausted: bool = True  # Flag constrained if sold out remaining low stock


def detect_stockout_periods(
    df: pd.DataFrame,
    config: Optional[StockoutConfig] = None,
) -> pd.DataFrame:
    """Identify stockout and low-stock periods across the daily demand dataset.

    Derives:
        - is_stockout (bool): Available inventory <= stockout_threshold
        - is_low_stock (bool): Available inventory <= low_stock_threshold
        - stockout_days (int): Running consecutive days in stockout for that event
        - stockout_event_id (str): Unique identifier for each contiguous stockout streak

    Args:
        df: Daily demand DataFrame with ['date', 'sku_id', 'warehouse_id', 'available_qty'].
        config: StockoutConfig instance with thresholds.

    Returns:
        pd.DataFrame: DataFrame augmented with stockout tracking columns.
    """
    cfg = config or StockoutConfig()
    result = df.copy()

    # Base indicators
    result["is_stockout"] = result["available_qty"] <= cfg.stockout_threshold
    result["is_low_stock"] = result["available_qty"] <= cfg.low_stock_threshold

    # Sort to ensure chronological order per series
    result = result.sort_values(["sku_id", "warehouse_id", "date"]).reset_index(drop=True)

    # Detect contiguous blocks of stockout
    # A new streak starts when is_stockout is True and prior row was not True
    prev_stockout = (
        result.groupby(["sku_id", "warehouse_id"])["is_stockout"].shift(1) == True
    )
    new_event = result["is_stockout"] & (~prev_stockout)

    # Cumulative sum gives an event counter per group
    result["_event_counter"] = (
        new_event.groupby([result["sku_id"], result["warehouse_id"]]).cumsum()
    )

    # Calculate consecutive days in stockout:
    # Cumsum within each contiguous True block
    # When is_stockout is False, reset to 0
    # In pandas: cumsum reset when False
    stockout_mask = result["is_stockout"]
    event_group = (~stockout_mask).cumsum()
    result["stockout_days"] = (
        stockout_mask.groupby(event_group).cumsum().astype(int)
    )

    # Generate readable stockout_event_id: e.g. "SO_SKU_0001_WH_01_1" or ""
    def format_event_id(row) -> str:
        if not row["is_stockout"] or row["_event_counter"] == 0:
            return ""
        return f"SO_{row['sku_id']}_{row['warehouse_id']}_{int(row['_event_counter'])}"

    # Vectorized event id construction
    event_id_series = np.where(
        result["is_stockout"] & (result["_event_counter"] > 0),
        "SO_" + result["sku_id"].astype(str) + "_" + result["warehouse_id"].astype(str) + "_" + result["_event_counter"].astype(str),
        "",
    )
    result["stockout_event_id"] = event_id_series
    result.drop(columns=["_event_counter"], inplace=True)

    return result


def mask_stockout_demand(
    df: pd.DataFrame,
    config: Optional[StockoutConfig] = None,
) -> pd.DataFrame:
    """Mask demand where inventory availability constrained sales.

    Derives:
        - is_demand_constrained (bool): True if demand was capped by lack of stock
        - forecast_training_eligible (bool): True if observation represents clean unconstrained demand

    Args:
        df: Daily demand DataFrame (will run detect_stockout_periods if not present).
        config: StockoutConfig instance with thresholds.

    Returns:
        pd.DataFrame: DataFrame with demand masking flags.
    """
    cfg = config or StockoutConfig()
    result = df.copy()

    if "is_stockout" not in result.columns:
        result = detect_stockout_periods(result, cfg)

    # Condition 1: Direct stockout (available_qty <= stockout_threshold)
    is_constrained = result["is_stockout"].copy()

    # Condition 2: Low stock exhaustion
    # If available inventory was low and units sold consumed available stock
    if cfg.mask_low_stock_exhausted and "units_sold" in result.columns:
        exhausted = result["is_low_stock"] & (result["units_sold"] > 0) & (result["units_sold"] >= result["available_qty"])
        is_constrained = is_constrained | exhausted

    result["is_demand_constrained"] = is_constrained
    result["forecast_training_eligible"] = ~is_constrained

    return result
