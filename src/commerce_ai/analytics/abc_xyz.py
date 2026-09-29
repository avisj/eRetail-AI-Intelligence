"""ABC and XYZ Demand Segmentation Engine.

Implements multi-dimensional product portfolio classification:
- ABC Analysis: Revenue/volume contribution ranking (Pareto principle)
- XYZ Analysis: Demand volatility and predictability (Coefficient of Variation)
- ABC-XYZ Matrix: 9-box operational segmentation for forecasting and inventory prioritization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ABCConfig:
    """Configurable thresholds for Pareto ABC classification."""

    a_threshold: float = 0.80  # Cumulative contribution cutoff for Class A
    b_threshold: float = 0.95  # Cumulative contribution cutoff for Class B
    metric: str = "revenue"     # Metric column to rank on ('revenue' or 'units_sold')


@dataclass
class XYZConfig:
    """Configurable thresholds for demand variability XYZ classification."""

    x_cv_threshold: float = 0.50  # Low variability (predictable): CV <= 0.50
    y_cv_threshold: float = 1.00  # Medium variability: 0.50 < CV <= 1.00
    filter_stockouts: bool = True  # Use only unconstrained demand (forecast_training_eligible == True)


def calculate_abc_classification(
    df: pd.DataFrame,
    config: Optional[ABCConfig] = None,
) -> pd.DataFrame:
    """Calculate dynamic ABC classification for SKUs based on cumulative contribution.

    Args:
        df: Daily demand or sales DataFrame containing 'sku_id' and the metric column.
        config: ABCConfig specifying cutoffs and target metric.

    Returns:
        pd.DataFrame: DataFrame with [sku_id, revenue, revenue_share, cumulative_revenue_share, abc_class].
    """
    cfg = config or ABCConfig()
    metric_col = cfg.metric

    if metric_col not in df.columns:
        if metric_col == "units_sold" and "quantity" in df.columns:
            metric_col = "quantity"
        elif metric_col == "quantity" and "units_sold" in df.columns:
            metric_col = "units_sold"
        else:
            raise ValueError(f"Metric column '{metric_col}' not found in DataFrame.")

    # Aggregate metric by SKU
    sku_agg = (
        df.groupby("sku_id", as_index=False)[metric_col]
        .sum()
        .rename(columns={metric_col: "revenue" if metric_col == "revenue" else metric_col})
    )

    # If ranking metric is not called revenue, rename internally
    val_col = "revenue" if metric_col == "revenue" else metric_col

    # Sort descending by metric, deterministic tie-breaking by sku_id
    sku_agg = sku_agg.sort_values([val_col, "sku_id"], ascending=[False, True]).reset_index(drop=True)

    total_val = sku_agg[val_col].sum()
    if total_val <= 0:
        # Edge case: zero total revenue/volume across all SKUs
        sku_agg["revenue_share"] = 0.0
        sku_agg["cumulative_revenue_share"] = 0.0
        sku_agg["abc_class"] = "C"
        return sku_agg

    sku_agg["revenue_share"] = sku_agg[val_col] / total_val
    sku_agg["cumulative_revenue_share"] = sku_agg["revenue_share"].cumsum()

    # Prior cumulative share before adding this SKU
    prior_cum_share = sku_agg["cumulative_revenue_share"].shift(1).fillna(0.0)

    # Class assignment based on standard Pareto boundaries
    conditions = [
        prior_cum_share < cfg.a_threshold,
        prior_cum_share < cfg.b_threshold,
    ]
    choices = ["A", "B"]
    sku_agg["abc_class"] = np.select(conditions, choices, default="C")

    return sku_agg


def calculate_xyz_classification(
    df: pd.DataFrame,
    config: Optional[XYZConfig] = None,
) -> pd.DataFrame:
    """Calculate XYZ classification based on demand coefficient of variation (CV = std / mean).

    Args:
        df: Daily demand or sales DataFrame containing 'sku_id' and 'units_sold' or 'quantity'.
        config: XYZConfig specifying CV thresholds and stockout filtering.

    Returns:
        pd.DataFrame: DataFrame with [sku_id, mean_daily_demand, std_daily_demand, coefficient_of_variation, xyz_class].
    """
    cfg = config or XYZConfig()
    data = df.copy()

    # Resolve demand column name (support units_sold or quantity)
    demand_col = "units_sold"
    if demand_col not in data.columns:
        if "quantity" in data.columns:
            demand_col = "quantity"
        else:
            raise ValueError("Neither 'units_sold' nor 'quantity' column found in DataFrame.")

    # Filter out stockout-constrained observations if requested and flag is present
    if cfg.filter_stockouts and "forecast_training_eligible" in data.columns:
        filtered = data[data["forecast_training_eligible"]]
        if not filtered.empty:
            data = filtered

    # Aggregate daily demand stats per SKU
    stats = (
        data.groupby("sku_id")[demand_col]
        .agg(
            mean_daily_demand="mean",
            std_daily_demand="std",
        )
        .reset_index()
    )

    stats["std_daily_demand"] = stats["std_daily_demand"].fillna(0.0)

    # Calculate Coefficient of Variation: CV = std / mean
    # Handle zero mean safely: if mean == 0, demand is dormant/zero, assigned to 'Z'
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_series = np.where(
            stats["mean_daily_demand"] > 0,
            stats["std_daily_demand"] / stats["mean_daily_demand"],
            np.nan,
        )

    stats["coefficient_of_variation"] = np.round(cv_series, 4)

    # Assign XYZ class
    # X: low variability, highly predictable (CV <= x_threshold)
    # Y: medium variability (x_threshold < CV <= y_threshold)
    # Z: high variability or zero demand (CV > y_threshold or NaN)
    conditions = [
        stats["coefficient_of_variation"].isna(),
        stats["coefficient_of_variation"] <= cfg.x_cv_threshold,
        stats["coefficient_of_variation"] <= cfg.y_cv_threshold,
    ]
    choices = ["Z", "X", "Y"]
    stats["xyz_class"] = np.select(conditions, choices, default="Z")

    return stats


def calculate_abc_xyz_matrix(
    df: pd.DataFrame,
    abc_config: Optional[ABCConfig] = None,
    xyz_config: Optional[XYZConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate unified ABC-XYZ classification, summary distribution, and 3x3 matrix crosstab.

    Args:
        df: Daily demand DataFrame with ['sku_id', 'revenue', 'units_sold'].
        abc_config: Configuration for ABC thresholding.
        xyz_config: Configuration for XYZ thresholding.

    Returns:
        Tuple containing:
            1. sku_classification_df: SKU-level table with both classifications and abc_xyz_class
            2. summary_table: Class-level metrics (sku_count, total_revenue, revenue_share, avg_demand, avg_cv)
            3. matrix_crosstab: 3x3 DataFrame matrix (rows: A, B, C; cols: X, Y, Z)
    """
    abc_df = calculate_abc_classification(df, abc_config)
    xyz_df = calculate_xyz_classification(df, xyz_config)

    # Merge on sku_id
    combined = pd.merge(abc_df, xyz_df, on="sku_id", how="inner")
    combined["abc_xyz_class"] = combined["abc_class"] + combined["xyz_class"]

    # Summary table
    summary = (
        combined.groupby("abc_xyz_class", as_index=False)
        .agg(
            sku_count=("sku_id", "count"),
            total_revenue=("revenue", "sum"),
            revenue_share=("revenue_share", "sum"),
            avg_daily_demand=("mean_daily_demand", "mean"),
            avg_cv=("coefficient_of_variation", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )

    # 3x3 Cross-tabulation Matrix (SKU counts)
    crosstab = (
        pd.crosstab(
            combined["abc_class"],
            combined["xyz_class"],
            margins=True,
            margins_name="Total",
        )
        .reindex(index=["A", "B", "C", "Total"], columns=["X", "Y", "Z", "Total"], fill_value=0)
    )

    return combined, summary, crosstab
