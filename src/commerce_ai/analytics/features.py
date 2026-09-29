"""Feature Engineering Engine for Time-Series Demand Forecasting.

Implements leakage-free lag features, rolling statistics, multi-horizon trend indicators,
demand intermittency categorization, and the master unified demand feature pipeline.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd

from commerce_ai.analytics.stockout import StockoutConfig, mask_stockout_demand
from commerce_ai.analytics.abc_xyz import (
    ABCConfig,
    XYZConfig,
    calculate_abc_xyz_matrix,
)
from commerce_ai.analytics.calendar import (
    BusinessEvent,
    build_calendar_features,
    build_event_features,
)


def build_lag_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    target_col: str = "units_sold",
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate leakage-free demand lag features.

    Lag k at day T is strictly the observation from day T - k.
    Features generated: demand_lag_{k}.

    Args:
        df: Daily demand DataFrame containing date and grouping columns.
        lags: List of integer lags (default: [1, 7, 14, 28, 30]).
        target_col: Target demand column to lag.
        group_cols: Columns defining the time series entity (default: ['sku_id', 'warehouse_id']).

    Returns:
        pd.DataFrame: DataFrame augmented with lag features.
    """
    result = df.copy()
    target_lags = lags or [1, 7, 14, 28, 30]
    groups = group_cols or ["sku_id", "warehouse_id"]

    # Ensure chronological sort within series
    result = result.sort_values(groups + ["date"]).reset_index(drop=True)

    grouped = result.groupby(groups)[target_col]
    for k in target_lags:
        result[f"demand_lag_{k}"] = grouped.shift(k)

    return result


def build_rolling_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    target_col: str = "units_sold",
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate leakage-free rolling window statistics (mean, std).

    CRITICAL LEAKAGE PROTECTION:
    All rolling windows operate on demand shifted by 1 day (shift(1)).
    The feature for day T reflects window statistics over [T - w, T - 1] and
    never includes day T itself.

    Args:
        df: Daily demand DataFrame containing date and grouping columns.
        windows: List of window sizes in days (default: [7, 14, 30, 90]).
        target_col: Target demand column.
        group_cols: Columns defining the time series entity (default: ['sku_id', 'warehouse_id']).

    Returns:
        pd.DataFrame: DataFrame augmented with rolling mean and std features.
    """
    result = df.copy()
    target_windows = windows or [7, 14, 30, 90]
    groups = group_cols or ["sku_id", "warehouse_id"]

    result = result.sort_values(groups + ["date"]).reset_index(drop=True)

    # Shift by 1 day first to guarantee strict causality (no data leakage)
    shifted = result.groupby(groups)[target_col].shift(1)

    for w in target_windows:
        # Grouped rolling calculation on shifted series
        rolling_obj = shifted.groupby([result[g] for g in groups]).rolling(window=w, min_periods=1)
        
        # Reset index to match original dataframe
        rolling_mean = rolling_obj.mean().reset_index(level=list(range(len(groups))), drop=True)
        rolling_std = rolling_obj.std().reset_index(level=list(range(len(groups))), drop=True).fillna(0.0)

        result[f"demand_rolling_mean_{w}"] = np.round(rolling_mean, 4)
        result[f"demand_rolling_std_{w}"] = np.round(rolling_std, 4)

    return result


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate demand trend ratios and percentage momentum indicators with safe division.

    Derived features:
        - trend_7_vs_30: rolling_mean_7 / rolling_mean_30
        - trend_30_vs_90: rolling_mean_30 / rolling_mean_90
        - demand_growth_7d: (lag_1 - lag_7) / lag_7
        - demand_growth_30d: (lag_1 - lag_30) / lag_30

    Returns:
        pd.DataFrame: DataFrame with trend indicators (no infinite values).
    """
    result = df.copy()

    # Trend 7 vs 30
    if "demand_rolling_mean_7" in result.columns and "demand_rolling_mean_30" in result.columns:
        denom = result["demand_rolling_mean_30"].replace(0, np.nan)
        ratio = result["demand_rolling_mean_7"] / denom
        result["trend_7_vs_30"] = np.round(ratio.replace([np.inf, -np.inf], np.nan), 4)

    # Trend 30 vs 90
    if "demand_rolling_mean_30" in result.columns and "demand_rolling_mean_90" in result.columns:
        denom = result["demand_rolling_mean_90"].replace(0, np.nan)
        ratio = result["demand_rolling_mean_30"] / denom
        result["trend_30_vs_90"] = np.round(ratio.replace([np.inf, -np.inf], np.nan), 4)

    # 7-day momentum
    if "demand_lag_1" in result.columns and "demand_lag_7" in result.columns:
        denom = result["demand_lag_7"].replace(0, np.nan)
        growth = (result["demand_lag_1"] - result["demand_lag_7"]) / denom
        result["demand_growth_7d"] = np.round(growth.replace([np.inf, -np.inf], np.nan), 4)

    # 30-day momentum
    if "demand_lag_1" in result.columns and "demand_lag_30" in result.columns:
        denom = result["demand_lag_30"].replace(0, np.nan)
        growth = (result["demand_lag_1"] - result["demand_lag_30"]) / denom
        result["demand_growth_30d"] = np.round(growth.replace([np.inf, -np.inf], np.nan), 4)

    return result


def calculate_intermittency_metrics(
    df: pd.DataFrame,
    target_col: str = "units_sold",
    group_col: str = "sku_id",
) -> pd.DataFrame:
    """Calculate demand intermittency and classify time series predictability.

    Metrics:
        - total_observations: total days recorded
        - zero_demand_days: count of zero-demand days
        - non_zero_demand_days: count of days with sales > 0
        - demand_occurrence_rate: non_zero_demand_days / total_observations
        - average_nonzero_demand: average sales volume on active selling days
        - intermittency_class: 'Regular' (>= 70%), 'Intermittent' (25% - 70%), 'Highly Intermittent' (< 25%)

    Args:
        df: Daily demand DataFrame.
        target_col: Target demand column.
        group_col: Entity grouping level (default: 'sku_id').

    Returns:
        pd.DataFrame: SKU-level intermittency table.
    """
    grouped = df.groupby(group_col)[target_col]

    total_days = grouped.count()
    non_zero = (df[df[target_col] > 0]).groupby(df[group_col])[target_col].count()
    non_zero = non_zero.reindex(total_days.index, fill_value=0)
    zero_days = total_days - non_zero

    nonzero_demand_sum = (df[df[target_col] > 0]).groupby(df[group_col])[target_col].sum()
    nonzero_demand_sum = nonzero_demand_sum.reindex(total_days.index, fill_value=0)

    occurrence_rate = np.where(total_days > 0, non_zero / total_days, 0.0)
    avg_nonzero = np.where(non_zero > 0, nonzero_demand_sum / non_zero, 0.0)

    metrics_df = pd.DataFrame({
        group_col: total_days.index,
        "total_days": total_days.values,
        "zero_demand_days": zero_days.values,
        "non_zero_demand_days": non_zero.values,
        "demand_occurrence_rate": np.round(occurrence_rate, 4),
        "average_nonzero_demand": np.round(avg_nonzero, 2),
    })

    # Intermittency classification
    conditions = [
        metrics_df["demand_occurrence_rate"] >= 0.70,
        metrics_df["demand_occurrence_rate"] >= 0.25,
    ]
    choices = ["Regular", "Intermittent"]
    metrics_df["intermittency_class"] = np.select(conditions, choices, default="Highly Intermittent")

    return metrics_df


def build_demand_features(
    daily_demand: pd.DataFrame,
    abc_config: Optional[ABCConfig] = None,
    xyz_config: Optional[XYZConfig] = None,
    stockout_config: Optional[StockoutConfig] = None,
    events: Optional[List[BusinessEvent]] = None,
    include_rolling: bool = True,
    include_lags: bool = True,
    include_calendar: bool = True,
    include_events: bool = True,
    include_intermittency: bool = True,
) -> pd.DataFrame:
    """Master pipeline constructing the unified Demand Feature Dataset.

    Combines:
        1. Daily Demand grid (units_sold, revenue, available_qty, reserved_qty, in_transit_qty)
        2. Stockout indicators & demand masking (is_stockout, is_low_stock, forecast_training_eligible)
        3. ABC / XYZ / ABC-XYZ segmentation
        4. Calendar temporal & cyclical features (sin/cos day, month, week, weekend flags)
        5. Retail business events & promotional campaign flags
        6. Leakage-free lag features (1, 7, 14, 28, 30 days)
        7. Leakage-free rolling features (mean & std across 7, 14, 30, 90 days)
        8. Trend momentum ratios (7 vs 30, 30 vs 90)
        9. SKU demand intermittency categorization

    Args:
        daily_demand: Daily regularized demand DataFrame from build_daily_demand.
        abc_config: Optional configuration for ABC classification.
        xyz_config: Optional configuration for XYZ classification.
        stockout_config: Optional configuration for stockout detection.
        events: Optional list of business/retail events.
        include_rolling: Whether to compute rolling statistics.
        include_lags: Whether to compute lag features.
        include_calendar: Whether to compute calendar features.
        include_events: Whether to compute retail event flags.
        include_intermittency: Whether to compute intermittency metrics.

    Returns:
        pd.DataFrame: Complete feature dataset ready for time-series forecasting.
    """
    # 1. Stockout detection and demand masking
    features_df = mask_stockout_demand(daily_demand, stockout_config)

    # 2. ABC / XYZ classification
    sku_classes, _, _ = calculate_abc_xyz_matrix(
        features_df,
        abc_config=abc_config,
        xyz_config=xyz_config,
    )
    abc_xyz_cols = [
        "sku_id",
        "abc_class",
        "xyz_class",
        "abc_xyz_class",
        "mean_daily_demand",
        "std_daily_demand",
        "coefficient_of_variation",
    ]
    features_df = pd.merge(features_df, sku_classes[abc_xyz_cols], on="sku_id", how="left")

    # 3. Intermittency classification
    if include_intermittency:
        intermittency_df = calculate_intermittency_metrics(features_df, target_col="units_sold")
        int_cols = ["sku_id", "demand_occurrence_rate", "average_nonzero_demand", "intermittency_class"]
        features_df = pd.merge(features_df, intermittency_df[int_cols], on="sku_id", how="left")

    # 4. Lag features
    if include_lags:
        features_df = build_lag_features(features_df, lags=[1, 7, 14, 28, 30])

    # 5. Rolling features
    if include_rolling:
        features_df = build_rolling_features(features_df, windows=[7, 14, 30, 90])
        # 6. Trend features (derived from rolling & lags)
        features_df = build_trend_features(features_df)

    # 7. Calendar features
    if include_calendar:
        features_df = build_calendar_features(features_df, date_col="date")

    # 8. Business events
    if include_events:
        features_df = build_event_features(features_df, events=events, date_col="date")

    # Clean ordering of final columns
    primary_keys = ["date", "sku_id", "warehouse_id"]
    priority_order = [
        # Keys
        "date", "sku_id", "warehouse_id",
        # Observed targets
        "units_sold", "revenue",
        # Inventory & Stockout
        "available_qty", "reserved_qty", "in_transit_qty", "damaged_qty",
        "is_stockout", "is_low_stock", "stockout_days", "is_demand_constrained", "forecast_training_eligible",
        # Segmentation & Intermittency
        "abc_class", "xyz_class", "abc_xyz_class",
        "mean_daily_demand", "std_daily_demand", "coefficient_of_variation",
        "demand_occurrence_rate", "average_nonzero_demand", "intermittency_class",
    ]

    existing_priority = [c for c in priority_order if c in features_df.columns]
    remaining_cols = [c for c in features_df.columns if c not in existing_priority]
    final_cols = existing_priority + remaining_cols

    return features_df[final_cols].sort_values(["date", "sku_id", "warehouse_id"]).reset_index(drop=True)
