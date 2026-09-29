"""Forecasting Dataset Construction and Time-Series Partitioning.

Prepares time-series panels for forecasting models and implements strict
chronological train/validation/test splits and rolling-origin backtesting folds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from commerce_ai.forecasting.config import BacktestConfig, ForecastConfig


def build_forecast_dataset(
    df: pd.DataFrame,
    config: Optional[ForecastConfig] = None,
) -> pd.DataFrame:
    """Prepare and validate a demand feature panel for time-series forecasting.

    Ensures:
        - Chronological sorting per entity
        - Presence of target and temporal columns
        - Retention of stockout and eligibility flags

    Args:
        df: Daily demand or feature dataset.
        config: ForecastConfig instance.

    Returns:
        pd.DataFrame: Formatted forecasting dataset.
    """
    cfg = config or ForecastConfig()
    result = df.copy()

    # Validate essential columns
    required = [cfg.date_column, cfg.target_column] + cfg.entity_columns
    missing = [c for c in required if c not in result.columns]
    if missing:
        raise ValueError(f"Forecast dataset missing required columns: {missing}")

    result[cfg.date_column] = pd.to_datetime(result[cfg.date_column])

    # Sort strictly chronologically within each entity
    sort_order = cfg.entity_columns + [cfg.date_column]
    result = result.sort_values(sort_order).reset_index(drop=True)

    # Ensure forecast_training_eligible exists
    if "forecast_training_eligible" not in result.columns:
        result["forecast_training_eligible"] = True

    return result


def time_series_train_test_split(
    df: pd.DataFrame,
    test_days: int = 30,
    val_days: int = 0,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Perform a strict chronological time-series train / validation / test split.

    Split structure:
        - Test: [max_date - test_days + 1, max_date]
        - Val:  [max_date - test_days - val_days + 1, max_date - test_days] (if val_days > 0)
        - Train: everything prior to validation/test

    Returns:
        Tuple of (train_df, test_df, val_df or None)
    """
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    max_date = data[date_col].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)

    if val_days > 0:
        val_start = test_start - pd.Timedelta(days=val_days)
        train_df = data[data[date_col] < val_start].reset_index(drop=True)
        val_df = data[(data[date_col] >= val_start) & (data[date_col] < test_start)].reset_index(drop=True)
    else:
        train_df = data[data[date_col] < test_start].reset_index(drop=True)
        val_df = None

    test_df = data[data[date_col] >= test_start].reset_index(drop=True)

    if val_days > 0:
        return train_df, test_df, val_df
    return train_df, test_df


def generate_rolling_origin_folds(
    df: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
) -> List[Dict[str, Any]]:
    """Generate chronological rolling-origin backtesting folds.

    Each fold consists of:
        - fold_index: 1 to K
        - train_df: History up to fold cutoff date
        - test_df: Out-of-sample window of `horizon` days
        - cutoff_date: Last date included in training
        - test_start_date: First date of forecast window
        - test_end_date: Last date of forecast window

    Args:
        df: Prepared forecast dataset.
        config: BacktestConfig specifying horizon, folds, and step size.

    Returns:
        List of fold dictionaries.
    """
    cfg = config or BacktestConfig()
    date_col = cfg.date_column
    step = cfg.step_size or cfg.horizon

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    max_date = data[date_col].max()
    min_date = data[date_col].min()
    total_days = (max_date - min_date).days + 1

    required_days = cfg.min_history + (cfg.folds * step)
    if total_days < required_days:
        raise ValueError(
            f"Insufficient historical span for {cfg.folds} folds: dataset has {total_days} days, "
            f"requires at least {required_days} days (min_history={cfg.min_history} + {cfg.folds}*{step})."
        )

    folds = []
    # Calculate cutoff dates moving backwards from max_date
    for fold_idx in range(cfg.folds):
        # Fold 1 is the oldest evaluation window, Fold K is the most recent
        shift_from_end = (cfg.folds - 1 - fold_idx) * step
        test_end = max_date - pd.Timedelta(days=shift_from_end)
        test_start = test_end - pd.Timedelta(days=cfg.horizon - 1)
        cutoff = test_start - pd.Timedelta(days=1)

        train_data = data[data[date_col] <= cutoff].reset_index(drop=True)
        test_data = data[(data[date_col] >= test_start) & (data[date_col] <= test_end)].reset_index(drop=True)

        folds.append({
            "fold_index": fold_idx + 1,
            "cutoff_date": cutoff.strftime("%Y-%m-%d"),
            "test_start_date": test_start.strftime("%Y-%m-%d"),
            "test_end_date": test_end.strftime("%Y-%m-%d"),
            "train_df": train_data,
            "test_df": test_data,
        })

    return folds
