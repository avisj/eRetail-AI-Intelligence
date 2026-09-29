"""Forecast Evaluation and Error Metrics.

Implements standard industry metrics for demand forecasting evaluation:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error, with zero-actual protection)
- WAPE (Weighted Absolute Percentage Error, volume-weighted error)
- Bias (Forecast Bias Percentage, tracking systematic over/under forecasting)
"""

from __future__ import annotations

from typing import Dict, Union
import numpy as np
import pandas as pd


def _to_numpy_pair(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate array-like inputs to matching 1D float numpy arrays."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true has length {len(y_t)}, y_pred has length {len(y_p)}")
    if len(y_t) == 0:
        raise ValueError("Cannot compute evaluation metrics on empty arrays.")

    return y_t, y_p


def mae(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
) -> float:
    """Mean Absolute Error (MAE): (1/n) * sum(|y_true - y_pred|)."""
    y_t, y_p = _to_numpy_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def rmse(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
) -> float:
    """Root Mean Squared Error (RMSE): sqrt((1/n) * sum((y_true - y_pred)^2))."""
    y_t, y_p = _to_numpy_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def mape(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    epsilon: float = 1e-5,
    ignore_zero_actuals: bool = True,
) -> float:
    """Mean Absolute Percentage Error (MAPE).

    Safe handling of zero demand:
    If ignore_zero_actuals=True, only non-zero actual points are evaluated to prevent
    infinite distortions typical of intermittent retail time series.
    Otherwise, regularizes zero denominators using epsilon.

    Returns:
        float: Percentage error (e.g. 15.2 for 15.2%).
    """
    y_t, y_p = _to_numpy_pair(y_true, y_pred)

    if ignore_zero_actuals:
        mask = np.abs(y_t) > epsilon
        if not np.any(mask):
            return 0.0 if np.all(np.abs(y_p) <= epsilon) else 100.0
        y_t = y_t[mask]
        y_p = y_p[mask]
        denom = np.abs(y_t)
    else:
        denom = np.maximum(np.abs(y_t), epsilon)

    ape = np.abs(y_t - y_p) / denom
    return float(np.mean(ape) * 100.0)


def wape(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
) -> float:
    """Weighted Absolute Percentage Error (WAPE): sum(|y_true - y_pred|) / sum(|y_true|).

    Standard KPI for supply chain forecasting as it weighs errors by sales volume.
    Safely handles zero denominator: returns 0.0 if both sum(y_true) and sum(y_pred) are 0.

    Returns:
        float: Percentage error (e.g. 18.5 for 18.5%).
    """
    y_t, y_p = _to_numpy_pair(y_true, y_pred)

    total_actual = float(np.sum(np.abs(y_t)))
    total_abs_error = float(np.sum(np.abs(y_t - y_p)))

    if total_actual == 0.0:
        return 0.0 if total_abs_error == 0.0 else 100.0

    return float((total_abs_error / total_actual) * 100.0)


def bias(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
) -> float:
    """Forecast Bias Percentage: sum(y_pred - y_true) / sum(|y_true|).

    Interpretation:
        - Positive (> 0): Systemic over-forecasting (leads to overstock risk)
        - Negative (< 0): Systemic under-forecasting (leads to stockout risk)
        - Zero (0.0): Unbiased forecast

    Returns:
        float: Bias percentage (e.g. +4.5% or -2.1%).
    """
    y_t, y_p = _to_numpy_pair(y_true, y_pred)

    total_actual = float(np.sum(np.abs(y_t)))
    net_error = float(np.sum(y_p - y_t))

    if total_actual == 0.0:
        return 0.0 if net_error == 0.0 else (100.0 if net_error > 0 else -100.0)

    return float((net_error / total_actual) * 100.0)


def evaluate_forecast(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
) -> Dict[str, float]:
    """Calculate all standard forecast evaluation metrics.

    Returns dictionary containing:
        - mae: Mean Absolute Error (units)
        - rmse: Root Mean Squared Error (units)
        - mape: Mean Absolute Percentage Error (%)
        - wape: Weighted Absolute Percentage Error (%)
        - bias: Forecast Bias Percentage (%)
    """
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 2),
        "wape": round(wape(y_true, y_pred), 2),
        "bias": round(bias(y_true, y_pred), 2),
    }
