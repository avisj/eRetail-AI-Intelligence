"""Inventory Risk and Health Assessment Engine.

Evaluates forward-looking stockout risk, days of supply (DOS), projected depletion curves,
runout dates, overstock, and dormant/dead stock with full diagnostic evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from commerce_ai.inventory.schemas import InventoryRiskRecord


@dataclass
class RiskThresholdConfig:
    """Configurable thresholds for inventory health and stockout risk classification."""

    critical_dos_threshold: float = 3.0  # Runout within this many days is CRITICAL
    understock_dos_threshold: float = 7.0  # Net position below ROP or DOS below this is UNDERSTOCK
    overstock_dos_threshold: float = 60.0  # DOS above this value is OVERSTOCK
    min_demand_threshold: float = 0.01  # Demand below this considered dormant / zero
    dead_stock_history_days: int = 90  # Zero sales over this period qualifies positive stock as DEAD_STOCK
    lead_time_buffer_factor: float = 1.0  # DOS < (lead_time * factor) triggers stockout risk


def simulate_depletion_curve(
    starting_on_hand: int,
    forecast_units: Union[List[float], np.ndarray],
    forecast_dates: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[str], Optional[float]]:
    """Simulate daily inventory depletion under forecasted demand.

    Args:
        starting_on_hand: Current available pickable stock.
        forecast_units: Daily forecasted demand sequence over horizon H.
        forecast_dates: Optional ISO date strings corresponding to forecast steps.

    Returns:
        Tuple:
            1. projected_trajectory (np.ndarray): Daily remaining on-hand stock (clamped at 0).
            2. runout_date (Optional[str]): First date inventory hits zero.
            3. days_to_runout (Optional[float]): Fractional days until stock exhaustion.
    """
    fc_arr = np.maximum(0.0, np.array(forecast_units, dtype=float))
    h = len(fc_arr)

    if h == 0 or starting_on_hand <= 0:
        traj = np.zeros(h)
        runout_dt = forecast_dates[0] if (forecast_dates and len(forecast_dates) > 0) else None
        return traj, runout_dt, 0.0

    cum_demand = np.cumsum(fc_arr)
    remaining_stock = np.maximum(0.0, starting_on_hand - cum_demand)

    # Detect runout
    depleted_indices = np.where(cum_demand >= starting_on_hand)[0]
    if len(depleted_indices) > 0:
        first_zero_idx = int(depleted_indices[0])
        prior_cum = cum_demand[first_zero_idx - 1] if first_zero_idx > 0 else 0.0
        day_demand = fc_arr[first_zero_idx]
        needed_on_day = starting_on_hand - prior_cum
        fraction = (needed_on_day / day_demand) if day_demand > 0 else 0.0
        days_to_runout = float(first_zero_idx + fraction)
        runout_date = forecast_dates[first_zero_idx] if (forecast_dates and len(forecast_dates) > first_zero_idx) else None
    else:
        # Stockout does not occur within the forecast horizon
        days_to_runout = None
        runout_date = None

    return remaining_stock, runout_date, days_to_runout


def compute_stockout_hazard_score(
    net_position: float,
    daily_demand_mean: float,
    daily_demand_std: float,
    lead_time_mean_days: float,
    lead_time_std_days: float = 0.0,
) -> float:
    """Calculate the statistical stockout hazard score (0.0 to 1.0).

    Uses the cumulative normal distribution to estimate:
        P(Lead Time Demand > Net Position)

    Where lead time demand is distributed as:
        Normal(Mean = D * L, Variance = L * sigma_D^2 + D^2 * sigma_L^2)
    """
    if daily_demand_mean <= 0.0:
        return 0.0
    if net_position <= 0.0:
        return 1.0

    lt_demand_mean = daily_demand_mean * lead_time_mean_days
    variance_term = (lead_time_mean_days * (daily_demand_std ** 2)) + (
        (daily_demand_mean ** 2) * (lead_time_std_days ** 2)
    )
    lt_demand_std = math.sqrt(max(1e-6, variance_term))

    # Standard normal CDF: P(Demand > NetPosition) = 1 - CDF(NetPosition)
    z = (net_position - lt_demand_mean) / lt_demand_std
    prob_stockout = float(1.0 - stats.norm.cdf(z))
    return float(np.clip(prob_stockout, 0.0, 1.0))


def assess_sku_inventory_risk(
    sku_id: str,
    warehouse_id: str,
    on_hand: int,
    net_position: int,
    daily_forecast: Union[List[float], np.ndarray],
    lead_time_mean_days: float,
    lead_time_std_days: float = 0.0,
    safety_stock: float = 0.0,
    reorder_point: float = 0.0,
    target_stock_level: Optional[float] = None,
    unit_cost: float = 0.0,
    selling_price: float = 0.0,
    forecast_dates: Optional[List[str]] = None,
    historical_sales_sum_recent: Optional[float] = None,
    daily_demand_std: Optional[float] = None,
    config: Optional[RiskThresholdConfig] = None,
) -> InventoryRiskRecord:
    """Evaluate comprehensive inventory risk for a SKU-warehouse pair without concealing evidence.

    Classifies into 5 discrete categories:
        1. CRITICAL_STOCKOUT: On-hand is depleted or will run out before replenishment arrives (DOS < lead_time).
        2. UNDERSTOCK: Net position is below Reorder Point (ROP); action required soon.
        3. HEALTHY: Net position covers safety stock and lead-time demand without excessive surplus.
        4. OVERSTOCK: Days of Supply exceeds configured threshold with active demand.
        5. DEAD_STOCK: Inventory on hand with zero or negligible demand over the observation period.

    Returns:
        InventoryRiskRecord containing the categorization, financial exposures, and full underlying metrics.
    """
    cfg = config or RiskThresholdConfig()
    fc_arr = np.maximum(0.0, np.array(daily_forecast, dtype=float))
    mean_fc = float(np.mean(fc_arr)) if len(fc_arr) > 0 else 0.0

    # Determine daily demand uncertainty (sigma_D) for stockout hazard:
    # 1. Use explicitly supplied daily_demand_std (e.g. from historical clean demand or model residuals)
    # 2. Backward-compatible fallback that does NOT treat a flat forecast as zero uncertainty
    if daily_demand_std is not None:
        effective_demand_std = max(0.0, float(daily_demand_std))
    else:
        forecast_spread = float(np.std(fc_arr, ddof=1)) if len(fc_arr) > 1 else 0.0
        if forecast_spread > 0.0:
            effective_demand_std = forecast_spread
        elif mean_fc > 0.0:
            # Safe non-zero fallback for flat forecast: assume Poisson dispersion sqrt(mean)
            effective_demand_std = math.sqrt(mean_fc)
        else:
            effective_demand_std = 0.0

    # 1. Days of Supply calculation
    if mean_fc > cfg.min_demand_threshold:
        dos = float(on_hand / mean_fc)
    else:
        dos = float("inf") if on_hand > 0 else 0.0

    # 2. Projected depletion trajectory
    _, runout_dt, days_to_runout = simulate_depletion_curve(
        starting_on_hand=on_hand,
        forecast_units=fc_arr,
        forecast_dates=forecast_dates,
    )

    # 3. Statistical stockout risk score
    stockout_hazard = compute_stockout_hazard_score(
        net_position=float(net_position),
        daily_demand_mean=mean_fc,
        daily_demand_std=effective_demand_std,
        lead_time_mean_days=lead_time_mean_days,
        lead_time_std_days=lead_time_std_days,
    )

    # 4. Classification logic
    is_dormant_demand = (mean_fc <= cfg.min_demand_threshold)
    if historical_sales_sum_recent is not None and historical_sales_sum_recent <= 0.0:
        is_dormant_demand = True

    effective_lt = max(1.0, lead_time_mean_days * cfg.lead_time_buffer_factor)

    if is_dormant_demand:
        category = "DEAD_STOCK" if on_hand > 0 else "HEALTHY"
        if on_hand <= 0:
            stockout_hazard = 0.0
    elif on_hand <= 0:
        category = "CRITICAL_STOCKOUT"
    elif (days_to_runout is not None and days_to_runout < effective_lt) or (dos < effective_lt):
        category = "CRITICAL_STOCKOUT"
    elif net_position < reorder_point:
        category = "UNDERSTOCK"
    elif dos > cfg.overstock_dos_threshold:
        category = "OVERSTOCK"
    else:
        category = "HEALTHY"

    # 5. Financial Exposure Calculations
    target_stock = target_stock_level if target_stock_level is not None else (reorder_point + (mean_fc * 14.0))

    # Excess capital tied up
    excess_units = max(0, int(net_position - target_stock)) if category == "OVERSTOCK" else (
        on_hand if category == "DEAD_STOCK" else 0
    )
    excess_capital = float(excess_units * unit_cost)

    # Stockout shortfall & lost revenue
    shortfall_units = max(0.0, reorder_point - net_position) if category in ["CRITICAL_STOCKOUT", "UNDERSTOCK"] else 0.0
    lost_revenue = float(shortfall_units * selling_price)

    # Underlying metrics bundle (diagnostic evidence)
    underlying = {
        "daily_demand_forecast_mean": round(mean_fc, 4),
        "daily_demand_std": round(effective_demand_std, 4),
        "daily_demand_forecast_std": round(effective_demand_std, 4),
        "daily_demand_forecast_spread": round(float(np.std(fc_arr, ddof=1)) if len(fc_arr) > 1 else 0.0, 4),
        "lead_time_mean_days": round(lead_time_mean_days, 2),
        "lead_time_std_days": round(lead_time_std_days, 2),
        "on_hand": on_hand,
        "net_position": net_position,
        "reorder_point": round(reorder_point, 2),
        "safety_stock": round(safety_stock, 2),
        "target_stock_level": round(target_stock, 2),
        "days_of_supply": round(dos, 2) if dos != float("inf") else 9999.0,
        "days_to_runout": round(days_to_runout, 2) if days_to_runout is not None else None,
        "stockout_probability": round(stockout_hazard, 4),
        "unit_cost": round(unit_cost, 2),
        "selling_price": round(selling_price, 2),
    }

    return InventoryRiskRecord(
        sku_id=str(sku_id),
        warehouse_id=str(warehouse_id),
        on_hand=int(on_hand),
        net_position=int(net_position),
        mean_daily_forecast=mean_fc,
        days_of_supply=dos,
        lead_time_days=lead_time_mean_days,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        risk_category=category,
        stockout_risk_score=stockout_hazard,
        runout_date=runout_dt,
        days_to_runout=days_to_runout,
        excess_units=excess_units,
        excess_capital=excess_capital,
        stockout_units_at_risk=shortfall_units,
        lost_revenue_risk=lost_revenue,
        underlying_metrics=underlying,
    )
