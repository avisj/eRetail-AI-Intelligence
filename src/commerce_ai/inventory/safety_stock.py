"""Safety Stock, Reorder Point, and Inventory Policy Engine.

Implements deterministic supply-chain calculations:
- Multi-uncertainty safety stock (demand variance + lead-time variance)
- Dynamic reorder points (ROP)
- Configurable service-level-to-Z-score mapping
- Configurable ABC/XYZ service-level policies
- Optional Economic Order Quantity (EOQ) helper (strictly non-inventive when costs are absent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, Optional, Union
import numpy as np
from scipy import stats

from commerce_ai.inventory.schemas import SafetyStockResult


# Default configurable 9-box service level matrix
DEFAULT_ABC_XYZ_SERVICE_LEVELS: Dict[str, float] = {
    "AX": 0.99,  # High revenue, predictable demand
    "AY": 0.95,  # High revenue, moderate variability
    "AZ": 0.92,  # High revenue, volatile demand
    "BX": 0.95,  # Medium revenue, predictable demand
    "BY": 0.92,  # Medium revenue, moderate variability
    "BZ": 0.90,  # Medium revenue, volatile demand
    "CX": 0.90,  # Low revenue, predictable demand
    "CY": 0.88,  # Low revenue, moderate variability
    "CZ": 0.85,  # Low revenue, volatile demand (minimize carrying cost)
}


@dataclass
class ServiceLevelPolicy:
    """Configurable policy mapping SKU segments or specific SKUs to Target Service Levels."""

    default_service_level: float = 0.95
    segment_service_levels: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ABC_XYZ_SERVICE_LEVELS)
    )
    sku_overrides: Dict[str, float] = field(default_factory=dict)

    def get_service_level(
        self,
        sku_id: Optional[str] = None,
        segment: Optional[str] = None,
    ) -> float:
        """Resolve service level following hierarchy: SKU override -> segment -> default."""
        if sku_id and sku_id in self.sku_overrides:
            return self.sku_overrides[sku_id]
        if segment and segment in self.segment_service_levels:
            return self.segment_service_levels[segment]
        return self.default_service_level


def service_level_to_z_score(
    service_level: float,
    custom_mapping: Optional[Union[Dict[float, float], Callable[[float], float]]] = None,
) -> float:
    """Convert a target cycle service level into a normal distribution Z-score.

    Args:
        service_level: Cycle service level probability (must be strictly in (0.0, 1.0)).
        custom_mapping: Optional custom lookup table or callable function.

    Returns:
        float: Standard normal inverse cumulative distribution value (Z-score).

    Raises:
        ValueError: If service_level is outside the valid range (0.0, 1.0).
    """
    if not (0.0 < service_level < 1.0):
        raise ValueError(
            f"Service level must be strictly between 0.0 and 1.0 (exclusive), got {service_level}"
        )

    if custom_mapping is not None:
        if callable(custom_mapping):
            return float(custom_mapping(service_level))
        if isinstance(custom_mapping, dict) and service_level in custom_mapping:
            return float(custom_mapping[service_level])

    # Standard Gaussian inverse CDF (probit function)
    return float(stats.norm.ppf(service_level))


def calculate_safety_stock(
    daily_demand_mean: float,
    daily_demand_std: float,
    lead_time_mean_days: float,
    lead_time_std_days: float = 0.0,
    service_level: float = 0.95,
    custom_z_mapping: Optional[Union[Dict[float, float], Callable[[float], float]]] = None,
) -> Tuple[float, float]:
    """Calculate deterministic safety stock considering dual demand and lead-time uncertainty.

    Mathematical Formula:
        SS = Z * sqrt( L * sigma_D^2  +  D^2 * sigma_L^2 )

    Where:
        Z = Inverse standard normal CDF evaluated at Target Service Level
        L = Average replenishment lead time in days
        sigma_D = Standard deviation of daily demand
        D = Average daily demand
        sigma_L = Standard deviation of replenishment lead time in days

    Validations & Edge Cases:
        - D == 0: SS = 0.0 (no demand requires no safety stock)
        - L == 0 and sigma_L == 0: SS = 0.0 (instantaneous supply requires no buffer)
        - sigma_D == 0 and sigma_L == 0: SS = 0.0 (perfect certainty requires no safety buffer)
        - Negative inputs: raises ValueError

    Returns:
        Tuple[float, float]: (safety_stock, z_score)
    """
    # Input validation
    if daily_demand_mean < 0.0:
        raise ValueError(f"Daily demand mean cannot be negative, got {daily_demand_mean}")
    if daily_demand_std < 0.0:
        raise ValueError(f"Daily demand std cannot be negative, got {daily_demand_std}")
    if lead_time_mean_days < 0.0:
        raise ValueError(f"Lead time mean cannot be negative, got {lead_time_mean_days}")
    if lead_time_std_days < 0.0:
        raise ValueError(f"Lead time std cannot be negative, got {lead_time_std_days}")

    z = service_level_to_z_score(service_level, custom_mapping=custom_z_mapping)

    # Edge cases
    if daily_demand_mean == 0.0:
        return 0.0, z
    if lead_time_mean_days == 0.0 and lead_time_std_days == 0.0:
        return 0.0, z
    if daily_demand_std == 0.0 and lead_time_std_days == 0.0:
        return 0.0, z

    # Dual uncertainty variance equation
    # Term 1: Demand variance over average lead time: L * sigma_D^2
    term_demand = lead_time_mean_days * (daily_demand_std ** 2)
    # Term 2: Lead time variance over average demand rate: D^2 * sigma_L^2
    term_lead_time = (daily_demand_mean ** 2) * (lead_time_std_days ** 2)

    combined_variance = term_demand + term_lead_time
    ss = z * math.sqrt(max(0.0, combined_variance))

    return max(0.0, ss), z


def calculate_reorder_point(
    daily_demand_mean: float,
    lead_time_mean_days: float,
    safety_stock: float,
) -> float:
    """Calculate dynamic Reorder Point (ROP).

    Formula:
        ROP = (D * L) + SS

    Args:
        daily_demand_mean: Average daily demand rate (D).
        lead_time_mean_days: Average replenishment lead time in days (L).
        safety_stock: Buffer units (SS).

    Returns:
        float: Reorder point in units.
    """
    if daily_demand_mean < 0.0:
        raise ValueError(f"Daily demand mean cannot be negative, got {daily_demand_mean}")
    if lead_time_mean_days < 0.0:
        raise ValueError(f"Lead time mean cannot be negative, got {lead_time_mean_days}")
    if safety_stock < 0.0:
        raise ValueError(f"Safety stock cannot be negative, got {safety_stock}")

    lead_time_demand = daily_demand_mean * lead_time_mean_days
    return lead_time_demand + safety_stock


def calculate_optional_eoq(
    annual_demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    holding_cost_per_unit_per_year: Optional[float] = None,
    unit_cost: Optional[float] = None,
    annual_holding_rate: Optional[float] = None,
) -> Optional[float]:
    """Calculate Economic Order Quantity (EOQ) strictly when cost parameters are provided.

    IMPORTANT ARCHITECTURAL RULE:
        If ordering cost and holding cost are unavailable, this function returns None.
        It NEVER invents or hallucinates cost figures.

    Formula:
        EOQ = sqrt( (2 * D_annual * S) / H )

    Required Inputs:
        - annual_demand > 0 (D_annual)
        - order_cost > 0 (S: fixed cost per purchase order)
        - EITHER holding_cost_per_unit_per_year > 0 (H)
          OR (unit_cost > 0 and annual_holding_rate > 0), where H = unit_cost * annual_holding_rate

    Returns:
        Optional[float]: Optimal order quantity in units, or None if inputs are incomplete.
    """
    if annual_demand is None or order_cost is None:
        return None
    if annual_demand <= 0.0 or order_cost <= 0.0:
        return None

    # Resolve holding cost H
    h_cost: Optional[float] = None
    if holding_cost_per_unit_per_year is not None and holding_cost_per_unit_per_year > 0.0:
        h_cost = holding_cost_per_unit_per_year
    elif (
        unit_cost is not None
        and annual_holding_rate is not None
        and unit_cost > 0.0
        and annual_holding_rate > 0.0
    ):
        h_cost = unit_cost * annual_holding_rate

    if h_cost is None or h_cost <= 0.0:
        return None

    eoq = math.sqrt((2.0 * annual_demand * order_cost) / h_cost)
    return max(1.0, eoq)


def calculate_safety_stock_result(
    sku_id: str,
    warehouse_id: str,
    daily_demand_mean: float,
    daily_demand_std: float,
    lead_time_mean_days: float,
    lead_time_std_days: float = 0.0,
    service_level: float = 0.95,
    custom_z_mapping: Optional[Union[Dict[float, float], Callable[[float], float]]] = None,
    order_cost: Optional[float] = None,
    holding_cost_per_unit_per_year: Optional[float] = None,
    unit_cost: Optional[float] = None,
    annual_holding_rate: Optional[float] = None,
    review_period_days: float = 7.0,
    demand_rate_source: str = "HISTORICAL",
) -> SafetyStockResult:
    """Build a comprehensive SafetyStockResult for a SKU-warehouse pair.

    Calculates:
        - Safety stock (dual uncertainty)
        - Reorder point (ROP)
        - Optional EOQ (if cost parameters provided)
        - Target stock level: ROP + EOQ (if EOQ available) else ROP + (D * review_period)

    Returns:
        SafetyStockResult
    """
    ss, z = calculate_safety_stock(
        daily_demand_mean=daily_demand_mean,
        daily_demand_std=daily_demand_std,
        lead_time_mean_days=lead_time_mean_days,
        lead_time_std_days=lead_time_std_days,
        service_level=service_level,
        custom_z_mapping=custom_z_mapping,
    )

    rop = calculate_reorder_point(
        daily_demand_mean=daily_demand_mean,
        lead_time_mean_days=lead_time_mean_days,
        safety_stock=ss,
    )

    annual_demand = daily_demand_mean * 365.0
    eoq = calculate_optional_eoq(
        annual_demand=annual_demand,
        order_cost=order_cost,
        holding_cost_per_unit_per_year=holding_cost_per_unit_per_year,
        unit_cost=unit_cost,
        annual_holding_rate=annual_holding_rate,
    )

    # Target stock level (order-up-to)
    if eoq is not None:
        target_stock = rop + eoq
    else:
        # Fall back to periodic review cycle stock (D * review_period)
        cycle_stock = daily_demand_mean * review_period_days
        target_stock = rop + cycle_stock

    return SafetyStockResult(
        sku_id=str(sku_id),
        warehouse_id=str(warehouse_id),
        service_level=service_level,
        z_score=z,
        lead_time_days=lead_time_mean_days,
        lead_time_std_days=lead_time_std_days,
        daily_demand_mean=daily_demand_mean,
        daily_demand_std=daily_demand_std,
        safety_stock=ss,
        reorder_point=rop,
        target_stock_level=target_stock,
        eoq=eoq,
        method="normal_dual_uncertainty",
        demand_rate_source=demand_rate_source,
    )
