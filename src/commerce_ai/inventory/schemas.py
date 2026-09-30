"""Data Contracts and Schema Definitions for Inventory Intelligence (Phase 4A).

Defines standardized Pydantic and dataclass models for:
- Inventory position records
- Lead-time metrics and supplier profiles
- Safety stock and Reorder Point (ROP) results
- Inventory risk assessment records
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class InventoryPositionRecord(BaseModel):
    """Standardized snapshot of inventory position for a SKU at a warehouse."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    sku_id: str = Field(..., description="Stock Keeping Unit ID")
    warehouse_id: str = Field(..., description="Fulfillment Warehouse ID")
    on_hand: int = Field(..., ge=0, description="Available pickable inventory on hand")
    reserved: int = Field(default=0, ge=0, description="Allocated or reserved units")
    on_order: int = Field(default=0, ge=0, description="Inbound replenishment units in transit or open POs")
    damaged: int = Field(default=0, ge=0, description="Damaged or quarantined units")
    as_of_date: Optional[Union[date, str]] = Field(default=None, description="Snapshot reference date")

    @property
    def net_position(self) -> int:
        """Effective inventory position: on_hand + on_order - reserved."""
        return self.on_hand + self.on_order - self.reserved

    @property
    def total_physical(self) -> int:
        """Physical inventory present in warehouse: on_hand + reserved + damaged."""
        return self.on_hand + self.reserved + self.damaged

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["net_position"] = self.net_position
        d["total_physical"] = self.total_physical
        if isinstance(d.get("as_of_date"), date):
            d["as_of_date"] = d["as_of_date"].isoformat()
        return d


@dataclass
class LeadTimeMetrics:
    """Empirical lead-time metrics for a supplier or SKU-supplier pair."""

    supplier_id: str
    sku_id: Optional[str] = None
    sample_size: int = 0
    mean_lead_time_days: float = 0.0
    std_lead_time_days: float = 0.0
    min_lead_time_days: Optional[float] = None
    max_lead_time_days: Optional[float] = None
    on_time_delivery_rate: Optional[float] = None
    is_fallback: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supplier_id": self.supplier_id,
            "sku_id": self.sku_id,
            "sample_size": self.sample_size,
            "mean_lead_time_days": round(self.mean_lead_time_days, 2),
            "std_lead_time_days": round(self.std_lead_time_days, 2),
            "min_lead_time_days": round(self.min_lead_time_days, 2) if self.min_lead_time_days is not None else None,
            "max_lead_time_days": round(self.max_lead_time_days, 2) if self.max_lead_time_days is not None else None,
            "on_time_delivery_rate": round(self.on_time_delivery_rate, 4) if self.on_time_delivery_rate is not None else None,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
        }


@dataclass
class SafetyStockResult:
    """Calculated safety stock, reorder point, and policy targets."""

    sku_id: str
    warehouse_id: str
    service_level: float
    z_score: float
    lead_time_days: float
    lead_time_std_days: float
    daily_demand_mean: float
    daily_demand_std: float
    safety_stock: float
    reorder_point: float
    target_stock_level: Optional[float] = None
    eoq: Optional[float] = None
    method: str = "normal_dual_uncertainty"
    demand_rate_source: str = "HISTORICAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku_id": self.sku_id,
            "warehouse_id": self.warehouse_id,
            "service_level": round(self.service_level, 4),
            "z_score": round(self.z_score, 4),
            "lead_time_days": round(self.lead_time_days, 2),
            "lead_time_std_days": round(self.lead_time_std_days, 2),
            "daily_demand_mean": round(self.daily_demand_mean, 4),
            "daily_demand_std": round(self.daily_demand_std, 4),
            "safety_stock": round(self.safety_stock, 2),
            "reorder_point": round(self.reorder_point, 2),
            "target_stock_level": round(self.target_stock_level, 2) if self.target_stock_level is not None else None,
            "eoq": round(self.eoq, 2) if self.eoq is not None else None,
            "method": self.method,
            "demand_rate_source": self.demand_rate_source,
        }


@dataclass
class InventoryRiskRecord:
    """Detailed risk assessment record for a SKU-warehouse entity with supporting evidence."""

    sku_id: str
    warehouse_id: str
    on_hand: int
    net_position: int
    mean_daily_forecast: float
    days_of_supply: float
    lead_time_days: float
    safety_stock: float
    reorder_point: float
    risk_category: str  # CRITICAL_STOCKOUT, UNDERSTOCK, HEALTHY, OVERSTOCK, DEAD_STOCK
    stockout_risk_score: float  # Normalized 0.0 to 1.0 hazard score
    runout_date: Optional[str] = None
    days_to_runout: Optional[float] = None
    excess_units: int = 0
    excess_capital: float = 0.0
    stockout_units_at_risk: float = 0.0
    lost_revenue_risk: float = 0.0
    underlying_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku_id": self.sku_id,
            "warehouse_id": self.warehouse_id,
            "on_hand": self.on_hand,
            "net_position": self.net_position,
            "mean_daily_forecast": round(self.mean_daily_forecast, 4),
            "days_of_supply": round(self.days_of_supply, 2) if self.days_of_supply != float("inf") else 9999.0,
            "lead_time_days": round(self.lead_time_days, 2),
            "safety_stock": round(self.safety_stock, 2),
            "reorder_point": round(self.reorder_point, 2),
            "risk_category": self.risk_category,
            "stockout_risk_score": round(self.stockout_risk_score, 4),
            "runout_date": self.runout_date,
            "days_to_runout": round(self.days_to_runout, 2) if self.days_to_runout is not None else None,
            "excess_units": self.excess_units,
            "excess_capital": round(self.excess_capital, 2),
            "stockout_units_at_risk": round(self.stockout_units_at_risk, 2),
            "lost_revenue_risk": round(self.lost_revenue_risk, 2),
            "underlying_metrics": self.underlying_metrics,
        }
