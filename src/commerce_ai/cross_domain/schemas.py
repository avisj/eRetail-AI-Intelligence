"""Data Contracts and Schema Definitions for Cross-Domain Business Intelligence (Phase 6E).

Defines standardized Pydantic models for:
- Cross-domain business records across multiple analytical grains (SKU × Warehouse, SKU, Channel, Warehouse)
- Descriptive, non-causal cross-domain signals with deterministic severity and IDs
- Signal categories and rule-based severity levels
- Domain coverage audits across financial, inventory, demand, forecasting, returns, and replenishment
- Cross-domain portfolio summaries and configuration
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SignalCategory(str, Enum):
    """Categorical classification of cross-domain and single-domain signals."""

    FINANCIAL = "FINANCIAL"
    INVENTORY = "INVENTORY"
    DEMAND = "DEMAND"
    FORECAST = "FORECAST"
    STOCKOUT = "STOCKOUT"
    RETURNS = "RETURNS"
    REPLENISHMENT = "REPLENISHMENT"
    WAREHOUSE = "WAREHOUSE"
    CROSS_DOMAIN = "CROSS_DOMAIN"


class SignalSeverity(str, Enum):
    """Deterministic, rule-based severity tiers (neutral analytical classification)."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CrossDomainDimension(str, Enum):
    """Supported analytical aggregation grains for cross-domain intelligence."""

    SKU_WAREHOUSE = "SKU_WAREHOUSE"
    SKU = "SKU"
    SKU_CHANNEL = "SKU_CHANNEL"
    SKU_WAREHOUSE_CHANNEL = "SKU_WAREHOUSE_CHANNEL"
    WAREHOUSE = "WAREHOUSE"
    CHANNEL = "CHANNEL"
    CATEGORY = "CATEGORY"
    BRAND = "BRAND"


class CrossDomainSignal(BaseModel):
    """Deterministic, factual cross-domain signal indicating co-occurring operational/financial conditions."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    signal_id: str = Field(..., description="Deterministic unique identifier for this signal occurrence")
    signal_type: str = Field(..., description="Classification code for the signal rule (e.g. HIGH_REVENUE_LOW_STOCK)")
    category: SignalCategory = Field(..., description="Primary domain or CROSS_DOMAIN category")
    severity: SignalSeverity = Field(..., description="Deterministic rule-based severity tier")
    dimension: str = Field(..., description="Analytical grain: SKU_WAREHOUSE, SKU, WAREHOUSE, etc.")
    sku_id: Optional[str] = Field(default=None, description="SKU identifier if applicable")
    warehouse_id: Optional[str] = Field(default=None, description="Warehouse identifier if applicable")
    channel_id: Optional[str] = Field(default=None, description="Sales channel identifier if applicable")
    observed_metrics: Dict[str, Any] = Field(default_factory=dict, description="Observed metric values triggering the signal")
    reason_codes: List[str] = Field(default_factory=list, description="Machine-readable rule conditions satisfied")
    description: str = Field(..., description="Descriptive, strictly non-causal explanation of co-occurring conditions")
    data_quality: str = Field(default="VALID", description="Data quality classification of underlying inputs")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time reference date")

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to serializable dictionary representation."""
        d = self.model_dump()
        d["category"] = self.category.value if isinstance(self.category, SignalCategory) else str(self.category)
        d["severity"] = self.severity.value if isinstance(self.severity, SignalSeverity) else str(self.severity)
        return d


class CrossDomainBusinessRecord(BaseModel):
    """Unified analytical record capturing cross-domain dimensions and observed metrics."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # IDENTITY
    sku_id: str = Field(..., description="Product SKU identifier")
    warehouse_id: Optional[str] = Field(default=None, description="Warehouse identifier")
    channel_id: Optional[str] = Field(default=None, description="Sales channel identifier")
    category_id: Optional[str] = Field(default=None, description="Product category identifier")
    brand: Optional[str] = Field(default=None, description="Product brand name")
    currency: str = Field(default="USD", description="ISO currency code for monetary metrics")

    # FINANCIAL
    gross_revenue: Optional[float] = Field(default=None, description="Observed gross revenue before discounts")
    net_revenue: Optional[float] = Field(default=None, description="Observed net realized revenue after discounts")
    product_cost: Optional[float] = Field(default=None, description="Product acquisition / COGS valuation")
    gross_margin: Optional[float] = Field(default=None, description="Gross profit: net_revenue - product_cost")
    gross_margin_pct: Optional[float] = Field(default=None, description="Gross margin percentage: gross_margin / net_revenue")
    known_variable_cost: Optional[float] = Field(default=None, description="Known operational variable fulfillment costs")
    known_contribution_margin: Optional[float] = Field(default=None, description="Known contribution margin: gross_margin - known_variable_cost")
    known_contribution_margin_pct: Optional[float] = Field(default=None, description="Known contribution margin percentage")
    final_contribution_margin: Optional[float] = Field(default=None, description="Final operational contribution margin where supported")
    final_contribution_margin_pct: Optional[float] = Field(default=None, description="Final contribution margin percentage")

    # SALES
    units_sold: Optional[int] = Field(default=None, ge=0, description="Total units sold in period")
    transaction_count: Optional[int] = Field(default=None, ge=0, description="Count of sales order lines")
    order_count: Optional[int] = Field(default=None, ge=0, description="Count of distinct customer orders")
    average_order_value: Optional[float] = Field(default=None, ge=0.0, description="Average order net revenue")

    # INVENTORY
    current_on_hand: Optional[int] = Field(default=None, description="Physical units on hand at snapshot")
    inventory_value: Optional[float] = Field(default=None, description="Valuation of current on-hand: on_hand * unit_cost")
    available_inventory: Optional[int] = Field(default=None, description="Available unreserved units for sale")
    reserved_inventory: Optional[int] = Field(default=None, description="Inventory units allocated to open customer orders")
    inventory_position: Optional[int] = Field(default=None, description="Net inventory position: on_hand + on_order - reserved")
    days_of_cover: Optional[float] = Field(default=None, description="Estimated days of inventory cover given daily demand")

    # DEMAND
    average_daily_demand: Optional[float] = Field(default=None, description="Average daily demand units")
    demand_std: Optional[float] = Field(default=None, description="Standard deviation of daily demand")
    demand_trend: Optional[str] = Field(default=None, description="Observed demand trend (INCREASING, DECREASING, STABLE)")
    intermittency_class: Optional[str] = Field(default=None, description="Demand pattern classification (SMOOTH, INTERMITTENT, LUMPY, ERRATIC)")
    abc_class: Optional[str] = Field(default=None, description="ABC revenue/volume Pareto tier (A, B, C)")
    xyz_class: Optional[str] = Field(default=None, description="XYZ demand predictability tier (X, Y, Z)")

    # STOCKOUT
    stockout_days: Optional[int] = Field(default=None, ge=0, description="Total stockout days in period")
    stockout_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Proportion of days with stockout")
    stockout_flag: Optional[bool] = Field(default=None, description="Binary indicator of any stockout occurrence")
    estimated_lost_demand: Optional[float] = Field(default=None, description="Estimated lost demand units where supported")

    # FORECAST
    forecast_mean: Optional[float] = Field(default=None, description="Projected demand forecast mean over horizon")
    forecast_horizon: Optional[int] = Field(default=None, description="Forecast horizon in days")
    forecast_model: Optional[str] = Field(default=None, description="Forecasting model used")
    forecast_available: bool = Field(default=False, description="Flag indicating forecast output is present")

    # RETURNS
    return_rate: Optional[float] = Field(default=None, ge=0.0, description="Unit return rate: returned_units / units_sold")
    return_count: Optional[int] = Field(default=None, ge=0, description="Count of return events")
    return_units: Optional[int] = Field(default=None, ge=0, description="Total returned units")
    return_anomaly_flag: Optional[bool] = Field(default=None, description="Observed return rate anomaly indicator")
    return_risk: Optional[str] = Field(default=None, description="Categorical return risk classification")

    # REPLENISHMENT
    reorder_point: Optional[float] = Field(default=None, description="Calculated replenishment reorder point")
    recommended_order_qty: Optional[int] = Field(default=None, description="Recommended purchase order quantity")
    replenishment_trigger: Optional[bool] = Field(default=None, description="Active trigger flag (net position <= ROP)")
    replenishment_risk: Optional[str] = Field(default=None, description="Replenishment risk classification")

    # WAREHOUSE
    warehouse_risk_indicators: List[str] = Field(default_factory=list, description="Warehouse operational risk flags")
    transfer_indicators: List[str] = Field(default_factory=list, description="Observed multi-warehouse rebalancing indicators")

    # QUALITY & COVERAGE
    data_quality_status: str = Field(default="VALID", description="Input data quality status")
    financial_available: bool = Field(default=False, description="Flag indicating financial data presence")
    inventory_available: bool = Field(default=False, description="Flag indicating inventory data presence")
    demand_available: bool = Field(default=False, description="Flag indicating demand data presence")
    returns_available: bool = Field(default=False, description="Flag indicating returns data presence")
    replenishment_available: bool = Field(default=False, description="Flag indicating replenishment data presence")
    missing_domain_count: int = Field(default=0, ge=0, le=6, description="Number of missing domains out of 6")
    domain_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Percentage of 6 domains available")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time reference date")

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary with rounded numeric fields."""
        d = self.model_dump()
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4) if "pct" in k or "rate" in k else round(v, 2)
        return d


class CrossDomainIntelligenceConfig(BaseModel):
    """Configurable thresholds and options for cross-domain business intelligence analysis."""

    model_config = ConfigDict(extra="allow")

    # Thresholds for Financial & Inventory Signals
    high_inventory_percentile: float = Field(default=0.80, ge=0.0, le=1.0, description="Percentile threshold for high inventory value")
    high_revenue_percentile: float = Field(default=0.80, ge=0.0, le=1.0, description="Percentile threshold for high net revenue")
    high_margin_percentile: float = Field(default=0.80, ge=0.0, le=1.0, description="Percentile threshold for high gross margin")
    low_margin_pct_threshold: float = Field(default=0.20, ge=0.0, le=1.0, description="Threshold below which gross margin % is low")
    high_return_rate_threshold: float = Field(default=0.10, ge=0.0, le=1.0, description="Threshold above which return rate is high")
    stockout_rate_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Threshold above which stockout rate is flagged")
    low_stock_threshold: int = Field(default=10, ge=0, description="Available units threshold for low stock")
    high_inventory_units_threshold: int = Field(default=100, ge=0, description="Inventory units threshold for high physical volume")
    low_velocity_threshold: float = Field(default=1.0, ge=0.0, description="Units/day threshold below which demand is low velocity")

    # Minimum Sample Sizes
    min_sample_size_units: int = Field(default=10, ge=0, description="Minimum units sold for reliable return rate evaluation")
    min_sample_size_orders: int = Field(default=5, ge=0, description="Minimum orders for reliable return rate evaluation")

    # Operational Options
    as_of_date: Optional[Union[str, date, datetime]] = Field(default=None, description="Historical anti-leakage cutoff date")
    default_currency: str = Field(default="USD", description="Expected default ISO currency code")
    allow_multi_currency: bool = Field(default=False, description="Whether to allow multi-currency aggregation without conversion")


class CrossDomainPortfolioSummary(BaseModel):
    """Portfolio-level summary metrics across records, signals, and domains."""

    model_config = ConfigDict(extra="allow")

    total_records: int = Field(default=0, ge=0, description="Total analytical records evaluated")
    total_signals: int = Field(default=0, ge=0, description="Total descriptive signals triggered")
    signals_by_category: Dict[str, int] = Field(default_factory=dict, description="Signal counts grouped by SignalCategory")
    signals_by_severity: Dict[str, int] = Field(default_factory=dict, description="Signal counts grouped by SignalSeverity")
    top_signal_types: Dict[str, int] = Field(default_factory=dict, description="Top signal types ordered by frequency")
    domain_coverage_summary: Dict[str, float] = Field(default_factory=dict, description="Average coverage percentage by domain")
    total_revenue: Optional[float] = Field(default=None, description="Total net revenue across evaluated portfolio")
    total_margin: Optional[float] = Field(default=None, description="Total gross margin across evaluated portfolio")
    total_inventory_value: Optional[float] = Field(default=None, description="Total inventory value across evaluated portfolio")
    currency: str = Field(default="USD", description="Portfolio reporting currency")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time reference date")


class CrossDomainIntelligenceResult(BaseModel):
    """Consolidated result bundle containing records, signals, and portfolio summary."""

    model_config = ConfigDict(extra="allow")

    records: List[CrossDomainBusinessRecord] = Field(default_factory=list, description="Cross-domain business records")
    signals: List[CrossDomainSignal] = Field(default_factory=list, description="Cross-domain descriptive signals")
    summary: CrossDomainPortfolioSummary = Field(..., description="Portfolio-level summary metrics")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time reference date")
