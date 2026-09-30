"""Data Contracts and Schema Definitions for Business Impact & Opportunity Quantification (Phase 6F).

Defines standardized Pydantic models for:
- Individual business impact records quantifying observed and estimated financial exposure
- Impact categories, impact types, confidence tiers, and calculation statuses
- Business impact portfolio summaries with double-counting protection and currency isolation
- Configuration parameters for deterministic quantification rules
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from commerce_ai.cross_domain.schemas import SignalSeverity


class ImpactCategory(str, Enum):
    """Broad domain classification of quantified business impact."""

    REVENUE = "REVENUE"
    MARGIN = "MARGIN"
    INVENTORY = "INVENTORY"
    STOCKOUT = "STOCKOUT"
    RETURNS = "RETURNS"
    REPLENISHMENT = "REPLENISHMENT"
    FORECAST = "FORECAST"
    OPERATIONAL_COST = "OPERATIONAL_COST"
    CROSS_DOMAIN = "CROSS_DOMAIN"


class ImpactType(str, Enum):
    """Specific measurable financial or operational exposure classification."""

    REVENUE_EXPOSURE = "REVENUE_EXPOSURE"
    MARGIN_EXPOSURE = "MARGIN_EXPOSURE"
    INVENTORY_CAPITAL_EXPOSURE = "INVENTORY_CAPITAL_EXPOSURE"
    STOCKOUT_REVENUE_EXPOSURE = "STOCKOUT_REVENUE_EXPOSURE"
    STOCKOUT_MARGIN_EXPOSURE = "STOCKOUT_MARGIN_EXPOSURE"
    RETURN_REVENUE_EXPOSURE = "RETURN_REVENUE_EXPOSURE"
    RETURN_MARGIN_EXPOSURE = "RETURN_MARGIN_EXPOSURE"
    RETURN_COST_EXPOSURE = "RETURN_COST_EXPOSURE"
    SLOW_MOVING_INVENTORY_EXPOSURE = "SLOW_MOVING_INVENTORY_EXPOSURE"
    HIGH_VALUE_INVENTORY_EXPOSURE = "HIGH_VALUE_INVENTORY_EXPOSURE"
    REPLENISHMENT_FINANCIAL_EXPOSURE = "REPLENISHMENT_FINANCIAL_EXPOSURE"
    FORECAST_COVERAGE_EXPOSURE = "FORECAST_COVERAGE_EXPOSURE"


class ImpactConfidence(str, Enum):
    """Methodological basis and provenance of the quantified impact figure."""

    DIRECT_OBSERVED = "DIRECT_OBSERVED"  # Directly computed from historical transactions or inventory snapshots
    ESTIMATED = "ESTIMATED"  # Derived using an explicit deterministic formula (e.g. ROQ x unit_cost)
    MODEL_BASED = "MODEL_BASED"  # Produced by an upstream predictive model (e.g. Phase 5C return model)
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Underlying inputs missing; exposure cannot be reliably quantified


class CalculationStatus(str, Enum):
    """Operational execution status of the impact quantification calculation."""

    CALCULATED = "CALCULATED"
    PARTIALLY_CALCULATED = "PARTIALLY_CALCULATED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class BusinessImpactRecord(BaseModel):
    """Auditable quantification record measuring financial/operational exposure for a signal."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Identifiers & Provenance
    impact_id: str = Field(..., description="Deterministic unique identifier for this impact record")
    signal_id: str = Field(..., description="Identifier of the triggering cross-domain signal")
    signal_type: str = Field(..., description="Classification code of the triggering signal")
    impact_category: ImpactCategory = Field(..., description="Broad category of business impact")
    impact_type: ImpactType = Field(..., description="Specific exposure classification")
    severity: SignalSeverity = Field(..., description="Severity level inherited from signal")

    # Dimensions
    sku_id: Optional[str] = Field(default=None, description="SKU identifier if applicable")
    warehouse_id: Optional[str] = Field(default=None, description="Warehouse identifier if applicable")
    channel_id: Optional[str] = Field(default=None, description="Channel identifier if applicable")
    category_id: Optional[str] = Field(default=None, description="Product category identifier if applicable")
    brand: Optional[str] = Field(default=None, description="Product brand name if applicable")

    # Quantified Physical Units
    affected_units: Optional[int] = Field(default=None, ge=0, description="Observed transaction units affected")
    affected_inventory_units: Optional[int] = Field(default=None, ge=0, description="Physical inventory units associated with exposure")
    affected_inventory_value: Optional[float] = Field(default=None, ge=0.0, description="Monetary valuation of affected physical stock ($)")

    # Associated Historical Commercial Outcomes
    associated_gross_revenue: Optional[float] = Field(default=None, description="Observed gross revenue of affected entity ($)")
    associated_net_revenue: Optional[float] = Field(default=None, description="Observed net realized revenue of affected entity ($)")
    associated_gross_margin: Optional[float] = Field(default=None, description="Observed gross margin of affected entity ($)")
    associated_gross_margin_pct: Optional[float] = Field(default=None, description="Observed gross margin percentage")
    known_contribution_margin: Optional[float] = Field(default=None, description="Observed known contribution margin ($)")
    known_contribution_margin_pct: Optional[float] = Field(default=None, description="Observed known contribution margin percentage")

    # Core Quantified Exposure Metrics
    exposure_value: Optional[float] = Field(
        default=None, description="Primary quantified monetary exposure metric ($); None if data is insufficient"
    )
    exposure_currency: str = Field(default="USD", description="Currency code of the exposure value")
    confidence_status: ImpactConfidence = Field(..., description="Confidence provenance of exposure value")
    calculation_method: str = Field(..., description="Explicit formula code used to compute exposure_value")
    calculation_inputs: Dict[str, Any] = Field(default_factory=dict, description="Key input parameters and values used")
    calculation_status: CalculationStatus = Field(default=CalculationStatus.CALCULATED, description="Completeness of calculation")

    # Opportunity Proxy & Non-Prescriptive Metrics
    opportunity_proxy_value: Optional[float] = Field(
        default=None, description="Neutral measurable monetary value associated with signal (NOT guaranteed savings)"
    )

    # Auditing, Coverage & Description
    data_quality_status: str = Field(default="VALID", description="Data quality classification of inputs")
    domain_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Completeness percentage of domain inputs")
    as_of_date: Optional[str] = Field(default=None, description="Historical anti-leakage cutoff date")
    description: str = Field(..., description="Factual, non-causal explanation of quantified exposure")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation with rounded numeric fields."""
        d = self.model_dump()
        d["impact_category"] = self.impact_category.value if isinstance(self.impact_category, ImpactCategory) else str(self.impact_category)
        d["impact_type"] = self.impact_type.value if isinstance(self.impact_type, ImpactType) else str(self.impact_type)
        d["confidence_status"] = self.confidence_status.value if isinstance(self.confidence_status, ImpactConfidence) else str(self.confidence_status)
        d["calculation_status"] = self.calculation_status.value if isinstance(self.calculation_status, CalculationStatus) else str(self.calculation_status)
        d["severity"] = self.severity.value if isinstance(self.severity, SignalSeverity) else str(self.severity)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4) if "pct" in k or "rate" in k else round(v, 2)
        return d


class BusinessImpactConfig(BaseModel):
    """Configuration parameters for business impact quantification."""

    model_config = ConfigDict(extra="allow")

    min_sample_size_units: int = Field(default=10, ge=0, description="Minimum units required for rate-based impact derivation")
    min_sample_size_orders: int = Field(default=5, ge=0, description="Minimum orders required for rate-based impact derivation")
    default_currency: str = Field(default="USD", description="Default ISO currency code")
    allow_multi_currency: bool = Field(default=False, description="Whether to allow multi-currency portfolio aggregation")
    deduplicate_by_entity: bool = Field(default=True, description="Whether to calculate deduplicated capital exposure across overlapping signals")
    as_of_date: Optional[Union[str, date, datetime]] = Field(default=None, description="Historical cutoff date")


class BusinessImpactPortfolioSummary(BaseModel):
    """Portfolio-level aggregation of quantified business impacts across signals."""

    model_config = ConfigDict(extra="allow")

    total_signal_count: int = Field(default=0, ge=0, description="Total input signals evaluated")
    impact_record_count: int = Field(default=0, ge=0, description="Total quantified impact records generated")
    currency_breakdown: Dict[str, float] = Field(default_factory=dict, description="Gross exposure summed by currency")
    impact_category_counts: Dict[str, int] = Field(default_factory=dict, description="Record counts by ImpactCategory")
    impact_type_counts: Dict[str, int] = Field(default_factory=dict, description="Record counts by ImpactType")
    severity_counts: Dict[str, int] = Field(default_factory=dict, description="Record counts by SignalSeverity")

    # Exposure Totals by Provenance
    direct_observed_exposure: float = Field(default=0.0, ge=0.0, description="Total exposure from directly observed metrics ($)")
    estimated_exposure: float = Field(default=0.0, ge=0.0, description="Total exposure from deterministic estimation formulas ($)")
    model_based_exposure: float = Field(default=0.0, ge=0.0, description="Total exposure from upstream predictive models ($)")
    insufficient_data_count: int = Field(default=0, ge=0, description="Count of impacts where exposure could not be quantified")

    # Double-Counting & Deduplication
    gross_signal_exposure: float = Field(default=0.0, ge=0.0, description="Naive arithmetic sum of all exposure values ($)")
    deduplicated_exposure: Optional[float] = Field(
        default=None, description="Deduplicated exposure eliminating double-counted economic or duplicate signal exposures ($)"
    )
    deduplicated_physical_capital_exposure: Optional[float] = Field(
        default=None, description="Deduplicated physical inventory capital exposure across overlapping inventory signals ($)"
    )
    deduplication_status: str = Field(
        default="DEDUPLICATED_BY_ECONOMIC_DIMENSION_AND_ENTITY", description="Methodology used to resolve overlapping signal exposures"
    )

    # Detailed Aggregate Distributions
    exposure_totals_by_category: Dict[str, float] = Field(
        default_factory=dict, description="Gross exposure summed by ImpactCategory ($)"
    )
    exposure_totals_by_type: Dict[str, float] = Field(
        default_factory=dict, description="Gross exposure summed by ImpactType ($)"
    )
    calculation_status_counts: Dict[str, int] = Field(
        default_factory=dict, description="Record counts by CalculationStatus"
    )

    currency: str = Field(default="USD", description="Reporting currency")
    as_of_date: Optional[str] = Field(default=None, description="Historical point-in-time reference date")


class BusinessImpactResult(BaseModel):
    """Consolidated result bundle containing all impact records and portfolio summary."""

    model_config = ConfigDict(extra="allow")

    impact_records: List[BusinessImpactRecord] = Field(default_factory=list, description="All quantified impact records")
    portfolio_summary: BusinessImpactPortfolioSummary = Field(..., description="High-level portfolio impact summary")
    as_of_date: Optional[str] = Field(default=None, description="Historical reference cutoff date")
