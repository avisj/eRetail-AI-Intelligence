"""Data Contracts and Schema Definitions for Financial Intelligence (Phase 6A).

Defines standardized Pydantic models and data classes for:
- Financial data statuses and reconciliation states
- Line-level revenue and gross margin records
- Dimension-level financial aggregation metrics (SKU, Category, Brand, Channel, Warehouse, Time)
- Portfolio-level financial summaries and rankings
- Financial data quality audits and leakage protections
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FinancialDataStatus(str, Enum):
    """Classification of financial input completeness and validity for a record."""

    SUFFICIENT = "SUFFICIENT"  # Both valid revenue and unit_cost available
    PARTIAL = "PARTIAL"        # Valid revenue available, but unit_cost missing
    INSUFFICIENT = "INSUFFICIENT"  # Core revenue/quantity fields missing
    INVALID = "INVALID"        # Negative quantity, invalid price, malformed currency, etc.


class RevenueReconciliationStatus(str, Enum):
    """Comparison result between source revenue and calculated net revenue."""

    MATCH = "MATCH"                        # |calc_net - source_rev| <= minor_tolerance
    MINOR_VARIANCE = "MINOR_VARIANCE"      # minor_tolerance < |diff| <= material_tolerance
    MATERIAL_VARIANCE = "MATERIAL_VARIANCE"# |diff| > material_tolerance
    UNAVAILABLE = "UNAVAILABLE"            # Source revenue unobserved for reconciliation


class MarginClassification(str, Enum):
    """Categorical classification of gross margin polarity."""

    POSITIVE_MARGIN = "POSITIVE_MARGIN"  # gross_margin > 0
    ZERO_MARGIN = "ZERO_MARGIN"          # gross_margin == 0
    NEGATIVE_MARGIN = "NEGATIVE_MARGIN"  # gross_margin < 0
    UNAVAILABLE = "UNAVAILABLE"          # Cost or revenue unavailable to compute margin


class TimeGrain(str, Enum):
    """Temporal aggregation granularity for financial analytics."""

    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class FinancialIntelligenceConfig(BaseModel):
    """Configuration parameters for Financial Intelligence calculations."""

    model_config = ConfigDict(extra="allow")

    minor_variance_tolerance: float = Field(
        default=0.02, ge=0.0, description="Tolerance ($) below which revenue variance is considered a MATCH"
    )
    material_variance_tolerance: float = Field(
        default=1.00, ge=0.0, description="Tolerance ($) above which revenue variance is a MATERIAL_VARIANCE"
    )
    as_of_date: Optional[Union[str, date]] = Field(
        default=None, description="Optional point-in-time historical cutoff date"
    )
    default_currency: str = Field(default="USD", description="Default expected ISO currency code")
    allow_multi_currency: bool = Field(
        default=False, description="Whether to allow multi-currency aggregation without explicit FX conversion"
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class RevenueMarginRecord(BaseModel):
    """Deterministic financial record for an individual sales transaction line."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    record_id: str = Field(..., description="Deterministic unique SHA-256 identifier")
    sale_id: str = Field(..., description="Primary sales transaction line ID")
    order_id: str = Field(..., description="Sales order ID")
    date: str = Field(..., description="Transaction date (YYYY-MM-DD)")
    sku_id: str = Field(..., description="Product SKU ID")
    warehouse_id: str = Field(..., description="Fulfillment facility ID")
    channel_id: str = Field(..., description="Sales channel ID")
    quantity: Optional[int] = Field(default=None, description="Observed sold quantity")
    unit_price: Optional[float] = Field(default=None, description="Observed unit selling price")
    discount: Optional[float] = Field(default=None, description="Observed promotional line discount")
    discount_rate: Optional[float] = Field(default=None, description="Discount as fraction of gross revenue")
    source_revenue: Optional[float] = Field(default=None, description="Directly observed source revenue")
    calculated_gross_revenue: Optional[float] = Field(
        default=None, description="Calculated gross revenue: quantity * unit_price"
    )
    calculated_net_revenue: Optional[float] = Field(
        default=None, description="Calculated net revenue: gross_revenue - discount"
    )
    revenue_variance: Optional[float] = Field(
        default=None, description="Variance between calculated net and source revenue"
    )
    revenue_reconciliation_status: RevenueReconciliationStatus = Field(
        default=RevenueReconciliationStatus.UNAVAILABLE, description="Audit reconciliation outcome"
    )
    unit_cost: Optional[float] = Field(
        default=None, description="Directly observed standard unit procurement cost"
    )
    estimated_cogs: Optional[float] = Field(
        default=None, description="Calculated estimated cost of goods: quantity * unit_cost"
    )
    gross_margin: Optional[float] = Field(
        default=None, description="Calculated gross margin: net_revenue - estimated_cogs"
    )
    gross_margin_percentage: Optional[float] = Field(
        default=None, description="Calculated gross margin percentage: gross_margin / net_revenue"
    )
    margin_classification: MarginClassification = Field(
        default=MarginClassification.UNAVAILABLE, description="Margin polarity classification"
    )
    financial_status: FinancialDataStatus = Field(..., description="Financial input completeness status")
    status_rationale: str = Field(..., description="Deterministic explanation of status and metrics")
    currency: str = Field(default="USD", description="Currency denomination")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time filter applied if any")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "unit_price",
            "discount",
            "discount_rate",
            "source_revenue",
            "calculated_gross_revenue",
            "calculated_net_revenue",
            "revenue_variance",
            "unit_cost",
            "estimated_cogs",
            "gross_margin",
            "gross_margin_percentage",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "rate" in k or "percentage" in k else 2)
        return d


class FinancialDimensionMetric(BaseModel):
    """Factual, aggregated financial metrics for any analytical dimension or segment."""

    model_config = ConfigDict(extra="allow")

    dimension: str = Field(..., description="Dimension: SKU, CATEGORY, BRAND, CHANNEL, WAREHOUSE, etc.")
    segment_key: str = Field(..., description="Unique dimension identifier (e.g. SKU_00001, Electronics, CH_AMZ)")
    record_count: int = Field(default=0, ge=0, description="Total order-line records in segment")
    order_count: int = Field(default=0, ge=0, description="Distinct sales orders in segment")
    total_units: int = Field(default=0, ge=0, description="Total units sold in segment")
    total_gross_revenue: Optional[float] = Field(default=None, description="Sum of gross revenue")
    total_discount: Optional[float] = Field(default=None, description="Sum of promotional discounts")
    total_net_revenue: Optional[float] = Field(default=None, description="Sum of net realized revenue")
    total_estimated_cogs: Optional[float] = Field(default=None, description="Sum of estimated COGS")
    total_gross_margin: Optional[float] = Field(default=None, description="Sum of gross margin")
    gross_margin_percentage: Optional[float] = Field(
        default=None, description="Aggregate gross margin %: total_gross_margin / total_net_revenue"
    )
    discount_rate: Optional[float] = Field(
        default=None, description="Aggregate discount %: total_discount / total_gross_revenue"
    )
    average_order_value: Optional[float] = Field(
        default=None, description="Average net revenue per distinct order: total_net_revenue / order_count"
    )
    revenue_contribution: Optional[float] = Field(
        default=None, description="Segment net revenue share of total portfolio revenue"
    )
    margin_contribution: Optional[float] = Field(
        default=None, description="Segment gross margin share of total portfolio gross margin"
    )
    negative_margin_order_line_count: int = Field(default=0, ge=0, description="Count of order lines with margin < 0")
    negative_margin_revenue: Optional[float] = Field(default=None, description="Revenue of negative-margin lines")
    negative_margin_units: int = Field(default=0, ge=0, description="Units sold on negative-margin lines")
    financial_completeness: float = Field(default=0.0, ge=0.0, le=100.0, description="% of records with SUFFICIENT data")
    records_with_cost: int = Field(default=0, ge=0, description="Number of order lines with unit_cost available")
    records_without_cost: int = Field(default=0, ge=0, description="Number of order lines without unit_cost")
    records_with_valid_revenue: int = Field(default=0, ge=0, description="Order lines with valid non-negative revenue")
    records_with_invalid_revenue: int = Field(default=0, ge=0, description="Order lines with missing/negative revenue")
    currency: str = Field(default="USD", description="Currency denomination")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "total_gross_revenue",
            "total_discount",
            "total_net_revenue",
            "total_estimated_cogs",
            "total_gross_margin",
            "gross_margin_percentage",
            "discount_rate",
            "average_order_value",
            "revenue_contribution",
            "margin_contribution",
            "negative_margin_revenue",
            "financial_completeness",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "percentage" in k or "rate" in k or "contribution" in k else 2)
        return d


class FinancialPortfolioSummary(BaseModel):
    """Executive portfolio-level summary of network revenue, COGS, and gross margin."""

    model_config = ConfigDict(extra="allow")

    total_order_lines: int = Field(default=0, ge=0, description="Total order-line records analyzed")
    total_orders: int = Field(default=0, ge=0, description="Total distinct orders analyzed")
    total_units: int = Field(default=0, ge=0, description="Total physical units sold")
    total_gross_revenue: Optional[float] = Field(default=None, description="Sum of gross revenue across records")
    total_discount: Optional[float] = Field(default=None, description="Sum of line discounts across records")
    total_net_revenue: Optional[float] = Field(default=None, description="Sum of net revenue across records")
    total_estimated_cogs: Optional[float] = Field(default=None, description="Sum of estimated COGS across records")
    total_gross_margin: Optional[float] = Field(default=None, description="Sum of gross margin: net_revenue - COGS")
    gross_margin_percentage: Optional[float] = Field(
        default=None, description="Portfolio gross margin %: total_gross_margin / total_net_revenue"
    )
    portfolio_discount_rate: Optional[float] = Field(
        default=None, description="Portfolio discount rate: total_discount / total_gross_revenue"
    )
    average_order_value: Optional[float] = Field(
        default=None, description="Portfolio average order value: total_net_revenue / total_orders"
    )
    negative_margin_revenue: Optional[float] = Field(
        default=None, description="Revenue generated on negative-margin order lines"
    )
    negative_margin_units: int = Field(default=0, ge=0, description="Units sold on negative-margin order lines")
    count_of_negative_margin_order_lines: int = Field(
        default=0, ge=0, description="Number of order lines resulting in commercial loss"
    )
    count_of_skus_with_negative_aggregate_margin: int = Field(
        default=0, ge=0, description="Number of distinct SKUs with net aggregate gross margin < 0"
    )
    records_with_cost: int = Field(default=0, ge=0, description="Lines with valid unit_cost observed")
    records_without_cost: int = Field(default=0, ge=0, description="Lines without unit_cost observed")
    records_with_valid_revenue: int = Field(default=0, ge=0, description="Lines with valid revenue observed")
    records_with_invalid_revenue: int = Field(default=0, ge=0, description="Lines with invalid or null revenue")
    financial_completeness_rate: float = Field(
        default=0.0, ge=0.0, le=100.0, description="% of records with complete financial data (cost & revenue)"
    )
    revenue_reconciliation_status: RevenueReconciliationStatus = Field(
        default=RevenueReconciliationStatus.UNAVAILABLE, description="Overall revenue reconciliation status"
    )
    reconciliation_match_count: int = Field(default=0, ge=0, description="Lines matching within tolerance")
    reconciliation_variance_count: int = Field(default=0, ge=0, description="Lines with minor variance")
    reconciliation_material_variance_count: int = Field(
        default=0, ge=0, description="Lines with material variance"
    )
    currency: str = Field(default="USD", description="Portfolio currency")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time filter applied if any")
    generated_at: str = Field(..., description="ISO 8601 generation timestamp")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "total_gross_revenue",
            "total_discount",
            "total_net_revenue",
            "total_estimated_cogs",
            "total_gross_margin",
            "gross_margin_percentage",
            "portfolio_discount_rate",
            "average_order_value",
            "negative_margin_revenue",
            "financial_completeness_rate",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "percentage" in k or "rate" in k else 2)
        return d


class RankingResult(BaseModel):
    """Deterministic ranking of items according to a financial metric."""

    model_config = ConfigDict(extra="allow")

    ranking_metric: str = Field(..., description="Neutral analytical metric: highest_revenue, highest_margin, etc.")
    dimension: str = Field(..., description="Ranked dimension: SKU, CATEGORY, CHANNEL, etc.")
    items: List[FinancialDimensionMetric] = Field(default_factory=list, description="Ranked dimension records")


class FinancialDataQualityReport(BaseModel):
    """Comprehensive data quality audit of transactional sales and product cost records."""

    model_config = ConfigDict(extra="allow")

    total_input_records: int = Field(default=0, ge=0, description="Total sales records evaluated")
    valid_records: int = Field(default=0, ge=0, description="Clean records satisfying validation rules")
    invalid_records: int = Field(default=0, ge=0, description="Records failing validation rules")
    missing_sku_count: int = Field(default=0, ge=0, description="Records with missing or empty sku_id")
    missing_quantity_count: int = Field(default=0, ge=0, description="Records with null quantity")
    zero_quantity_count: int = Field(default=0, ge=0, description="Records with quantity == 0")
    negative_quantity_count: int = Field(default=0, ge=0, description="Records with quantity < 0")
    missing_unit_price_count: int = Field(default=0, ge=0, description="Records with null unit_price")
    negative_unit_price_count: int = Field(default=0, ge=0, description="Records with unit_price < 0")
    missing_discount_count: int = Field(default=0, ge=0, description="Records with null discount")
    negative_discount_count: int = Field(default=0, ge=0, description="Records with discount < 0")
    missing_revenue_count: int = Field(default=0, ge=0, description="Records with null source revenue")
    negative_revenue_count: int = Field(default=0, ge=0, description="Records with revenue < 0")
    missing_unit_cost_count: int = Field(default=0, ge=0, description="Records where unit_cost is not in products")
    negative_unit_cost_count: int = Field(default=0, ge=0, description="Records where unit_cost < 0")
    missing_currency_count: int = Field(default=0, ge=0, description="Records with null or blank currency")
    duplicate_sale_id_count: int = Field(default=0, ge=0, description="Duplicate sale_id records detected")
    future_date_leakage_count: int = Field(default=0, ge=0, description="Records dated after as_of_date")
    mismatched_currencies: List[str] = Field(
        default_factory=list, description="Currencies discovered in cohort differing from default"
    )
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed field-level audit issues")
    is_clean: bool = Field(default=True, description="True if no data quality errors were discovered")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class FinancialIntelligenceResult(BaseModel):
    """Consolidated container for complete financial intelligence analysis outputs."""

    model_config = ConfigDict(extra="allow")

    portfolio_summary: FinancialPortfolioSummary = Field(..., description="High-level portfolio financial metrics")
    dimension_metrics: Dict[str, List[FinancialDimensionMetric]] = Field(
        default_factory=dict, description="Aggregated metrics by dimension (SKU, Channel, Warehouse, etc.)"
    )
    time_series: List[FinancialDimensionMetric] = Field(
        default_factory=list, description="Temporal aggregated metrics across requested time grain"
    )
    rankings: Dict[str, RankingResult] = Field(
        default_factory=dict, description="Neutral analytical rankings (highest_revenue, lowest_margin, etc.)"
    )
    negative_margin_records: List[RevenueMarginRecord] = Field(
        default_factory=list, description="Individual order-line records with gross margin < 0"
    )
    data_quality_report: FinancialDataQualityReport = Field(
        ..., description="Data quality validation audit and issue inventory"
    )
    records: Optional[List[RevenueMarginRecord]] = Field(
        default=None, description="Optional granular line-level evaluated records"
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# =====================================================================
# Phase 6B: Cost Breakdown & True Unit Economics Schemas
# =====================================================================


class CostComponent(str, Enum):
    """Supported variable and direct cost components for unit economics."""

    PRODUCT_COST = "PRODUCT_COST"
    SHIPPING_COST = "SHIPPING_COST"
    PAYMENT_PROCESSING_COST = "PAYMENT_PROCESSING_COST"
    PACKAGING_COST = "PACKAGING_COST"
    WAREHOUSE_HANDLING_COST = "WAREHOUSE_HANDLING_COST"
    RETURN_PROCESSING_COST = "RETURN_PROCESSING_COST"
    OTHER_VARIABLE_COST = "OTHER_VARIABLE_COST"


class CostComponentStatus(str, Enum):
    """Audit status of a cost component's availability and derivation method."""

    AVAILABLE_SOURCE = "AVAILABLE_SOURCE"          # Observed directly in transactional data
    AVAILABLE_ESTIMATED = "AVAILABLE_ESTIMATED"    # Standard catalog or procurement rate
    AVAILABLE_ASSUMED = "AVAILABLE_ASSUMED"        # Applied from configurable business assumption/default
    UNAVAILABLE = "UNAVAILABLE"                    # Cost component not observed or provided
    NOT_APPLICABLE = "NOT_APPLICABLE"              # Cost component not required/applicable for segment


class CostSourceType(str, Enum):
    """Source origin of cost data for audit traceability."""

    SOURCE_DATA = "SOURCE_DATA"                    # Transaction-level feed
    CATALOG_ESTIMATE = "CATALOG_ESTIMATE"          # Catalog master table (e.g. products.csv)
    CONFIGURED_ASSUMPTION = "CONFIGURED_ASSUMPTION"# Parametric rate / fee in CostModelConfig
    UNAVAILABLE = "UNAVAILABLE"                    # Unobserved
    NOT_APPLICABLE = "NOT_APPLICABLE"              # Excluded by business policy


class ContributionMarginStatus(str, Enum):
    """Calculability state of contribution margin."""

    CALCULABLE = "CALCULABLE"                      # All required cost components available
    INSUFFICIENT_COST_DATA = "INSUFFICIENT_COST_DATA" # Required cost components are missing
    NOT_CALCULABLE = "NOT_CALCULABLE"              # Revenue or critical inputs missing/invalid


class UnitEconomicsStatus(str, Enum):
    """Comprehensive analytical status of unit economics calculations."""

    FULLY_CALCULABLE = "FULLY_CALCULABLE"          # Valid revenue & all required cost components available
    PARTIALLY_CALCULABLE = "PARTIALLY_CALCULABLE"  # Valid revenue & some costs available (e.g. product cost only)
    INSUFFICIENT_COST_DATA = "INSUFFICIENT_COST_DATA" # Core product cost or revenue is missing
    INVALID_DATA = "INVALID_DATA"                  # Negative quantities, malformed prices, invalid currency


class CostComponentDetail(BaseModel):
    """Traceable metadata and monetary amount for an individual cost component."""

    model_config = ConfigDict(extra="allow")

    component: CostComponent = Field(..., description="Cost component type")
    amount: Optional[float] = Field(default=None, description="Total monetary amount allocated to transaction")
    unit_amount: Optional[float] = Field(default=None, description="Per-unit monetary amount")
    status: CostComponentStatus = Field(default=CostComponentStatus.UNAVAILABLE, description="Availability status")
    source_type: CostSourceType = Field(default=CostSourceType.UNAVAILABLE, description="Source traceability classification")
    source_reference: Optional[str] = Field(default=None, description="Field, table, or parameter reference")
    description: Optional[str] = Field(default=None, description="Explanatory context on cost derivation")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        if d.get("amount") is not None:
            d["amount"] = round(float(d["amount"]), 2)
        if d.get("unit_amount") is not None:
            d["unit_amount"] = round(float(d["unit_amount"]), 4)
        return d


class CostModelConfig(BaseModel):
    """Configuration parameters and assumptions for Cost Breakdown & Unit Economics."""

    model_config = ConfigDict(extra="allow")

    required_cost_components: List[CostComponent] = Field(
        default_factory=lambda: [
            CostComponent.PRODUCT_COST,
            CostComponent.SHIPPING_COST,
            CostComponent.PAYMENT_PROCESSING_COST,
            CostComponent.PACKAGING_COST,
            CostComponent.WAREHOUSE_HANDLING_COST,
        ],
        description="List of cost components strictly required before contribution margin is marked CALCULABLE",
    )
    # Configurable assumptions / defaults (used only when explicit values are missing and assumptions enabled)
    default_shipping_cost_per_order: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed flat outbound shipping fee per order"
    )
    default_shipping_cost_per_unit: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed variable outbound shipping fee per unit"
    )
    channel_shipping_costs: Dict[str, float] = Field(
        default_factory=dict, description="Channel-specific shipping cost per order or unit"
    )
    channel_payment_processing_rates: Dict[str, float] = Field(
        default_factory=dict, description="Channel-specific payment processing percentage (e.g. 0.029 for 2.9%)"
    )
    channel_payment_processing_fixed_fees: Dict[str, float] = Field(
        default_factory=dict, description="Channel-specific payment gateway fixed fee per transaction ($)"
    )
    default_payment_processing_rate: Optional[float] = Field(
        default=None, ge=0.0, description="Default payment processing fee rate across all channels"
    )
    default_payment_processing_fixed_fee: Optional[float] = Field(
        default=None, ge=0.0, description="Default payment processing fixed fee per transaction ($)"
    )
    default_packaging_cost_per_unit: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed packaging box/material cost per unit"
    )
    warehouse_handling_cost_per_unit: Dict[str, float] = Field(
        default_factory=dict, description="Warehouse-specific pick/pack labor cost per unit"
    )
    default_warehouse_handling_cost_per_unit: Optional[float] = Field(
        default=None, ge=0.0, description="Default warehouse pick/pack handling cost per unit"
    )
    default_return_processing_cost_per_unit: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed reverse logistics & restocking fee per unit returned"
    )
    default_other_variable_cost_per_unit: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed other miscellaneous variable cost per unit"
    )
    as_of_date: Optional[Union[str, date]] = Field(
        default=None, description="Optional point-in-time historical cutoff date"
    )
    default_currency: str = Field(default="USD", description="Default expected ISO currency code")
    allow_multi_currency: bool = Field(
        default=False, description="Whether to allow multi-currency aggregation without explicit FX conversion"
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class UnitEconomicsRecord(BaseModel):
    """Deterministic unit economics evaluation for an individual transaction line."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    record_id: str = Field(..., description="Deterministic unique SHA-256 identifier (FIN-UE-...)")
    sale_id: str = Field(..., description="Primary sales transaction line ID")
    order_id: str = Field(..., description="Sales order ID")
    date: str = Field(..., description="Transaction date (YYYY-MM-DD)")
    sku_id: str = Field(..., description="Product SKU ID")
    warehouse_id: str = Field(..., description="Fulfillment facility ID")
    channel_id: str = Field(..., description="Sales channel ID")
    currency: str = Field(default="USD", description="Currency denomination")
    quantity: Optional[int] = Field(default=None, description="Observed units sold")
    unit_price: Optional[float] = Field(default=None, description="Observed unit selling price")

    # Revenue metrics
    gross_revenue: Optional[float] = Field(default=None, description="Calculated gross revenue: quantity * unit_price")
    discount: Optional[float] = Field(default=None, description="Promotional discount applied")
    net_revenue: Optional[float] = Field(default=None, description="Realized net revenue: gross_revenue - discount")
    source_revenue: Optional[float] = Field(default=None, description="Observed source revenue feed")

    # Product economics
    unit_product_cost: Optional[float] = Field(default=None, description="Observed standard unit procurement cost")
    product_cost: Optional[float] = Field(default=None, description="Total estimated COGS: quantity * unit_product_cost")
    gross_margin: Optional[float] = Field(default=None, description="Gross margin: net_revenue - product_cost")
    gross_margin_pct: Optional[float] = Field(default=None, description="Gross margin %: gross_margin / net_revenue")

    # Variable cost breakdown
    shipping_cost: Optional[float] = Field(default=None, description="Outbound fulfillment freight cost")
    payment_processing_cost: Optional[float] = Field(default=None, description="Payment gateway transaction fee")
    packaging_cost: Optional[float] = Field(default=None, description="Packaging materials & dunnage cost")
    warehouse_handling_cost: Optional[float] = Field(default=None, description="Pick, pack & handling labor cost")
    return_processing_cost: Optional[float] = Field(default=None, description="Reverse logistics & inspection fee")
    other_variable_cost: Optional[float] = Field(default=None, description="Other miscellaneous variable costs")
    total_known_variable_cost: float = Field(
        default=0.0, ge=0.0, description="Sum of all currently available variable cost components"
    )

    # Contribution economics
    known_contribution_margin: Optional[float] = Field(
        default=None, description="Net revenue minus product cost and all known variable costs"
    )
    known_contribution_margin_pct: Optional[float] = Field(
        default=None, description="Known contribution margin as percentage of net revenue"
    )
    contribution_margin: Optional[float] = Field(
        default=None, description="Final contribution margin (populated only when all required costs are observed)"
    )
    contribution_margin_pct: Optional[float] = Field(
        default=None, description="Final contribution margin % (populated only when all required costs are observed)"
    )
    contribution_margin_status: ContributionMarginStatus = Field(
        default=ContributionMarginStatus.INSUFFICIENT_COST_DATA, description="Calculability state of contribution margin"
    )

    # Data quality / confidence
    cost_completeness_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="% of required cost components that are available"
    )
    required_cost_components: List[str] = Field(default_factory=list, description="Cost components configured as required")
    available_cost_components: List[str] = Field(default_factory=list, description="Cost components currently available")
    unavailable_cost_components: List[str] = Field(default_factory=list, description="Cost components currently missing")
    estimated_cost_components: List[str] = Field(default_factory=list, description="Cost components derived from catalog/standard rates")
    assumed_cost_components: List[str] = Field(default_factory=list, description="Cost components derived from configurable assumptions")
    economics_status: UnitEconomicsStatus = Field(..., description="Overall unit economics calculation status")
    status_rationale: str = Field(..., description="Deterministic explanation of status and cost coverage")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time filter applied if any")
    cost_details: Dict[str, CostComponentDetail] = Field(
        default_factory=dict, description="Detailed per-component metadata and traceability"
    )

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "unit_price", "gross_revenue", "discount", "net_revenue", "source_revenue",
            "unit_product_cost", "product_cost", "gross_margin", "gross_margin_pct",
            "shipping_cost", "payment_processing_cost", "packaging_cost",
            "warehouse_handling_cost", "return_processing_cost", "other_variable_cost",
            "total_known_variable_cost", "known_contribution_margin", "known_contribution_margin_pct",
            "contribution_margin", "contribution_margin_pct", "cost_completeness_pct",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "pct" in k else 2)
        return d


class UnitEconomicsDimensionMetric(BaseModel):
    """Aggregated unit economics metrics for an analytical dimension (SKU, Channel, Warehouse, etc.)."""

    model_config = ConfigDict(extra="allow")

    dimension: str = Field(..., description="Dimension: SKU, CATEGORY, BRAND, CHANNEL, WAREHOUSE, etc.")
    segment_key: str = Field(..., description="Dimension identifier (e.g. SKU_001, CH_AMZ, WH_EAST)")
    record_count: int = Field(default=0, ge=0, description="Total order-line records in segment")
    order_count: int = Field(default=0, ge=0, description="Distinct sales orders in segment")
    total_units: int = Field(default=0, ge=0, description="Total units sold in segment")
    total_gross_revenue: Optional[float] = Field(default=None, description="Sum of gross revenue")
    total_discount: Optional[float] = Field(default=None, description="Sum of promotional discounts")
    total_net_revenue: Optional[float] = Field(default=None, description="Sum of net realized revenue")
    total_product_cost: Optional[float] = Field(default=None, description="Sum of standard procurement product costs")
    total_gross_margin: Optional[float] = Field(default=None, description="Sum of gross margin: net_revenue - product_cost")
    gross_margin_pct: Optional[float] = Field(
        default=None, description="Aggregate gross margin %: total_gross_margin / total_net_revenue"
    )

    # Variable cost totals
    total_shipping_cost: Optional[float] = Field(default=None, description="Sum of available shipping costs")
    total_payment_processing_cost: Optional[float] = Field(default=None, description="Sum of available payment fees")
    total_packaging_cost: Optional[float] = Field(default=None, description="Sum of available packaging costs")
    total_warehouse_handling_cost: Optional[float] = Field(default=None, description="Sum of available handling costs")
    total_return_processing_cost: Optional[float] = Field(default=None, description="Sum of available return fees")
    total_other_variable_cost: Optional[float] = Field(default=None, description="Sum of other variable costs")
    total_known_variable_cost: float = Field(default=0.0, ge=0.0, description="Sum of all available variable costs")

    # Contribution economics totals
    total_known_contribution_margin: Optional[float] = Field(
        default=None, description="Sum of known contribution margin: net_revenue - product_cost - known_var_costs"
    )
    known_contribution_margin_pct: Optional[float] = Field(
        default=None, description="Aggregate known contribution margin %: total_known_cm / total_net_revenue"
    )
    total_contribution_margin: Optional[float] = Field(
        default=None, description="Sum of contribution margin (only when fully calculable across slice)"
    )
    contribution_margin_pct: Optional[float] = Field(
        default=None, description="Aggregate contribution margin %: total_cm / total_net_revenue"
    )
    contribution_margin_status: ContributionMarginStatus = Field(
        default=ContributionMarginStatus.INSUFFICIENT_COST_DATA, description="Calculability status for segment"
    )

    # Completeness and descriptive metrics
    cost_completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Average cost completeness %")
    average_order_value: Optional[float] = Field(default=None, description="Net revenue / distinct order count")
    average_unit_revenue: Optional[float] = Field(default=None, description="Net revenue / total physical units")
    average_unit_cost: Optional[float] = Field(default=None, description="Product cost / total physical units")
    revenue_contribution: Optional[float] = Field(default=None, description="Segment net revenue share of portfolio")
    margin_contribution: Optional[float] = Field(default=None, description="Segment gross margin share of portfolio")
    economics_status: UnitEconomicsStatus = Field(
        default=UnitEconomicsStatus.PARTIALLY_CALCULABLE, description="Overall economics status for segment"
    )
    currency: str = Field(default="USD", description="Currency denomination")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "total_gross_revenue", "total_discount", "total_net_revenue", "total_product_cost",
            "total_gross_margin", "gross_margin_pct", "total_shipping_cost",
            "total_payment_processing_cost", "total_packaging_cost", "total_warehouse_handling_cost",
            "total_return_processing_cost", "total_other_variable_cost", "total_known_variable_cost",
            "total_known_contribution_margin", "known_contribution_margin_pct",
            "total_contribution_margin", "contribution_margin_pct", "cost_completeness_pct",
            "average_order_value", "average_unit_revenue", "average_unit_cost",
            "revenue_contribution", "margin_contribution",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "pct" in k or "contribution" in k else 2)
        return d


class UnitEconomicsPortfolioSummary(BaseModel):
    """Executive portfolio summary of network unit economics and cost coverage."""

    model_config = ConfigDict(extra="allow")

    total_order_lines: int = Field(default=0, ge=0, description="Total order-line transactions analyzed")
    total_orders: int = Field(default=0, ge=0, description="Total distinct customer sales orders")
    total_units: int = Field(default=0, ge=0, description="Total physical units sold")
    total_gross_revenue: Optional[float] = Field(default=None, description="Total gross revenue: quantity * price")
    total_discount: Optional[float] = Field(default=None, description="Total discounts granted")
    total_net_revenue: Optional[float] = Field(default=None, description="Total net realized revenue")
    total_product_cost: Optional[float] = Field(default=None, description="Total standard procurement product costs")
    total_gross_margin: Optional[float] = Field(default=None, description="Total gross margin")
    gross_margin_pct: Optional[float] = Field(default=None, description="Portfolio gross margin %")

    # Variable costs
    total_shipping_cost: Optional[float] = Field(default=None, description="Total outbound shipping cost")
    total_payment_processing_cost: Optional[float] = Field(default=None, description="Total payment processing fees")
    total_packaging_cost: Optional[float] = Field(default=None, description="Total packaging materials cost")
    total_warehouse_handling_cost: Optional[float] = Field(default=None, description="Total warehouse pick/pack labor")
    total_return_processing_cost: Optional[float] = Field(default=None, description="Total return reverse fees")
    total_other_variable_cost: Optional[float] = Field(default=None, description="Total other variable expenses")
    total_known_variable_cost: float = Field(default=0.0, ge=0.0, description="Total known variable expenses")

    # Contribution economics
    total_known_contribution_margin: Optional[float] = Field(
        default=None, description="Portfolio net revenue minus product costs and known variable costs"
    )
    known_contribution_margin_pct: Optional[float] = Field(
        default=None, description="Portfolio known contribution margin %"
    )
    total_contribution_margin: Optional[float] = Field(
        default=None, description="Final contribution margin (None unless all required costs available)"
    )
    contribution_margin_pct: Optional[float] = Field(
        default=None, description="Final contribution margin % (None unless all required costs available)"
    )
    contribution_margin_status: ContributionMarginStatus = Field(
        default=ContributionMarginStatus.INSUFFICIENT_COST_DATA, description="Calculability state"
    )

    # Coverage and completeness
    cost_completeness_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Portfolio average cost component completeness %"
    )
    economics_status: UnitEconomicsStatus = Field(
        default=UnitEconomicsStatus.PARTIALLY_CALCULABLE, description="Overall portfolio economics status"
    )
    average_order_value: Optional[float] = Field(default=None, description="Average net revenue per order")
    average_unit_revenue: Optional[float] = Field(default=None, description="Average net revenue per unit")
    average_unit_cost: Optional[float] = Field(default=None, description="Average product cost per unit")
    records_with_product_cost: int = Field(default=0, ge=0, description="Count of order lines with product cost")
    records_without_product_cost: int = Field(default=0, ge=0, description="Count of order lines without product cost")
    currency: str = Field(default="USD", description="Currency denomination")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time filter applied if any")
    generated_at: str = Field(..., description="ISO 8601 generation timestamp")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "total_gross_revenue", "total_discount", "total_net_revenue", "total_product_cost",
            "total_gross_margin", "gross_margin_pct", "total_shipping_cost",
            "total_payment_processing_cost", "total_packaging_cost", "total_warehouse_handling_cost",
            "total_return_processing_cost", "total_other_variable_cost", "total_known_variable_cost",
            "total_known_contribution_margin", "known_contribution_margin_pct",
            "total_contribution_margin", "contribution_margin_pct", "cost_completeness_pct",
            "average_order_value", "average_unit_revenue", "average_unit_cost",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "pct" in k else 2)
        return d


class MarginErosionSegment(BaseModel):
    """Descriptive identification of a dimension segment exhibiting margin erosion."""

    model_config = ConfigDict(extra="allow")

    segment_dimension: str = Field(..., description="Dimension: SKU, CHANNEL, WAREHOUSE, etc.")
    segment_key: str = Field(..., description="Identifier (e.g. SKU_001, CH_AMZ)")
    revenue: float = Field(..., description="Realized net revenue ($)")
    revenue_share: float = Field(..., description="Segment share of portfolio net revenue (0.0 to 1.0)")
    gross_margin: float = Field(..., description="Realized gross margin ($)")
    margin_share: float = Field(..., description="Segment share of portfolio gross margin (0.0 to 1.0)")
    gross_margin_pct: Optional[float] = Field(default=None, description="Segment gross margin %")
    erosion_type: str = Field(..., description="Erosion category: NEGATIVE_MARGIN, LOW_MARGIN, HIGH_DISCOUNT, CONTRIBUTION_MISMATCH")
    description: str = Field(..., description="Descriptive factual rationale of the erosion pattern")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["revenue"] = round(float(d["revenue"]), 2)
        d["revenue_share"] = round(float(d["revenue_share"]), 4)
        d["gross_margin"] = round(float(d["gross_margin"]), 2)
        d["margin_share"] = round(float(d["margin_share"]), 4)
        if d.get("gross_margin_pct") is not None:
            d["gross_margin_pct"] = round(float(d["gross_margin_pct"]), 4)
        return d


class MarginErosionReport(BaseModel):
    """Consolidated descriptive report of margin erosion patterns across the commercial portfolio."""

    model_config = ConfigDict(extra="allow")

    negative_margin_skus: List[MarginErosionSegment] = Field(default_factory=list, description="SKUs generating gross margin < 0")
    low_margin_skus: List[MarginErosionSegment] = Field(default_factory=list, description="SKUs with gross margin % below threshold")
    high_discount_skus: List[MarginErosionSegment] = Field(default_factory=list, description="SKUs with high promotional discount dilution")
    low_margin_channels: List[MarginErosionSegment] = Field(default_factory=list, description="Channels with below-average gross margin %")
    low_margin_warehouses: List[MarginErosionSegment] = Field(default_factory=list, description="Warehouses with below-average gross margin %")
    margin_contribution_mismatches: List[MarginErosionSegment] = Field(
        default_factory=list, description="Segments where revenue share significantly outpaces gross margin share"
    )
    summary_notes: List[str] = Field(default_factory=list, description="Descriptive observations on portfolio economics")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class UnitEconomicsResult(BaseModel):
    """Consolidated container for complete Phase 6B unit economics outputs."""

    model_config = ConfigDict(extra="allow")

    portfolio_summary: UnitEconomicsPortfolioSummary = Field(..., description="Executive unit economics summary")
    dimension_metrics: Dict[str, List[UnitEconomicsDimensionMetric]] = Field(
        default_factory=dict, description="Aggregated metrics by dimension (SKU, Channel, Warehouse, etc.)"
    )
    time_series: List[UnitEconomicsDimensionMetric] = Field(
        default_factory=list, description="Temporal unit economics time series"
    )
    margin_erosion: MarginErosionReport = Field(..., description="Descriptive margin erosion analysis")
    data_quality_report: FinancialDataQualityReport = Field(..., description="Data quality audit report")
    cost_model_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary of cost components, availability, and assumptions applied"
    )
    records: Optional[List[UnitEconomicsRecord]] = Field(
        default=None, description="Optional granular line-level unit economics records"
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# =====================================================================
# Phase 6C: Profitability Attribution & Margin Drivers Schemas
# =====================================================================


class MarginDriverClassification(str, Enum):
    """Deterministic analytical classification of segment margin drivers."""

    HIGH_REVENUE_LOW_MARGIN_SHARE = "HIGH_REVENUE_LOW_MARGIN_SHARE"
    LOW_REVENUE_HIGH_MARGIN_SHARE = "LOW_REVENUE_HIGH_MARGIN_SHARE"
    HIGH_DISCOUNT = "HIGH_DISCOUNT"
    NEGATIVE_MARGIN = "NEGATIVE_MARGIN"
    LOW_MARGIN = "LOW_MARGIN"
    HIGH_MARGIN_CONTRIBUTOR = "HIGH_MARGIN_CONTRIBUTOR"
    HIGH_REVENUE_HIGH_MARGIN = "HIGH_REVENUE_HIGH_MARGIN"
    LOW_REVENUE_LOW_MARGIN = "LOW_REVENUE_LOW_MARGIN"


class AttributionReasonCode(str, Enum):
    """Factual, descriptive reason codes for segment margin behavior."""

    NEGATIVE_GROSS_MARGIN = "NEGATIVE_GROSS_MARGIN"
    LOW_GROSS_MARGIN_PERCENT = "LOW_GROSS_MARGIN_PERCENT"
    HIGH_DISCOUNT = "HIGH_DISCOUNT"
    LOW_MARGIN_CONTRIBUTION = "LOW_MARGIN_CONTRIBUTION"
    HIGH_REVENUE_CONTRIBUTION = "HIGH_REVENUE_CONTRIBUTION"
    CONTRIBUTION_GAP_DEFICIT = "CONTRIBUTION_GAP_DEFICIT"


class ProfitabilityAttributionConfig(BaseModel):
    """Configuration parameters for Profitability Attribution & Margin Drivers."""

    model_config = ConfigDict(extra="allow")

    low_margin_threshold: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Gross margin percentage below which a segment is flagged LOW_MARGIN"
    )
    high_discount_threshold: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Discount rate above which a segment is flagged HIGH_DISCOUNT"
    )
    contribution_gap_threshold: float = Field(
        default=0.02, ge=0.0, description="Contribution gap threshold (margin_share - rev_share) to flag disparity"
    )
    concentration_percentiles: List[float] = Field(
        default_factory=lambda: [0.01, 0.05, 0.10, 0.20],
        description="Top concentration percentiles to evaluate (e.g. 1%, 5%, 10%, 20%)",
    )
    discount_bucket_boundaries: List[Tuple[float, float, str]] = Field(
        default_factory=lambda: [
            (0.0, 0.0, "0%"),
            (0.0, 0.05, "0-5%"),
            (0.05, 0.10, "5-10%"),
            (0.10, 0.20, "10-20%"),
            (0.20, 0.30, "20-30%"),
            (0.30, 1.00, "30%+"),
        ],
        description="List of (min_rate, max_rate, label) tuples defining discount rate buckets",
    )
    as_of_date: Optional[Union[str, date]] = Field(
        default=None, description="Point-in-time historical cutoff date"
    )
    default_currency: str = Field(default="USD", description="Default expected ISO currency code")
    allow_multi_currency: bool = Field(
        default=False, description="Whether to allow multi-currency aggregation without explicit FX conversion"
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ProfitabilityAttributionRecord(BaseModel):
    """Factual profitability attribution metrics for any dimensional slice."""

    model_config = ConfigDict(extra="allow")

    dimension: str = Field(..., description="Attribution dimension (SKU, CATEGORY, CHANNEL, WAREHOUSE, etc.)")
    segment_key: str = Field(..., description="Segment identifier (e.g. SKU_001, CH_AMZ, WH_EAST)")
    record_count: int = Field(default=0, ge=0, description="Order-line transactions count")
    order_count: int = Field(default=0, ge=0, description="Distinct sales orders count")
    total_units: int = Field(default=0, ge=0, description="Physical units sold")
    gross_revenue: Optional[float] = Field(default=None, description="Gross revenue: quantity * unit_price")
    discount: Optional[float] = Field(default=None, description="Promotional discount")
    net_revenue: Optional[float] = Field(default=None, description="Net realized revenue: gross_revenue - discount")
    product_cost: Optional[float] = Field(default=None, description="Standard procurement product cost")
    gross_margin: Optional[float] = Field(default=None, description="Gross margin: net_revenue - product_cost")
    gross_margin_pct: Optional[float] = Field(default=None, description="Gross margin %: gross_margin / net_revenue")
    revenue_contribution_pct: Optional[float] = Field(
        default=None, description="Segment net revenue share of portfolio net revenue"
    )
    margin_contribution_pct: Optional[float] = Field(
        default=None, description="Segment gross margin share of portfolio gross margin"
    )
    contribution_gap: Optional[float] = Field(
        default=None, description="margin_contribution_pct - revenue_contribution_pct"
    )
    absolute_contribution_gap: Optional[float] = Field(
        default=None, description="abs(contribution_gap)"
    )
    discount_rate: Optional[float] = Field(
        default=None, description="Discount rate: discount / gross_revenue"
    )
    margin_per_unit: Optional[float] = Field(
        default=None, description="Gross margin / total_units"
    )
    revenue_per_unit: Optional[float] = Field(
        default=None, description="Net revenue / total_units"
    )
    product_cost_per_unit: Optional[float] = Field(
        default=None, description="Product cost / total_units"
    )
    known_variable_cost: Optional[float] = Field(
        default=None, description="Total known variable expenses from Phase 6B if observed"
    )
    known_contribution_margin: Optional[float] = Field(
        default=None, description="Known contribution margin: net_revenue - product_cost - known_var_costs"
    )
    known_contribution_margin_pct: Optional[float] = Field(
        default=None, description="Known contribution margin %: known_cm / net_revenue"
    )
    average_order_value: Optional[float] = Field(
        default=None, description="Net revenue / distinct order_count"
    )
    currency: str = Field(default="USD", description="Currency denomination")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "gross_revenue", "discount", "net_revenue", "product_cost", "gross_margin",
            "gross_margin_pct", "revenue_contribution_pct", "margin_contribution_pct",
            "contribution_gap", "absolute_contribution_gap", "discount_rate",
            "margin_per_unit", "revenue_per_unit", "product_cost_per_unit",
            "known_variable_cost", "known_contribution_margin", "known_contribution_margin_pct",
            "average_order_value",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "pct" in k or "gap" in k or "rate" in k else 2)
        return d


class SKUProfitabilityProfile(BaseModel):
    """Reusable comprehensive profitability profile for an individual SKU."""

    model_config = ConfigDict(extra="allow")

    sku_id: str = Field(..., description="Unique product SKU ID")
    product_name: Optional[str] = Field(default=None, description="Catalog product description")
    category_id: Optional[str] = Field(default=None, description="Category classification")
    brand: Optional[str] = Field(default=None, description="Brand name")
    total_units: int = Field(default=0, ge=0, description="Total physical units sold")
    order_count: int = Field(default=0, ge=0, description="Distinct customer orders count")
    gross_revenue: Optional[float] = Field(default=None, description="Gross revenue")
    discount: Optional[float] = Field(default=None, description="Promotional discount")
    net_revenue: Optional[float] = Field(default=None, description="Net realized revenue")
    product_cost: Optional[float] = Field(default=None, description="Standard procurement product cost")
    gross_margin: Optional[float] = Field(default=None, description="Gross margin")
    gross_margin_pct: Optional[float] = Field(default=None, description="Gross margin %")
    margin_per_unit: Optional[float] = Field(default=None, description="Gross margin / total_units")
    revenue_per_unit: Optional[float] = Field(default=None, description="Net revenue / total_units")
    product_cost_per_unit: Optional[float] = Field(default=None, description="Product cost / total_units")
    discount_rate: Optional[float] = Field(default=None, description="discount / gross_revenue")
    revenue_contribution_pct: Optional[float] = Field(default=None, description="SKU share of portfolio revenue")
    margin_contribution_pct: Optional[float] = Field(default=None, description="SKU share of portfolio margin")
    contribution_gap: Optional[float] = Field(default=None, description="margin_contribution_pct - revenue_contribution_pct")
    driver_classifications: List[MarginDriverClassification] = Field(
        default_factory=list, description="Assigned margin driver classifications"
    )
    reason_codes: List[AttributionReasonCode] = Field(
        default_factory=list, description="Descriptive attribution reason codes"
    )
    economics_status: Optional[str] = Field(default=None, description="Phase 6B unit economics status")
    known_variable_cost: Optional[float] = Field(default=None, description="Known variable costs if available")
    known_contribution_margin: Optional[float] = Field(default=None, description="Known contribution margin")
    cost_completeness_pct: Optional[float] = Field(default=None, description="Cost component completeness %")
    currency: str = Field(default="USD", description="Currency denomination")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "gross_revenue", "discount", "net_revenue", "product_cost", "gross_margin",
            "gross_margin_pct", "margin_per_unit", "revenue_per_unit", "product_cost_per_unit",
            "discount_rate", "revenue_contribution_pct", "margin_contribution_pct",
            "contribution_gap", "known_variable_cost", "known_contribution_margin", "cost_completeness_pct",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "pct" in k or "gap" in k or "rate" in k else 2)
        return d


class MarginContributionPoint(BaseModel):
    """Ranked data point on the cumulative margin contribution curve."""

    model_config = ConfigDict(extra="allow")

    rank: int = Field(..., ge=1, description="1-indexed segment contribution rank")
    segment_id: str = Field(..., description="Segment identifier (e.g. SKU ID)")
    segment_dimension: str = Field(default="SKU", description="Dimension type")
    segment_margin: float = Field(..., description="Segment gross margin ($)")
    segment_revenue: float = Field(..., description="Segment net revenue ($)")
    margin_contribution_pct: float = Field(..., description="Segment margin share of portfolio total")
    revenue_contribution_pct: float = Field(..., description="Segment revenue share of portfolio total")
    cumulative_margin_contribution_pct: float = Field(..., description="Running cumulative margin contribution")
    cumulative_revenue_contribution_pct: float = Field(..., description="Running cumulative revenue contribution")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["segment_margin"] = round(float(d["segment_margin"]), 2)
        d["segment_revenue"] = round(float(d["segment_revenue"]), 2)
        d["margin_contribution_pct"] = round(float(d["margin_contribution_pct"]), 4)
        d["revenue_contribution_pct"] = round(float(d["revenue_contribution_pct"]), 4)
        d["cumulative_margin_contribution_pct"] = round(float(d["cumulative_margin_contribution_pct"]), 4)
        d["cumulative_revenue_contribution_pct"] = round(float(d["cumulative_revenue_contribution_pct"]), 4)
        return d


class MarginConcentrationTier(BaseModel):
    """Concentration summary for a specific top percentile tier of contributors."""

    model_config = ConfigDict(extra="allow")

    percentile: float = Field(..., description="Percentile fraction (e.g. 0.01 for 1%, 0.05 for 5%)")
    tier_label: str = Field(..., description="Descriptive label (e.g. 'Top 1%', 'Top 5%')")
    segment_count: int = Field(..., ge=0, description="Total segments evaluated in cohort")
    top_n_count: int = Field(..., ge=0, description="Count of top segments in this tier")
    cumulative_margin_contribution_pct: float = Field(
        ..., description="Cumulative share of portfolio gross margin generated by tier"
    )
    cumulative_revenue_contribution_pct: float = Field(
        ..., description="Cumulative share of portfolio net revenue generated by tier"
    )
    cumulative_gross_margin: float = Field(..., description="Total gross margin generated by tier ($)")
    cumulative_net_revenue: float = Field(..., description="Total net revenue generated by tier ($)")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["cumulative_margin_contribution_pct"] = round(float(d["cumulative_margin_contribution_pct"]), 4)
        d["cumulative_revenue_contribution_pct"] = round(float(d["cumulative_revenue_contribution_pct"]), 4)
        d["cumulative_gross_margin"] = round(float(d["cumulative_gross_margin"]), 2)
        d["cumulative_net_revenue"] = round(float(d["cumulative_net_revenue"]), 2)
        return d


class MarginConcentrationResult(BaseModel):
    """Consolidated portfolio concentration analysis across defined percentiles."""

    model_config = ConfigDict(extra="allow")

    dimension: str = Field(default="SKU", description="Dimension evaluated (e.g. SKU)")
    total_segments: int = Field(..., ge=0, description="Total distinct segments evaluated")
    total_portfolio_margin: float = Field(..., description="Portfolio gross margin ($)")
    total_portfolio_revenue: float = Field(..., description="Portfolio net revenue ($)")
    tiers: List[MarginConcentrationTier] = Field(default_factory=list, description="Concentration tiers")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class DiscountBucketMetric(BaseModel):
    """Aggregated financial performance metrics within a specific discount rate bucket."""

    model_config = ConfigDict(extra="allow")

    bucket_label: str = Field(..., description="Bucket label (e.g. '0%', '0-5%', '5-10%', '10-20%', '20-30%', '30%+')")
    min_discount_rate: float = Field(..., ge=0.0, description="Lower bound of discount rate (inclusive for 0%, exclusive otherwise)")
    max_discount_rate: float = Field(..., ge=0.0, description="Upper bound of discount rate (inclusive)")
    transaction_count: int = Field(default=0, ge=0, description="Order-line count in bucket")
    total_units: int = Field(default=0, ge=0, description="Physical units sold in bucket")
    total_gross_revenue: float = Field(default=0.0, description="Gross revenue in bucket ($)")
    total_discount: float = Field(default=0.0, description="Total discount granted in bucket ($)")
    total_net_revenue: float = Field(default=0.0, description="Net realized revenue in bucket ($)")
    total_product_cost: float = Field(default=0.0, description="Standard procurement product cost in bucket ($)")
    total_gross_margin: float = Field(default=0.0, description="Gross margin in bucket ($)")
    gross_margin_pct: Optional[float] = Field(default=None, description="Gross margin % in bucket")
    margin_per_unit: Optional[float] = Field(default=None, description="Gross margin per unit in bucket")
    discount_margin_impact: float = Field(default=0.0, description="Margin reduction directly attributable to discount (-discount)")
    currency: str = Field(default="USD", description="Currency denomination")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "total_gross_revenue", "total_discount", "total_net_revenue", "total_product_cost",
            "total_gross_margin", "gross_margin_pct", "margin_per_unit", "discount_margin_impact",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "pct" in k else 2)
        return d


class DiscountImpactSummary(BaseModel):
    """Executive summary of promotional discount impact on portfolio margins."""

    model_config = ConfigDict(extra="allow")

    total_gross_revenue: float = Field(default=0.0, description="Total portfolio gross revenue ($)")
    total_discount: float = Field(default=0.0, description="Total portfolio promotional discounts ($)")
    discount_rate: float = Field(default=0.0, description="Portfolio aggregate discount rate: discount / gross_revenue")
    total_net_revenue: float = Field(default=0.0, description="Total portfolio net revenue ($)")
    margin_before_discount: float = Field(default=0.0, description="gross_revenue - product_cost ($)")
    margin_after_discount: float = Field(default=0.0, description="net_revenue - product_cost ($)")
    discount_margin_impact: float = Field(
        default=0.0, description="Direct margin reduction from discount: margin_after - margin_before = -discount ($)"
    )
    buckets: List[DiscountBucketMetric] = Field(default_factory=list, description="Performance broken down by discount rate bucket")
    currency: str = Field(default="USD", description="Currency denomination")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "total_gross_revenue", "total_discount", "discount_rate", "total_net_revenue",
            "margin_before_discount", "margin_after_discount", "discount_margin_impact",
        ]:
            d[k] = round(float(d[k]), 4 if "rate" in k else 2)
        return d


class MarginDriverSegment(BaseModel):
    """Segment classified under a deterministic margin driver category."""

    model_config = ConfigDict(extra="allow")

    segment_dimension: str = Field(..., description="Dimension: SKU, CHANNEL, WAREHOUSE, etc.")
    segment_key: str = Field(..., description="Segment identifier (e.g. SKU_001, CH_AMZ)")
    driver_classification: MarginDriverClassification = Field(..., description="Assigned driver category")
    gross_margin: float = Field(..., description="Realized gross margin ($)")
    net_revenue: float = Field(..., description="Realized net revenue ($)")
    gross_margin_pct: Optional[float] = Field(default=None, description="Gross margin %")
    revenue_contribution_pct: Optional[float] = Field(default=None, description="Revenue share of portfolio")
    margin_contribution_pct: Optional[float] = Field(default=None, description="Margin share of portfolio")
    contribution_gap: Optional[float] = Field(default=None, description="margin_contribution_pct - revenue_contribution_pct")
    reason_codes: List[AttributionReasonCode] = Field(default_factory=list, description="Triggered descriptive reason codes")
    description: str = Field(..., description="Descriptive factual explanation of classification")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["gross_margin"] = round(float(d["gross_margin"]), 2)
        d["net_revenue"] = round(float(d["net_revenue"]), 2)
        if d.get("gross_margin_pct") is not None:
            d["gross_margin_pct"] = round(float(d["gross_margin_pct"]), 4)
        if d.get("revenue_contribution_pct") is not None:
            d["revenue_contribution_pct"] = round(float(d["revenue_contribution_pct"]), 4)
        if d.get("margin_contribution_pct") is not None:
            d["margin_contribution_pct"] = round(float(d["margin_contribution_pct"]), 4)
        if d.get("contribution_gap") is not None:
            d["contribution_gap"] = round(float(d["contribution_gap"]), 4)
        return d


class MarginWaterfallStage(BaseModel):
    """An individual numerical stage in the portfolio commercial margin waterfall."""

    model_config = ConfigDict(extra="allow")

    stage_name: str = Field(..., description="Stage title (e.g. Gross Revenue, Discounts, Net Revenue, COGS, etc.)")
    stage_order: int = Field(..., ge=1, description="Sequential sequence index")
    amount: Optional[float] = Field(default=None, description="Monetary value of stage ($)")
    percentage_of_gross_revenue: Optional[float] = Field(
        default=None, description="Stage amount as percentage of gross revenue"
    )
    stage_type: str = Field(..., description="Stage classification: SUBTOTAL, DEDUCTION, RESULT")
    description: str = Field(..., description="Descriptive definition of stage calculation")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        if d.get("amount") is not None:
            d["amount"] = round(float(d["amount"]), 2)
        if d.get("percentage_of_gross_revenue") is not None:
            d["percentage_of_gross_revenue"] = round(float(d["percentage_of_gross_revenue"]), 4)
        return d


class MarginWaterfall(BaseModel):
    """Executive portfolio margin waterfall representing economic progression from gross revenue to margin."""

    model_config = ConfigDict(extra="allow")

    stages: List[MarginWaterfallStage] = Field(default_factory=list, description="Ordered waterfall stages")
    gross_revenue: float = Field(default=0.0, description="Gross revenue ($)")
    discount: float = Field(default=0.0, description="Promotional discount deduction ($)")
    net_revenue: float = Field(default=0.0, description="Net realized revenue ($)")
    product_cost: float = Field(default=0.0, description="Standard procurement product cost deduction ($)")
    gross_margin: float = Field(default=0.0, description="Gross commercial margin ($)")
    known_variable_costs: float = Field(default=0.0, description="Total known variable expenses deduction ($)")
    known_contribution_margin: float = Field(default=0.0, description="Known contribution margin ($)")
    final_contribution_margin: Optional[float] = Field(
        default=None, description="Final contribution margin (None when cost components are unavailable)"
    )
    currency: str = Field(default="USD", description="Currency denomination")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "gross_revenue", "discount", "net_revenue", "product_cost", "gross_margin",
            "known_variable_costs", "known_contribution_margin",
        ]:
            d[k] = round(float(d[k]), 2)
        if d.get("final_contribution_margin") is not None:
            d["final_contribution_margin"] = round(float(d["final_contribution_margin"]), 2)
        return d


class ProfitabilityAttributionResult(BaseModel):
    """Consolidated container for complete Phase 6C Profitability Attribution outputs."""

    model_config = ConfigDict(extra="allow")

    portfolio_attribution: ProfitabilityAttributionRecord = Field(
        ..., description="Overall network profitability attribution metrics"
    )
    dimension_attributions: Dict[str, List[ProfitabilityAttributionRecord]] = Field(
        default_factory=dict, description="Attribution records partitioned by dimension"
    )
    sku_profiles: List[SKUProfitabilityProfile] = Field(
        default_factory=list, description="Comprehensive SKU-level profitability profiles"
    )
    margin_concentration: MarginConcentrationResult = Field(
        ..., description="Portfolio margin and revenue concentration analysis"
    )
    contribution_curve: List[MarginContributionPoint] = Field(
        default_factory=list, description="Ordered cumulative margin contribution curve data"
    )
    discount_impact: DiscountImpactSummary = Field(
        ..., description="Comprehensive promotional discount impact analysis and bucket breakdown"
    )
    margin_drivers: List[MarginDriverSegment] = Field(
        default_factory=list, description="Segments classified under deterministic margin driver categories"
    )
    margin_waterfall: MarginWaterfall = Field(
        ..., description="Portfolio margin waterfall progression"
    )
    time_attributions: List[ProfitabilityAttributionRecord] = Field(
        default_factory=list, description="Temporal profitability attribution series"
    )
    data_quality_report: FinancialDataQualityReport = Field(
        ..., description="Data quality validation audit report"
    )
    currency: str = Field(default="USD", description="Currency denomination")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time filter applied if any")
    generated_at: str = Field(..., description="ISO 8601 generation timestamp")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# =============================================================================
# Phase 6D: Cost Completeness & Operational Economics Schemas
# =============================================================================


class CostGrain(str, Enum):
    """Grain / level of granularity at which an operational cost is naturally incurred or captured."""

    UNIT = "UNIT"
    TRANSACTION = "TRANSACTION"
    ORDER = "ORDER"
    RETURN_EVENT = "RETURN_EVENT"
    UNKNOWN = "UNKNOWN"


class CostCompletenessStatus(str, Enum):
    """Categorical classification of cost completeness based on required components."""

    COMPLETE = "COMPLETE"
    PARTIALLY_COMPLETE = "PARTIALLY_COMPLETE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class OperationalEconomicsStatus(str, Enum):
    """Calculation health and calculability state for operational economics."""

    COMPLETE = "COMPLETE"
    PARTIALLY_CALCULABLE = "PARTIALLY_CALCULABLE"
    INSUFFICIENT_COST_DATA = "INSUFFICIENT_COST_DATA"
    INVALID_INPUT = "INVALID_INPUT"


class OperationalCostDetail(BaseModel):
    """Detailed metadata and audit trail for a single operational cost component."""

    model_config = ConfigDict(extra="allow")

    component: CostComponent = Field(..., description="Cost component type")
    amount: Optional[float] = Field(default=None, description="Monetary cost amount in currency units")
    unit_amount: Optional[float] = Field(default=None, description="Per-unit allocated cost amount")
    currency: str = Field(default="USD", description="Currency denomination")
    source_type: CostSourceType = Field(..., description="Cost source provenance")
    availability_status: CostComponentStatus = Field(..., description="Component availability status")
    cost_grain: CostGrain = Field(default=CostGrain.UNKNOWN, description="Natural grain of cost incurrence")
    is_estimated: bool = Field(default=False, description="True if derived from catalog or standard reference tables")
    is_assumed: bool = Field(default=False, description="True if derived from configurable operational assumptions")
    included_in_known_contribution: bool = Field(
        default=False, description="True if included in known contribution margin calculation"
    )
    included_in_final_contribution: bool = Field(
        default=False, description="True if component satisfies criteria for final contribution margin"
    )
    source_reference: Optional[str] = Field(default=None, description="Field, table, or assumption reference name")
    description: str = Field(default="", description="Audit description of how cost was determined")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        if d.get("amount") is not None:
            d["amount"] = round(float(d["amount"]), 4)
        if d.get("unit_amount") is not None:
            d["unit_amount"] = round(float(d["unit_amount"]), 4)
        return d


class CostAssumptionsConfig(BaseModel):
    """Configurable operational cost assumptions. None of these are hardcoded."""

    model_config = ConfigDict(extra="allow")

    # Shipping assumptions
    shipping_cost_per_order: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed flat outbound shipping fee per sales order"
    )
    shipping_cost_per_unit: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed outbound shipping cost per unit shipped"
    )
    shipping_pct_of_net_revenue: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Assumed shipping cost as percentage of net revenue"
    )
    shipping_cost_by_channel: Dict[str, float] = Field(
        default_factory=dict, description="Assumed flat shipping fee keyed by channel_id"
    )

    # Payment processing assumptions
    payment_processing_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Assumed merchant payment gateway fee rate (e.g. 0.029 for 2.9%)"
    )
    payment_processing_fixed_per_order: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed fixed transaction fee per sales order (e.g. 0.30)"
    )
    payment_rate_by_channel: Dict[str, float] = Field(
        default_factory=dict, description="Assumed payment fee rate keyed by channel_id"
    )
    payment_fixed_by_channel: Dict[str, float] = Field(
        default_factory=dict, description="Assumed fixed transaction fee keyed by channel_id"
    )

    # Packaging assumptions
    packaging_cost_per_order: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed packaging box/materials cost per sales order"
    )
    packaging_cost_per_unit: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed packaging materials cost per unit"
    )

    # Warehouse handling assumptions
    warehouse_handling_cost_per_order: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed pick/pack labor cost per sales order"
    )
    warehouse_handling_cost_per_unit: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed pick/pack labor cost per unit"
    )
    warehouse_handling_by_facility: Dict[str, float] = Field(
        default_factory=dict, description="Assumed handling fee keyed by warehouse_id"
    )

    # Return processing assumptions
    return_cost_per_returned_unit: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed reverse logistics & inspection fee per returned unit"
    )
    return_cost_per_return_event: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed fixed return handling fee per return event"
    )

    # Other variable cost assumptions
    other_variable_cost_per_unit: Optional[float] = Field(
        default=None, ge=0.0, description="Assumed other miscellaneous variable cost per unit"
    )
    other_variable_cost_pct: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Assumed other variable cost as % of net revenue"
    )

    @field_validator(
        "payment_processing_rate",
        "shipping_pct_of_net_revenue",
        "other_variable_cost_pct",
        mode="before",
    )
    @classmethod
    def validate_rate_bounds(cls, v: Any) -> Any:
        if v is not None:
            v_float = float(v)
            if v_float < 0.0 or v_float > 1.0:
                raise ValueError(f"Rate must be between 0.0 and 1.0, got {v_float}")
            return v_float
        return v

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class OperationalEconomicsConfig(BaseModel):
    """Configuration governing required cost components, completeness policy, and assumption models."""

    model_config = ConfigDict(extra="allow")

    required_cost_components: List[CostComponent] = Field(
        default_factory=lambda: [
            CostComponent.PRODUCT_COST,
            CostComponent.SHIPPING_COST,
            CostComponent.PAYMENT_PROCESSING_COST,
            CostComponent.PACKAGING_COST,
            CostComponent.WAREHOUSE_HANDLING_COST,
            CostComponent.RETURN_PROCESSING_COST,
        ],
        description="List of cost components strictly required for 100% cost completeness",
    )
    allow_estimated_costs_for_completeness: bool = Field(
        default=True,
        description="If True, CATALOG_ESTIMATE (e.g. standard product cost from catalog) counts as available",
    )
    allow_assumed_costs_for_completeness: bool = Field(
        default=False,
        description="If True, CONFIGURED_ASSUMPTION counts as available for completeness. Strict default is False.",
    )
    strict_source_only: bool = Field(
        default=False,
        description="If True, only observed SOURCE_DATA satisfies completeness. Overrides allow_estimated.",
    )
    min_completeness_threshold: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Minimum completeness percentage required to populate final contribution margin",
    )
    assumptions: CostAssumptionsConfig = Field(
        default_factory=CostAssumptionsConfig,
        description="Configurable operational cost assumptions",
    )
    as_of_date: Optional[str] = Field(
        default=None,
        description="Point-in-time filter date (YYYY-MM-DD); events after this date are excluded",
    )
    default_currency: str = Field(
        default="USD",
        description="Expected currency denomination. Mixed currencies will be flagged or segregated.",
    )
    order_cost_allocation_method: str = Field(
        default="NET_REVENUE",
        description="Method to allocate order-level costs to line items: NET_REVENUE, QUANTITY, or EQUAL",
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class CostCompletenessReport(BaseModel):
    """Comprehensive cost completeness audit report."""

    model_config = ConfigDict(extra="allow")

    completeness_status: CostCompletenessStatus = Field(
        ..., description="Overall cost completeness state: COMPLETE, PARTIALLY_COMPLETE, INSUFFICIENT_DATA"
    )
    completeness_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Percentage of required components available"
    )
    total_required_components: int = Field(default=0, ge=0, description="Number of required cost components")
    available_required_components: int = Field(default=0, ge=0, description="Count of required components available")
    missing_required_components: int = Field(default=0, ge=0, description="Count of required components missing")
    required_components: List[str] = Field(default_factory=list, description="Names of required components")
    available_components: List[str] = Field(default_factory=list, description="Names of available components")
    unavailable_components: List[str] = Field(default_factory=list, description="Names of missing components")
    source_components: List[str] = Field(default_factory=list, description="Components sourced from observed data")
    estimated_components: List[str] = Field(default_factory=list, description="Components derived from catalog estimates")
    assumed_components: List[str] = Field(default_factory=list, description="Components derived from assumptions")
    audit_notes: List[str] = Field(default_factory=list, description="Audit notes and diagnostic messages")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["completeness_pct"] = round(float(d["completeness_pct"]), 2)
        return d


class OperationalEconomicsRecord(BaseModel):
    """Deterministic transaction-level operational economics record."""

    model_config = ConfigDict(extra="allow")

    record_id: str = Field(..., description="Unique deterministic record ID (hash of keys)")
    transaction_id: str = Field(..., description="Order-line sales transaction ID")
    order_id: str = Field(..., description="Sales order identifier")
    order_date: str = Field(..., description="Sales order date (YYYY-MM-DD)")
    sku_id: str = Field(..., description="Product SKU ID")
    warehouse_id: str = Field(..., description="Fulfillment facility ID")
    channel_id: str = Field(..., description="Sales channel ID")
    currency: str = Field(default="USD", description="Currency denomination")

    # Volume and pricing
    quantity: Optional[int] = Field(default=None, description="Observed units sold")
    unit_price: Optional[float] = Field(default=None, description="Observed unit selling price")
    gross_revenue: Optional[float] = Field(default=None, description="Gross revenue: quantity * unit_price")
    discount: Optional[float] = Field(default=None, description="Promotional discount applied")
    net_revenue: Optional[float] = Field(default=None, description="Net realized revenue: gross_revenue - discount")

    # Product economics
    unit_product_cost: Optional[float] = Field(default=None, description="Standard procurement unit cost")
    product_cost: Optional[float] = Field(default=None, description="Total COGS: quantity * unit_product_cost")
    gross_margin: Optional[float] = Field(default=None, description="Gross commercial margin: net_revenue - product_cost")
    gross_margin_pct: Optional[float] = Field(default=None, description="Gross margin %: gross_margin / net_revenue")

    # Operational cost lines
    shipping_cost: Optional[float] = Field(default=None, description="Outbound shipping cost allocated to line")
    payment_processing_cost: Optional[float] = Field(default=None, description="Payment gateway transaction fee allocated to line")
    packaging_cost: Optional[float] = Field(default=None, description="Packaging materials cost allocated to line")
    warehouse_handling_cost: Optional[float] = Field(default=None, description="Warehouse pick/pack labor cost allocated to line")
    return_processing_cost: Optional[float] = Field(default=None, description="Return processing & inspection fee")
    other_variable_cost: Optional[float] = Field(default=None, description="Other miscellaneous variable costs")
    total_known_variable_cost: float = Field(
        default=0.0, ge=0.0, description="Sum of all currently available variable cost components"
    )

    # Contribution economics
    known_contribution_margin: Optional[float] = Field(
        default=None, description="Net revenue - product cost - total known variable costs"
    )
    known_contribution_margin_pct: Optional[float] = Field(
        default=None, description="Known contribution margin as % of net revenue"
    )
    final_contribution_margin: Optional[float] = Field(
        default=None,
        description="Final contribution margin; populated ONLY when all required cost components meet completeness policy",
    )
    final_contribution_margin_pct: Optional[float] = Field(
        default=None,
        description="Final contribution margin %; populated ONLY when all required cost components meet completeness policy",
    )
    final_contribution_margin_status: ContributionMarginStatus = Field(
        default=ContributionMarginStatus.INSUFFICIENT_COST_DATA,
        description="Calculability state: COMPLETE, PARTIALLY_CALCULABLE, or INSUFFICIENT_COST_DATA",
    )

    # Per-unit metrics
    net_revenue_per_unit: Optional[float] = Field(default=None, description="Net revenue per unit")
    gross_margin_per_unit: Optional[float] = Field(default=None, description="Gross margin per unit")
    variable_cost_per_unit: Optional[float] = Field(default=None, description="Known variable cost per unit")
    contribution_margin_per_unit: Optional[float] = Field(default=None, description="Contribution margin per unit")

    # Quality and provenance
    cost_completeness_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Percentage of required cost components available"
    )
    cost_completeness_status: CostCompletenessStatus = Field(
        default=CostCompletenessStatus.INSUFFICIENT_DATA,
        description="Cost completeness state: COMPLETE, PARTIALLY_COMPLETE, or INSUFFICIENT_DATA",
    )
    economics_status: OperationalEconomicsStatus = Field(
        ..., description="Overall operational economics status"
    )
    status_rationale: str = Field(..., description="Deterministic explanation of status and cost coverage")
    cost_details: Dict[str, OperationalCostDetail] = Field(
        default_factory=dict, description="Detailed per-component metadata and traceability"
    )

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "unit_price", "gross_revenue", "discount", "net_revenue",
            "unit_product_cost", "product_cost", "gross_margin", "gross_margin_pct",
            "shipping_cost", "payment_processing_cost", "packaging_cost",
            "warehouse_handling_cost", "return_processing_cost", "other_variable_cost",
            "total_known_variable_cost", "known_contribution_margin", "known_contribution_margin_pct",
            "final_contribution_margin", "final_contribution_margin_pct",
            "net_revenue_per_unit", "gross_margin_per_unit", "variable_cost_per_unit", "contribution_margin_per_unit",
            "cost_completeness_pct",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "pct" in k else 2)
        return d


class OrderOperationalEconomicsRecord(BaseModel):
    """Deterministic order-level operational economics record aggregating without multi-line double-counting."""

    model_config = ConfigDict(extra="allow")

    order_id: str = Field(..., description="Sales order identifier")
    order_date: str = Field(..., description="Sales order date (YYYY-MM-DD)")
    channel_id: str = Field(..., description="Sales channel ID")
    warehouse_id: Optional[str] = Field(default=None, description="Fulfillment facility ID (or primary facility)")
    currency: str = Field(default="USD", description="Currency denomination")

    # Order volumes
    line_count: int = Field(default=1, ge=1, description="Number of order line items in sales order")
    total_quantity: int = Field(default=0, ge=0, description="Total units across all lines")

    # Order revenue
    gross_revenue: Optional[float] = Field(default=None, description="Total order gross revenue")
    discount: Optional[float] = Field(default=None, description="Total order promotional discounts")
    net_revenue: Optional[float] = Field(default=None, description="Total order net revenue")

    # Order product economics
    product_cost: Optional[float] = Field(default=None, description="Total order COGS across all lines")
    gross_margin: Optional[float] = Field(default=None, description="Total order gross margin")
    gross_margin_pct: Optional[float] = Field(default=None, description="Order gross margin %")

    # True order-level operational costs (un-allocated, exact)
    shipping_cost: Optional[float] = Field(default=None, description="Total outbound order shipping cost")
    payment_processing_cost: Optional[float] = Field(default=None, description="Total payment processing fee for order")
    packaging_cost: Optional[float] = Field(default=None, description="Total packaging materials cost for order")
    warehouse_handling_cost: Optional[float] = Field(default=None, description="Total warehouse handling cost for order")
    return_processing_cost: Optional[float] = Field(default=None, description="Total return processing cost for order")
    other_variable_cost: Optional[float] = Field(default=None, description="Total other variable costs for order")
    total_known_variable_cost: float = Field(
        default=0.0, ge=0.0, description="Sum of all available variable cost components for order"
    )

    # Order contribution economics
    known_contribution_margin: Optional[float] = Field(default=None, description="Order known contribution margin")
    known_contribution_margin_pct: Optional[float] = Field(default=None, description="Order known contribution margin %")
    final_contribution_margin: Optional[float] = Field(default=None, description="Order final contribution margin")
    final_contribution_margin_pct: Optional[float] = Field(default=None, description="Order final contribution margin %")
    final_contribution_margin_status: ContributionMarginStatus = Field(
        default=ContributionMarginStatus.INSUFFICIENT_COST_DATA, description="Calculability state of order contribution margin"
    )

    # Completeness and quality
    cost_completeness_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Percentage of required components available at order level"
    )
    cost_completeness_status: CostCompletenessStatus = Field(
        default=CostCompletenessStatus.INSUFFICIENT_DATA, description="Order cost completeness status"
    )
    economics_status: OperationalEconomicsStatus = Field(
        ..., description="Overall order operational economics status"
    )
    cost_details: Dict[str, OperationalCostDetail] = Field(
        default_factory=dict, description="Detailed per-component metadata and traceability for order"
    )

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "gross_revenue", "discount", "net_revenue", "product_cost", "gross_margin", "gross_margin_pct",
            "shipping_cost", "payment_processing_cost", "packaging_cost", "warehouse_handling_cost",
            "return_processing_cost", "other_variable_cost", "total_known_variable_cost",
            "known_contribution_margin", "known_contribution_margin_pct",
            "final_contribution_margin", "final_contribution_margin_pct", "cost_completeness_pct",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "pct" in k else 2)
        return d


class OperationalEconomicsSegment(BaseModel):
    """Aggregated operational economics metrics for an analytical dimension or slice."""

    model_config = ConfigDict(extra="allow")

    dimension: str = Field(..., description="Dimension: SKU, CATEGORY, BRAND, CHANNEL, WAREHOUSE, etc.")
    segment_key: str = Field(..., description="Segment identifier value")
    record_count: int = Field(default=0, ge=0, description="Total order-line records in segment")
    order_count: int = Field(default=0, ge=0, description="Distinct sales orders in segment")
    total_units: int = Field(default=0, ge=0, description="Total units sold in segment")

    # Revenue metrics
    gross_revenue: Optional[float] = Field(default=None, description="Sum of gross revenue")
    discount: Optional[float] = Field(default=None, description="Sum of promotional discounts")
    net_revenue: Optional[float] = Field(default=None, description="Sum of net revenue")

    # Product economics
    product_cost: Optional[float] = Field(default=None, description="Sum of product costs")
    gross_margin: Optional[float] = Field(default=None, description="Sum of gross margin")
    gross_margin_pct: Optional[float] = Field(default=None, description="Aggregate gross margin %")

    # Variable costs
    shipping_cost: Optional[float] = Field(default=None, description="Sum of allocated shipping costs")
    payment_processing_cost: Optional[float] = Field(default=None, description="Sum of allocated payment fees")
    packaging_cost: Optional[float] = Field(default=None, description="Sum of packaging costs")
    warehouse_handling_cost: Optional[float] = Field(default=None, description="Sum of handling costs")
    return_processing_cost: Optional[float] = Field(default=None, description="Sum of return costs")
    other_variable_cost: Optional[float] = Field(default=None, description="Sum of other variable costs")
    total_known_variable_cost: float = Field(default=0.0, ge=0.0, description="Sum of total known variable costs")

    # Contribution economics
    known_contribution_margin: Optional[float] = Field(default=None, description="Sum of known contribution margin")
    known_contribution_margin_pct: Optional[float] = Field(default=None, description="Known contribution margin %")
    final_contribution_margin: Optional[float] = Field(default=None, description="Sum of final contribution margin")
    final_contribution_margin_pct: Optional[float] = Field(default=None, description="Final contribution margin %")
    final_contribution_margin_status: ContributionMarginStatus = Field(
        default=ContributionMarginStatus.INSUFFICIENT_COST_DATA, description="Calculability state of segment contribution margin"
    )

    # Completeness breakdown
    avg_cost_completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Average completeness % across records")
    complete_records_count: int = Field(default=0, ge=0, description="Count of complete records")
    partially_complete_records_count: int = Field(default=0, ge=0, description="Count of partially complete records")
    insufficient_records_count: int = Field(default=0, ge=0, description="Count of insufficient data records")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "gross_revenue", "discount", "net_revenue", "product_cost", "gross_margin", "gross_margin_pct",
            "shipping_cost", "payment_processing_cost", "packaging_cost", "warehouse_handling_cost",
            "return_processing_cost", "other_variable_cost", "total_known_variable_cost",
            "known_contribution_margin", "known_contribution_margin_pct",
            "final_contribution_margin", "final_contribution_margin_pct", "avg_cost_completeness_pct",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "pct" in k else 2)
        return d


class OperationalEconomicsSummary(BaseModel):
    """Network-level operational economics executive portfolio summary."""

    model_config = ConfigDict(extra="allow")

    total_records: int = Field(default=0, ge=0, description="Total order-line records analyzed")
    total_orders: int = Field(default=0, ge=0, description="Total distinct sales orders")
    total_units: int = Field(default=0, ge=0, description="Total units sold")

    # Portfolio revenues
    gross_revenue: float = Field(default=0.0, description="Total gross revenue ($)")
    discount: float = Field(default=0.0, description="Total promotional discounts ($)")
    net_revenue: float = Field(default=0.0, description="Total net realized revenue ($)")

    # Portfolio product cost and gross margin
    product_cost: float = Field(default=0.0, description="Total procurement product cost ($)")
    gross_margin: float = Field(default=0.0, description="Total commercial gross margin ($)")
    gross_margin_pct: Optional[float] = Field(default=None, description="Overall gross margin %")

    # Portfolio operational costs
    shipping_cost: float = Field(default=0.0, description="Total shipping expenses ($)")
    payment_processing_cost: float = Field(default=0.0, description="Total payment processing fees ($)")
    packaging_cost: float = Field(default=0.0, description="Total packaging materials cost ($)")
    warehouse_handling_cost: float = Field(default=0.0, description="Total warehouse labor handling cost ($)")
    return_processing_cost: float = Field(default=0.0, description="Total return processing cost ($)")
    other_variable_cost: float = Field(default=0.0, description="Total other variable expenses ($)")
    total_known_variable_cost: float = Field(default=0.0, description="Total known variable operational costs ($)")

    # Portfolio contribution economics
    known_contribution_margin: float = Field(default=0.0, description="Total known contribution margin ($)")
    known_contribution_margin_pct: Optional[float] = Field(default=None, description="Known contribution margin %")
    final_contribution_margin: Optional[float] = Field(
        default=None, description="Final contribution margin ($); None if any required cost is unavailable"
    )
    final_contribution_margin_pct: Optional[float] = Field(
        default=None, description="Final contribution margin %; None if any required cost is unavailable"
    )
    final_contribution_margin_status: ContributionMarginStatus = Field(
        default=ContributionMarginStatus.INSUFFICIENT_COST_DATA, description="Calculability state of portfolio contribution margin"
    )

    # Auditing & metadata
    cost_completeness_report: CostCompletenessReport = Field(
        ..., description="Complete audit report on cost component availability and provenance"
    )
    cost_details: Dict[str, OperationalCostDetail] = Field(
        default_factory=dict, description="Detailed per-component metadata and traceability"
    )
    currency: str = Field(default="USD", description="Currency denomination")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time filter applied if any")
    generated_at: str = Field(..., description="ISO 8601 generation timestamp")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "gross_revenue", "discount", "net_revenue", "product_cost", "gross_margin", "gross_margin_pct",
            "shipping_cost", "payment_processing_cost", "packaging_cost", "warehouse_handling_cost",
            "return_processing_cost", "other_variable_cost", "total_known_variable_cost",
            "known_contribution_margin", "known_contribution_margin_pct",
        ]:
            d[k] = round(float(d[k]), 4 if "pct" in k else 2)
        if d.get("final_contribution_margin") is not None:
            d["final_contribution_margin"] = round(float(d["final_contribution_margin"]), 2)
        if d.get("final_contribution_margin_pct") is not None:
            d["final_contribution_margin_pct"] = round(float(d["final_contribution_margin_pct"]), 4)
        return d
