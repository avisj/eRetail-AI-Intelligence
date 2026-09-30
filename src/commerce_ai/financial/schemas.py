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
from typing import Any, Dict, List, Optional, Sequence, Union
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


