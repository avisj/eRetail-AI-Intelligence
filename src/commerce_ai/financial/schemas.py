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

