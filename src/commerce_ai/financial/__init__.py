"""Financial Intelligence & Unit Economics Module (Phases 6A & 6B).

Provides deterministic, factual financial analytics for eRetail transactions:
- Revenue calculation and reconciliation (gross revenue, discounts, net revenue, source variance)
- Standard procurement cost of goods sold (estimated COGS / product cost)
- Gross margin and gross margin percentage with zero-denominator safety
- Variable cost breakdown (shipping, payment fees, packaging, handling, return fees, other)
- Known and full contribution margin with strict calculability status
- Cost completeness evaluation and source traceability
- Multi-dimensional aggregations (SKU, Category, Brand, Channel, Warehouse, Matrix cross-slices)
- Time intelligence across daily, weekly, and monthly granularities
- Neutral analytical rankings & descriptive margin erosion analysis
- Point-in-time anti-leakage filtering
- Comprehensive data quality auditing
"""

from commerce_ai.financial.schemas import (
    # Phase 6A Schemas
    FinancialDataQualityReport,
    FinancialDataStatus,
    FinancialDimensionMetric,
    FinancialIntelligenceConfig,
    FinancialIntelligenceResult,
    FinancialPortfolioSummary,
    MarginClassification,
    RankingResult,
    RevenueMarginRecord,
    RevenueReconciliationStatus,
    TimeGrain,
    # Phase 6B Schemas
    ContributionMarginStatus,
    CostComponent,
    CostComponentDetail,
    CostComponentStatus,
    CostModelConfig,
    CostSourceType,
    MarginErosionReport,
    MarginErosionSegment,
    UnitEconomicsDimensionMetric,
    UnitEconomicsPortfolioSummary,
    UnitEconomicsRecord,
    UnitEconomicsResult,
    UnitEconomicsStatus,
)
from commerce_ai.financial.revenue_margin import (
    aggregate_financial_dimension,
    aggregate_time_series,
    analyze_negative_margins,
    audit_financial_data_quality,
    calculate_revenue_margin,
    compute_revenue_margin_dataframe,
    compute_revenue_margin_records,
    filter_sales_by_as_of_date,
    generate_financial_portfolio_id,
    generate_financial_record_id,
    generate_financial_segment_id,
    rank_segments,
    summarize_financial_portfolio,
)
from commerce_ai.financial.cost_model import CostModelResolver
from commerce_ai.financial.unit_economics import (
    aggregate_unit_economics_dimension,
    aggregate_unit_economics_time_series,
    analyze_margin_erosion,
    calculate_unit_economics_record,
    compute_unit_economics_dataframe,
    compute_unit_economics_records,
    generate_unit_economics_record_id,
    generate_unit_economics_segment_id,
    summarize_unit_economics_portfolio,
)
from commerce_ai.financial.service import FinancialIntelligenceService, UnitEconomicsService

__all__ = [
    # Schemas (Phase 6A)
    "FinancialDataQualityReport",
    "FinancialDataStatus",
    "FinancialDimensionMetric",
    "FinancialIntelligenceConfig",
    "FinancialIntelligenceResult",
    "FinancialPortfolioSummary",
    "MarginClassification",
    "RankingResult",
    "RevenueMarginRecord",
    "RevenueReconciliationStatus",
    "TimeGrain",
    # Schemas (Phase 6B)
    "ContributionMarginStatus",
    "CostComponent",
    "CostComponentDetail",
    "CostComponentStatus",
    "CostModelConfig",
    "CostSourceType",
    "MarginErosionReport",
    "MarginErosionSegment",
    "UnitEconomicsDimensionMetric",
    "UnitEconomicsPortfolioSummary",
    "UnitEconomicsRecord",
    "UnitEconomicsResult",
    "UnitEconomicsStatus",
    # Functions (Phase 6A)
    "aggregate_financial_dimension",
    "aggregate_time_series",
    "analyze_negative_margins",
    "audit_financial_data_quality",
    "calculate_revenue_margin",
    "compute_revenue_margin_dataframe",
    "compute_revenue_margin_records",
    "filter_sales_by_as_of_date",
    "generate_financial_portfolio_id",
    "generate_financial_record_id",
    "generate_financial_segment_id",
    "rank_segments",
    "summarize_financial_portfolio",
    # Functions (Phase 6B)
    "CostModelResolver",
    "aggregate_unit_economics_dimension",
    "aggregate_unit_economics_time_series",
    "analyze_margin_erosion",
    "calculate_unit_economics_record",
    "compute_unit_economics_dataframe",
    "compute_unit_economics_records",
    "generate_unit_economics_record_id",
    "generate_unit_economics_segment_id",
    "summarize_unit_economics_portfolio",
    # Services
    "FinancialIntelligenceService",
    "UnitEconomicsService",
]
