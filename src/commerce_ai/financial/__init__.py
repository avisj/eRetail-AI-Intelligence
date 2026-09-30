"""Financial Intelligence Module (Phase 6A).

Provides deterministic, factual financial analytics for eRetail transactions:
- Revenue calculation and reconciliation (gross revenue, discounts, net revenue, source variance)
- Standard procurement cost of goods sold (estimated COGS)
- Gross margin and gross margin percentage with zero-denominator safety
- Multi-dimensional aggregations (SKU, Category, Brand, Channel, Warehouse, Matrix slices)
- Time intelligence across daily, weekly, and monthly granularities
- Neutral analytical rankings (top revenue, top margin, top units, lowest margin %, negative margin)
- Point-in-time anti-leakage filtering
- Comprehensive data quality auditing
"""

from commerce_ai.financial.schemas import (
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
from commerce_ai.financial.service import FinancialIntelligenceService

__all__ = [
    # Schemas
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
    # Functions
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
    # Service
    "FinancialIntelligenceService",
]
