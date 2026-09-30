"""Financial Intelligence, Unit Economics & Profitability Attribution (Phases 6A, 6B & 6C).

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
- Profitability attribution and contribution gap analysis (Phase 6C)
- Margin concentration tiers and ordered cumulative contribution curves (Phase 6C)
- Promotional discount impact analysis and bucket distributions (Phase 6C)
- Margin driver classifications and negative/low margin root-cause flagging (Phase 6C)
- Executive commercial portfolio margin waterfall (Phase 6C)
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
    # Phase 6C Schemas
    AttributionReasonCode,
    DiscountBucketMetric,
    DiscountImpactSummary,
    MarginConcentrationResult,
    MarginConcentrationTier,
    MarginContributionPoint,
    MarginDriverClassification,
    MarginDriverSegment,
    MarginWaterfall,
    MarginWaterfallStage,
    ProfitabilityAttributionConfig,
    ProfitabilityAttributionRecord,
    ProfitabilityAttributionResult,
    SKUProfitabilityProfile,
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
from commerce_ai.financial.concentration import (
    calculate_margin_concentration,
    calculate_margin_contribution_curve,
)
from commerce_ai.financial.discount_analysis import (
    analyze_discount_buckets,
    analyze_discount_impact,
)
from commerce_ai.financial.margin_drivers import (
    analyze_negative_and_low_margins,
    build_margin_waterfall,
    classify_margin_drivers,
    evaluate_sku_driver_classifications,
)
from commerce_ai.financial.attribution import (
    ProfitabilityAttributionService,
    compute_dimensional_profitability_attribution,
    compute_sku_profitability_profiles,
    compute_temporal_profitability_attribution,
)
from commerce_ai.financial.service import (
    FinancialIntelligenceService,
    UnitEconomicsService,
)

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
    # Schemas (Phase 6C)
    "AttributionReasonCode",
    "DiscountBucketMetric",
    "DiscountImpactSummary",
    "MarginConcentrationResult",
    "MarginConcentrationTier",
    "MarginContributionPoint",
    "MarginDriverClassification",
    "MarginDriverSegment",
    "MarginWaterfall",
    "MarginWaterfallStage",
    "ProfitabilityAttributionConfig",
    "ProfitabilityAttributionRecord",
    "ProfitabilityAttributionResult",
    "SKUProfitabilityProfile",
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
    # Functions & Classes (Phase 6C)
    "calculate_margin_concentration",
    "calculate_margin_contribution_curve",
    "analyze_discount_buckets",
    "analyze_discount_impact",
    "analyze_negative_and_low_margins",
    "build_margin_waterfall",
    "classify_margin_drivers",
    "evaluate_sku_driver_classifications",
    "compute_dimensional_profitability_attribution",
    "compute_sku_profitability_profiles",
    "compute_temporal_profitability_attribution",
    # Services
    "FinancialIntelligenceService",
    "UnitEconomicsService",
    "ProfitabilityAttributionService",
]
