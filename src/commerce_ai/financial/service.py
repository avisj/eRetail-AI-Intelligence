"""Financial Intelligence & Unit Economics Services (Phases 6A & 6B).

Orchestrates:
1. Revenue and Gross Margin Analysis (Phase 6A)
2. Cost Breakdown & True Unit Economics Analysis (Phase 6B)
3. Data Quality Auditing & Referential Integrity Validation
4. Point-in-Time Anti-Leakage Filtering
5. Multi-Dimensional Aggregations & Temporal Slicing
6. Neutral Analytical Rankings & Margin Erosion Reporting
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd

from commerce_ai.financial.schemas import (
    CostComponent,
    CostComponentDetail,
    CostModelConfig,
    FinancialDataQualityReport,
    FinancialDimensionMetric,
    FinancialIntelligenceConfig,
    FinancialIntelligenceResult,
    FinancialPortfolioSummary,
    MarginErosionReport,
    RankingResult,
    RevenueMarginRecord,
    TimeGrain,
    UnitEconomicsDimensionMetric,
    UnitEconomicsPortfolioSummary,
    UnitEconomicsRecord,
    UnitEconomicsResult,
)
from commerce_ai.financial.revenue_margin import (
    aggregate_financial_dimension,
    aggregate_time_series,
    analyze_negative_margins,
    audit_financial_data_quality,
    compute_revenue_margin_dataframe,
    compute_revenue_margin_records,
    filter_sales_by_as_of_date,
    rank_segments,
    summarize_financial_portfolio,
)
from commerce_ai.financial.cost_model import CostModelResolver
from commerce_ai.financial.unit_economics import (
    aggregate_unit_economics_dimension,
    aggregate_unit_economics_time_series,
    analyze_margin_erosion as analyze_margin_erosion_fn,
    compute_unit_economics_dataframe,
    compute_unit_economics_records,
    summarize_unit_economics_portfolio,
)


class FinancialIntelligenceService:
    """Service providing deterministic, factual financial analytics for eRetail transactions (Phase 6A)."""

    def __init__(self, config: Optional[FinancialIntelligenceConfig] = None):
        self.config = config or FinancialIntelligenceConfig()

    def audit_quality(
        self,
        sales: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[FinancialIntelligenceConfig] = None,
    ) -> FinancialDataQualityReport:
        """Run standalone data quality audit on input sales and product cost records."""
        cfg = config or (self.config.model_copy(update={"as_of_date": as_of_date}) if as_of_date else self.config)
        return audit_financial_data_quality(sales_df=sales, products_df=products, config=cfg)

    def analyze(
        self,
        sales: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        channels: Optional[pd.DataFrame] = None,
        warehouses: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[FinancialIntelligenceConfig] = None,
        dimensions: Optional[List[str]] = None,
        time_grain: Union[str, TimeGrain] = TimeGrain.MONTHLY,
        ranking_top_n: int = 10,
        include_records: bool = False,
    ) -> FinancialIntelligenceResult:
        """Run the comprehensive financial intelligence analysis (Phase 6A)."""
        effective_cfg = config or self.config
        if as_of_date is not None:
            effective_cfg = effective_cfg.model_copy(update={"as_of_date": as_of_date})

        # 1. Audit Data Quality on raw input before filtering
        quality_report = audit_financial_data_quality(
            sales_df=sales,
            products_df=products,
            config=effective_cfg,
        )

        # 2. Apply Chronological Anti-Leakage Filtering
        filtered_sales = filter_sales_by_as_of_date(sales, effective_cfg.as_of_date)

        # 3. Compute Vectorized Revenue, Estimated COGS, and Margin
        calc_df = compute_revenue_margin_dataframe(
            sales_df=filtered_sales,
            products_df=products,
            config=effective_cfg,
        )

        # 4. Generate Executive Portfolio Summary
        portfolio_summary = summarize_financial_portfolio(
            df=calc_df,
            quality_report=quality_report,
            config=effective_cfg,
        )

        total_portfolio_rev = portfolio_summary.total_net_revenue
        total_portfolio_margin = portfolio_summary.total_gross_margin

        # 5. Multi-dimensional Aggregations
        default_dims = [
            "SKU",
            "CATEGORY",
            "BRAND",
            "CHANNEL",
            "WAREHOUSE",
            "SKU_CHANNEL",
            "SKU_WAREHOUSE",
            "CHANNEL_WAREHOUSE",
        ]
        active_dims = dimensions or default_dims
        dim_metrics: Dict[str, List[FinancialDimensionMetric]] = {}

        for dim in active_dims:
            dim_key = dim.strip().upper()
            metrics = aggregate_financial_dimension(
                df=calc_df,
                dimension=dim_key,
                total_portfolio_revenue=total_portfolio_rev,
                total_portfolio_margin=total_portfolio_margin,
                config=effective_cfg,
            )
            dim_metrics[dim_key] = metrics

        # 6. Temporal Aggregation
        time_series = aggregate_time_series(
            df=calc_df,
            grain=time_grain,
            total_portfolio_revenue=total_portfolio_rev,
            total_portfolio_margin=total_portfolio_margin,
            config=effective_cfg,
        )

        # 7. Deterministic Rankings
        rankings: Dict[str, RankingResult] = {}
        sku_metrics = dim_metrics.get("SKU", [])
        if sku_metrics:
            rankings["top_skus_by_revenue"] = rank_segments(sku_metrics, "highest_revenue", top_n=ranking_top_n)
            rankings["top_skus_by_margin"] = rank_segments(sku_metrics, "highest_margin", top_n=ranking_top_n)
            rankings["top_skus_by_units"] = rank_segments(sku_metrics, "highest_units", top_n=ranking_top_n)
            rankings["bottom_skus_by_margin_pct"] = rank_segments(sku_metrics, "lowest_margin_percentage", top_n=ranking_top_n)
            rankings["skus_with_negative_margin"] = rank_segments(sku_metrics, "negative_margin", top_n=ranking_top_n)

        # 8. Negative Margin Order Lines
        negative_margin_records: List[RevenueMarginRecord] = []
        if not calc_df.empty:
            neg_mask = calc_df["gross_margin"] < 0
            if neg_mask.any():
                neg_df = calc_df.loc[neg_mask]
                negative_margin_records = compute_revenue_margin_records(neg_df, products, effective_cfg)

        # 9. Optional granular records
        records = None
        if include_records:
            records = compute_revenue_margin_records(calc_df, products, effective_cfg)

        return FinancialIntelligenceResult(
            portfolio_summary=portfolio_summary,
            dimension_metrics=dim_metrics,
            time_series=time_series,
            rankings=rankings,
            negative_margin_records=negative_margin_records,
            data_quality_report=quality_report,
            records=records,
        )

    def analyze_unit_economics(
        self,
        sales: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        channels: Optional[pd.DataFrame] = None,
        warehouses: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        cost_config: Optional[CostModelConfig] = None,
        dimensions: Optional[List[str]] = None,
        time_grain: Union[str, TimeGrain] = TimeGrain.MONTHLY,
        include_records: bool = False,
    ) -> UnitEconomicsResult:
        """Convenience method delegating to UnitEconomicsService (Phase 6B)."""
        svc = UnitEconomicsService(config=cost_config)
        return svc.calculate_unit_economics(
            sales=sales,
            products=products,
            channels=channels,
            warehouses=warehouses,
            as_of_date=as_of_date,
            config=cost_config,
            dimensions=dimensions,
            time_grain=time_grain,
            include_records=include_records,
        )


class UnitEconomicsService:
    """Service providing deterministic unit economics, variable cost breakdowns, and margin erosion analysis (Phase 6B)."""

    def __init__(self, config: Optional[CostModelConfig] = None):
        self.config = config or CostModelConfig()
        self.resolver = CostModelResolver(self.config)

    def run_financial_quality_checks(
        self,
        sales: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[CostModelConfig] = None,
    ) -> FinancialDataQualityReport:
        """Run data quality checks against sales and catalog master datasets."""
        cfg = config or self.config
        fin_cfg = FinancialIntelligenceConfig(
            as_of_date=as_of_date or cfg.as_of_date,
            default_currency=cfg.default_currency,
            allow_multi_currency=cfg.allow_multi_currency,
        )
        return audit_financial_data_quality(sales_df=sales, products_df=products, config=fin_cfg)

    def calculate_cost_breakdown(
        self,
        sales: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[CostModelConfig] = None,
    ) -> pd.DataFrame:
        """Compute vectorized unit economics and variable cost breakdown DataFrame."""
        cfg = config or self.config
        if as_of_date is not None:
            cfg = cfg.model_copy(update={"as_of_date": as_of_date})

        filtered_sales = filter_sales_by_as_of_date(sales, cfg.as_of_date)
        return compute_unit_economics_dataframe(
            sales_df=filtered_sales,
            products_df=products,
            config=cfg,
        )

    def calculate_cost_completeness(
        self,
        cost_details: Dict[str, CostComponentDetail],
        required_components: Optional[List[CostComponent]] = None,
    ) -> Tuple[float, List[str], List[str], List[str], List[str]]:
        """Calculate deterministic cost completeness for a set of cost component details."""
        return self.resolver.calculate_cost_completeness(cost_details, required_components)

    def aggregate_unit_economics(
        self,
        df: pd.DataFrame,
        dimension: str,
        total_portfolio_revenue: Optional[float] = None,
        total_portfolio_margin: Optional[float] = None,
        config: Optional[CostModelConfig] = None,
    ) -> List[UnitEconomicsDimensionMetric]:
        """Aggregate unit economics across a specified dimension."""
        cfg = config or self.config
        return aggregate_unit_economics_dimension(
            df=df,
            dimension=dimension,
            total_portfolio_revenue=total_portfolio_revenue,
            total_portfolio_margin=total_portfolio_margin,
            config=cfg,
        )

    def analyze_margin_erosion(
        self,
        sku_metrics: Sequence[UnitEconomicsDimensionMetric],
        channel_metrics: Sequence[UnitEconomicsDimensionMetric],
        warehouse_metrics: Sequence[UnitEconomicsDimensionMetric],
        portfolio_margin_pct: Optional[float] = None,
        low_margin_threshold: float = 0.20,
        mismatch_threshold: float = 0.02,
    ) -> MarginErosionReport:
        """Run descriptive margin erosion analysis across SKU, channel, and warehouse metrics."""
        return analyze_margin_erosion_fn(
            sku_metrics=sku_metrics,
            channel_metrics=channel_metrics,
            warehouse_metrics=warehouse_metrics,
            portfolio_margin_pct=portfolio_margin_pct,
            low_margin_threshold=low_margin_threshold,
            mismatch_threshold=mismatch_threshold,
        )

    def calculate_unit_economics(
        self,
        sales: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        channels: Optional[pd.DataFrame] = None,
        warehouses: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[CostModelConfig] = None,
        dimensions: Optional[List[str]] = None,
        time_grain: Union[str, TimeGrain] = TimeGrain.MONTHLY,
        low_margin_threshold: float = 0.20,
        mismatch_threshold: float = 0.02,
        include_records: bool = False,
    ) -> UnitEconomicsResult:
        """Run end-to-end Unit Economics analysis pipeline (Phase 6B).

        Args:
            sales: Transactional sales DataFrame.
            products: Optional product catalog with unit_cost and metadata.
            channels: Optional channels dimension metadata.
            warehouses: Optional warehouse facilities metadata.
            as_of_date: Point-in-time historical cutoff date.
            config: Optional CostModelConfig overriding default configuration.
            dimensions: List of analytical dimensions to aggregate.
            time_grain: Time series aggregation grain ('DAILY', 'WEEKLY', 'MONTHLY').
            low_margin_threshold: Gross margin % below which a SKU is flagged as low margin.
            mismatch_threshold: Difference between revenue share and margin share to flag dilution.
            include_records: Whether to populate individual UnitEconomicsRecord objects.

        Returns:
            UnitEconomicsResult containing portfolio summary, dimension metrics, time series,
            margin erosion report, data quality audit, cost model summary, and optional records.
        """
        effective_cfg = config or self.config
        if as_of_date is not None:
            effective_cfg = effective_cfg.model_copy(update={"as_of_date": as_of_date})

        resolver = CostModelResolver(effective_cfg)

        # 1. Data Quality Audit
        quality_report = self.run_financial_quality_checks(
            sales=sales,
            products=products,
            as_of_date=effective_cfg.as_of_date,
            config=effective_cfg,
        )

        # 2. Anti-leakage chronological filtering
        filtered_sales = filter_sales_by_as_of_date(sales, effective_cfg.as_of_date)

        # 3. Vectorized unit economics calculation
        calc_df = compute_unit_economics_dataframe(
            sales_df=filtered_sales,
            products_df=products,
            config=effective_cfg,
        )

        # 4. Executive portfolio summary
        portfolio_summary = summarize_unit_economics_portfolio(
            df=calc_df,
            quality_report=quality_report,
            config=effective_cfg,
        )

        total_portfolio_rev = portfolio_summary.total_net_revenue
        total_portfolio_margin = portfolio_summary.total_gross_margin
        portfolio_margin_pct = portfolio_summary.gross_margin_pct

        # 5. Multi-dimensional aggregations
        default_dims = [
            "SKU",
            "CATEGORY",
            "BRAND",
            "CHANNEL",
            "WAREHOUSE",
            "SKU_CHANNEL",
            "SKU_WAREHOUSE",
            "CHANNEL_WAREHOUSE",
        ]
        active_dims = dimensions or default_dims
        dim_metrics: Dict[str, List[UnitEconomicsDimensionMetric]] = {}

        for dim in active_dims:
            dim_key = dim.strip().upper()
            metrics = aggregate_unit_economics_dimension(
                df=calc_df,
                dimension=dim_key,
                total_portfolio_revenue=total_portfolio_rev,
                total_portfolio_margin=total_portfolio_margin,
                config=effective_cfg,
            )
            dim_metrics[dim_key] = metrics

        # 6. Temporal aggregations
        time_series = aggregate_unit_economics_time_series(
            df=calc_df,
            grain=time_grain,
            total_portfolio_revenue=total_portfolio_rev,
            total_portfolio_margin=total_portfolio_margin,
            config=effective_cfg,
        )

        # 7. Descriptive Margin Erosion Report
        sku_metrics = dim_metrics.get("SKU", [])
        channel_metrics = dim_metrics.get("CHANNEL", [])
        warehouse_metrics = dim_metrics.get("WAREHOUSE", [])

        erosion_report = analyze_margin_erosion_fn(
            sku_metrics=sku_metrics,
            channel_metrics=channel_metrics,
            warehouse_metrics=warehouse_metrics,
            portfolio_margin_pct=portfolio_margin_pct,
            low_margin_threshold=low_margin_threshold,
            mismatch_threshold=mismatch_threshold,
        )

        # 8. Cost model summary
        cost_summary = resolver.get_cost_model_summary()

        # 9. Optional granular records
        records = None
        if include_records:
            records = compute_unit_economics_records(calc_df, products, effective_cfg)

        return UnitEconomicsResult(
            portfolio_summary=portfolio_summary,
            dimension_metrics=dim_metrics,
            time_series=time_series,
            margin_erosion=erosion_report,
            data_quality_report=quality_report,
            cost_model_summary=cost_summary,
            records=records,
        )


from commerce_ai.financial.attribution import ProfitabilityAttributionService
from commerce_ai.financial.operational_service import OperationalEconomicsService
