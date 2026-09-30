"""Financial Intelligence Service (Phase 6A).

Orchestrates the foundational Financial Intelligence pipeline:
1. Data Quality Audit & Referential Integrity Validation
2. Anti-Leakage Point-in-Time Cutoff Filtering
3. Unit Cost & Metadata Enrichment
4. Deterministic Revenue, COGS, and Gross Margin Calculation
5. Multi-Dimensional Aggregation (SKU, Category, Brand, Channel, Warehouse, Matrix slices)
6. Temporal Aggregation (Daily, Weekly, Monthly)
7. Neutral Analytical Rankings
8. Negative Margin Component Analysis
9. Executive Portfolio Summary Generation
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd

from commerce_ai.financial.schemas import (
    FinancialDataQualityReport,
    FinancialDimensionMetric,
    FinancialIntelligenceConfig,
    FinancialIntelligenceResult,
    FinancialPortfolioSummary,
    RankingResult,
    RevenueMarginRecord,
    TimeGrain,
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


class FinancialIntelligenceService:
    """Service providing deterministic, factual financial analytics for eRetail transactions."""

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
        """Run the comprehensive financial intelligence analysis.

        Args:
            sales: Historical sales transactions DataFrame.
            products: Optional product catalog with unit_cost and metadata.
            channels: Optional channels dimension metadata.
            warehouses: Optional warehouse facilities metadata.
            as_of_date: Point-in-time historical cutoff date (excludes records after this date).
            config: Optional override configuration.
            dimensions: List of dimensions to aggregate. Defaults to all supported dimensions.
            time_grain: Temporal grain for time series ('DAILY', 'WEEKLY', 'MONTHLY').
            ranking_top_n: Number of top/bottom items to return in rankings.
            include_records: Whether to include granular line-level RevenueMarginRecord objects.

        Returns:
            FinancialIntelligenceResult with summary, dimension aggregates, time series, rankings,
            negative-margin records, and data quality audit.
        """
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
