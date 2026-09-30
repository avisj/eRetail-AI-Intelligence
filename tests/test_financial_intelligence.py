"""Comprehensive Test Suite for Financial Intelligence Foundation (Phase 6A).

Validates all 36 required test conditions:
1. gross revenue calculation
2. net revenue calculation
3. source revenue reconciliation
4. matching source revenue
5. revenue variance
6. COGS calculation
7. missing unit cost
8. gross margin
9. positive margin
10. zero margin
11. negative margin
12. gross margin percentage
13. zero revenue
14. discount rate
15. missing discount
16. SKU aggregation
17. channel aggregation
18. warehouse aggregation
19. SKU x channel aggregation
20. monthly aggregation
21. distinct order count
22. units aggregation
23. negative-margin detection
24. portfolio summary
25. missing values not treated as zero
26. invalid quantity
27. invalid price
28. invalid cost
29. duplicate sale_id
30. as_of_date leakage protection
31. deterministic IDs
32. deterministic outputs
33. empty dataset
34. multi-currency handling if applicable
35. revenue reconciliation status
36. financial completeness
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

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


@pytest.fixture
def sample_sales():
    return pd.DataFrame([
        {
            "sale_id": "SALE_001",
            "order_id": "ORD_001",
            "date": "2026-03-01",
            "sku_id": "SKU_A",
            "warehouse_id": "WH_EAST",
            "channel_id": "CH_AMZ",
            "quantity": 10,
            "unit_price": 25.0,
            "discount": 10.0,
            "revenue": 240.0,
            "currency": "USD",
        },
        {
            "sale_id": "SALE_002",
            "order_id": "ORD_001",
            "date": "2026-03-01",
            "sku_id": "SKU_B",
            "warehouse_id": "WH_EAST",
            "channel_id": "CH_AMZ",
            "quantity": 5,
            "unit_price": 50.0,
            "discount": 0.0,
            "revenue": 250.0,
            "currency": "USD",
        },
        {
            "sale_id": "SALE_003",
            "order_id": "ORD_002",
            "date": "2026-03-15",
            "sku_id": "SKU_A",
            "warehouse_id": "WH_WEST",
            "channel_id": "CH_SHOPIFY",
            "quantity": 2,
            "unit_price": 30.0,
            "discount": 5.0,
            "revenue": 55.0,
            "currency": "USD",
        },
    ])


@pytest.fixture
def sample_products():
    return pd.DataFrame([
        {
            "sku_id": "SKU_A",
            "product_name": "Product A",
            "category_id": "CAT_TECH",
            "brand": "BrandAlpha",
            "unit_cost": 15.0,
            "selling_price": 25.0,
            "currency": "USD",
        },
        {
            "sku_id": "SKU_B",
            "product_name": "Product B",
            "category_id": "CAT_TECH",
            "brand": "BrandBeta",
            "unit_cost": 30.0,
            "selling_price": 50.0,
            "currency": "USD",
        },
    ])


class TestFinancialIntelligenceSuite:
    """Test suite executing all 36 test requirements for Phase 6A Financial Intelligence."""

    # 1. gross revenue calculation
    def test_gross_revenue_calculation(self):
        """Gross revenue is strictly quantity * unit_price."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 10, "unit_price": 20.0, "discount": 0.0}
        rec = calculate_revenue_margin(row, unit_cost=10.0)
        assert rec.calculated_gross_revenue == 200.0

    # 2. net revenue calculation
    def test_net_revenue_calculation(self):
        """Net revenue is strictly gross_revenue - discount."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 10, "unit_price": 20.0, "discount": 15.0}
        rec = calculate_revenue_margin(row, unit_cost=10.0)
        assert rec.calculated_gross_revenue == 200.0
        assert rec.calculated_net_revenue == 185.0

    # 3. source revenue reconciliation
    def test_source_revenue_reconciliation(self):
        """Source revenue reconciliation calculates variance against calculated net revenue."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 5, "unit_price": 10.0, "discount": 2.0, "revenue": 48.0}
        rec = calculate_revenue_margin(row, unit_cost=5.0)
        assert rec.calculated_net_revenue == 48.0
        assert rec.source_revenue == 48.0
        assert rec.revenue_variance == 0.0
        assert rec.revenue_reconciliation_status == RevenueReconciliationStatus.MATCH

    # 4. matching source revenue
    def test_matching_source_revenue(self):
        """Reconciliation status is MATCH when |variance| <= minor_variance_tolerance."""
        config = FinancialIntelligenceConfig(minor_variance_tolerance=0.05)
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 10.0, "discount": 0.0, "revenue": 10.03}
        rec = calculate_revenue_margin(row, unit_cost=5.0, config=config)
        assert rec.revenue_reconciliation_status == RevenueReconciliationStatus.MATCH

    # 5. revenue variance
    def test_revenue_variance(self):
        """Detects minor and material variances correctly based on thresholds."""
        config = FinancialIntelligenceConfig(minor_variance_tolerance=0.02, material_variance_tolerance=1.00)

        # Minor variance ($0.50 diff)
        row_minor = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 10.0, "discount": 0.0, "revenue": 9.50}
        rec_minor = calculate_revenue_margin(row_minor, unit_cost=5.0, config=config)
        assert rec_minor.revenue_reconciliation_status == RevenueReconciliationStatus.MINOR_VARIANCE
        assert rec_minor.revenue_variance == 0.50

        # Material variance ($2.00 diff)
        row_mat = {"sale_id": "S2", "order_id": "O2", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 10.0, "discount": 0.0, "revenue": 8.00}
        rec_mat = calculate_revenue_margin(row_mat, unit_cost=5.0, config=config)
        assert rec_mat.revenue_reconciliation_status == RevenueReconciliationStatus.MATERIAL_VARIANCE
        assert rec_mat.revenue_variance == 2.00

    # 6. COGS calculation
    def test_cogs_calculation(self):
        """Estimated COGS is quantity * unit_cost and explicitly named estimated_cogs."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 4, "unit_price": 25.0, "discount": 0.0}
        rec = calculate_revenue_margin(row, unit_cost=12.50)
        assert rec.estimated_cogs == 50.0

    # 7. missing unit cost
    def test_missing_unit_cost(self):
        """When unit_cost is missing, estimated_cogs is None and status is PARTIAL."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 4, "unit_price": 25.0, "discount": 0.0}
        rec = calculate_revenue_margin(row, unit_cost=None)
        assert rec.unit_cost is None
        assert rec.estimated_cogs is None
        assert rec.gross_margin is None
        assert rec.gross_margin_percentage is None
        assert rec.financial_status == FinancialDataStatus.PARTIAL
        assert rec.margin_classification == MarginClassification.UNAVAILABLE

    # 8. gross margin
    def test_gross_margin(self):
        """Gross margin is calculated_net_revenue - estimated_cogs."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 2, "unit_price": 50.0, "discount": 10.0}
        rec = calculate_revenue_margin(row, unit_cost=30.0)
        # Net revenue = 100 - 10 = 90. COGS = 2 * 30 = 60. Margin = 90 - 60 = 30.
        assert rec.calculated_net_revenue == 90.0
        assert rec.estimated_cogs == 60.0
        assert rec.gross_margin == 30.0

    # 9. positive margin
    def test_positive_margin(self):
        """Positive margin is classified as POSITIVE_MARGIN."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 100.0, "discount": 10.0}
        rec = calculate_revenue_margin(row, unit_cost=50.0)
        assert rec.gross_margin == 40.0
        assert rec.margin_classification == MarginClassification.POSITIVE_MARGIN

    # 10. zero margin
    def test_zero_margin(self):
        """Zero margin is classified as ZERO_MARGIN."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 50.0, "discount": 0.0}
        rec = calculate_revenue_margin(row, unit_cost=50.0)
        assert rec.gross_margin == 0.0
        assert rec.margin_classification == MarginClassification.ZERO_MARGIN

    # 11. negative margin
    def test_negative_margin(self):
        """Negative margin is classified as NEGATIVE_MARGIN."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 40.0, "discount": 10.0}
        rec = calculate_revenue_margin(row, unit_cost=50.0)
        # Net = 30. COGS = 50. Margin = -20.
        assert rec.gross_margin == -20.0
        assert rec.margin_classification == MarginClassification.NEGATIVE_MARGIN

    # 12. gross margin percentage
    def test_gross_margin_percentage(self):
        """Gross margin percentage is gross_margin / net_revenue."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 2, "unit_price": 100.0, "discount": 0.0}
        rec = calculate_revenue_margin(row, unit_cost=75.0)
        # Net = 200. COGS = 150. Margin = 50. Pct = 50 / 200 = 0.25 (25%)
        assert rec.gross_margin == 50.0
        assert rec.gross_margin_percentage == 0.25

    # 13. zero revenue
    def test_zero_revenue(self):
        """Zero net revenue safely handles gross margin percentage as None (no divide-by-zero)."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 0, "unit_price": 100.0, "discount": 0.0}
        rec = calculate_revenue_margin(row, unit_cost=50.0)
        assert rec.calculated_net_revenue == 0.0
        assert rec.gross_margin == 0.0
        assert rec.gross_margin_percentage is None

    # 14. discount rate
    def test_discount_rate(self):
        """Discount rate is discount / gross_revenue."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 2, "unit_price": 50.0, "discount": 20.0}
        rec = calculate_revenue_margin(row, unit_cost=20.0)
        # Gross = 100. Discount = 20. Rate = 0.20
        assert rec.discount_rate == 0.20

    # 15. missing discount
    def test_missing_discount(self):
        """Missing discount defaults to 0.0 without failing."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 2, "unit_price": 50.0, "discount": None}
        rec = calculate_revenue_margin(row, unit_cost=20.0)
        assert rec.calculated_gross_revenue == 100.0
        assert rec.calculated_net_revenue == 100.0
        assert rec.discount is None

    # 16. SKU aggregation
    def test_sku_aggregation(self, sample_sales, sample_products):
        """Aggregates metrics accurately at the SKU dimension."""
        df = compute_revenue_margin_dataframe(sample_sales, sample_products)
        sku_metrics = aggregate_financial_dimension(df, dimension="SKU")
        assert len(sku_metrics) == 2
        sku_a = next(m for m in sku_metrics if m.segment_key == "SKU_A")
        # SKU_A: 2 lines. Total units = 10 + 2 = 12. Net rev = 240 + 55 = 295.
        assert sku_a.record_count == 2
        assert sku_a.total_units == 12
        assert sku_a.total_net_revenue == 295.0

    # 17. channel aggregation
    def test_channel_aggregation(self, sample_sales, sample_products):
        """Aggregates metrics accurately at the Channel dimension."""
        df = compute_revenue_margin_dataframe(sample_sales, sample_products)
        ch_metrics = aggregate_financial_dimension(df, dimension="CHANNEL")
        assert len(ch_metrics) == 2
        amz = next(m for m in ch_metrics if m.segment_key == "CH_AMZ")
        assert amz.record_count == 2
        assert amz.total_units == 15

    # 18. warehouse aggregation
    def test_warehouse_aggregation(self, sample_sales, sample_products):
        """Aggregates metrics accurately at the Warehouse dimension."""
        df = compute_revenue_margin_dataframe(sample_sales, sample_products)
        wh_metrics = aggregate_financial_dimension(df, dimension="WAREHOUSE")
        assert len(wh_metrics) == 2
        east = next(m for m in wh_metrics if m.segment_key == "WH_EAST")
        assert east.record_count == 2
        assert east.total_units == 15

    # 19. SKU x channel aggregation
    def test_sku_x_channel_aggregation(self, sample_sales, sample_products):
        """Aggregates metrics accurately at the SKU x Channel cross slice."""
        df = compute_revenue_margin_dataframe(sample_sales, sample_products)
        matrix_metrics = aggregate_financial_dimension(df, dimension="SKU_CHANNEL")
        assert len(matrix_metrics) == 3
        keys = {m.segment_key for m in matrix_metrics}
        assert "SKU_A:CH_AMZ" in keys
        assert "SKU_A:CH_SHOPIFY" in keys
        assert "SKU_B:CH_AMZ" in keys

    # 20. monthly aggregation
    def test_monthly_aggregation(self, sample_sales, sample_products):
        """Aggregates metrics at monthly time granularity with distinct order counts."""
        df = compute_revenue_margin_dataframe(sample_sales, sample_products)
        month_metrics = aggregate_time_series(df, grain=TimeGrain.MONTHLY)
        assert len(month_metrics) == 1
        m = month_metrics[0]
        assert m.segment_key == "2026-03"
        assert m.record_count == 3
        assert m.order_count == 2  # ORD_001 and ORD_002

    # 21. distinct order count
    def test_distinct_order_count(self, sample_sales, sample_products):
        """Distinguishes distinct order_id count from order-line record count."""
        df = compute_revenue_margin_dataframe(sample_sales, sample_products)
        overall = aggregate_financial_dimension(df, dimension="OVERALL")[0]
        assert overall.record_count == 3
        assert overall.order_count == 2

    # 22. units aggregation
    def test_units_aggregation(self, sample_sales, sample_products):
        """Sums physical units sold across transactions accurately."""
        df = compute_revenue_margin_dataframe(sample_sales, sample_products)
        overall = aggregate_financial_dimension(df, dimension="OVERALL")[0]
        assert overall.total_units == 17  # 10 + 5 + 2

    # 23. negative-margin detection
    def test_negative_margin_detection(self):
        """Detects negative-margin lines and analyzes components accurately."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_LOSS", "quantity": 10, "unit_price": 20.0, "discount": 100.0, "revenue": 100.0, "currency": "USD"},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-01-01", "sku_id": "SKU_GAIN", "quantity": 5, "unit_price": 50.0, "discount": 0.0, "revenue": 250.0, "currency": "USD"},
        ])
        products = pd.DataFrame([
            {"sku_id": "SKU_LOSS", "unit_cost": 15.0},
            {"sku_id": "SKU_GAIN", "unit_cost": 20.0},
        ])
        df = compute_revenue_margin_dataframe(sales, products)
        # S1: Gross = 200, Disc = 100, Net = 100, COGS = 150, Margin = -50.
        analysis = analyze_negative_margins(df)
        assert analysis["negative_margin_order_lines"] == 1
        assert analysis["negative_margin_revenue"] == 100.0
        assert analysis["negative_margin_cogs"] == 150.0
        assert analysis["negative_margin_loss"] == 50.0
        assert analysis["negative_margin_units"] == 10

    # 24. portfolio summary
    def test_portfolio_summary(self, sample_sales, sample_products):
        """Generates executive portfolio summary matching total aggregates."""
        df = compute_revenue_margin_dataframe(sample_sales, sample_products)
        summary = summarize_financial_portfolio(df)
        assert summary.total_order_lines == 3
        assert summary.total_orders == 2
        assert summary.total_units == 17
        assert summary.total_gross_revenue == 560.0  # 250 + 250 + 60
        assert summary.total_discount == 15.0        # 10 + 0 + 5
        assert summary.total_net_revenue == 545.0    # 560 - 15
        assert summary.total_estimated_cogs == 330.0 # (10*15) + (5*30) + (2*15) = 150 + 150 + 30
        assert summary.total_gross_margin == 215.0   # 545 - 330
        assert round(summary.gross_margin_percentage, 4) == round(215.0 / 545.0, 4)

    # 25. missing values not treated as zero
    def test_missing_values_not_treated_as_zero(self):
        """Unobserved inputs are preserved as None rather than silently converted to 0."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_X", "quantity": 10, "unit_price": 20.0, "discount": None, "revenue": None}
        ])
        rec = calculate_revenue_margin(sales.iloc[0], unit_cost=None)
        assert rec.unit_cost is None
        assert rec.estimated_cogs is None
        assert rec.source_revenue is None
        assert rec.revenue_variance is None
        assert rec.revenue_reconciliation_status == RevenueReconciliationStatus.UNAVAILABLE

    # 26. invalid quantity
    def test_invalid_quantity(self):
        """Negative quantity is flagged as INVALID and excluded from valid financial metrics."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": -5, "unit_price": 20.0}
        rec = calculate_revenue_margin(row, unit_cost=10.0)
        assert rec.financial_status == FinancialDataStatus.INVALID
        assert "Negative quantity" in rec.status_rationale

    # 27. invalid price
    def test_invalid_price(self):
        """Negative unit price is flagged as INVALID."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 5, "unit_price": -20.0}
        rec = calculate_revenue_margin(row, unit_cost=10.0)
        assert rec.financial_status == FinancialDataStatus.INVALID
        assert "Negative unit price" in rec.status_rationale

    # 28. invalid cost
    def test_invalid_cost(self):
        """Negative unit cost is flagged as INVALID."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 5, "unit_price": 20.0}
        rec = calculate_revenue_margin(row, unit_cost=-10.0)
        assert rec.financial_status == FinancialDataStatus.INVALID
        assert "Negative unit cost" in rec.status_rationale

    # 29. duplicate sale_id
    def test_duplicate_sale_id(self):
        """Data quality audit detects duplicate transaction lines."""
        sales = pd.DataFrame([
            {"sale_id": "DUP_1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 10.0, "revenue": 10.0},
            {"sale_id": "DUP_1", "order_id": "O2", "date": "2026-01-02", "sku_id": "SKU_1", "quantity": 1, "unit_price": 10.0, "revenue": 10.0},
        ])
        report = audit_financial_data_quality(sales)
        assert report.duplicate_sale_id_count == 1
        assert not report.is_clean

    # 30. as_of_date leakage protection
    def test_as_of_date_leakage_protection(self, sample_sales, sample_products):
        """Point-in-time filtering strictly excludes sales occurring after as_of_date."""
        service = FinancialIntelligenceService()
        # Cutoff between SALE_001/002 (2026-03-01) and SALE_003 (2026-03-15)
        res = service.analyze(sales=sample_sales, products=sample_products, as_of_date="2026-03-05")
        assert res.portfolio_summary.total_order_lines == 2
        assert res.portfolio_summary.total_orders == 1
        assert res.portfolio_summary.total_units == 15

    # 31. deterministic IDs
    def test_deterministic_ids(self):
        """Record and segment IDs are reproducible SHA-256 digests."""
        id1 = generate_financial_record_id("S1", "SKU1", "O1", "2026-01-01", "USD")
        id2 = generate_financial_record_id("S1", "SKU1", "O1", "2026-01-01", "USD")
        id3 = generate_financial_record_id("S2", "SKU1", "O1", "2026-01-01", "USD")
        assert id1 == id2
        assert id1 != id3
        assert id1.startswith("FIN-REC-")

    # 32. deterministic outputs
    def test_deterministic_outputs(self, sample_sales, sample_products):
        """Running the service multiple times on the same inputs produces identical output."""
        service = FinancialIntelligenceService()
        res1 = service.analyze(sales=sample_sales, products=sample_products)
        res2 = service.analyze(sales=sample_sales, products=sample_products)
        assert res1.portfolio_summary.total_net_revenue == res2.portfolio_summary.total_net_revenue
        assert res1.portfolio_summary.total_gross_margin == res2.portfolio_summary.total_gross_margin

    # 33. empty dataset
    def test_empty_dataset(self):
        """Empty sales input is handled gracefully without exceptions."""
        empty_sales = pd.DataFrame(columns=["sale_id", "order_id", "date", "sku_id", "quantity", "unit_price"])
        service = FinancialIntelligenceService()
        res = service.analyze(sales=empty_sales)
        assert res.portfolio_summary.total_order_lines == 0
        assert res.portfolio_summary.total_orders == 0
        assert res.portfolio_summary.total_net_revenue is None
        assert res.data_quality_report.total_input_records == 0

    # 34. multi-currency handling
    def test_multi_currency_handling(self):
        """Audits and flags mismatched currencies if present in the sales dataset."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 10.0, "currency": "USD"},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-01-02", "sku_id": "SKU_2", "quantity": 1, "unit_price": 10.0, "currency": "EUR"},
        ])
        config = FinancialIntelligenceConfig(default_currency="USD")
        report = audit_financial_data_quality(sales, config=config)
        assert "EUR" in report.mismatched_currencies

    # 35. revenue reconciliation status
    def test_revenue_reconciliation_status(self):
        """Checks UNAVAILABLE when source revenue is missing."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 10.0, "revenue": None}
        rec = calculate_revenue_margin(row, unit_cost=5.0)
        assert rec.revenue_reconciliation_status == RevenueReconciliationStatus.UNAVAILABLE

    # 36. financial completeness
    def test_financial_completeness(self):
        """Reports financial completeness percentage based on complete revenue and cost."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 10.0, "discount": 0.0, "revenue": 10.0, "currency": "USD"},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-01-01", "sku_id": "SKU_2", "quantity": 1, "unit_price": 10.0, "discount": 0.0, "revenue": 10.0, "currency": "USD"},
        ])
        products = pd.DataFrame([
            {"sku_id": "SKU_1", "unit_cost": 5.0}  # SKU_2 cost missing
        ])
        df = compute_revenue_margin_dataframe(sales, products)
        summary = summarize_financial_portfolio(df)
        assert summary.records_with_cost == 1
        assert summary.records_without_cost == 1
        assert summary.financial_completeness_rate == 50.0

    # 37. Neutral analytical rankings test
    def test_neutral_analytical_rankings(self, sample_sales, sample_products):
        """Ensures rankings use neutral labels and order properly."""
        service = FinancialIntelligenceService()
        res = service.analyze(sales=sample_sales, products=sample_products)
        rankings = res.rankings
        assert "top_skus_by_revenue" in rankings
        assert "top_skus_by_margin" in rankings
        assert "bottom_skus_by_margin_pct" in rankings
        top_rev = rankings["top_skus_by_revenue"]
        assert top_rev.ranking_metric == "highest_revenue"
        assert len(top_rev.items) > 0
        assert top_rev.items[0].segment_key == "SKU_A"
