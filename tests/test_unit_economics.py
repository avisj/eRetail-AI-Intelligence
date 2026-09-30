"""Comprehensive Test Suite for Cost Breakdown & True Unit Economics (Phase 6B).

Validates all 30 required test conditions:
1. gross revenue calculation
2. net revenue calculation
3. product cost calculation
4. gross margin
5. gross margin %
6. known variable cost aggregation
7. contribution margin when all costs exist
8. contribution margin when costs are missing
9. missing costs are NOT treated as zero
10. cost completeness
11. cost source metadata
12. SKU aggregation
13. channel aggregation
14. warehouse aggregation
15. SKU x channel
16. SKU x warehouse
17. negative margin
18. zero revenue
19. zero quantity
20. invalid cost
21. missing unit cost
22. currency isolation
23. as_of_date leakage
24. duplicate sale detection
25. data quality reporting
26. deterministic output
27. empty dataset handling
28. partial cost data
29. fully available cost data using injected test costs
30. regression tests proving Phase 6A behavior remains unchanged
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from commerce_ai.financial.schemas import (
    ContributionMarginStatus,
    CostComponent,
    CostComponentDetail,
    CostComponentStatus,
    CostModelConfig,
    CostSourceType,
    FinancialIntelligenceConfig,
    MarginClassification,
    MarginErosionReport,
    TimeGrain,
    UnitEconomicsDimensionMetric,
    UnitEconomicsPortfolioSummary,
    UnitEconomicsRecord,
    UnitEconomicsStatus,
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
    summarize_unit_economics_portfolio,
)
from commerce_ai.financial.service import FinancialIntelligenceService, UnitEconomicsService


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


@pytest.fixture
def fully_costed_sales():
    """Synthetic sales fixture with all 7 cost components manually injected at transaction level."""
    return pd.DataFrame([
        {
            "sale_id": "SALE_FULL_1",
            "order_id": "ORD_FULL_1",
            "date": "2026-03-01",
            "sku_id": "SKU_A",
            "warehouse_id": "WH_EAST",
            "channel_id": "CH_AMZ",
            "quantity": 10,
            "unit_price": 50.0,
            "discount": 20.0,
            "revenue": 480.0,
            "currency": "USD",
            "unit_cost": 20.0,  # Product cost = 10 * 20 = 200
            "shipping_cost": 15.0,
            "payment_processing_cost": 14.22,
            "packaging_cost": 5.0,
            "warehouse_handling_cost": 8.0,
            "return_processing_cost": 2.50,
            "other_variable_cost": 1.50,
        },
        {
            "sale_id": "SALE_FULL_2",
            "order_id": "ORD_FULL_2",
            "date": "2026-03-02",
            "sku_id": "SKU_B",
            "warehouse_id": "WH_WEST",
            "channel_id": "CH_SHOPIFY",
            "quantity": 4,
            "unit_price": 100.0,
            "discount": 0.0,
            "revenue": 400.0,
            "currency": "USD",
            "unit_cost": 40.0,  # Product cost = 4 * 40 = 160
            "shipping_cost": 12.0,
            "payment_processing_cost": 11.90,
            "packaging_cost": 4.0,
            "warehouse_handling_cost": 6.0,
            "return_processing_cost": 0.0,
            "other_variable_cost": 0.0,
        },
    ])


class TestUnitEconomicsSuite:
    """Test suite executing all 30 requirements for Phase 6B Unit Economics."""

    # 1. gross revenue calculation
    def test_gross_revenue_calculation(self):
        """Gross revenue is strictly quantity * unit_price."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 10, "unit_price": 25.0}
        rec = calculate_unit_economics_record(row, unit_cost=15.0)
        assert rec.gross_revenue == 250.0

    # 2. net revenue calculation
    def test_net_revenue_calculation(self):
        """Net revenue is gross_revenue - discount."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 10, "unit_price": 25.0, "discount": 20.0}
        rec = calculate_unit_economics_record(row, unit_cost=15.0)
        assert rec.gross_revenue == 250.0
        assert rec.net_revenue == 230.0

    # 3. product cost calculation
    def test_product_cost_calculation(self):
        """Product cost is quantity * unit_product_cost."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 8, "unit_price": 30.0}
        rec = calculate_unit_economics_record(row, unit_cost=12.50)
        assert rec.product_cost == 100.0  # 8 * 12.50

    # 4. gross margin
    def test_gross_margin(self):
        """Gross margin is net_revenue - product_cost."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 5, "unit_price": 40.0, "discount": 10.0}
        rec = calculate_unit_economics_record(row, unit_cost=20.0)
        # Net = 190. Product cost = 100. Gross margin = 90.
        assert rec.net_revenue == 190.0
        assert rec.product_cost == 100.0
        assert rec.gross_margin == 90.0

    # 5. gross margin %
    def test_gross_margin_pct(self):
        """Gross margin % is gross_margin / net_revenue."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 2, "unit_price": 100.0, "discount": 0.0}
        rec = calculate_unit_economics_record(row, unit_cost=60.0)
        # Net = 200. Cost = 120. Margin = 80. Pct = 80 / 200 = 0.40 (40%)
        assert rec.gross_margin == 80.0
        assert rec.gross_margin_pct == 0.40

    # 6. known variable cost aggregation
    def test_known_variable_cost_aggregation(self):
        """Sums all available variable cost components accurately."""
        row = {
            "sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1",
            "quantity": 2, "unit_price": 50.0, "discount": 0.0,
            "shipping_cost": 10.0, "payment_processing_cost": 3.0, "packaging_cost": 2.0,
        }
        rec = calculate_unit_economics_record(row, unit_cost=20.0)
        assert rec.shipping_cost == 10.0
        assert rec.payment_processing_cost == 3.0
        assert rec.packaging_cost == 2.0
        assert rec.total_known_variable_cost == 15.0

    # 7. contribution margin when all costs exist
    def test_contribution_margin_when_all_costs_exist(self, fully_costed_sales):
        """When all required cost components exist, contribution_margin is CALCULABLE."""
        cfg = CostModelConfig(
            required_cost_components=[
                CostComponent.PRODUCT_COST,
                CostComponent.SHIPPING_COST,
                CostComponent.PAYMENT_PROCESSING_COST,
                CostComponent.PACKAGING_COST,
                CostComponent.WAREHOUSE_HANDLING_COST,
            ]
        )
        row = fully_costed_sales.iloc[0]
        rec = calculate_unit_economics_record(row, unit_cost=float(row["unit_cost"]), config=cfg)
        # Net = 480. Product cost = 200. Known var = 15 + 14.22 + 5 + 8 + 2.5 + 1.5 = 46.22
        # CM = 480 - 200 - 46.22 = 233.78
        assert rec.contribution_margin_status == ContributionMarginStatus.CALCULABLE
        assert rec.contribution_margin == pytest.approx(233.78, 0.01)
        assert rec.contribution_margin_pct == pytest.approx(233.78 / 480.0, 0.001)
        assert rec.economics_status == UnitEconomicsStatus.FULLY_CALCULABLE

    # 8. contribution margin when costs are missing
    def test_contribution_margin_when_costs_are_missing(self, sample_sales, sample_products):
        """When required variable costs are missing, contribution_margin is None and status is INSUFFICIENT_COST_DATA."""
        cfg = CostModelConfig(
            required_cost_components=[
                CostComponent.PRODUCT_COST,
                CostComponent.SHIPPING_COST,
                CostComponent.PAYMENT_PROCESSING_COST,
            ]
        )
        row = sample_sales.iloc[0]
        rec = calculate_unit_economics_record(row, unit_cost=15.0, config=cfg)
        assert rec.contribution_margin is None
        assert rec.contribution_margin_pct is None
        assert rec.contribution_margin_status == ContributionMarginStatus.INSUFFICIENT_COST_DATA
        assert rec.economics_status == UnitEconomicsStatus.PARTIALLY_CALCULABLE
        # But known_contribution_margin IS populated (net_revenue - product_cost - 0 = 90.0)
        assert rec.known_contribution_margin == 90.0

    # 9. missing costs are NOT treated as zero
    def test_missing_costs_are_not_treated_as_zero(self, sample_sales):
        """Missing cost components remain None and are not fabricated as 0.0 in component fields."""
        row = sample_sales.iloc[0]
        rec = calculate_unit_economics_record(row, unit_cost=15.0)
        assert rec.shipping_cost is None
        assert rec.payment_processing_cost is None
        assert rec.packaging_cost is None
        assert rec.warehouse_handling_cost is None
        assert rec.return_processing_cost is None
        assert rec.other_variable_cost is None

    # 10. cost completeness
    def test_cost_completeness(self):
        """Calculates cost completeness percentage correctly against required components."""
        cfg = CostModelConfig(
            required_cost_components=[
                CostComponent.PRODUCT_COST,
                CostComponent.SHIPPING_COST,
                CostComponent.PAYMENT_PROCESSING_COST,
                CostComponent.PACKAGING_COST,
                CostComponent.WAREHOUSE_HANDLING_COST,
            ]
        )
        # 1 of 5 available (PRODUCT_COST only)
        row_1 = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 50.0}
        rec_1 = calculate_unit_economics_record(row_1, unit_cost=20.0, config=cfg)
        assert rec_1.cost_completeness_pct == 20.0

        # 3 of 5 available (PRODUCT_COST, SHIPPING_COST, PACKAGING_COST)
        row_3 = {
            "sale_id": "S2", "order_id": "O2", "date": "2026-01-01", "sku_id": "SKU_1",
            "quantity": 1, "unit_price": 50.0, "shipping_cost": 5.0, "packaging_cost": 1.0,
        }
        rec_3 = calculate_unit_economics_record(row_3, unit_cost=20.0, config=cfg)
        assert rec_3.cost_completeness_pct == 60.0

    # 11. cost source metadata
    def test_cost_source_metadata(self):
        """Identifies SOURCE_DATA, CATALOG_ESTIMATE, and CONFIGURED_ASSUMPTION correctly."""
        cfg = CostModelConfig(
            default_shipping_cost_per_unit=3.0,  # CONFIGURED_ASSUMPTION
        )
        row = {
            "sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1",
            "quantity": 2, "unit_price": 50.0, "packaging_cost": 4.0,  # SOURCE_DATA
        }
        rec = calculate_unit_economics_record(row, unit_cost=15.0, config=cfg)  # CATALOG_ESTIMATE
        details = rec.cost_details

        assert details[CostComponent.PRODUCT_COST.value].source_type == CostSourceType.CATALOG_ESTIMATE
        assert details[CostComponent.SHIPPING_COST.value].source_type == CostSourceType.CONFIGURED_ASSUMPTION
        assert details[CostComponent.PACKAGING_COST.value].source_type == CostSourceType.SOURCE_DATA
        assert details[CostComponent.WAREHOUSE_HANDLING_COST.value].source_type == CostSourceType.UNAVAILABLE

    # 12. SKU aggregation
    def test_sku_aggregation(self, sample_sales, sample_products):
        """Aggregates unit economics across SKU dimension."""
        df = compute_unit_economics_dataframe(sample_sales, sample_products)
        sku_metrics = aggregate_unit_economics_dimension(df, dimension="SKU")
        assert len(sku_metrics) == 2
        sku_a = next(m for m in sku_metrics if m.segment_key == "SKU_A")
        assert sku_a.record_count == 2
        assert sku_a.total_units == 12
        assert sku_a.total_net_revenue == 295.0
        assert sku_a.total_product_cost == 180.0  # (10 * 15) + (2 * 15)
        assert sku_a.total_gross_margin == 115.0  # 295 - 180

    # 13. channel aggregation
    def test_channel_aggregation(self, sample_sales, sample_products):
        """Aggregates unit economics across Channel dimension."""
        df = compute_unit_economics_dataframe(sample_sales, sample_products)
        ch_metrics = aggregate_unit_economics_dimension(df, dimension="CHANNEL")
        assert len(ch_metrics) == 2
        amz = next(m for m in ch_metrics if m.segment_key == "CH_AMZ")
        assert amz.record_count == 2
        assert amz.total_units == 15
        assert amz.total_net_revenue == 490.0

    # 14. warehouse aggregation
    def test_warehouse_aggregation(self, sample_sales, sample_products):
        """Aggregates unit economics across Warehouse dimension."""
        df = compute_unit_economics_dataframe(sample_sales, sample_products)
        wh_metrics = aggregate_unit_economics_dimension(df, dimension="WAREHOUSE")
        assert len(wh_metrics) == 2
        east = next(m for m in wh_metrics if m.segment_key == "WH_EAST")
        assert east.record_count == 2
        assert east.total_units == 15

    # 15. SKU x channel
    def test_sku_x_channel_aggregation(self, sample_sales, sample_products):
        """Aggregates across SKU × Channel cross slice."""
        df = compute_unit_economics_dataframe(sample_sales, sample_products)
        cross_metrics = aggregate_unit_economics_dimension(df, dimension="SKU_CHANNEL")
        assert len(cross_metrics) == 3
        keys = {m.segment_key for m in cross_metrics}
        assert "SKU_A:CH_AMZ" in keys
        assert "SKU_A:CH_SHOPIFY" in keys
        assert "SKU_B:CH_AMZ" in keys

    # 16. SKU x warehouse
    def test_sku_x_warehouse_aggregation(self, sample_sales, sample_products):
        """Aggregates across SKU × Warehouse cross slice."""
        df = compute_unit_economics_dataframe(sample_sales, sample_products)
        cross_metrics = aggregate_unit_economics_dimension(df, dimension="SKU_WAREHOUSE")
        assert len(cross_metrics) == 3
        keys = {m.segment_key for m in cross_metrics}
        assert "SKU_A:WH_EAST" in keys
        assert "SKU_A:WH_WEST" in keys
        assert "SKU_B:WH_EAST" in keys

    # 17. negative margin
    def test_negative_margin(self):
        """Identifies negative margin and computes known contribution margin accordingly."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_LOSS", "quantity": 10, "unit_price": 20.0, "discount": 100.0}
        rec = calculate_unit_economics_record(row, unit_cost=15.0)
        # Net = 100. Cost = 150. Margin = -50.
        assert rec.gross_margin == -50.0
        assert rec.known_contribution_margin == -50.0

    # 18. zero revenue
    def test_zero_revenue(self):
        """Zero net revenue safely handles margin percentages as None without ZeroDivisionError."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_ZERO", "quantity": 0, "unit_price": 50.0}
        rec = calculate_unit_economics_record(row, unit_cost=25.0)
        assert rec.net_revenue == 0.0
        assert rec.gross_margin == 0.0
        assert rec.gross_margin_pct is None
        assert rec.known_contribution_margin_pct is None

    # 19. zero quantity
    def test_zero_quantity(self):
        """Zero quantity results in zero revenue and zero product cost."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 0, "unit_price": 100.0}
        rec = calculate_unit_economics_record(row, unit_cost=50.0)
        assert rec.gross_revenue == 0.0
        assert rec.product_cost == 0.0
        assert rec.gross_margin == 0.0

    # 20. invalid cost
    def test_invalid_cost(self):
        """Negative unit cost is flagged as INVALID_DATA."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 5, "unit_price": 20.0}
        rec = calculate_unit_economics_record(row, unit_cost=-10.0)
        assert rec.economics_status == UnitEconomicsStatus.INVALID_DATA
        assert "Negative unit cost" in rec.status_rationale

    # 21. missing unit cost
    def test_missing_unit_cost(self):
        """Missing unit cost sets product_cost to None and status to INSUFFICIENT_COST_DATA."""
        row = {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 5, "unit_price": 20.0}
        rec = calculate_unit_economics_record(row, unit_cost=None)
        assert rec.unit_product_cost is None
        assert rec.product_cost is None
        assert rec.gross_margin is None
        assert rec.economics_status == UnitEconomicsStatus.INSUFFICIENT_COST_DATA

    # 22. currency isolation
    def test_currency_isolation(self):
        """Preserves currency and checks currency consistency."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 10.0, "currency": "USD"},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-01-01", "sku_id": "SKU_2", "quantity": 1, "unit_price": 10.0, "currency": "EUR"},
        ])
        svc = UnitEconomicsService()
        report = svc.run_financial_quality_checks(sales)
        assert "EUR" in report.mismatched_currencies

    # 23. as_of_date leakage
    def test_as_of_date_leakage(self, sample_sales, sample_products):
        """Point-in-time filtering strictly excludes sales occurring after as_of_date."""
        svc = UnitEconomicsService()
        res = svc.calculate_unit_economics(sales=sample_sales, products=sample_products, as_of_date="2026-03-05")
        assert res.portfolio_summary.total_order_lines == 2
        assert res.portfolio_summary.total_orders == 1
        assert res.portfolio_summary.total_units == 15

    # 24. duplicate sale detection
    def test_duplicate_sale_detection(self):
        """Data quality report detects duplicate sale_id entries."""
        sales = pd.DataFrame([
            {"sale_id": "DUP_1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1", "quantity": 1, "unit_price": 10.0},
            {"sale_id": "DUP_1", "order_id": "O2", "date": "2026-01-02", "sku_id": "SKU_1", "quantity": 1, "unit_price": 10.0},
        ])
        svc = UnitEconomicsService()
        report = svc.run_financial_quality_checks(sales)
        assert report.duplicate_sale_id_count == 1
        assert not report.is_clean

    # 25. data quality reporting
    def test_data_quality_reporting(self):
        """Data quality report captures negative values and missing identifiers."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "", "quantity": -2, "unit_price": -5.0},
        ])
        svc = UnitEconomicsService()
        report = svc.run_financial_quality_checks(sales)
        assert report.missing_sku_count == 1
        assert report.negative_quantity_count == 1
        assert report.negative_unit_price_count == 1

    # 26. deterministic output
    def test_deterministic_output(self, sample_sales, sample_products):
        """Running the service multiple times produces identical numeric and ID output."""
        svc = UnitEconomicsService()
        res1 = svc.calculate_unit_economics(sales=sample_sales, products=sample_products)
        res2 = svc.calculate_unit_economics(sales=sample_sales, products=sample_products)
        assert res1.portfolio_summary.total_net_revenue == res2.portfolio_summary.total_net_revenue
        assert res1.portfolio_summary.total_gross_margin == res2.portfolio_summary.total_gross_margin
        assert res1.portfolio_summary.cost_completeness_pct == res2.portfolio_summary.cost_completeness_pct

    # 27. empty dataset handling
    def test_empty_dataset_handling(self):
        """Empty sales input is handled cleanly without exceptions."""
        empty_sales = pd.DataFrame(columns=["sale_id", "order_id", "date", "sku_id", "quantity", "unit_price"])
        svc = UnitEconomicsService()
        res = svc.calculate_unit_economics(sales=empty_sales)
        assert res.portfolio_summary.total_order_lines == 0
        assert res.portfolio_summary.total_net_revenue is None
        assert res.portfolio_summary.cost_completeness_pct == 0.0

    # 28. partial cost data
    def test_partial_cost_data(self, sample_sales, sample_products):
        """When only product cost is available (as in canonical dataset), status is PARTIALLY_CALCULABLE."""
        svc = UnitEconomicsService()
        res = svc.calculate_unit_economics(sales=sample_sales, products=sample_products)
        summary = res.portfolio_summary
        assert summary.economics_status == UnitEconomicsStatus.PARTIALLY_CALCULABLE
        assert summary.contribution_margin_status == ContributionMarginStatus.INSUFFICIENT_COST_DATA
        assert summary.total_contribution_margin is None
        # But known contribution margin IS calculated
        assert summary.total_known_contribution_margin == 215.0

    # 29. fully available cost data using injected test costs
    def test_fully_available_cost_data_using_injected_test_costs(self, fully_costed_sales):
        """Synthetic fixture with all costs injected achieves FULLY_CALCULABLE and 100% completeness."""
        cfg = CostModelConfig(
            required_cost_components=[
                CostComponent.PRODUCT_COST,
                CostComponent.SHIPPING_COST,
                CostComponent.PAYMENT_PROCESSING_COST,
                CostComponent.PACKAGING_COST,
                CostComponent.WAREHOUSE_HANDLING_COST,
            ]
        )
        svc = UnitEconomicsService(config=cfg)
        res = svc.calculate_unit_economics(sales=fully_costed_sales)
        summary = res.portfolio_summary
        assert summary.economics_status == UnitEconomicsStatus.FULLY_CALCULABLE
        assert summary.contribution_margin_status == ContributionMarginStatus.CALCULABLE
        assert summary.cost_completeness_pct == 100.0
        assert summary.total_contribution_margin is not None
        assert summary.total_contribution_margin > 0

    # 30. regression tests proving Phase 6A behavior remains unchanged
    def test_regression_phase_6a_behavior_unchanged(self, sample_sales, sample_products):
        """Verifies Phase 6A FinancialIntelligenceService produces exact same revenue and margin results."""
        fin_svc = FinancialIntelligenceService()
        res_6a = fin_svc.analyze(sales=sample_sales, products=sample_products)

        ue_svc = UnitEconomicsService()
        res_6b = ue_svc.calculate_unit_economics(sales=sample_sales, products=sample_products)

        assert res_6a.portfolio_summary.total_gross_revenue == res_6b.portfolio_summary.total_gross_revenue
        assert res_6a.portfolio_summary.total_discount == res_6b.portfolio_summary.total_discount
        assert res_6a.portfolio_summary.total_net_revenue == res_6b.portfolio_summary.total_net_revenue
        assert res_6a.portfolio_summary.total_estimated_cogs == res_6b.portfolio_summary.total_product_cost
        assert res_6a.portfolio_summary.total_gross_margin == res_6b.portfolio_summary.total_gross_margin
        assert res_6a.portfolio_summary.gross_margin_percentage == res_6b.portfolio_summary.gross_margin_pct

    # 31. Margin erosion analysis
    def test_margin_erosion_analysis(self):
        """Verifies descriptive margin erosion report flags low margin, negative margin, and mismatches."""
        m1 = UnitEconomicsDimensionMetric(
            dimension="SKU", segment_key="SKU_LOSS", total_net_revenue=100.0, total_gross_margin=-50.0,
            revenue_contribution=0.10, margin_contribution=-0.20, gross_margin_pct=-0.50,
        )
        m2 = UnitEconomicsDimensionMetric(
            dimension="SKU", segment_key="SKU_LOW", total_net_revenue=200.0, total_gross_margin=20.0,
            revenue_contribution=0.20, margin_contribution=0.08, gross_margin_pct=0.10,
        )
        m3 = UnitEconomicsDimensionMetric(
            dimension="SKU", segment_key="SKU_MISMATCH", total_net_revenue=500.0, total_gross_margin=100.0,
            revenue_contribution=0.50, margin_contribution=0.40, gross_margin_pct=0.20,
        )
        m4 = UnitEconomicsDimensionMetric(
            dimension="CHANNEL", segment_key="CH_LOW", total_net_revenue=300.0, total_gross_margin=90.0,
            revenue_contribution=0.30, margin_contribution=0.36, gross_margin_pct=0.30,
        )
        report = analyze_margin_erosion(
            sku_metrics=[m1, m2, m3],
            channel_metrics=[m4],
            warehouse_metrics=[],
            portfolio_margin_pct=0.50,
            low_margin_threshold=0.20,
            mismatch_threshold=0.05,
        )
        assert len(report.negative_margin_skus) == 1
        assert report.negative_margin_skus[0].segment_key == "SKU_LOSS"
        assert len(report.low_margin_skus) == 1
        assert report.low_margin_skus[0].segment_key == "SKU_LOW"
        assert len(report.margin_contribution_mismatches) == 1
        assert report.margin_contribution_mismatches[0].segment_key == "SKU_MISMATCH"
        assert len(report.low_margin_channels) == 1
        assert report.low_margin_channels[0].segment_key == "CH_LOW"

    # 32. Configurable channel and warehouse defaults
    def test_configurable_channel_and_warehouse_defaults(self):
        """Verifies parametric assumptions in CostModelConfig apply properly to transactions."""
        cfg = CostModelConfig(
            channel_shipping_costs={"CH_AMZ": 4.50},
            channel_payment_processing_rates={"CH_AMZ": 0.03},
            warehouse_handling_cost_per_unit={"WH_EAST": 1.25},
        )
        row = {
            "sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_1",
            "warehouse_id": "WH_EAST", "channel_id": "CH_AMZ", "quantity": 10, "unit_price": 20.0,
        }
        rec = calculate_unit_economics_record(row, unit_cost=10.0, config=cfg)
        # Net = 200.
        # Shipping = 10 * 4.50 = 45.0
        # Payment = 0.03 * 200 = 6.0
        # Handling = 10 * 1.25 = 12.50
        assert rec.shipping_cost == 45.0
        assert rec.payment_processing_cost == 6.0
        assert rec.warehouse_handling_cost == 12.50
        assert rec.total_known_variable_cost == 63.50
        assert rec.known_contribution_margin == 200.0 - 100.0 - 63.50  # 36.50
