"""Comprehensive unit tests for all domain Query Tools (Phase 7A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from commerce_ai.forecasting.base import ForecastMetadata, ForecastOutput, ForecastRecord
from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.schemas import CalculationStatus
from commerce_ai.query_layer import (
    sales,
    financial,
    inventory,
    demand,
    forecasting,
    returns,
    operations,
    impact,
    recommendations,
    decisions,
)
from datetime import date, timedelta
from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendation,
    BusinessRecommendationResult,
    ConfidenceProvenance,
    RecommendationConflict,
    RecommendationPriority,
    RecommendationStatus,
    RecommendationType,
)
from commerce_ai.decision_intelligence.schemas import (
    DecisionBusinessContext,
    DecisionEntityContext,
    DecisionIntelligenceResult,
    DecisionOption,
    DecisionOptionType,
    DecisionPackage,
    DecisionRiskFlag,
    DecisionStatus,
    DecisionTradeOff,
    RiskSeverity,
)
from commerce_ai.business_impact.schemas import (
    BusinessImpactPortfolioSummary,
    BusinessImpactResult,
)


@pytest.fixture
def mock_sales_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2024-01-10", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_DIR", "quantity": 5, "unit_price": 100.0, "discount": 10.0, "revenue": 490.0, "currency": "USD"},
        {"sale_id": "S2", "order_id": "O1", "date": "2024-01-10", "sku_id": "SKU_02", "warehouse_id": "WH_01", "channel_id": "CH_DIR", "quantity": 2, "unit_price": 50.0, "discount": 0.0, "revenue": 100.0, "currency": "USD"},
        {"sale_id": "S3", "order_id": "O2", "date": "2024-01-15", "sku_id": "SKU_01", "warehouse_id": "WH_02", "channel_id": "CH_AMZ", "quantity": 3, "unit_price": 100.0, "discount": 0.0, "revenue": 300.0, "currency": "USD"},
    ])


@pytest.fixture
def mock_products_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"sku_id": "SKU_01", "product_name": "Premium Tech", "category_id": "Electronics", "brand": "HyperTech", "currency": "USD", "unit_cost": 60.0, "selling_price": 100.0, "velocity_tier": "FAST"},
        {"sku_id": "SKU_02", "product_name": "Home Comfort", "category_id": "Home", "brand": "CozyLiving", "currency": "USD", "unit_cost": 25.0, "selling_price": 50.0, "velocity_tier": "SLOW"},
    ])


@pytest.fixture
def mock_inventory_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"snapshot_date": "2024-01-15", "sku_id": "SKU_01", "warehouse_id": "WH_01", "available_qty": 50, "reserved_qty": 5, "in_transit_qty": 20, "damaged_qty": 1},
        {"snapshot_date": "2024-01-15", "sku_id": "SKU_01", "warehouse_id": "WH_02", "available_qty": 0, "reserved_qty": 0, "in_transit_qty": 0, "damaged_qty": 0},
        {"snapshot_date": "2024-01-15", "sku_id": "SKU_02", "warehouse_id": "WH_01", "available_qty": 100, "reserved_qty": 0, "in_transit_qty": 0, "damaged_qty": 0},
    ])


@pytest.fixture
def mock_returns_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2024-01-12", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_DIR", "quantity": 1, "reason": "Defective Item"},
        {"return_id": "R2", "order_id": "O2", "return_date": "2024-01-18", "sku_id": "SKU_01", "warehouse_id": "WH_02", "channel_id": "CH_AMZ", "quantity": 1, "reason": "Wrong Size"},
    ])


class TestSalesTools:
    def test_get_sales_summary_kpi_contract(self, mock_sales_df, mock_products_df):
        ctx = QueryContext(currency="USD")
        res = sales.get_sales_summary(ctx, mock_sales_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        assert len(res.metrics) >= 5

        metric_map = {m.metric_name: m.value for m in res.metrics}
        # Gross revenue: (5*100) + (2*50) + (3*100) = 500 + 100 + 300 = 900
        assert metric_map["gross_revenue"] == 900.0
        assert metric_map["total_discount"] == 10.0
        assert metric_map["net_revenue"] == 890.0
        assert metric_map["units_sold"] == 10
        assert metric_map["order_count"] == 2
        assert metric_map["average_order_value"] == 445.0

    def test_get_sales_summary_point_in_time_filtering(self, mock_sales_df, mock_products_df):
        # as_of_date is 2024-01-12 -> S3 (Jan 15) must be excluded
        ctx = QueryContext(as_of_date="2024-01-12", currency="USD")
        res = sales.get_sales_summary(ctx, mock_sales_df, mock_products_df)
        metric_map = {m.metric_name: m.value for m in res.metrics}
        # Only S1 & S2: 500 + 100 = 600 gross, 590 net
        assert metric_map["gross_revenue"] == 600.0
        assert metric_map["net_revenue"] == 590.0
        assert metric_map["units_sold"] == 7

    def test_get_sales_trend_daily(self, mock_sales_df):
        ctx = QueryContext(time_grain="daily")
        res = sales.get_sales_trend(ctx, mock_sales_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.time_series is not None
        assert len(res.time_series.points) == 2  # 2024-01-10 and 2024-01-15
        assert res.time_series.points[0].date == "2024-01-10"
        assert res.time_series.points[0].value == 590.0  # 490 + 100

    def test_get_sales_by_sku_table(self, mock_sales_df, mock_products_df):
        ctx = QueryContext()
        res = sales.get_sales_by_sku(ctx, mock_sales_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.table is not None
        assert res.table.total_rows == 2
        sku_ids = [r["sku_id"] for r in res.table.rows]
        assert sku_ids == ["SKU_01", "SKU_02"]

    def test_get_sales_by_channel_breakdown(self, mock_sales_df):
        ctx = QueryContext()
        res = sales.get_sales_by_channel(ctx, mock_sales_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.breakdown is not None
        assert len(res.breakdown.items) == 2
        pct_sum = sum(i.percentage_of_total for i in res.breakdown.items)
        assert round(pct_sum, 1) == 100.0

    def test_get_sales_by_warehouse_breakdown(self, mock_sales_df):
        ctx = QueryContext()
        res = sales.get_sales_by_warehouse(ctx, mock_sales_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.breakdown is not None
        assert len(res.breakdown.items) == 2

    def test_get_sales_by_category_breakdown(self, mock_sales_df, mock_products_df):
        ctx = QueryContext()
        res = sales.get_sales_by_category(ctx, mock_sales_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.breakdown is not None
        assert len(res.breakdown.items) == 2
        cats = {i.dimension_value for i in res.breakdown.items}
        assert cats == {"Electronics", "Home"}

    def test_get_sales_by_brand_breakdown(self, mock_sales_df, mock_products_df):
        ctx = QueryContext()
        res = sales.get_sales_by_brand(ctx, mock_sales_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.breakdown is not None
        assert len(res.breakdown.items) == 2
        brands = {i.dimension_value for i in res.breakdown.items}
        assert brands == {"HyperTech", "CozyLiving"}


class TestFinancialTools:
    def test_get_revenue_summary(self, mock_sales_df, mock_products_df):
        ctx = QueryContext(currency="USD")
        res = financial.get_revenue_summary(ctx, mock_sales_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        assert len(res.metrics) >= 4
        m = {k.metric_name: k.value for k in res.metrics}
        assert m["gross_revenue"] == 900.0
        assert m["net_revenue"] == 890.0

    def test_get_margin_summary(self, mock_sales_df, mock_products_df):
        ctx = QueryContext(currency="USD")
        res = financial.get_margin_summary(ctx, mock_sales_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        m = {k.metric_name: k.value for k in res.metrics}
        # Product costs: (5*60) + (2*25) + (3*60) = 300 + 50 + 180 = 530
        assert m["product_cost"] == 530.0
        # Gross margin: 890 - 530 = 360
        assert m["gross_margin"] == 360.0
        assert round(m["gross_margin_pct"], 4) == round(360.0 / 890.0, 4)

    def test_get_unit_economics(self, mock_sales_df, mock_products_df):
        ctx = QueryContext(currency="USD")
        res = financial.get_unit_economics(ctx, mock_sales_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        m = {k.metric_name: k.value for k in res.metrics}
        assert "shipping_cost" in m
        assert "payment_processing_fee" in m
        assert "known_contribution_margin" in m

    def test_get_margin_drivers_waterfall(self, mock_sales_df, mock_products_df):
        ctx = QueryContext(currency="USD")
        res = financial.get_margin_drivers(ctx, mock_sales_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.breakdown is not None
        assert len(res.breakdown.items) >= 4  # gross rev, disc, cogs, var costs

    def test_get_operational_economics(self, mock_sales_df, mock_products_df):
        ctx = QueryContext(currency="USD")
        res = financial.get_operational_economics(ctx, mock_sales_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        m = {k.metric_name: k.value for k in res.metrics}
        assert "total_operational_cost" in m
        assert "cost_completeness_score" in m

    def test_get_profitability_attribution(self, mock_sales_df, mock_products_df):
        ctx = QueryContext(currency="USD")
        res = financial.get_profitability_attribution(ctx, mock_sales_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.table is not None
        assert len(res.table.rows) == 2


class TestInventoryTools:
    def test_get_inventory_summary(self, mock_inventory_df, mock_products_df):
        ctx = QueryContext(currency="USD")
        res = inventory.get_inventory_summary(ctx, mock_inventory_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        m = {k.metric_name: k.value for k in res.metrics}
        # On hand: 50 + 0 + 100 = 150
        assert m["on_hand_units"] == 150
        assert m["reserved_units"] == 5
        assert m["on_order_units"] == 20
        # Valuation: (50*60) + (0*60) + (100*25) = 3000 + 2500 = 5500
        assert m["inventory_valuation"] == 5500.0

    def test_get_inventory_position(self, mock_inventory_df, mock_products_df):
        ctx = QueryContext(currency="USD")
        res = inventory.get_inventory_position(ctx, mock_inventory_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.table is not None
        assert res.table.total_rows == 3

    def test_get_stockout_risk(self, mock_inventory_df, mock_products_df, mock_sales_df):
        ctx = QueryContext()
        res = inventory.get_stockout_risk(ctx, mock_inventory_df, mock_products_df, mock_sales_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.table is not None
        # SKU_01 at WH_02 has 0 stock -> CRITICAL_STOCKOUT
        stockouts = [r for r in res.table.rows if r["sku_id"] == "SKU_01" and r["warehouse_id"] == "WH_02"]
        assert len(stockouts) == 1
        assert stockouts[0]["stockout_risk_tier"] == "CRITICAL_STOCKOUT"

    def test_get_slow_moving_inventory(self, mock_inventory_df, mock_products_df):
        ctx = QueryContext()
        res = inventory.get_slow_moving_inventory(ctx, mock_inventory_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.table is not None
        # SKU_02 is SLOW tier with 100 units
        assert any(r["sku_id"] == "SKU_02" for r in res.table.rows)

    def test_get_high_value_inventory(self, mock_inventory_df, mock_products_df):
        ctx = QueryContext()
        res = inventory.get_high_value_inventory(ctx, mock_inventory_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.table is not None
        assert res.table.total_rows >= 2

    def test_get_inventory_risk_breakdown(self, mock_inventory_df, mock_products_df):
        ctx = QueryContext()
        res = inventory.get_inventory_risk(ctx, mock_inventory_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.breakdown is not None
        tiers = {i.dimension_value for i in res.breakdown.items}
        assert "CRITICAL_STOCKOUT" in tiers


class TestDemandTools:
    def test_get_demand_summary(self, mock_sales_df):
        ctx = QueryContext()
        res = demand.get_demand_summary(ctx, mock_sales_df)
        assert res.status == CalculationStatus.SUCCESS
        m = {k.metric_name: k.value for k in res.metrics}
        assert m["total_demand_units"] == 10
        assert m["active_sku_count"] == 2

    def test_get_demand_trend(self, mock_sales_df):
        ctx = QueryContext(time_grain="daily")
        res = demand.get_demand_trend(ctx, mock_sales_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.time_series is not None
        assert len(res.time_series.points) == 2

    def test_get_demand_profile(self, mock_sales_df):
        ctx = QueryContext()
        res = demand.get_demand_profile(ctx, mock_sales_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.table is not None
        assert res.table.total_rows == 2

    def test_get_abc_xyz_distribution(self, mock_sales_df):
        ctx = QueryContext()
        res = demand.get_abc_xyz_distribution(ctx, mock_sales_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.breakdown is not None
        assert len(res.breakdown.items) > 0


class TestForecastingTools:
    def test_get_forecast_unavailable_when_missing(self):
        ctx = QueryContext()
        res = forecasting.get_forecast(ctx)
        assert res.status == CalculationStatus.UNAVAILABLE
        assert "unavailable" in res.error_message.lower()

    def test_get_forecast_with_forecast_output(self):
        ctx = QueryContext()
        out = ForecastOutput(
            sku_id="SKU_01",
            warehouse_id="WH_01",
            forecast_dates=["2024-02-01", "2024-02-02"],
            forecast_units=np.array([15.0, 16.0]),
            model_name="MovingAverage",
        )
        res = forecasting.get_forecast(ctx, forecast_output=out)
        assert res.status == CalculationStatus.SUCCESS
        assert res.time_series is not None
        assert len(res.time_series.points) == 2
        assert res.time_series.points[0].value == 15.0

    def test_get_forecast_accuracy(self):
        ctx = QueryContext()
        actuals = [10.0, 20.0, 30.0]
        preds = [12.0, 18.0, 31.0]
        res = forecasting.get_forecast_accuracy(ctx, actual_series=actuals, forecast_series=preds)
        assert res.status == CalculationStatus.SUCCESS
        m = {k.metric_name: k.value for k in res.metrics}
        assert "wape" in m
        assert "mae" in m

    def test_get_forecast_bias(self):
        ctx = QueryContext()
        actuals = [10.0, 20.0, 30.0]
        preds = [15.0, 25.0, 35.0]  # consistently higher
        res = forecasting.get_forecast_bias(ctx, actual_series=actuals, forecast_series=preds)
        assert res.status == CalculationStatus.SUCCESS
        assert res.metrics[0].status == "OVER_FORECASTING"


class TestReturnsTools:
    def test_get_return_summary(self, mock_returns_df, mock_sales_df, mock_products_df):
        ctx = QueryContext(currency="USD")
        res = returns.get_return_summary(ctx, mock_returns_df, mock_sales_df, mock_products_df)
        assert res.status == CalculationStatus.SUCCESS
        m = {k.metric_name: k.value for k in res.metrics}
        assert m["total_return_events"] == 2
        assert m["total_returned_units"] == 2

    def test_get_return_trend(self, mock_returns_df):
        ctx = QueryContext(time_grain="daily")
        res = returns.get_return_trend(ctx, mock_returns_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.time_series is not None
        assert len(res.time_series.points) == 2

    def test_get_return_anomalies(self, mock_returns_df, mock_sales_df):
        ctx = QueryContext()
        res = returns.get_return_anomalies(ctx, mock_returns_df, mock_sales_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.table is not None

    def test_get_return_reason_breakdown(self, mock_returns_df):
        ctx = QueryContext()
        res = returns.get_return_reason_breakdown(ctx, mock_returns_df)
        assert res.status == CalculationStatus.SUCCESS
        assert res.breakdown is not None
        assert len(res.breakdown.items) == 2
        reasons = {i.dimension_value for i in res.breakdown.items}
        assert reasons == {"Defective Item", "Wrong Size"}


class TestImpactAndGovernanceTools:
    def test_get_business_impact_summary(self):
        ctx = QueryContext(currency="USD")
        summary = BusinessImpactPortfolioSummary(
            impact_record_count=10,
            gross_signal_exposure=50000.0,
            deduplicated_exposure=40000.0,
            deduplicated_physical_capital_exposure=15000.0,
            impact_category_counts={"INVENTORY_EXCESS": 5, "STOCKOUT_REVENUE_LOSS": 5},
            impact_type_counts={"CAPITAL_TIED_UP": 5, "AT_RISK_REVENUE": 5},
            exposure_totals_by_category={"INVENTORY_EXCESS": 30000.0, "STOCKOUT_REVENUE_LOSS": 20000.0},
            exposure_totals_by_type={"CAPITAL_TIED_UP": 30000.0, "AT_RISK_REVENUE": 20000.0},
            currency="USD",
        )
        res_mock = BusinessImpactResult(
            impact_records=[],
            portfolio_summary=summary,
            as_of_date="2026-06-30",
        )
        res = impact.get_business_impact_summary(ctx, impact_result=res_mock)
        assert res.status == CalculationStatus.SUCCESS
        m = {k.metric_name: k.value for k in res.metrics}
        assert m["gross_signal_exposure"] == 50000.0
        assert m["deduplicated_exposure"] == 40000.0
        assert m["deduplicated_physical_capital_exposure"] == 15000.0

    def test_get_recommendation_summary_governance(self):
        ctx = QueryContext()
        rec = BusinessRecommendation(
            recommendation_id="REC-0010000000000001",
            recommendation_type=RecommendationType.STOCKOUT_REVIEW,
            priority=RecommendationPriority.CRITICAL,
            sku_id="SKU_01",
            warehouse_id="WH_01",
            title="Stockout Review",
            summary="Stockout risk detected",
            reason="Critical runout detected",
            action="Review inventory and safety stock",
            action_category="REVIEW",
            source_engine="inventory_solver",
            rule_id="RULE-01",
            confidence=ConfidenceProvenance.DETERMINISTIC_DERIVED,
            confidence_reason="Inventory math",
            created_as_of=date(2026, 6, 30),
            valid_until=date(2026, 7, 7),
            approval_required=True,
            execution_allowed=False,
            requires_human_review=True,
            currency="USD",
        )
        rec_res = BusinessRecommendationResult(
            recommendations=[rec],
            conflicts=[],
            as_of_date=date(2026, 6, 30),
            total_recommendations=1,
            recommendations_by_type={"STOCKOUT_REVIEW": 1},
            recommendations_by_priority={"CRITICAL": 1},
        )
        res = recommendations.get_recommendation_summary(ctx, recommendation_result=rec_res)
        assert res.status == CalculationStatus.SUCCESS
        m = {k.metric_name: k.value for k in res.metrics}
        assert m["total_recommendations"] == 1
        assert m["human_review_count"] == 1
        assert m["critical_priority_count"] == 1

        res_list = recommendations.get_recommendations(ctx, recommendation_result=rec_res)
        assert res_list.status == CalculationStatus.SUCCESS
        assert res_list.table is not None
        assert res_list.table.total_rows == 1

        res_id = recommendations.get_recommendation_by_id(ctx, "REC-0010000000000001", recommendation_result=rec_res)
        assert res_id.status == CalculationStatus.SUCCESS
        assert len(res_id.insights) == 1
        assert res_id.insights[0].insight_id == "REC-0010000000000001"

    def test_get_decision_summary_governance_no_winner(self):
        ctx = QueryContext()
        pkg = DecisionPackage(
            decision_id="DEC-0010000000000001",
            recommendation_id="REC-0010000000000001",
            recommendation_type=RecommendationType.STOCKOUT_REVIEW,
            decision_title="Stockout Decision",
            decision_summary="Evaluate stockout response",
            entity_context=DecisionEntityContext(
                sku_id="SKU_01",
                warehouse_id="WH_01",
                entity_key="SKU_01:WH_01",
            ),
            business_context=DecisionBusinessContext(
                priority=RecommendationPriority.CRITICAL,
                source_engine="test",
                rule_id="RULE_001",
                confidence=ConfidenceProvenance.DETERMINISTIC_DERIVED,
                confidence_reason="test",
                currency="USD",
            ),
            decision_options=[
                DecisionOption(
                    option_id="OPT-001000000001",
                    title="Observe",
                    option_type=DecisionOptionType.NO_CHANGE,
                    description="Observe",
                    expected_effect="Observe stock without immediate intervention",
                    is_baseline=True,
                )
            ],
            trade_offs=[],
            required_information=[],
            risk_flags=[],
            approval_required=True,
            execution_allowed=False,
            selected_option=None,
            requires_human_review=True,
            created_as_of=date(2026, 6, 30),
            valid_until=date(2026, 7, 7),
        )
        dec_res = DecisionIntelligenceResult(
            decision_packages=[pkg],
            as_of_date=date(2026, 6, 30),
            total_decisions=1,
            decisions_by_status={"PENDING_REVIEW": 1},
            decisions_by_type={"STOCKOUT_REVIEW": 1},
            decisions_by_risk_severity={"CRITICAL": 1},
            human_review_count=1,
            conflict_count=0,
            insufficient_information_count=0,
        )
        res = decisions.get_decision_summary(ctx, decision_result=dec_res)
        assert res.status == CalculationStatus.SUCCESS
        m = {k.metric_name: k.value for k in res.metrics}
        assert m["total_decision_packages"] == 1
        assert m["pending_review_count"] == 1
        assert m["unselected_winner_count"] == 1  # 100% selected_option=None

        res_list = decisions.get_decision_packages(ctx, decision_result=dec_res)
        assert res_list.status == CalculationStatus.SUCCESS
        assert res_list.table is not None
        assert res_list.table.total_rows == 1

        res_id = decisions.get_decision_package_by_id(ctx, "DEC-0010000000000001", decision_result=dec_res)
        assert res_id.status == CalculationStatus.SUCCESS
        assert len(res_id.insights) == 1
        assert res_id.insights[0].insight_id == "DEC-0010000000000001"
