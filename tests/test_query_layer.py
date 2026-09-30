"""Integration and Governance tests for QueryLayerService and ToolRegistry (Phase 7A)."""

from __future__ import annotations

import pandas as pd
import pytest

from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.registry import ToolMetadata, ToolRegistry, default_registry
from commerce_ai.query_layer.schemas import (
    CalculationStatus,
    EvidenceReference,
    MetricResult,
    QueryDomain,
    QueryResponse,
    TableResult,
    TimeSeriesResult,
)
from commerce_ai.query_layer.service import QueryLayerService


@pytest.fixture
def mock_service() -> QueryLayerService:
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2024-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_DIR", "quantity": 10, "unit_price": 50.0, "currency": "USD"},
        {"sale_id": "S2", "order_id": "O2", "date": "2024-01-02", "sku_id": "SKU_02", "warehouse_id": "WH_02", "channel_id": "CH_AMZ", "quantity": 5, "unit_price": 40.0, "currency": "USD"},
    ])
    products = pd.DataFrame([
        {"sku_id": "SKU_01", "product_name": "Item 1", "category_id": "Electronics", "brand": "HyperTech", "currency": "USD", "unit_cost": 25.0},
        {"sku_id": "SKU_02", "product_name": "Item 2", "category_id": "Home", "brand": "CozyLiving", "currency": "USD", "unit_cost": 20.0},
    ])
    inventory = pd.DataFrame([
        {"snapshot_date": "2024-01-02", "sku_id": "SKU_01", "warehouse_id": "WH_01", "available_qty": 100, "reserved_qty": 10, "in_transit_qty": 20, "damaged_qty": 0},
        {"snapshot_date": "2024-01-02", "sku_id": "SKU_02", "warehouse_id": "WH_02", "available_qty": 50, "reserved_qty": 0, "in_transit_qty": 0, "damaged_qty": 0},
    ])
    return QueryLayerService(
        sales_df=sales,
        products_df=products,
        inventory_df=inventory,
    )


class TestQueryLayerService:
    def test_service_initialization(self, mock_service):
        assert not mock_service.sales_df.empty
        assert not mock_service.products_df.empty
        assert not mock_service.inventory_df.empty
        assert mock_service.registry is not None

    def test_service_execute_tool_dynamic_dispatch(self, mock_service):
        ctx = QueryContext(currency="USD")
        res = mock_service.execute_tool("get_sales_summary", context=ctx)
        assert res.status == CalculationStatus.SUCCESS
        assert res.tool_name == "get_sales_summary"
        assert len(res.metrics) >= 5

    def test_service_direct_method_dispatch(self, mock_service):
        ctx = QueryContext(currency="USD")
        res = mock_service.get_sales_summary(context=ctx)
        assert res.status == CalculationStatus.SUCCESS
        res_inv = mock_service.get_inventory_summary(context=ctx)
        assert res_inv.status == CalculationStatus.SUCCESS

    def test_service_execute_unknown_tool_returns_error(self, mock_service):
        ctx = QueryContext()
        res = mock_service.execute_tool("non_existent_tool", context=ctx)
        assert res.status == CalculationStatus.ERROR
        assert "not registered" in res.error_message

    def test_service_does_not_mutate_underlying_data(self, mock_service):
        sales_copy = mock_service.sales_df.copy(deep=True)
        ctx = QueryContext(sku_id="SKU_01", currency="USD")
        res = mock_service.get_sales_summary(context=ctx)
        assert res.status == CalculationStatus.SUCCESS
        # Underlying dataset must have original length and values
        pd.testing.assert_frame_equal(mock_service.sales_df, sales_copy)


class TestToolRegistry:
    def test_default_registry_has_all_standard_tools(self):
        tool_names = default_registry.get_all_tool_names()
        assert len(tool_names) >= 25
        assert "get_sales_summary" in tool_names
        assert "get_inventory_summary" in tool_names
        assert "get_demand_summary" in tool_names
        assert "get_forecast" in tool_names
        assert "get_return_summary" in tool_names
        assert "get_revenue_summary" in tool_names
        assert "get_replenishment_reviews" in tool_names
        assert "get_business_impact_summary" in tool_names
        assert "get_recommendation_summary" in tool_names
        assert "get_decision_summary" in tool_names

    def test_tool_registry_list_by_domain(self):
        sales_tools = default_registry.list_tools(domain=QueryDomain.SALES)
        assert len(sales_tools) >= 5
        assert all(t.domain == "sales" for t in sales_tools)

        inv_tools = default_registry.list_tools(domain="inventory")
        assert len(inv_tools) >= 5
        assert all(t.domain == "inventory" for t in inv_tools)

    def test_registry_enforces_read_only(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError) as exc:
            reg.register(
                tool_name="illegal_write_tool",
                handler=lambda: None,
                metadata=ToolMetadata(
                    tool_name="illegal_write_tool",
                    display_name="Write Tool",
                    description="Attempts to modify state",
                    domain="operations",
                    read_only=False,  # ILLEGAL
                    source_engine="illegal",
                ),
            )
        assert "must be read_only=True" in str(exc.value)

    def test_to_copilot_declarations(self):
        declarations = default_registry.to_copilot_declarations()
        assert len(declarations) >= 25
        for decl in declarations:
            assert "name" in decl
            assert "description" in decl
            assert "parameters" in decl
            assert decl["read_only"] is True


class TestResponseContracts:
    def test_metric_result_kpi_card_contract(self):
        m = MetricResult(
            metric_name="net_revenue",
            display_name="Net Revenue",
            value=1000.0,
            unit="USD",
            currency="USD",
            previous_period_value=800.0,
            absolute_change=200.0,
            percentage_change=25.0,
            source="FinancialIntelligenceService",
            as_of_date="2026-06-30",
            evidence=[
                EvidenceReference(
                    source_engine="commerce_ai.financial",
                    source_type="portfolio",
                    source_id="port_1",
                    metric="net_revenue",
                    value=1000.0,
                )
            ],
        )
        assert m.metric_name == "net_revenue"
        assert m.value == 1000.0
        assert m.percentage_change == 25.0
        assert len(m.evidence) == 1

    def test_time_series_result_contract(self):
        ts = TimeSeriesResult(
            metric_name="demand_volume",
            display_name="Daily Demand",
            time_grain="daily",
            points=[],
            metadata=QueryResponse(
                query_id="q1",
                tool_name="get_demand_trend",
                domain="demand",
                metadata={"query_id": "q1", "tool_name": "get_demand_trend", "domain": "demand", "generated_as_of": "2026-06-30"},
            ).metadata,
        )
        assert ts.time_grain == "daily"
        assert ts.metric_name == "demand_volume"

    def test_table_result_contract(self):
        tb = TableResult(
            columns=["sku_id", "on_hand"],
            column_types={"sku_id": "string", "on_hand": "integer"},
            rows=[{"sku_id": "SKU_01", "on_hand": 50}],
            total_rows=1,
            metadata=QueryResponse(
                query_id="q2",
                tool_name="get_inventory_position",
                domain="inventory",
                metadata={"query_id": "q2", "tool_name": "get_inventory_position", "domain": "inventory", "generated_as_of": "2026-06-30"},
            ).metadata,
        )
        assert tb.total_rows == 1
        assert tb.rows[0]["on_hand"] == 50
