"""Mapping test suite for Business Query Contracts to Phase 7A Tools (Phase 7B).

Tests:
- Deterministic resolution of all 28 canonical intents
- Verification that all resolved tools are registered in Phase 7A canonical catalog
- Dimensional routing (SKU, Warehouse, Channel, Category, Brand, Trend)
- WHY-style diagnostic investigation multi-tool mappings
- Multi-domain cross-cutting request tool synthesis
- Operational review and governance proposal tool mappings
"""

from __future__ import annotations

import pytest

from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    OutputGrain,
    QueryIntent,
    RequestedOutput,
)
from commerce_ai.query_contracts.schemas import BusinessQueryContract
from commerce_ai.query_contracts.service import QueryContractService
from commerce_ai.query_contracts.tool_mapping import (
    PHASE_7A_CANONICAL_TOOLS,
    map_contract_to_tools,
)


class TestIntentToToolMapping:
    @pytest.fixture
    def service(self) -> QueryContractService:
        return QueryContractService()

    def test_all_intents_resolve_to_canonical_phase_7a_tools(self, service: QueryContractService):
        """Verify every canonical QueryIntent maps to valid tools registered in Phase 7A registry."""
        from commerce_ai.query_layer.registry import default_registry
        registry_tools = set(default_registry.get_all_tool_names())

        for intent in QueryIntent:
            contract, val = service.build_contract(intent=intent)
            tools = map_contract_to_tools(contract)
            if intent == QueryIntent.DATA_QUALITY_ANALYSIS:
                # DATA_QUALITY_ANALYSIS has no exposed read-only tool in Phase 7A
                assert len(tools) == 0
                assert any(w.code == "CAPABILITY_UNAVAILABLE_IN_PHASE_7A" for w in val.warnings)
            else:
                assert len(tools) > 0, f"Intent {intent.value} resolved to 0 tools"
                for t in tools:
                    assert t in PHASE_7A_CANONICAL_TOOLS, f"Tool '{t}' for intent '{intent.value}' not canonical"
                    assert t in registry_tools, f"Tool '{t}' for intent '{intent.value}' not in Phase 7A registry"
                    # Strictly no ranking tools
                    assert "ranking" not in t.lower(), f"Tool '{t}' implies ranking which is prohibited"

    def test_sales_dimensional_mapping(self, service: QueryContractService):
        # Default -> get_sales_summary
        c1, _ = service.build_contract(intent=QueryIntent.SALES_PERFORMANCE)
        assert c1.required_tools == ["get_sales_summary"]

        # By SKU -> get_sales_by_sku
        c2, _ = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            dimensions=[BusinessDimension.SKU],
        )
        assert c2.required_tools == ["get_sales_by_sku"]

        # By Warehouse -> get_sales_by_warehouse
        c3, _ = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            dimensions=[BusinessDimension.WAREHOUSE],
        )
        assert c3.required_tools == ["get_sales_by_warehouse"]

        # By Channel -> get_sales_by_channel
        c4, _ = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            dimensions=[BusinessDimension.CHANNEL],
        )
        assert c4.required_tools == ["get_sales_by_channel"]

        # By Category -> get_sales_by_category
        c5, _ = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            dimensions=[BusinessDimension.CATEGORY],
        )
        assert c5.required_tools == ["get_sales_by_category"]

        # By Brand -> get_sales_by_brand
        c6, _ = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            dimensions=[BusinessDimension.BRAND],
        )
        assert c6.required_tools == ["get_sales_by_brand"]

        # Time series / grain day -> get_sales_trend
        c7, _ = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            requested_output=RequestedOutput.TIME_SERIES,
        )
        assert c7.required_tools == ["get_sales_trend"]

    def test_financial_intent_mapping(self, service: QueryContractService):
        c_rev, _ = service.build_contract(intent=QueryIntent.REVENUE_ANALYSIS)
        assert "get_revenue_summary" in c_rev.required_tools

        c_margin, _ = service.build_contract(
            intent=QueryIntent.MARGIN_ANALYSIS,
            requested_output=RequestedOutput.BREAKDOWN,
        )
        assert "get_margin_summary" in c_margin.required_tools
        assert "get_margin_drivers" in c_margin.required_tools

        c_prof, _ = service.build_contract(intent=QueryIntent.PROFITABILITY_ANALYSIS)
        assert "get_profitability_attribution" in c_prof.required_tools

        c_ue, _ = service.build_contract(intent=QueryIntent.UNIT_ECONOMICS_ANALYSIS)
        assert "get_unit_economics" in c_ue.required_tools

    def test_inventory_intent_mapping(self, service: QueryContractService):
        c_inv, _ = service.build_contract(intent=QueryIntent.INVENTORY_STATUS)
        assert "get_inventory_summary" in c_inv.required_tools

        c_pos, _ = service.build_contract(
            intent=QueryIntent.INVENTORY_STATUS,
            requested_output=RequestedOutput.TABLE,
        )
        assert "get_inventory_position" in c_pos.required_tools

        c_stockout, _ = service.build_contract(intent=QueryIntent.STOCKOUT_ANALYSIS)
        assert "get_stockout_risk" in c_stockout.required_tools

        c_slow, _ = service.build_contract(intent=QueryIntent.SLOW_MOVING_INVENTORY)
        assert "get_slow_moving_inventory" in c_slow.required_tools

        c_hv, _ = service.build_contract(intent=QueryIntent.HIGH_VALUE_INVENTORY)
        assert "get_high_value_inventory" in c_hv.required_tools

    def test_demand_and_forecast_intent_mapping(self, service: QueryContractService):
        c_dem, _ = service.build_contract(intent=QueryIntent.DEMAND_ANALYSIS)
        assert "get_demand_summary" in c_dem.required_tools

        c_trend, _ = service.build_contract(intent=QueryIntent.DEMAND_TREND)
        assert "get_demand_trend" in c_trend.required_tools

        c_abc, _ = service.build_contract(intent=QueryIntent.ABC_XYZ_ANALYSIS)
        assert "get_abc_xyz_distribution" in c_abc.required_tools

        c_fc, _ = service.build_contract(intent=QueryIntent.FORECAST_ANALYSIS)
        assert "get_forecast" in c_fc.required_tools
        assert "get_forecast_summary" in c_fc.required_tools

        c_acc, _ = service.build_contract(intent=QueryIntent.FORECAST_ACCURACY)
        assert "get_forecast_accuracy" in c_acc.required_tools

        c_bias, _ = service.build_contract(intent=QueryIntent.FORECAST_BIAS)
        assert "get_forecast_bias" in c_bias.required_tools

    def test_returns_and_impact_intent_mapping(self, service: QueryContractService):
        c_ret, _ = service.build_contract(intent=QueryIntent.RETURN_ANALYSIS)
        assert "get_return_summary" in c_ret.required_tools
        assert "get_return_rate" in c_ret.required_tools

        c_anom, _ = service.build_contract(intent=QueryIntent.RETURN_ANOMALY_ANALYSIS)
        assert "get_return_anomalies" in c_anom.required_tools

        c_risk, _ = service.build_contract(intent=QueryIntent.RETURN_RISK)
        assert "get_return_risk" in c_risk.required_tools

        c_rsn, _ = service.build_contract(intent=QueryIntent.RETURN_REASON_ANALYSIS)
        assert "get_return_reason_breakdown" in c_rsn.required_tools

        c_imp, _ = service.build_contract(
            intent=QueryIntent.BUSINESS_IMPACT_ANALYSIS,
            requested_output=RequestedOutput.BREAKDOWN,
        )
        assert "get_business_impact_summary" in c_imp.required_tools
        assert "get_business_impact_by_category" in c_imp.required_tools
        assert "get_business_impact_by_type" in c_imp.required_tools

    def test_governance_reviews_mapping(self, service: QueryContractService):
        c_rep, _ = service.build_contract(intent=QueryIntent.REPLENISHMENT_REVIEW)
        assert c_rep.required_tools == ["get_replenishment_reviews"]
        assert c_rep.governance.approval_required is True

        c_po, _ = service.build_contract(intent=QueryIntent.PURCHASE_ORDER_REVIEW)
        assert c_po.required_tools == ["get_purchase_order_reviews"]
        assert c_po.governance.approval_required is True

        c_reb, _ = service.build_contract(intent=QueryIntent.WAREHOUSE_REBALANCING_REVIEW)
        assert c_reb.required_tools == ["get_warehouse_rebalancing_reviews"]
        assert c_reb.governance.approval_required is True

        c_rec, _ = service.build_contract(intent=QueryIntent.RECOMMENDATION_REVIEW)
        assert "get_recommendation_summary" in c_rec.required_tools
        assert "get_recommendations" in c_rec.required_tools

        c_dec, _ = service.build_contract(intent=QueryIntent.DECISION_REVIEW)
        assert "get_decision_summary" in c_dec.required_tools
        assert "get_decision_packages" in c_dec.required_tools


class TestWhyStyleInvestigationMapping:
    def test_why_margin_down_mapping(self):
        service = QueryContractService()
        contract, val = service.build_contract(
            intent=QueryIntent.MARGIN_ANALYSIS,
            explanation_context="Why is margin down investigation",
        )
        assert val.is_valid is True
        expected = [
            "get_margin_drivers",
            "get_margin_summary",
            "get_sales_by_channel",
            "get_sales_by_warehouse",
        ]
        assert contract.required_tools == expected

    def test_why_returns_increasing_mapping(self):
        service = QueryContractService()
        contract, val = service.build_contract(
            intent=QueryIntent.RETURN_ANALYSIS,
            explanation_context="Why are returns increasing investigation",
        )
        assert val.is_valid is True
        expected = [
            "get_return_anomalies",
            "get_return_reason_breakdown",
            "get_return_summary",
            "get_return_trend",
        ]
        assert contract.required_tools == expected

    def test_why_inventory_risk_high_mapping(self):
        service = QueryContractService()
        contract, val = service.build_contract(
            intent=QueryIntent.INVENTORY_RISK,
            explanation_context="Why is inventory risk high investigation",
        )
        assert val.is_valid is True
        expected = [
            "get_inventory_risk",
            "get_inventory_summary",
            "get_slow_moving_inventory",
            "get_stockout_risk",
        ]
        assert contract.required_tools == expected


class TestMultiDomainMapping:
    def test_multi_domain_revenue_margin_inventory(self):
        service = QueryContractService()
        contract, val = service.build_contract(
            intent=QueryIntent.MULTI_DOMAIN_ANALYSIS,
            domains=[BusinessDomain.FINANCIAL, BusinessDomain.INVENTORY],
        )
        assert val.is_valid is True
        expected = [
            "get_inventory_summary",
            "get_margin_summary",
            "get_revenue_summary",
        ]
        assert contract.required_tools == expected

    def test_multi_domain_demand_returns(self):
        service = QueryContractService()
        contract, val = service.build_contract(
            intent=QueryIntent.MULTI_DOMAIN_ANALYSIS,
            domains=[BusinessDomain.DEMAND, BusinessDomain.RETURNS],
        )
        assert val.is_valid is True
        assert "get_demand_summary" in contract.required_tools
        assert "get_return_summary" in contract.required_tools


class TestContractGovernanceAndRankingProhibition:
    def test_no_ranking_tool_exists_in_any_mapping(self):
        """Verify across all 28 intents that zero mapped tools contain ranking or winner concepts (Point 8 & 21-I)."""
        service = QueryContractService()
        for intent in QueryIntent:
            contract, _ = service.build_contract(intent=intent)
            for tool_name in contract.required_tools:
                assert "ranking" not in tool_name.lower(), f"Tool {tool_name} implies ranking"
                assert "top_n" not in tool_name.lower(), f"Tool {tool_name} implies top-N ranking"
                assert "winner" not in tool_name.lower(), f"Tool {tool_name} implies winner selection"
                assert "leaderboard" not in tool_name.lower(), f"Tool {tool_name} implies leaderboard"

    def test_business_impact_analysis_has_no_ranking(self):
        """Verify BUSINESS_IMPACT_ANALYSIS uses only aggregate tools, NEVER ranking (Point 8 & 21-J)."""
        service = QueryContractService()
        for output in [RequestedOutput.KPI, RequestedOutput.BREAKDOWN, RequestedOutput.DETAIL]:
            contract, val = service.build_contract(
                intent=QueryIntent.BUSINESS_IMPACT_ANALYSIS,
                requested_output=output,
            )
            assert val.is_valid is True
            for t in contract.required_tools:
                assert "ranking" not in t.lower()
                assert t in {
                    "get_business_impact_summary",
                    "get_business_impact_by_category",
                    "get_business_impact_by_type",
                    "get_physical_capital_exposure",
                }

    def test_replenishment_review_is_read_only(self):
        """Verify REPLENISHMENT_REVIEW is strictly read-only without execution (Point 10 & 21-K)."""
        service = QueryContractService()
        contract, val = service.build_contract(intent=QueryIntent.REPLENISHMENT_REVIEW)
        assert val.is_valid is True
        assert contract.governance.read_only is True
        assert contract.governance.execution_allowed is False
        assert contract.governance.action_execution is False
        assert contract.required_tools == ["get_replenishment_reviews"]

    def test_purchase_order_review_is_read_only(self):
        """Verify PURCHASE_ORDER_REVIEW is strictly read-only without PO creation (Point 10 & 21-L)."""
        service = QueryContractService()
        contract, val = service.build_contract(intent=QueryIntent.PURCHASE_ORDER_REVIEW)
        assert val.is_valid is True
        assert contract.governance.read_only is True
        assert contract.governance.execution_allowed is False
        assert contract.governance.action_execution is False
        assert contract.required_tools == ["get_purchase_order_reviews"]

    def test_warehouse_rebalancing_review_is_read_only(self):
        """Verify WAREHOUSE_REBALANCING_REVIEW is strictly read-only without stock transfer (Point 10 & 21-M)."""
        service = QueryContractService()
        contract, val = service.build_contract(intent=QueryIntent.WAREHOUSE_REBALANCING_REVIEW)
        assert val.is_valid is True
        assert contract.governance.read_only is True
        assert contract.governance.execution_allowed is False
        assert contract.governance.action_execution is False
        assert contract.required_tools == ["get_warehouse_rebalancing_reviews"]

    def test_direct_phase_7a_registry_compatibility(self):
        """Verify all 46 canonical tools are present in live Phase 7A registry (Point 1 & 21-X)."""
        from commerce_ai.query_layer.registry import default_registry
        from commerce_ai.query_contracts.tool_mapping import PHASE_7A_CANONICAL_TOOLS, PHASE_7A_APPROVED_ALIASES

        reg_tools = set(default_registry.get_all_tool_names())
        assert len(PHASE_7A_CANONICAL_TOOLS) == 46
        assert len(PHASE_7A_APPROVED_ALIASES) == 3
        # Assert canonical tools exist in registry
        for t in PHASE_7A_CANONICAL_TOOLS:
            assert t in reg_tools, f"Canonical tool '{t}' missing from Phase 7A registry"
        # Assert aliases exist in registry
        for a in PHASE_7A_APPROVED_ALIASES:
            assert a in reg_tools, f"Approved alias '{a}' missing from Phase 7A registry"

    def test_phase_7a_aliases_recognized(self):
        """Verify the 3 approved aliases from Phase 7A are recognized (Point 1)."""
        from commerce_ai.query_contracts.tool_mapping import PHASE_7A_APPROVED_ALIASES
        assert "get_inventory_risk_breakdown" in PHASE_7A_APPROVED_ALIASES
        assert "get_recommendation_list" in PHASE_7A_APPROVED_ALIASES
        assert "get_decision_list" in PHASE_7A_APPROVED_ALIASES
