"""QueryPlan test suite for Business Query Contracts (Phase 7B).

Tests:
- Deterministic QueryPlan generation from BusinessQueryContract
- Plan ID format and deterministic reproducibility
- Step sequencing (summary steps ordered first)
- Step dependency graphs
- QueryContext argument generation (dates, currency, dimensions, pagination)
- Governance assertions inherited by plan
- Plan rejection on invalid contracts
- Complex multi-step plans for investigation and multi-domain contracts
"""

from __future__ import annotations

import pytest

from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    BusinessGrain,
    OutputGrain,
    PlanStatus,
    QueryIntent,
    RequestedOutput,
    TimeGranularity,
)
from commerce_ai.query_contracts.planner import create_query_plan
from commerce_ai.query_contracts.schemas import (
    BusinessQueryContract,
    BusinessQueryFilter,
    TimeRangeContract,
)
from commerce_ai.query_contracts.service import QueryContractService


class TestQueryPlanSynthesis:
    @pytest.fixture
    def service(self) -> QueryContractService:
        return QueryContractService()

    def test_basic_plan_generation(self, service: QueryContractService):
        contract, val = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            as_of_date="2026-06-30",
            currency="USD",
        )
        assert val.is_valid is True

        plan = service.plan(contract)
        assert plan.plan_id.startswith("PLAN-")
        assert plan.contract_id == contract.query_id
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "get_sales_summary"
        assert plan.steps[0].arguments["as_of_date"] == "2026-06-30"
        assert plan.steps[0].arguments["currency"] == "USD"
        assert plan.governance.read_only is True
        assert plan.governance.execution_allowed is False
        assert plan.status == PlanStatus.PLANNED

    def test_plan_id_determinism(self, service: QueryContractService):
        c1, _ = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            as_of_date="2026-06-30",
            currency="USD",
        )
        c2, _ = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            as_of_date="2026-06-30",
            currency="USD",
        )
        p1 = service.plan(c1)
        p2 = service.plan(c2)
        assert p1.plan_id == p2.plan_id

    def test_step_sequencing_and_dependencies(self, service: QueryContractService):
        # Why is margin down has 4 tools: get_margin_summary (summary), get_margin_drivers, get_sales_by_channel, get_sales_by_warehouse
        contract, _ = service.build_contract(
            intent=QueryIntent.MARGIN_ANALYSIS,
            explanation_context="Why is margin down investigation",
            as_of_date="2026-06-30",
            currency="USD",
        )
        plan = service.plan(contract)
        assert len(plan.steps) == 4

        # Summary tool must be first (sequence 1)
        assert plan.steps[0].tool_name == "get_margin_summary"
        assert plan.steps[0].sequence == 1
        assert plan.steps[0].required is True

        # Secondary tools have dependencies on get_margin_summary
        assert "get_margin_drivers" in plan.dependencies
        assert plan.dependencies["get_margin_drivers"] == ["get_margin_summary"]

    def test_filter_arguments_propagation(self, service: QueryContractService):
        contract, _ = service.build_contract(
            intent=QueryIntent.INVENTORY_STATUS,
            dimensions=[BusinessDimension.SKU],
            requested_output=RequestedOutput.TABLE,
            filters={
                "sku_id": "SKU_001",
                "warehouse_id": "WH_01",
                "limit": 25,
                "offset": 50,
            },
            as_of_date="2026-06-30",
            currency="USD",
        )
        plan = service.plan(contract)
        step = plan.steps[0]
        assert step.tool_name == "get_inventory_position"
        args = step.arguments
        assert args["sku_id"] == "SKU_001"
        assert args["warehouse_id"] == "WH_01"
        assert args["limit"] == 25
        assert args["offset"] == 50
        assert args["as_of_date"] == "2026-06-30"

    def test_time_grain_argument_propagation(self, service: QueryContractService):
        contract, _ = service.build_contract(
            intent=QueryIntent.DEMAND_TREND,
            requested_grain=OutputGrain.MONTH,
            time_range={"start_date": "2026-01-01", "end_date": "2026-06-30"},
            as_of_date="2026-06-30",
        )
        plan = service.plan(contract)
        assert plan.steps[0].tool_name == "get_demand_trend"
        assert plan.steps[0].arguments["time_grain"] == "month"
        assert plan.steps[0].arguments["start_date"] == "2026-01-01"
        assert plan.steps[0].arguments["end_date"] == "2026-06-30"

    def test_multi_domain_plan(self, service: QueryContractService):
        contract, _ = service.build_contract(
            intent=QueryIntent.MULTI_DOMAIN_ANALYSIS,
            domains=[BusinessDomain.FINANCIAL, BusinessDomain.INVENTORY],
            as_of_date="2026-06-30",
        )
        plan = service.plan(contract)
        tools = {s.tool_name for s in plan.steps}
        assert "get_inventory_summary" in tools
        assert "get_margin_summary" in tools
        assert "get_revenue_summary" in tools

    def test_plan_fails_on_invalid_contract(self, service: QueryContractService):
        # Force invalid contract with unregistered tool
        contract, _ = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            required_tools=["completely_invalid_tool_name"],
        )
        with pytest.raises(ValueError, match="Cannot generate QueryPlan for invalid contract"):
            service.plan(contract)

    def test_data_quality_plan_unavailable(self, service: QueryContractService):
        contract, val = service.build_contract(
            intent=QueryIntent.DATA_QUALITY_ANALYSIS,
            as_of_date="2026-06-30",
        )
        assert any(w.code == "CAPABILITY_UNAVAILABLE_IN_PHASE_7A" for w in val.warnings)
        plan = service.plan(contract)
        assert plan.status == PlanStatus.UNAVAILABLE
        assert len(plan.steps) == 0
        assert plan.dependencies == {}
        assert plan.governance.read_only is True
        assert plan.governance.execution_allowed is False

    def test_grain_and_granularity_orthogonal_propagation(self, service: QueryContractService):
        contract, _ = service.build_contract(
            intent=QueryIntent.DEMAND_TREND,
            requested_grain=BusinessGrain.SKU,
            time_granularity=TimeGranularity.DAY,
            time_range={"start_date": "2026-06-01", "end_date": "2026-06-30"},
            as_of_date="2026-06-30",
        )
        plan = service.plan(contract)
        assert plan.steps[0].arguments["time_grain"] == "day"
        assert contract.requested_grain == BusinessGrain.SKU
        assert contract.time_granularity == TimeGranularity.DAY

    def test_why_inventory_risk_high_plan(self, service: QueryContractService):
        contract, _ = service.build_contract(
            intent=QueryIntent.INVENTORY_RISK,
            explanation_context="Why is inventory risk high investigation",
            as_of_date="2026-06-30",
            currency="USD",
        )
        plan = service.plan(contract)
        assert len(plan.steps) == 4
        tool_names = [s.tool_name for s in plan.steps]
        assert "get_inventory_risk" in tool_names
        assert "get_inventory_summary" in tool_names
        assert "get_slow_moving_inventory" in tool_names
        assert "get_stockout_risk" in tool_names
        # First step is summary or primary risk
        assert plan.steps[0].sequence == 1
        assert plan.steps[0].required is True

    def test_readonly_review_plans_governance(self, service: QueryContractService):
        for intent in [
            QueryIntent.REPLENISHMENT_REVIEW,
            QueryIntent.PURCHASE_ORDER_REVIEW,
            QueryIntent.WAREHOUSE_REBALANCING_REVIEW,
        ]:
            contract, _ = service.build_contract(intent=intent, as_of_date="2026-06-30")
            plan = service.plan(contract)
            assert plan.governance.read_only is True
            assert plan.governance.execution_allowed is False
            assert plan.governance.action_execution is False
            for step in plan.steps:
                assert step.arguments.get("execution_allowed", False) is False

    def test_canonical_question_templates_registry(self, service: QueryContractService):
        from commerce_ai.query_contracts.templates import get_canonical_templates
        templates = get_canonical_templates()
        assert len(templates) >= 15
        ids = {t.template_id for t in templates}
        assert "TPL_WHY_MARGIN_DOWN" in ids
        assert "TPL_WHY_INVENTORY_RISK_HIGH" in ids
        assert "TPL_WHY_RETURNS_INCREASING" in ids

