"""Unit tests for Copilot Decomposition & Phase 7B Plan Synthesis (Phase 7C).

Tests:
- Translation of interpretations into Phase 7B BusinessQueryContract and QueryPlan
- Preservation of filters, as_of_date, and currency across the contract boundary
- Multi-step reasoning plan synthesis for why-style inquiries
- Correct handling of unavailable capabilities (DATA_QUALITY_ANALYSIS)
- Governance inheritance from Phase 7B contracts
"""

from __future__ import annotations

import pytest

from commerce_ai.copilot.decomposition import decompose_and_plan
from commerce_ai.copilot.enums import CopilotState, StepStatus
from commerce_ai.copilot.interpreter import interpret_question
from commerce_ai.copilot.schemas import CopilotInterpretation, CopilotRequest
from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    BusinessGrain,
    MetricIdentifier,
    PlanStatus,
    QueryIntent,
    RequestedOutput,
    TimeGranularity,
)
from commerce_ai.query_contracts.service import QueryContractService


class TestDecompositionAndPlanning:
    @pytest.fixture
    def contract_service(self) -> QueryContractService:
        return QueryContractService()

    def test_single_step_sales_performance(self, contract_service: QueryContractService):
        req = CopilotRequest(question="How are sales performing?", as_of_date="2026-06-30", currency="USD")
        interp = interpret_question(req)
        plan, warnings, errors = decompose_and_plan(interp, req, contract_service=contract_service)

        assert plan is not None
        assert len(errors) == 0
        assert plan.state == CopilotState.PLANNED
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "get_sales_summary"
        assert plan.steps[0].arguments["as_of_date"] == "2026-06-30"
        assert plan.steps[0].arguments["currency"] == "USD"
        assert plan.governance.read_only is True
        assert plan.governance.execution_allowed is False

    def test_why_margin_down_four_step_plan(self, contract_service: QueryContractService):
        req = CopilotRequest(question="Why did margin decline?", as_of_date="2026-06-30", currency="USD")
        interp = interpret_question(req)
        plan, warnings, errors = decompose_and_plan(interp, req, contract_service=contract_service)

        assert plan is not None
        assert len(errors) == 0
        assert len(plan.steps) == 4
        tool_names = [s.tool_name for s in plan.steps]
        assert "get_margin_summary" in tool_names
        assert "get_margin_drivers" in tool_names
        assert "get_sales_by_channel" in tool_names
        assert "get_sales_by_warehouse" in tool_names
        # First step is summary
        assert plan.steps[0].tool_name == "get_margin_summary"
        assert plan.steps[0].status == StepStatus.READY

    def test_why_inventory_risk_high_four_step_plan(self, contract_service: QueryContractService):
        req = CopilotRequest(question="Why is inventory risk high?", as_of_date="2026-06-30")
        interp = interpret_question(req)
        plan, warnings, errors = decompose_and_plan(interp, req, contract_service=contract_service)

        assert plan is not None
        assert len(errors) == 0
        assert len(plan.steps) == 4
        tool_names = [s.tool_name for s in plan.steps]
        assert "get_inventory_risk" in tool_names
        assert "get_inventory_summary" in tool_names
        assert "get_slow_moving_inventory" in tool_names
        assert "get_stockout_risk" in tool_names

    def test_why_returns_increasing_four_step_plan(self, contract_service: QueryContractService):
        req = CopilotRequest(question="Why are returns increasing?", as_of_date="2026-06-30")
        interp = interpret_question(req)
        plan, warnings, errors = decompose_and_plan(interp, req, contract_service=contract_service)

        assert plan is not None
        assert len(errors) == 0
        assert len(plan.steps) == 4
        tool_names = [s.tool_name for s in plan.steps]
        assert "get_return_summary" in tool_names
        assert "get_return_trend" in tool_names
        assert "get_return_reason_breakdown" in tool_names
        assert "get_return_anomalies" in tool_names

    def test_filter_propagation_to_steps(self, contract_service: QueryContractService):
        req = CopilotRequest(
            question="Show inventory for SKU_999 in warehouse WH_01",
            as_of_date="2026-06-30",
        )
        interp = interpret_question(req)
        plan, _, _ = decompose_and_plan(interp, req, contract_service=contract_service)

        assert plan is not None
        step_args = plan.steps[0].arguments
        assert step_args["sku_id"] == "SKU_999"
        assert step_args["warehouse_id"] == "WH_01"
        assert step_args["as_of_date"] == "2026-06-30"

    def test_unavailable_capability_data_quality(self, contract_service: QueryContractService):
        req = CopilotRequest(question="Check data quality completeness", as_of_date="2026-06-30")
        interp = interpret_question(req)
        plan, warnings, errors = decompose_and_plan(interp, req, contract_service=contract_service)

        assert plan is not None
        assert plan.state == CopilotState.UNAVAILABLE
        assert len(plan.steps) == 0
        assert any("CAPABILITY_UNAVAILABLE_IN_PHASE_7A" in w for w in warnings)

    def test_operational_reviews_governance_approval_flag(self, contract_service: QueryContractService):
        req = CopilotRequest(question="What recommendations are pending review?", as_of_date="2026-06-30")
        interp = interpret_question(req)
        plan, _, _ = decompose_and_plan(interp, req, contract_service=contract_service)

        assert plan is not None
        assert plan.governance.approval_required is True
        assert plan.governance.execution_allowed is False

    def test_plan_id_determinism(self, contract_service: QueryContractService):
        req1 = CopilotRequest(question="How are sales performing?", as_of_date="2026-06-30", currency="USD")
        req2 = CopilotRequest(question="How are sales performing?", as_of_date="2026-06-30", currency="USD")
        i1 = interpret_question(req1)
        i2 = interpret_question(req2)
        p1, _, _ = decompose_and_plan(i1, req1, contract_service=contract_service)
        p2, _, _ = decompose_and_plan(i2, req2, contract_service=contract_service)

        assert p1 is not None and p2 is not None
        assert p1.plan_id == p2.plan_id
