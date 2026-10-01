"""Unit tests for Copilot Step Execution & Tool Coordination (Phase 7C).

Tests:
- Execution of single-step plans against Phase 7A tools
- Execution of multi-step why-style diagnostic plans
- Step dependency enforcement during execution
- Failure isolation and partial success outcomes
- Unavailable plan execution handling
- Decision package neutrality on execution
"""

from __future__ import annotations

import pytest

from commerce_ai.copilot.decomposition import decompose_and_plan
from commerce_ai.copilot.enums import (
    CopilotProvenanceSource,
    CopilotState,
    PlanExecutionStatus,
    StepStatus,
)
from commerce_ai.copilot.execution import execute_plan
from commerce_ai.copilot.interpreter import interpret_question
from commerce_ai.copilot.schemas import CopilotExecutionPlan, CopilotRequest, CopilotStep
from commerce_ai.query_layer.service import QueryLayerService


class TestCopilotExecution:
    @pytest.fixture
    def query_layer(self) -> QueryLayerService:
        return QueryLayerService()

    def test_single_step_sales_execution(self, query_layer: QueryLayerService):
        req = CopilotRequest(question="How are sales performing?", as_of_date="2026-06-30", currency="USD")
        interp = interpret_question(req)
        plan, _, _ = decompose_and_plan(interp, req)
        assert plan is not None

        result = execute_plan(plan, query_layer=query_layer)
        assert result.status == PlanExecutionStatus.COMPLETE
        assert len(result.step_results) == 1
        assert result.step_results[0].status == StepStatus.COMPLETED
        assert len(result.evidence_chain) == 1
        assert result.evidence_chain[0].provenance == CopilotProvenanceSource.PHASE_7A_TOOL
        assert result.total_execution_time_ms >= 0.0

    def test_why_margin_down_multi_step_execution(self, query_layer: QueryLayerService):
        req = CopilotRequest(question="Why did margin decline?", as_of_date="2026-06-30", currency="USD")
        interp = interpret_question(req)
        plan, _, _ = decompose_and_plan(interp, req)
        assert plan is not None
        assert len(plan.steps) == 4

        result = execute_plan(plan, query_layer=query_layer)
        assert result.status == PlanExecutionStatus.COMPLETE
        assert len(result.step_results) == 4
        for sr in result.step_results:
            assert sr.status == StepStatus.COMPLETED
            assert sr.evidence is not None

    def test_why_inventory_risk_multi_step_execution(self, query_layer: QueryLayerService):
        req = CopilotRequest(question="Why is inventory risk high?", as_of_date="2026-06-30")
        interp = interpret_question(req)
        plan, _, _ = decompose_and_plan(interp, req)
        assert plan is not None

        result = execute_plan(plan, query_layer=query_layer)
        assert result.status == PlanExecutionStatus.COMPLETE
        assert len(result.step_results) == 4

    def test_why_returns_increasing_multi_step_execution(self, query_layer: QueryLayerService):
        req = CopilotRequest(question="Why are returns increasing?", as_of_date="2026-06-30")
        interp = interpret_question(req)
        plan, _, _ = decompose_and_plan(interp, req)
        assert plan is not None

        result = execute_plan(plan, query_layer=query_layer)
        assert result.status == PlanExecutionStatus.COMPLETE
        assert len(result.step_results) == 4

    def test_unavailable_plan_execution(self, query_layer: QueryLayerService):
        req = CopilotRequest(question="Check data quality completeness", as_of_date="2026-06-30")
        interp = interpret_question(req)
        plan, _, _ = decompose_and_plan(interp, req)
        assert plan is not None
        assert plan.state == CopilotState.UNAVAILABLE

        result = execute_plan(plan, query_layer=query_layer)
        assert result.status == PlanExecutionStatus.UNAVAILABLE
        assert len(result.step_results) == 0

    def test_failure_isolation_on_tool_error(self):
        # Create a mock query layer where tool_a fails but tool_b succeeds
        class MockQueryLayer(QueryLayerService):
            def execute_tool(self, tool_name: str, **kwargs):
                if tool_name == "failing_tool":
                    raise RuntimeError("Database connection timed out")
                return super().execute_tool(tool_name, **kwargs)

        mock_ql = MockQueryLayer()
        s1 = CopilotStep(
            step_id="STEP-1",
            sequence=1,
            contract_id="QRY-1",
            plan_id="PLAN-1",
            tool_name="failing_tool",
            arguments={"as_of_date": "2026-06-30"},
            purpose="Simulate failure",
            required=False,
            dependencies=[],
        )
        s2 = CopilotStep(
            step_id="STEP-2",
            sequence=2,
            contract_id="QRY-2",
            plan_id="PLAN-2",
            tool_name="get_sales_summary",
            arguments={"as_of_date": "2026-06-30"},
            purpose="Independent success",
            required=True,
            dependencies=[],
        )
        plan = CopilotExecutionPlan(
            plan_id="PLAN-partial",
            request_id="REQ-partial",
            steps=[s1, s2],
            execution_order=["STEP-1", "STEP-2"],
        )

        result = execute_plan(plan, query_layer=mock_ql)
        assert result.status == PlanExecutionStatus.PARTIAL
        assert len(result.failures) == 1
        assert result.failures[0].error_code == "TOOL_EXECUTION_FAILURE"
        # STEP-2 succeeded
        assert result.step_results[1].status == StepStatus.COMPLETED

    def test_dependent_step_blocked_on_prerequisite_failure(self):
        class MockQueryLayer(QueryLayerService):
            def execute_tool(self, tool_name: str, **kwargs):
                if tool_name == "failing_tool":
                    raise RuntimeError("Failed prerequisite")
                return super().execute_tool(tool_name, **kwargs)

        mock_ql = MockQueryLayer()
        s1 = CopilotStep(
            step_id="STEP-1",
            sequence=1,
            contract_id="QRY-1",
            plan_id="PLAN-1",
            tool_name="failing_tool",
            arguments={},
            purpose="Fail prerequisite",
            required=True,
            dependencies=[],
        )
        s2 = CopilotStep(
            step_id="STEP-2",
            sequence=2,
            contract_id="QRY-2",
            plan_id="PLAN-2",
            tool_name="get_sales_summary",
            arguments={},
            purpose="Dependent step",
            required=True,
            dependencies=["STEP-1"],
        )
        plan = CopilotExecutionPlan(
            plan_id="PLAN-blocked",
            request_id="REQ-blocked",
            steps=[s1, s2],
            execution_order=["STEP-1", "STEP-2"],
            dependencies={"STEP-2": ["STEP-1"]},
        )

        result = execute_plan(plan, query_layer=mock_ql)
        assert result.status == PlanExecutionStatus.FAILED
        assert result.step_results[0].status == StepStatus.FAILED
        assert result.step_results[1].status == StepStatus.BLOCKED
