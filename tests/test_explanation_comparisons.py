"""Unit tests for Comparison Explanations (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import (
    CopilotExecutionResult,
    CopilotResponse,
    CopilotStepResult,
)
from commerce_ai.explanations.enums import ExplanationConfidence, ExplanationStatus, ExplanationType
from commerce_ai.explanations.templates import build_comparison_explanation
from commerce_ai.query_layer.schemas import MetricResult, QueryMetadata, QueryResponse


@pytest.fixture
def comparison_response() -> CopilotResponse:
    meta = QueryMetadata(
        query_id="Q-COMP",
        tool_name="get_revenue_summary",
        domain="financial",
        generated_as_of="2026-06-30",
        currency="USD",
        source_engine="commerce_ai.financial",
    )
    metric = MetricResult(
        metric_name="net_revenue",
        display_name="Net Revenue",
        value=91600.0,
        unit="USD",
        currency="USD",
        previous_period_value=100000.0,
        absolute_change=-8400.0,
        percentage_change=-8.4,
    )
    q_resp = QueryResponse(
        query_id="Q-COMP",
        tool_name="get_revenue_summary",
        domain="financial",
        metadata=meta,
        metrics=[metric],
    )
    step = CopilotStepResult(
        step_id="STEP-COMP",
        tool_name="get_revenue_summary",
        status=StepStatus.COMPLETED,
        response=q_resp,
    )
    exec_res = CopilotExecutionResult(
        result_id="RES-COMP",
        request_id="REQ-COMP",
        status=PlanExecutionStatus.COMPLETE,
        step_results=[step],
    )
    return CopilotResponse(
        response_id="RESP-COMP",
        request_id="REQ-COMP",
        state=CopilotState.COMPLETED,
        normalized_question="Compare net revenue to prior period.",
        execution_result=exec_res,
    )


class TestComparisonExplanations:
    def test_comparison_delta_and_percentage(self, comparison_response):
        exp = build_comparison_explanation(comparison_response)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.COMPARISON_EXPLANATION
        assert len(exp.key_findings) == 1
        finding_stmt = exp.key_findings[0].statement
        assert "$91,600.00" in finding_stmt
        assert "$100,000.00" in finding_stmt
        assert "-8.4%" in finding_stmt

    def test_comparisons_structured_data(self, comparison_response):
        exp = build_comparison_explanation(comparison_response)
        assert len(exp.comparisons) == 1
        comp = exp.comparisons[0]
        assert comp["current_value"] == 91600.0
        assert comp["comparison_value"] == 100000.0
        assert comp["percentage_change"] == -8.4
        assert comp["absolute_change"] == -8400.0

    def test_zero_baseline_safe_division(self):
        meta = QueryMetadata(
            query_id="Q-ZERO",
            tool_name="get_revenue_summary",
            domain="financial",
            generated_as_of="2026-06-30",
            currency="USD",
            source_engine="commerce_ai.financial",
        )
        metric = MetricResult(
            metric_name="new_channel_revenue",
            display_name="New Channel Revenue",
            value=5000.0,
            previous_period_value=0.0,
        )
        q_resp = QueryResponse(
            query_id="Q-ZERO",
            tool_name="get_revenue_summary",
            domain="financial",
            metadata=meta,
            metrics=[metric],
        )
        step = CopilotStepResult(
            step_id="STEP-ZERO",
            tool_name="get_revenue_summary",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-ZERO",
            request_id="REQ-ZERO",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-ZERO",
            request_id="REQ-ZERO",
            state=CopilotState.COMPLETED,
            normalized_question="Compare new channel revenue.",
            execution_result=exec_res,
        )

        exp = build_comparison_explanation(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.comparisons) == 1
        assert exp.comparisons[0]["percentage_change"] is None  # no ZeroDivisionError
