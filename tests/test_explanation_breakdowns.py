"""Unit tests for Breakdown Explanations (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import (
    CopilotExecutionResult,
    CopilotResponse,
    CopilotStepResult,
)
from commerce_ai.explanations.enums import ExplanationConfidence, ExplanationStatus, ExplanationType
from commerce_ai.explanations.templates import build_breakdown_explanation
from commerce_ai.query_layer.schemas import (
    BreakdownItem,
    BreakdownResult,
    QueryMetadata,
    QueryResponse,
)


@pytest.fixture
def breakdown_response() -> CopilotResponse:
    meta = QueryMetadata(
        query_id="Q-BK",
        tool_name="get_sales_by_channel",
        domain="sales",
        generated_as_of="2026-06-30",
        currency="USD",
        source_engine="commerce_ai.sales",
    )
    items = [
        BreakdownItem(dimension_name="channel_id", dimension_value="ONLINE", metric_value=60000.0, metric_name="revenue", percentage_of_total=60.0, currency="USD"),
        BreakdownItem(dimension_name="channel_id", dimension_value="RETAIL", metric_value=30000.0, metric_name="revenue", percentage_of_total=30.0, currency="USD"),
        BreakdownItem(dimension_name="channel_id", dimension_value="WHOLESALE", metric_value=10000.0, metric_name="revenue", percentage_of_total=10.0, currency="USD"),
    ]
    bk = BreakdownResult(
        metric_name="revenue",
        display_name="Revenue by Channel",
        dimension_name="channel_id",
        items=items,
        metadata=meta,
    )
    q_resp = QueryResponse(
        query_id="Q-BK",
        tool_name="get_sales_by_channel",
        domain="sales",
        metadata=meta,
        breakdown=bk,
    )
    step = CopilotStepResult(
        step_id="STEP-BK",
        tool_name="get_sales_by_channel",
        status=StepStatus.COMPLETED,
        response=q_resp,
    )
    exec_res = CopilotExecutionResult(
        result_id="RES-BK",
        request_id="REQ-BK",
        status=PlanExecutionStatus.COMPLETE,
        step_results=[step],
    )
    return CopilotResponse(
        response_id="RESP-BK",
        request_id="REQ-BK",
        state=CopilotState.COMPLETED,
        normalized_question="Show revenue by channel.",
        execution_result=exec_res,
    )


class TestBreakdownExplanations:
    def test_breakdown_findings_count(self, breakdown_response):
        exp = build_breakdown_explanation(breakdown_response)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.BREAKDOWN_EXPLANATION
        assert len(exp.key_findings) == 3

    def test_breakdown_values_and_shares(self, breakdown_response):
        exp = build_breakdown_explanation(breakdown_response)
        stmts = [f.statement for f in exp.key_findings]
        assert any("ONLINE" in s and "$60,000.00" in s and "60.0% share" in s for s in stmts)
        assert any("RETAIL" in s and "$30,000.00" in s and "30.0% share" in s for s in stmts)
        assert any("WHOLESALE" in s and "$10,000.00" in s and "10.0% share" in s for s in stmts)

    def test_breakdown_prohibits_ranking_language(self, breakdown_response):
        exp = build_breakdown_explanation(breakdown_response)
        full_text = exp.headline + " " + exp.summary + " " + " ".join(f.statement for f in exp.key_findings)
        assert "winner" not in full_text.lower()
        assert "loser" not in full_text.lower()
        assert "top 1" not in full_text.lower()
        assert "best" not in full_text.lower()
        assert "worst" not in full_text.lower()
