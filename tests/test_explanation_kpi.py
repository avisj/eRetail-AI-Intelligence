"""Unit tests for KPI Explanations (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import (
    CopilotExecutionResult,
    CopilotResponse,
    CopilotStepResult,
)
from commerce_ai.explanations.enums import ExplanationConfidence, ExplanationStatus, ExplanationType
from commerce_ai.explanations.templates import build_kpi_explanation
from commerce_ai.query_layer.schemas import MetricResult, QueryMetadata, QueryResponse


@pytest.fixture
def multi_kpi_response() -> CopilotResponse:
    meta = QueryMetadata(
        query_id="Q-KPI",
        tool_name="get_sales_summary",
        domain="sales",
        generated_as_of="2026-06-30",
        currency="USD",
        source_engine="commerce_ai.sales",
    )
    metrics = [
        MetricResult(metric_name="net_revenue", display_name="Net Revenue", value=250000.0, unit="USD", currency="USD"),
        MetricResult(metric_name="orders_count", display_name="Orders Count", value=1250, unit="orders"),
        MetricResult(metric_name="aov", display_name="Average Order Value", value=200.0, unit="USD", currency="USD"),
    ]
    q_resp = QueryResponse(
        query_id="Q-KPI",
        tool_name="get_sales_summary",
        domain="sales",
        metadata=meta,
        metrics=metrics,
    )
    step = CopilotStepResult(
        step_id="STEP-1",
        tool_name="get_sales_summary",
        status=StepStatus.COMPLETED,
        response=q_resp,
    )
    exec_res = CopilotExecutionResult(
        result_id="RES-KPI",
        request_id="REQ-KPI",
        status=PlanExecutionStatus.COMPLETE,
        step_results=[step],
    )
    return CopilotResponse(
        response_id="RESP-KPI",
        request_id="REQ-KPI",
        state=CopilotState.COMPLETED,
        normalized_question="How are sales performing?",
        execution_result=exec_res,
    )


class TestKPIExplanations:
    def test_multi_kpi_findings_count(self, multi_kpi_response):
        exp = build_kpi_explanation(multi_kpi_response)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.KPI_EXPLANATION
        assert len(exp.key_findings) == 3
        assert exp.confidence == ExplanationConfidence.HIGH

    def test_multi_kpi_metrics_extracted(self, multi_kpi_response):
        exp = build_kpi_explanation(multi_kpi_response)
        assert "net_revenue" in exp.metrics
        assert "orders_count" in exp.metrics
        assert "aov" in exp.metrics
        assert exp.metrics["net_revenue"] == 250000.0
        assert exp.metrics["orders_count"] == 1250

    def test_kpi_finding_statement_formatting(self, multi_kpi_response):
        exp = build_kpi_explanation(multi_kpi_response)
        statements = [f.statement for f in exp.key_findings]
        assert any("$250,000.00" in s for s in statements)
        assert any("1,250" in s for s in statements)
        assert any("$200.00" in s for s in statements)

    def test_empty_kpi_results_produce_insufficient_status(self):
        meta = QueryMetadata(
            query_id="Q-EMPTY",
            tool_name="get_sales_summary",
            domain="sales",
            generated_as_of="2026-06-30",
            source_engine="commerce_ai.sales",
        )
        q_resp = QueryResponse(
            query_id="Q-EMPTY",
            tool_name="get_sales_summary",
            domain="sales",
            metadata=meta,
            metrics=[],
        )
        step = CopilotStepResult(
            step_id="STEP-EMPTY",
            tool_name="get_sales_summary",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-EMPTY",
            request_id="REQ-EMPTY",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-EMPTY",
            request_id="REQ-EMPTY",
            state=CopilotState.COMPLETED,
            normalized_question="How are sales performing?",
            execution_result=exec_res,
        )
        exp = build_kpi_explanation(resp)
        assert exp.status == ExplanationStatus.INSUFFICIENT_DATA
        assert "unavailable" in exp.headline.lower()
