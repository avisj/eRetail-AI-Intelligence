"""Unit tests for Phase 7D Deterministic Explanation Template Builders."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import (
    CopilotExecutionResult,
    CopilotResponse,
    CopilotStepResult,
)
from commerce_ai.explanations.enums import (
    ExplanationConfidence,
    ExplanationStatus,
    ExplanationType,
    ProvenanceType,
)
from commerce_ai.explanations.templates import (
    _format_metric_val,
    _make_explanation_id,
    _make_finding_id,
    build_insufficient_data_explanation,
    build_kpi_explanation,
    build_operational_review_explanation,
    build_unavailable_explanation,
)
from commerce_ai.query_layer.schemas import (
    MetricResult,
    QueryMetadata,
    QueryResponse,
    TableResult,
)


@pytest.fixture
def sample_kpi_response() -> CopilotResponse:
    meta = QueryMetadata(
        query_id="Q-1",
        tool_name="get_margin_summary",
        domain="financial",
        generated_as_of="2026-06-30",
        currency="USD",
        source_engine="commerce_ai.financial",
    )
    metric = MetricResult(
        metric_name="gross_margin",
        display_name="Gross Margin",
        value=50.1,
        unit="%",
        currency="USD",
    )
    q_resp = QueryResponse(
        query_id="Q-1",
        tool_name="get_margin_summary",
        domain="financial",
        metadata=meta,
        metrics=[metric],
    )
    step = CopilotStepResult(
        step_id="STEP-1",
        tool_name="get_margin_summary",
        status=StepStatus.COMPLETED,
        response=q_resp,
    )
    exec_res = CopilotExecutionResult(
        result_id="RES-1",
        request_id="REQ-1",
        status=PlanExecutionStatus.COMPLETE,
        step_results=[step],
    )
    return CopilotResponse(
        response_id="RESP-1",
        request_id="REQ-1",
        state=CopilotState.COMPLETED,
        normalized_question="What is our gross margin?",
        execution_result=exec_res,
    )


class TestExplanationTemplates:
    def test_format_metric_val_currency(self):
        assert _format_metric_val(12345.678, currency="USD") == "$12,345.68"
        assert _format_metric_val(100.0, unit="USD") == "$100.00"

    def test_format_metric_val_percentage(self):
        assert _format_metric_val(42.54, unit="%") == "42.5%"

    def test_format_metric_val_integer_and_none(self):
        assert _format_metric_val(5000) == "5,000"
        assert _format_metric_val(None) == "N/A"

    def test_deterministic_id_generation(self):
        id1 = _make_explanation_id("REQ-123", "KPI")
        id2 = _make_explanation_id("REQ-123", "KPI")
        id3 = _make_explanation_id("REQ-456", "KPI")
        assert id1 == id2
        assert id1 != id3
        assert id1.startswith("EXP-")

        f_id = _make_finding_id("EXP-TEST", 1)
        assert f_id == "EXP-TEST-FND-01"

    def test_build_kpi_explanation(self, sample_kpi_response):
        exp = build_kpi_explanation(sample_kpi_response)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.KPI_EXPLANATION
        assert "50.1%" in exp.headline
        assert len(exp.key_findings) == 1
        assert "Gross Margin was 50.1%." in exp.key_findings[0].statement
        assert exp.confidence == ExplanationConfidence.HIGH
        assert len(exp.limitations) > 0  # Financial limitations attached

    def test_build_insufficient_data_explanation(self):
        resp = CopilotResponse(
            response_id="RESP-ERR",
            request_id="REQ-ERR",
            state=CopilotState.FAILED,
            normalized_question="Show missing data.",
        )
        exp = build_insufficient_data_explanation(resp, reason="Missing underlying records.")
        assert exp.status == ExplanationStatus.INSUFFICIENT_DATA
        assert exp.explanation_type == ExplanationType.INSUFFICIENT_DATA_EXPLANATION
        assert "Insufficient data" in exp.headline
        assert exp.confidence == ExplanationConfidence.INSUFFICIENT
        assert len(exp.key_findings) == 1

    def test_build_unavailable_explanation(self):
        resp = CopilotResponse(
            response_id="RESP-UNAV",
            request_id="REQ-UNAV",
            state=CopilotState.UNAVAILABLE,
            normalized_question="Run unsupported analysis.",
        )
        exp = build_unavailable_explanation(resp)
        assert exp.status == ExplanationStatus.UNAVAILABLE
        assert exp.explanation_type == ExplanationType.UNAVAILABLE_EXPLANATION
        assert "unavailable" in exp.headline.lower()
        assert exp.confidence == ExplanationConfidence.INSUFFICIENT

    def test_build_operational_review_explanation(self):
        meta = QueryMetadata(
            query_id="Q-REV",
            tool_name="get_replenishment_reviews",
            domain="operations",
            generated_as_of="2026-06-30",
            source_engine="commerce_ai.operations",
        )
        tbl = TableResult(
            columns=["sku_id", "warehouse_id", "recommended_qty"],
            rows=[{"sku_id": "SKU_01", "recommended_qty": 500}],
            total_rows=1,
            metadata=meta,
        )
        q_resp = QueryResponse(
            query_id="Q-REV",
            tool_name="get_replenishment_reviews",
            domain="operations",
            metadata=meta,
            table=tbl,
        )
        step = CopilotStepResult(
            step_id="STEP-REV",
            tool_name="get_replenishment_reviews",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-REV",
            request_id="REQ-REV",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-REV",
            request_id="REQ-REV",
            state=CopilotState.COMPLETED,
            normalized_question="Review replenishment proposals.",
            execution_result=exec_res,
        )

        exp = build_operational_review_explanation(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.governance.approval_required is True
        assert exp.governance.execution_allowed is False
        assert "1 operational proposals" in exp.summary or "1 candidate review" in exp.key_findings[0].statement
