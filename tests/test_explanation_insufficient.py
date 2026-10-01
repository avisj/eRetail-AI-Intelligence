"""Dedicated tests for Insufficient Data and Unavailable Capabilities (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import (
    CopilotExecutionResult,
    CopilotInterpretation,
    CopilotResponse,
    CopilotStepResult,
)
from commerce_ai.copilot.service import CopilotService
from commerce_ai.explanations.enums import (
    ExplanationConfidence,
    ExplanationStatus,
    ExplanationType,
    ProvenanceType,
)
from commerce_ai.explanations.service import ExplanationService
from commerce_ai.query_contracts.enums import BusinessDomain, QueryIntent
from commerce_ai.query_layer.schemas import (
    CalculationStatus,
    MetricResult,
    QueryMetadata,
    QueryResponse,
)


class TestExplanationInsufficientData:
    @pytest.fixture
    def service(self) -> ExplanationService:
        return ExplanationService()

    def test_missing_financial_cost_produces_insufficient_data(self, service: ExplanationService):
        meta = QueryMetadata(
            query_id="Q-EMPTY-COST",
            tool_name="get_margin_summary",
            domain="financial",
            generated_as_of="2026-06-30",
            calculation_status=CalculationStatus.EMPTY_RESULT,
        )
        q_resp = QueryResponse(
            query_id="Q-EMPTY-COST",
            tool_name="get_margin_summary",
            domain="financial",
            status=CalculationStatus.EMPTY_RESULT,
            metadata=meta,
            error_message="Cost data unavailable for margin evaluation.",
        )
        step = CopilotStepResult(
            step_id="S-1",
            tool_name="get_margin_summary",
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
            normalized_question="What is our gross margin?",
            execution_result=exec_res,
        )
        exp = service.explain(resp)
        assert exp.status == ExplanationStatus.INSUFFICIENT_DATA
        assert exp.confidence == ExplanationConfidence.INSUFFICIENT
        assert len(exp.key_findings) == 0 or any("insufficient" in f.statement.lower() for f in exp.key_findings)

    def test_missing_forecast_does_not_fabricate_zero(self, service: ExplanationService):
        meta = QueryMetadata(
            query_id="Q-UNAV-FC",
            tool_name="get_forecast",
            domain="forecasting",
            generated_as_of="2026-06-30",
            calculation_status=CalculationStatus.UNAVAILABLE,
        )
        q_resp = QueryResponse(
            query_id="Q-UNAV-FC",
            tool_name="get_forecast",
            domain="forecasting",
            status=CalculationStatus.UNAVAILABLE,
            metadata=meta,
            error_message="Forecast models not trained.",
        )
        step = CopilotStepResult(
            step_id="S-FC",
            tool_name="get_forecast",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-UNAV",
            request_id="REQ-UNAV",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-UNAV",
            request_id="REQ-UNAV",
            state=CopilotState.COMPLETED,
            normalized_question="Show demand forecast",
            execution_result=exec_res,
        )
        exp = service.explain(resp)
        assert exp.status == ExplanationStatus.UNAVAILABLE
        assert exp.explanation_type == ExplanationType.UNAVAILABLE_EXPLANATION
        # Assert forecast was not fabricated as 0
        full_text = exp.headline + " " + exp.summary + " " + " ".join(f.statement for f in exp.key_findings)
        assert "forecast is 0" not in full_text.lower()
        assert "0.0" not in full_text

    def test_empty_step_results_yields_insufficient_data(self, service: ExplanationService):
        exec_res = CopilotExecutionResult(
            result_id="RES-NO-STEPS",
            request_id="REQ-NO-STEPS",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[],
        )
        resp = CopilotResponse(
            response_id="RESP-NO-STEPS",
            request_id="REQ-NO-STEPS",
            state=CopilotState.COMPLETED,
            normalized_question="Any data?",
            execution_result=exec_res,
        )
        exp = service.explain(resp)
        assert exp.status == ExplanationStatus.INSUFFICIENT_DATA
        assert exp.confidence == ExplanationConfidence.INSUFFICIENT

    def test_data_quality_capability_unavailable(self, service: ExplanationService):
        interp = CopilotInterpretation(
            interpretation_id="INTERP-DQ",
            normalized_question="Check data quality completeness",
            intent=QueryIntent.DATA_QUALITY_ANALYSIS,
            domain=BusinessDomain.OPERATIONS,
        )
        resp = CopilotResponse(
            response_id="RESP-DQ",
            request_id="REQ-DQ",
            state=CopilotState.UNAVAILABLE,
            interpretation=interp,
            normalized_question="Check data quality completeness",
        )
        exp = service.explain(resp)
        assert exp.status == ExplanationStatus.UNAVAILABLE
        assert exp.explanation_type == ExplanationType.UNAVAILABLE_EXPLANATION
        assert "unavailable" in exp.headline.lower()

    def test_failed_plan_execution_honestly_reports_insufficiency(self, service: ExplanationService):
        exec_res = CopilotExecutionResult(
            result_id="RES-FAIL",
            request_id="REQ-FAIL",
            status=PlanExecutionStatus.FAILED,
            step_results=[],
        )
        resp = CopilotResponse(
            response_id="RESP-FAIL",
            request_id="REQ-FAIL",
            state=CopilotState.FAILED,
            normalized_question="Execute broad analysis",
            execution_result=exec_res,
        )
        exp = service.explain(resp)
        assert exp.status == ExplanationStatus.INSUFFICIENT_DATA
        assert exp.confidence == ExplanationConfidence.INSUFFICIENT
        assert "cannot be completed" in exp.summary.lower()

    def test_insufficient_data_governance_invariants(self, service: ExplanationService):
        resp = CopilotResponse(
            response_id="RESP-INSUF",
            request_id="REQ-INSUF",
            state=CopilotState.CLARIFICATION_REQUIRED,
            normalized_question="Broad question",
        )
        exp = service.explain(resp)
        assert exp.governance.read_only is True
        assert exp.governance.action_execution is False
        assert exp.governance.no_ranking_enforced is True
        assert exp.governance.no_causality_invented is True
        assert exp.governance.no_recommendation_generated is True
