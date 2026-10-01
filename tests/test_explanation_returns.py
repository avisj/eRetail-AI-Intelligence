"""Unit tests for Returns Explanations (Phase 7D)."""

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
from commerce_ai.query_contracts.enums import BusinessDomain, QueryIntent, RequestedOutput
from commerce_ai.explanations.enums import (
    EvidenceType,
    ExplanationStatus,
    ExplanationType,
    ProvenanceType,
)
from commerce_ai.explanations.service import ExplanationService
from commerce_ai.query_layer.schemas import (
    CalculationStatus,
    MetricResult,
    QueryMetadata,
    QueryResponse,
    TableResult,
    TimeSeriesPoint,
    TimeSeriesResult,
)
from commerce_ai.query_layer.service import QueryLayerService


class TestReturnsExplanations:
    def test_return_rate_explanation(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process(
            "What is our return rate?",
            as_of_date="2026-06-30",
        )
        assert copilot_resp.execution_result is not None

        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.key_findings) > 0

    def test_return_reasons_breakdown_explanation(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process(
            "What are the main return reasons?",
            as_of_date="2026-06-30",
        )
        assert copilot_resp.execution_result is not None

        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.BREAKDOWN_EXPLANATION

    def test_return_trend_explanation(self):
        meta = QueryMetadata(
            query_id="Q-RET-TR",
            tool_name="get_return_trend",
            domain="returns",
            generated_as_of="2026-06-30",
            source_engine="commerce_ai.returns",
        )
        points = [
            TimeSeriesPoint(date="2026-06-01", value=12.0, metric_name="return_count"),
            TimeSeriesPoint(date="2026-06-15", value=18.0, metric_name="return_count"),
        ]
        ts = TimeSeriesResult(
            metric_name="return_count",
            display_name="Return Count Trend",
            time_grain="biweekly",
            points=points,
            metadata=meta,
        )
        q_resp = QueryResponse(
            query_id="Q-RET-TR",
            tool_name="get_return_trend",
            domain="returns",
            metadata=meta,
            time_series=ts,
        )
        step = CopilotStepResult(
            step_id="S-RET-TR",
            tool_name="get_return_trend",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-RET-TR",
            request_id="REQ-RET-TR",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        interp = CopilotInterpretation(
            interpretation_id="I-RET-TR",
            normalized_question="Show return trend",
            intent=QueryIntent.RETURN_ANALYSIS,
            domain=BusinessDomain.RETURNS,
            requested_output=RequestedOutput.TIME_SERIES,
        )
        resp = CopilotResponse(
            response_id="RESP-RET-TR",
            request_id="REQ-RET-TR",
            state=CopilotState.COMPLETED,
            interpretation=interp,
            normalized_question="Show return trend",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.TREND_EXPLANATION
        assert len(exp.trends) == 1

    def test_return_anomalies_explanation(self):
        meta = QueryMetadata(
            query_id="Q-RET-ANOM",
            tool_name="get_return_anomalies",
            domain="returns",
            generated_as_of="2026-06-30",
            source_engine="commerce_ai.returns",
        )
        table = TableResult(
            columns=["return_date", "sku_id", "z_score", "anomaly_severity"],
            rows=[
                {"return_date": "2026-06-12", "sku_id": "SKU_01", "z_score": 3.4, "anomaly_severity": "HIGH"},
            ],
            total_rows=1,
            metadata=meta,
        )
        q_resp = QueryResponse(
            query_id="Q-RET-ANOM",
            tool_name="get_return_anomalies",
            domain="returns",
            metadata=meta,
            table=table,
        )
        step = CopilotStepResult(
            step_id="S-RET-ANOM",
            tool_name="get_return_anomalies",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-RET-ANOM",
            request_id="REQ-RET-ANOM",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-RET-ANOM",
            request_id="REQ-RET-ANOM",
            state=CopilotState.COMPLETED,
            normalized_question="Show return anomalies",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert any("HIGH" in ev.notes for ev in exp.supporting_evidence)

    def test_return_risk_distinguishes_predicted_risk_from_observed_rate(self):
        meta = QueryMetadata(
            query_id="Q-RET-RISK",
            tool_name="get_return_risk",
            domain="returns",
            generated_as_of="2026-06-30",
            source_engine="commerce_ai.returns",
        )
        metrics = [
            MetricResult(
                metric_name="predicted_return_rate",
                display_name="Predicted Return Rate",
                value=0.082,
                unit="ratio",
                source="ReturnRiskModel",
            ),
        ]
        q_resp = QueryResponse(
            query_id="Q-RET-RISK",
            tool_name="get_return_risk",
            domain="returns",
            metadata=meta,
            metrics=metrics,
        )
        step = CopilotStepResult(
            step_id="S-RET-RISK",
            tool_name="get_return_risk",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-RET-RISK",
            request_id="REQ-RET-RISK",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-RET-RISK",
            request_id="REQ-RET-RISK",
            state=CopilotState.COMPLETED,
            normalized_question="What is the return risk?",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        # Must attach return risk limitations
        assert any("return" in lim.lower() for lim in exp.limitations)

    def test_return_summary_kpi_explanation(self):
        meta = QueryMetadata(
            query_id="Q-RET-SUM",
            tool_name="get_return_summary",
            domain="returns",
            generated_as_of="2026-06-30",
            currency="USD",
            source_engine="commerce_ai.returns",
        )
        metrics = [
            MetricResult(metric_name="total_return_events", display_name="Total Return Events", value=14, unit="events"),
            MetricResult(metric_name="return_rate", display_name="Return Rate", value=3.2, unit="%"),
        ]
        q_resp = QueryResponse(
            query_id="Q-RET-SUM",
            tool_name="get_return_summary",
            domain="returns",
            metadata=meta,
            metrics=metrics,
        )
        step = CopilotStepResult(
            step_id="S-RET-SUM",
            tool_name="get_return_summary",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-RET-SUM",
            request_id="REQ-RET-SUM",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-RET-SUM",
            request_id="REQ-RET-SUM",
            state=CopilotState.COMPLETED,
            normalized_question="Show return summary",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.key_findings) == 2
