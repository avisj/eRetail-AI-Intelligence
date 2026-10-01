import numpy as np
from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import (
    CopilotExecutionResult,
    CopilotInterpretation,
    CopilotResponse,
    CopilotStepResult,
)
from commerce_ai.copilot.service import CopilotService
from commerce_ai.query_contracts.enums import BusinessDomain, QueryIntent
from commerce_ai.explanations.enums import (
    EvidenceType,
    ExplanationConfidence,
    ExplanationStatus,
    ExplanationType,
    ProvenanceType,
)
from commerce_ai.explanations.service import ExplanationService
from commerce_ai.forecasting.base import ForecastMetadata, ForecastOutput
from commerce_ai.query_layer.schemas import MetricResult, QueryMetadata, QueryResponse
from commerce_ai.query_layer.service import QueryLayerService


class TestForecastingExplanations:
    def test_forecast_explanation_labeled_as_model(self):
        out = ForecastOutput(
            sku_id="SKU_01",
            warehouse_id="WH_01",
            forecast_dates=["2026-07-01", "2026-07-02"],
            forecast_units=np.array([15.0, 16.0]),
            model_name="MovingAverage",
            metadata=ForecastMetadata(model_name="MovingAverage", model_version="1.0.0"),
        )
        out.records = out.to_records()
        ql = QueryLayerService.from_sample_data()
        ql.forecast_output = out
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process(
            "Show the demand forecast.",
            as_of_date="2026-06-30",
        )
        assert copilot_resp.execution_result is not None

        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.FORECAST_EXPLANATION
        assert exp.provenance == ProvenanceType.MODEL_BASED

        # Check forecast honesty
        for finding in exp.key_findings:
            assert finding.evidence_type == EvidenceType.MODEL_PROJECTED
            assert any(
                w in finding.statement.lower()
                for w in ["forecast", "project", "estimate", "model", "expected"]
            )

        # Check limitations attached
        assert any("projection" in lim.lower() or "statistical" in lim.lower() for lim in exp.limitations)

    def test_forecast_accuracy_explanation(self):
        meta = QueryMetadata(
            query_id="Q-ACC",
            tool_name="get_forecast_accuracy",
            domain="forecasting",
            generated_as_of="2026-06-30",
            source_engine="commerce_ai.analytics.metrics",
        )
        metrics = [
            MetricResult(
                metric_name="wape",
                display_name="Weighted Absolute Percentage Error (WAPE)",
                value=0.152,
                unit="ratio",
                source="ForecastMetrics",
            ),
            MetricResult(
                metric_name="mae",
                display_name="Mean Absolute Error (MAE)",
                value=4.5,
                unit="units",
                source="ForecastMetrics",
            ),
        ]
        q_resp = QueryResponse(
            query_id="Q-ACC",
            tool_name="get_forecast_accuracy",
            domain="forecasting",
            metadata=meta,
            metrics=metrics,
        )
        step = CopilotStepResult(
            step_id="STEP-ACC",
            tool_name="get_forecast_accuracy",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-ACC",
            request_id="REQ-ACC",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        interp = CopilotInterpretation(
            interpretation_id="INTERP-ACC",
            normalized_question="How accurate is the forecast?",
            intent=QueryIntent.FORECAST_ACCURACY,
            domain=BusinessDomain.FORECASTING,
        )
        copilot_resp = CopilotResponse(
            response_id="RESP-ACC",
            request_id="REQ-ACC",
            state=CopilotState.COMPLETED,
            interpretation=interp,
            normalized_question="How accurate is the forecast?",
            execution_result=exec_res,
        )

        explanation_service = ExplanationService()
        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.FORECAST_EXPLANATION
        assert exp.provenance == ProvenanceType.MODEL_BASED
