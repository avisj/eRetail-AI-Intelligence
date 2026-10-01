"""Dedicated tests for Evidence-Support Confidence Evaluation (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import CopilotExecutionResult, CopilotResponse, CopilotStepResult
from commerce_ai.explanations.confidence import evaluate_evidence_confidence
from commerce_ai.explanations.enums import (
    EvidenceType,
    ExplanationConfidence,
    ProvenanceType,
)
from commerce_ai.explanations.schemas import ExplanationEvidence


class TestExplanationConfidence:
    def test_empty_evidence_yields_insufficient_confidence(self):
        conf = evaluate_evidence_confidence([], None)
        assert conf == ExplanationConfidence.INSUFFICIENT

    def test_failed_execution_yields_insufficient_confidence(self):
        exec_res = CopilotExecutionResult(
            result_id="RES-FAIL",
            request_id="REQ-FAIL",
            status=PlanExecutionStatus.FAILED,
            step_results=[],
        )
        ev = ExplanationEvidence(
            evidence_id="EV-1",
            source_step_id="S-1",
            source_tool="get_sales_summary",
            source_engine="commerce_ai.sales",
            metric="gross_revenue",
            value=1000.0,
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
        )
        conf = evaluate_evidence_confidence([ev], exec_res)
        assert conf == ExplanationConfidence.INSUFFICIENT

    def test_all_direct_observed_yields_high_confidence(self):
        exec_res = CopilotExecutionResult(
            result_id="RES-OK",
            request_id="REQ-OK",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[
                CopilotStepResult(
                    step_id="S-1",
                    tool_name="get_sales_summary",
                    status=StepStatus.COMPLETED,
                )
            ],
        )
        ev1 = ExplanationEvidence(
            evidence_id="EV-1",
            source_step_id="S-1",
            source_tool="get_sales_summary",
            source_engine="commerce_ai.sales",
            metric="gross_revenue",
            value=1000.0,
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
        )
        ev2 = ExplanationEvidence(
            evidence_id="EV-2",
            source_step_id="S-1",
            source_tool="get_sales_summary",
            source_engine="commerce_ai.sales",
            metric="net_revenue",
            value=950.0,
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
        )
        conf = evaluate_evidence_confidence([ev1, ev2], exec_res)
        assert conf == ExplanationConfidence.HIGH

    def test_model_projected_evidence_yields_medium_confidence(self):
        exec_res = CopilotExecutionResult(
            result_id="RES-OK",
            request_id="REQ-OK",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[
                CopilotStepResult(
                    step_id="S-1",
                    tool_name="get_forecast",
                    status=StepStatus.COMPLETED,
                )
            ],
        )
        ev = ExplanationEvidence(
            evidence_id="EV-1",
            source_step_id="S-1",
            source_tool="get_forecast",
            source_engine="commerce_ai.forecasting",
            metric="forecast_units",
            value=250.0,
            provenance=ProvenanceType.MODEL_BASED,
            evidence_type=EvidenceType.MODEL_PROJECTED,
        )
        conf = evaluate_evidence_confidence([ev], exec_res)
        assert conf == ExplanationConfidence.MEDIUM

    def test_partial_step_failure_yields_low_confidence(self):
        exec_res = CopilotExecutionResult(
            result_id="RES-PARTIAL",
            request_id="REQ-PARTIAL",
            status=PlanExecutionStatus.PARTIAL,
            step_results=[
                CopilotStepResult(
                    step_id="S-1",
                    tool_name="get_sales_summary",
                    status=StepStatus.COMPLETED,
                ),
                CopilotStepResult(
                    step_id="S-2",
                    tool_name="get_margin_summary",
                    status=StepStatus.FAILED,
                ),
            ],
        )
        ev = ExplanationEvidence(
            evidence_id="EV-1",
            source_step_id="S-1",
            source_tool="get_sales_summary",
            source_engine="commerce_ai.sales",
            metric="gross_revenue",
            value=1000.0,
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
        )
        conf = evaluate_evidence_confidence([ev], exec_res)
        assert conf == ExplanationConfidence.LOW

    def test_deterministic_confidence_assignment_no_randomness(self):
        exec_res = CopilotExecutionResult(
            result_id="RES-OK",
            request_id="REQ-OK",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[],
        )
        ev = ExplanationEvidence(
            evidence_id="EV-1",
            source_step_id="S-1",
            source_tool="get_sales_summary",
            source_engine="commerce_ai.sales",
            metric="units_sold",
            value=50.0,
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
        )
        results = [evaluate_evidence_confidence([ev], exec_res) for _ in range(10)]
        assert all(c == ExplanationConfidence.HIGH for c in results)
