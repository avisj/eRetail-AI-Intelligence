"""Deterministic Confidence Evaluation for Business Explanations (Phase 7D).

Assigns structured ExplanationConfidence based strictly on empirical evidence completeness.
Does NOT compute arbitrary or manufactured probabilistic percentages.
"""

from __future__ import annotations

from typing import List, Optional

from commerce_ai.copilot.enums import PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import CopilotExecutionResult
from commerce_ai.explanations.enums import (
    EvidenceType,
    ExplanationConfidence,
    ProvenanceType,
)
from commerce_ai.explanations.schemas import ExplanationEvidence
from commerce_ai.query_layer.schemas import CalculationStatus


def evaluate_evidence_confidence(
    evidence_items: List[ExplanationEvidence],
    execution_result: Optional[CopilotExecutionResult] = None,
) -> ExplanationConfidence:
    """Deterministically determine the overall explanation confidence from evidence support."""
    if not evidence_items:
        return ExplanationConfidence.INSUFFICIENT

    # Check for execution result failures
    if execution_result:
        if execution_result.status == PlanExecutionStatus.FAILED:
            return ExplanationConfidence.INSUFFICIENT
        if execution_result.status == PlanExecutionStatus.UNAVAILABLE:
            return ExplanationConfidence.INSUFFICIENT
        if execution_result.status == PlanExecutionStatus.PARTIAL:
            # Check if any step succeeded
            successful_steps = [
                s for s in execution_result.step_results if s.status == StepStatus.COMPLETED
            ]
            if not successful_steps:
                return ExplanationConfidence.INSUFFICIENT
            return ExplanationConfidence.LOW

    # Check if any evidence is purely insufficient
    has_insufficient = any(e.provenance == ProvenanceType.INSUFFICIENT_DATA for e in evidence_items)
    valid_evidence = [e for e in evidence_items if e.provenance != ProvenanceType.INSUFFICIENT_DATA]

    if not valid_evidence:
        return ExplanationConfidence.INSUFFICIENT

    # Check for partial data flags or warnings in underlying responses
    has_partial_status = False
    if execution_result:
        for s in execution_result.step_results:
            if s.response and s.response.status in (
                CalculationStatus.PARTIAL_DATA,
                CalculationStatus.CURRENCY_INCONSISTENCY,
                CalculationStatus.EMPTY_RESULT,
            ):
                has_partial_status = True
                break

    if has_partial_status:
        return ExplanationConfidence.LOW

    if has_insufficient:
        return ExplanationConfidence.MEDIUM

    # Direct observed facts with non-null values
    all_observed = all(
        e.evidence_type in (EvidenceType.DIRECT_OBSERVED, EvidenceType.DETERMINISTIC_DERIVED)
        for e in valid_evidence
    )
    has_values = any(e.value is not None for e in valid_evidence)

    if all_observed and has_values:
        return ExplanationConfidence.HIGH

    # Model projected or statistical derivations
    if any(e.evidence_type in (EvidenceType.MODEL_PROJECTED, EvidenceType.STATISTICAL_DERIVED) for e in valid_evidence):
        return ExplanationConfidence.MEDIUM

    return ExplanationConfidence.MEDIUM
