"""Business Explanation Layer (Phase 7D).

Converts structured analytical results produced by Phase 7A, 7B, and 7C into
grounded, auditable, and human-readable business explanations.
"""

from __future__ import annotations

from commerce_ai.explanations.auditor import ClaimAuditor, default_auditor
from commerce_ai.explanations.confidence import evaluate_evidence_confidence
from commerce_ai.explanations.enums import (
    ClaimCategory,
    EvidenceType,
    ExplanationConfidence,
    ExplanationStatus,
    ExplanationType,
    ProvenanceType,
)
from commerce_ai.explanations.evidence import extract_evidence_from_step_result
from commerce_ai.explanations.governance import validate_explanation_governance
from commerce_ai.explanations.renderer import render_explanation_to_text
from commerce_ai.explanations.schemas import (
    AuditResult,
    BusinessExplanation,
    ExplanationEvidence,
    ExplanationGovernance,
    Finding,
)
from commerce_ai.explanations.service import ExplanationService

__all__ = [
    "ExplanationService",
    "BusinessExplanation",
    "Finding",
    "ExplanationEvidence",
    "ExplanationGovernance",
    "AuditResult",
    "ClaimAuditor",
    "default_auditor",
    "ExplanationType",
    "ExplanationStatus",
    "ProvenanceType",
    "EvidenceType",
    "ExplanationConfidence",
    "ClaimCategory",
    "extract_evidence_from_step_result",
    "evaluate_evidence_confidence",
    "validate_explanation_governance",
    "render_explanation_to_text",
]
