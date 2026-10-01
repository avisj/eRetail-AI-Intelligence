"""Core Pydantic Schemas for Business Explanations (Phase 7D).

Provides validated data structures for:
- ExplanationEvidence: Granular factual evidence extracted from Phase 7A/7C outputs
- Finding: Traceable business statement linked directly to supporting evidence
- ExplanationGovernance: Immutable policy constraints (read-only, zero causality, no rankings)
- BusinessExplanation: Full structured explanation container
- AuditResult: Verification outcome from semantic claim auditing
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from commerce_ai.explanations.enums import (
    ClaimCategory,
    EvidenceType,
    ExplanationConfidence,
    ExplanationStatus,
    ExplanationType,
    ProvenanceType,
)


class ExplanationEvidence(BaseModel):
    """Granular evidence item supporting a business explanation claim."""
    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(description="Deterministic evidence identifier")
    source_step_id: str = Field(description="Originating CopilotStep ID")
    source_tool: str = Field(description="Phase 7A canonical tool name")
    source_engine: str = Field(description="Upstream calculation engine or service")
    metric: Optional[str] = Field(default=None, description="Observed metric name")
    value: Optional[Union[float, int, str]] = Field(default=None, description="Recorded metric value")
    comparison_value: Optional[Union[float, int, str]] = Field(default=None, description="Comparison or baseline value")
    unit: Optional[str] = Field(default=None, description="Measurement unit (e.g. USD, %, units, days)")
    currency: Optional[str] = Field(default=None, description="Currency ISO code if monetary")
    dimension: Optional[str] = Field(default=None, description="Dimension name if sliced")
    entity: Optional[str] = Field(default=None, description="Entity or dimension value (e.g. WH_01, ONLINE)")
    time_range: Optional[str] = Field(default=None, description="Time range or as-of date")
    provenance: ProvenanceType = Field(default=ProvenanceType.PHASE_7A_OBSERVED, description="Pedigree of evidence")
    evidence_type: EvidenceType = Field(default=EvidenceType.DIRECT_OBSERVED, description="Empirical nature of evidence")
    notes: Optional[str] = Field(default=None, description="Contextual note or calculation details")


class Finding(BaseModel):
    """An individual factual finding linked directly to supporting evidence."""
    model_config = ConfigDict(frozen=True)

    finding_id: str = Field(description="Deterministic finding identifier")
    statement: str = Field(description="Human-readable business observation")
    evidence: List[ExplanationEvidence] = Field(default_factory=list, description="Linked supporting evidence items")
    provenance: ProvenanceType = Field(description="Provenance tier of the statement")
    evidence_type: EvidenceType = Field(description="Empirical type of the statement")
    confidence: ExplanationConfidence = Field(description="Evidence support confidence tier")
    claim_category: ClaimCategory = Field(default=ClaimCategory.OBSERVED_FACT, description="Category of the claim")
    supporting_metric: Optional[str] = Field(default=None, description="Primary supporting metric identifier")
    source_step: Optional[str] = Field(default=None, description="Originating execution step ID")
    source_tool: Optional[str] = Field(default=None, description="Originating Phase 7A tool name")


class ExplanationGovernance(BaseModel):
    """Platform governance constraints strictly enforced across all business explanations."""
    model_config = ConfigDict(frozen=True)

    read_only: bool = Field(default=True, description="Strictly True. Explanations never mutate state.")
    action_execution: bool = Field(default=False, description="Strictly False. Autonomous action execution is prohibited.")
    execution_allowed: bool = Field(default=False, description="Strictly False. Business mutations (PO, transfer, pricing) are forbidden.")
    approval_required: bool = Field(default=False, description="True if operational review items require human authorization.")
    no_ranking_enforced: bool = Field(default=True, description="Strictly True. Subjective entity rankings (top-N, winners) are forbidden.")
    no_decision_selection_enforced: bool = Field(default=True, description="Strictly True. Explanations never select a decision option.")
    no_causality_invented: bool = Field(default=True, description="Strictly True. Explanations never claim causality without causal models.")
    no_recommendation_generated: bool = Field(default=True, description="Strictly True. Explanations explain existing data, never invent actions.")


class BusinessExplanation(BaseModel):
    """Complete, structured, and auditable business explanation container."""
    model_config = ConfigDict(frozen=True)

    explanation_id: str = Field(description="Deterministic explanation identifier")
    status: ExplanationStatus = Field(description="Lifecycle status of explanation")
    explanation_type: ExplanationType = Field(description="Type of explanation generated")
    headline: str = Field(description="Concise, non-sensational headline statement")
    summary: str = Field(description="Executive business summary paragraph")
    key_findings: List[Finding] = Field(default_factory=list, description="Structured factual findings")
    supporting_evidence: List[ExplanationEvidence] = Field(default_factory=list, description="All underlying evidence records")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Key metric values dictionary")
    comparisons: List[Dict[str, Any]] = Field(default_factory=list, description="Temporal or baseline comparisons")
    trends: List[Dict[str, Any]] = Field(default_factory=list, description="Directional trend observations")
    risks: List[str] = Field(default_factory=list, description="Identified operational or financial risk notes")
    limitations: List[str] = Field(default_factory=list, description="Data, accounting, or causal limitations")
    data_quality_notes: List[str] = Field(default_factory=list, description="Data availability or completeness observations")
    provenance: ProvenanceType = Field(default=ProvenanceType.PHASE_7A_OBSERVED, description="Overall explanation provenance")
    confidence: ExplanationConfidence = Field(default=ExplanationConfidence.HIGH, description="Overall evidence support confidence")
    generated_from: Dict[str, str] = Field(default_factory=dict, description="Lineage pointers to request, response, template, and intent")
    governance: ExplanationGovernance = Field(default_factory=ExplanationGovernance, description="Enforced governance invariants")
    warnings: List[str] = Field(default_factory=list, description="Aggregated advisory warnings")


class AuditResult(BaseModel):
    """Verification outcome produced by the semantic claim auditor."""
    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(description="Whether the explanation passed all audit rules")
    passed_checks: List[str] = Field(default_factory=list, description="List of passed verification checks")
    violations: List[str] = Field(default_factory=list, description="List of detected policy or grounding violations")
    unsupported_claims: List[str] = Field(default_factory=list, description="Findings lacking supporting evidence")
    causality_warnings: List[str] = Field(default_factory=list, description="Detected ungrounded causal claims")
    ranking_violations: List[str] = Field(default_factory=list, description="Detected ranking or leaderboard violations")
    forecast_as_fact_warnings: List[str] = Field(default_factory=list, description="Detected forecasts stated as facts")
