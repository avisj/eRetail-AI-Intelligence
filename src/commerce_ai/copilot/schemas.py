"""Core Pydantic Schemas for Copilot Reasoning & Orchestration (Phase 7C).

Provides strongly typed, validated data structures for:
- CopilotRequest: Intake of user business questions
- CopilotInterpretation: Structured intent, domain, entity, and parameter representation
- CopilotClarification: Explicit clarification requirements when questions are ambiguous
- CopilotGovernance: Immutability, read-only guarantees, ranking prohibitions
- CopilotStep & CopilotExecutionPlan: Coordinated execution blueprint across Phase 7B plans
- CopilotEvidence & CopilotStepResult: Evidence preservation and result envelopes
- CopilotExecutionResult: Aggregated execution outcome, partial failure status, and timings
- CopilotResponse: End-to-end orchestration response
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from commerce_ai.copilot.enums import (
    ClarificationSeverity,
    CopilotProvenanceSource,
    CopilotState,
    InterpretationConfidence,
    InterpretationSource,
    PlanExecutionStatus,
    StepStatus,
)
from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    BusinessGrain,
    MetricIdentifier,
    QueryIntent,
    RequestedOutput,
    TimeGranularity,
)
from commerce_ai.query_contracts.schemas import (
    BusinessQueryContract,
    QueryPlan,
)
from commerce_ai.query_layer.schemas import QueryResponse


class CopilotGovernance(BaseModel):
    """Platform governance constraints strictly enforced across all Copilot workflows."""
    model_config = ConfigDict(frozen=True)

    read_only: bool = Field(default=True, description="Strictly True. Copilot cannot mutate data.")
    action_execution: bool = Field(default=False, description="Strictly False. Autonomous system action is prohibited.")
    execution_allowed: bool = Field(default=False, description="Strictly False. Business mutations (PO, transfer, pricing) are forbidden.")
    approval_required: bool = Field(default=False, description="True if operational review items require human authorization.")
    no_ranking_enforced: bool = Field(default=True, description="Strictly True. Subjective entity rankings (top-N, winners) are forbidden.")
    no_decision_selection_enforced: bool = Field(default=True, description="Strictly True. Copilot never autonomously selects a decision option.")


class CopilotRequest(BaseModel):
    """Inbound Copilot question request specification."""
    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Natural-language business question from the user")
    request_id: Optional[str] = Field(default=None, description="Deterministic or client-provided request identifier")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time reference date (YYYY-MM-DD)")
    currency: Optional[str] = Field(default=None, description="Target currency code (e.g. USD)")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Explicit caller-provided filters")
    requested_output: Optional[RequestedOutput] = Field(default=None, description="Explicitly requested output format")
    context: Dict[str, Any] = Field(default_factory=dict, description="Request-scoped caller context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operational metadata")


class CopilotClarification(BaseModel):
    """Specification of required clarification when a question is ambiguous or underspecified."""
    model_config = ConfigDict(frozen=True)

    clarification_id: str = Field(description="Deterministic clarification identifier")
    question: str = Field(description="Clarification prompt to present to the user")
    reason: str = Field(description="Technical rationale why clarification is necessary")
    missing_fields: List[str] = Field(default_factory=list, description="List of underspecified or missing fields")
    severity: ClarificationSeverity = Field(default=ClarificationSeverity.BLOCKING, description="Clarification severity")
    possible_interpretations: List[str] = Field(default_factory=list, description="Plausible interpretations offered to user")


class CopilotInterpretation(BaseModel):
    """Structured linguistic and business interpretation of a normalized question."""
    model_config = ConfigDict(frozen=True)

    interpretation_id: str = Field(description="Deterministic interpretation identifier")
    normalized_question: str = Field(description="Canonical normalized form of user question")
    intent: QueryIntent = Field(description="Target Phase 7B QueryIntent")
    domain: BusinessDomain = Field(description="Target primary Phase 7B BusinessDomain")
    domains: List[BusinessDomain] = Field(default_factory=list, description="Target domains for multi-domain inquiries")
    metrics: List[MetricIdentifier] = Field(default_factory=list, description="Identified business metrics")
    dimensions: List[BusinessDimension] = Field(default_factory=list, description="Identified slice-and-dice dimensions")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Extracted entity/dimensional filters")
    time_range: Optional[Dict[str, Any]] = Field(default=None, description="Extracted time range bounds or presets")
    time_granularity: TimeGranularity = Field(default=TimeGranularity.NONE, description="Extracted temporal reporting frequency")
    requested_grain: BusinessGrain = Field(default=BusinessGrain.PORTFOLIO, description="Extracted business aggregation entity")
    requested_output: RequestedOutput = Field(default=RequestedOutput.KPI, description="Extracted response format")
    explanation_context: Optional[str] = Field(default=None, description="Diagnostic explanation context for why-style queries")
    is_investigation: bool = Field(default=False, description="True if question is a diagnostic why-style query")
    is_multi_domain: bool = Field(default=False, description="True if inquiry spans multiple business domains")
    confidence: InterpretationConfidence = Field(default=InterpretationConfidence.HIGH, description="Rule-based interpretation confidence")
    source: InterpretationSource = Field(default=InterpretationSource.DETERMINISTIC_RULE, description="Provenance source of interpretation")
    clarification: Optional[CopilotClarification] = Field(default=None, description="Associated clarification if ambiguous")
    matched_template_id: Optional[str] = Field(default=None, description="Phase 7B QuestionTemplate ID if matched")


class CopilotStep(BaseModel):
    """An individual execution step within a Copilot reasoning plan."""
    model_config = ConfigDict(frozen=True)

    step_id: str = Field(description="Unique step identifier (e.g. STEP-1)")
    sequence: int = Field(description="Execution sequence order (1-indexed)")
    contract_id: str = Field(description="Associated Phase 7B BusinessQueryContract ID")
    plan_id: str = Field(description="Associated Phase 7B QueryPlan ID")
    tool_name: str = Field(description="Phase 7A canonical query tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Parameters passed to Phase 7A tool")
    purpose: str = Field(description="Business explanation of why this step is executed")
    required: bool = Field(default=True, description="Whether failure of this step blocks subsequent steps")
    dependencies: List[str] = Field(default_factory=list, description="Step IDs that must complete before this step")
    status: StepStatus = Field(default=StepStatus.PENDING, description="Current lifecycle state of the step")


class CopilotExecutionPlan(BaseModel):
    """Composite execution plan coordinating one or more Phase 7B query contracts and plans."""
    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(description="Deterministic execution plan fingerprint")
    request_id: str = Field(description="Reference to source CopilotRequest")
    steps: List[CopilotStep] = Field(default_factory=list, description="Ordered execution steps")
    execution_order: List[str] = Field(default_factory=list, description="List of step IDs in execution order")
    dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="Dependency graph mapping step_id to prerequisites")
    governance: CopilotGovernance = Field(default_factory=CopilotGovernance, description="Enforced governance invariants")
    contracts: List[BusinessQueryContract] = Field(default_factory=list, description="Underlying Phase 7B query contracts")
    query_plans: List[QueryPlan] = Field(default_factory=list, description="Underlying Phase 7B query plans")
    state: CopilotState = Field(default=CopilotState.PLANNED, description="Current plan status")


class CopilotEvidence(BaseModel):
    """Structured evidence package extracted from tool execution results."""
    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(description="Deterministic evidence record ID")
    step_id: str = Field(description="Source execution step ID")
    tool_name: str = Field(description="Executing Phase 7A tool name")
    source_engine: str = Field(description="Underlying commerce_ai calculation engine")
    metrics: List[str] = Field(default_factory=list, description="Metric names present in evidence")
    entity_count: int = Field(default=0, description="Count of data records or entities retrieved")
    as_of_date: Optional[str] = Field(default=None, description="Effective point-in-time date of evidence")
    currency: Optional[str] = Field(default=None, description="Currency of financial values")
    data_summary: Dict[str, Any] = Field(default_factory=dict, description="High-level structured data summary")
    provenance: CopilotProvenanceSource = Field(default=CopilotProvenanceSource.PHASE_7A_TOOL, description="Evidence pedigree")


class CopilotFailure(BaseModel):
    """Structured failure representation for isolated step execution errors."""
    model_config = ConfigDict(frozen=True)

    step_id: str = Field(description="Step ID that encountered failure")
    tool_name: str = Field(description="Tool that encountered failure")
    error_code: str = Field(description="Standardized error code")
    error_message: str = Field(description="Human-readable error description")
    fatal: bool = Field(default=False, description="Whether this failure halted the entire plan")


class CopilotStepResult(BaseModel):
    """Execution outcome of a single Copilot reasoning step."""
    model_config = ConfigDict(frozen=True)

    step_id: str = Field(description="Step identifier")
    tool_name: str = Field(description="Executed tool name")
    status: StepStatus = Field(description="Final execution status")
    response: Optional[QueryResponse] = Field(default=None, description="Underlying Phase 7A response envelope")
    evidence: Optional[CopilotEvidence] = Field(default=None, description="Extracted evidence package")
    execution_time_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    failure: Optional[CopilotFailure] = Field(default=None, description="Failure details if step did not succeed")
    warnings: List[str] = Field(default_factory=list, description="Advisory execution warnings")


class CopilotExecutionResult(BaseModel):
    """Aggregated execution results across all planned reasoning steps."""
    model_config = ConfigDict(frozen=True)

    result_id: str = Field(description="Unique deterministic result identifier")
    request_id: str = Field(description="Source request identifier")
    status: PlanExecutionStatus = Field(description="Overall execution status (COMPLETE, PARTIAL, FAILED, UNAVAILABLE)")
    step_results: List[CopilotStepResult] = Field(default_factory=list, description="Individual step outcomes")
    evidence_chain: List[CopilotEvidence] = Field(default_factory=list, description="Preserved evidence lineage")
    failures: List[CopilotFailure] = Field(default_factory=list, description="All encountered failures")
    total_execution_time_ms: float = Field(default=0.0, description="Total execution duration in milliseconds")


class CopilotResponse(BaseModel):
    """Complete end-to-end response returned by the CopilotService."""
    model_config = ConfigDict(frozen=True)

    response_id: str = Field(description="Deterministic response identifier")
    request_id: str = Field(description="Source request identifier")
    state: CopilotState = Field(description="Final lifecycle state of Copilot workflow")
    normalized_question: str = Field(description="Normalized question text")
    interpretation: Optional[CopilotInterpretation] = Field(default=None, description="Structured interpretation")
    clarification: Optional[CopilotClarification] = Field(default=None, description="Clarification if question was ambiguous")
    execution_plan: Optional[CopilotExecutionPlan] = Field(default=None, description="Executed or synthesized plan")
    execution_result: Optional[CopilotExecutionResult] = Field(default=None, description="Execution outcome and evidence chain")
    governance: CopilotGovernance = Field(default_factory=CopilotGovernance, description="Enforced governance invariants")
    warnings: List[str] = Field(default_factory=list, description="Aggregated advisory warnings")
    errors: List[str] = Field(default_factory=list, description="Aggregated error messages")
