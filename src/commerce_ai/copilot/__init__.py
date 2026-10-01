"""Copilot Reasoning & Orchestration Layer (Phase 7C).

Provides deterministic question intake, natural-language question normalization,
structured intent interpretation, ambiguity detection, Phase 7B query plan synthesis,
tool execution coordination against Phase 7A tools, failure isolation, and evidence collection.
"""

from __future__ import annotations

from commerce_ai.copilot.ambiguity import check_query_ambiguity
from commerce_ai.copilot.decomposition import decompose_and_plan
from commerce_ai.copilot.dependencies import DependencyGraph
from commerce_ai.copilot.enums import (
    ClarificationSeverity,
    CopilotProvenanceSource,
    CopilotState,
    InterpretationConfidence,
    InterpretationSource,
    PlanExecutionStatus,
    StepStatus,
)
from commerce_ai.copilot.execution import execute_plan
from commerce_ai.copilot.governance import (
    GovernanceViolation,
    enforce_decision_package_neutrality,
    validate_plan_governance,
    validate_request_governance,
)
from commerce_ai.copilot.interpreter import interpret_question
from commerce_ai.copilot.normalization import (
    detect_malicious_intent,
    normalize_question,
)
from commerce_ai.copilot.orchestration import CopilotOrchestrator
from commerce_ai.copilot.results import assemble_execution_result
from commerce_ai.copilot.schemas import (
    CopilotClarification,
    CopilotEvidence,
    CopilotExecutionPlan,
    CopilotExecutionResult,
    CopilotFailure,
    CopilotGovernance,
    CopilotInterpretation,
    CopilotRequest,
    CopilotResponse,
    CopilotStep,
    CopilotStepResult,
)
from commerce_ai.copilot.service import CopilotService

__all__ = [
    # Main Service & Orchestrator
    "CopilotService",
    "CopilotOrchestrator",
    # Enums
    "CopilotState",
    "StepStatus",
    "PlanExecutionStatus",
    "InterpretationSource",
    "InterpretationConfidence",
    "ClarificationSeverity",
    "CopilotProvenanceSource",
    # Schemas
    "CopilotRequest",
    "CopilotClarification",
    "CopilotInterpretation",
    "CopilotGovernance",
    "CopilotStep",
    "CopilotExecutionPlan",
    "CopilotEvidence",
    "CopilotFailure",
    "CopilotStepResult",
    "CopilotExecutionResult",
    "CopilotResponse",
    # Functions & Components
    "normalize_question",
    "detect_malicious_intent",
    "check_query_ambiguity",
    "interpret_question",
    "decompose_and_plan",
    "execute_plan",
    "assemble_execution_result",
    "validate_request_governance",
    "validate_plan_governance",
    "enforce_decision_package_neutrality",
    "DependencyGraph",
    "GovernanceViolation",
]
