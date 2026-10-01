"""Canonical Enums for Copilot Reasoning & Orchestration (Phase 7C).

Defines lifecycle states, step execution statuses, interpretation sources,
confidence levels, and provenance sources for the Copilot orchestration pipeline.
"""

from __future__ import annotations

from enum import Enum


class CopilotState(str, Enum):
    """Lifecycle state of a Copilot request as it transitions through the orchestration pipeline."""
    RECEIVED = "RECEIVED"
    NORMALIZED = "NORMALIZED"
    INTERPRETED = "INTERPRETED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    CONTRACT_BUILT = "CONTRACT_BUILT"
    VALIDATED = "VALIDATED"
    PLANNED = "PLANNED"
    READY = "READY"
    EXECUTING = "EXECUTING"
    PARTIAL = "PARTIAL"
    COMPLETED = "COMPLETED"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"


class StepStatus(str, Enum):
    """Execution status of an individual Copilot reasoning or tool execution step."""
    PENDING = "PENDING"
    READY = "READY"
    BLOCKED = "BLOCKED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    SKIPPED = "SKIPPED"


class PlanExecutionStatus(str, Enum):
    """Overall execution completion status of a Copilot execution plan."""
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNAVAILABLE = "UNAVAILABLE"


class InterpretationSource(str, Enum):
    """Provenance source of a question interpretation."""
    CANONICAL_TEMPLATE = "CANONICAL_TEMPLATE"
    DETERMINISTIC_RULE = "DETERMINISTIC_RULE"
    USER_PROVIDED_CONTEXT = "USER_PROVIDED_CONTEXT"
    FALLBACK = "FALLBACK"


class InterpretationConfidence(str, Enum):
    """Confidence level of intent and entity interpretation (strictly linguistic/rule confidence, not business correctness)."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ClarificationSeverity(str, Enum):
    """Severity level of an ambiguity or missing parameter clarification requirement."""
    INFO = "INFO"
    WARNING = "WARNING"
    BLOCKING = "BLOCKING"


class CopilotProvenanceSource(str, Enum):
    """Pedigree source for evidence and orchestration outputs."""
    DETERMINISTIC_RULE = "DETERMINISTIC_RULE"
    PHASE_7B_CONTRACT = "PHASE_7B_CONTRACT"
    PHASE_7B_PLAN = "PHASE_7B_PLAN"
    PHASE_7A_TOOL = "PHASE_7A_TOOL"
    USER_PROVIDED_CONTEXT = "USER_PROVIDED_CONTEXT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
