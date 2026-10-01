"""Controlled Conversation Context Module (Phase 7E).

Provides strongly-typed, auditable, non-authoritative conversational state tracking:
- Strict separation between Authoritative Business Truth and ephemeral conversation memory
- Scope management (TURN, QUERY, CONVERSATION, SESSION)
- In-memory thread-safe store with zero external database dependencies
- Deterministic context resolution (inheritance, overrides, corrections, time shifting)
- Lineage provenance tracking and governance safety enforcement
"""

from commerce_ai.context.enums import (
    CompatibilityStatus,
    ConflictResolution,
    ContextCategory,
    ContextConfidence,
    ContextEvictionPolicy,
    ContextProvenance,
    ContextScope,
)
from commerce_ai.context.governance import (
    enforce_non_authoritative_memory,
    sanitize_input_for_injection,
    validate_context_governance,
)
from commerce_ai.context.lineage import (
    ContextLineageEvent,
    ContextLineageTracker,
)
from commerce_ai.context.memory import (
    ConversationContextStore,
    InMemoryConversationContextStore,
)
from commerce_ai.context.resolver import ContextResolver
from commerce_ai.context.schemas import (
    AmbiguityStateValue,
    ContextGovernance,
    ContextItem,
    ContextResolutionResult,
    ConversationContextSnapshot,
    CorrectionEvent,
    EntityContextValue,
    EvidenceReferenceValue,
    FilterContextValue,
    ResultReferenceValue,
    SessionContextValue,
    TimeRangeContextValue,
)
from commerce_ai.context.scope import ScopeManager
from commerce_ai.context.service import ConversationContextService

__all__ = [
    # Enums
    "CompatibilityStatus",
    "ConflictResolution",
    "ContextCategory",
    "ContextConfidence",
    "ContextEvictionPolicy",
    "ContextProvenance",
    "ContextScope",
    # Governance
    "ContextGovernance",
    "validate_context_governance",
    "sanitize_input_for_injection",
    "enforce_non_authoritative_memory",
    # Schemas
    "ContextItem",
    "SessionContextValue",
    "EntityContextValue",
    "TimeRangeContextValue",
    "FilterContextValue",
    "EvidenceReferenceValue",
    "ResultReferenceValue",
    "CorrectionEvent",
    "AmbiguityStateValue",
    "ConversationContextSnapshot",
    "ContextResolutionResult",
    # Scope & Lineage
    "ScopeManager",
    "ContextLineageEvent",
    "ContextLineageTracker",
    # Memory
    "ConversationContextStore",
    "InMemoryConversationContextStore",
    # Resolver & Service
    "ContextResolver",
    "ConversationContextService",
]
