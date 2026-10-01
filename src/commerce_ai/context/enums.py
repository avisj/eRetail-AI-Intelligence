"""Enumerations for Controlled Conversation Context (Phase 7E).

Defines strongly-typed lifecycle, categorization, scoping, provenance,
confidence, compatibility, and conflict resolution enumerations for conversation context.
"""

from __future__ import annotations

from enum import Enum


class ContextCategory(str, Enum):
    """Semantic category of information stored in conversation context."""
    SESSION_CONTEXT = "SESSION_CONTEXT"          # Session-level metadata (user, tenant, currency)
    QUERY_CONTEXT = "QUERY_CONTEXT"              # Prior query structure (intent, domain, output)
    ENTITY_CONTEXT = "ENTITY_CONTEXT"            # Active business entities (SKU, Warehouse, Channel, etc.)
    FILTER_CONTEXT = "FILTER_CONTEXT"            # Active analytical filters (time range, grain, dimension)
    EVIDENCE_CONTEXT = "EVIDENCE_CONTEXT"        # Citations and evidence references from prior executions
    RESULT_CONTEXT = "RESULT_CONTEXT"            # Metadata references to prior result sets (not raw data)
    DECISION_CONTEXT = "DECISION_CONTEXT"        # Operational review / scenario state
    AMBIGUITY_CONTEXT = "AMBIGUITY_CONTEXT"      # Pending clarification state
    CORRECTION_CONTEXT = "CORRECTION_CONTEXT"    # User corrections and overrides


class ContextScope(str, Enum):
    """Lifetime and boundary scope of a context item."""
    TURN = "TURN"                    # Single turn only (immediate next question)
    QUERY = "QUERY"                  # Bound to current query plan decomposition
    CONVERSATION = "CONVERSATION"    # Multi-turn within same conversational topic
    SESSION = "SESSION"              # Across entire user session


class ContextProvenance(str, Enum):
    """Origin source of a context item for auditability and lineage."""
    EXPLICIT_USER = "EXPLICIT_USER"                          # Stated directly by user in prompt
    DERIVED_FROM_VALIDATED_QUERY = "DERIVED_FROM_VALIDATED_QUERY"  # Extracted from executed Phase 7B contract
    DERIVED_FROM_SYSTEM_RESULT = "DERIVED_FROM_SYSTEM_RESULT"      # Inferred from tool execution result
    USER_CORRECTION = "USER_CORRECTION"                      # Explicit user correction overriding prior value
    SYSTEM_DEFAULT = "SYSTEM_DEFAULT"                        # Platform default (e.g., currency, as_of_date)


class ContextConfidence(str, Enum):
    """Confidence level in context resolution."""
    HIGH = "HIGH"              # Explicitly provided or validated
    MEDIUM = "MEDIUM"          # Inferred from context or heuristics
    LOW = "LOW"                # Weak match or assumed
    UNRESOLVED = "UNRESOLVED"  # Requires clarification


class CompatibilityStatus(str, Enum):
    """Status of context item compatibility with a target query intent/domain."""
    REUSABLE = "REUSABLE"                          # Directly reusable without adaptation
    CONDITIONALLY_REUSABLE = "CONDITIONALLY_REUSABLE"  # Reusable if adapted or partially filtered
    INCOMPATIBLE = "INCOMPATIBLE"                  # Incompatible with new query domain/intent
    EXPIRED = "EXPIRED"                            # Exceeded its lifetime scope
    OVERRIDDEN = "OVERRIDDEN"                      # Superseded by newer turn or user correction


class ConflictResolution(str, Enum):
    """Deterministic strategy for resolving conflicting context attributes."""
    USER_OVERRIDE = "USER_OVERRIDE"                  # User input always overrides inherited context
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"  # Cannot resolve safely; prompt user
    INVALIDATED = "INVALIDATED"                      # Stale or conflicting context item discarded


class ContextEvictionPolicy(str, Enum):
    """Policy for evicting items when context capacity is reached."""
    LRU = "LRU"                    # Least recently used
    FIFO = "FIFO"                  # First in, first out
    SCOPE_BASED = "SCOPE_BASED"    # Expire short-lived scopes first
