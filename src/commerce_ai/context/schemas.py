"""Pydantic Schemas for Controlled Conversation Context (Phase 7E).

Provides strongly typed, validated, and auditable models for:
- ContextGovernance: Immutability, read-only guarantees, non-authoritative memory enforcement
- ContextItem: Granular context element with provenance, scope, and compatibility tracking
- SessionContextValue: User session metadata and defaults
- EntityContextValue: Active business entities (SKU, Warehouse, Channel, etc.)
- TimeRangeContextValue: Temporal parameters and shifting metadata
- FilterContextValue: Analytical filters and dimensions
- EvidenceReferenceValue: Citations to tool execution results
- ResultReferenceValue: Non-authoritative metadata references to prior result sets
- CorrectionEvent: Explicit user corrections overriding prior state
- AmbiguityStateValue: Unresolved ambiguity state awaiting user clarification
- ConversationContextSnapshot: Immutable full snapshot of conversation state
- ContextResolutionResult: Structured output of context resolution for a follow-up query
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from commerce_ai.context.enums import (
    CompatibilityStatus,
    ContextCategory,
    ContextConfidence,
    ContextProvenance,
    ContextScope,
)


def _current_timestamp() -> str:
    """Return ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class ContextGovernance(BaseModel):
    """Platform governance constraints enforced across all conversation context operations."""
    model_config = ConfigDict(frozen=True)

    read_only: bool = Field(
        default=True,
        description="Strictly True. Conversation context can never mutate database or business records.",
    )
    action_execution: bool = Field(
        default=False,
        description="Strictly False. Context can never autonomously trigger actions.",
    )
    execution_allowed: bool = Field(
        default=False,
        description="Strictly False. Business state mutations are prohibited.",
    )
    no_ranking_enforced: bool = Field(
        default=True,
        description="Strictly True. Subjective entity rankings (top-N, best) are forbidden.",
    )
    no_decision_selection_enforced: bool = Field(
        default=True,
        description="Strictly True. Autonomous selection among decision options is prohibited.",
    )
    allow_injection: bool = Field(
        default=False,
        description="Strictly False. Prompt injection patterns and fake business facts are rejected.",
    )


class ContextItem(BaseModel):
    """Granular context item tracking key, value, provenance, and lifecycle scope."""
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(description="Unique identifier for this context item.")
    category: ContextCategory = Field(description="Semantic category of context.")
    scope: ContextScope = Field(default=ContextScope.CONVERSATION, description="Lifetime scope.")
    key: str = Field(description="Semantic attribute key (e.g. 'sku_id', 'warehouse_id', 'date_range').")
    value: Any = Field(description="Value stored in context.")
    provenance: ContextProvenance = Field(
        default=ContextProvenance.EXPLICIT_USER,
        description="Origin source of this context item.",
    )
    confidence: ContextConfidence = Field(
        default=ContextConfidence.HIGH,
        description="Confidence level in resolution.",
    )
    compatibility_status: CompatibilityStatus = Field(
        default=CompatibilityStatus.REUSABLE,
        description="Compatibility with current or subsequent query intents.",
    )
    turn_index: int = Field(default=0, description="Conversation turn index when this item was recorded.")
    query_id: Optional[str] = Field(default=None, description="Query identifier associated with this item.")
    created_at: str = Field(default_factory=_current_timestamp, description="UTC creation timestamp.")
    updated_at: str = Field(default_factory=_current_timestamp, description="UTC last update timestamp.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional audit metadata.")


class SessionContextValue(BaseModel):
    """Session-level configuration and default parameters."""
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Unique session identifier.")
    user_id: Optional[str] = Field(default=None, description="Authenticated user identifier.")
    tenant_id: Optional[str] = Field(default=None, description="Tenant / organization identifier.")
    default_currency: str = Field(default="USD", description="Default operational currency.")
    created_at: str = Field(default_factory=_current_timestamp, description="Session start timestamp.")
    last_active_at: str = Field(default_factory=_current_timestamp, description="Last activity timestamp.")


class EntityContextValue(BaseModel):
    """Active business entities tracked across conversation turns."""
    model_config = ConfigDict(extra="forbid")

    sku_id: Optional[str] = Field(default=None, description="Active SKU identifier.")
    warehouse_id: Optional[str] = Field(default=None, description="Active Warehouse identifier.")
    channel_id: Optional[str] = Field(default=None, description="Active Sales Channel identifier.")
    supplier_id: Optional[str] = Field(default=None, description="Active Supplier identifier.")
    category: Optional[str] = Field(default=None, description="Active Product Category.")
    brand: Optional[str] = Field(default=None, description="Active Brand.")
    store_id: Optional[str] = Field(default=None, description="Active Store identifier.")
    additional_entities: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary additional entity key-value pairs.",
    )

    def to_dict(self) -> Dict[str, str]:
        """Convert populated entities to dictionary."""
        res: Dict[str, str] = {}
        for k in ["sku_id", "warehouse_id", "channel_id", "supplier_id", "category", "brand", "store_id"]:
            v = getattr(self, k)
            if v is not None:
                res[k] = v
        res.update(self.additional_entities)
        return res


class TimeRangeContextValue(BaseModel):
    """Temporal window parameters and shifting metadata."""
    model_config = ConfigDict(extra="forbid")

    preset: Optional[str] = Field(default=None, description="Named preset (e.g. 'LAST_30_DAYS', 'PRIOR_MONTH').")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD).")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD).")
    as_of_date: Optional[str] = Field(default=None, description="Anchor as-of date (YYYY-MM-DD).")
    grain: Optional[str] = Field(default=None, description="Time grain (e.g. 'DAY', 'WEEK', 'MONTH').")
    granularity: Optional[str] = Field(default=None, description="Time granularity.")
    shift_description: Optional[str] = Field(default=None, description="Human description of relative time shift.")


class FilterContextValue(BaseModel):
    """Active analytical filters tracked across conversation turns."""
    model_config = ConfigDict(extra="forbid")

    time_range: Optional[TimeRangeContextValue] = Field(default=None, description="Temporal filter window.")
    dimension: Optional[str] = Field(default=None, description="Active breakdown dimension.")
    grain: Optional[str] = Field(default=None, description="Business grain.")
    currency: Optional[str] = Field(default=None, description="Currency filter.")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Generic filter key-value pairs.")


class EvidenceReferenceValue(BaseModel):
    """Auditable citation reference to past tool execution evidence."""
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(description="Evidence identifier from Phase 7C execution.")
    tool_name: str = Field(description="Phase 7A canonical tool name.")
    query_id: str = Field(description="Query ID that generated this evidence.")
    turn_index: int = Field(description="Turn index when evidence was produced.")
    intent: Optional[str] = Field(default=None, description="Intent behind execution.")
    calculation_status: Optional[str] = Field(default=None, description="Calculation status.")
    as_of_date: Optional[str] = Field(default=None, description="As-of date used in calculation.")


class ResultReferenceValue(BaseModel):
    """Non-authoritative metadata reference to prior result sets.
    
    CRITICAL: This stores ONLY metadata summaries and references, NEVER raw data
    to be substituted as authoritative truth in lieu of tool execution.
    """
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(description="Query ID of result.")
    turn_index: int = Field(description="Turn index of result.")
    intent: str = Field(description="Query intent.")
    record_count: int = Field(default=0, description="Number of records returned by tool.")
    key_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary metadata only (not authoritative calculation truth).",
    )
    summary_text: Optional[str] = Field(default=None, description="Brief explanation summary.")
    evidence_ids: List[str] = Field(default_factory=list, description="Associated evidence IDs.")
    timestamp: str = Field(default_factory=_current_timestamp, description="Timestamp of execution.")


class CorrectionEvent(BaseModel):
    """Record of an explicit user correction overriding existing context."""
    model_config = ConfigDict(extra="forbid")

    correction_id: str = Field(description="Unique correction event ID.")
    turn_index: int = Field(description="Turn index where correction occurred.")
    target_category: ContextCategory = Field(description="Category being corrected.")
    target_key: str = Field(description="Attribute key being corrected.")
    previous_value: Any = Field(description="Value prior to correction.")
    new_value: Any = Field(description="Corrected value provided by user.")
    reason: str = Field(description="Correction rationale (e.g. 'User negated WH_01 in favor of WH_02').")
    timestamp: str = Field(default_factory=_current_timestamp, description="UTC timestamp of correction.")


class AmbiguityStateValue(BaseModel):
    """Pending ambiguity state requiring user clarification before execution."""
    model_config = ConfigDict(extra="forbid")

    ambiguity_id: str = Field(description="Unique ambiguity state ID.")
    turn_index: int = Field(description="Turn index where ambiguity was detected.")
    original_question: str = Field(description="User question that was ambiguous.")
    missing_fields: List[str] = Field(default_factory=list, description="Missing required parameters.")
    candidate_intents: List[str] = Field(default_factory=list, description="Possible candidate intents.")
    candidate_entities: Dict[str, List[str]] = Field(default_factory=dict, description="Candidate entities.")
    clarification_prompt: str = Field(description="Prompt presented to user asking for clarification.")
    status: str = Field(default="PENDING", description="'PENDING' or 'RESOLVED'.")


class ConversationContextSnapshot(BaseModel):
    """Immutable, auditable snapshot of entire conversation context state."""
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Session identifier.")
    conversation_id: str = Field(description="Conversation identifier.")
    version: int = Field(default=1, description="Monotonically increasing state version number.")
    current_turn_index: int = Field(default=0, description="Active turn index.")
    session_context: Optional[SessionContextValue] = Field(default=None, description="Session parameters.")
    entities: EntityContextValue = Field(default_factory=EntityContextValue, description="Active entities.")
    filters: FilterContextValue = Field(default_factory=FilterContextValue, description="Active filters.")
    recent_queries: List[Dict[str, Any]] = Field(default_factory=list, description="Recent query history.")
    evidence_refs: List[EvidenceReferenceValue] = Field(default_factory=list, description="Evidence citations.")
    result_refs: List[ResultReferenceValue] = Field(default_factory=list, description="Result metadata citations.")
    active_ambiguity: Optional[AmbiguityStateValue] = Field(default=None, description="Active pending clarification.")
    corrections: List[CorrectionEvent] = Field(default_factory=list, description="Audit log of user corrections.")
    items: Dict[str, ContextItem] = Field(default_factory=dict, description="Granular context items.")
    governance: ContextGovernance = Field(default_factory=ContextGovernance, description="Governance constraints.")
    created_at: str = Field(default_factory=_current_timestamp, description="Snapshot creation timestamp.")
    updated_at: str = Field(default_factory=_current_timestamp, description="Snapshot last update timestamp.")


class ContextResolutionResult(BaseModel):
    """Structured result of resolving conversational cues against active context."""
    model_config = ConfigDict(extra="forbid")

    original_question: str = Field(description="Original user question.")
    resolved_question: str = Field(description="Question expanded with resolved context entities/filters.")
    inherited_entities: Dict[str, Any] = Field(default_factory=dict, description="Entities inherited from context.")
    overridden_entities: Dict[str, Any] = Field(default_factory=dict, description="Entities explicitly overridden.")
    inherited_filters: Dict[str, Any] = Field(default_factory=dict, description="Filters inherited from context.")
    overridden_filters: Dict[str, Any] = Field(default_factory=dict, description="Filters explicitly overridden.")
    incompatible_pruned: Dict[str, Any] = Field(default_factory=dict, description="Incompatible context pruned.")
    corrections_applied: List[CorrectionEvent] = Field(default_factory=list, description="User corrections applied.")
    time_shift_applied: Optional[Dict[str, Any]] = Field(default=None, description="Time shift details if applied.")
    ambiguity_resolved: bool = Field(default=False, description="True if this turn resolved a pending ambiguity.")
    inferred_intent: Optional[str] = Field(default=None, description="Intent inferred during context resolution.")
    inferred_domain: Optional[str] = Field(default=None, description="Domain inferred during context resolution.")
    lineage_events: List[Dict[str, Any]] = Field(default_factory=list, description="Lineage audit records.")
    confidence: ContextConfidence = Field(default=ContextConfidence.HIGH, description="Resolution confidence.")
    sanitized_violations: List[str] = Field(default_factory=list, description="Security violations detected and sanitized.")
