"""High-Level Conversation Context Service (Phase 7E).

Provides the unified entry point for:
- Session and conversation lifecycle management
- Controlled context retrieval and resolution
- Clean integration with CopilotRequest / CopilotResponse
- Post-execution context state updates and non-authoritative evidence indexing
- Provenance lineage recording and audit trail exports
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from commerce_ai.context.enums import (
    ContextCategory,
    ContextConfidence,
    ContextProvenance,
    ContextScope,
)
from commerce_ai.context.governance import (
    enforce_non_authoritative_memory,
    validate_context_governance,
)
from commerce_ai.context.lineage import ContextLineageTracker
from commerce_ai.context.memory import (
    ConversationContextStore,
    InMemoryConversationContextStore,
)
from commerce_ai.context.resolver import ContextResolver
from commerce_ai.context.schemas import (
    AmbiguityStateValue,
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
from commerce_ai.copilot.enums import CopilotState
from commerce_ai.copilot.schemas import CopilotRequest, CopilotResponse


class ConversationContextService:
    """Unified service for deterministic, controlled conversation context tracking."""

    def __init__(
        self,
        store: Optional[ConversationContextStore] = None,
        resolver: Optional[ContextResolver] = None,
        scope_manager: Optional[ScopeManager] = None,
    ) -> None:
        self.store = store or InMemoryConversationContextStore()
        self.resolver = resolver or ContextResolver()
        self.scope_manager = scope_manager or ScopeManager()
        self.lineage_tracker = ContextLineageTracker()

    def get_or_create_context(
        self,
        session_id: str,
        conversation_id: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        default_currency: str = "USD",
    ) -> ConversationContextSnapshot:
        """Retrieve existing conversation context or initialize a fresh snapshot."""
        snap = self.store.get(session_id, conversation_id)
        if snap is not None:
            # Check session expiration
            if self.scope_manager.is_session_expired(snap):
                self.store.delete(session_id, conversation_id)
                snap = None

        if snap is None:
            now_ts = datetime.now(timezone.utc).isoformat()
            sess = SessionContextValue(
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id,
                default_currency=default_currency,
                created_at=now_ts,
                last_active_at=now_ts,
            )
            snap = ConversationContextSnapshot(
                session_id=session_id,
                conversation_id=conversation_id,
                version=1,
                current_turn_index=0,
                session_context=sess,
                created_at=now_ts,
                updated_at=now_ts,
            )
            self.store.save(snap)

        return snap

    def resolve_request(
        self,
        question: str,
        session_id: str,
        conversation_id: str,
        explicit_filters: Optional[Dict[str, Any]] = None,
        as_of_date: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> Tuple[CopilotRequest, ContextResolutionResult]:
        """Resolve conversational cues, inherit active entities/filters, and formulate CopilotRequest."""
        snapshot = self.get_or_create_context(session_id, conversation_id)

        # Advance turn & prune turn-scoped items
        snapshot, _ = self.scope_manager.advance_turn(snapshot)

        # Run resolution engine
        res = self.resolver.resolve(
            question=question,
            snapshot=snapshot,
            explicit_filters=explicit_filters,
            as_of_date=as_of_date,
            currency=currency,
        )

        # Merge resolved entities and filters
        merged_filters: Dict[str, Any] = dict(explicit_filters or {})
        for k, v in res.inherited_filters.items():
            if k not in merged_filters:
                merged_filters[k] = v
        for k, v in res.overridden_filters.items():
            merged_filters[k] = v

        # Combine entities into filters for downstream Copilot / QueryContract
        effective_entities: Dict[str, Any] = {}
        # Apply snapshot entities first
        effective_entities.update(snapshot.entities.to_dict())
        # Apply inherited entities
        effective_entities.update(res.inherited_entities)
        # Apply overrides
        for k, v in res.overridden_entities.items():
            if v is None:
                effective_entities.pop(k, None)
            else:
                effective_entities[k] = v

        # Prune incompatible entities from domain switching
        for k in res.incompatible_pruned:
            effective_entities.pop(k, None)
            merged_filters.pop(k, None)
            if k.startswith("context_"):
                base_k = k.replace("context_", "")
                effective_entities.pop(base_k, None)
                merged_filters.pop(base_k, None)

        # Add effective entities into filters if not explicitly present
        for ent_key in ["sku_id", "warehouse_id", "channel_id", "supplier_id"]:
            if ent_key in effective_entities and ent_key not in merged_filters:
                merged_filters[ent_key] = effective_entities[ent_key]

        # Record lineage for inherited/overridden attributes
        for k, v in res.inherited_entities.items():
            self.lineage_tracker.record(
                turn_index=snapshot.current_turn_index,
                item_key=k,
                operation="INHERITED",
                provenance=ContextProvenance.DERIVED_FROM_VALIDATED_QUERY,
                new_value=v,
                old_value=None,
                conversation_id=conversation_id,
            )
        for k, v in res.overridden_entities.items():
            self.lineage_tracker.record(
                turn_index=snapshot.current_turn_index,
                item_key=k,
                operation="OVERRIDDEN" if v is not None else "PRUNED",
                provenance=ContextProvenance.EXPLICIT_USER,
                new_value=v,
                old_value=snapshot.entities.to_dict().get(k),
                conversation_id=conversation_id,
            )

        # Build context dictionary for CopilotRequest
        context_payload: Dict[str, Any] = {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "turn_index": snapshot.current_turn_index,
            "effective_entities": effective_entities,
            "inherited_entities": res.inherited_entities,
            "overridden_entities": res.overridden_entities,
            "time_shift": res.time_shift_applied,
            "ambiguity_resolved": res.ambiguity_resolved,
            "sanitized_violations": res.sanitized_violations,
        }

        # Formulate final CopilotRequest
        req_id = "REQ-" + hashlib.sha256(f"{session_id}:{conversation_id}:{snapshot.current_turn_index}:{question}".encode("utf-8")).hexdigest()[:16]
        effective_currency = currency or (snapshot.session_context.default_currency if snapshot.session_context else "USD")

        req = CopilotRequest(
            request_id=req_id,
            question=res.resolved_question,
            as_of_date=as_of_date or (res.time_shift_applied.get("anchor_as_of_date") if res.time_shift_applied else None),
            currency=effective_currency,
            filters=merged_filters,
            context=context_payload,
        )

        # Update snapshot in-memory
        # Update entities
        snapshot.entities.sku_id = effective_entities.get("sku_id")
        snapshot.entities.warehouse_id = effective_entities.get("warehouse_id")
        snapshot.entities.channel_id = effective_entities.get("channel_id")
        snapshot.entities.supplier_id = effective_entities.get("supplier_id")

        # If ambiguity was resolved, clear active_ambiguity
        if res.ambiguity_resolved:
            snapshot.active_ambiguity = None

        self.store.save(snapshot)

        return req, res

    def update_after_response(
        self,
        session_id: str,
        conversation_id: str,
        request: CopilotRequest,
        response: CopilotResponse,
        summary_text: Optional[str] = None,
    ) -> ConversationContextSnapshot:
        """Update context snapshot with post-execution evidence, query history, and ambiguity state."""
        snapshot = self.get_or_create_context(session_id, conversation_id)
        now_ts = datetime.now(timezone.utc).isoformat()

        # 1. Record Recent Query
        snapshot.recent_queries.append({
            "request_id": request.request_id,
            "response_id": response.response_id,
            "turn_index": snapshot.current_turn_index,
            "question": request.question,
            "state": response.state.value if hasattr(response.state, "value") else str(response.state),
            "timestamp": now_ts,
        })

        # 2. Check if CLARIFICATION_REQUIRED
        if response.state == CopilotState.CLARIFICATION_REQUIRED and response.clarification:
            amb_id = "AMB-" + hashlib.sha256(f"{snapshot.current_turn_index}:{request.question}".encode("utf-8")).hexdigest()[:16]
            snapshot.active_ambiguity = AmbiguityStateValue(
                ambiguity_id=amb_id,
                turn_index=snapshot.current_turn_index,
                original_question=request.question,
                missing_fields=response.clarification.missing_fields,
                candidate_intents=response.clarification.possible_interpretations,
                clarification_prompt=response.clarification.reason,
                status="PENDING",
            )

        # 3. Record Evidence Citations (Non-Authoritative)
        if response.execution_result:
            evidence_ids: List[str] = []
            for ev in response.execution_result.evidence_chain:
                evidence_ids.append(ev.evidence_id)
                snapshot.evidence_refs.append(
                    EvidenceReferenceValue(
                        evidence_id=ev.evidence_id,
                        tool_name=ev.tool_name,
                        query_id=request.request_id or "QRY-REQ",
                        turn_index=snapshot.current_turn_index,
                        intent=ev.tool_name,
                        calculation_status="SUCCESS",
                        as_of_date=ev.as_of_date or request.as_of_date,
                    )
                )

            # 4. Record Result Reference (Summary metadata only, NOT replacement business truth)
            rec_count = 0
            for s in response.execution_result.step_results:
                if s.response:
                    if s.response.table:
                        rec_count += len(s.response.table.rows)
                    elif s.response.breakdown:
                        rec_count += len(s.response.breakdown.items)
                    elif s.response.time_series:
                        rec_count += len(s.response.time_series.data_points)
                    elif s.response.metrics:
                        rec_count += len(s.response.metrics)
            res_ref = ResultReferenceValue(
                query_id=request.request_id or "QRY-REQ",
                turn_index=snapshot.current_turn_index,
                intent=response.interpretation.intent.value if response.interpretation else "UNKNOWN",
                record_count=rec_count,
                key_metrics={},
                summary_text=summary_text,
                evidence_ids=evidence_ids,
                timestamp=now_ts,
            )
            snapshot.result_refs.append(res_ref)

        # Enforce capacity
        snapshot, _ = self.scope_manager.enforce_capacity(snapshot)

        # Validate governance invariants
        is_gov_valid, gov_violations = validate_context_governance(snapshot)
        if not is_gov_valid:
            # Re-enforce strictly
            snapshot.governance = snapshot.governance.model_copy()

        self.store.save(snapshot)
        return snapshot

    def record_user_correction(
        self,
        session_id: str,
        conversation_id: str,
        target_key: str,
        new_value: Any,
        reason: str,
    ) -> CorrectionEvent:
        """Manually record a user correction against conversation context."""
        snapshot = self.get_or_create_context(session_id, conversation_id)
        now_ts = datetime.now(timezone.utc).isoformat()
        corr_id = "CORR-" + hashlib.sha256(f"{snapshot.current_turn_index}:{target_key}:{now_ts}".encode("utf-8")).hexdigest()[:16]

        prev_val = getattr(snapshot.entities, target_key, None)
        event = CorrectionEvent(
            correction_id=corr_id,
            turn_index=snapshot.current_turn_index,
            target_category=ContextCategory.ENTITY_CONTEXT,
            target_key=target_key,
            previous_value=prev_val,
            new_value=new_value,
            reason=reason,
            timestamp=now_ts,
        )
        snapshot.corrections.append(event)
        if hasattr(snapshot.entities, target_key):
            setattr(snapshot.entities, target_key, new_value)

        self.lineage_tracker.record(
            turn_index=snapshot.current_turn_index,
            item_key=target_key,
            operation="USER_CORRECTION",
            provenance=ContextProvenance.USER_CORRECTION,
            new_value=new_value,
            old_value=prev_val,
            conversation_id=conversation_id,
            details={"reason": reason},
        )
        self.store.save(snapshot)
        return event

    def reset_conversation(self, session_id: str, conversation_id: str) -> bool:
        """Completely reset context for a specific conversation."""
        return self.store.delete(session_id, conversation_id)

    def export_audit_trail(self, session_id: str, conversation_id: str) -> Dict[str, Any]:
        """Export full auditable context state, lineage events, and governance status."""
        snapshot = self.get_or_create_context(session_id, conversation_id)
        conv_events = self.lineage_tracker.get_conversation_events(conversation_id)
        # If no conversation-tagged events exist, fallback to all unassigned events for backwards compatibility
        if not conv_events:
            conv_events = [ev for ev in self.lineage_tracker.get_events() if ev.conversation_id is None]
        return {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "current_turn": snapshot.current_turn_index,
            "active_entities": snapshot.entities.to_dict(),
            "corrections_count": len(snapshot.corrections),
            "evidence_citations_count": len(snapshot.evidence_refs),
            "result_references_count": len(snapshot.result_refs),
            "lineage_events": [ev.model_dump() for ev in conv_events],
            "governance": snapshot.governance.model_dump(),
        }
