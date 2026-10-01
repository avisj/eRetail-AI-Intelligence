"""Auditability and Lineage Tracking for Conversation Context (Phase 7E).

Provides end-to-end provenance tracking for all conversation context attributes,
recording every creation, inheritance, override, user correction, and pruning event.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from commerce_ai.context.enums import ContextProvenance


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContextLineageEvent(BaseModel):
    """Immutable audit record of a single context mutation or inheritance event."""
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(description="Unique identifier for this lineage event.")
    timestamp: str = Field(default_factory=_utc_now, description="UTC timestamp of the event.")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID associated with this event.")
    turn_index: int = Field(description="Conversation turn index when event occurred.")
    item_key: str = Field(description="Context key mutated or inherited (e.g., 'sku_id', 'warehouse_id').")
    operation: str = Field(description="Operation type: CREATED, INHERITED, OVERRIDDEN, CORRECTED, PRUNED, EXPIRED.")
    provenance: ContextProvenance = Field(description="Origin source of the value.")
    old_value: Any = Field(default=None, description="Previous value before operation.")
    new_value: Any = Field(default=None, description="New value after operation.")
    query_id: Optional[str] = Field(default=None, description="Associated query ID if applicable.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional structured audit details.")


class ContextLineageTracker:
    """Tracks and explains the lifecycle and inheritance history of context items."""

    def __init__(self, events: Optional[List[ContextLineageEvent]] = None) -> None:
        self._events: List[ContextLineageEvent] = list(events) if events else []

    def record(
        self,
        turn_index: int,
        item_key: str,
        operation: str,
        provenance: ContextProvenance,
        new_value: Any,
        old_value: Any = None,
        query_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ContextLineageEvent:
        """Record a context lineage event."""
        seed = f"{conversation_id or ''}:{turn_index}:{item_key}:{operation}:{_utc_now()}:{len(self._events)}"
        ev_id = "LIN-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

        event = ContextLineageEvent(
            event_id=ev_id,
            timestamp=_utc_now(),
            conversation_id=conversation_id,
            turn_index=turn_index,
            item_key=item_key,
            operation=operation,
            provenance=provenance,
            old_value=old_value,
            new_value=new_value,
            query_id=query_id,
            details=details or {},
        )
        self._events.append(event)
        return event

    def get_events(self) -> List[ContextLineageEvent]:
        """Return all recorded lineage events."""
        return list(self._events)

    def get_conversation_events(self, conversation_id: str) -> List[ContextLineageEvent]:
        """Return all lineage events for a specific conversation ID."""
        return [e for e in self._events if e.conversation_id == conversation_id]

    def get_item_history(self, item_key: str) -> List[ContextLineageEvent]:
        """Return all lineage events for a specific item key in chronological order."""
        return [e for e in self._events if e.item_key == item_key]

    def get_turn_events(self, turn_index: int) -> List[ContextLineageEvent]:
        """Return all lineage events that occurred in a specific turn."""
        return [e for e in self._events if e.turn_index == turn_index]

    def explain_item_lineage(self, item_key: str) -> str:
        """Produce a human-readable audit narrative explaining where a context value came from."""
        history = self.get_item_history(item_key)
        if not history:
            return f"No context lineage recorded for attribute '{item_key}'."

        steps: List[str] = []
        for ev in history:
            val_repr = repr(ev.new_value)
            steps.append(
                f"[Turn {ev.turn_index}] {ev.operation} ({ev.provenance.value}): {val_repr}"
                + (f" (prior: {repr(ev.old_value)})" if ev.old_value is not None else "")
                + (f" [query: {ev.query_id}]" if ev.query_id else "")
            )
        return f"Lineage for '{item_key}':\n" + "\n".join(f"  - {s}" for s in steps)
