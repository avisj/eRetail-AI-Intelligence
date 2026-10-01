"""Scope and Lifecycle Management for Conversation Context (Phase 7E).

Implements deterministic scoping rules, lifetime expiration, and eviction policies:
- TURN: Immediate single-turn lifetime; automatically pruned when turn advances
- QUERY: Lifetime bounded to single multi-step query execution
- CONVERSATION: Persists across turns within conversational thread
- SESSION: Persists across entire user session until TTL expiration
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from commerce_ai.context.enums import ContextEvictionPolicy, ContextScope
from commerce_ai.context.schemas import ContextItem, ConversationContextSnapshot


def _parse_iso(ts: str) -> datetime:
    """Parse ISO UTC timestamp."""
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.now(timezone.utc)


class ScopeManager:
    """Manages scoping rules, TTL expiration, and capacity limits for conversation context."""

    def __init__(
        self,
        default_session_ttl_seconds: int = 3600,
        max_turn_history: int = 50,
        max_items_per_conversation: int = 200,
        eviction_policy: ContextEvictionPolicy = ContextEvictionPolicy.SCOPE_BASED,
    ) -> None:
        self.session_ttl_seconds = default_session_ttl_seconds
        self.max_turn_history = max_turn_history
        self.max_items_per_conversation = max_items_per_conversation
        self.eviction_policy = eviction_policy

    def is_session_expired(self, snapshot: ConversationContextSnapshot, now: Optional[datetime] = None) -> bool:
        """Check whether the session has expired past its TTL."""
        if not snapshot.session_context:
            return False

        current_time = now or datetime.now(timezone.utc)
        last_active = _parse_iso(snapshot.session_context.last_active_at)
        delta = (current_time - last_active).total_seconds()
        return delta > self.session_ttl_seconds

    def advance_turn(self, snapshot: ConversationContextSnapshot) -> Tuple[ConversationContextSnapshot, List[str]]:
        """Advance the conversation turn index and prune expired TURN-scoped context items.
        
        Returns:
            Tuple of (updated_snapshot, list_of_pruned_item_keys)
        """
        snapshot.current_turn_index += 1
        snapshot.version += 1
        now_ts = datetime.now(timezone.utc).isoformat()
        snapshot.updated_at = now_ts
        if snapshot.session_context:
            snapshot.session_context.last_active_at = now_ts

        pruned_keys: List[str] = []
        new_items: Dict[str, ContextItem] = {}

        for key, item in snapshot.items.items():
            if item.scope == ContextScope.TURN:
                # Expire items that only apply to the immediate turn
                pruned_keys.append(key)
            else:
                new_items[key] = item

        snapshot.items = new_items

        # Prune turn history if exceeding max_turn_history
        if len(snapshot.recent_queries) > self.max_turn_history:
            snapshot.recent_queries = snapshot.recent_queries[-self.max_turn_history:]

        return snapshot, pruned_keys

    def enforce_capacity(self, snapshot: ConversationContextSnapshot) -> Tuple[ConversationContextSnapshot, List[str]]:
        """Evict context items if capacity limits are exceeded based on eviction policy."""
        evicted_keys: List[str] = []
        if len(snapshot.items) <= self.max_items_per_conversation:
            return snapshot, evicted_keys

        excess = len(snapshot.items) - self.max_items_per_conversation

        if self.eviction_policy == ContextEvictionPolicy.SCOPE_BASED:
            # First evict QUERY, then CONVERSATION
            candidates = sorted(
                snapshot.items.items(),
                key=lambda x: (
                    0 if x[1].scope == ContextScope.QUERY else
                    1 if x[1].scope == ContextScope.CONVERSATION else 2,
                    x[1].turn_index,
                ),
            )
            for k, _ in candidates[:excess]:
                evicted_keys.append(k)
                del snapshot.items[k]

        elif self.eviction_policy == ContextEvictionPolicy.FIFO:
            candidates = sorted(snapshot.items.items(), key=lambda x: x[1].turn_index)
            for k, _ in candidates[:excess]:
                evicted_keys.append(k)
                del snapshot.items[k]

        return snapshot, evicted_keys
