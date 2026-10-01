"""Unit tests for Context Scope and Lifecycle Management (Phase 7E)."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from commerce_ai.context.enums import ContextCategory, ContextEvictionPolicy, ContextScope
from commerce_ai.context.schemas import (
    ContextItem,
    ConversationContextSnapshot,
    SessionContextValue,
)
from commerce_ai.context.scope import ScopeManager


def test_scope_manager_initialization() -> None:
    sm = ScopeManager(default_session_ttl_seconds=1800, max_turn_history=30)
    assert sm.session_ttl_seconds == 1800
    assert sm.max_turn_history == 30
    assert sm.eviction_policy == ContextEvictionPolicy.SCOPE_BASED


def test_scope_advance_turn_increments_turn_and_version() -> None:
    sm = ScopeManager()
    snap = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
        version=1,
        current_turn_index=0,
    )
    updated, pruned = sm.advance_turn(snap)
    assert updated.current_turn_index == 1
    assert updated.version == 2
    assert len(pruned) == 0


def test_scope_advance_turn_prunes_turn_scoped_items() -> None:
    sm = ScopeManager()
    snap = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
    )
    # Add a TURN-scoped item
    snap.items["comparison_target"] = ContextItem(
        item_id="ITM-1",
        category=ContextCategory.FILTER_CONTEXT,
        scope=ContextScope.TURN,
        key="comparison_target",
        value="WH_02",
    )
    # Add a CONVERSATION-scoped item
    snap.items["warehouse_id"] = ContextItem(
        item_id="ITM-2",
        category=ContextCategory.ENTITY_CONTEXT,
        scope=ContextScope.CONVERSATION,
        key="warehouse_id",
        value="WH_01",
    )

    updated, pruned = sm.advance_turn(snap)
    assert "comparison_target" in pruned
    assert "comparison_target" not in updated.items
    assert "warehouse_id" in updated.items
    assert updated.items["warehouse_id"].value == "WH_01"


def test_scope_advance_turn_trims_history_excess() -> None:
    sm = ScopeManager(max_turn_history=3)
    snap = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
    )
    snap.recent_queries = [
        {"turn_index": 1},
        {"turn_index": 2},
        {"turn_index": 3},
        {"turn_index": 4},
    ]
    updated, _ = sm.advance_turn(snap)
    assert len(updated.recent_queries) == 3
    assert updated.recent_queries[0]["turn_index"] == 2


def test_scope_is_session_expired_false_when_active() -> None:
    sm = ScopeManager(default_session_ttl_seconds=3600)
    now_ts = datetime.now(timezone.utc).isoformat()
    snap = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
        session_context=SessionContextValue(
            session_id="S-1",
            created_at=now_ts,
            last_active_at=now_ts,
        ),
    )
    assert sm.is_session_expired(snap) is False


def test_scope_is_session_expired_true_when_stale() -> None:
    sm = ScopeManager(default_session_ttl_seconds=3600)
    stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=7200)).isoformat()
    snap = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
        session_context=SessionContextValue(
            session_id="S-1",
            created_at=stale_ts,
            last_active_at=stale_ts,
        ),
    )
    assert sm.is_session_expired(snap) is True


def test_scope_is_session_expired_without_session_context() -> None:
    sm = ScopeManager()
    snap = ConversationContextSnapshot(session_id="S-1", conversation_id="C-1")
    assert sm.is_session_expired(snap) is False


def test_scope_enforce_capacity_no_op_under_limit() -> None:
    sm = ScopeManager(max_items_per_conversation=5)
    snap = ConversationContextSnapshot(session_id="S-1", conversation_id="C-1")
    snap.items["k1"] = ContextItem(item_id="1", category=ContextCategory.ENTITY_CONTEXT, key="k1", value="v1")
    updated, evicted = sm.enforce_capacity(snap)
    assert len(evicted) == 0
    assert len(updated.items) == 1


def test_scope_enforce_capacity_scope_based_eviction() -> None:
    sm = ScopeManager(max_items_per_conversation=2, eviction_policy=ContextEvictionPolicy.SCOPE_BASED)
    snap = ConversationContextSnapshot(session_id="S-1", conversation_id="C-1")
    snap.items["query_item"] = ContextItem(
        item_id="1", category=ContextCategory.FILTER_CONTEXT, scope=ContextScope.QUERY, key="q", value="v1", turn_index=1
    )
    snap.items["conv_item"] = ContextItem(
        item_id="2", category=ContextCategory.ENTITY_CONTEXT, scope=ContextScope.CONVERSATION, key="c", value="v2", turn_index=1
    )
    snap.items["session_item"] = ContextItem(
        item_id="3", category=ContextCategory.SESSION_CONTEXT, scope=ContextScope.SESSION, key="s", value="v3", turn_index=1
    )

    updated, evicted = sm.enforce_capacity(snap)
    assert len(evicted) == 1
    # QUERY scope should be evicted first
    assert "query_item" in evicted
    assert "query_item" not in updated.items
    assert len(updated.items) == 2


def test_scope_enforce_capacity_fifo_eviction() -> None:
    sm = ScopeManager(max_items_per_conversation=2, eviction_policy=ContextEvictionPolicy.FIFO)
    snap = ConversationContextSnapshot(session_id="S-1", conversation_id="C-1")
    snap.items["oldest"] = ContextItem(
        item_id="1", category=ContextCategory.ENTITY_CONTEXT, key="o", value="v1", turn_index=1
    )
    snap.items["middle"] = ContextItem(
        item_id="2", category=ContextCategory.ENTITY_CONTEXT, key="m", value="v2", turn_index=2
    )
    snap.items["newest"] = ContextItem(
        item_id="3", category=ContextCategory.ENTITY_CONTEXT, key="n", value="v3", turn_index=3
    )

    updated, evicted = sm.enforce_capacity(snap)
    assert len(evicted) == 1
    assert "oldest" in evicted
    assert "oldest" not in updated.items
    assert len(updated.items) == 2
