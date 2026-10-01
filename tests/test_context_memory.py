"""Unit tests for Context Memory and Storage Abstraction (Phase 7E)."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone, timedelta

from commerce_ai.context.memory import InMemoryConversationContextStore
from commerce_ai.context.schemas import (
    ConversationContextSnapshot,
    EntityContextValue,
    SessionContextValue,
)


def test_memory_store_save_and_get() -> None:
    store = InMemoryConversationContextStore()
    snap = ConversationContextSnapshot(
        session_id="S-01",
        conversation_id="C-01",
        entities=EntityContextValue(sku_id="SKU_100"),
    )
    store.save(snap)

    retrieved = store.get("S-01", "C-01")
    assert retrieved is not None
    assert retrieved.session_id == "S-01"
    assert retrieved.conversation_id == "C-01"
    assert retrieved.entities.sku_id == "SKU_100"


def test_memory_store_not_found() -> None:
    store = InMemoryConversationContextStore()
    assert store.get("NONEXISTENT", "NONEXISTENT") is None


def test_memory_store_deep_copy_isolation() -> None:
    store = InMemoryConversationContextStore()
    snap = ConversationContextSnapshot(
        session_id="S-01",
        conversation_id="C-01",
        entities=EntityContextValue(sku_id="SKU_ORIG"),
    )
    store.save(snap)

    retrieved1 = store.get("S-01", "C-01")
    assert retrieved1 is not None
    retrieved1.entities.sku_id = "SKU_MUTATED"

    retrieved2 = store.get("S-01", "C-01")
    assert retrieved2 is not None
    assert retrieved2.entities.sku_id == "SKU_ORIG"


def test_memory_store_version_increment() -> None:
    store = InMemoryConversationContextStore()
    snap = ConversationContextSnapshot(
        session_id="S-01",
        conversation_id="C-01",
        version=1,
    )
    store.save(snap)
    first_ver = store.get("S-01", "C-01").version  # type: ignore

    snap2 = store.get("S-01", "C-01")
    assert snap2 is not None
    store.save(snap2)
    second_ver = store.get("S-01", "C-01").version  # type: ignore

    assert second_ver > first_ver


def test_memory_store_delete() -> None:
    store = InMemoryConversationContextStore()
    snap = ConversationContextSnapshot(
        session_id="S-01",
        conversation_id="C-01",
    )
    store.save(snap)
    assert store.get("S-01", "C-01") is not None

    deleted = store.delete("S-01", "C-01")
    assert deleted is True
    assert store.get("S-01", "C-01") is None

    # Deleting again returns False
    assert store.delete("S-01", "C-01") is False


def test_memory_store_list_conversations() -> None:
    store = InMemoryConversationContextStore()
    snap1 = ConversationContextSnapshot(session_id="S-01", conversation_id="C-01")
    snap2 = ConversationContextSnapshot(session_id="S-01", conversation_id="C-02")
    snap3 = ConversationContextSnapshot(session_id="S-02", conversation_id="C-03")
    store.save(snap1)
    store.save(snap2)
    store.save(snap3)

    convs = store.list_conversations("S-01")
    assert set(convs) == {"C-01", "C-02"}
    assert store.list_conversations("S-02") == ["C-03"]
    assert store.list_conversations("S-03") == []


def test_memory_store_clear_expired() -> None:
    store = InMemoryConversationContextStore()
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=7200)).isoformat()
    fresh_ts = datetime.now(timezone.utc).isoformat()

    expired_snap = ConversationContextSnapshot(
        session_id="S-EXPIRED",
        conversation_id="C-01",
        session_context=SessionContextValue(
            session_id="S-EXPIRED",
            created_at=old_ts,
            last_active_at=old_ts,
        ),
    )
    active_snap = ConversationContextSnapshot(
        session_id="S-ACTIVE",
        conversation_id="C-02",
        session_context=SessionContextValue(
            session_id="S-ACTIVE",
            created_at=fresh_ts,
            last_active_at=fresh_ts,
        ),
    )
    store.save(expired_snap)
    store.save(active_snap)

    # Prune with TTL=3600
    pruned_count = store.clear_expired(ttl_seconds=3600)
    assert pruned_count == 1
    assert store.get("S-EXPIRED", "C-01") is None
    assert store.get("S-ACTIVE", "C-02") is not None


def test_memory_store_thread_safety() -> None:
    store = InMemoryConversationContextStore()
    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            for j in range(20):
                c_id = f"CONV-{idx}-{j}"
                snap = ConversationContextSnapshot(
                    session_id=f"SESS-{idx}",
                    conversation_id=c_id,
                    entities=EntityContextValue(sku_id=f"SKU-{idx}"),
                )
                store.save(snap)
                ret = store.get(f"SESS-{idx}", c_id)
                assert ret is not None
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0


def test_memory_store_clear() -> None:
    store = InMemoryConversationContextStore()
    snap = ConversationContextSnapshot(session_id="S-1", conversation_id="C-1")
    store.save(snap)
    assert store.get("S-1", "C-1") is not None
    store.clear()
    assert store.get("S-1", "C-1") is None
