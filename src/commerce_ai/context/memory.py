"""Storage Abstraction and In-Memory Store for Conversation Context (Phase 7E).

Provides:
- ConversationContextStore: Abstract interface for context storage and lifecycle
- InMemoryConversationContextStore: Thread-safe, non-persistent, local in-memory store
  strictly complying with zero external dependency (no Redis, no SQL, no vector DB).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from commerce_ai.context.schemas import ConversationContextSnapshot


class ConversationContextStore(ABC):
    """Abstract interface defining required contract for conversation context persistence."""

    @abstractmethod
    def get(self, session_id: str, conversation_id: str) -> Optional[ConversationContextSnapshot]:
        """Retrieve conversation context snapshot by session and conversation IDs."""
        pass

    @abstractmethod
    def save(self, snapshot: ConversationContextSnapshot) -> None:
        """Persist or update conversation context snapshot."""
        pass

    @abstractmethod
    def delete(self, session_id: str, conversation_id: str) -> bool:
        """Remove conversation context snapshot from store."""
        pass

    @abstractmethod
    def list_conversations(self, session_id: str) -> List[str]:
        """List all active conversation IDs belonging to a session."""
        pass

    @abstractmethod
    def clear_expired(self, ttl_seconds: int) -> int:
        """Prune all sessions whose last activity exceeds the given TTL in seconds."""
        pass


class InMemoryConversationContextStore(ConversationContextStore):
    """Thread-safe, high-performance in-memory context store.
    
    Zero external dependencies: strictly uses native Python dictionaries and thread locks.
    Never connects to external vector databases, document stores, or caches.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: Dict[Tuple[str, str], ConversationContextSnapshot] = {}

    def get(self, session_id: str, conversation_id: str) -> Optional[ConversationContextSnapshot]:
        with self._lock:
            snap = self._store.get((session_id, conversation_id))
            return snap.model_copy(deep=True) if snap else None

    def save(self, snapshot: ConversationContextSnapshot) -> None:
        with self._lock:
            # Monotonically bump version on save if not already bumped
            key = (snapshot.session_id, snapshot.conversation_id)
            existing = self._store.get(key)
            if existing and existing.version >= snapshot.version:
                snapshot.version = existing.version + 1
            snapshot.updated_at = datetime.now(timezone.utc).isoformat()
            self._store[key] = snapshot.model_copy(deep=True)

    def delete(self, session_id: str, conversation_id: str) -> bool:
        with self._lock:
            key = (session_id, conversation_id)
            if key in self._store:
                del self._store[key]
                return True
            return False

    def list_conversations(self, session_id: str) -> List[str]:
        with self._lock:
            return [c_id for (s_id, c_id) in self._store.keys() if s_id == session_id]

    def clear_expired(self, ttl_seconds: int) -> int:
        with self._lock:
            now = datetime.now(timezone.utc)
            expired_keys: List[Tuple[str, str]] = []

            for key, snap in self._store.items():
                if snap.session_context:
                    try:
                        last_active = datetime.fromisoformat(snap.session_context.last_active_at)
                        if (now - last_active).total_seconds() > ttl_seconds:
                            expired_keys.append(key)
                    except Exception:
                        pass

            for key in expired_keys:
                del self._store[key]

            return len(expired_keys)

    def clear(self) -> None:
        """Clear all stored contexts (useful for testing)."""
        with self._lock:
            self._store.clear()
