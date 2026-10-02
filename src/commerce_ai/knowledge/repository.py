"""Knowledge Repository Storage Abstraction (Phase 7F).

Provides an abstract interface and thread-safe in-memory implementation for storing
and querying BusinessKnowledgeDocuments and KnowledgeChunks with strict tenant isolation.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from commerce_ai.knowledge.enums import DocumentStatus, KnowledgeDomain
from commerce_ai.knowledge.schemas import BusinessKnowledgeDocument, KnowledgeChunk


class KnowledgeRepository(ABC):
    """Abstract interface for business knowledge storage."""

    @abstractmethod
    def add_document(self, document: BusinessKnowledgeDocument) -> bool:
        """Store or update a document version."""

    @abstractmethod
    def get_document(
        self,
        document_id: str,
        version: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> Optional[BusinessKnowledgeDocument]:
        """Retrieve a specific document version or the active version."""

    @abstractmethod
    def list_documents(
        self,
        organization_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
        domain: Optional[KnowledgeDomain] = None,
    ) -> List[BusinessKnowledgeDocument]:
        """List documents matching optional filters and tenant scope."""

    @abstractmethod
    def add_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        """Store chunk records."""

    @abstractmethod
    def get_chunks(
        self,
        document_id: str,
        version: Optional[str] = None,
    ) -> List[KnowledgeChunk]:
        """Retrieve chunks for a given document and version."""

    @abstractmethod
    def get_all_chunks(
        self,
        organization_id: Optional[str] = None,
        include_inactive: bool = False,
    ) -> List[KnowledgeChunk]:
        """Retrieve all indexed chunks respecting tenant boundaries and status."""

    @abstractmethod
    def delete_document(
        self,
        document_id: str,
        version: Optional[str] = None,
    ) -> bool:
        """Delete a specific document version or all versions of a document."""

    @abstractmethod
    def clear(self) -> None:
        """Reset repository to empty state."""


class InMemoryKnowledgeRepository(KnowledgeRepository):
    """Thread-safe in-memory repository implementation for local knowledge storage."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Key: (document_id, version) -> BusinessKnowledgeDocument
        self._documents: Dict[Tuple[str, str], BusinessKnowledgeDocument] = {}
        # Key: (document_id, version) -> List[KnowledgeChunk]
        self._chunks: Dict[Tuple[str, str], List[KnowledgeChunk]] = {}
        # Fast chunk lookup by chunk_id
        self._chunk_by_id: Dict[str, KnowledgeChunk] = {}

    def add_document(self, document: BusinessKnowledgeDocument) -> bool:
        with self._lock:
            key = (document.document_id, document.version)
            self._documents[key] = document.model_copy(deep=True)
            return True

    def get_document(
        self,
        document_id: str,
        version: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> Optional[BusinessKnowledgeDocument]:
        with self._lock:
            if version is not None:
                doc = self._documents.get((document_id, version))
                if doc is None:
                    return None
                if not self._check_org_access(doc.organization_id, organization_id):
                    return None
                return doc.model_copy(deep=True)

            # Version not specified: prefer ACTIVE version, otherwise latest by updated_at
            candidates = [
                d for (did, _), d in self._documents.items()
                if did == document_id and self._check_org_access(d.organization_id, organization_id)
            ]
            if not candidates:
                return None

            active_docs = [d for d in candidates if d.status == DocumentStatus.ACTIVE]
            if active_docs:
                # Return newest active version
                chosen = sorted(active_docs, key=lambda d: (d.version, d.updated_at), reverse=True)[0]
                return chosen.model_copy(deep=True)

            # Fallback to newest overall
            chosen = sorted(candidates, key=lambda d: (d.version, d.updated_at), reverse=True)[0]
            return chosen.model_copy(deep=True)

    def list_documents(
        self,
        organization_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
        domain: Optional[KnowledgeDomain] = None,
    ) -> List[BusinessKnowledgeDocument]:
        with self._lock:
            results: List[BusinessKnowledgeDocument] = []
            for doc in self._documents.values():
                if not self._check_org_access(doc.organization_id, organization_id):
                    continue
                if status is not None and doc.status != status:
                    continue
                if domain is not None and doc.domain != domain:
                    continue
                results.append(doc.model_copy(deep=True))
            return sorted(results, key=lambda d: (d.document_id, d.version))

    def add_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        with self._lock:
            added = 0
            for chk in chunks:
                key = (chk.document_id, chk.document_version)
                if key not in self._chunks:
                    self._chunks[key] = []
                # Replace if chunk_id already exists in this document version
                self._chunks[key] = [c for c in self._chunks[key] if c.chunk_id != chk.chunk_id]
                self._chunks[key].append(chk.model_copy(deep=True))
                self._chunk_by_id[chk.chunk_id] = chk.model_copy(deep=True)
                added += 1
            return added

    def get_chunks(
        self,
        document_id: str,
        version: Optional[str] = None,
    ) -> List[KnowledgeChunk]:
        with self._lock:
            if version is not None:
                chunks = self._chunks.get((document_id, version), [])
                return [c.model_copy(deep=True) for c in sorted(chunks, key=lambda x: x.sequence_number)]

            # If no version specified, find active document version
            doc = self.get_document(document_id)
            if not doc:
                return []
            chunks = self._chunks.get((document_id, doc.version), [])
            return [c.model_copy(deep=True) for c in sorted(chunks, key=lambda x: x.sequence_number)]

    def get_all_chunks(
        self,
        organization_id: Optional[str] = None,
        include_inactive: bool = False,
    ) -> List[KnowledgeChunk]:
        with self._lock:
            results: List[KnowledgeChunk] = []
            for chk in self._chunk_by_id.values():
                if not self._check_org_access(chk.organization_id, organization_id):
                    continue
                if not include_inactive and chk.status != DocumentStatus.ACTIVE:
                    continue
                results.append(chk.model_copy(deep=True))
            return sorted(results, key=lambda c: (c.document_id, c.document_version, c.sequence_number))

    def delete_document(
        self,
        document_id: str,
        version: Optional[str] = None,
    ) -> bool:
        with self._lock:
            if version is not None:
                key = (document_id, version)
                doc_existed = key in self._documents
                self._documents.pop(key, None)
                chunks = self._chunks.pop(key, [])
                for chk in chunks:
                    self._chunk_by_id.pop(chk.chunk_id, None)
                return doc_existed

            # Delete all versions
            keys_to_del = [k for k in self._documents if k[0] == document_id]
            if not keys_to_del:
                return False
            for k in keys_to_del:
                self._documents.pop(k, None)
                chunks = self._chunks.pop(k, [])
                for chk in chunks:
                    self._chunk_by_id.pop(chk.chunk_id, None)
            return True

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._chunks.clear()
            self._chunk_by_id.clear()

    @staticmethod
    def _check_org_access(doc_org: Optional[str], query_org: Optional[str]) -> bool:
        """Evaluate organization tenancy isolation.

        Rules:
        - If query specifies an organization, it can see global documents (doc_org is None)
          AND documents belonging to its own organization (doc_org == query_org).
        - If query specifies an organization, it CANNOT see documents from a different organization!
        - If query does NOT specify an organization (e.g. system/public admin query), it can only
          see global documents (doc_org is None) unless explicitly scoped.
        """
        if doc_org is None:
            return True
        if query_org is None:
            return False
        return doc_org == query_org
