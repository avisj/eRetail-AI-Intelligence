"""Controlled Knowledge Ingestion Service (Phase 7F).

Validates business document metadata, normalizes content, verifies cryptographic
checksums, performs deterministic chunking, enforces version non-overwriting,
and stores document chunks idempotently.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from commerce_ai.knowledge.chunking import DeterministicChunker
from commerce_ai.knowledge.enums import DocumentStatus
from commerce_ai.knowledge.repository import KnowledgeRepository
from commerce_ai.knowledge.schemas import (
    BusinessKnowledgeDocument,
    KnowledgeIngestionResult,
)


_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_document_content(content: str) -> str:
    """Normalize line endings and whitespace deterministically."""
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def compute_content_checksum(content: str) -> str:
    """Compute SHA-256 hash of normalized text."""
    norm = normalize_document_content(content)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class KnowledgeIngestionService:
    """Service governing validation, versioning, and chunking of knowledge documents."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        chunker: Optional[DeterministicChunker] = None,
    ) -> None:
        self.repository = repository
        self.chunker = chunker or DeterministicChunker()
        self._ingestion_log: List[Dict[str, Any]] = []

    def ingest_document(
        self,
        document: BusinessKnowledgeDocument,
        supersede_prior: bool = False,
    ) -> KnowledgeIngestionResult:
        """Validate, version, chunk, and index a business knowledge document."""
        # 1. Metadata & Date Validation
        self._validate_document_metadata(document)

        # 2. Content Normalization & Checksum
        normalized_content = normalize_document_content(document.content)
        calculated_checksum = compute_content_checksum(normalized_content)

        # 3. Check existing version in repository
        existing = self.repository.get_document(
            document_id=document.document_id,
            version=document.version,
            organization_id=document.organization_id,
        )

        if existing is not None:
            if existing.checksum == calculated_checksum:
                # Idempotent replay: return unchanged receipt
                existing_chunks = self.repository.get_chunks(document.document_id, version=document.version)
                return KnowledgeIngestionResult(
                    document_id=document.document_id,
                    version=document.version,
                    status="REPLAYED_UNCHANGED",
                    chunk_count=len(existing_chunks),
                    checksum=calculated_checksum,
                    is_idempotent_replay=True,
                    message="Document version already indexed with identical checksum; idempotent replay complete.",
                )
            else:
                # Checksum differs -> forbidden silent overwrite
                raise ValueError(
                    f"Document '{document.document_id}' version '{document.version}' already exists with differing "
                    f"checksum ({existing.checksum[:8]}... vs {calculated_checksum[:8]}...). "
                    f"Silent modification of existing versions is prohibited. Please bump document version."
                )

        # 4. Handle Supersession of Prior Active Versions if requested
        if supersede_prior and document.status == DocumentStatus.ACTIVE:
            self._supersede_prior_versions(document.document_id, document.version, document.organization_id)

        # 5. Build standardized document with normalized content & checksum
        now_ts = datetime.now(timezone.utc).isoformat()
        normalized_doc = document.model_copy(
            update={
                "content": normalized_content,
                "checksum": calculated_checksum,
                "created_at": document.created_at or now_ts,
                "updated_at": now_ts,
            }
        )

        # 6. Chunk content
        chunks = self.chunker.chunk_document(normalized_doc)

        # 7. Store document and chunks
        self.repository.add_document(normalized_doc)
        self.repository.add_chunks(chunks)

        # 8. Record audit lineage
        self._ingestion_log.append({
            "timestamp": now_ts,
            "document_id": document.document_id,
            "version": document.version,
            "organization_id": document.organization_id,
            "chunk_count": len(chunks),
            "checksum": calculated_checksum,
            "superseded_prior": supersede_prior,
        })

        return KnowledgeIngestionResult(
            document_id=document.document_id,
            version=document.version,
            status="INGESTED",
            chunk_count=len(chunks),
            checksum=calculated_checksum,
            is_idempotent_replay=False,
            message=f"Successfully ingested and indexed {len(chunks)} chunks.",
        )

    def _validate_document_metadata(self, document: BusinessKnowledgeDocument) -> None:
        """Validate dates and core business constraints."""
        if document.effective_from:
            if not _DATE_REGEX.match(document.effective_from):
                raise ValueError(f"effective_from '{document.effective_from}' must follow YYYY-MM-DD calendar format.")
        if document.effective_to:
            if not _DATE_REGEX.match(document.effective_to):
                raise ValueError(f"effective_to '{document.effective_to}' must follow YYYY-MM-DD calendar format.")
        if document.effective_from and document.effective_to:
            if document.effective_from > document.effective_to:
                raise ValueError(
                    f"effective_from ({document.effective_from}) cannot be after effective_to ({document.effective_to})."
                )

    def _supersede_prior_versions(
        self,
        document_id: str,
        current_version: str,
        organization_id: Optional[str],
    ) -> None:
        """Mark earlier active versions of the same document as SUPERSEDED."""
        existing_docs = self.repository.list_documents(organization_id=organization_id)
        for doc in existing_docs:
            if doc.document_id == document_id and doc.version != current_version and doc.status == DocumentStatus.ACTIVE:
                updated_doc = doc.model_copy(update={"status": DocumentStatus.SUPERSEDED})
                self.repository.add_document(updated_doc)
                # Also update status of associated chunks
                chunks = self.repository.get_chunks(doc.document_id, version=doc.version)
                superseded_chunks = [c.model_copy(update={"status": DocumentStatus.SUPERSEDED}) for c in chunks]
                self.repository.add_chunks(superseded_chunks)

    def get_ingestion_log(self) -> List[Dict[str, Any]]:
        """Return history of ingestion events."""
        return list(self._ingestion_log)
