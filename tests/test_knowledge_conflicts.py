"""Unit tests for Phase 7F Knowledge Conflict Detection Engine."""

from __future__ import annotations

import pytest

from commerce_ai.knowledge.conflicts import detect_knowledge_conflicts
from commerce_ai.knowledge.enums import (
    ConflictSeverity,
    ConflictStatus,
    DocumentStatus,
    DocumentType,
    KnowledgeDomain,
)
from commerce_ai.knowledge.ingestion import KnowledgeIngestionService
from commerce_ai.knowledge.repository import InMemoryKnowledgeRepository
from commerce_ai.knowledge.retrieval import KnowledgeRetriever
from commerce_ai.knowledge.schemas import (
    BusinessKnowledgeDocument,
    KnowledgeChunk,
    KnowledgeQueryContract,
)


def _make_chunk(doc_id: str, ver: str, content: str, domain: KnowledgeDomain = KnowledgeDomain.RETURNS, doc_type: DocumentType = DocumentType.RETURN_POLICY) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=f"CHK-{doc_id}-v{ver}-0000",
        document_id=doc_id,
        document_version=ver,
        sequence_number=0,
        title=f"Doc {doc_id}",
        content=content,
        checksum="hash",
        domain=domain,
        document_type=doc_type,
        status=DocumentStatus.ACTIVE,
    )


def test_conflict_empty_chunks_returns_empty_list() -> None:
    assert detect_knowledge_conflicts([]) == []
    chk = _make_chunk("D1", "1.0", "Return window is 30 days.")
    assert detect_knowledge_conflicts([chk]) == []


def test_conflict_multi_version_coexistence_detected() -> None:
    # Two chunks from different versions of the SAME document
    c1 = _make_chunk("DOC-POL-SAME", "1.0", "Legacy policy text.")
    c2 = _make_chunk("DOC-POL-SAME", "2.0", "Updated policy text.")

    conflicts = detect_knowledge_conflicts([c1, c2])

    assert len(conflicts) == 1
    conf = conflicts[0]
    assert conf.severity == ConflictSeverity.BLOCKING
    assert conf.status == ConflictStatus.DETECTED
    assert conf.requires_human_review is True
    assert conf.document_ids == ["DOC-POL-SAME"]
    assert "coexisting_versions" in conf.conflicting_attributes


def test_conflict_differing_numerical_rules_detected() -> None:
    # Two distinct documents in the same domain claiming different return windows
    c1 = _make_chunk("DOC-RET-ONLINE", "1.0", "Our standard customer return window is 30 days.")
    c2 = _make_chunk("DOC-RET-STORE", "1.0", "Our standard customer return window is 14 days.")

    conflicts = detect_knowledge_conflicts([c1, c2])

    assert len(conflicts) >= 1
    conf = conflicts[0]
    assert conf.severity == ConflictSeverity.HIGH
    assert "DOC-RET-ONLINE" in conf.document_ids
    assert "DOC-RET-STORE" in conf.document_ids
    assert conf.requires_human_review is True


def test_conflict_non_conflicting_chunks_produce_no_conflicts() -> None:
    # Chunks covering different topics or agreeing on values
    c1 = _make_chunk("DOC-A", "1.0", "Standard safety stock window is 14 days.", domain=KnowledgeDomain.INVENTORY, doc_type=DocumentType.REPLENISHMENT_POLICY)
    c2 = _make_chunk("DOC-B", "1.0", "Standard customer return window is 30 days.", domain=KnowledgeDomain.RETURNS, doc_type=DocumentType.RETURN_POLICY)

    conflicts = detect_knowledge_conflicts([c1, c2])
    assert len(conflicts) == 0


def test_conflict_blocking_severity_marks_retrieval_insufficient() -> None:
    repo = InMemoryKnowledgeRepository()
    svc = KnowledgeIngestionService(repository=repo)

    # Ingest two active versions of same document without supersession
    d1 = BusinessKnowledgeDocument(
        document_id="DOC-COLLIDE",
        title="Collision Doc",
        document_type=DocumentType.POLICY,
        version="1.0",
        source="Legal",
        status=DocumentStatus.ACTIVE,
        content="Return window is 10 days.",
        checksum="dummy",
    )
    d2 = BusinessKnowledgeDocument(
        document_id="DOC-COLLIDE",
        title="Collision Doc",
        document_type=DocumentType.POLICY,
        version="2.0",
        source="Legal",
        status=DocumentStatus.ACTIVE,
        content="Return window is 20 days.",
        checksum="dummy",
    )
    svc.ingest_document(d1)
    svc.ingest_document(d2)

    retriever = KnowledgeRetriever(repository=repo)
    result = retriever.retrieve(KnowledgeQueryContract(query="return window"))

    assert len(result.conflicts) > 0
    assert result.insufficient_evidence is True
    assert "Blocking knowledge conflict" in (result.insufficient_reason or "")
