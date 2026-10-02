"""Unit tests for Phase 7F Knowledge Ingestion Service."""

from __future__ import annotations

import pytest

from commerce_ai.knowledge.enums import DocumentStatus, DocumentType, KnowledgeDomain
from commerce_ai.knowledge.ingestion import (
    KnowledgeIngestionService,
    compute_content_checksum,
    normalize_document_content,
)
from commerce_ai.knowledge.repository import InMemoryKnowledgeRepository
from commerce_ai.knowledge.schemas import BusinessKnowledgeDocument


@pytest.fixture
def repo_and_service() -> tuple[InMemoryKnowledgeRepository, KnowledgeIngestionService]:
    repo = InMemoryKnowledgeRepository()
    svc = KnowledgeIngestionService(repository=repo)
    return repo, svc


def test_normalization_whitespace_and_newlines() -> None:
    text_crlf = "Heading\r\nLine 1   \r\n\r\nLine 2  \r\n"
    norm = normalize_document_content(text_crlf)
    assert "\r" not in norm
    assert "Line 1   " not in norm
    assert "Line 1\n\nLine 2" in norm


def test_compute_checksum_deterministic() -> None:
    t1 = "Standardized business policy text."
    t2 = "Standardized business policy text.   \r\n"
    # Should normalize to identical hash
    assert compute_content_checksum(t1) == compute_content_checksum(t2)


def test_ingestion_successful_initial_run(repo_and_service: tuple[InMemoryKnowledgeRepository, KnowledgeIngestionService]) -> None:
    repo, svc = repo_and_service

    doc = BusinessKnowledgeDocument(
        document_id="DOC-ING-01",
        title="Ingestion Test Policy",
        document_type=DocumentType.POLICY,
        version="1.0",
        source="Engineering",
        domain=KnowledgeDomain.DATA_QUALITY,
        content="All data pipelines must log daily verification hashes.",
        checksum="dummy",
    )

    result = svc.ingest_document(doc)

    assert result.status == "INGESTED"
    assert result.is_idempotent_replay is False
    assert result.chunk_count >= 1

    stored = repo.get_document("DOC-ING-01", version="1.0")
    assert stored is not None
    assert stored.checksum == result.checksum


def test_ingestion_idempotent_replay(repo_and_service: tuple[InMemoryKnowledgeRepository, KnowledgeIngestionService]) -> None:
    repo, svc = repo_and_service

    doc = BusinessKnowledgeDocument(
        document_id="DOC-ING-02",
        title="Idempotence Policy",
        document_type=DocumentType.SOP,
        version="1.0",
        source="Ops",
        content="Step 1: Check inbound manifests.\nStep 2: Scan items.",
        checksum="dummy",
    )

    res1 = svc.ingest_document(doc)
    assert res1.is_idempotent_replay is False

    # Second ingestion of identical document
    res2 = svc.ingest_document(doc)
    assert res2.is_idempotent_replay is True
    assert res2.status == "REPLAYED_UNCHANGED"
    assert res2.chunk_count == res1.chunk_count

    # Verify chunks weren't duplicated in repository
    chunks = repo.get_chunks("DOC-ING-02", version="1.0")
    assert len(chunks) == res1.chunk_count


def test_ingestion_differing_checksum_overwrite_rejected(repo_and_service: tuple[InMemoryKnowledgeRepository, KnowledgeIngestionService]) -> None:
    repo, svc = repo_and_service

    doc_v1 = BusinessKnowledgeDocument(
        document_id="DOC-ING-03",
        title="Immutable Policy",
        document_type=DocumentType.POLICY,
        version="1.0",
        source="Legal",
        content="Original legal text.",
        checksum="dummy",
    )
    svc.ingest_document(doc_v1)

    doc_v1_tampered = BusinessKnowledgeDocument(
        document_id="DOC-ING-03",
        title="Immutable Policy",
        document_type=DocumentType.POLICY,
        version="1.0",
        source="Legal",
        content="Altered legal text without version bump.",
        checksum="dummy",
    )

    with pytest.raises(ValueError, match="already exists with differing checksum"):
        svc.ingest_document(doc_v1_tampered)


def test_ingestion_updates_status_on_supersede(repo_and_service: tuple[InMemoryKnowledgeRepository, KnowledgeIngestionService]) -> None:
    repo, svc = repo_and_service

    doc_v1 = BusinessKnowledgeDocument(
        document_id="DOC-ING-04",
        title="Warehouse Guide",
        document_type=DocumentType.WAREHOUSE_GUIDE,
        version="1.0",
        source="Logistics",
        status=DocumentStatus.ACTIVE,
        content="Old warehouse safety guidelines.",
        checksum="dummy",
    )
    doc_v2 = BusinessKnowledgeDocument(
        document_id="DOC-ING-04",
        title="Warehouse Guide",
        document_type=DocumentType.WAREHOUSE_GUIDE,
        version="2.0",
        source="Logistics",
        status=DocumentStatus.ACTIVE,
        content="New warehouse safety guidelines with automated forklifts.",
        checksum="dummy",
    )

    svc.ingest_document(doc_v1)
    svc.ingest_document(doc_v2, supersede_prior=True)

    v1_chunks = repo.get_chunks("DOC-ING-04", version="1.0")
    v2_chunks = repo.get_chunks("DOC-ING-04", version="2.0")

    assert all(c.status == DocumentStatus.SUPERSEDED for c in v1_chunks)
    assert all(c.status == DocumentStatus.ACTIVE for c in v2_chunks)
