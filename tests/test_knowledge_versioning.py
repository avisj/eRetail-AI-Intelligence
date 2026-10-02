"""Unit tests for Phase 7F Document Versioning and Effective Calendar Dates."""

from __future__ import annotations

import pytest

from commerce_ai.knowledge.enums import DocumentStatus, DocumentType, KnowledgeDomain
from commerce_ai.knowledge.ingestion import KnowledgeIngestionService
from commerce_ai.knowledge.repository import InMemoryKnowledgeRepository
from commerce_ai.knowledge.schemas import BusinessKnowledgeDocument


@pytest.fixture
def repo_and_service() -> tuple[InMemoryKnowledgeRepository, KnowledgeIngestionService]:
    repo = InMemoryKnowledgeRepository()
    svc = KnowledgeIngestionService(repository=repo)
    return repo, svc


def test_versioning_explicit_version_retrieval(repo_and_service: tuple[InMemoryKnowledgeRepository, KnowledgeIngestionService]) -> None:
    repo, svc = repo_and_service

    doc_v1 = BusinessKnowledgeDocument(
        document_id="DOC-RET-POL",
        title="Return Policy",
        document_type=DocumentType.RETURN_POLICY,
        version="1.0",
        source="Customer Operations",
        domain=KnowledgeDomain.RETURNS,
        content="Items can be returned within 14 days.",
        checksum="dummy",
    )
    doc_v2 = BusinessKnowledgeDocument(
        document_id="DOC-RET-POL",
        title="Return Policy",
        document_type=DocumentType.RETURN_POLICY,
        version="2.0",
        source="Customer Operations",
        domain=KnowledgeDomain.RETURNS,
        content="Items can be returned within 30 days.",
        checksum="dummy",
    )

    svc.ingest_document(doc_v1)
    svc.ingest_document(doc_v2)

    res_v1 = repo.get_document("DOC-RET-POL", version="1.0")
    res_v2 = repo.get_document("DOC-RET-POL", version="2.0")

    assert res_v1 is not None and res_v1.version == "1.0"
    assert "14 days" in res_v1.content
    assert res_v2 is not None and res_v2.version == "2.0"
    assert "30 days" in res_v2.content


def test_versioning_default_selects_active_version(repo_and_service: tuple[InMemoryKnowledgeRepository, KnowledgeIngestionService]) -> None:
    repo, svc = repo_and_service

    doc_v1 = BusinessKnowledgeDocument(
        document_id="DOC-REP",
        title="Replenishment Policy",
        document_type=DocumentType.REPLENISHMENT_POLICY,
        version="1.0",
        source="Supply Chain",
        status=DocumentStatus.SUPERSEDED,
        content="Lead time threshold is 7 days.",
        checksum="dummy",
    )
    doc_v2 = BusinessKnowledgeDocument(
        document_id="DOC-REP",
        title="Replenishment Policy",
        document_type=DocumentType.REPLENISHMENT_POLICY,
        version="2.0",
        source="Supply Chain",
        status=DocumentStatus.ACTIVE,
        content="Lead time threshold is 14 days.",
        checksum="dummy",
    )

    svc.ingest_document(doc_v1)
    svc.ingest_document(doc_v2)

    # When version is omitted, active version must be returned
    active_doc = repo.get_document("DOC-REP")
    assert active_doc is not None
    assert active_doc.version == "2.0"
    assert "14 days" in active_doc.content


def test_versioning_supersede_prior_flag(repo_and_service: tuple[InMemoryKnowledgeRepository, KnowledgeIngestionService]) -> None:
    repo, svc = repo_and_service

    doc_v1 = BusinessKnowledgeDocument(
        document_id="DOC-SOP",
        title="Receiving SOP",
        document_type=DocumentType.SOP,
        version="1.0",
        source="Warehouse",
        status=DocumentStatus.ACTIVE,
        content="Check inbound pallets manually.",
        checksum="dummy",
    )
    doc_v2 = BusinessKnowledgeDocument(
        document_id="DOC-SOP",
        title="Receiving SOP",
        document_type=DocumentType.SOP,
        version="2.0",
        source="Warehouse",
        status=DocumentStatus.ACTIVE,
        content="Scan inbound pallets with RFID.",
        checksum="dummy",
    )

    svc.ingest_document(doc_v1)
    assert repo.get_document("DOC-SOP", version="1.0").status == DocumentStatus.ACTIVE

    # Ingest v2 with supersede_prior=True
    svc.ingest_document(doc_v2, supersede_prior=True)

    v1_updated = repo.get_document("DOC-SOP", version="1.0")
    v2_doc = repo.get_document("DOC-SOP", version="2.0")

    assert v1_updated.status == DocumentStatus.SUPERSEDED
    assert v2_doc.status == DocumentStatus.ACTIVE


def test_effective_dates_validation_format_and_order(repo_and_service: tuple[InMemoryKnowledgeRepository, KnowledgeIngestionService]) -> None:
    _, svc = repo_and_service

    # Invalid date format
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        doc_bad_date = BusinessKnowledgeDocument(
            document_id="DOC-ERR-1",
            title="Bad Date",
            document_type=DocumentType.POLICY,
            source="Legal",
            effective_from="10/15/2026",
            content="Content",
            checksum="dummy",
        )
        svc.ingest_document(doc_bad_date)

    # Inverted date window
    with pytest.raises(ValueError, match="cannot be after"):
        doc_inv_date = BusinessKnowledgeDocument(
            document_id="DOC-ERR-2",
            title="Inverted Dates",
            document_type=DocumentType.POLICY,
            source="Legal",
            effective_from="2026-12-01",
            effective_to="2026-01-01",
            content="Content",
            checksum="dummy",
        )
        svc.ingest_document(doc_inv_date)


def test_effective_dates_calendar_window_boundaries() -> None:
    from commerce_ai.knowledge.retrieval import _is_date_effective

    chk_from = "2026-06-01"
    chk_to = "2026-08-31"

    assert _is_date_effective(chk_from, chk_to, "2026-06-01") is True
    assert _is_date_effective(chk_from, chk_to, "2026-07-15") is True
    assert _is_date_effective(chk_from, chk_to, "2026-08-31") is True
    assert _is_date_effective(chk_from, chk_to, "2026-05-31") is False
    assert _is_date_effective(chk_from, chk_to, "2026-09-01") is False


def test_effective_dates_open_ended_windows() -> None:
    from commerce_ai.knowledge.retrieval import _is_date_effective

    # Only effective_from specified
    assert _is_date_effective("2026-01-01", None, "2026-05-10") is True
    assert _is_date_effective("2026-01-01", None, "2025-12-31") is False

    # Only effective_to specified
    assert _is_date_effective(None, "2026-12-31", "2026-06-15") is True
    assert _is_date_effective(None, "2026-12-31", "2027-01-01") is False

    # Neither specified
    assert _is_date_effective(None, None, "2026-10-15") is True


def test_versioning_auditable_ingestion_log(repo_and_service: tuple[InMemoryKnowledgeRepository, KnowledgeIngestionService]) -> None:
    _, svc = repo_and_service

    doc = BusinessKnowledgeDocument(
        document_id="DOC-AUD-1",
        title="Audit Policy",
        document_type=DocumentType.POLICY,
        version="1.0",
        source="Compliance",
        content="Compliance rules for eRetail.",
        checksum="dummy",
    )
    svc.ingest_document(doc)

    log = svc.get_ingestion_log()
    assert len(log) == 1
    assert log[0]["document_id"] == "DOC-AUD-1"
    assert log[0]["version"] == "1.0"
    assert "timestamp" in log[0]
