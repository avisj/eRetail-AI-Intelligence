"""Unit tests for Phase 7F BusinessKnowledgeDocument lifecycle, metadata, and serialization."""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from commerce_ai.knowledge.enums import DocumentStatus, DocumentType, KnowledgeDomain
from commerce_ai.knowledge.schemas import BusinessKnowledgeDocument


def test_document_all_types_instantiation() -> None:
    types = list(DocumentType)
    assert len(types) >= 10
    for dt in types:
        doc = BusinessKnowledgeDocument(
            document_id=f"DOC-TYPE-{dt.value}",
            title=f"Sample {dt.value}",
            document_type=dt,
            source="Legal & Ops",
            content=f"Content for {dt.value}",
            checksum="hash_sample",
        )
        assert doc.document_type == dt


def test_document_all_domains_instantiation() -> None:
    domains = list(KnowledgeDomain)
    assert len(domains) >= 10
    for dom in domains:
        doc = BusinessKnowledgeDocument(
            document_id=f"DOC-DOM-{dom.value}",
            title=f"Sample {dom.value}",
            document_type=DocumentType.POLICY,
            domain=dom,
            source="Enterprise",
            content=f"Content for {dom.value}",
            checksum="hash_sample",
        )
        assert doc.domain == dom


def test_document_json_serialization_roundtrip() -> None:
    doc = BusinessKnowledgeDocument(
        document_id="DOC-SER-01",
        title="Serialization Test Document",
        document_type=DocumentType.GLOSSARY,
        version="1.2",
        source="Analytics",
        domain=KnowledgeDomain.FINANCIAL,
        tags=["ebitda", "gross_margin", "finance"],
        content="Gross Margin is calculated as Revenue minus Cost of Goods Sold.",
        checksum="hash_ser_01",
    )
    raw_json = doc.model_dump_json()
    data = json.loads(raw_json)
    assert data["document_id"] == "DOC-SER-01"
    assert data["domain"] == "FINANCIAL"

    doc_restored = BusinessKnowledgeDocument.model_validate_json(raw_json)
    assert doc_restored == doc


def test_document_status_transitions_via_copy() -> None:
    doc = BusinessKnowledgeDocument(
        document_id="DOC-TRANS-01",
        title="Transition Policy",
        document_type=DocumentType.POLICY,
        version="1.0",
        source="Ops",
        status=DocumentStatus.DRAFT,
        content="Initial draft.",
        checksum="hash_trans",
    )
    assert doc.status == DocumentStatus.DRAFT

    doc_active = doc.model_copy(update={"status": DocumentStatus.ACTIVE})
    assert doc_active.status == DocumentStatus.ACTIVE

    doc_superseded = doc_active.model_copy(update={"status": DocumentStatus.SUPERSEDED})
    assert doc_superseded.status == DocumentStatus.SUPERSEDED


def test_document_source_uri_handling() -> None:
    doc = BusinessKnowledgeDocument(
        document_id="DOC-URI-01",
        title="Sharepoint Policy Document",
        document_type=DocumentType.POLICY,
        source="Corporate Compliance",
        source_uri="https://sharepoint.internal.eratail.ai/policies/returns-2026.pdf",
        content="Official policy on customer returns.",
        checksum="hash_uri",
    )
    assert doc.source_uri is not None
    assert doc.source_uri.startswith("https://")


def test_document_language_code() -> None:
    doc_es = BusinessKnowledgeDocument(
        document_id="DOC-LANG-ES",
        title="Política de Devoluciones",
        document_type=DocumentType.RETURN_POLICY,
        source="Servicio al Cliente",
        language="es",
        content="Los artículos pueden ser devueltos dentro de los 30 días.",
        checksum="hash_es",
    )
    assert doc_es.language == "es"


def test_document_tags_list_immutability() -> None:
    doc = BusinessKnowledgeDocument(
        document_id="DOC-TAG-01",
        title="Tagged Doc",
        document_type=DocumentType.SOP,
        source="Ops",
        tags=["tag1", "tag2"],
        content="Text",
        checksum="hash_tag",
    )
    assert len(doc.tags) == 2
    assert "tag1" in doc.tags


def test_document_timestamp_generation() -> None:
    doc = BusinessKnowledgeDocument(
        document_id="DOC-TS-01",
        title="Timestamp Test",
        document_type=DocumentType.POLICY,
        source="Ops",
        content="Text",
        checksum="hash_ts",
    )
    assert doc.created_at is not None
    assert doc.updated_at is not None
    assert "T" in doc.created_at  # ISO format


def test_document_empty_title_raises() -> None:
    with pytest.raises(ValidationError):
        BusinessKnowledgeDocument(
            document_id="DOC-BAD",
            title="",
            document_type=DocumentType.POLICY,
            source="Ops",
            content="Text",
            checksum="hash",
        )


def test_document_empty_checksum_raises() -> None:
    with pytest.raises(ValidationError):
        BusinessKnowledgeDocument(
            document_id="DOC-BAD",
            title="Title",
            document_type=DocumentType.POLICY,
            source="Ops",
            content="Text",
            checksum="",
        )


def test_document_equality() -> None:
    doc1 = BusinessKnowledgeDocument(
        document_id="DOC-EQ",
        title="Equality Doc",
        document_type=DocumentType.POLICY,
        source="Ops",
        content="Same content.",
        checksum="hash_eq",
        created_at="2026-10-01T00:00:00Z",
        updated_at="2026-10-01T00:00:00Z",
    )
    doc2 = BusinessKnowledgeDocument(
        document_id="DOC-EQ",
        title="Equality Doc",
        document_type=DocumentType.POLICY,
        source="Ops",
        content="Same content.",
        checksum="hash_eq",
        created_at="2026-10-01T00:00:00Z",
        updated_at="2026-10-01T00:00:00Z",
    )
    assert doc1 == doc2
