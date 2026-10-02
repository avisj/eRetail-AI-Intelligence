"""Unit tests for Phase 7F Deterministic Chunking Engine."""

from __future__ import annotations

import pytest

from commerce_ai.knowledge.chunking import DeterministicChunker
from commerce_ai.knowledge.enums import DocumentStatus, DocumentType, KnowledgeDomain
from commerce_ai.knowledge.schemas import BusinessKnowledgeDocument


def _make_sample_doc(content: str, doc_id: str = "DOC-TEST-01", version: str = "1.0") -> BusinessKnowledgeDocument:
    return BusinessKnowledgeDocument(
        document_id=doc_id,
        title="Test Operational Manual",
        document_type=DocumentType.SOP,
        version=version,
        source="Operations",
        domain=KnowledgeDomain.OPERATIONS,
        tags=["sop", "operations", "manual"],
        content=content,
        checksum="dummy_checksum",
    )


def test_chunker_empty_content_returns_empty_list() -> None:
    chunker = DeterministicChunker()
    # BusinessKnowledgeDocument schema rejects empty/whitespace content
    with pytest.raises(Exception):
        _make_sample_doc("   \n\n  ")

    class DummyDoc:
        content = ""
    assert chunker.chunk_document(DummyDoc()) == []  # type: ignore


def test_chunker_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        DeterministicChunker(chunk_size=30)
    with pytest.raises(ValueError):
        DeterministicChunker(chunk_size=500, chunk_overlap=500)
    with pytest.raises(ValueError):
        DeterministicChunker(chunk_size=500, chunk_overlap=-10)


def test_chunker_deterministic_repeated_ingestion() -> None:
    content = (
        "# Overview\nThis is the operational overview.\n\n"
        "## Procedure A\nFollow steps 1, 2, and 3 carefully to ensure warehouse compliance.\n\n"
        "## Procedure B\nVerify all inbound shipments against physical packing slips."
    )
    doc = _make_sample_doc(content)
    chunker = DeterministicChunker(chunk_size=300, chunk_overlap=50)

    run_1 = chunker.chunk_document(doc)
    run_2 = chunker.chunk_document(doc)

    assert len(run_1) == len(run_2)
    for c1, c2 in zip(run_1, run_2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.content == c2.content
        assert c1.checksum == c2.checksum
        assert c1.section_heading == c2.section_heading


def test_chunker_stable_chunk_id_format() -> None:
    doc = _make_sample_doc("Short content", doc_id="DOC-POL-99", version="2.5")
    chunker = DeterministicChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "CHK-DOC-POL-99-v2.5-0000"
    assert chunks[0].sequence_number == 0


def test_chunker_heading_extraction_and_preservation() -> None:
    content = (
        "# Replenishment Thresholds\n"
        "Safety stock levels must be reviewed every Monday morning.\n\n"
        "## Order Frequency\n"
        "Purchase orders should be placed on a bi-weekly cadence."
    )
    doc = _make_sample_doc(content)
    chunker = DeterministicChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    headings = [c.section_heading for c in chunks]
    assert "Replenishment Thresholds" in headings
    assert "Order Frequency" in headings


def test_chunker_metadata_propagation() -> None:
    doc = BusinessKnowledgeDocument(
        document_id="DOC-RET-01",
        title="Customer Return Policy",
        document_type=DocumentType.RETURN_POLICY,
        version="3.0",
        source="Customer Care",
        organization_id="ORG_RETAIL_1",
        domain=KnowledgeDomain.RETURNS,
        tags=["returns", "refunds"],
        effective_from="2026-01-01",
        effective_to="2026-12-31",
        status=DocumentStatus.ACTIVE,
        content="Items in original packaging can be returned within 30 days of purchase.",
        checksum="chk_returns",
    )
    chunker = DeterministicChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    chk = chunks[0]
    assert chk.organization_id == "ORG_RETAIL_1"
    assert chk.domain == KnowledgeDomain.RETURNS
    assert chk.document_type == DocumentType.RETURN_POLICY
    assert chk.effective_from == "2026-01-01"
    assert chk.effective_to == "2026-12-31"
    assert chk.status == DocumentStatus.ACTIVE
    assert "returns" in chk.tags


def test_chunker_long_text_sentence_boundary_splitting() -> None:
    sentence = "All inventory items must be checked for barcode readability before stowing. "
    long_content = sentence * 25  # ~1900 chars
    doc = _make_sample_doc(long_content)
    chunker = DeterministicChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 3
    # Check that sentences aren't chopped in the middle of words
    for c in chunks:
        assert not c.content.startswith("before stowing")
        assert not c.content.endswith("barcode read")


def test_chunker_sequence_monotonicity() -> None:
    paragraphs = [f"Paragraph {i}: Detailed description of procedure step {i}." for i in range(15)]
    doc = _make_sample_doc("\n\n".join(paragraphs))
    chunker = DeterministicChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 5
    for idx, c in enumerate(chunks):
        assert c.sequence_number == idx
        assert c.chunk_id.endswith(f"-{idx:04d}")


def test_chunker_checksum_uniqueness_for_distinct_content() -> None:
    doc1 = _make_sample_doc("Content for section one.")
    doc2 = _make_sample_doc("Content for section two.")
    chunker = DeterministicChunker()

    c1 = chunker.chunk_document(doc1)[0]
    c2 = chunker.chunk_document(doc2)[0]

    assert c1.checksum != c2.checksum
