"""Unit tests for Phase 7F Knowledge Citations and Evidence Packaging."""

from __future__ import annotations

import pytest

from commerce_ai.knowledge.citations import build_citation, build_evidence, create_snippet
from commerce_ai.knowledge.enums import (
    DocumentStatus,
    DocumentType,
    KnowledgeDomain,
    KnowledgeProvenanceType,
    RetrievalMethod,
)
from commerce_ai.knowledge.schemas import KnowledgeChunk


@pytest.fixture
def sample_chunk() -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id="CHK-DOC-SOP-01-v1.0-0001",
        document_id="DOC-SOP-01",
        document_version="1.0",
        sequence_number=1,
        title="Warehouse Quarantine Procedures",
        section_heading="Hazardous Goods",
        page_or_location="Section: Hazardous Goods",
        content="Any compromised chemicals must be moved to Zone Z within 15 minutes of spill detection.",
        checksum="hash987654321",
        organization_id="ORG_WEST",
        domain=KnowledgeDomain.OPERATIONS,
        document_type=DocumentType.SOP,
        effective_from="2026-01-01",
        effective_to="2026-12-31",
        status=DocumentStatus.ACTIVE,
        tags=["quarantine", "hazard"],
    )


def test_create_snippet_short_text() -> None:
    text = "Short text under limits."
    snippet = create_snippet(text, max_chars=100)
    assert snippet == text


def test_create_snippet_long_text_word_boundary_truncation() -> None:
    text = "This is a very long descriptive paragraph intended to verify that snippets truncate cleanly at word boundaries rather than breaking words in half."
    snippet = create_snippet(text, max_chars=50)
    assert len(snippet) <= 53  # with '...'
    assert snippet.endswith("...")
    assert not snippet.endswith("  ...")


def test_build_citation_fields_and_traceability(sample_chunk: KnowledgeChunk) -> None:
    cit = build_citation(sample_chunk, relevance_score=0.87654)

    assert cit.citation_id.startswith("CIT-")
    assert cit.document_id == "DOC-SOP-01"
    assert cit.document_version == "1.0"
    assert cit.chunk_id == "CHK-DOC-SOP-01-v1.0-0001"
    assert cit.document_title == "Warehouse Quarantine Procedures"
    assert cit.section_heading == "Hazardous Goods"
    assert cit.relevance_score == 0.8765
    assert "compromised chemicals" in cit.snippet


def test_build_evidence_fields_and_provenance(sample_chunk: KnowledgeChunk) -> None:
    ev = build_evidence(
        chunk=sample_chunk,
        relevance_score=0.91234,
        retrieval_method=RetrievalMethod.LEXICAL_TOKEN_MATCH,
    )

    assert ev.evidence_id.startswith("EVD-KNOW-")
    assert ev.document_id == "DOC-SOP-01"
    assert ev.document_version == "1.0"
    assert ev.chunk_id == "CHK-DOC-SOP-01-v1.0-0001"
    assert ev.title == "Warehouse Quarantine Procedures"
    assert ev.retrieval_method == RetrievalMethod.LEXICAL_TOKEN_MATCH
    assert ev.relevance_score == 0.9123
    assert ev.checksum == "hash987654321"
    assert ev.provenance_type == KnowledgeProvenanceType.DOCUMENT_DERIVED
    assert ev.organization_id == "ORG_WEST"
    assert ev.content == sample_chunk.content


def test_citation_and_evidence_immutability(sample_chunk: KnowledgeChunk) -> None:
    cit = build_citation(sample_chunk, 0.75)
    ev = build_evidence(sample_chunk, 0.75, RetrievalMethod.LEXICAL_TOKEN_MATCH)

    with pytest.raises(Exception):
        cit.relevance_score = 1.0  # type: ignore # frozen
    with pytest.raises(Exception):
        ev.content = "Mutated content"  # type: ignore # frozen
