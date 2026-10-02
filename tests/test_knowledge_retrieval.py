"""Unit tests for Phase 7F Knowledge Retrieval Engine."""

from __future__ import annotations

import pytest

from commerce_ai.knowledge.enums import (
    DocumentStatus,
    DocumentType,
    KnowledgeDomain,
    RetrievalMethod,
)
from commerce_ai.knowledge.ingestion import KnowledgeIngestionService
from commerce_ai.knowledge.repository import InMemoryKnowledgeRepository
from commerce_ai.knowledge.retrieval import KnowledgeRetriever
from commerce_ai.knowledge.schemas import (
    BusinessKnowledgeDocument,
    KnowledgeQueryContract,
)


@pytest.fixture
def populated_retriever() -> tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]:
    repo = InMemoryKnowledgeRepository()
    svc = KnowledgeIngestionService(repository=repo)

    doc_ret = BusinessKnowledgeDocument(
        document_id="DOC-RET-POLICY",
        title="Customer Return and Refund Policy",
        document_type=DocumentType.RETURN_POLICY,
        version="1.0",
        source="Customer Experience",
        domain=KnowledgeDomain.RETURNS,
        tags=["return", "refund", "customer"],
        effective_from="2026-01-01",
        effective_to="2026-12-31",
        status=DocumentStatus.ACTIVE,
        content=(
            "# Return Timeframe\nCustomers may return unworn items within 30 days of purchase.\n\n"
            "# Damaged Goods\nDamaged items will receive a full immediate refund upon photo verification."
        ),
        checksum="dummy",
    )

    doc_rep = BusinessKnowledgeDocument(
        document_id="DOC-REP-POLICY",
        title="Automated Replenishment Guidelines",
        document_type=DocumentType.REPLENISHMENT_POLICY,
        version="1.0",
        source="Supply Chain",
        domain=KnowledgeDomain.INVENTORY,
        tags=["replenishment", "reorder", "safety stock"],
        effective_from="2026-06-01",
        effective_to="2026-12-31",
        status=DocumentStatus.ACTIVE,
        content=(
            "# Safety Stock\nSafety stock is calibrated to 14 days of average forecast demand.\n\n"
            "# Reorder Point\nReorder point triggers when inventory falls below 200 units."
        ),
        checksum="dummy",
    )

    doc_old = BusinessKnowledgeDocument(
        document_id="DOC-OLD-POLICY",
        title="Archived Warehouse SOP",
        document_type=DocumentType.SOP,
        version="0.9",
        source="Warehouse",
        domain=KnowledgeDomain.OPERATIONS,
        status=DocumentStatus.ARCHIVED,
        content="Old forklift operation rules.",
        checksum="dummy",
    )

    svc.ingest_document(doc_ret)
    svc.ingest_document(doc_rep)
    svc.ingest_document(doc_old)

    retriever = KnowledgeRetriever(repository=repo)
    return repo, retriever


def test_retrieval_exact_keyword_match(populated_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = populated_retriever
    contract = KnowledgeQueryContract(query="What is the return timeframe?")
    result = retriever.retrieve(contract)

    assert result.insufficient_evidence is False
    assert len(result.chunks) >= 1
    assert result.chunks[0].document_id == "DOC-RET-POLICY"
    assert result.retrieval_method == RetrievalMethod.LEXICAL_TOKEN_MATCH
    assert result.scores[0] > 0.3


def test_retrieval_domain_filtering(populated_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = populated_retriever
    # Query return but restrict domain to INVENTORY -> should return nothing
    contract = KnowledgeQueryContract(query="return policy", domain=KnowledgeDomain.INVENTORY)
    result = retriever.retrieve(contract)

    assert result.insufficient_evidence is True
    assert len(result.chunks) == 0


def test_retrieval_document_type_filtering(populated_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = populated_retriever
    # Restrict to REPLENISHMENT_POLICY
    contract = KnowledgeQueryContract(
        query="safety stock trigger",
        document_types=[DocumentType.REPLENISHMENT_POLICY],
    )
    result = retriever.retrieve(contract)

    assert result.insufficient_evidence is False
    assert result.chunks[0].document_type == DocumentType.REPLENISHMENT_POLICY
    assert result.chunks[0].document_id == "DOC-REP-POLICY"


def test_retrieval_effective_date_inside_window(populated_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = populated_retriever
    # 2026-07-15 is within 2026-06-01 .. 2026-12-31 for Replenishment
    contract = KnowledgeQueryContract(
        query="safety stock",
        effective_date="2026-07-15",
    )
    result = retriever.retrieve(contract)

    assert result.insufficient_evidence is False
    assert any(c.document_id == "DOC-REP-POLICY" for c in result.chunks)


def test_retrieval_effective_date_outside_window(populated_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = populated_retriever
    # 2026-03-01 is BEFORE 2026-06-01 effective_from for Replenishment
    contract = KnowledgeQueryContract(
        query="safety stock calibration",
        effective_date="2026-03-01",
    )
    result = retriever.retrieve(contract)

    # Replenishment was not effective in March 2026
    assert not any(c.document_id == "DOC-REP-POLICY" for c in result.chunks)


def test_retrieval_inactive_excluded_by_default(populated_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = populated_retriever
    contract = KnowledgeQueryContract(query="forklift operation rules")
    result = retriever.retrieve(contract)

    # DOC-OLD-POLICY is ARCHIVED -> must not be retrieved by default
    assert result.insufficient_evidence is True
    assert len(result.chunks) == 0


def test_retrieval_include_inactive_flag(populated_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = populated_retriever
    contract = KnowledgeQueryContract(query="forklift operation rules", include_inactive=True)
    result = retriever.retrieve(contract)

    assert result.insufficient_evidence is False
    assert result.chunks[0].document_id == "DOC-OLD-POLICY"


def test_retrieval_top_k_bound(populated_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = populated_retriever
    contract = KnowledgeQueryContract(query="return safety stock", top_k=1)
    result = retriever.retrieve(contract)

    assert len(result.chunks) <= 1


def test_retrieval_minimum_relevance_threshold(populated_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = populated_retriever
    # Very high threshold that nothing meets
    contract = KnowledgeQueryContract(query="return policy", minimum_relevance=0.999)
    result = retriever.retrieve(contract)

    assert result.insufficient_evidence is True
    assert "minimum relevance threshold" in (result.insufficient_reason or "")


def test_retrieval_evidence_and_citation_generation(populated_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = populated_retriever
    contract = KnowledgeQueryContract(query="damaged goods refund")
    result = retriever.retrieve(contract)

    assert len(result.citations) > 0
    assert len(result.evidence) > 0
    assert result.citations[0].document_title == "Customer Return and Refund Policy"
    assert "damaged" in result.citations[0].snippet.lower()
    assert result.evidence[0].provenance_type.value == "DOCUMENT_DERIVED"
