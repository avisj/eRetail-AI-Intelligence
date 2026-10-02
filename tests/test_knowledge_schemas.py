"""Unit tests for Phase 7F Knowledge Schemas and Data Models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from commerce_ai.knowledge.enums import (
    ConflictSeverity,
    ConflictStatus,
    DocumentStatus,
    DocumentType,
    KnowledgeDomain,
    KnowledgeProvenanceType,
    QueryCapability,
    RetrievalMethod,
)
from commerce_ai.knowledge.schemas import (
    BusinessKnowledgeDocument,
    HybridQueryResult,
    KnowledgeChunk,
    KnowledgeCitation,
    KnowledgeConflict,
    KnowledgeEvidence,
    KnowledgeGovernance,
    KnowledgeIngestionResult,
    KnowledgeQueryContract,
    KnowledgeRetrievalResult,
)


def test_business_knowledge_document_valid() -> None:
    doc = BusinessKnowledgeDocument(
        document_id="DOC-POL-01",
        title="Standard Inventory Replenishment Policy",
        document_type=DocumentType.REPLENISHMENT_POLICY,
        version="1.0",
        source="Supply Chain Operations",
        domain=KnowledgeDomain.INVENTORY,
        content="All regional warehouses must maintain a minimum 14-day safety stock.",
        checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    assert doc.document_id == "DOC-POL-01"
    assert doc.status == DocumentStatus.ACTIVE
    assert doc.language == "en"
    assert doc.domain == KnowledgeDomain.INVENTORY


def test_business_knowledge_document_empty_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        BusinessKnowledgeDocument(
            document_id="",
            title="Valid Title",
            document_type=DocumentType.POLICY,
            source="Ops",
            content="Content",
            checksum="hash",
        )
    with pytest.raises(ValidationError):
        BusinessKnowledgeDocument(
            document_id="DOC-01",
            title="   ",
            document_type=DocumentType.POLICY,
            source="Ops",
            content="Content",
            checksum="hash",
        )


def test_business_knowledge_document_immutability() -> None:
    doc = BusinessKnowledgeDocument(
        document_id="DOC-01",
        title="Title",
        document_type=DocumentType.SOP,
        source="Ops",
        content="Content",
        checksum="hash",
    )
    with pytest.raises(ValidationError):
        doc.title = "New Title"  # type: ignore # frozen


def test_knowledge_chunk_valid() -> None:
    chk = KnowledgeChunk(
        chunk_id="CHK-DOC-01-v1.0-0000",
        document_id="DOC-01",
        document_version="1.0",
        sequence_number=0,
        title="Return SOP",
        section_heading="Damaged Items",
        content="Damaged items must be photographed within 48 hours of receipt.",
        checksum="hash123",
        document_type=DocumentType.RETURN_POLICY,
    )
    assert chk.chunk_id == "CHK-DOC-01-v1.0-0000"
    assert chk.section_heading == "Damaged Items"
    assert chk.status == DocumentStatus.ACTIVE


def test_knowledge_citation_model() -> None:
    cit = KnowledgeCitation(
        citation_id="CIT-001",
        document_id="DOC-01",
        document_version="1.0",
        chunk_id="CHK-001",
        document_title="Return Policy",
        source="Customer Service",
        relevance_score=0.88,
        snippet="Items may be returned within 30 days.",
    )
    assert cit.relevance_score == 0.88
    assert cit.document_title == "Return Policy"


def test_knowledge_evidence_model() -> None:
    ev = KnowledgeEvidence(
        evidence_id="EVD-01",
        document_id="DOC-01",
        document_version="1.0",
        chunk_id="CHK-01",
        title="Replenishment Guide",
        source="Inventory Ops",
        retrieval_method=RetrievalMethod.LEXICAL_TOKEN_MATCH,
        relevance_score=0.92,
        checksum="chk999",
        provenance_type=KnowledgeProvenanceType.DOCUMENT_DERIVED,
        content="Safety stock targets are reviewed weekly.",
    )
    assert ev.provenance_type == KnowledgeProvenanceType.DOCUMENT_DERIVED
    assert ev.retrieval_method == RetrievalMethod.LEXICAL_TOKEN_MATCH


def test_knowledge_conflict_model() -> None:
    conf = KnowledgeConflict(
        conflict_id="CONF-01",
        document_ids=["DOC-A", "DOC-B"],
        chunk_ids=["CHK-A1", "CHK-B1"],
        conflicting_attributes={"DOC-A": "30 days", "DOC-B": "14 days"},
        description="Return window mismatch.",
        severity=ConflictSeverity.HIGH,
        status=ConflictStatus.DETECTED,
        requires_human_review=True,
    )
    assert conf.severity == ConflictSeverity.HIGH
    assert conf.requires_human_review is True


def test_knowledge_query_contract_valid_and_bounds() -> None:
    qc = KnowledgeQueryContract(query="What is the return window?", top_k=10, minimum_relevance=0.1)
    assert qc.top_k == 10
    assert qc.minimum_relevance == 0.1
    assert qc.include_inactive is False

    with pytest.raises(ValidationError):
        KnowledgeQueryContract(query="")
    with pytest.raises(ValidationError):
        KnowledgeQueryContract(query="Valid", top_k=0)
    with pytest.raises(ValidationError):
        KnowledgeQueryContract(query="Valid", top_k=51)
    with pytest.raises(ValidationError):
        KnowledgeQueryContract(query="Valid", minimum_relevance=-0.1)
    with pytest.raises(ValidationError):
        KnowledgeQueryContract(query="Valid", minimum_relevance=1.5)


def test_knowledge_retrieval_result_model() -> None:
    res = KnowledgeRetrievalResult(
        query="return policy",
        insufficient_evidence=True,
        insufficient_reason="No documents found.",
    )
    assert res.insufficient_evidence is True
    assert len(res.chunks) == 0


def test_knowledge_governance_invariants() -> None:
    gov = KnowledgeGovernance()
    assert gov.read_only is True
    assert gov.action_execution is False
    assert gov.execution_allowed is False
    assert gov.no_ranking_enforced is True
    assert gov.metric_fabrication_prohibited is True
    assert gov.recommendation_fabrication_prohibited is True


def test_knowledge_ingestion_result_model() -> None:
    res = KnowledgeIngestionResult(
        document_id="DOC-01",
        version="1.0",
        status="INGESTED",
        chunk_count=5,
        checksum="abc123",
        is_idempotent_replay=False,
        message="OK",
    )
    assert res.status == "INGESTED"
    assert res.chunk_count == 5


def test_hybrid_query_result_model() -> None:
    res = HybridQueryResult(
        query="What is current stock and the policy?",
        query_capability=QueryCapability.HYBRID_QUERY,
        provenance_type=KnowledgeProvenanceType.MIXED,
        data_summary="Stock is 42 units.",
        knowledge_summary="Policy requires 50 units.",
        combined_narrative="Stock is 42 units, which is below the policy required 50 units.",
    )
    assert res.query_capability == QueryCapability.HYBRID_QUERY
    assert res.provenance_type == KnowledgeProvenanceType.MIXED
