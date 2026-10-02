"""Unit tests for Phase 7F Multi-Tenant / Organization Knowledge Isolation."""

from __future__ import annotations

import pytest

from commerce_ai.knowledge.enums import DocumentStatus, DocumentType, KnowledgeDomain
from commerce_ai.knowledge.ingestion import KnowledgeIngestionService
from commerce_ai.knowledge.repository import InMemoryKnowledgeRepository
from commerce_ai.knowledge.retrieval import KnowledgeRetriever
from commerce_ai.knowledge.schemas import (
    BusinessKnowledgeDocument,
    KnowledgeQueryContract,
)


@pytest.fixture
def multi_tenant_retriever() -> tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]:
    repo = InMemoryKnowledgeRepository()
    svc = KnowledgeIngestionService(repository=repo)

    doc_org_a = BusinessKnowledgeDocument(
        document_id="DOC-ORG-A-01",
        title="Org A Proprietary Discount Policy",
        document_type=DocumentType.POLICY,
        version="1.0",
        source="Finance Org A",
        organization_id="TENANT_ALPHA",
        domain=KnowledgeDomain.FINANCIAL,
        content="Tenant Alpha offers a maximum partner discount of 25 percent.",
        checksum="dummy",
    )

    doc_org_b = BusinessKnowledgeDocument(
        document_id="DOC-ORG-B-01",
        title="Org B Confidential Markup Rules",
        document_type=DocumentType.POLICY,
        version="1.0",
        source="Finance Org B",
        organization_id="TENANT_BETA",
        domain=KnowledgeDomain.FINANCIAL,
        content="Tenant Beta applies a strict minimum 40 percent markup on luxury apparel.",
        checksum="dummy",
    )

    doc_global = BusinessKnowledgeDocument(
        document_id="DOC-GLOBAL-01",
        title="Standard Retail Safety Glossary",
        document_type=DocumentType.GLOSSARY,
        version="1.0",
        source="Standards Body",
        organization_id=None,  # Global / unassigned
        domain=KnowledgeDomain.GENERAL,
        content="OSHA compliance requires fire exits to be kept clear at all times.",
        checksum="dummy",
    )

    svc.ingest_document(doc_org_a)
    svc.ingest_document(doc_org_b)
    svc.ingest_document(doc_global)

    retriever = KnowledgeRetriever(repository=repo)
    return repo, retriever


def test_tenant_isolation_org_a_can_retrieve_own(multi_tenant_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = multi_tenant_retriever
    contract = KnowledgeQueryContract(
        query="partner discount",
        organization_id="TENANT_ALPHA",
    )
    result = retriever.retrieve(contract)

    assert result.insufficient_evidence is False
    assert len(result.chunks) >= 1
    assert result.chunks[0].document_id == "DOC-ORG-A-01"
    assert result.chunks[0].organization_id == "TENANT_ALPHA"


def test_tenant_isolation_org_b_cannot_retrieve_org_a(multi_tenant_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = multi_tenant_retriever
    # Tenant Beta queries Tenant Alpha's specific policy terms
    contract = KnowledgeQueryContract(
        query="partner discount of 25 percent",
        organization_id="TENANT_BETA",
    )
    result = retriever.retrieve(contract)

    # Must be completely hidden from Tenant Beta
    assert not any(c.document_id == "DOC-ORG-A-01" for c in result.chunks)
    assert not any(c.organization_id == "TENANT_ALPHA" for c in result.chunks)


def test_tenant_isolation_org_a_cannot_retrieve_org_b(multi_tenant_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = multi_tenant_retriever
    # Tenant Alpha queries Tenant Beta's markup terms
    contract = KnowledgeQueryContract(
        query="luxury apparel markup 40 percent",
        organization_id="TENANT_ALPHA",
    )
    result = retriever.retrieve(contract)

    assert not any(c.document_id == "DOC-ORG-B-01" for c in result.chunks)
    assert not any(c.organization_id == "TENANT_BETA" for c in result.chunks)


def test_tenant_isolation_global_knowledge_accessible_to_all(multi_tenant_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    _, retriever = multi_tenant_retriever

    contract_a = KnowledgeQueryContract(query="OSHA fire exits", organization_id="TENANT_ALPHA")
    res_a = retriever.retrieve(contract_a)
    assert any(c.document_id == "DOC-GLOBAL-01" for c in res_a.chunks)

    contract_b = KnowledgeQueryContract(query="OSHA fire exits", organization_id="TENANT_BETA")
    res_b = retriever.retrieve(contract_b)
    assert any(c.document_id == "DOC-GLOBAL-01" for c in res_b.chunks)


def test_repository_get_document_tenant_checks(multi_tenant_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    repo, _ = multi_tenant_retriever

    # Direct repository fetch with matching tenant
    doc_a = repo.get_document("DOC-ORG-A-01", organization_id="TENANT_ALPHA")
    assert doc_a is not None
    assert doc_a.document_id == "DOC-ORG-A-01"

    # Direct repository fetch with mismatched tenant -> None
    doc_a_forbidden = repo.get_document("DOC-ORG-A-01", organization_id="TENANT_BETA")
    assert doc_a_forbidden is None


def test_repository_list_documents_tenant_isolation(multi_tenant_retriever: tuple[InMemoryKnowledgeRepository, KnowledgeRetriever]) -> None:
    repo, _ = multi_tenant_retriever

    docs_alpha = repo.list_documents(organization_id="TENANT_ALPHA")
    doc_ids_alpha = {d.document_id for d in docs_alpha}
    assert "DOC-ORG-A-01" in doc_ids_alpha
    assert "DOC-GLOBAL-01" in doc_ids_alpha
    assert "DOC-ORG-B-01" not in doc_ids_alpha

    docs_beta = repo.list_documents(organization_id="TENANT_BETA")
    doc_ids_beta = {d.document_id for d in docs_beta}
    assert "DOC-ORG-B-01" in doc_ids_beta
    assert "DOC-GLOBAL-01" in doc_ids_beta
    assert "DOC-ORG-A-01" not in doc_ids_beta
