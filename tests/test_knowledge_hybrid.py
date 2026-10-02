"""Unit tests for Phase 7F Hybrid Data + Knowledge Orchestration."""

from __future__ import annotations

import pytest

from commerce_ai.context.service import ConversationContextService
from commerce_ai.copilot.service import CopilotService
from commerce_ai.knowledge.enums import (
    DocumentStatus,
    DocumentType,
    KnowledgeDomain,
    KnowledgeProvenanceType,
    QueryCapability,
)
from commerce_ai.knowledge.schemas import BusinessKnowledgeDocument
from commerce_ai.knowledge.service import KnowledgeService


@pytest.fixture
def hybrid_env() -> tuple[KnowledgeService, CopilotService, ConversationContextService]:
    know_svc = KnowledgeService()
    copilot_svc = CopilotService()
    ctx_svc = ConversationContextService()

    # Ingest representative policy documents
    doc_inv_policy = BusinessKnowledgeDocument(
        document_id="DOC-POL-INV-HIGH",
        title="High Inventory Risk Mitigation Policy",
        document_type=DocumentType.INVENTORY_POLICY if hasattr(DocumentType, "INVENTORY_POLICY") else DocumentType.POLICY,
        version="1.0",
        source="Supply Chain Governance",
        domain=KnowledgeDomain.INVENTORY,
        tags=["inventory", "risk", "policy"],
        content=(
            "# High Risk Procedure\n"
            "When inventory risk is flagged as HIGH or CRITICAL, the inventory manager must "
            "conduct an immediate root-cause review within 24 hours. No new purchase orders "
            "may be dispatched until holding cost and demand velocity are verified."
        ),
        checksum="dummy",
    )

    doc_ret_policy = BusinessKnowledgeDocument(
        document_id="DOC-POL-RET-THRESH",
        title="Return Rate Threshold and Review Guidelines",
        document_type=DocumentType.RETURN_POLICY,
        version="1.0",
        source="Customer Experience",
        domain=KnowledgeDomain.RETURNS,
        tags=["return", "rate", "threshold"],
        content=(
            "# Return Rate Thresholds\n"
            "If the customer return rate for an apparel SKU exceeds 10 percent over a 30-day "
            "period, quality assurance must inspect the physical warehouse stock for sizing defects."
        ),
        checksum="dummy",
    )

    know_svc.ingest(doc_inv_policy)
    know_svc.ingest(doc_ret_policy)

    return know_svc, copilot_svc, ctx_svc


def test_hybrid_query_provenance_mixed(hybrid_env: tuple[KnowledgeService, CopilotService, ConversationContextService]) -> None:
    know_svc, copilot_svc, _ = hybrid_env

    # Hybrid inquiry: Inventory risk for SKU_001 + policy guidelines
    question = "Inventory risk is high for SKU_001. What does our policy say should happen?"
    hybrid_res = know_svc.process_hybrid(
        question=question,
        copilot_service=copilot_svc,
        as_of_date="2026-06-30",
    )

    assert hybrid_res.query_capability == QueryCapability.HYBRID_QUERY
    assert hybrid_res.provenance_type == KnowledgeProvenanceType.MIXED
    assert hybrid_res.data_summary is not None
    assert hybrid_res.knowledge_summary is not None
    assert "High Inventory Risk Mitigation Policy" in hybrid_res.knowledge_summary
    assert "Evaluation:" in (hybrid_res.combined_narrative or "")


def test_pure_knowledge_via_hybrid_pipeline(hybrid_env: tuple[KnowledgeService, CopilotService, ConversationContextService]) -> None:
    know_svc, copilot_svc, _ = hybrid_env

    question = "What does our policy say regarding high inventory risk?"
    hybrid_res = know_svc.process_hybrid(
        question=question,
        copilot_service=copilot_svc,
    )

    assert hybrid_res.query_capability == QueryCapability.KNOWLEDGE_QUERY
    # Since data inquiry yields no step execution results for pure policy question
    assert hybrid_res.knowledge_evidence is not None
    assert len(hybrid_res.knowledge_evidence) > 0


def test_hybrid_preserves_7e_conversation_context(hybrid_env: tuple[KnowledgeService, CopilotService, ConversationContextService]) -> None:
    know_svc, copilot_svc, ctx_svc = hybrid_env
    session_id = "SESS-HYBRID-7E"
    conv_id = "CONV-HYBRID-7E"

    # Turn 1: 7E establishes entity SKU_001
    req1, res1 = ctx_svc.resolve_request("Show inventory position for SKU_001 in warehouse WH_01", session_id, conv_id)
    resp1 = copilot_svc.process(req1)
    ctx_svc.update_after_response(session_id, conv_id, req1, resp1)

    # Turn 2: Follow-up pronoun inquiry requiring hybrid reasoning
    req2, res2 = ctx_svc.resolve_request("Its inventory risk is high. What does our policy say?", session_id, conv_id)
    assert res2.inherited_entities.get("sku_id") == "SKU_001"

    # Process hybrid inquiry
    hybrid_res = know_svc.process_hybrid(
        question=req2.question,
        copilot_service=copilot_svc,
        as_of_date="2026-06-30",
    )

    assert hybrid_res.provenance_type == KnowledgeProvenanceType.MIXED
    assert "SKU_001" in hybrid_res.query


def test_audit_trail_export(hybrid_env: tuple[KnowledgeService, CopilotService, ConversationContextService]) -> None:
    know_svc, _, _ = hybrid_env
    trail = know_svc.export_audit_trail()

    assert trail["indexed_documents_count"] >= 2
    assert len(trail["ingestion_events"]) >= 2
    assert trail["governance"]["read_only"] is True
    assert trail["governance"]["action_execution"] is False
