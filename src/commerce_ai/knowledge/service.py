"""Unified Business Knowledge / RAG Service (Phase 7F).

Integrates document ingestion, repository storage, lexical retrieval, capability
routing, conflict detection, governance, and hybrid data + knowledge orchestration.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from commerce_ai.copilot.enums import CopilotState
from commerce_ai.copilot.schemas import CopilotResponse
from commerce_ai.copilot.service import CopilotService
from commerce_ai.knowledge.chunking import DeterministicChunker
from commerce_ai.knowledge.enums import (
    DocumentStatus,
    DocumentType,
    KnowledgeDomain,
    KnowledgeProvenanceType,
    QueryCapability,
)
from commerce_ai.knowledge.governance import (
    get_default_knowledge_governance,
    validate_no_recommendation_fabrication,
)
from commerce_ai.knowledge.ingestion import KnowledgeIngestionService
from commerce_ai.knowledge.repository import (
    InMemoryKnowledgeRepository,
    KnowledgeRepository,
)
from commerce_ai.knowledge.retrieval import KnowledgeRetriever
from commerce_ai.knowledge.routing import classify_query_capability
from commerce_ai.knowledge.schemas import (
    BusinessKnowledgeDocument,
    HybridQueryResult,
    KnowledgeIngestionResult,
    KnowledgeQueryContract,
    KnowledgeRetrievalResult,
)


class KnowledgeService:
    """Unified service interface for Phase 7F Business Knowledge."""

    def __init__(
        self,
        repository: Optional[KnowledgeRepository] = None,
        chunker: Optional[DeterministicChunker] = None,
        ingestion_service: Optional[KnowledgeIngestionService] = None,
        retriever: Optional[KnowledgeRetriever] = None,
    ) -> None:
        self.repository = repository or InMemoryKnowledgeRepository()
        self.chunker = chunker or DeterministicChunker()
        self.ingestion_service = ingestion_service or KnowledgeIngestionService(
            repository=self.repository, chunker=self.chunker
        )
        self.retriever = retriever or KnowledgeRetriever(repository=self.repository)
        self.governance = get_default_knowledge_governance()

    def ingest(
        self,
        document: BusinessKnowledgeDocument,
        supersede_prior: bool = False,
    ) -> KnowledgeIngestionResult:
        """Ingest, validate, chunk, and index a business knowledge document."""
        return self.ingestion_service.ingest_document(
            document=document,
            supersede_prior=supersede_prior,
        )

    def query(self, contract: KnowledgeQueryContract) -> KnowledgeRetrievalResult:
        """Execute knowledge retrieval using a structured query contract."""
        return self.retriever.retrieve(contract)

    def query_by_text(
        self,
        query: str,
        organization_id: Optional[str] = None,
        effective_date: Optional[str] = None,
        domain: Optional[KnowledgeDomain] = None,
        document_types: Optional[List[DocumentType]] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 5,
        minimum_relevance: float = 0.05,
        include_inactive: bool = False,
    ) -> KnowledgeRetrievalResult:
        """Convenience method to query knowledge directly via natural language string."""
        contract = KnowledgeQueryContract(
            query=query,
            organization_id=organization_id,
            effective_date=effective_date,
            domain=domain,
            document_types=document_types,
            tags=tags,
            top_k=top_k,
            minimum_relevance=minimum_relevance,
            include_inactive=include_inactive,
        )
        return self.retriever.retrieve(contract)

    def classify(self, question: str) -> QueryCapability:
        """Classify inbound question capability."""
        return classify_query_capability(question)

    def process_hybrid(
        self,
        question: str,
        copilot_service: Optional[CopilotService] = None,
        as_of_date: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> HybridQueryResult:
        """Orchestrate hybrid queries requiring both transactional data and documented knowledge."""
        cap = self.classify(question)
        copilot = copilot_service or CopilotService()

        # 1. Execute Data Branch through Copilot / Phase 7A
        data_resp: Optional[CopilotResponse] = None
        data_summary: Optional[str] = None
        data_evidence: List[Any] = []

        try:
            data_resp = copilot.process(question, as_of_date=as_of_date)
            if data_resp.state == CopilotState.COMPLETED and data_resp.execution_result:
                data_evidence = list(data_resp.execution_result.evidence_chain)
                data_summary = f"Observed business metrics computed via {len(data_resp.execution_result.step_results)} tool steps."
            elif data_resp.state == CopilotState.UNAVAILABLE:
                data_summary = "Requested business data metric is currently unavailable in the data layer."
            else:
                data_summary = f"Data query returned state: {data_resp.state.value}."
        except Exception as ex:
            data_summary = f"Data query execution could not be completed: {ex}"

        # 2. Execute Knowledge Branch through Phase 7F
        know_contract = KnowledgeQueryContract(
            query=question,
            organization_id=organization_id,
            effective_date=as_of_date,
            top_k=3,
            minimum_relevance=0.05,
        )
        know_res = self.query(know_contract)

        know_summary: Optional[str] = None
        if not know_res.insufficient_evidence and know_res.evidence:
            first_ev = know_res.evidence[0]
            know_summary = f"Policy '{first_ev.title}' (v{first_ev.document_version}) states: {first_ev.content[:160]}..."
        else:
            know_summary = know_res.insufficient_reason or "No relevant policy or business knowledge document was found."

        # 3. Determine Provenance and Synthesize Hybrid Narrative
        has_data = len(data_evidence) > 0
        has_know = len(know_res.evidence) > 0 and not know_res.insufficient_evidence

        if has_data and has_know:
            prov = KnowledgeProvenanceType.MIXED
            narrative = (
                f"Data Fact: {data_summary}\n"
                f"Documented Policy: {know_summary}\n"
                f"Evaluation: Observed business performance should be interpreted according to the documented policy above."
            )
        elif has_data:
            prov = KnowledgeProvenanceType.DATA_DERIVED
            narrative = f"Data Fact: {data_summary} (No documented policy retrieved)."
        elif has_know:
            prov = KnowledgeProvenanceType.DOCUMENT_DERIVED
            narrative = f"Documented Policy: {know_summary} (No transactional data requested or available)."
        else:
            prov = KnowledgeProvenanceType.INSUFFICIENT_DATA
            narrative = "Insufficient evidence to answer query from either data or documented knowledge."

        # Ensure no unauthorized recommendations were synthesized
        is_safe_rec, rec_err = validate_no_recommendation_fabrication(narrative)
        if not is_safe_rec:
            narrative = "[REDACTED_UNAUTHORIZED_OPERATIONAL_DIRECTIVE]"

        return HybridQueryResult(
            query=question,
            query_capability=cap,
            data_evidence=data_evidence if has_data else None,
            knowledge_evidence=know_res.evidence if has_know else None,
            provenance_type=prov,
            data_summary=data_summary,
            knowledge_summary=know_summary,
            combined_narrative=narrative,
            conflicts=know_res.conflicts,
            insufficient_evidence=(not has_data and not has_know),
        )

    def get_document(
        self,
        document_id: str,
        version: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> Optional[BusinessKnowledgeDocument]:
        """Fetch document version from repository."""
        return self.repository.get_document(
            document_id=document_id,
            version=version,
            organization_id=organization_id,
        )

    def list_documents(
        self,
        organization_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
        domain: Optional[KnowledgeDomain] = None,
    ) -> List[BusinessKnowledgeDocument]:
        """List documents from repository."""
        return self.repository.list_documents(
            organization_id=organization_id,
            status=status,
            domain=domain,
        )

    def export_audit_trail(self) -> Dict[str, Any]:
        """Export comprehensive audit trail of knowledge ingestions and governance settings."""
        docs = self.repository.list_documents()
        return {
            "indexed_documents_count": len(docs),
            "ingestion_events": self.ingestion_service.get_ingestion_log(),
            "governance": self.governance.model_dump(),
        }
