"""Phase 7F — Controlled Business Knowledge & RAG Foundation.

Provides structured, version-aware, tenant-isolated business knowledge retrieval
operating above the existing Phase 7A tools, 7B contracts, 7C copilot, 7D explanations,
and 7E conversation context without converting documents into transactional truth.
"""

from __future__ import annotations

from commerce_ai.knowledge.chunking import DeterministicChunker
from commerce_ai.knowledge.citations import build_citation, build_evidence
from commerce_ai.knowledge.conflicts import detect_knowledge_conflicts
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
from commerce_ai.knowledge.governance import (
    get_default_knowledge_governance,
    sanitize_document_text,
    validate_knowledge_query_governance,
    validate_no_recommendation_fabrication,
)
from commerce_ai.knowledge.ingestion import (
    KnowledgeIngestionService,
    compute_content_checksum,
    normalize_document_content,
)
from commerce_ai.knowledge.repository import (
    InMemoryKnowledgeRepository,
    KnowledgeRepository,
)
from commerce_ai.knowledge.retrieval import KnowledgeRetriever
from commerce_ai.knowledge.routing import classify_query_capability
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
from commerce_ai.knowledge.service import KnowledgeService

__all__ = [
    # Enums
    "DocumentType",
    "DocumentStatus",
    "KnowledgeDomain",
    "RetrievalMethod",
    "KnowledgeProvenanceType",
    "ConflictSeverity",
    "ConflictStatus",
    "QueryCapability",
    # Schemas
    "BusinessKnowledgeDocument",
    "KnowledgeChunk",
    "KnowledgeCitation",
    "KnowledgeEvidence",
    "KnowledgeConflict",
    "KnowledgeQueryContract",
    "KnowledgeRetrievalResult",
    "KnowledgeGovernance",
    "KnowledgeIngestionResult",
    "HybridQueryResult",
    # Chunking & Processing
    "DeterministicChunker",
    "normalize_document_content",
    "compute_content_checksum",
    # Repository
    "KnowledgeRepository",
    "InMemoryKnowledgeRepository",
    # Services & Engines
    "KnowledgeIngestionService",
    "KnowledgeRetriever",
    "detect_knowledge_conflicts",
    "build_citation",
    "build_evidence",
    "classify_query_capability",
    "KnowledgeService",
    # Governance
    "validate_knowledge_query_governance",
    "sanitize_document_text",
    "validate_no_recommendation_fabrication",
    "get_default_knowledge_governance",
]
