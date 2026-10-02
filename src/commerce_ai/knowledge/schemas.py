"""Schemas and Data Models for Business Knowledge / RAG (Phase 7F).

Defines structured models for business knowledge documents, deterministic chunks,
citations, evidence lineage records, conflicts, query contracts, retrieval results,
ingestion receipts, and hybrid query containers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BusinessKnowledgeDocument(BaseModel):
    """Authoritative document containing approved organizational knowledge or policy."""
    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Unique deterministic document identifier.")
    title: str = Field(description="Human-readable title of the document.")
    document_type: DocumentType = Field(description="Controlled classification of the document.")
    version: str = Field(default="1.0", description="Semantic or sequential version string.")
    source: str = Field(description="Authoring body, system, or organization department.")
    source_uri: Optional[str] = Field(default=None, description="Optional canonical URI or file path.")
    organization_id: Optional[str] = Field(default=None, description="Tenant / Organization scope for strict multi-tenant isolation.")
    domain: KnowledgeDomain = Field(default=KnowledgeDomain.GENERAL, description="Business domain alignment.")
    category: str = Field(default="General", description="Subcategory or business topic.")
    tags: List[str] = Field(default_factory=list, description="Descriptive search and taxonomy tags.")
    effective_from: Optional[str] = Field(default=None, description="Calendar effective start date (YYYY-MM-DD).")
    effective_to: Optional[str] = Field(default=None, description="Calendar effective end date (YYYY-MM-DD).")
    status: DocumentStatus = Field(default=DocumentStatus.ACTIVE, description="Lifecycle status.")
    language: str = Field(default="en", description="Language code.")
    content: str = Field(description="Raw text content of the document.")
    checksum: str = Field(description="SHA-256 deterministic checksum of the normalized content.")
    created_at: str = Field(default_factory=_utc_now, description="UTC creation timestamp.")
    updated_at: str = Field(default_factory=_utc_now, description="UTC last updated timestamp.")

    @field_validator("document_id", "title", "content", "checksum")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or blank.")
        return v.strip()


class KnowledgeChunk(BaseModel):
    """Deterministic, immutable unit of text derived from a BusinessKnowledgeDocument."""
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique deterministic chunk identifier (e.g. CHK-{doc_id}-v{ver}-{seq}).")
    document_id: str = Field(description="Associated parent document ID.")
    document_version: str = Field(description="Associated parent document version.")
    sequence_number: int = Field(description="0-indexed sequence position within document.")
    title: str = Field(description="Document title.")
    section_heading: Optional[str] = Field(default=None, description="Section heading under which chunk appears.")
    page_or_location: Optional[str] = Field(default=None, description="Structural location or anchor within source.")
    content: str = Field(description="Text segment content.")
    checksum: str = Field(description="SHA-256 checksum of chunk content.")
    organization_id: Optional[str] = Field(default=None, description="Tenant scope inherited from document.")
    domain: KnowledgeDomain = Field(default=KnowledgeDomain.GENERAL, description="Domain inherited from document.")
    document_type: DocumentType = Field(description="Document type inherited from document.")
    effective_from: Optional[str] = Field(default=None, description="Effective from date.")
    effective_to: Optional[str] = Field(default=None, description="Effective to date.")
    status: DocumentStatus = Field(default=DocumentStatus.ACTIVE, description="Status inherited from document.")
    tags: List[str] = Field(default_factory=list, description="Tags inherited from document.")


class KnowledgeCitation(BaseModel):
    """Fine-grained audit citation pointing back to specific source chunk."""
    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(description="Unique citation identifier.")
    document_id: str = Field(description="Source document ID.")
    document_version: str = Field(description="Source document version.")
    chunk_id: str = Field(description="Source chunk ID.")
    document_title: str = Field(description="Title of cited document.")
    section_heading: Optional[str] = Field(default=None, description="Cited section heading.")
    source: str = Field(description="Original authoring body / department.")
    effective_from: Optional[str] = Field(default=None, description="Effective start date.")
    effective_to: Optional[str] = Field(default=None, description="Effective end date.")
    relevance_score: float = Field(description="Relevance score of retrieved chunk.")
    snippet: str = Field(description="Representative snippet of cited text.")


class KnowledgeEvidence(BaseModel):
    """Structured evidence envelope for downstream explanation and auditability."""
    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(description="Unique evidence identifier.")
    document_id: str = Field(description="Source document ID.")
    document_version: str = Field(description="Source document version.")
    chunk_id: str = Field(description="Source chunk ID.")
    title: str = Field(description="Document title.")
    source: str = Field(description="Originating authority.")
    section: Optional[str] = Field(default=None, description="Section heading.")
    retrieval_method: RetrievalMethod = Field(description="Actual retrieval algorithm used.")
    relevance_score: float = Field(description="Numerical relevance score [0.0, 1.0].")
    effective_from: Optional[str] = Field(default=None, description="Effective start date.")
    effective_to: Optional[str] = Field(default=None, description="Effective end date.")
    organization_id: Optional[str] = Field(default=None, description="Tenant organization scope.")
    checksum: str = Field(description="Content checksum.")
    provenance_type: KnowledgeProvenanceType = Field(default=KnowledgeProvenanceType.DOCUMENT_DERIVED)
    content: str = Field(description="Full text content of the evidence chunk.")


class KnowledgeConflict(BaseModel):
    """Explicit representation of conflicting business knowledge or overlapping policies."""
    model_config = ConfigDict(frozen=True)

    conflict_id: str = Field(description="Unique conflict identifier.")
    document_ids: List[str] = Field(description="List of document IDs involved in conflict.")
    chunk_ids: List[str] = Field(description="List of chunk IDs involved in conflict.")
    conflicting_attributes: Dict[str, Any] = Field(default_factory=dict, description="Conflicting attributes or values.")
    description: str = Field(description="Detailed explanation of the contradiction.")
    severity: ConflictSeverity = Field(default=ConflictSeverity.MEDIUM, description="Impact severity.")
    status: ConflictStatus = Field(default=ConflictStatus.DETECTED, description="Current resolution state.")
    requires_human_review: bool = Field(default=True, description="Whether human review is required.")


class KnowledgeQueryContract(BaseModel):
    """Structured contract governing knowledge retrieval requests."""
    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Natural language or keyword query string.")
    organization_id: Optional[str] = Field(default=None, description="Mandatory tenant scope for isolated retrieval.")
    domain: Optional[KnowledgeDomain] = Field(default=None, description="Optional domain filter.")
    document_types: Optional[List[DocumentType]] = Field(default=None, description="Optional document type filters.")
    tags: Optional[List[str]] = Field(default=None, description="Optional required tags.")
    effective_date: Optional[str] = Field(default=None, description="Target calendar date (YYYY-MM-DD) for time-valid retrieval.")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum chunks to retrieve (1..50).")
    minimum_relevance: float = Field(default=0.05, ge=0.0, le=1.0, description="Minimum relevance threshold.")
    conversation_id: Optional[str] = Field(default=None, description="Optional context tracking ID.")
    include_inactive: bool = Field(default=False, description="Whether to include deprecated/archived knowledge.")

    @field_validator("query")
    @classmethod
    def _validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string cannot be empty.")
        return v.strip()


class KnowledgeRetrievalResult(BaseModel):
    """Deterministic output package of a knowledge retrieval operation."""
    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Original query string.")
    organization_id: Optional[str] = Field(default=None, description="Tenant scope applied.")
    effective_date: Optional[str] = Field(default=None, description="Effective date applied.")
    retrieval_method: RetrievalMethod = Field(default=RetrievalMethod.LEXICAL_TOKEN_MATCH)
    chunks: List[KnowledgeChunk] = Field(default_factory=list, description="Retrieved chunks ordered deterministically.")
    scores: List[float] = Field(default_factory=list, description="Relevance scores matching chunks.")
    citations: List[KnowledgeCitation] = Field(default_factory=list, description="Audit citations.")
    evidence: List[KnowledgeEvidence] = Field(default_factory=list, description="Evidence envelopes.")
    conflicts: List[KnowledgeConflict] = Field(default_factory=list, description="Surfaced conflicts if any.")
    insufficient_evidence: bool = Field(default=False, description="Flag indicating lack of sufficient grounded knowledge.")
    insufficient_reason: Optional[str] = Field(default=None, description="Reason for insufficient evidence.")
    timestamp: str = Field(default_factory=_utc_now, description="UTC retrieval timestamp.")


class KnowledgeGovernance(BaseModel):
    """Immutable governance guarantees enforced across knowledge retrieval."""
    model_config = ConfigDict(frozen=True)

    read_only: bool = Field(default=True, description="Strict read-only semantics.")
    action_execution: bool = Field(default=False, description="Autonomous actions strictly prohibited.")
    execution_allowed: bool = Field(default=False, description="Execution permissions denied.")
    no_ranking_enforced: bool = Field(default=True, description="Subjective top-N / ranking logic prohibited.")
    metric_fabrication_prohibited: bool = Field(default=True, description="Documents cannot generate synthetic metrics.")
    recommendation_fabrication_prohibited: bool = Field(default=True, description="Documents cannot generate unauthorized operational actions.")


class KnowledgeIngestionResult(BaseModel):
    """Receipt returned following ingestion of a business knowledge document."""
    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Ingested document ID.")
    version: str = Field(description="Ingested document version.")
    status: str = Field(description="Ingestion status (e.g. INGESTED, REPLAYED_UNCHANGED).")
    chunk_count: int = Field(description="Number of chunks created and stored.")
    checksum: str = Field(description="Checksum of content.")
    is_idempotent_replay: bool = Field(default=False, description="True if identical document already existed.")
    message: str = Field(description="Informative message.")


class HybridQueryResult(BaseModel):
    """Unified container for inquiries combining transactional data and business knowledge."""
    model_config = ConfigDict(frozen=True)

    query: str = Field(description="User query.")
    query_capability: QueryCapability = Field(description="Classified query capability.")
    data_evidence: Optional[List[Any]] = Field(default=None, description="Authoritative transactional evidence from Phase 7A.")
    knowledge_evidence: Optional[List[KnowledgeEvidence]] = Field(default=None, description="Document evidence from Phase 7F.")
    provenance_type: KnowledgeProvenanceType = Field(description="Overall provenance classification.")
    data_summary: Optional[str] = Field(default=None, description="Summary of data facts.")
    knowledge_summary: Optional[str] = Field(default=None, description="Summary of documented policy / knowledge.")
    combined_narrative: Optional[str] = Field(default=None, description="Synthesized multi-provenance narrative.")
    conflicts: List[KnowledgeConflict] = Field(default_factory=list, description="Any detected policy or data contradictions.")
    insufficient_evidence: bool = Field(default=False, description="True if either data or policy evidence was missing.")
