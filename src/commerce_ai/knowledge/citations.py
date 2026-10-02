"""Knowledge Citations and Evidence Packaging (Phase 7F).

Builds immutable citations and evidence records grounded strictly in retrieved
document chunks, preventing citation fabrication and preserving end-to-end lineage.
"""

from __future__ import annotations

import hashlib
from typing import List

from commerce_ai.knowledge.enums import KnowledgeProvenanceType, RetrievalMethod
from commerce_ai.knowledge.schemas import (
    KnowledgeCitation,
    KnowledgeChunk,
    KnowledgeEvidence,
)


def create_snippet(text: str, max_chars: int = 180) -> str:
    """Generate a clean excerpt preserving word boundaries."""
    clean = " ".join(text.split()).strip()
    if len(clean) <= max_chars:
        return clean
    truncated = clean[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space != -1 and last_space > 50:
        return truncated[:last_space] + "..."
    return truncated + "..."


def build_citation(chunk: KnowledgeChunk, relevance_score: float) -> KnowledgeCitation:
    """Construct an auditable KnowledgeCitation from a retrieved chunk."""
    seed = f"{chunk.chunk_id}:{relevance_score}:{chunk.checksum[:8]}"
    cit_id = "CIT-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]

    return KnowledgeCitation(
        citation_id=cit_id,
        document_id=chunk.document_id,
        document_version=chunk.document_version,
        chunk_id=chunk.chunk_id,
        document_title=chunk.title,
        section_heading=chunk.section_heading,
        source=f"Document {chunk.document_id} (v{chunk.document_version})",
        effective_from=chunk.effective_from,
        effective_to=chunk.effective_to,
        relevance_score=round(relevance_score, 4),
        snippet=create_snippet(chunk.content),
    )


def build_evidence(
    chunk: KnowledgeChunk,
    relevance_score: float,
    retrieval_method: RetrievalMethod,
) -> KnowledgeEvidence:
    """Construct a formal KnowledgeEvidence record for downstream explanation and auditing."""
    seed = f"{chunk.chunk_id}:{retrieval_method.value}:{chunk.checksum[:8]}"
    ev_id = "EVD-KNOW-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]

    return KnowledgeEvidence(
        evidence_id=ev_id,
        document_id=chunk.document_id,
        document_version=chunk.document_version,
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        source=f"KnowledgeRepo:{chunk.document_id}#v{chunk.document_version}",
        section=chunk.section_heading,
        retrieval_method=retrieval_method,
        relevance_score=round(relevance_score, 4),
        effective_from=chunk.effective_from,
        effective_to=chunk.effective_to,
        organization_id=chunk.organization_id,
        checksum=chunk.checksum,
        provenance_type=KnowledgeProvenanceType.DOCUMENT_DERIVED,
        content=chunk.content,
    )
