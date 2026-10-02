"""Deterministic Knowledge Retrieval Engine (Phase 7F).

Executes metadata pre-filtering, calendar effective-date validation, lexical token relevance
scoring, conflict detection, and evidence lineage generation without external vector dependencies.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from commerce_ai.knowledge.citations import build_citation, build_evidence
from commerce_ai.knowledge.conflicts import detect_knowledge_conflicts
from commerce_ai.knowledge.enums import (
    ConflictSeverity,
    DocumentStatus,
    RetrievalMethod,
)
from commerce_ai.knowledge.governance import validate_knowledge_query_governance
from commerce_ai.knowledge.repository import KnowledgeRepository
from commerce_ai.knowledge.schemas import (
    KnowledgeChunk,
    KnowledgeCitation,
    KnowledgeEvidence,
    KnowledgeQueryContract,
    KnowledgeRetrievalResult,
)


_WORD_REGEX = re.compile(r"\b[A-Za-z0-9_]{2,}\b")
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "shall", "will", "should", "would", "may", "might", "must", "can", "could", "of",
    "it", "its", "what", "which", "who", "whom", "this", "that", "these", "those", "how",
    "our", "my", "your", "their", "tell", "me", "show", "give", "explain", "policy", "sop",
}


def _tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into searchable lowercase terms."""
    return [m.group(0).lower() for m in _WORD_REGEX.finditer(text)]


def _is_date_effective(chk_from: Optional[str], chk_to: Optional[str], target_date: str) -> bool:
    """Check if target_date falls within [chk_from, chk_to] inclusive."""
    if chk_from and chk_from > target_date:
        return False
    if chk_to and chk_to < target_date:
        return False
    return True


class KnowledgeRetriever:
    """Retrieval engine executing grounded, deterministic knowledge searches."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository

    def retrieve(self, contract: KnowledgeQueryContract) -> KnowledgeRetrievalResult:
        """Execute knowledge search governed by the structured contract."""
        now_ts = datetime.now(timezone.utc).isoformat()

        # 1. Governance & Security check
        is_safe, violations = validate_knowledge_query_governance(contract)
        if not is_safe:
            return KnowledgeRetrievalResult(
                query=contract.query,
                organization_id=contract.organization_id,
                effective_date=contract.effective_date,
                retrieval_method=RetrievalMethod.LEXICAL_TOKEN_MATCH,
                chunks=[],
                scores=[],
                citations=[],
                evidence=[],
                conflicts=[],
                insufficient_evidence=True,
                insufficient_reason="; ".join(violations),
                timestamp=now_ts,
            )

        # 2. Candidate Chunk Fetching & Tenant Isolation
        all_candidates = self.repository.get_all_chunks(
            organization_id=contract.organization_id,
            include_inactive=contract.include_inactive,
        )

        # 3. Metadata Pre-Filtering
        filtered_chunks: List[KnowledgeChunk] = []
        for chk in all_candidates:
            # Document status
            if not contract.include_inactive and chk.status != DocumentStatus.ACTIVE:
                continue
            # Domain filter
            if contract.domain is not None and chk.domain != contract.domain:
                continue
            # Document types filter
            if contract.document_types and chk.document_type not in contract.document_types:
                continue
            # Tags filter
            if contract.tags:
                if not any(t.lower() in [ct.lower() for ct in chk.tags] for t in contract.tags):
                    continue
            # Effective date filter
            if contract.effective_date:
                if not _is_date_effective(chk.effective_from, chk.effective_to, contract.effective_date):
                    continue

            filtered_chunks.append(chk)

        if not filtered_chunks:
            return KnowledgeRetrievalResult(
                query=contract.query,
                organization_id=contract.organization_id,
                effective_date=contract.effective_date,
                retrieval_method=RetrievalMethod.LEXICAL_TOKEN_MATCH,
                chunks=[],
                scores=[],
                citations=[],
                evidence=[],
                conflicts=[],
                insufficient_evidence=True,
                insufficient_reason="No knowledge documents matched the filter criteria or effective date window.",
                timestamp=now_ts,
            )

        # 4. Tokenization & Relevance Scoring
        query_terms = [t for t in _tokenize(contract.query) if t not in _STOPWORDS]
        if not query_terms:
            # Fallback to all terms if every word was a stopword
            query_terms = _tokenize(contract.query)

        scored_candidates: List[Tuple[KnowledgeChunk, float]] = []
        query_lower = contract.query.lower()

        for chk in filtered_chunks:
            score = self._compute_lexical_score(chk, query_terms, query_lower)
            if score >= contract.minimum_relevance:
                scored_candidates.append((chk, score))

        if not scored_candidates:
            return KnowledgeRetrievalResult(
                query=contract.query,
                organization_id=contract.organization_id,
                effective_date=contract.effective_date,
                retrieval_method=RetrievalMethod.LEXICAL_TOKEN_MATCH,
                chunks=[],
                scores=[],
                citations=[],
                evidence=[],
                conflicts=[],
                insufficient_evidence=True,
                insufficient_reason="No documents met the minimum relevance threshold.",
                timestamp=now_ts,
            )

        # 5. Deterministic Ranking & Tie-Breaking (No LLM / probabilistic order)
        # Sort key: (-score, document_id, sequence_number)
        scored_candidates.sort(key=lambda item: (-item[1], item[0].document_id, item[0].sequence_number))

        # Truncate to top_k
        top_candidates = scored_candidates[: contract.top_k]
        top_chunks = [c for c, _ in top_candidates]
        top_scores = [s for _, s in top_candidates]

        # 6. Conflict Detection
        conflicts = detect_knowledge_conflicts(top_chunks, effective_date=contract.effective_date)
        blocking_conflict = any(c.severity == ConflictSeverity.BLOCKING for c in conflicts)

        # 7. Package Citations and Evidence
        citations: List[KnowledgeCitation] = []
        evidence_list: List[KnowledgeEvidence] = []
        for chk, sc in zip(top_chunks, top_scores):
            citations.append(build_citation(chk, sc))
            evidence_list.append(build_evidence(chk, sc, RetrievalMethod.LEXICAL_TOKEN_MATCH))

        insufficient_ev = blocking_conflict
        insufficient_msg = (
            "Blocking knowledge conflict detected among retrieved documents. Human review required."
            if blocking_conflict
            else None
        )

        return KnowledgeRetrievalResult(
            query=contract.query,
            organization_id=contract.organization_id,
            effective_date=contract.effective_date,
            retrieval_method=RetrievalMethod.LEXICAL_TOKEN_MATCH,
            chunks=top_chunks,
            scores=top_scores,
            citations=citations,
            evidence=evidence_list,
            conflicts=conflicts,
            insufficient_evidence=insufficient_ev,
            insufficient_reason=insufficient_msg,
            timestamp=now_ts,
        )

    def _compute_lexical_score(
        self,
        chunk: KnowledgeChunk,
        query_terms: List[str],
        raw_query: str,
    ) -> float:
        """Compute normalized [0.0, 1.0] lexical score incorporating headers, keywords, and exact phrases."""
        if not query_terms:
            return 0.0

        content_lower = chunk.content.lower()
        title_lower = chunk.title.lower()
        heading_lower = (chunk.section_heading or "").lower()

        chunk_tokens = _tokenize(content_lower)
        token_count = len(chunk_tokens) or 1

        term_hits = 0
        exact_freq = 0

        for t in query_terms:
            hits = chunk_tokens.count(t)
            if hits > 0:
                term_hits += 1
                exact_freq += hits

        # Term coverage ratio
        coverage = term_hits / len(query_terms)

        # Frequency boost (diminishing returns)
        freq_factor = math.log1p(exact_freq) / math.log1p(token_count)

        score = coverage * 0.5 + freq_factor * 0.2

        # Title match bonus
        if any(t in title_lower for t in query_terms):
            score += 0.15

        # Section heading bonus
        if heading_lower and any(t in heading_lower for t in query_terms):
            score += 0.10

        # Exact phrase bonus
        if len(raw_query) > 5 and raw_query in content_lower:
            score += 0.25

        return min(1.0, round(score, 4))
