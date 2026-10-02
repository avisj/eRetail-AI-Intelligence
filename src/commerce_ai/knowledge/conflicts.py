"""Deterministic Knowledge Conflict Detection Engine (Phase 7F).

Identifies contradictions, version collisions, and overlapping policy discrepancies
among retrieved knowledge chunks without silently choosing arbitrary winners.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from commerce_ai.knowledge.enums import ConflictSeverity, ConflictStatus
from commerce_ai.knowledge.schemas import KnowledgeChunk, KnowledgeConflict


# Pattern to identify key numerical rules (e.g., "30 days", "15% threshold", "safety stock: 20 units")
_NUMERICAL_RULE_PATTERN = re.compile(
    r"\b(?:window|period|lead\s+time|safety\s+stock|threshold|target|minimum|maximum)\s*(?:is|of|:|=)?\s*(\d+(?:\.\d+)?)\s*(days?|weeks?|units?|%|percent)?\b",
    re.IGNORECASE,
)


def detect_knowledge_conflicts(
    chunks: List[KnowledgeChunk],
    effective_date: Optional[str] = None,
) -> List[KnowledgeConflict]:
    """Examine retrieved chunks and surface any deterministic contradictions."""
    conflicts: List[KnowledgeConflict] = []

    if not chunks or len(chunks) < 2:
        return conflicts

    # 1. Version Collision Check: Chunks from different versions of the SAME document
    versions_by_doc: Dict[str, Dict[str, List[KnowledgeChunk]]] = {}
    for chk in chunks:
        doc_id = chk.document_id
        ver = chk.document_version
        if doc_id not in versions_by_doc:
            versions_by_doc[doc_id] = {}
        if ver not in versions_by_doc[doc_id]:
            versions_by_doc[doc_id][ver] = []
        versions_by_doc[doc_id][ver].append(chk)

    for doc_id, ver_dict in versions_by_doc.items():
        if len(ver_dict) > 1:
            all_chk_ids = [c.chunk_id for v in ver_dict.values() for c in v]
            cid = "CONF-VER-" + hashlib.sha256(f"{doc_id}:{sorted(ver_dict.keys())}".encode("utf-8")).hexdigest()[:12]
            conflicts.append(
                KnowledgeConflict(
                    conflict_id=cid,
                    document_ids=[doc_id],
                    chunk_ids=all_chk_ids,
                    conflicting_attributes={
                        "document_id": doc_id,
                        "coexisting_versions": list(ver_dict.keys()),
                    },
                    description=(
                        f"Retrieved chunks span multiple competing versions ({list(ver_dict.keys())}) of the same "
                        f"document '{doc_id}'. System cannot safely combine mutually incompatible versions."
                    ),
                    severity=ConflictSeverity.BLOCKING,
                    status=ConflictStatus.DETECTED,
                    requires_human_review=True,
                )
            )

    # 2. Inter-Document Policy Conflict Check: Distinct active documents in same domain/type with differing values
    # Group chunks by (domain, document_type)
    by_category: Dict[str, List[KnowledgeChunk]] = {}
    for chk in chunks:
        cat_key = f"{chk.domain.value}:{chk.document_type.value}"
        by_category.setdefault(cat_key, []).append(chk)

    for cat_key, cat_chunks in by_category.items():
        distinct_docs = {c.document_id for c in cat_chunks}
        if len(distinct_docs) < 2:
            continue

        # Extract numerical rules from each document's chunks in this category
        rules_by_doc: Dict[str, Dict[str, str]] = {}
        for c in cat_chunks:
            matches = _NUMERICAL_RULE_PATTERN.findall(c.content)
            for val, unit in matches:
                rule_desc = f"{val} {unit}".strip()
                rules_by_doc.setdefault(c.document_id, {})[rule_desc] = c.chunk_id

        # If two distinct documents state mutually differing numerical rules in the same category
        doc_ids = list(rules_by_doc.keys())
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                doc_a, doc_b = doc_ids[i], doc_ids[j]
                rules_a = set(rules_by_doc[doc_a].keys())
                rules_b = set(rules_by_doc[doc_b].keys())
                # If both have extracted rules and they are completely disjoint
                if rules_a and rules_b and not (rules_a & rules_b):
                    cid = "CONF-POL-" + hashlib.sha256(f"{doc_a}:{doc_b}:{cat_key}".encode("utf-8")).hexdigest()[:12]
                    chunk_ids = [rules_by_doc[doc_a][r] for r in rules_a] + [rules_by_doc[doc_b][r] for r in rules_b]
                    conflicts.append(
                        KnowledgeConflict(
                            conflict_id=cid,
                            document_ids=[doc_a, doc_b],
                            chunk_ids=chunk_ids,
                            conflicting_attributes={
                                doc_a: list(rules_a),
                                doc_b: list(rules_b),
                                "category": cat_key,
                            },
                            description=(
                                f"Documents '{doc_a}' and '{doc_b}' define conflicting operational thresholds or rules "
                                f"for {cat_key} ({list(rules_a)} vs {list(rules_b)})."
                            ),
                            severity=ConflictSeverity.HIGH,
                            status=ConflictStatus.DETECTED,
                            requires_human_review=True,
                        )
                    )

    return conflicts
