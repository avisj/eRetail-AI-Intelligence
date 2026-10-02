"""Governance, Security, and Prompt Injection Defense for Knowledge / RAG (Phase 7F).

Enforces strict read-only execution invariants, sanitizes malicious instructions
embedded inside documents or queries, prevents metric/recommendation fabrication,
and ensures documents cannot grant execution authority.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from commerce_ai.knowledge.schemas import KnowledgeGovernance, KnowledgeQueryContract


# Security patterns for prompt injection and system override attempts
_INJECTION_PATTERNS = [
    (re.compile(r"\b(?:ignore|disregard|forget|bypass)\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules|prompts|guardrails)\b", re.IGNORECASE), "PROMPT_INJECTION_BYPASS"),
    (re.compile(r"\b(?:system\s+override|admin\s+override|sudo\s+mode|root\s+access|jailbreak)\b", re.IGNORECASE), "SYSTEM_OVERRIDE_ATTEMPT"),
    (re.compile(r"\b(?:you\s+are\s+now|act\s+as)\s+(?:an?\s+)?(?:admin|root|unrestricted|unfiltered)\b", re.IGNORECASE), "ROLE_HIJACK_ATTEMPT"),
    (re.compile(r"\b(?:execute\s+order\s+66|execute\s+immediate\s+action)\b", re.IGNORECASE), "MALICIOUS_EXECUTION_TRIGGER"),
]

# Action execution requests that cannot be triggered via knowledge retrieval
_ACTION_PATTERNS = [
    (re.compile(r"\b(?:create|issue|place|generate)\s+(?:a\s+)?(?:purchase\s+order|po)\b", re.IGNORECASE), "ACTION_PROHIBITED_PO_CREATION"),
    (re.compile(r"\b(?:transfer|move|shift)\s+(?:stock|inventory|units)\b", re.IGNORECASE), "ACTION_PROHIBITED_STOCK_TRANSFER"),
    (re.compile(r"\b(?:change|update|alter|set)\s+(?:price|pricing|discount)\b", re.IGNORECASE), "ACTION_PROHIBITED_PRICING_CHANGE"),
    (re.compile(r"\b(?:drop|truncate|delete\s+from)\s+\w+\b", re.IGNORECASE), "ACTION_PROHIBITED_DB_MUTATION"),
]

# Prohibited operational directives masquerading as knowledge
_UNAUTHORIZED_DIRECTIVE_PATTERN = re.compile(
    r"\b(?:therefore|hereby|immediately)(?:\s+(?:therefore|hereby|immediately))*\s+(?:create|approve|execute|dispatch|authorize)\s+(?:a\s+)?(?:po|purchase\s+order|stock\s+transfer|transfer|purchase|order)\b",
    re.IGNORECASE,
)


def validate_knowledge_query_governance(contract: KnowledgeQueryContract) -> Tuple[bool, List[str]]:
    """Validate query contract against platform governance rules."""
    violations: List[str] = []

    # Check for prompt injection in query
    for pat, v_type in _INJECTION_PATTERNS:
        if pat.search(contract.query):
            violations.append(f"{v_type}: Query contains prohibited instruction override pattern.")

    # Check for action execution in query
    for pat, v_type in _ACTION_PATTERNS:
        if pat.search(contract.query):
            violations.append(f"{v_type}: Knowledge queries are strictly read-only and cannot trigger action execution.")

    # Bounds validation
    if contract.top_k < 1 or contract.top_k > 50:
        violations.append(f"INVALID_TOP_K: top_k must be between 1 and 50 (got {contract.top_k}).")

    if contract.minimum_relevance < 0.0 or contract.minimum_relevance > 1.0:
        violations.append(f"INVALID_RELEVANCE: minimum_relevance must be between 0.0 and 1.0.")

    return len(violations) == 0, violations


def sanitize_document_text(content: str) -> Tuple[str, List[str]]:
    """Sanitize document text to ensure embedded instructions cannot be executed as commands."""
    violations: List[str] = []
    sanitized = content

    for pat, v_type in _INJECTION_PATTERNS:
        if pat.search(sanitized):
            violations.append(f"{v_type}: Document content contained hostile instruction attempt.")
            sanitized = pat.sub("[REDACTED_DOCUMENT_INSTRUCTION_OVERRIDE]", sanitized)

    return sanitized, violations


def validate_no_recommendation_fabrication(text: str) -> Tuple[bool, Optional[str]]:
    """Ensure knowledge output does not synthesize operational directives."""
    if _UNAUTHORIZED_DIRECTIVE_PATTERN.search(text):
        return False, "PROHIBITED_DIRECTIVE: RAG output cannot issue operational action recommendations."
    return True, None


def get_default_knowledge_governance() -> KnowledgeGovernance:
    """Return default immutable governance envelope for Phase 7F."""
    return KnowledgeGovernance(
        read_only=True,
        action_execution=False,
        execution_allowed=False,
        no_ranking_enforced=True,
        metric_fabrication_prohibited=True,
        recommendation_fabrication_prohibited=True,
    )
