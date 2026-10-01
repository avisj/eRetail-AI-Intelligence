"""Governance and Security Constraints for Controlled Conversation Context (Phase 7E).

Enforces:
- Absolute read-only guarantees (no database mutation)
- Non-authoritative memory enforcement (context numbers are never business truth)
- Prompt injection defense & fake business fact sanitization
- Autonomous action & ranking prohibitions
"""

from __future__ import annotations

import re
from typing import List, Tuple

from commerce_ai.context.enums import ContextCategory
from commerce_ai.context.schemas import ContextGovernance, ContextItem, ConversationContextSnapshot

# Known prompt injection & system override patterns
_INJECTION_PATTERNS = [
    re.compile(r"\b(ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|rules))\b", re.IGNORECASE),
    re.compile(r"\b(system\s+override|sudo\s+mode|admin\s+mode|developer\s+mode)\b", re.IGNORECASE),
    re.compile(r"\b(execute\s+(sql|code|command|script|action|order|po|transfer|pricing))\b", re.IGNORECASE),
    re.compile(r"\b(drop\s+table|delete\s+from|insert\s+into|update\s+\w+\s+set)\b", re.IGNORECASE),
    re.compile(r"\b(grant\s+(all\s+)?permissions|escalate\s+privilege)\b", re.IGNORECASE),
    re.compile(r"\b(you\s+are\s+now\s+(an\s+admin|root|unrestricted))\b", re.IGNORECASE),
    re.compile(r"\b(bypass\s+(governance|safety|validation|authorization))\b", re.IGNORECASE),
]

# Patterns attempting to fabricate business facts into context
_FAKE_FACT_PATTERNS = [
    re.compile(r"\b(remember\s+that\s+(sales|revenue|inventory|margin|profit)\s+(is|are|was|were))\b", re.IGNORECASE),
    re.compile(r"\b(record\s+that\s+(we\s+made|sales\s+reached|inventory\s+is))\b", re.IGNORECASE),
    re.compile(r"\b(set\s+(margin|sales|revenue|units)\s+to\s+[\$0-9])\b", re.IGNORECASE),
    re.compile(r"\b(assume\s+the\s+truth\s+is\s+that)\b", re.IGNORECASE),
]

# Subjective ranking patterns prohibited by governance
_RANKING_PATTERNS = [
    re.compile(r"\b(who\s+is\s+the\s+best\s+performing|rank\s+the\s+top\s+[0-9]+|show\s+the\s+worst\s+performing)\b", re.IGNORECASE),
    re.compile(r"\b(winner\s+and\s+loser|top\s+performers|bottom\s+performers)\b", re.IGNORECASE),
]


def sanitize_input_for_injection(text: str) -> Tuple[str, List[str]]:
    """Inspect text for prompt injection, fake business facts, or unauthorized commands.
    
    Returns:
        Tuple of (sanitized_text, list_of_violations)
    """
    violations: List[str] = []
    sanitized = text

    # Check prompt injection patterns
    for pat in _INJECTION_PATTERNS:
        matches = [m.group(0) for m in pat.finditer(sanitized)]
        for m_str in matches:
            violations.append(f"Prompt injection pattern detected: '{m_str}'")
        if matches:
            sanitized = pat.sub("[REDACTED_SECURITY_OVERRIDE]", sanitized)

    # Check fake business facts
    for pat in _FAKE_FACT_PATTERNS:
        matches = [m.group(0) for m in pat.finditer(sanitized)]
        for m_str in matches:
            violations.append(f"Attempt to inject fake business fact into memory: '{m_str}'")
        if matches:
            sanitized = pat.sub("[REDACTED_FAKE_FACT]", sanitized)

    # Check subjective ranking patterns
    for pat in _RANKING_PATTERNS:
        matches = [m.group(0) for m in pat.finditer(sanitized)]
        for m_str in matches:
            violations.append(f"Subjective ranking request detected: '{m_str}'")

    return sanitized, violations


def validate_context_governance(snapshot: ConversationContextSnapshot) -> Tuple[bool, List[str]]:
    """Validate that conversation context strictly conforms to platform governance invariants.
    
    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    gov = snapshot.governance
    violations: List[str] = []

    if not gov.read_only:
        violations.append("Violation: Context governance read_only must be True.")
    if gov.action_execution:
        violations.append("Violation: Context governance action_execution must be False.")
    if gov.execution_allowed:
        violations.append("Violation: Context governance execution_allowed must be False.")
    if not gov.no_ranking_enforced:
        violations.append("Violation: Context governance no_ranking_enforced must be True.")
    if not gov.no_decision_selection_enforced:
        violations.append("Violation: Context governance no_decision_selection_enforced must be True.")
    if gov.allow_injection:
        violations.append("Violation: Context governance allow_injection must be False.")

    # Check individual context items for security violations
    for item_key, item in snapshot.items.items():
        if isinstance(item.value, str):
            _, item_violations = sanitize_input_for_injection(item.value)
            for v in item_violations:
                violations.append(f"Item '{item_key}': {v}")

    return len(violations) == 0, violations


def enforce_non_authoritative_memory(item: ContextItem) -> bool:
    """Verify that a context item does not masquerade as authoritative business truth.
    
    Context is strictly ephemeral and non-authoritative:
    - ENTITY_CONTEXT and FILTER_CONTEXT store search/filter dimensions, never metric truths.
    - RESULT_CONTEXT and EVIDENCE_CONTEXT store citations, never replacement data.
    """
    if item.category in (ContextCategory.ENTITY_CONTEXT, ContextCategory.FILTER_CONTEXT):
        # Must not contain calculated metric data
        if item.key in {"sales", "revenue", "gross_margin", "units", "inventory_balance"}:
            return False
    return True
