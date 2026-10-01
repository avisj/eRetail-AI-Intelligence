"""Deterministic Question Normalization for Copilot Intake (Phase 7C).

Provides reproducible string normalization:
- Trims and collapses duplicate whitespace
- Normalizes punctuation while preserving identifiers (SKU_001, WH_02), numbers, and dates (YYYY-MM-DD)
- Preserves business acronyms (SKU, WH, PO, AOV, COGS, ABC, XYZ, USD, EUR)
- Detects prompt injection keywords and malicious pattern attempts
"""

from __future__ import annotations

import re
from typing import Tuple


# Disallowed injection and jailbreak phrases
_INJECTION_PATTERNS = [
    re.compile(r"\b(ignore\s+(all\s+)?(previous\s+)?instructions|ignore\s+rules|system\s+prompt|developer\s+mode)\b", re.IGNORECASE),
    re.compile(r"\b(you\s+are\s+now|override\s+governance|bypass\s+security|jailbreak)\b", re.IGNORECASE),
    re.compile(r"\b(execute\s+arbitrary|run\s+shell|eval\(|exec\(|import\s+os)\b", re.IGNORECASE),
    re.compile(r"\b(drop\s+table|delete\s+from|insert\s+into|truncate\s+table|delete\s+inventory)\b", re.IGNORECASE),
]

# Prohibited action command phrases in text
_PROHIBITED_ACTION_PATTERNS = [
    re.compile(r"\b((create|execute|submit|issue|place|generate)\s+(a\s+)?(purchase\s+order|po)|order\s+\d+\s+units|purchase\s+\d+\s+units)\b", re.IGNORECASE),
    re.compile(r"\b((transfer|move|ship|dispatch|rebalance)\s+(\d+\s+units|stock|inventory|units)\b|transfer\s+from\s+\w+\s+to\s+\w+)", re.IGNORECASE),
    re.compile(r"\b((change|modify|update|adjust|set|reduce|increase)\s+(the\s+)?(price|pricing|discount|cost)|apply\s+(a\s+)?discount)\b", re.IGNORECASE),
    re.compile(r"\b(delete|drop|truncate|remove)\s+(inventory|sales|records|data|table|schema)\b", re.IGNORECASE),
]


def normalize_question(text: str) -> str:
    """Deterministically normalize question text for structured intent interpretation."""
    if not text:
        return ""

    # Replace newlines, tabs, and duplicate whitespace with single spaces
    normalized = re.sub(r"[\r\n\t]+", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Remove trailing punctuation (question marks, exclamation points, periods)
    normalized = re.sub(r"[?!.]+$", "", normalized).strip()

    return normalized


def detect_malicious_intent(text: str) -> Tuple[bool, str]:
    """Check question text for prompt injection, jailbreak, or unauthorized action execution attempts."""
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, f"Prompt injection / jailbreak pattern detected: '{match.group(0)}'"

    for pattern in _PROHIBITED_ACTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, f"Prohibited direct action execution command detected: '{match.group(0)}'"

    return False, ""
