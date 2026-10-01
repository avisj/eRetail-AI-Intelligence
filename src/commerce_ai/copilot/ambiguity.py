"""Ambiguity Detection & Clarification Generator for Copilot (Phase 7C).

Identifies underspecified, ambiguous, or multi-meaning business queries where
executing a guessed contract could mislead the user. Generates structured
CopilotClarification objects offering explicit options without guessing.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Tuple

from commerce_ai.copilot.enums import ClarificationSeverity
from commerce_ai.copilot.schemas import CopilotClarification


# Patterns of queries that are severely underspecified and require explicit clarification
_AMBIGUOUS_PATTERNS = [
    (
        re.compile(r"^(show|check|get|view)\s+(inventory|stock)$", re.IGNORECASE),
        "The inventory request is underspecified. Please specify whether you want inventory status/position, risk tier distribution, stockout exposure, or slow-moving capital.",
        ["intent", "grain"],
        [
            "Show overall inventory position summary (PORTFOLIO)",
            "Show inventory position by SKU (TABLE)",
            "Analyze network inventory risk tiers",
            "Show stockout risk exposure",
            "Show slow-moving inventory capital",
        ],
    ),
    (
        re.compile(r"^(what\s+about|check|how\s+are)\s+returns\??$", re.IGNORECASE),
        "The returns request is underspecified. Please specify if you want overall return rate, return anomalies, customer return reasons, or high-risk SKUs.",
        ["intent", "requested_output"],
        [
            "Overall return rate and KPI summary",
            "Statistically anomalous return volume spikes",
            "Breakdown of customer-reported return reasons",
            "SKUs with elevated return risk",
        ],
    ),
    (
        re.compile(r"^(status|status\s+report|how\s+are\s+things|overall\s+status|health\s+check)$", re.IGNORECASE),
        "The query does not specify a business domain. Please select the functional area you wish to analyze.",
        ["domain", "intent"],
        [
            "Commercial sales performance",
            "Gross margin and financial economics",
            "Network inventory status and coverage",
            "Multi-domain executive overview (revenue, margin, inventory)",
        ],
    ),
    (
        re.compile(r"^(show\s+me\s+data|give\s+me\s+data|look\s+into\s+this|analyze|report)$", re.IGNORECASE),
        "The question is completely unspecified. Please describe the specific business metric or area to analyze.",
        ["domain", "intent", "metrics"],
        [
            "Sales performance and revenue trends",
            "Inventory risk and stockout analysis",
            "Margin drivers and profitability",
        ],
    ),
    (
        re.compile(r"^(tell\s+me\s+about\s+(our\s+)?performance|how\s+is\s+(the\s+)?(business|company)\s+doing|how\s+are\s+we\s+doing|what\s+are\s+the\s+numbers|what\s+do\s+the\s+numbers\s+look\s+like)$", re.IGNORECASE),
        "The inquiry is broad and underspecified. Please specify which dimension of performance you wish to review.",
        ["domain", "intent"],
        [
            "Commercial sales performance (revenue, orders, units)",
            "Financial gross margin and contribution margin economics",
            "Inventory position and network risk breakdown",
            "Executive cross-domain operational overview",
        ],
    ),
]


def check_query_ambiguity(normalized_question: str) -> Optional[CopilotClarification]:
    """Inspect normalized question and return a CopilotClarification if the query is materially ambiguous."""
    clean = normalized_question.strip().lower()

    for pattern, reason, missing_fields, interpretations in _AMBIGUOUS_PATTERNS:
        if pattern.match(clean):
            cid = "CLR-" + hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]
            return CopilotClarification(
                clarification_id=cid,
                question=f"Could you clarify your request? {reason}",
                reason=reason,
                missing_fields=missing_fields,
                severity=ClarificationSeverity.BLOCKING,
                possible_interpretations=interpretations,
            )

    return None
