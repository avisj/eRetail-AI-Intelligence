"""Deterministic Capability Routing Engine (Phase 7F).

Classifies inbound questions into DATA_QUERY, KNOWLEDGE_QUERY, HYBRID_QUERY,
AMBIGUOUS_QUERY, or UNSUPPORTED_QUERY without probabilistic LLM classifiers.
"""

from __future__ import annotations

import re
from commerce_ai.knowledge.enums import QueryCapability


# Unsupported off-topic keywords (politics, weather, sports, entertainment)
_UNSUPPORTED_PATTERNS = [
    re.compile(r"\b(weather|forecast\s+tomorrow|rain|temperature|climate)\b", re.IGNORECASE),
    re.compile(r"\b(sports|football|basketball|soccer|baseball|olympics|world\s+cup)\b", re.IGNORECASE),
    re.compile(r"\b(politics|election|president|congress|senate|parliament|democrat|republican)\b", re.IGNORECASE),
    re.compile(r"\b(movie|cinema|actor|actress|song|music|recipe|bake|cook)\b", re.IGNORECASE),
    re.compile(r"\b(write\s+(?:a\s+)?(?:poem|joke|story|essay))\b", re.IGNORECASE),
]

# Knowledge / Policy indicators
_KNOWLEDGE_CUES = re.compile(
    r"\b(policy|policies|sop|sops|procedure|procedures|guideline|guidelines|handbook|manual|guide|"
    r"business\s+rule|business\s+rules|rules?|formula|formulas?|define|definition|definitions|glossary|terminology|meaning\s+of|"
    r"how\s+to|what\s+is\s+our\s+policy|what\s+does\s+(?:our\s+)?policy\s+say|what\s+should\s+happen|"
    r"protocol|protocols|standards?|regulations?|compliance|terms\s+and\s+conditions)\b",
    re.IGNORECASE,
)

# Transactional / Analytical Data indicators
_DATA_CUES = re.compile(
    r"\b(sales|revenue|orders|order\s+count|units\s+sold|gmv|gross\s+margin|cogs|margin|profit|"
    r"inventory|stock|stockout|on\s+hand|in\s+transit|inventory\s+risk|holding\s+cost|turnover|"
    r"demand|forecast|lead\s+time|return\s+rate|return\s+volume|returns\s+by\s+channel|"
    r"sku[_-]\w+|wh[_-]\w+|warehouse\s+\w+|supplier\s+\w+|yesterday|last\s+month|last\s+quarter)\b",
    re.IGNORECASE,
)

# Ambiguity cues (very sparse queries)
_AMBIGUOUS_SHORT_PATTERNS = [
    re.compile(r"^(what|why|how|show|tell|help|status|details|query|data|check|info)\??$", re.IGNORECASE),
    re.compile(r"^(inventory|sales|margin|policy|rules)\??$", re.IGNORECASE),
]


# Specific entity cues
_SPECIFIC_ENTITY = re.compile(r"\b(sku[_-]\w+|wh[_-]\w+|sku\d+|wh\d+|warehouse\s+[0-9]+)\b", re.IGNORECASE)

# Data condition / metric assertion cues
_DATA_ASSERTION_CUES = re.compile(
    r"\b((?:risk|margin|stock|rate|sales)\s+is\s+(?:high|critical|low|negative|dropping|increasing|spiking)|"
    r"dropped|increased|spiked|fell|declined|surged|"
    r"current\s+(?:return\s+rate|stock|inventory|margin|sales|rate|demand)\s+is|"
    r"detected|"
    r"\d+(?:\.\d+)?%|\$\d+|\d+\s+units)\b",
    re.IGNORECASE,
)


def classify_query_capability(question: str) -> QueryCapability:
    """Deterministically categorize inbound query into appropriate capability channel."""
    q_clean = question.strip()
    if not q_clean or len(q_clean) < 3:
        return QueryCapability.AMBIGUOUS_QUERY

    # 1. Ambiguity Check
    for pat in _AMBIGUOUS_SHORT_PATTERNS:
        if pat.match(q_clean):
            return QueryCapability.AMBIGUOUS_QUERY

    # 2. Unsupported / Off-topic Check
    for pat in _UNSUPPORTED_PATTERNS:
        if pat.search(q_clean):
            return QueryCapability.UNSUPPORTED_QUERY

    has_knowledge = bool(_KNOWLEDGE_CUES.search(q_clean))
    has_data = bool(_DATA_CUES.search(q_clean))
    has_entity = bool(_SPECIFIC_ENTITY.search(q_clean))
    has_assertion = bool(_DATA_ASSERTION_CUES.search(q_clean))

    # 3. Hybrid Check: Requires knowledge cues combined with specific entity, live metric condition, or dual request
    if has_knowledge and (has_entity or has_assertion):
        return QueryCapability.HYBRID_QUERY

    if has_knowledge and has_data and re.search(r"\b(and\s+show|and\s+what\s+(?:are|is)|show\s+.*\s+and)\b", q_clean, re.IGNORECASE):
        return QueryCapability.HYBRID_QUERY

    # 4. Pure Knowledge Check
    if has_knowledge:
        return QueryCapability.KNOWLEDGE_QUERY

    # 5. Pure Data Check
    if has_data:
        return QueryCapability.DATA_QUERY

    # If neither explicit cues match, check if query starts with definition inquiry
    if re.match(r"^(?:define|explain|what\s+is|what\s+are)\b", q_clean, re.IGNORECASE):
        return QueryCapability.KNOWLEDGE_QUERY

    return QueryCapability.AMBIGUOUS_QUERY
