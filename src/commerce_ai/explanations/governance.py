"""Governance Validation and Prohibition Enforcement for Business Explanations (Phase 7D).

Strictly verifies that explanations do not violate:
1. Ranking prohibitions (no top-N, winners, losers, leaderboards)
2. Causality invention prohibitions (no ungrounded causal claims)
3. Action generation prohibitions (no PO, transfer, or price recommendations)
4. Decision selection prohibitions (selected_option must remain null)
5. Sensationalism prohibitions (no disaster/catastrophic language)
"""

from __future__ import annotations

import re
from typing import List, Tuple

from commerce_ai.explanations.schemas import BusinessExplanation, Finding

# Prohibited ranking patterns
_RANKING_PATTERNS = [
    re.compile(r"\b(top\s+\d+|bottom\s+\d+|best\s+performing|worst\s+performing|leaderboard|winner(s)?|loser(s)?|rank(ed)?\s+#?\d+)\b", re.IGNORECASE),
]

# Prohibited ungrounded causal claim patterns
_CAUSAL_PATTERNS = [
    re.compile(r"\b((is|are|was|were)\s+caused\s+by|caused\s+the\s+decline|caused\s+the\s+increase|directly\s+caused|the\s+root\s+cause\s+is)\b", re.IGNORECASE),
]

# Prohibited action generation patterns
_ACTION_PATTERNS = [
    re.compile(r"\b(you\s+should\s+(order|buy|transfer|raise|lower|discount)|place\s+(a\s+)?purchase\s+order|execute\s+(a\s+)?transfer|change\s+the\s+price)\b", re.IGNORECASE),
]

# Prohibited sensationalism patterns
_SENSATIONAL_PATTERNS = [
    re.compile(r"\b(critical\s+disaster|massive\s+failure|terrible\s+performance|catastrophe|complete\s+disaster)\b", re.IGNORECASE),
]


def check_text_for_prohibited_patterns(text: str) -> List[str]:
    """Inspect text for prohibited ranking, ungrounded causal, action, or sensational phrases."""
    violations: List[str] = []

    for pat in _RANKING_PATTERNS:
        match = pat.search(text)
        if match:
            violations.append(f"Ranking prohibition violation: found '{match.group(0)}'")

    for pat in _CAUSAL_PATTERNS:
        match = pat.search(text)
        if match:
            violations.append(f"Ungrounded causality violation: found '{match.group(0)}'")

    for pat in _ACTION_PATTERNS:
        match = pat.search(text)
        if match:
            violations.append(f"Action generation prohibition violation: found '{match.group(0)}'")

    for pat in _SENSATIONAL_PATTERNS:
        match = pat.search(text)
        if match:
            violations.append(f"Sensational language violation: found '{match.group(0)}'")

    return violations


def validate_explanation_governance(explanation: BusinessExplanation) -> Tuple[bool, List[str]]:
    """Validate full BusinessExplanation container against governance invariants."""
    violations: List[str] = []

    # 1. Invariant flags
    gov = explanation.governance
    if not gov.read_only:
        violations.append("Governance error: explanation read_only must be True.")
    if gov.action_execution:
        violations.append("Governance error: action_execution must be False.")
    if gov.execution_allowed:
        violations.append("Governance error: execution_allowed must be False.")
    if not gov.no_ranking_enforced:
        violations.append("Governance error: no_ranking_enforced must be True.")
    if not gov.no_decision_selection_enforced:
        violations.append("Governance error: no_decision_selection_enforced must be True.")
    if not gov.no_causality_invented:
        violations.append("Governance error: no_causality_invented must be True.")
    if not gov.no_recommendation_generated:
        violations.append("Governance error: no_recommendation_generated must be True.")

    # 2. Check headline and summary text
    violations.extend(check_text_for_prohibited_patterns(explanation.headline))
    violations.extend(check_text_for_prohibited_patterns(explanation.summary))

    # 3. Check each finding statement
    for finding in explanation.key_findings:
        v_list = check_text_for_prohibited_patterns(finding.statement)
        for v in v_list:
            violations.append(f"Finding '{finding.finding_id}': {v}")

    # 4. Check risks and limitations
    for r in explanation.risks:
        violations.extend(check_text_for_prohibited_patterns(r))
    for l in explanation.limitations:
        violations.extend(check_text_for_prohibited_patterns(l))

    is_valid = len(violations) == 0
    return is_valid, violations
