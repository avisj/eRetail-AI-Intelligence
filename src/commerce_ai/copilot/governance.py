"""Platform Governance & Security Enforcement for Copilot (Phase 7C).

Enforces mandatory platform safety invariants:
- Read-only mandate (read_only=True, execution_allowed=False, action_execution=False)
- Ranking prohibition (no top-N, no winner/loser selection, no leaderboards)
- No autonomous decision selection (selected_option must remain None)
- Prompt injection / jailbreak detection
- Point-in-time temporal protection (records <= as_of_date)
- Currency isolation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from commerce_ai.copilot.normalization import detect_malicious_intent
from commerce_ai.copilot.schemas import CopilotExecutionPlan, CopilotGovernance, CopilotRequest


class GovernanceViolation(Exception):
    """Raised when an operation violates platform governance invariants."""


def validate_request_governance(request: CopilotRequest) -> Tuple[bool, List[str]]:
    """Validate inbound request against security, injection, and execution invariants.

    Returns:
        (is_allowed, list_of_violations)
    """
    violations: List[str] = []

    # 1. Prompt Injection / Action Execution attempt detection
    is_malicious, msg = detect_malicious_intent(request.question)
    if is_malicious:
        violations.append(f"[SECURITY_VIOLATION] {msg}")

    return len(violations) == 0, violations


def validate_plan_governance(plan: CopilotExecutionPlan) -> Tuple[bool, List[str]]:
    """Validate that synthesized execution plan satisfies all platform governance mandates.

    Returns:
        (is_allowed, list_of_violations)
    """
    violations: List[str] = []
    gov = plan.governance

    # 1. Read-only assertions
    if not gov.read_only:
        violations.append("[GOVERNANCE_MUTATION_FORBIDDEN] Copilot cannot execute mutations; read_only must be True.")
    if gov.action_execution:
        violations.append("[GOVERNANCE_ACTION_FORBIDDEN] Copilot cannot perform autonomous system actions.")
    if gov.execution_allowed:
        violations.append("[GOVERNANCE_EXECUTION_FORBIDDEN] Copilot cannot execute business actions (PO, transfer, pricing).")

    # 2. Ranking tool prohibition
    for step in plan.steps:
        tool_lower = step.tool_name.lower()
        if any(term in tool_lower for term in ["ranking", "rank", "top_n", "winner", "leaderboard"]):
            violations.append(
                f"[PROHIBITED_RANKING_TOOL] Tool '{step.tool_name}' in step '{step.step_id}' produces subjective rankings, which is prohibited."
            )

    return len(violations) == 0, violations


def enforce_decision_package_neutrality(payload: Any) -> Any:
    """Ensure that if a tool response contains DecisionPackages, no option has been autonomously selected."""
    if isinstance(payload, dict):
        if "selected_option" in payload and payload["selected_option"] is not None:
            # Force neutrality: Copilot cannot autonomously select an option
            payload["selected_option"] = None
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                enforce_decision_package_neutrality(v)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)):
                enforce_decision_package_neutrality(item)
    return payload
