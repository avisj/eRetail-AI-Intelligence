"""Query Decomposition & Phase 7B Plan Synthesis (Phase 7C).

Transforms a CopilotInterpretation into one or more formal Phase 7B BusinessQueryContract
and QueryPlan instances. Strictly enforces that all executable steps originate from
Phase 7B, guaranteeing zero architectural bypass.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from commerce_ai.copilot.enums import CopilotState, StepStatus
from commerce_ai.copilot.schemas import (
    CopilotExecutionPlan,
    CopilotGovernance,
    CopilotInterpretation,
    CopilotRequest,
    CopilotStep,
)
from commerce_ai.query_contracts.enums import PlanStatus
from commerce_ai.query_contracts.schemas import BusinessQueryContract, QueryPlan
from commerce_ai.query_contracts.service import QueryContractService


def decompose_and_plan(
    interpretation: CopilotInterpretation,
    request: CopilotRequest,
    contract_service: Optional[QueryContractService] = None,
) -> Tuple[Optional[CopilotExecutionPlan], List[str], List[str]]:
    """Decompose an interpretation into validated Phase 7B contracts and synthesize a CopilotExecutionPlan.

    Returns:
        (plan, warnings, errors)
    """
    service = contract_service or QueryContractService()
    warnings: List[str] = []
    errors: List[str] = []

    # Build primary Phase 7B contract
    contract, val_res = service.build_contract(
        intent=interpretation.intent,
        domain=interpretation.domain,
        domains=interpretation.domains,
        metrics=interpretation.metrics,
        dimensions=interpretation.dimensions,
        filters=interpretation.filters,
        time_range=interpretation.time_range,
        as_of_date=request.as_of_date or (interpretation.time_range.get("as_of_date") if interpretation.time_range else None),
        currency=request.currency or interpretation.filters.get("currency"),
        requested_grain=interpretation.requested_grain,
        time_granularity=interpretation.time_granularity,
        requested_output=interpretation.requested_output,
        explanation_context=interpretation.explanation_context,
    )

    # Collect validation warnings and errors
    for w in val_res.warnings:
        warnings.append(f"[{w.code}] {w.message}")
    for e in val_res.errors:
        errors.append(f"[{e.code}] {e.message}")

    if not val_res.is_valid:
        return None, warnings, errors

    # Synthesize Phase 7B QueryPlan
    try:
        q_plan = service.plan(contract)
    except Exception as exc:
        errors.append(f"Phase 7B QueryPlan generation failed: {str(exc)}")
        return None, warnings, errors

    req_id = request.request_id or "REQ-anonymous"
    cplan_id = "CPLAN-" + hashlib.sha256((contract.query_id + q_plan.plan_id).encode("utf-8")).hexdigest()[:16]

    steps: List[CopilotStep] = []
    step_deps: Dict[str, List[str]] = {}

    if q_plan.status == PlanStatus.UNAVAILABLE:
        # Capability is recognized by 7B but unavailable in Phase 7A registry (e.g. DATA_QUALITY_ANALYSIS)
        plan_state = CopilotState.UNAVAILABLE
    else:
        plan_state = CopilotState.PLANNED
        for i, tool_step in enumerate(q_plan.steps, start=1):
            sid = f"STEP-{i}"
            deps = list(q_plan.dependencies.get(tool_step.tool_name, []))
            step_deps[sid] = deps
            steps.append(
                CopilotStep(
                    step_id=sid,
                    sequence=tool_step.sequence,
                    contract_id=contract.query_id,
                    plan_id=q_plan.plan_id,
                    tool_name=tool_step.tool_name,
                    arguments=tool_step.arguments,
                    purpose=tool_step.purpose,
                    required=tool_step.required,
                    dependencies=deps,
                    status=StepStatus.READY if not deps else StepStatus.PENDING,
                )
            )

    execution_order = [s.step_id for s in steps]

    gov = CopilotGovernance(
        read_only=True,
        action_execution=False,
        execution_allowed=False,
        approval_required=contract.governance.approval_required,
        no_ranking_enforced=True,
        no_decision_selection_enforced=True,
    )

    exec_plan = CopilotExecutionPlan(
        plan_id=cplan_id,
        request_id=req_id,
        steps=steps,
        execution_order=execution_order,
        dependencies=step_deps,
        governance=gov,
        contracts=[contract],
        query_plans=[q_plan],
        state=plan_state,
    )

    return exec_plan, warnings, errors
