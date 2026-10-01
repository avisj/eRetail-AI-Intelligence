"""Copilot Orchestration State Machine & Pipeline Coordinator (Phase 7C).

Coordinates the end-to-end Copilot request lifecycle through deterministic state transitions:
RECEIVED -> NORMALIZED -> INTERPRETED -> (CLARIFICATION_REQUIRED) ->
CONTRACT_BUILT -> PLANNED -> EXECUTING -> COMPLETED / PARTIAL / UNAVAILABLE / FAILED.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

from commerce_ai.copilot.ambiguity import check_query_ambiguity
from commerce_ai.copilot.decomposition import decompose_and_plan
from commerce_ai.copilot.enums import ClarificationSeverity, CopilotState, PlanExecutionStatus
from commerce_ai.copilot.execution import execute_plan
from commerce_ai.copilot.governance import (
    validate_plan_governance,
    validate_request_governance,
)
from commerce_ai.copilot.interpreter import interpret_question
from commerce_ai.copilot.normalization import normalize_question
from commerce_ai.copilot.schemas import (
    CopilotClarification,
    CopilotExecutionPlan,
    CopilotExecutionResult,
    CopilotGovernance,
    CopilotInterpretation,
    CopilotRequest,
    CopilotResponse,
)
from commerce_ai.query_contracts.service import QueryContractService
from commerce_ai.query_layer.service import QueryLayerService


class CopilotOrchestrator:
    """Deterministic orchestrator managing state transitions and workflow execution."""

    def __init__(
        self,
        query_layer: Optional[QueryLayerService] = None,
        contract_service: Optional[QueryContractService] = None,
    ) -> None:
        self.query_layer = query_layer or QueryLayerService()
        self.contract_service = contract_service or QueryContractService()

    def orchestrate(
        self,
        request: CopilotRequest,
        execute: bool = True,
    ) -> CopilotResponse:
        """Execute the complete deterministic orchestration pipeline for an inbound request."""
        req_id = request.request_id or "REQ-" + hashlib.sha256(request.question.encode("utf-8")).hexdigest()[:16]
        resp_id = "RSP-" + hashlib.sha256((req_id + request.question).encode("utf-8")).hexdigest()[:16]
        warnings: List[str] = []
        errors: List[str] = []

        gov = CopilotGovernance(
            read_only=True,
            action_execution=False,
            execution_allowed=False,
            no_ranking_enforced=True,
            no_decision_selection_enforced=True,
        )

        # 1. State: RECEIVED -> NORMALIZED
        norm_q = normalize_question(request.question)

        # 2. Security & Prompt Injection check
        is_safe, sec_violations = validate_request_governance(request)
        if not is_safe:
            errors.extend(sec_violations)
            return CopilotResponse(
                response_id=resp_id,
                request_id=req_id,
                state=CopilotState.FAILED,
                normalized_question=norm_q,
                governance=gov,
                warnings=warnings,
                errors=errors,
            )

        # 3. Ambiguity check -> CLARIFICATION_REQUIRED
        clarification = check_query_ambiguity(norm_q)
        if clarification:
            return CopilotResponse(
                response_id=resp_id,
                request_id=req_id,
                state=CopilotState.CLARIFICATION_REQUIRED,
                normalized_question=norm_q,
                clarification=clarification,
                governance=gov,
                warnings=warnings,
                errors=errors,
            )

        # 4. State: INTERPRETED
        interpretation = interpret_question(request)
        if not interpretation:
            unrec_clar = CopilotClarification(
                clarification_id="CLR-" + hashlib.sha256(norm_q.encode("utf-8")).hexdigest()[:16],
                question=norm_q,
                reason="The question could not be safely mapped to an authorized business intent or capability. Please select from the supported analytical inquiries.",
                missing_fields=["domain", "intent"],
                possible_interpretations=[
                    "Commercial sales performance (revenue, orders, units)",
                    "Financial gross margin and profitability analysis",
                    "Network inventory position and risk analysis",
                    "Demand trends and statistical forecast",
                    "Customer return rates and reason breakdown",
                ],
                severity=ClarificationSeverity.BLOCKING,
            )
            return CopilotResponse(
                response_id=resp_id,
                request_id=req_id,
                state=CopilotState.CLARIFICATION_REQUIRED,
                normalized_question=norm_q,
                clarification=unrec_clar,
                governance=gov,
                warnings=["QUESTION_UNRECOGNIZED: Question cannot be safely mapped to a canonical Phase 7B intent."],
                errors=[],
            )

        # 5. State: CONTRACT_BUILT -> PLANNED
        plan, decomp_warnings, decomp_errors = decompose_and_plan(
            interpretation=interpretation,
            request=request,
            contract_service=self.contract_service,
        )
        warnings.extend(decomp_warnings)
        errors.extend(decomp_errors)

        if not plan:
            return CopilotResponse(
                response_id=resp_id,
                request_id=req_id,
                state=CopilotState.FAILED,
                normalized_question=norm_q,
                interpretation=interpretation,
                governance=gov,
                warnings=warnings,
                errors=errors,
            )

        # If capability is unavailable in Phase 7A (e.g. DATA_QUALITY_ANALYSIS)
        if plan.state == CopilotState.UNAVAILABLE:
            return CopilotResponse(
                response_id=resp_id,
                request_id=req_id,
                state=CopilotState.UNAVAILABLE,
                normalized_question=norm_q,
                interpretation=interpretation,
                execution_plan=plan,
                governance=gov,
                warnings=warnings,
                errors=errors,
            )

        # 6. If execution is not requested, return the PLANNED state
        if not execute:
            return CopilotResponse(
                response_id=resp_id,
                request_id=req_id,
                state=CopilotState.PLANNED,
                normalized_question=norm_q,
                interpretation=interpretation,
                execution_plan=plan,
                governance=gov,
                warnings=warnings,
                errors=errors,
            )

        # 7. State: EXECUTING -> COMPLETED / PARTIAL / FAILED
        exec_result = execute_plan(plan, query_layer=self.query_layer)

        if exec_result.status == PlanExecutionStatus.COMPLETE:
            final_state = CopilotState.COMPLETED
        elif exec_result.status == PlanExecutionStatus.PARTIAL:
            final_state = CopilotState.PARTIAL
        elif exec_result.status == PlanExecutionStatus.UNAVAILABLE:
            final_state = CopilotState.UNAVAILABLE
        else:
            final_state = CopilotState.FAILED

        return CopilotResponse(
            response_id=resp_id,
            request_id=req_id,
            state=final_state,
            normalized_question=norm_q,
            interpretation=interpretation,
            execution_plan=plan,
            execution_result=exec_result,
            governance=plan.governance,
            warnings=warnings,
            errors=errors,
        )
