"""High-Level Copilot Service API Boundary (Phase 7C).

Provides the unified entry point for dashboards, chat interfaces, and future agentic workflows:
- CopilotService.process(request): End-to-end question-to-evidence pipeline
- CopilotService.interpret(request): Deterministic question parsing & intent mapping
- CopilotService.plan(request): Synthesize validated Phase 7B execution plans without execution
- CopilotService.execute(plan): Execute pre-planned reasoning steps with failure isolation
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Union

from commerce_ai.copilot.execution import execute_plan
from commerce_ai.copilot.interpreter import interpret_question
from commerce_ai.copilot.orchestration import CopilotOrchestrator
from commerce_ai.copilot.schemas import (
    CopilotExecutionPlan,
    CopilotExecutionResult,
    CopilotInterpretation,
    CopilotRequest,
    CopilotResponse,
)
from commerce_ai.query_contracts.enums import RequestedOutput
from commerce_ai.query_contracts.service import QueryContractService
from commerce_ai.query_layer.service import QueryLayerService


class CopilotService:
    """Unified service for deterministic Copilot reasoning and tool execution orchestration."""

    def __init__(
        self,
        query_layer: Optional[QueryLayerService] = None,
        contract_service: Optional[QueryContractService] = None,
    ) -> None:
        self.query_layer = query_layer or QueryLayerService()
        self.contract_service = contract_service or QueryContractService()
        self.orchestrator = CopilotOrchestrator(
            query_layer=self.query_layer,
            contract_service=self.contract_service,
        )

    def _normalize_request(
        self,
        request: Union[CopilotRequest, str],
        as_of_date: Optional[str] = None,
        currency: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        requested_output: Optional[RequestedOutput] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CopilotRequest:
        """Coerce string or CopilotRequest into a validated CopilotRequest object."""
        if isinstance(request, CopilotRequest):
            return request

        req_id = "REQ-" + hashlib.sha256(request.encode("utf-8")).hexdigest()[:16]
        return CopilotRequest(
            question=request,
            request_id=req_id,
            as_of_date=as_of_date,
            currency=currency,
            filters=filters or {},
            requested_output=requested_output,
            context=context or {},
        )

    def process(
        self,
        request: Union[CopilotRequest, str],
        as_of_date: Optional[str] = None,
        currency: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        requested_output: Optional[RequestedOutput] = None,
        context: Optional[Dict[str, Any]] = None,
        execute: bool = True,
    ) -> CopilotResponse:
        """Process a business question through the full Copilot reasoning and orchestration pipeline."""
        req = self._normalize_request(
            request=request,
            as_of_date=as_of_date,
            currency=currency,
            filters=filters,
            requested_output=requested_output,
            context=context,
        )
        return self.orchestrator.orchestrate(req, execute=execute)

    def interpret(
        self,
        request: Union[CopilotRequest, str],
        as_of_date: Optional[str] = None,
        currency: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> CopilotInterpretation:
        """Interpret a business question into structured intent and entity specifications."""
        req = self._normalize_request(
            request=request,
            as_of_date=as_of_date,
            currency=currency,
            filters=filters,
        )
        return interpret_question(req)

    def plan(
        self,
        request: Union[CopilotRequest, str],
        as_of_date: Optional[str] = None,
        currency: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        requested_output: Optional[RequestedOutput] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CopilotExecutionPlan]:
        """Synthesize a validated CopilotExecutionPlan without executing tools."""
        resp = self.process(
            request=request,
            as_of_date=as_of_date,
            currency=currency,
            filters=filters,
            requested_output=requested_output,
            context=context,
            execute=False,
        )
        return resp.execution_plan

    def execute(
        self,
        plan: CopilotExecutionPlan,
    ) -> CopilotExecutionResult:
        """Execute a previously synthesized CopilotExecutionPlan."""
        return execute_plan(plan, query_layer=self.query_layer)
