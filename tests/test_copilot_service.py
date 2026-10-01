"""Unit tests for CopilotService High-Level API (Phase 7C).

Tests:
- CopilotService.process() with string questions and CopilotRequest objects
- CopilotService.interpret() extraction and parameter mapping
- CopilotService.plan() deterministic plan generation
- CopilotService.execute() execution of synthesized plans
- Parameter forwarding (as_of_date, currency, filters, context)
- Deterministic execution repeatability across repeated invocations
- Comprehensive coverage across diverse business domains
"""

from __future__ import annotations

import pytest

from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus
from commerce_ai.copilot.schemas import CopilotRequest
from commerce_ai.copilot.service import CopilotService
from commerce_ai.query_contracts.enums import (
    BusinessDomain,
    QueryIntent,
    RequestedOutput,
)


class TestCopilotService:
    @pytest.fixture
    def service(self) -> CopilotService:
        return CopilotService()

    def test_process_with_string_question(self, service: CopilotService):
        resp = service.process("How are sales performing?", as_of_date="2026-06-30", currency="USD")
        assert resp.state == CopilotState.COMPLETED
        assert resp.interpretation.intent == QueryIntent.SALES_PERFORMANCE
        assert resp.execution_result.status == PlanExecutionStatus.COMPLETE

    def test_process_with_copilot_request_object(self, service: CopilotService):
        req = CopilotRequest(question="What is our margin?", as_of_date="2026-06-30", currency="USD")
        resp = service.process(req)
        assert resp.state == CopilotState.COMPLETED
        assert resp.interpretation.intent == QueryIntent.MARGIN_ANALYSIS

    def test_interpret_api(self, service: CopilotService):
        interp = service.interpret("Show revenue trend for SKU_001", as_of_date="2026-06-30")
        assert interp.intent == QueryIntent.SALES_PERFORMANCE
        assert interp.filters["sku_id"] == "SKU_001"
        assert interp.requested_output == RequestedOutput.TIME_SERIES

    def test_plan_api_without_execution(self, service: CopilotService):
        plan = service.plan("Why did margin decline?", as_of_date="2026-06-30")
        assert plan is not None
        assert len(plan.steps) == 4
        assert plan.state == CopilotState.PLANNED

    def test_execute_pre_synthesized_plan(self, service: CopilotService):
        plan = service.plan("What is our margin?", as_of_date="2026-06-30", currency="USD")
        assert plan is not None
        result = service.execute(plan)
        assert result.status == PlanExecutionStatus.COMPLETE
        assert len(result.step_results) >= 1

    def test_replenishment_review_inquiry(self, service: CopilotService):
        resp = service.process("What should I review regarding replenishment?", as_of_date="2026-06-30")
        assert resp.state == CopilotState.COMPLETED
        assert resp.interpretation.intent == QueryIntent.REPLENISHMENT_REVIEW
        assert resp.governance.approval_required is True
        assert resp.governance.execution_allowed is False

    def test_purchase_order_review_inquiry(self, service: CopilotService):
        resp = service.process("Review purchase orders awaiting approval", as_of_date="2026-06-30")
        assert resp.state == CopilotState.COMPLETED
        assert resp.interpretation.intent == QueryIntent.PURCHASE_ORDER_REVIEW

    def test_warehouse_rebalancing_review_inquiry(self, service: CopilotService):
        resp = service.process("Show warehouse rebalancing review", as_of_date="2026-06-30")
        assert resp.state == CopilotState.COMPLETED
        assert resp.interpretation.intent == QueryIntent.WAREHOUSE_REBALANCING_REVIEW

    def test_forecast_accuracy_inquiry(self, service: CopilotService):
        resp = service.process("How accurate is the forecast?", as_of_date="2026-06-30")
        assert resp.state == CopilotState.COMPLETED
        assert resp.interpretation.intent == QueryIntent.FORECAST_ACCURACY

    def test_multi_domain_inquiry(self, service: CopilotService):
        resp = service.process(
            "How are sales, inventory, returns and margin connected?",
            as_of_date="2026-06-30",
            currency="USD",
        )
        assert resp.state == CopilotState.COMPLETED
        assert resp.interpretation.intent == QueryIntent.MULTI_DOMAIN_ANALYSIS
        assert resp.interpretation.domain == BusinessDomain.CROSS_DOMAIN

    def test_currency_isolation_forwarding(self, service: CopilotService):
        resp = service.process("What is our margin in EUR?", currency="EUR", as_of_date="2026-06-30")
        assert resp.state == CopilotState.COMPLETED
        assert resp.interpretation.filters["currency"] == "EUR"
        assert resp.execution_plan.steps[0].arguments["currency"] == "EUR"

    def test_as_of_date_forwarding(self, service: CopilotService):
        resp = service.process("How are sales performing?", as_of_date="2026-03-31")
        assert resp.state == CopilotState.COMPLETED
        assert resp.execution_plan.steps[0].arguments["as_of_date"] == "2026-03-31"

    def test_deterministic_repeatability(self, service: CopilotService):
        q = "Why did margin decline?"
        resp1 = service.process(q, as_of_date="2026-06-30", currency="USD")
        resp2 = service.process(q, as_of_date="2026-06-30", currency="USD")

        assert resp1.interpretation.interpretation_id == resp2.interpretation.interpretation_id
        assert resp1.execution_plan.plan_id == resp2.execution_plan.plan_id
        assert len(resp1.execution_plan.steps) == len(resp2.execution_plan.steps)
        assert resp1.execution_result.status == resp2.execution_result.status
