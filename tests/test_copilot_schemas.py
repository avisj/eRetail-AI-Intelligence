"""Unit tests for Copilot Pydantic Schemas (Phase 7C).

Tests validation, serialization, immutability, and governance invariants
for all Copilot data models.
"""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from commerce_ai.copilot.enums import (
    ClarificationSeverity,
    CopilotProvenanceSource,
    CopilotState,
    InterpretationConfidence,
    InterpretationSource,
    PlanExecutionStatus,
    StepStatus,
)
from commerce_ai.copilot.schemas import (
    CopilotClarification,
    CopilotEvidence,
    CopilotExecutionPlan,
    CopilotExecutionResult,
    CopilotFailure,
    CopilotGovernance,
    CopilotInterpretation,
    CopilotRequest,
    CopilotResponse,
    CopilotStep,
    CopilotStepResult,
)
from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    BusinessGrain,
    MetricIdentifier,
    QueryIntent,
    RequestedOutput,
    TimeGranularity,
)


class TestCopilotSchemas:
    def test_copilot_request_minimal(self):
        req = CopilotRequest(question="How are sales performing?")
        assert req.question == "How are sales performing?"
        assert req.request_id is None
        assert req.as_of_date is None
        assert req.currency is None
        assert req.filters == {}
        assert req.context == {}

    def test_copilot_request_full(self):
        req = CopilotRequest(
            question="Show margin for warehouse WH_01",
            request_id="REQ-12345",
            as_of_date="2026-06-30",
            currency="USD",
            filters={"warehouse_id": "WH_01"},
            requested_output=RequestedOutput.BREAKDOWN,
            context={"user_role": "analyst"},
            metadata={"source": "slack"},
        )
        assert req.request_id == "REQ-12345"
        assert req.filters["warehouse_id"] == "WH_01"
        assert req.requested_output == RequestedOutput.BREAKDOWN

    def test_copilot_request_forbids_extra(self):
        with pytest.raises(ValidationError):
            CopilotRequest(question="Valid question", unauthorized_field="bad")

    def test_copilot_clarification_frozen(self):
        clar = CopilotClarification(
            clarification_id="CLR-001",
            question="Which warehouse do you mean?",
            reason="Multiple warehouses available",
            missing_fields=["warehouse_id"],
            severity=ClarificationSeverity.BLOCKING,
            possible_interpretations=["Warehouse WH_01", "Warehouse WH_02"],
        )
        assert clar.clarification_id == "CLR-001"
        assert clar.severity == ClarificationSeverity.BLOCKING
        with pytest.raises(ValidationError):
            clar.question = "Mutated question"

    def test_copilot_governance_invariants(self):
        gov = CopilotGovernance()
        assert gov.read_only is True
        assert gov.action_execution is False
        assert gov.execution_allowed is False
        assert gov.no_ranking_enforced is True
        assert gov.no_decision_selection_enforced is True

    def test_copilot_interpretation(self):
        interp = CopilotInterpretation(
            interpretation_id="INT-001",
            normalized_question="how are sales performing",
            intent=QueryIntent.SALES_PERFORMANCE,
            domain=BusinessDomain.SALES,
            metrics=[MetricIdentifier.NET_REVENUE],
            dimensions=[BusinessDimension.WAREHOUSE],
            filters={"warehouse_id": "WH_01"},
            time_granularity=TimeGranularity.DAY,
            requested_grain=BusinessGrain.WAREHOUSE,
            requested_output=RequestedOutput.KPI,
            confidence=InterpretationConfidence.HIGH,
            source=InterpretationSource.DETERMINISTIC_RULE,
        )
        assert interp.intent == QueryIntent.SALES_PERFORMANCE
        assert interp.domain == BusinessDomain.SALES
        assert interp.confidence == InterpretationConfidence.HIGH

    def test_copilot_step(self):
        step = CopilotStep(
            step_id="STEP-1",
            sequence=1,
            contract_id="QRY-abc",
            plan_id="PLAN-xyz",
            tool_name="get_sales_summary",
            arguments={"as_of_date": "2026-06-30"},
            purpose="Retrieve sales baseline KPIs",
            required=True,
            dependencies=[],
            status=StepStatus.PENDING,
        )
        assert step.step_id == "STEP-1"
        assert step.sequence == 1
        assert step.tool_name == "get_sales_summary"
        assert step.status == StepStatus.PENDING

    def test_copilot_evidence(self):
        evi = CopilotEvidence(
            evidence_id="EVI-001",
            step_id="STEP-1",
            tool_name="get_sales_summary",
            source_engine="commerce_ai.sales_intelligence",
            metrics=["net_revenue", "order_count"],
            entity_count=10,
            as_of_date="2026-06-30",
            currency="USD",
            data_summary={"net_revenue": 100000.0},
            provenance=CopilotProvenanceSource.PHASE_7A_TOOL,
        )
        assert evi.evidence_id == "EVI-001"
        assert evi.provenance == CopilotProvenanceSource.PHASE_7A_TOOL
        assert evi.source_engine == "commerce_ai.sales_intelligence"

    def test_copilot_failure(self):
        fail = CopilotFailure(
            step_id="STEP-2",
            tool_name="get_margin_drivers",
            error_code="DATA_NOT_FOUND",
            error_message="No transaction records for period",
            fatal=False,
        )
        assert fail.error_code == "DATA_NOT_FOUND"
        assert fail.fatal is False

    def test_copilot_step_result(self):
        res = CopilotStepResult(
            step_id="STEP-1",
            tool_name="get_sales_summary",
            status=StepStatus.COMPLETED,
            execution_time_ms=12.5,
            warnings=["Low sample size"],
        )
        assert res.status == StepStatus.COMPLETED
        assert res.execution_time_ms == 12.5
        assert len(res.warnings) == 1

    def test_copilot_execution_result_aggregation(self):
        res = CopilotExecutionResult(
            result_id="RES-001",
            request_id="REQ-001",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[],
            evidence_chain=[],
            failures=[],
            total_execution_time_ms=45.2,
        )
        assert res.status == PlanExecutionStatus.COMPLETE
        assert res.total_execution_time_ms == 45.2

    def test_copilot_response_serialization(self):
        resp = CopilotResponse(
            response_id="RSP-001",
            request_id="REQ-001",
            state=CopilotState.COMPLETED,
            normalized_question="how are sales performing",
            warnings=["Advisory note"],
            errors=[],
        )
        data = resp.model_dump()
        assert data["state"] == "COMPLETED"
        assert data["response_id"] == "RSP-001"
        # Check JSON roundtrip
        json_str = resp.model_dump_json()
        deserialized = json.loads(json_str)
        assert deserialized["state"] == "COMPLETED"
