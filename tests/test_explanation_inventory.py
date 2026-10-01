"""Unit tests for Inventory Explanations (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import (
    CopilotExecutionResult,
    CopilotResponse,
    CopilotStepResult,
)
from commerce_ai.copilot.service import CopilotService
from commerce_ai.explanations.enums import ExplanationStatus, ExplanationType
from commerce_ai.explanations.service import ExplanationService
from commerce_ai.query_layer.schemas import (
    CalculationStatus,
    MetricResult,
    QueryMetadata,
    QueryResponse,
    TableResult,
)
from commerce_ai.query_layer.service import QueryLayerService


class TestInventoryExplanations:
    def test_inventory_position_explanation(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process(
            "What is our inventory position?",
            as_of_date="2026-06-30",
        )
        assert copilot_resp.execution_result is not None

        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.supporting_evidence) > 0

    def test_inventory_risk_does_not_generate_unauthorized_actions(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process(
            "Where is stockout risk?",
            as_of_date="2026-06-30",
        )
        assert copilot_resp.execution_result is not None

        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS

        full_text = exp.headline + " " + exp.summary + " " + " ".join(f.statement for f in exp.key_findings)
        assert "place order" not in full_text.lower()
        assert "buy" not in full_text.lower()
        assert "transfer immediately" not in full_text.lower()

    def test_slow_moving_inventory_explanation(self):
        meta = QueryMetadata(
            query_id="Q-SLOW",
            tool_name="get_slow_moving_inventory",
            domain="inventory",
            generated_as_of="2026-06-30",
            source_engine="commerce_ai.inventory",
        )
        table = TableResult(
            columns=["sku_id", "warehouse_id", "days_of_supply", "on_hand_units"],
            rows=[
                {"sku_id": "SKU_02", "warehouse_id": "WH_01", "days_of_supply": 120, "on_hand_units": 100},
                {"sku_id": "SKU_03", "warehouse_id": "WH_02", "days_of_supply": 95, "on_hand_units": 50},
            ],
            total_rows=2,
            metadata=meta,
        )
        q_resp = QueryResponse(
            query_id="Q-SLOW",
            tool_name="get_slow_moving_inventory",
            domain="inventory",
            metadata=meta,
            table=table,
        )
        step = CopilotStepResult(
            step_id="S-SLOW",
            tool_name="get_slow_moving_inventory",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-SLOW",
            request_id="REQ-SLOW",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-SLOW",
            request_id="REQ-SLOW",
            state=CopilotState.COMPLETED,
            normalized_question="Show slow moving inventory",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.key_findings) >= 1
        # Assert no subjective ranking is used
        assert "worst" not in exp.summary.lower()

    def test_high_value_inventory_explanation(self):
        meta = QueryMetadata(
            query_id="Q-HV",
            tool_name="get_high_value_inventory",
            domain="inventory",
            generated_as_of="2026-06-30",
            currency="USD",
            source_engine="commerce_ai.inventory",
        )
        table = TableResult(
            columns=["sku_id", "warehouse_id", "inventory_valuation"],
            rows=[
                {"sku_id": "SKU_01", "warehouse_id": "WH_01", "inventory_valuation": 3000.0},
            ],
            total_rows=1,
            metadata=meta,
        )
        q_resp = QueryResponse(
            query_id="Q-HV",
            tool_name="get_high_value_inventory",
            domain="inventory",
            metadata=meta,
            table=table,
        )
        step = CopilotStepResult(
            step_id="S-HV",
            tool_name="get_high_value_inventory",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-HV",
            request_id="REQ-HV",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-HV",
            request_id="REQ-HV",
            state=CopilotState.COMPLETED,
            normalized_question="Show high value inventory",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.key_findings) >= 1

    def test_inventory_summary_kpi_explanation(self):
        meta = QueryMetadata(
            query_id="Q-INV-SUM",
            tool_name="get_inventory_summary",
            domain="inventory",
            generated_as_of="2026-06-30",
            currency="USD",
            source_engine="commerce_ai.inventory",
        )
        metrics = [
            MetricResult(metric_name="on_hand_units", display_name="On Hand Units", value=150, unit="units"),
            MetricResult(metric_name="reserved_units", display_name="Reserved Units", value=5, unit="units"),
            MetricResult(metric_name="inventory_valuation", display_name="Inventory Valuation", value=5500.0, unit="USD", currency="USD"),
        ]
        q_resp = QueryResponse(
            query_id="Q-INV-SUM",
            tool_name="get_inventory_summary",
            domain="inventory",
            metadata=meta,
            metrics=metrics,
        )
        step = CopilotStepResult(
            step_id="S-INV-SUM",
            tool_name="get_inventory_summary",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-INV-SUM",
            request_id="REQ-INV-SUM",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-INV-SUM",
            request_id="REQ-INV-SUM",
            state=CopilotState.COMPLETED,
            normalized_question="Show inventory summary",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.key_findings) == 3

    def test_stockout_risk_findings_content(self):
        meta = QueryMetadata(
            query_id="Q-STK",
            tool_name="get_stockout_risk",
            domain="inventory",
            generated_as_of="2026-06-30",
            source_engine="commerce_ai.inventory",
        )
        table = TableResult(
            columns=["sku_id", "warehouse_id", "stockout_risk_tier"],
            rows=[
                {"sku_id": "SKU_01", "warehouse_id": "WH_02", "stockout_risk_tier": "CRITICAL_STOCKOUT"},
            ],
            total_rows=1,
            metadata=meta,
        )
        q_resp = QueryResponse(
            query_id="Q-STK",
            tool_name="get_stockout_risk",
            domain="inventory",
            metadata=meta,
            table=table,
        )
        step = CopilotStepResult(
            step_id="S-STK",
            tool_name="get_stockout_risk",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-STK",
            request_id="REQ-STK",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-STK",
            request_id="REQ-STK",
            state=CopilotState.COMPLETED,
            normalized_question="Show stockout risk",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.supporting_evidence) >= 1
        assert any("CRITICAL_STOCKOUT" in ev.notes for ev in exp.supporting_evidence)
