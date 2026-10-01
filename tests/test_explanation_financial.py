"""Unit tests for Financial Explanations and Limitations (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import (
    CopilotExecutionResult,
    CopilotResponse,
    CopilotStepResult,
)
from commerce_ai.copilot.service import CopilotService
from commerce_ai.explanations.enums import ExplanationStatus, ExplanationType, ProvenanceType
from commerce_ai.explanations.service import ExplanationService
from commerce_ai.query_layer.schemas import (
    BreakdownItem,
    BreakdownResult,
    CalculationStatus,
    MetricResult,
    QueryMetadata,
    QueryResponse,
    TableResult,
)
from commerce_ai.query_layer.service import QueryLayerService


class TestFinancialExplanations:
    def test_gross_margin_explanation_attaches_financial_limitations(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process(
            "What is our gross margin?",
            as_of_date="2026-06-30",
            currency="USD",
        )
        assert copilot_resp.execution_result is not None

        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.limitations) >= 2

        # Verify specific project financial limitations are preserved
        assert any("fifo/lifo" in lim.lower() for lim in exp.limitations)
        assert any("procurement cost" in lim.lower() for lim in exp.limitations)

    def test_revenue_explanation(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process(
            "What is our gross revenue?",
            as_of_date="2026-06-30",
            currency="USD",
        )
        assert copilot_resp.execution_result is not None

        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert any("revenue" in f.statement.lower() for f in exp.key_findings)

    def test_unit_economics_explanation(self):
        meta = QueryMetadata(
            query_id="Q-UE",
            tool_name="get_unit_economics",
            domain="financial",
            generated_as_of="2026-06-30",
            currency="USD",
            source_engine="commerce_ai.financial",
        )
        metrics = [
            MetricResult(
                metric_name="average_selling_price",
                display_name="Average Selling Price",
                value=45.5,
                unit="USD",
                currency="USD",
            ),
            MetricResult(
                metric_name="unit_cost",
                display_name="Average Unit Cost",
                value=25.0,
                unit="USD",
                currency="USD",
            ),
            MetricResult(
                metric_name="unit_margin",
                display_name="Unit Margin",
                value=20.5,
                unit="USD",
                currency="USD",
            ),
        ]
        q_resp = QueryResponse(
            query_id="Q-UE",
            tool_name="get_unit_economics",
            domain="financial",
            metadata=meta,
            metrics=metrics,
        )
        step = CopilotStepResult(
            step_id="S-UE",
            tool_name="get_unit_economics",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-UE",
            request_id="REQ-UE",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-UE",
            request_id="REQ-UE",
            state=CopilotState.COMPLETED,
            normalized_question="Show unit economics",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.key_findings) == 3
        assert any("procurement cost" in lim.lower() for lim in exp.limitations)

    def test_operational_economics_cost_completeness(self):
        meta = QueryMetadata(
            query_id="Q-OE",
            tool_name="get_operational_economics",
            domain="financial",
            generated_as_of="2026-06-30",
            currency="USD",
            source_engine="commerce_ai.financial",
        )
        metrics = [
            MetricResult(
                metric_name="total_operational_cost",
                display_name="Total Operational Cost",
                value=125000.0,
                unit="USD",
                currency="USD",
            ),
            MetricResult(
                metric_name="cost_completeness_score",
                display_name="Cost Completeness Score",
                value=0.85,
                unit="ratio",
            ),
        ]
        q_resp = QueryResponse(
            query_id="Q-OE",
            tool_name="get_operational_economics",
            domain="financial",
            metadata=meta,
            metrics=metrics,
        )
        step = CopilotStepResult(
            step_id="S-OE",
            tool_name="get_operational_economics",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-OE",
            request_id="REQ-OE",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-OE",
            request_id="REQ-OE",
            state=CopilotState.COMPLETED,
            normalized_question="Show operational economics",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert any("cost completeness" in f.statement.lower() for f in exp.key_findings)

    def test_profitability_attribution_table_explanation(self):
        meta = QueryMetadata(
            query_id="Q-PA",
            tool_name="get_profitability_attribution",
            domain="financial",
            generated_as_of="2026-06-30",
            source_engine="commerce_ai.financial",
        )
        table = TableResult(
            columns=["segment", "gross_profit", "margin_rate"],
            rows=[
                {"segment": "Direct", "gross_profit": 50000.0, "margin_rate": 0.45},
                {"segment": "Wholesale", "gross_profit": 20000.0, "margin_rate": 0.30},
            ],
            total_rows=2,
            metadata=meta,
        )
        q_resp = QueryResponse(
            query_id="Q-PA",
            tool_name="get_profitability_attribution",
            domain="financial",
            metadata=meta,
            table=table,
        )
        step = CopilotStepResult(
            step_id="S-PA",
            tool_name="get_profitability_attribution",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-PA",
            request_id="REQ-PA",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-PA",
            request_id="REQ-PA",
            state=CopilotState.COMPLETED,
            normalized_question="Show profitability attribution",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.key_findings) >= 2

    def test_financial_explanation_distinguishes_cogs_from_catalog_cost(self):
        meta = QueryMetadata(
            query_id="Q-COGS",
            tool_name="get_margin_summary",
            domain="financial",
            generated_as_of="2026-06-30",
            currency="USD",
            source_engine="commerce_ai.financial",
        )
        metrics = [
            MetricResult(
                metric_name="gross_margin_rate",
                display_name="Gross Margin Rate",
                value=42.1,
                unit="%",
            ),
        ]
        q_resp = QueryResponse(
            query_id="Q-COGS",
            tool_name="get_margin_summary",
            domain="financial",
            metadata=meta,
            metrics=metrics,
        )
        step = CopilotStepResult(
            step_id="S-COGS",
            tool_name="get_margin_summary",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-COGS",
            request_id="REQ-COGS",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-COGS",
            request_id="REQ-COGS",
            state=CopilotState.COMPLETED,
            normalized_question="What is gross margin?",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        full_text = exp.headline + " " + exp.summary + " " + " ".join(f.statement for f in exp.key_findings)
        # Ensure it does NOT conflate catalog unit cost with accounting GAAP COGS
        assert "gaap cogs" not in full_text.lower()
        assert any("fifo/lifo" in lim.lower() for lim in exp.limitations)

    def test_margin_drivers_breakdown_explanation(self):
        meta = QueryMetadata(
            query_id="Q-MD",
            tool_name="get_margin_drivers",
            domain="financial",
            generated_as_of="2026-06-30",
            currency="USD",
            source_engine="commerce_ai.financial",
        )
        bk = BreakdownResult(
            metric_name="margin_impact",
            display_name="Margin Drivers",
            dimension_name="driver",
            items=[
                BreakdownItem(dimension_name="driver", dimension_value="Discounting", metric_value=-15000.0, metric_name="margin_impact", currency="USD"),
                BreakdownItem(dimension_name="driver", dimension_value="Volume Shift", metric_value=25000.0, metric_name="margin_impact", currency="USD"),
            ],
            metadata=meta,
        )
        q_resp = QueryResponse(
            query_id="Q-MD",
            tool_name="get_margin_drivers",
            domain="financial",
            metadata=meta,
            breakdown=bk,
        )
        step = CopilotStepResult(
            step_id="S-MD",
            tool_name="get_margin_drivers",
            status=StepStatus.COMPLETED,
            response=q_resp,
        )
        exec_res = CopilotExecutionResult(
            result_id="RES-MD",
            request_id="REQ-MD",
            status=PlanExecutionStatus.COMPLETE,
            step_results=[step],
        )
        resp = CopilotResponse(
            response_id="RESP-MD",
            request_id="REQ-MD",
            state=CopilotState.COMPLETED,
            normalized_question="What drove margin?",
            execution_result=exec_res,
        )
        exp = ExplanationService().explain(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert len(exp.key_findings) == 2
