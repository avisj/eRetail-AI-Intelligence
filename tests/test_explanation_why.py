"""Unit tests for Diagnostic Why Explanations (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.service import CopilotService
from commerce_ai.explanations.enums import (
    ExplanationConfidence,
    ExplanationStatus,
    ExplanationType,
    ProvenanceType,
)
from commerce_ai.explanations.service import ExplanationService


from commerce_ai.query_layer.service import QueryLayerService


@pytest.fixture
def copilot_service() -> CopilotService:
    ql = QueryLayerService.from_sample_data()
    return CopilotService(query_layer=ql)


@pytest.fixture
def explanation_service() -> ExplanationService:
    return ExplanationService()


class TestWhyExplanations:
    def test_why_margin_down_explanation(self, copilot_service, explanation_service):
        """1. 'Why did margin decline?' combines 4 upstream branches with causal honesty."""
        copilot_resp = copilot_service.process(
            "Why did margin decline?",
            as_of_date="2026-06-30",
            currency="USD",
        )
        assert copilot_resp.execution_result is not None
        assert len(copilot_resp.execution_result.step_results) == 4

        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.WHY_EXPLANATION
        assert "margin" in exp.headline.lower()

        # Check that evidence from all 4 branches is represented
        tools_in_evidence = {e.source_tool for e in exp.supporting_evidence}
        assert "get_margin_summary" in tools_in_evidence
        assert "get_margin_drivers" in tools_in_evidence
        assert "get_sales_by_channel" in tools_in_evidence
        assert "get_sales_by_warehouse" in tools_in_evidence

        # Check absence of ungrounded causal claims
        full_text = exp.headline + " " + exp.summary + " " + " ".join(f.statement for f in exp.key_findings)
        assert "caused by" not in full_text.lower()
        assert "directly caused" not in full_text.lower()
        assert "the root cause is" not in full_text.lower()

        # Check that causal and financial limitations are attached
        assert any("causal" in lim.lower() for lim in exp.limitations)
        assert any("fifo/lifo" in lim.lower() for lim in exp.limitations)

    def test_why_inventory_risk_high_explanation(self, copilot_service, explanation_service):
        """2. 'Why is inventory risk high?' combines 4 inventory tools without causal leap."""
        copilot_resp = copilot_service.process(
            "Why is inventory risk high?",
            as_of_date="2026-06-30",
        )
        assert copilot_resp.execution_result is not None
        assert len(copilot_resp.execution_result.step_results) == 4

        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.WHY_EXPLANATION

        tools_in_evidence = {e.source_tool for e in exp.supporting_evidence}
        assert "get_inventory_summary" in tools_in_evidence
        assert "get_inventory_risk" in tools_in_evidence
        assert "get_slow_moving_inventory" in tools_in_evidence
        assert "get_stockout_risk" in tools_in_evidence

        # Check risks are identified
        assert len(exp.risks) >= 2

        # Check absence of ungrounded causal claims
        full_text = exp.headline + " " + exp.summary + " " + " ".join(f.statement for f in exp.key_findings)
        assert "caused" not in full_text.lower()

    def test_why_returns_increasing_explanation(self, copilot_service, explanation_service):
        """3. 'Why are returns increasing?' combines 4 returns tools."""
        copilot_resp = copilot_service.process(
            "Why are returns increasing?",
            as_of_date="2026-06-30",
        )
        assert copilot_resp.execution_result is not None
        assert len(copilot_resp.execution_result.step_results) == 4

        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.WHY_EXPLANATION

        tools_in_evidence = {e.source_tool for e in exp.supporting_evidence}
        assert "get_return_summary" in tools_in_evidence
        assert "get_return_trend" in tools_in_evidence
        assert "get_return_anomalies" in tools_in_evidence
        assert "get_return_reason_breakdown" in tools_in_evidence

        # Check absence of ungrounded causal claims
        full_text = exp.headline + " " + exp.summary + " " + " ".join(f.statement for f in exp.key_findings)
        assert "caused by" not in full_text.lower()
