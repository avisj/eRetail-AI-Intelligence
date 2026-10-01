"""Unit tests for Multi-Domain Explanations (Phase 7D)."""

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


class TestMultiDomainExplanations:
    def test_multi_domain_synthesis(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process(
            "Show sales, margin, inventory and returns together",
            as_of_date="2026-06-30",
            currency="USD",
        )
        assert copilot_resp.execution_result is not None
        assert len(copilot_resp.execution_result.step_results) == 4

        exp = explanation_service.explain(copilot_resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.MULTI_DOMAIN_EXPLANATION
        assert exp.provenance == ProvenanceType.PHASE_7C_ORCHESTRATED

        # Verify domain representation across findings
        findings_text = " ".join(f.statement for f in exp.key_findings)
        assert "revenue" in findings_text.lower() or "sales" in findings_text.lower()
        assert "margin" in findings_text.lower()
        assert "inventory" in findings_text.lower()
        assert "return" in findings_text.lower()

        # Verify evidence tools
        tools_in_evidence = {e.source_tool for e in exp.supporting_evidence}
        assert "get_revenue_summary" in tools_in_evidence
        assert "get_margin_summary" in tools_in_evidence
        assert "get_inventory_summary" in tools_in_evidence
        assert "get_return_summary" in tools_in_evidence

        # Verify no ungrounded cross-domain causal claims
        assert "caused" not in exp.summary.lower()
