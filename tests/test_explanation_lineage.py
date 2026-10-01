"""Unit tests for Explanation Lineage and Auditability (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.service import CopilotService
from commerce_ai.explanations.auditor import ClaimAuditor
from commerce_ai.explanations.enums import (
    EvidenceType,
    ExplanationConfidence,
    ExplanationStatus,
    ExplanationType,
    ProvenanceType,
)
from commerce_ai.explanations.schemas import (
    BusinessExplanation,
    ExplanationEvidence,
    Finding,
)
from commerce_ai.explanations.service import ExplanationService
from commerce_ai.query_layer.registry import default_registry
from commerce_ai.query_layer.service import QueryLayerService


class TestExplanationLineage:
    def test_lineage_traceability_on_why_margin(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process("Why did margin decline?", as_of_date="2026-06-30", currency="USD")
        exp = explanation_service.explain(copilot_resp)

        all_registered_tools = set(default_registry.get_all_tool_names())

        # 1. Verify every finding has linked evidence
        for finding in exp.key_findings:
            assert len(finding.evidence) > 0, f"Finding '{finding.finding_id}' has no linked evidence"
            assert finding.source_step is not None
            assert finding.source_tool is not None
            assert finding.source_tool in all_registered_tools

        # 2. Verify every supporting evidence record carries full provenance
        for ev in exp.supporting_evidence:
            assert ev.evidence_id.startswith("EVI-")
            assert ev.source_step_id.startswith("STEP-")
            assert ev.source_tool in all_registered_tools
            assert ev.source_engine.startswith("commerce_ai.")

    def test_lineage_traceability_on_multi_domain(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process(
            "Show sales, margin, inventory and returns together",
            as_of_date="2026-06-30",
            currency="USD",
        )
        exp = explanation_service.explain(copilot_resp)

        all_registered_tools = set(default_registry.get_all_tool_names())

        for finding in exp.key_findings:
            assert len(finding.evidence) > 0
            for ev in finding.evidence:
                assert ev.source_tool in all_registered_tools
                assert ev.source_engine.startswith("commerce_ai.")

    def test_auditor_detects_orphan_findings(self):
        auditor = ClaimAuditor()
        orphan_finding = Finding(
            finding_id="FND-ORPHAN",
            statement="Revenue grew 25% without evidence.",
            evidence=[],  # orphan!
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
            confidence=ExplanationConfidence.HIGH,
        )
        exp = BusinessExplanation(
            explanation_id="EXP-ORPHAN",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.KPI_EXPLANATION,
            headline="Revenue report.",
            summary="Revenue summary.",
            key_findings=[orphan_finding],
            supporting_evidence=[],
        )

        res = auditor.audit(exp)
        assert res.is_valid is False
        assert len(res.unsupported_claims) > 0

    def test_auditor_detects_missing_evidence_chain_id(self):
        auditor = ClaimAuditor()
        ev = ExplanationEvidence(
            evidence_id="EVI-A",
            source_step_id="S-1",
            source_tool="get_sales_summary",
            source_engine="commerce_ai.sales",
            metric="gross_revenue",
            value=100.0,
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
        )
        finding = Finding(
            finding_id="FND-BROKEN",
            statement="Gross revenue was $100.00.",
            evidence=[ev],  # ev is in finding, but omitted from supporting_evidence!
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
            confidence=ExplanationConfidence.HIGH,
        )
        exp = BusinessExplanation(
            explanation_id="EXP-BROKEN",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.KPI_EXPLANATION,
            headline="Headline",
            summary="Summary",
            key_findings=[finding],
            supporting_evidence=[],  # Broken chain!
        )
        res = auditor.audit(exp)
        assert res.is_valid is False
        assert any("not found in explanation evidence chain" in c for c in res.unsupported_claims)

    def test_auditor_detects_unsupported_numeric_value_in_statement(self):
        auditor = ClaimAuditor()
        ev = ExplanationEvidence(
            evidence_id="EVI-NUM",
            source_step_id="S-1",
            source_tool="get_sales_summary",
            source_engine="commerce_ai.sales",
            metric="gross_revenue",
            value=100.0,
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
        )
        finding = Finding(
            finding_id="FND-NUM-FAB",
            statement="Gross revenue was $999.00 and profit was 50.0.",  # 999.00 is not in evidence!
            evidence=[ev],
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
            confidence=ExplanationConfidence.HIGH,
        )
        exp = BusinessExplanation(
            explanation_id="EXP-NUM-FAB",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.KPI_EXPLANATION,
            headline="Headline",
            summary="Summary",
            key_findings=[finding],
            supporting_evidence=[ev],
        )
        res = auditor.audit(exp)
        assert res.is_valid is False
        assert any("numeric value" in c.lower() for c in res.unsupported_claims)

    def test_lineage_traceability_on_inventory_position(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process("What is our inventory position?", as_of_date="2026-06-30")
        exp = explanation_service.explain(copilot_resp)

        assert len(exp.supporting_evidence) > 0
        for ev in exp.supporting_evidence:
            assert ev.source_tool == "get_inventory_position"
            assert ev.provenance in (ProvenanceType.PHASE_7A_OBSERVED, ProvenanceType.PHASE_7B_DERIVED)
