"""Unit tests for Phase 7D Business Explanation Schemas and Data Models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from commerce_ai.explanations.enums import (
    ClaimCategory,
    EvidenceType,
    ExplanationConfidence,
    ExplanationStatus,
    ExplanationType,
    ProvenanceType,
)
from commerce_ai.explanations.schemas import (
    AuditResult,
    BusinessExplanation,
    ExplanationEvidence,
    ExplanationGovernance,
    Finding,
)


class TestExplanationSchemas:
    def test_all_15_explanation_types_exist(self):
        """Verify all 15 required explanation types are present in ExplanationType enum."""
        required_types = [
            "KPI_EXPLANATION",
            "TREND_EXPLANATION",
            "BREAKDOWN_EXPLANATION",
            "COMPARISON_EXPLANATION",
            "WHY_EXPLANATION",
            "MULTI_DOMAIN_EXPLANATION",
            "RISK_EXPLANATION",
            "FORECAST_EXPLANATION",
            "RETURN_EXPLANATION",
            "INVENTORY_EXPLANATION",
            "FINANCIAL_EXPLANATION",
            "OPERATIONAL_REVIEW_EXPLANATION",
            "DATA_QUALITY_EXPLANATION",
            "INSUFFICIENT_DATA_EXPLANATION",
            "UNAVAILABLE_EXPLANATION",
        ]
        enum_values = {e.value for e in ExplanationType}
        for rt in required_types:
            assert rt in enum_values, f"Missing required explanation type: {rt}"

    def test_provenance_types_exist(self):
        """Verify all required provenance tiers exist."""
        required_prov = [
            "USER_PROVIDED",
            "PHASE_7A_OBSERVED",
            "PHASE_7B_DERIVED",
            "PHASE_7C_ORCHESTRATED",
            "MODEL_BASED",
            "INSUFFICIENT_DATA",
        ]
        enum_values = {p.value for p in ProvenanceType}
        for rp in required_prov:
            assert rp in enum_values

    def test_confidence_tiers_exist(self):
        """Verify all 4 deterministic confidence tiers exist."""
        assert ExplanationConfidence.HIGH.value == "HIGH"
        assert ExplanationConfidence.MEDIUM.value == "MEDIUM"
        assert ExplanationConfidence.LOW.value == "LOW"
        assert ExplanationConfidence.INSUFFICIENT.value == "INSUFFICIENT"

    def test_explanation_evidence_immutability(self):
        """ExplanationEvidence should be immutable (frozen=True)."""
        ev = ExplanationEvidence(
            evidence_id="EVI-1",
            source_step_id="STEP-1",
            source_tool="get_sales_summary",
            source_engine="commerce_ai.sales",
            metric="gross_revenue",
            value=100000.0,
            currency="USD",
        )
        assert ev.evidence_id == "EVI-1"
        with pytest.raises(ValidationError):
            ev.value = 200000.0  # type: ignore

    def test_finding_immutability(self):
        """Finding should be immutable (frozen=True)."""
        finding = Finding(
            finding_id="FND-1",
            statement="Revenue was $100,000.",
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
            confidence=ExplanationConfidence.HIGH,
        )
        assert finding.finding_id == "FND-1"
        with pytest.raises(ValidationError):
            finding.statement = "Modified statement."  # type: ignore

    def test_governance_invariants_defaults(self):
        """ExplanationGovernance should enforce strict read-only and no-ranking defaults."""
        gov = ExplanationGovernance()
        assert gov.read_only is True
        assert gov.action_execution is False
        assert gov.execution_allowed is False
        assert gov.approval_required is False
        assert gov.no_ranking_enforced is True
        assert gov.no_decision_selection_enforced is True
        assert gov.no_causality_invented is True
        assert gov.no_recommendation_generated is True

    def test_business_explanation_minimal(self):
        """BusinessExplanation should instantiate with valid required fields."""
        exp = BusinessExplanation(
            explanation_id="EXP-001",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.KPI_EXPLANATION,
            headline="Revenue performance is stable.",
            summary="Total revenue reached $100,000 for the period.",
        )
        assert exp.explanation_id == "EXP-001"
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.KPI_EXPLANATION
        assert exp.key_findings == []
        assert exp.governance.read_only is True

    def test_audit_result_model(self):
        """AuditResult should correctly record validation checks."""
        res = AuditResult(
            is_valid=True,
            passed_checks=["CHECK_GOVERNANCE", "CHECK_EVIDENCE"],
        )
        assert res.is_valid is True
        assert len(res.passed_checks) == 2
        assert len(res.violations) == 0
