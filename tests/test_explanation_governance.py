"""Unit tests for Explanation Governance and Prohibition Enforcement (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.explanations.auditor import ClaimAuditor
from commerce_ai.explanations.enums import (
    EvidenceType,
    ExplanationConfidence,
    ExplanationStatus,
    ExplanationType,
    ProvenanceType,
)
from commerce_ai.explanations.governance import (
    check_text_for_prohibited_patterns,
    validate_explanation_governance,
)
from commerce_ai.explanations.schemas import (
    BusinessExplanation,
    ExplanationEvidence,
    ExplanationGovernance,
    Finding,
)


class TestExplanationGovernance:
    def test_ranking_prohibition_check(self):
        """Prohibit ranking keywords."""
        assert len(check_text_for_prohibited_patterns("This is the top 10 list.")) > 0
        assert len(check_text_for_prohibited_patterns("Warehouse A is the winner.")) > 0
        assert len(check_text_for_prohibited_patterns("Show leaderboard.")) > 0
        assert len(check_text_for_prohibited_patterns("This was the best performing channel.")) > 0
        assert len(check_text_for_prohibited_patterns("Standard observed metrics.")) == 0

    def test_causality_prohibition_check(self):
        """Prohibit ungrounded causal phrasing."""
        assert len(check_text_for_prohibited_patterns("The decline was caused by warehouse A.")) > 0
        assert len(check_text_for_prohibited_patterns("Channel B directly caused the drop.")) > 0
        assert len(check_text_for_prohibited_patterns("The root cause is increased returns.")) > 0
        # Allowed co-occurrence phrasing
        assert len(check_text_for_prohibited_patterns("The decline coincided with channel shifts.")) == 0
        assert len(check_text_for_prohibited_patterns("Channel B accounted for a larger share.")) == 0

    def test_action_prohibition_check(self):
        """Prohibit unauthorized action generation."""
        assert len(check_text_for_prohibited_patterns("You should order 500 units.")) > 0
        assert len(check_text_for_prohibited_patterns("Place a purchase order now.")) > 0
        assert len(check_text_for_prohibited_patterns("Execute a transfer to WH_02.")) > 0

    def test_sensationalism_prohibition_check(self):
        """Prohibit sensational crisis language."""
        assert len(check_text_for_prohibited_patterns("This is a critical disaster.")) > 0
        assert len(check_text_for_prohibited_patterns("A massive failure occurred.")) > 0

    def test_auditor_rejects_ranking_finding(self):
        """ClaimAuditor must reject explanation with ranking finding."""
        auditor = ClaimAuditor()
        ev = ExplanationEvidence(
            evidence_id="EVI-1",
            source_step_id="STEP-1",
            source_tool="get_sales_by_channel",
            source_engine="commerce_ai.sales",
            metric="revenue",
            value=50000.0,
        )
        finding = Finding(
            finding_id="FND-1",
            statement="Channel Online is the winner and best performing.",
            evidence=[ev],
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
            confidence=ExplanationConfidence.HIGH,
        )
        exp = BusinessExplanation(
            explanation_id="EXP-TEST",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.BREAKDOWN_EXPLANATION,
            headline="Breakdown evaluation.",
            summary="Channel breakdown summary.",
            key_findings=[finding],
            supporting_evidence=[ev],
        )

        res = auditor.audit(exp)
        assert res.is_valid is False
        assert len(res.ranking_violations) > 0

    def test_auditor_rejects_causal_finding(self):
        """ClaimAuditor must reject explanation with ungrounded causal claim."""
        auditor = ClaimAuditor()
        ev = ExplanationEvidence(
            evidence_id="EVI-2",
            source_step_id="STEP-1",
            source_tool="get_margin_summary",
            source_engine="commerce_ai.financial",
            metric="gross_margin",
            value=45.0,
        )
        finding = Finding(
            finding_id="FND-2",
            statement="Margin decline was caused by freight cost.",
            evidence=[ev],
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
            confidence=ExplanationConfidence.HIGH,
        )
        exp = BusinessExplanation(
            explanation_id="EXP-TEST2",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.WHY_EXPLANATION,
            headline="Margin inquiry.",
            summary="Margin evaluation summary.",
            key_findings=[finding],
            supporting_evidence=[ev],
        )

        res = auditor.audit(exp)
        assert res.is_valid is False
        assert len(res.causality_warnings) > 0

    def test_auditor_rejects_forecast_as_fact(self):
        """ClaimAuditor must reject forecast finding without model qualification."""
        auditor = ClaimAuditor()
        ev = ExplanationEvidence(
            evidence_id="EVI-3",
            source_step_id="STEP-1",
            source_tool="get_forecast",
            source_engine="commerce_ai.forecasting",
            metric="future_demand",
            value=1000.0,
            provenance=ProvenanceType.MODEL_BASED,
            evidence_type=EvidenceType.MODEL_PROJECTED,
        )
        finding = Finding(
            finding_id="FND-3",
            statement="Demand will definitely be 1000 units.",
            evidence=[ev],
            provenance=ProvenanceType.MODEL_BASED,
            evidence_type=EvidenceType.MODEL_PROJECTED,
            confidence=ExplanationConfidence.HIGH,
        )
        exp = BusinessExplanation(
            explanation_id="EXP-TEST3",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.FORECAST_EXPLANATION,
            headline="Demand forecast.",
            summary="Demand forecast summary.",
            key_findings=[finding],
            supporting_evidence=[ev],
        )

        res = auditor.audit(exp)
        assert res.is_valid is False
        assert len(res.forecast_as_fact_warnings) > 0

    def test_governance_validator_rejects_action_execution_enabled(self):
        gov = ExplanationGovernance(action_execution=True)
        exp = BusinessExplanation(
            explanation_id="EXP-GOV-ERR",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.KPI_EXPLANATION,
            headline="Headline",
            summary="Summary",
            governance=gov,
        )
        is_valid, violations = validate_explanation_governance(exp)
        assert not is_valid
        assert len(violations) > 0
        assert any("action_execution" in v for v in violations)

    def test_governance_validator_rejects_missing_no_ranking_enforcement(self):
        gov = ExplanationGovernance(no_ranking_enforced=False)
        exp = BusinessExplanation(
            explanation_id="EXP-GOV-ERR2",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.KPI_EXPLANATION,
            headline="Headline",
            summary="Summary",
            governance=gov,
        )
        is_valid, violations = validate_explanation_governance(exp)
        assert not is_valid
        assert len(violations) > 0
        assert any("no_ranking_enforced" in v for v in violations)

    def test_auditor_rejects_purchase_order_action_phrasing_in_statement(self):
        auditor = ClaimAuditor()
        ev = ExplanationEvidence(
            evidence_id="EVI-ACT",
            source_step_id="STEP-ACT",
            source_tool="get_sales_summary",
            source_engine="commerce_ai.sales",
            metric="gross_revenue",
            value=500.0,
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
        )
        finding = Finding(
            finding_id="FND-ACT",
            statement="You should order 500 units immediately.",
            evidence=[ev],
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
            confidence=ExplanationConfidence.HIGH,
        )
        exp = BusinessExplanation(
            explanation_id="EXP-ACT",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.KPI_EXPLANATION,
            headline="Action test",
            summary="Summary",
            key_findings=[finding],
            supporting_evidence=[ev],
        )
        res = auditor.audit(exp)
        assert res.is_valid is False
        assert any("Action generation" in v for v in res.violations)

    def test_auditor_passes_clean_evidence_grounded_explanation(self):
        auditor = ClaimAuditor()
        ev = ExplanationEvidence(
            evidence_id="EVI-CLEAN",
            source_step_id="STEP-CLEAN",
            source_tool="get_sales_summary",
            source_engine="commerce_ai.sales",
            metric="gross_revenue",
            value=250000.0,
            currency="USD",
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
        )
        finding = Finding(
            finding_id="FND-CLEAN",
            statement="Gross revenue was $250,000.00.",
            evidence=[ev],
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
            confidence=ExplanationConfidence.HIGH,
        )
        exp = BusinessExplanation(
            explanation_id="EXP-CLEAN",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.KPI_EXPLANATION,
            headline="Commercial revenue summary.",
            summary="Observed revenue metrics without subjective ordering.",
            key_findings=[finding],
            supporting_evidence=[ev],
        )
        res = auditor.audit(exp)
        assert res.is_valid is True
        assert len(res.violations) == 0

    def test_auditor_records_all_passed_checks(self):
        auditor = ClaimAuditor()
        ev = ExplanationEvidence(
            evidence_id="EVI-PASS",
            source_step_id="STEP-PASS",
            source_tool="get_margin_summary",
            source_engine="commerce_ai.financial",
            metric="gross_margin",
            value=45.0,
            unit="%",
            provenance=ProvenanceType.PHASE_7B_DERIVED,
            evidence_type=EvidenceType.DETERMINISTIC_DERIVED,
        )
        finding = Finding(
            finding_id="FND-PASS",
            statement="Gross margin was 45.0%.",
            evidence=[ev],
            provenance=ProvenanceType.PHASE_7B_DERIVED,
            evidence_type=EvidenceType.DETERMINISTIC_DERIVED,
            confidence=ExplanationConfidence.HIGH,
        )
        exp = BusinessExplanation(
            explanation_id="EXP-PASS",
            status=ExplanationStatus.SUCCESS,
            explanation_type=ExplanationType.KPI_EXPLANATION,
            headline="Margin evaluation.",
            summary="Margin evaluation summary.",
            key_findings=[finding],
            supporting_evidence=[ev],
        )
        res = auditor.audit(exp)
        assert "CHECK_GOVERNANCE_INVARIANTS" in res.passed_checks
        assert "CHECK_ALL_CLAIMS_GROUNDED_IN_EVIDENCE" in res.passed_checks
        assert "CHECK_ZERO_UNGROUNDED_CAUSALITY" in res.passed_checks
        assert "CHECK_ZERO_RANKINGS_OR_LEADERBOARDS" in res.passed_checks
        assert "CHECK_FORECAST_HONESTY_AS_MODEL" in res.passed_checks

