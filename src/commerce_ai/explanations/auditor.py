"""Semantic Claim Auditor for Business Explanations (Phase 7D).

Rigorously verifies every finding and claim in a BusinessExplanation against:
1. Evidence grounding: claim must link to existing evidence.
2. Value consistency: referenced numbers/metrics must exist in evidence.
3. Provenance validation: pedigree must match the nature of the data.
4. Causality prohibition: claims of causation are rejected unless causal models exist.
5. Forecast honesty: forecasts must be explicitly identified as models, never facts.
6. Ranking prohibition: rankings and winners are rejected.
7. Recommendation/Action prohibition: newly invented actions are rejected.
8. Silence prohibition: missing data must not be silently fabricated.
"""

from __future__ import annotations

import re
from typing import List

from commerce_ai.explanations.enums import (
    EvidenceType,
    ExplanationStatus,
    ProvenanceType,
)
from commerce_ai.explanations.governance import (
    _CAUSAL_PATTERNS,
    _RANKING_PATTERNS,
    validate_explanation_governance,
)
from commerce_ai.explanations.schemas import AuditResult, BusinessExplanation, Finding


class ClaimAuditor:
    """Audits business explanations for factual grounding, lineage, and governance compliance."""

    def audit(self, explanation: BusinessExplanation) -> AuditResult:
        """Perform comprehensive semantic claim audit on a BusinessExplanation."""
        passed_checks: List[str] = []
        violations: List[str] = []
        unsupported_claims: List[str] = []
        causality_warnings: List[str] = []
        ranking_violations: List[str] = []
        forecast_as_fact_warnings: List[str] = []

        # 1. Base Governance Audit
        gov_valid, gov_violations = validate_explanation_governance(explanation)
        if gov_valid:
            passed_checks.append("CHECK_GOVERNANCE_INVARIANTS")
        else:
            violations.extend(gov_violations)

        # 2. Check each finding
        if explanation.status == ExplanationStatus.INSUFFICIENT_DATA:
            passed_checks.append("CHECK_INSUFFICIENT_DATA_STATUS_HONEST")
            return AuditResult(
                is_valid=len(violations) == 0,
                passed_checks=passed_checks,
                violations=violations,
                unsupported_claims=unsupported_claims,
                causality_warnings=causality_warnings,
                ranking_violations=ranking_violations,
                forecast_as_fact_warnings=forecast_as_fact_warnings,
            )

        if explanation.status == ExplanationStatus.UNAVAILABLE:
            passed_checks.append("CHECK_UNAVAILABLE_STATUS_HONEST")
            return AuditResult(
                is_valid=len(violations) == 0,
                passed_checks=passed_checks,
                violations=violations,
                unsupported_claims=unsupported_claims,
                causality_warnings=causality_warnings,
                ranking_violations=ranking_violations,
                forecast_as_fact_warnings=forecast_as_fact_warnings,
            )

        all_evidence_ids = {e.evidence_id for e in explanation.supporting_evidence}

        for finding in explanation.key_findings:
            f_id = finding.finding_id
            stmt = finding.statement

            # A. Evidence presence check
            if not finding.evidence and finding.provenance != ProvenanceType.USER_PROVIDED:
                unsupported_claims.append(f"Finding '{f_id}' has no linked supporting evidence.")
                violations.append(f"Ungrounded claim: finding '{f_id}' lacks evidence lineage.")

            # B. Verify linked evidence exists in supporting_evidence
            for ev in finding.evidence:
                if ev.evidence_id not in all_evidence_ids:
                    unsupported_claims.append(
                        f"Finding '{f_id}' links to evidence '{ev.evidence_id}' not found in explanation evidence chain."
                    )

            # C. Check for ranking phrasing
            for pat in _RANKING_PATTERNS:
                if pat.search(stmt):
                    ranking_violations.append(f"Finding '{f_id}' contains ranking phrasing: '{stmt}'")
                    violations.append(f"Ranking prohibition violation in finding '{f_id}'.")

            # D. Check for ungrounded causality phrasing
            for pat in _CAUSAL_PATTERNS:
                if pat.search(stmt):
                    causality_warnings.append(f"Finding '{f_id}' makes ungrounded causal claim: '{stmt}'")
                    violations.append(f"Ungrounded causality violation in finding '{f_id}'.")

            # E. Forecast presented as fact check
            is_forecast_evidence = any(
                ev.provenance == ProvenanceType.MODEL_BASED or ev.evidence_type == EvidenceType.MODEL_PROJECTED
                for ev in finding.evidence
            )
            if is_forecast_evidence:
                # Statement must qualify that it is a projection/estimate/model
                has_qualifier = any(
                    w in stmt.lower()
                    for w in ["forecast", "project", "estimate", "model", "expected", "anticipate"]
                )
                if not has_qualifier:
                    forecast_as_fact_warnings.append(
                        f"Finding '{f_id}' presents model forecast as definitive historical fact: '{stmt}'"
                    )
                    violations.append(f"Forecast honesty violation in finding '{f_id}'.")

            # F. Check value consistency
            if finding.evidence:
                # Verify that any explicit numbers in the statement appear in the evidence
                # Strip ISO dates/periods (e.g. 2026-07-01, 2026-07) so date components aren't treated as metric values
                stmt_clean = re.sub(r"\b\d{4}(?:-\d{2})?(?:-\d{2})?\b", "", stmt)
                # Also strip entity IDs with numbers like SKU_01, WH_01
                stmt_clean = re.sub(r"\b[A-Za-z]+_\d+\b", "", stmt_clean)
                raw_numbers = re.findall(r"-?\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|-?\b\d+(?:\.\d+)?\b", stmt_clean)
                numbers_in_stmt = [n.replace(",", "") for n in raw_numbers if n.strip("-")]
                evidence_values_str = " ".join(
                    str(ev.value) + " " + str(ev.comparison_value or "") for ev in finding.evidence
                )
                for num_str in numbers_in_stmt:
                    if num_str not in evidence_values_str:
                        try:
                            val_float = float(num_str)
                            ev_floats = []
                            for ev in finding.evidence:
                                if ev.value is not None:
                                    try:
                                        ev_floats.append(float(ev.value))
                                    except (ValueError, TypeError):
                                        pass
                                if ev.comparison_value is not None:
                                    try:
                                        ev_floats.append(float(ev.comparison_value))
                                    except (ValueError, TypeError):
                                        pass
                            if not any(abs(val_float - ef) < 0.06 or abs(abs(val_float) - abs(ef)) < 0.06 for ef in ev_floats):
                                unsupported_claims.append(f"Finding '{f_id}' contains ungrounded numeric value '{num_str}'.")
                                violations.append(f"Value mismatch in finding '{f_id}'.")
                        except ValueError:
                            pass

        if not unsupported_claims:
            passed_checks.append("CHECK_ALL_CLAIMS_GROUNDED_IN_EVIDENCE")
        if not causality_warnings:
            passed_checks.append("CHECK_ZERO_UNGROUNDED_CAUSALITY")
        if not ranking_violations:
            passed_checks.append("CHECK_ZERO_RANKINGS_OR_LEADERBOARDS")
        if not forecast_as_fact_warnings:
            passed_checks.append("CHECK_FORECAST_HONESTY_AS_MODEL")

        is_valid = len(violations) == 0 and len(unsupported_claims) == 0

        return AuditResult(
            is_valid=is_valid,
            passed_checks=passed_checks,
            violations=violations,
            unsupported_claims=unsupported_claims,
            causality_warnings=causality_warnings,
            ranking_violations=ranking_violations,
            forecast_as_fact_warnings=forecast_as_fact_warnings,
        )


default_auditor = ClaimAuditor()
