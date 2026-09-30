"""Decision Intelligence and Action Planning Service (Phase 6H).

Transforms deterministic Phase 6G business recommendations into structured, auditable
decision support packages for human review and action planning.

Guarantees:
- Strict Human-in-the-Loop governance: approval_required=True, execution_allowed=False
- NEVER declares a winner or selects an optimal option (selected_option=None)
- Deterministic SHA-256 decision ID generation preventing duplicates
- Comprehensive multi-dimensional trade-off evaluations using non-causal language
- Information completeness tracking without data fabrication
- Point-in-time safety (no future-data contamination)
- Single currency isolation with automatic multi-currency risk flagging
- No LLMs, no agents, no RAG, no autonomous execution
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from commerce_ai.decision_intelligence.options import DecisionOptionGenerator
from commerce_ai.decision_intelligence.risks import DecisionRiskEvaluator
from commerce_ai.decision_intelligence.schemas import (
    DecisionBusinessContext,
    DecisionEntityContext,
    DecisionIntelligenceConfig,
    DecisionIntelligenceResult,
    DecisionOption,
    DecisionPackage,
    DecisionStatus,
    DecisionTraceability,
    InformationAvailability,
    RiskSeverity,
    generate_decision_id,
)
from commerce_ai.decision_intelligence.templates import (
    format_decision_summary,
    format_decision_title,
)
from commerce_ai.decision_intelligence.tradeoffs import DecisionTradeOffEvaluator
from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendation,
    ConfidenceProvenance,
    RecommendationConflict,
    RecommendationEvidence,
    RecommendationPriority,
    RecommendationType,
)


def _parse_date(val: Union[str, date, datetime, None]) -> Optional[date]:
    """Parse date, datetime, or ISO string safely to a date object."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        try:
            return date.fromisoformat(val[:10])
        except Exception:
            return None
    return None


class DecisionIntelligenceService:
    """Orchestrates the conversion of recommendations into structured decision packages."""

    def __init__(self, config: Optional[DecisionIntelligenceConfig] = None) -> None:
        self.config = config or DecisionIntelligenceConfig()
        self.option_generator = DecisionOptionGenerator()
        self.tradeoff_evaluator = DecisionTradeOffEvaluator()
        self.risk_evaluator = DecisionRiskEvaluator()

    def package_single_recommendation(
        self,
        rec: BusinessRecommendation,
        conflicts: Optional[List[RecommendationConflict]] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> DecisionPackage:
        """Transform an individual business recommendation into an auditable decision package."""
        effective_date = _parse_date(as_of_date) or rec.created_as_of

        # Point-in-time safety check (Section 27)
        if rec.created_as_of > effective_date:
            raise ValueError(
                f"Future-data contamination: recommendation {rec.recommendation_id} "
                f"created_as_of ({rec.created_as_of}) is later than as_of_date ({effective_date})."
            )

        conflicts_list = conflicts or []
        decision_id = generate_decision_id(
            recommendation_id=rec.recommendation_id,
            decision_type=rec.recommendation_type,
            as_of_date=effective_date,
            rule_version=rec.rule_version,
        )

        # Entity Context
        entity_key = f"{rec.sku_id or 'GLOBAL'}:{rec.warehouse_id or rec.channel_id or 'ALL'}"
        entity_ctx = DecisionEntityContext(
            sku_id=rec.sku_id,
            warehouse_id=rec.warehouse_id,
            channel_id=rec.channel_id,
            category_id=rec.category_id,
            brand=rec.brand,
            entity_key=entity_key,
        )

        # Business Context
        biz_ctx = DecisionBusinessContext(
            priority=rec.priority,
            severity=rec.severity,
            source_engine=rec.source_engine,
            rule_id=rec.rule_id,
            rule_version=rec.rule_version,
            confidence=rec.confidence,
            confidence_reason=rec.confidence_reason,
            currency=rec.currency,
        )

        # Expiration Check (Section 25)
        status = DecisionStatus.PENDING_REVIEW
        if self.config.auto_expire_decisions and effective_date > rec.valid_until:
            status = DecisionStatus.EXPIRED

        # Explainability & Titles (Section 24)
        title = format_decision_title(rec)
        summary = format_decision_summary(rec)

        # Options & Trade-offs (Sections 6-17)
        options = self.option_generator.generate_options(decision_id, rec)
        tradeoffs = self.tradeoff_evaluator.evaluate_tradeoffs(decision_id, options, rec)

        # Required Information & Risks (Sections 18-20, 26)
        required_info = self.risk_evaluator.evaluate_required_information(decision_id, rec)
        risk_flags = self.risk_evaluator.evaluate_risks(decision_id, rec, conflicts_list, self.config)

        # Traceability (Section 23)
        traceability = DecisionTraceability(
            decision_id=decision_id,
            recommendation_id=rec.recommendation_id,
            supporting_signal_ids=list(rec.supporting_signal_ids),
            supporting_impact_ids=list(rec.supporting_impact_ids),
            domain_metrics=rec.traceability.domain_metrics if rec.traceability else [],
            source_datasets=rec.traceability.source_datasets if rec.traceability else [],
        )

        # Associated conflicts
        rec_conflicts = [
            c for c in conflicts_list if rec.recommendation_id in c.conflicting_recommendation_ids
        ]

        # Multi-currency detection requires human review
        currencies_found = {rec.currency}
        if rec.financial_context and rec.financial_context.currency:
            currencies_found.add(rec.financial_context.currency)
        for e in rec.evidence:
            if e.currency:
                currencies_found.add(e.currency)
        has_multi_currency = len(currencies_found) > 1

        return DecisionPackage(
            decision_id=decision_id,
            recommendation_id=rec.recommendation_id,
            recommendation_type=rec.recommendation_type,
            decision_status=status,
            decision_title=title,
            decision_summary=summary,
            entity_context=entity_ctx,
            business_context=biz_ctx,
            financial_context=rec.financial_context,
            operational_context=rec.operational_context,
            evidence=rec.evidence,
            decision_options=options,
            trade_offs=tradeoffs,
            required_information=required_info,
            risk_flags=risk_flags,
            conflicts=rec_conflicts,
            approval_required=True,
            execution_allowed=False,
            selected_option=None,  # NEVER automatically select a winner!
            requires_human_review=True or has_multi_currency or bool(rec_conflicts),
            created_as_of=effective_date,
            valid_until=rec.valid_until,
            rule_version=rec.rule_version,
            traceability=traceability,
        )

    def package_recommendations(
        self,
        recommendations: List[BusinessRecommendation],
        conflicts: Optional[List[RecommendationConflict]] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> DecisionIntelligenceResult:
        """Package a portfolio of recommendations into structured decision support packages."""
        if not recommendations:
            eff_date = _parse_date(as_of_date) or date.today()
            return DecisionIntelligenceResult(
                decision_packages=[],
                total_decisions=0,
                decisions_by_type={},
                decisions_by_status={},
                decisions_by_risk_severity={},
                decisions_by_option_count={},
                human_review_count=0,
                conflict_count=0,
                insufficient_information_count=0,
                decisions_by_confidence={},
                as_of_date=eff_date,
            )

        effective_date = _parse_date(as_of_date) or recommendations[0].created_as_of
        conflicts_list = conflicts or []

        packages: List[DecisionPackage] = []
        seen_decision_ids: Set[str] = set()

        decisions_by_type: Dict[str, int] = {}
        decisions_by_status: Dict[str, int] = {}
        decisions_by_risk_severity: Dict[str, int] = {
            RiskSeverity.CRITICAL.value: 0,
            RiskSeverity.HIGH.value: 0,
            RiskSeverity.MEDIUM.value: 0,
            RiskSeverity.LOW.value: 0,
        }
        decisions_by_option_count: Dict[int, int] = {}
        decisions_by_confidence: Dict[str, int] = {}
        human_review_count = 0
        conflict_decisions_count = 0
        insufficient_info_count = 0

        for rec in recommendations:
            pkg = self.package_single_recommendation(
                rec, conflicts=conflicts_list, as_of_date=effective_date
            )

            # Prevent duplicates deterministically (Section 28)
            if pkg.decision_id in seen_decision_ids:
                continue
            seen_decision_ids.add(pkg.decision_id)
            packages.append(pkg)

            # Aggregate statistics
            t_key = pkg.recommendation_type.value
            decisions_by_type[t_key] = decisions_by_type.get(t_key, 0) + 1

            s_key = pkg.decision_status.value
            decisions_by_status[s_key] = decisions_by_status.get(s_key, 0) + 1

            c_key = pkg.business_context.confidence.value
            decisions_by_confidence[c_key] = decisions_by_confidence.get(c_key, 0) + 1

            opt_cnt = len(pkg.decision_options)
            decisions_by_option_count[opt_cnt] = decisions_by_option_count.get(opt_cnt, 0) + 1

            if pkg.requires_human_review:
                human_review_count += 1

            if pkg.conflicts:
                conflict_decisions_count += 1

            has_unavailable_info = any(
                info.availability == InformationAvailability.UNAVAILABLE
                for info in pkg.required_information
            )
            if has_unavailable_info:
                insufficient_info_count += 1

            # Highest risk severity
            highest_sev = RiskSeverity.LOW
            for rf in pkg.risk_flags:
                if rf.severity == RiskSeverity.CRITICAL:
                    highest_sev = RiskSeverity.CRITICAL
                    break
                elif rf.severity == RiskSeverity.HIGH and highest_sev != RiskSeverity.CRITICAL:
                    highest_sev = RiskSeverity.HIGH
                elif rf.severity == RiskSeverity.MEDIUM and highest_sev not in (RiskSeverity.CRITICAL, RiskSeverity.HIGH):
                    highest_sev = RiskSeverity.MEDIUM
            decisions_by_risk_severity[highest_sev.value] = decisions_by_risk_severity.get(highest_sev.value, 0) + 1

        return DecisionIntelligenceResult(
            decision_packages=packages,
            total_decisions=len(packages),
            decisions_by_type=decisions_by_type,
            decisions_by_status=decisions_by_status,
            decisions_by_risk_severity=decisions_by_risk_severity,
            decisions_by_option_count=decisions_by_option_count,
            human_review_count=human_review_count,
            conflict_count=conflict_decisions_count,
            insufficient_information_count=insufficient_info_count,
            decisions_by_confidence=decisions_by_confidence,
            as_of_date=effective_date,
        )
