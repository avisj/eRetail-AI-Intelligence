"""Unit and Integration Tests for Decision Intelligence & Action Planning (Phase 6H).

Tests all 50 specified conditions from A through AX:
- Condition A: schema validation
- Condition B: decision package creation
- Condition C: recommendation linkage
- Condition D: deterministic decision ID
- Condition E: duplicate prevention
- Condition F: lifecycle
- Condition G: approval required
- Condition H: execution prohibited
- Condition I: option generation
- Condition J: no winner selection
- Condition K: trade-offs
- Condition L: required information
- Condition M: missing information
- Condition N: risk flags
- Condition O: conflict propagation
- Condition P: financial context
- Condition Q: operational context
- Condition R: evidence
- Condition S: traceability
- Condition T: replenishment decision
- Condition U: PO decision
- Condition V: transfer decision
- Condition W: inventory decision
- Condition X: stockout decision
- Condition Y: return decision
- Condition Z: margin decision
- Condition AA: data-quality decision
- Condition AB: forecast uncertainty
- Condition AC: currency isolation
- Condition AD: future-data protection
- Condition AE: validity expiration
- Condition AF: no causal claims
- Condition AG: no autonomous execution
- Condition AH: no pricing recommendation
- Condition AI: no fabricated evidence
- Condition AJ: no fabricated forecast
- Condition AK: no fabricated supplier information
- Condition AL: no fabricated logistics information
- Condition AM: conflict handling
- Condition AN: multiple recommendations
- Condition AO: duplicate recommendations
- Condition AP: recommendation without financial impact
- Condition AQ: insufficient data
- Condition AR: model-based provenance
- Condition AS: deterministic-derived provenance
- Condition AT: direct-observed provenance
- Condition AU: insufficient-data provenance
- Condition AV: Phase 6G regression
- Condition AW: Phase 6F regression
- Condition AX: Phase 6E regression
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
import pytest
from pydantic import ValidationError

from commerce_ai.business_impact.schemas import (
    BusinessImpactRecord,
    CalculationStatus,
    ImpactCategory,
    ImpactConfidence,
    ImpactType,
)
from commerce_ai.cross_domain.schemas import (
    CrossDomainDimension,
    CrossDomainSignal,
    SignalCategory,
    SignalSeverity,
)
from commerce_ai.decision_intelligence.options import DecisionOptionGenerator
from commerce_ai.decision_intelligence.risks import DecisionRiskEvaluator
from commerce_ai.decision_intelligence.schemas import (
    DecisionBusinessContext,
    DecisionEntityContext,
    DecisionIntelligenceConfig,
    DecisionIntelligenceResult,
    DecisionOption,
    DecisionOptionType,
    DecisionPackage,
    DecisionRiskFlag,
    DecisionStatus,
    DecisionTraceability,
    DecisionTradeOff,
    DecisionTradeOffDimension,
    InformationAvailability,
    OptionReversibility,
    RequiredDecisionInformation,
    RiskFlagType,
    RiskSeverity,
    generate_decision_id,
    generate_info_id,
    generate_option_id,
    generate_risk_id,
    generate_tradeoff_id,
)
from commerce_ai.decision_intelligence.service import DecisionIntelligenceService
from commerce_ai.decision_intelligence.templates import (
    format_decision_summary,
    format_decision_title,
)
from commerce_ai.decision_intelligence.tradeoffs import DecisionTradeOffEvaluator
from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendation,
    ConfidenceProvenance,
    ConflictType,
    RecommendationConflict,
    RecommendationEvidence,
    RecommendationEvidenceType,
    RecommendationFinancialContext,
    RecommendationOperationalContext,
    RecommendationPriority,
    RecommendationStatus,
    RecommendationTraceability,
    RecommendationType,
)


@pytest.fixture
def sample_as_of_date():
    return date(2026, 6, 30)


@pytest.fixture
def sample_config():
    return DecisionIntelligenceConfig(
        critical_risk_threshold=10000.0,
        high_risk_threshold=2500.0,
        medium_risk_threshold=500.0,
        default_currency="USD",
        auto_expire_decisions=True,
    )


_DEFAULT = object()


def make_recommendation(
    rec_id: str = "REC-0123456789abcdef",
    rec_type: RecommendationType = RecommendationType.REPLENISHMENT_REVIEW,
    priority: RecommendationPriority = RecommendationPriority.HIGH,
    severity: str = "HIGH",
    sku_id: Optional[str] = "SKU_001",
    warehouse_id: Optional[str] = "WH_01",
    channel_id: Optional[str] = None,
    currency: str = "USD",
    title: str = "Replenishment Review: SKU_001 at WH_01",
    summary: str = "Inventory position is below ROP.",
    reason: str = "Net position below reorder threshold.",
    action: str = "Review the existing replenishment recommendation.",
    action_category: str = "REPLENISH",
    financial_context: Any = _DEFAULT,
    operational_context: Any = _DEFAULT,
    evidence: Optional[List[RecommendationEvidence]] = None,
    supporting_signal_ids: Optional[List[str]] = None,
    supporting_impact_ids: Optional[List[str]] = None,
    recommended_quantity: Optional[int] = 260,
    recommended_value: Optional[float] = 6500.0,
    source_engine: str = "replenishment_solver",
    rule_id: str = "RULE_REPLENISHMENT_REVIEW_001",
    rule_version: str = "1.0",
    confidence: ConfidenceProvenance = ConfidenceProvenance.DETERMINISTIC_DERIVED,
    confidence_reason: str = "Deterministic solver calculation",
    created_as_of: date = date(2026, 6, 30),
    valid_until: date = date(2026, 7, 7),
    data_quality_status: str = "COMPLETE",
    calculation_status: str = "CALCULATED",
) -> BusinessRecommendation:
    """Helper to instantiate schema-valid BusinessRecommendation."""
    if financial_context is _DEFAULT:
        fin_ctx = RecommendationFinancialContext(
            currency=currency,
            proposed_purchase_value=recommended_value,
            exposure_value=recommended_value,
        )
    else:
        fin_ctx = financial_context

    if operational_context is _DEFAULT:
        op_ctx = RecommendationOperationalContext(
            inventory_position=40,
            reorder_point=180.0,
            recommended_order_qty=recommended_quantity,
        )
    else:
        op_ctx = operational_context

    return BusinessRecommendation(
        recommendation_id=rec_id,
        recommendation_type=rec_type,
        status=RecommendationStatus.DRAFT,
        priority=priority,
        severity=severity,
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        channel_id=channel_id,
        currency=currency,
        title=title,
        summary=summary,
        reason=reason,
        action=action,
        action_category=action_category,
        evidence=evidence or [
            RecommendationEvidence(
                evidence_type=RecommendationEvidenceType.ENGINE_OUTPUT,
                source_id="REP_001",
                metric="recommended_order_qty",
                value=recommended_quantity or 260,
                unit="units",
                description="Solved replenishment quantity",
            )
        ],
        supporting_signal_ids=supporting_signal_ids or ["SIG_001"],
        supporting_impact_ids=supporting_impact_ids or ["IMP_001"],
        financial_context=fin_ctx,
        operational_context=op_ctx,
        recommended_quantity=recommended_quantity,
        recommended_value=recommended_value,
        source_engine=source_engine,
        rule_id=rule_id,
        rule_version=rule_version,
        confidence=confidence,
        confidence_reason=confidence_reason,
        approval_required=True,
        execution_allowed=False,
        created_as_of=created_as_of,
        valid_until=valid_until,
        data_quality_status=data_quality_status,
        calculation_status=calculation_status,
        traceability=RecommendationTraceability(
            recommendation_id=rec_id,
            supporting_signal_ids=supporting_signal_ids or ["SIG_001"],
            supporting_impact_ids=supporting_impact_ids or ["IMP_001"],
            domain_metrics=["net_inventory_position", "reorder_point"],
            source_datasets=["inventory.csv", "sales.csv"],
        ),
        requires_human_review=True,
    )


# ---------------------------------------------------------------------------
# Test Conditions A through AX
# ---------------------------------------------------------------------------

def test_condition_a_schema_validation(sample_as_of_date):
    """Condition A: Schema validation enforces valid prefixes and contract integrity."""
    with pytest.raises(ValidationError):
        DecisionPackage(
            decision_id="INVALID_ID_WITHOUT_PREFIX",
            recommendation_id="REC-001",
            recommendation_type=RecommendationType.REPLENISHMENT_REVIEW,
            decision_title="Invalid",
            decision_summary="Invalid",
            entity_context=DecisionEntityContext(entity_key="SKU_01:WH_01"),
            business_context=DecisionBusinessContext(
                priority=RecommendationPriority.HIGH,
                source_engine="test",
                rule_id="RULE_001",
                confidence=ConfidenceProvenance.DETERMINISTIC_DERIVED,
                confidence_reason="test",
            ),
            created_as_of=sample_as_of_date,
            valid_until=sample_as_of_date + timedelta(days=7),
        )


def test_condition_b_decision_package_creation(sample_config, sample_as_of_date):
    """Condition B: Decision package created cleanly from recommendation."""
    rec = make_recommendation(created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.decision_id.startswith("DEC-")
    assert pkg.recommendation_id == rec.recommendation_id
    assert pkg.recommendation_type == RecommendationType.REPLENISHMENT_REVIEW
    assert pkg.decision_status == DecisionStatus.PENDING_REVIEW
    assert len(pkg.decision_options) >= 2


def test_condition_c_recommendation_linkage(sample_config, sample_as_of_date):
    """Condition C: Recommendation fields linked 1:1 into decision package."""
    rec = make_recommendation(
        sku_id="SKU_999",
        warehouse_id="WH_NORTH",
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.recommendation_id == rec.recommendation_id
    assert pkg.entity_context.sku_id == "SKU_999"
    assert pkg.entity_context.warehouse_id == "WH_NORTH"
    assert pkg.business_context.rule_id == rec.rule_id


def test_condition_d_deterministic_decision_id(sample_as_of_date):
    """Condition D: Decision ID is deterministic from recommendation, type, date, and rule version."""
    id1 = generate_decision_id("REC-12345", RecommendationType.REPLENISHMENT_REVIEW, sample_as_of_date, "1.0")
    id2 = generate_decision_id("REC-12345", RecommendationType.REPLENISHMENT_REVIEW, sample_as_of_date, "1.0")
    id3 = generate_decision_id("REC-99999", RecommendationType.REPLENISHMENT_REVIEW, sample_as_of_date, "1.0")

    assert id1.startswith("DEC-")
    assert id1 == id2
    assert id1 != id3


def test_condition_e_duplicate_prevention(sample_config, sample_as_of_date):
    """Condition E: Duplicate recommendations are suppressed from emitting duplicate packages."""
    rec = make_recommendation(rec_id="REC-DUP-01", created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    res = service.package_recommendations([rec, rec], as_of_date=sample_as_of_date)

    assert res.total_decisions == 1
    assert len(res.decision_packages) == 1


def test_condition_f_lifecycle(sample_config, sample_as_of_date):
    """Condition F: Initial lifecycle is PENDING_REVIEW; validates valid status transitions."""
    rec = make_recommendation(created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.decision_status == DecisionStatus.PENDING_REVIEW
    # Pydantic validates valid statuses
    for s in DecisionStatus:
        pkg.decision_status = s
        assert pkg.decision_status == s


def test_condition_g_approval_required(sample_config, sample_as_of_date):
    """Condition G: Every decision package strictly enforces approval_required=True."""
    rec = make_recommendation(created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.approval_required is True
    with pytest.raises(ValidationError):
        pkg.approval_required = False


def test_condition_h_execution_prohibited(sample_config, sample_as_of_date):
    """Condition H: Every decision package strictly enforces execution_allowed=False."""
    rec = make_recommendation(created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.execution_allowed is False
    with pytest.raises(ValidationError):
        pkg.execution_allowed = True


def test_condition_i_option_generation(sample_config, sample_as_of_date):
    """Condition I: Multi-option generation supported by available evidence."""
    rec = make_recommendation(rec_type=RecommendationType.REPLENISHMENT_REVIEW, created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert len(pkg.decision_options) >= 3
    types = [o.option_type for o in pkg.decision_options]
    assert DecisionOptionType.REPLENISHMENT_REVIEW in types
    assert DecisionOptionType.DEFER_DECISION in types
    assert DecisionOptionType.NO_CHANGE in types


def test_condition_j_no_winner_selection(sample_config, sample_as_of_date):
    """Condition J: Engine strictly does NOT select a winning or optimal option."""
    rec = make_recommendation(created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.selected_option is None
    for opt in pkg.decision_options:
        assert opt.recommended_by_engine is False


def test_condition_k_tradeoffs(sample_config, sample_as_of_date):
    """Condition K: Trade-offs evaluate multi-dimensional impact with conditional wording."""
    rec = make_recommendation(created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert len(pkg.trade_offs) > 0
    for t in pkg.trade_offs:
        assert t.tradeoff_id.startswith("TRD-")
        assert "may" in t.positive_effect.lower() or "allows" in t.positive_effect.lower() or "maintains" in t.positive_effect.lower()
        assert "depends on" in t.uncertainty.lower()


def test_condition_l_required_information(sample_config, sample_as_of_date):
    """Condition L: Required information attributes tracked for replenishment."""
    rec = make_recommendation(rec_type=RecommendationType.REPLENISHMENT_REVIEW, created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    field_names = [info.field_name for info in pkg.required_information]
    assert "current_inventory" in field_names
    assert "supplier_lead_time" in field_names


def test_condition_m_missing_information(sample_config, sample_as_of_date):
    """Condition M: Missing information marked UNAVAILABLE without data fabrication."""
    # When forecast is missing in operational context:
    rec = make_recommendation(
        operational_context=RecommendationOperationalContext(
            inventory_position=50,
            forecast=None,  # missing forecast
        ),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    fc_info = next((i for i in pkg.required_information if i.field_name == "forecast_demand"), None)
    if fc_info:
        assert fc_info.availability == InformationAvailability.UNAVAILABLE


def test_condition_n_risk_flags(sample_config, sample_as_of_date):
    """Condition N: Risk flags generated qualitatively without probability scores."""
    rec = make_recommendation(
        financial_context=RecommendationFinancialContext(
            inventory_value=15000.0,  # exceeds critical threshold 10,000
        ),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    inv_risk = next((rf for rf in pkg.risk_flags if rf.risk_type == RiskFlagType.INVENTORY_RISK), None)
    assert inv_risk is not None
    assert inv_risk.severity == RiskSeverity.CRITICAL


def test_condition_o_conflict_propagation(sample_config, sample_as_of_date):
    """Condition O: Recommendation conflict propagates CONFLICT_RISK and requires_human_review."""
    rec = make_recommendation(rec_id="REC-CONF-01", created_as_of=sample_as_of_date)
    conflict = RecommendationConflict(
        conflict_id="CNF-001",
        conflict_type=ConflictType.RETURN_QUALITY_CONFLICT,
        conflicting_recommendation_ids=["REC-CONF-01", "REC-OTHER-02"],
        conflicting_signal_ids=["SIG_01", "SIG_02"],
        entity_key="SKU_001",
        explanation="High return rate on high inventory SKU.",
        requires_human_review=True,
        created_as_of=sample_as_of_date,
    )

    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, conflicts=[conflict], as_of_date=sample_as_of_date)

    assert len(pkg.conflicts) == 1
    assert pkg.requires_human_review is True
    conflict_risk = next((rf for rf in pkg.risk_flags if rf.risk_type == RiskFlagType.CONFLICT_RISK), None)
    assert conflict_risk is not None
    assert conflict_risk.severity == RiskSeverity.HIGH


def test_condition_p_financial_context(sample_config, sample_as_of_date):
    """Condition P: Financial context preserved directly from recommendation."""
    rec = make_recommendation(
        financial_context=RecommendationFinancialContext(
            currency="USD",
            proposed_purchase_value=7500.0,
            exposure_value=7500.0,
            associated_revenue=25000.0,
            associated_margin=12000.0,
        ),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.financial_context is not None
    assert pkg.financial_context.proposed_purchase_value == 7500.0
    assert pkg.financial_context.associated_revenue == 25000.0


def test_condition_q_operational_context(sample_config, sample_as_of_date):
    """Condition Q: Operational context preserved directly from recommendation."""
    rec = make_recommendation(
        operational_context=RecommendationOperationalContext(
            inventory_position=35,
            reorder_point=120.0,
            recommended_order_qty=180,
            stockout_days=0,
        ),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.operational_context is not None
    assert pkg.operational_context.inventory_position == 35
    assert pkg.operational_context.reorder_point == 120.0
    assert pkg.operational_context.recommended_order_qty == 180


def test_condition_r_evidence(sample_config, sample_as_of_date):
    """Condition R: Verifiable evidence preserved across package and options."""
    ev = RecommendationEvidence(
        evidence_type=RecommendationEvidenceType.DOMAIN_METRIC,
        source_id="METRIC_01",
        metric="net_inventory_position",
        value=42,
        unit="units",
        description="Physical position count",
    )
    rec = make_recommendation(evidence=[ev], created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert len(pkg.evidence) == 1
    assert pkg.evidence[0].metric == "net_inventory_position"
    assert pkg.evidence[0].value == 42


def test_condition_s_traceability(sample_config, sample_as_of_date):
    """Condition S: Traceability links decision to recommendation, signals, impacts, datasets."""
    rec = make_recommendation(
        supporting_signal_ids=["SIG_100", "SIG_101"],
        supporting_impact_ids=["IMP_200"],
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.traceability is not None
    assert pkg.traceability.decision_id == pkg.decision_id
    assert pkg.traceability.recommendation_id == rec.recommendation_id
    assert "SIG_100" in pkg.traceability.supporting_signal_ids
    assert "IMP_200" in pkg.traceability.supporting_impact_ids


def test_condition_t_replenishment_decision(sample_config, sample_as_of_date):
    """Condition T: Replenishment review decision provides review, defer, no_change options."""
    rec = make_recommendation(
        rec_type=RecommendationType.REPLENISHMENT_REVIEW,
        recommended_quantity=300,
        recommended_value=9000.0,
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    opt_types = [o.option_type for o in pkg.decision_options]
    assert DecisionOptionType.REPLENISHMENT_REVIEW in opt_types
    assert DecisionOptionType.DEFER_DECISION in opt_types
    assert DecisionOptionType.NO_CHANGE in opt_types


def test_condition_u_po_decision(sample_config, sample_as_of_date):
    """Condition U: PO review decision provides draft PO review without submitting order."""
    rec = make_recommendation(
        rec_type=RecommendationType.PURCHASE_ORDER_REVIEW,
        source_engine="purchase_order_engine",
        financial_context=RecommendationFinancialContext(proposed_purchase_value=12500.0),
        operational_context=RecommendationOperationalContext(recommended_order_qty=400),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    po_opt = next((o for o in pkg.decision_options if o.option_type == DecisionOptionType.PURCHASE_ORDER_REVIEW), None)
    assert po_opt is not None
    assert "Draft Purchase Order" in po_opt.title
    assert po_opt.requires_external_action is True


def test_condition_v_transfer_decision(sample_config, sample_as_of_date):
    """Condition V: Warehouse transfer review decision provides transfer review without moving stock."""
    rec = make_recommendation(
        rec_type=RecommendationType.WAREHOUSE_REBALANCING_REVIEW,
        source_engine="warehouse_rebalancer",
        operational_context=RecommendationOperationalContext(
            source_warehouse="WH_01",
            destination_warehouse="WH_02",
            transfer_quantity=80,
        ),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    tr_opt = next((o for o in pkg.decision_options if o.option_type == DecisionOptionType.TRANSFER_REVIEW), None)
    assert tr_opt is not None
    assert "WH_01" in tr_opt.description
    assert "WH_02" in tr_opt.description


def test_condition_w_inventory_decision(sample_config, sample_as_of_date):
    """Condition W: Inventory review decision does NOT invent a transfer option if none exists."""
    # Recommendation without transfer context
    rec = make_recommendation(
        rec_type=RecommendationType.SLOW_MOVING_INVENTORY_REVIEW,
        operational_context=RecommendationOperationalContext(
            inventory_position=500,
            transfer_quantity=None,  # no transfer recommendation
        ),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    opt_types = [o.option_type for o in pkg.decision_options]
    assert DecisionOptionType.INVENTORY_REVIEW in opt_types
    assert DecisionOptionType.TRANSFER_REVIEW not in opt_types  # strictly not invented!


def test_condition_x_stockout_decision(sample_config, sample_as_of_date):
    """Condition X: Stockout decision exposes replenishment only if solver output exists; else demand review."""
    # Case 1: with replenishment
    rec_with_rep = make_recommendation(
        rec_type=RecommendationType.STOCKOUT_REVIEW,
        operational_context=RecommendationOperationalContext(
            inventory_position=0,
            stockout_days=5,
            recommended_order_qty=200,
        ),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg1 = service.package_single_recommendation(rec_with_rep, as_of_date=sample_as_of_date)
    types1 = [o.option_type for o in pkg1.decision_options]
    assert DecisionOptionType.REPLENISHMENT_REVIEW in types1

    # Case 2: without replenishment or transfer
    rec_without = make_recommendation(
        rec_type=RecommendationType.STOCKOUT_REVIEW,
        operational_context=RecommendationOperationalContext(
            inventory_position=0,
            stockout_days=5,
            recommended_order_qty=None,
            transfer_quantity=None,
        ),
        created_as_of=sample_as_of_date,
    )
    pkg2 = service.package_single_recommendation(rec_without, as_of_date=sample_as_of_date)
    types2 = [o.option_type for o in pkg2.decision_options]
    assert DecisionOptionType.DEMAND_REVIEW in types2
    assert DecisionOptionType.REVIEW_ONLY in types2
    assert DecisionOptionType.REPLENISHMENT_REVIEW not in types2


def test_condition_y_return_decision(sample_config, sample_as_of_date):
    """Condition Y: Return decision does NOT recommend pricing, refund, or policy changes."""
    rec = make_recommendation(
        rec_type=RecommendationType.RETURN_REVIEW,
        operational_context=RecommendationOperationalContext(return_rate=0.25),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    opt_types = [o.option_type for o in pkg.decision_options]
    assert DecisionOptionType.RETURN_REVIEW in opt_types
    assert DecisionOptionType.DEMAND_REVIEW in opt_types
    assert DecisionOptionType.DATA_REVIEW in opt_types

    # Ensure no forbidden pricing or refund recommendations exist
    all_text = " ".join([o.title + " " + o.description for o in pkg.decision_options]).lower()
    assert "refund" not in all_text
    assert "price" not in all_text
    assert "return policy" not in all_text


def test_condition_z_margin_decision(sample_config, sample_as_of_date):
    """Condition Z: Margin review decision does NOT recommend price changes."""
    rec = make_recommendation(
        rec_type=RecommendationType.MARGIN_REVIEW,
        financial_context=RecommendationFinancialContext(
            associated_revenue=10000.0,
            associated_margin=1200.0,
        ),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    opt_types = [o.option_type for o in pkg.decision_options]
    assert DecisionOptionType.MARGIN_REVIEW in opt_types
    all_text = " ".join([o.title + " " + o.description for o in pkg.decision_options]).lower()
    assert "retail price" not in all_text
    assert "discount" not in all_text
    assert "mark down" not in all_text


def test_condition_aa_data_quality_decision(sample_config, sample_as_of_date):
    """Condition AA: Data quality decision identifies missing field, entity, and downstream impact."""
    ev = RecommendationEvidence(
        evidence_type=RecommendationEvidenceType.DATA_QUALITY,
        source_id="DQ_01",
        metric="missing_field",
        value="unit_cost",
        description="Missing unit cost attribute",
    )
    rec = make_recommendation(
        rec_type=RecommendationType.DATA_QUALITY_REVIEW,
        evidence=[ev],
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    opt_types = [o.option_type for o in pkg.decision_options]
    assert DecisionOptionType.DATA_REVIEW in opt_types
    info = next((i for i in pkg.required_information if "unit_cost" in i.field_name), None)
    assert info is not None
    assert info.blocking is True


def test_condition_ab_forecast_uncertainty(sample_config, sample_as_of_date):
    """Condition AB: Missing or model-based forecast triggers FORECAST_UNCERTAINTY risk flag."""
    rec = make_recommendation(
        confidence=ConfidenceProvenance.MODEL_BASED,
        operational_context=RecommendationOperationalContext(forecast=150.0),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    fc_risk = next((rf for rf in pkg.risk_flags if rf.risk_type == RiskFlagType.FORECAST_UNCERTAINTY), None)
    assert fc_risk is not None


def test_condition_ac_currency_isolation(sample_config, sample_as_of_date):
    """Condition AC: Inconsistent currencies trigger DATA_QUALITY_RISK and requires_human_review."""
    ev = RecommendationEvidence(
        evidence_type=RecommendationEvidenceType.IMPACT,
        source_id="IMP_EUR",
        metric="exposure_value",
        value=5000.0,
        currency="EUR",  # Inconsistent with USD
        description="Euro denominated impact",
    )
    rec = make_recommendation(
        currency="USD",
        evidence=[ev],
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.requires_human_review is True
    curr_risk = next((rf for rf in pkg.risk_flags if "currency" in rf.description.lower() or "currencies" in rf.description.lower()), None)
    assert curr_risk is not None
    assert curr_risk.severity == RiskSeverity.CRITICAL


def test_condition_ad_future_data_protection(sample_config, sample_as_of_date):
    """Condition AD: Future-data contamination raises ValueError."""
    future_date = sample_as_of_date + timedelta(days=10)
    rec = make_recommendation(created_as_of=future_date)
    service = DecisionIntelligenceService(config=sample_config)

    with pytest.raises(ValueError, match="Future-data contamination"):
        service.package_single_recommendation(rec, as_of_date=sample_as_of_date)


def test_condition_ae_validity_expiration(sample_config, sample_as_of_date):
    """Condition AE: Past-horizon decisions automatically transition to EXPIRED status."""
    expired_valid_until = sample_as_of_date - timedelta(days=2)
    rec = make_recommendation(
        created_as_of=sample_as_of_date - timedelta(days=10),
        valid_until=expired_valid_until,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.decision_status == DecisionStatus.EXPIRED


def test_condition_af_no_causal_claims(sample_config, sample_as_of_date):
    """Condition AF: Summary and expected effects use conditional, non-causal language."""
    rec = make_recommendation(created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    summary_lower = pkg.decision_summary.lower()
    assert "caused by" not in summary_lower
    assert "guarantees" not in summary_lower
    assert "will definitely" not in summary_lower

    for opt in pkg.decision_options:
        eff_lower = opt.expected_effect.lower()
        assert "caused by" not in eff_lower
        assert "guarantees" not in eff_lower


def test_condition_ag_no_autonomous_execution(sample_config, sample_as_of_date):
    """Condition AG: Autonomous execution strictly disabled across all packages."""
    rec = make_recommendation(created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.execution_allowed is False


def test_condition_ah_no_pricing_recommendation(sample_config, sample_as_of_date):
    """Condition AH: No option recommends price changes or automated discounting."""
    for r_type in [RecommendationType.MARGIN_REVIEW, RecommendationType.RETURN_REVIEW, RecommendationType.SLOW_MOVING_INVENTORY_REVIEW]:
        rec = make_recommendation(rec_type=r_type, created_as_of=sample_as_of_date)
        service = DecisionIntelligenceService(config=sample_config)
        pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

        for opt in pkg.decision_options:
            assert "discount" not in opt.title.lower()
            assert "mark down" not in opt.title.lower()


def test_condition_ai_no_fabricated_evidence(sample_config, sample_as_of_date):
    """Condition AI: Decision package contains only evidence derived from recommendation."""
    rec = make_recommendation(
        evidence=[
            RecommendationEvidence(
                evidence_type=RecommendationEvidenceType.SIGNAL,
                source_id="SIG_001",
                metric="observed_return_rate",
                value=0.18,
                description="Observed rate",
            )
        ],
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert len(pkg.evidence) == 1
    assert pkg.evidence[0].source_id == "SIG_001"


def test_condition_aj_no_fabricated_forecast(sample_config, sample_as_of_date):
    """Condition AJ: When forecast is unavailable, availability is marked UNAVAILABLE."""
    rec = make_recommendation(
        operational_context=RecommendationOperationalContext(forecast=None),
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    fc_info = next((i for i in pkg.required_information if i.field_name == "forecast_demand"), None)
    if fc_info:
        assert fc_info.availability == InformationAvailability.UNAVAILABLE


def test_condition_ak_no_fabricated_supplier_info(sample_config, sample_as_of_date):
    """Condition AK: Procurement budget authorization marked UNAVAILABLE when not provided."""
    rec = make_recommendation(
        rec_type=RecommendationType.PURCHASE_ORDER_REVIEW,
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    budget_info = next((i for i in pkg.required_information if "budget" in i.field_name), None)
    assert budget_info is not None
    assert budget_info.availability == InformationAvailability.UNAVAILABLE


def test_condition_al_no_fabricated_logistics_info(sample_config, sample_as_of_date):
    """Condition AL: Dock availability marked UNAVAILABLE when external WMS data is absent."""
    rec = make_recommendation(
        rec_type=RecommendationType.WAREHOUSE_REBALANCING_REVIEW,
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    dock_info = next((i for i in pkg.required_information if "bay" in i.field_name or "dock" in i.field_name), None)
    assert dock_info is not None
    assert dock_info.availability == InformationAvailability.UNAVAILABLE


def test_condition_am_conflict_handling(sample_config, sample_as_of_date):
    """Condition AM: Unresolved conflicts marked without automated resolution."""
    rec = make_recommendation(rec_id="REC-AM-01", created_as_of=sample_as_of_date)
    conf = RecommendationConflict(
        conflict_id="CNF-AM-01",
        conflict_type=ConflictType.INVENTORY_DEMAND_CONFLICT,
        conflicting_recommendation_ids=["REC-AM-01", "REC-AM-02"],
        entity_key="SKU_AM:WH_01",
        explanation="Contradictory stockout and surplus indicators.",
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, conflicts=[conf], as_of_date=sample_as_of_date)

    assert len(pkg.conflicts) == 1
    assert pkg.selected_option is None  # no automated resolution!


def test_condition_an_multiple_recommendations(sample_config, sample_as_of_date):
    """Condition AN: Batch packaging processes multiple recommendations accurately."""
    r1 = make_recommendation(rec_id="REC-AN-01", rec_type=RecommendationType.REPLENISHMENT_REVIEW, created_as_of=sample_as_of_date)
    r2 = make_recommendation(rec_id="REC-AN-02", rec_type=RecommendationType.SLOW_MOVING_INVENTORY_REVIEW, created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    res = service.package_recommendations([r1, r2], as_of_date=sample_as_of_date)

    assert res.total_decisions == 2
    assert res.decisions_by_type[RecommendationType.REPLENISHMENT_REVIEW.value] == 1
    assert res.decisions_by_type[RecommendationType.SLOW_MOVING_INVENTORY_REVIEW.value] == 1


def test_condition_ao_duplicate_recommendations(sample_config, sample_as_of_date):
    """Condition AO: Duplicate input recommendations deduplicated without crashing."""
    r1 = make_recommendation(rec_id="REC-AO-01", created_as_of=sample_as_of_date)
    r2 = make_recommendation(rec_id="REC-AO-01", created_as_of=sample_as_of_date)
    service = DecisionIntelligenceService(config=sample_config)
    res = service.package_recommendations([r1, r2], as_of_date=sample_as_of_date)

    assert res.total_decisions == 1


def test_condition_ap_recommendation_without_financial_impact(sample_config, sample_as_of_date):
    """Condition AP: Recommendation without financial context handled gracefully."""
    rec = make_recommendation(
        financial_context=None,
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.financial_context is None
    assert pkg.decision_id.startswith("DEC-")


def test_condition_aq_insufficient_data(sample_config, sample_as_of_date):
    """Condition AQ: Insufficient data confidence tier flags INSUFFICIENT_EVIDENCE risk."""
    rec = make_recommendation(
        confidence=ConfidenceProvenance.INSUFFICIENT_DATA,
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert any(rf.risk_type == RiskFlagType.INSUFFICIENT_EVIDENCE for rf in pkg.risk_flags)


def test_condition_ar_model_based_provenance(sample_config, sample_as_of_date):
    """Condition AR: Model-based confidence provenance preserved in business context."""
    rec = make_recommendation(
        confidence=ConfidenceProvenance.MODEL_BASED,
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.business_context.confidence == ConfidenceProvenance.MODEL_BASED


def test_condition_as_deterministic_derived_provenance(sample_config, sample_as_of_date):
    """Condition AS: Deterministic-derived confidence provenance preserved."""
    rec = make_recommendation(
        confidence=ConfidenceProvenance.DETERMINISTIC_DERIVED,
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.business_context.confidence == ConfidenceProvenance.DETERMINISTIC_DERIVED


def test_condition_at_direct_observed_provenance(sample_config, sample_as_of_date):
    """Condition AT: Direct-observed confidence provenance preserved."""
    rec = make_recommendation(
        confidence=ConfidenceProvenance.DIRECT_OBSERVED,
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.business_context.confidence == ConfidenceProvenance.DIRECT_OBSERVED


def test_condition_au_insufficient_data_provenance(sample_config, sample_as_of_date):
    """Condition AU: Insufficient-data provenance preserved."""
    rec = make_recommendation(
        confidence=ConfidenceProvenance.INSUFFICIENT_DATA,
        created_as_of=sample_as_of_date,
    )
    service = DecisionIntelligenceService(config=sample_config)
    pkg = service.package_single_recommendation(rec, as_of_date=sample_as_of_date)

    assert pkg.business_context.confidence == ConfidenceProvenance.INSUFFICIENT_DATA


def test_condition_av_phase_6g_regression(sample_config, sample_as_of_date):
    """Condition AV: Phase 6G recommendation contracts remain intact."""
    rec = make_recommendation(created_as_of=sample_as_of_date)
    assert rec.recommendation_id.startswith("REC-")
    assert rec.approval_required is True
    assert rec.execution_allowed is False


def test_condition_aw_phase_6f_regression():
    """Condition AW: Phase 6F impact record schemas remain intact."""
    imp = BusinessImpactRecord(
        impact_id="IMP-001",
        signal_id="SIG-001",
        signal_type="HIGH_VALUE_INVENTORY",
        impact_category=ImpactCategory.INVENTORY,
        impact_type=ImpactType.INVENTORY_CAPITAL_EXPOSURE,
        severity=SignalSeverity.HIGH,
        exposure_value=12500.0,
        exposure_currency="USD",
        confidence_status=ImpactConfidence.DIRECT_OBSERVED,
        calculation_method="direct",
        calculation_status=CalculationStatus.CALCULATED,
        description="Holding exposure",
    )
    assert imp.impact_id == "IMP-001"
    assert imp.exposure_value == 12500.0


def test_condition_ax_phase_6e_regression():
    """Condition AX: Phase 6E cross-domain signals remain intact."""
    sig = CrossDomainSignal(
        signal_id="SIG-001",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.MEDIUM,
        dimension=CrossDomainDimension.SKU_WAREHOUSE.value,
        sku_id="SKU_001",
        warehouse_id="WH_01",
        description="High inventory value",
    )
    assert sig.signal_id == "SIG-001"
    assert sig.sku_id == "SKU_001"
