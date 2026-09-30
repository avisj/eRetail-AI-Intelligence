"""Unit and Integration Tests for Business Recommendations Engine (Phase 6G).

Tests all 56 specified conditions from A through BD:
- Condition A: schema validation
- Condition B: replenishment recommendation
- Condition C: purchase order review
- Condition D: warehouse rebalancing review
- Condition E: inventory review
- Condition F: slow-moving inventory review
- Condition G: high-value inventory review
- Condition H: stockout review
- Condition I: return review
- Condition J: return root-cause review
- Condition K: margin review
- Condition L: forecast review
- Condition M: data-quality review
- Condition N: missing evidence handling
- Condition O: insufficient data handling
- Condition P: recommendation priority
- Condition Q: deterministic priority
- Condition R: approval_required strictly True
- Condition S: execution_allowed strictly False
- Condition T: recommendation ID determinism
- Condition U: duplicate prevention
- Condition V: signal traceability
- Condition W: impact traceability
- Condition X: evidence creation
- Condition Y: financial context
- Condition Z: operational context
- Condition AA: confidence provenance
- Condition AB: rule ID
- Condition AC: rule version
- Condition AD: validity horizon
- Condition AE: currency isolation
- Condition AF: as_of_date handling
- Condition AG: future-data protection
- Condition AH: conflict detection
- Condition AI: conflicting recommendations
- Condition AJ: human-review flag
- Condition AK: high inventory + replenishment conflict
- Condition AL: stockout + replenishment linkage
- Condition AM: stockout + rebalancing linkage
- Condition AN: return anomaly linkage
- Condition AO: margin signal linkage
- Condition AP: forecast linkage
- Condition AQ: missing supplier handling
- Condition AR: missing unit cost handling
- Condition AS: missing forecast handling
- Condition AT: missing inventory handling
- Condition AU: duplicate signals handling
- Condition AV: multiple signals for same entity
- Condition AW: no autonomous execution
- Condition AX: no PO creation
- Condition AY: no transfer execution
- Condition AZ: no pricing recommendation
- Condition BA: no causal wording
- Condition BB: no ranking
- Condition BC: Phase 6E regression
- Condition BD: Phase 6F regression
"""

from datetime import date, timedelta
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
from commerce_ai.recommendations.business_recommendations import (
    BusinessRecommendationEngine,
    RecommendationConflictDetector,
    calculate_valid_until,
    generate_conflict_id,
    generate_recommendation_id,
)
from commerce_ai.recommendations.business_rules import (
    ALL_RULES,
    RULES_BY_ID,
    RULE_DATA_QUALITY_REVIEW_001,
    RULE_FORECAST_REVIEW_001,
    RULE_HIGH_VALUE_INVENTORY_001,
    RULE_INVENTORY_REVIEW_001,
    RULE_MARGIN_REVIEW_001,
    RULE_PURCHASE_ORDER_REVIEW_001,
    RULE_REPLENISHMENT_REVIEW_001,
    RULE_RETURN_REVIEW_001,
    RULE_RETURN_ROOT_CAUSE_001,
    RULE_SLOW_MOVING_INVENTORY_001,
    RULE_STOCKOUT_REVIEW_001,
    RULE_WAREHOUSE_REBALANCING_REVIEW_001,
)
from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendation,
    BusinessRecommendationConfig,
    BusinessRecommendationResult,
    ConfidenceProvenance,
    ConflictType,
    RecommendationConflict,
    RecommendationEvidence,
    RecommendationEvidenceType,
    RecommendationFinancialContext,
    RecommendationOperationalContext,
    RecommendationPriority,
    RecommendationStatus,
    RecommendationType,
)
from commerce_ai.recommendations.schemas import (
    PurchaseOrderLineProposal,
    PurchaseOrderProposal,
    ReplenishmentRecommendation,
    WarehouseTransferRecommendation,
)


def make_signal(
    signal_id: str,
    signal_type: str,
    category: SignalCategory = SignalCategory.CROSS_DOMAIN,
    severity: SignalSeverity = SignalSeverity.MEDIUM,
    dimension: str = CrossDomainDimension.SKU_WAREHOUSE.value,
    sku_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    description: str = "Test signal",
    observed_metrics: Optional[Dict[str, Any]] = None,
    as_of_date: Optional[str] = "2026-06-30",
) -> CrossDomainSignal:
    """Helper to instantiate schema-valid CrossDomainSignal."""
    return CrossDomainSignal(
        signal_id=signal_id,
        signal_type=signal_type,
        category=category,
        severity=severity,
        dimension=dimension,
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        channel_id=channel_id,
        description=description,
        observed_metrics=observed_metrics or {},
        as_of_date=as_of_date,
    )


def make_impact(
    impact_id: str,
    signal_id: str,
    signal_type: str,
    impact_category: ImpactCategory,
    impact_type: ImpactType,
    severity: SignalSeverity = SignalSeverity.MEDIUM,
    sku_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    exposure_value: Optional[float] = None,
    exposure_currency: str = "USD",
    confidence_status: ImpactConfidence = ImpactConfidence.DIRECT_OBSERVED,
    calculation_method: str = "direct_calculation",
    calculation_status: CalculationStatus = CalculationStatus.CALCULATED,
    description: str = "Quantified financial exposure",
    as_of_date: Optional[str] = "2026-06-30",
) -> BusinessImpactRecord:
    """Helper to instantiate schema-valid BusinessImpactRecord."""
    return BusinessImpactRecord(
        impact_id=impact_id,
        signal_id=signal_id,
        signal_type=signal_type,
        impact_category=impact_category,
        impact_type=impact_type,
        severity=severity,
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        exposure_value=exposure_value,
        exposure_currency=exposure_currency,
        confidence_status=confidence_status,
        calculation_method=calculation_method,
        calculation_status=calculation_status,
        description=description,
        as_of_date=as_of_date,
    )


@pytest.fixture
def sample_as_of_date():
    return date(2026, 6, 30)


@pytest.fixture
def sample_config():
    return BusinessRecommendationConfig(
        critical_impact_threshold=10000.0,
        high_impact_threshold=2500.0,
        medium_impact_threshold=500.0,
        validity_days_replenishment=7,
        validity_days_po=7,
        validity_days_rebalancing=7,
        validity_days_inventory=7,
        validity_days_stockout=7,
        validity_days_returns=30,
        validity_days_forecast=14,
        validity_days_data_quality=3,
        conflict_detection_enabled=True,
        default_currency="USD",
    )


@pytest.fixture
def sample_replenishment(sample_as_of_date):
    return ReplenishmentRecommendation(
        sku_id="SKU_001",
        warehouse_id="WH_01",
        supplier_id="SUP_01",
        recommendation_required=True,
        urgency="HIGH",
        risk_category="REORDER_REQUIRED",
        on_hand=50,
        reserved=10,
        on_order=0,
        net_inventory_position=40,
        forecast_daily_demand=15.0,
        lead_time_days=10.0,
        reorder_point=180.0,
        target_stock_level=300.0,
        shortfall_qty=140,
        required_qty=260,
        recommended_order_qty=260,
        unit_cost=25.0,
        estimated_order_cost=6500.0,
        currency="USD",
        rationale="Net position (40) below ROP (180.0). Deficit is 260 units.",
        constraints_applied=[],
        status="COMPLETED",
        recommendation_id="REC_SKU_001_WH_01",
    )


@pytest.fixture
def sample_po_proposal(sample_as_of_date):
    line = PurchaseOrderLineProposal(
        line_id="LINE_001",
        sku_id="SKU_001",
        supplier_id="SUP_01",
        warehouse_id="WH_01",
        quantity=260,
        unit_cost=25.0,
        estimated_line_value=6500.0,
        currency="USD",
        urgency="HIGH",
        risk_category="REORDER_REQUIRED",
        expected_delivery_date=(sample_as_of_date + timedelta(days=10)).isoformat(),
        source_recommendation_id="REC_SKU_001_WH_01",
        rationale="Urgent replenishment",
        constraints_applied=[],
    )
    return PurchaseOrderProposal(
        proposal_id="PROP_SUP_01_WH_01_USD",
        supplier_id="SUP_01",
        warehouse_id="WH_01",
        proposal_date=sample_as_of_date.isoformat(),
        currency="USD",
        lines=[line],
        total_quantity=260,
        total_value=6500.0,
        cost_data_status="COMPLETE_COST_DATA",
        expected_delivery_date=(sample_as_of_date + timedelta(days=10)).isoformat(),
        urgency="HIGH",
        status="DRAFT",
        approval_required=True,
        source_recommendation_ids=["REC_SKU_001_WH_01"],
        rationale="Supplier SUP_01 proposal for warehouse WH_01",
        constraints_applied=[],
    )


@pytest.fixture
def sample_warehouse_transfer(sample_as_of_date):
    return WarehouseTransferRecommendation(
        recommendation_id="TR_SKU_001_WH_02_WH_01",
        sku_id="SKU_001",
        source_warehouse_id="WH_02",
        destination_warehouse_id="WH_01",
        transfer_quantity=80.0,
        source_net_inventory_position=480.0,
        source_reorder_point=150.0,
        source_transferable_surplus=330.0,
        destination_net_inventory_position=30.0,
        destination_reorder_point=180.0,
        destination_required_qty=150.0,
        destination_risk_category="REORDER_REQUIRED",
        priority="HIGH",
        forecast_daily_demand=12.0,
        estimated_transfer_cost=160.0,
        rationale="Transfer 80 units from surplus WH_02 to deficit WH_01",
        constraints_applied=[],
        approval_required=True,
        status="DRAFT",
    )


# ---------------------------------------------------------------------------
# Test Conditions A through BD
# ---------------------------------------------------------------------------


def test_condition_a_schema_validation(sample_as_of_date):
    """Condition A: Schema validation rejects invalid prefixes and enforces contracts."""
    with pytest.raises(ValidationError):
        BusinessRecommendation(
            recommendation_id="INVALID_ID_WITHOUT_PREFIX",
            recommendation_type=RecommendationType.REPLENISHMENT_REVIEW,
            priority=RecommendationPriority.HIGH,
            title="Invalid",
            summary="Invalid",
            reason="Invalid",
            action="Review",
            action_category="REVIEW",
            source_engine="test",
            rule_id="RULE_001",
            confidence=ConfidenceProvenance.DETERMINISTIC_DERIVED,
            confidence_reason="Test",
            created_as_of=sample_as_of_date,
            valid_until=sample_as_of_date + timedelta(days=7),
        )


def test_condition_b_replenishment_recommendation(sample_replenishment, sample_config, sample_as_of_date):
    """Condition B: Replenishment review recommendation generated correctly."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.REPLENISHMENT_REVIEW
    assert rec.sku_id == "SKU_001"
    assert rec.warehouse_id == "WH_01"
    assert rec.recommended_quantity == 260
    assert rec.recommended_value == 6500.0
    assert rec.action == "Review the existing replenishment recommendation."
    assert rec.action_category == "REPLENISH"


def test_condition_c_purchase_order_review(sample_po_proposal, sample_config, sample_as_of_date):
    """Condition C: Purchase order review recommendation generated correctly."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        purchase_order_proposals=[sample_po_proposal],
        as_of_date=sample_as_of_date,
    )
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.PURCHASE_ORDER_REVIEW
    assert rec.recommended_quantity == 260
    assert rec.recommended_value == 6500.0
    assert rec.action == "Review the draft purchase order proposal."


def test_condition_d_warehouse_rebalancing_review(sample_warehouse_transfer, sample_config, sample_as_of_date):
    """Condition D: Warehouse rebalancing review recommendation generated correctly."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        warehouse_transfers=[sample_warehouse_transfer],
        as_of_date=sample_as_of_date,
    )
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.WAREHOUSE_REBALANCING_REVIEW
    assert rec.recommended_quantity == 80
    assert rec.operational_context.source_warehouse == "WH_02"
    assert rec.operational_context.destination_warehouse == "WH_01"
    assert rec.action == "Review the existing warehouse transfer recommendation."


def test_condition_e_inventory_review(sample_config, sample_as_of_date):
    """Condition E: General inventory review generated from inventory signal."""
    sig = make_signal(
        signal_id="SIG_001",
        signal_type="INVENTORY_IMBALANCE",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_002",
        warehouse_id="WH_01",
        description="Inventory imbalance across regions",
        observed_metrics={"inventory_position": 150},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.INVENTORY_REVIEW


def test_condition_f_slow_moving_inventory_review(sample_config, sample_as_of_date):
    """Condition F: Slow-moving inventory review generated from signal."""
    sig = make_signal(
        signal_id="SIG_SLOW_01",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_SLOW",
        warehouse_id="WH_02",
        description="High inventory holding volume coexisting with low demand rate",
        observed_metrics={"inventory_position": 1200, "inventory_value": 36000.0, "daily_demand": 0.2},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.SLOW_MOVING_INVENTORY_REVIEW
    assert rec.action == "Review inventory disposition or rebalancing options."


def test_condition_g_high_value_inventory_review(sample_config, sample_as_of_date):
    """Condition G: High-value inventory review generated from signal."""
    sig = make_signal(
        signal_id="SIG_VAL_01",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_HIGH_VAL",
        warehouse_id="WH_01",
        description="High capital concentration in inventory",
        observed_metrics={"inventory_position": 400, "inventory_value": 40000.0},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.HIGH_VALUE_INVENTORY_REVIEW
    assert rec.action == "Review the high-value inventory position."


def test_condition_h_stockout_review(sample_config, sample_as_of_date):
    """Condition H: Stockout review generated from signal."""
    sig = make_signal(
        signal_id="SIG_SO_01",
        signal_type="HIGH_REVENUE_STOCKOUT_EXPOSURE",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.CRITICAL,
        sku_id="SKU_SO",
        warehouse_id="WH_01",
        description="High revenue SKU facing prolonged stockout",
        observed_metrics={"inventory_position": 0, "stockout_days": 12, "gross_revenue": 15000.0},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.STOCKOUT_REVIEW
    assert rec.priority == RecommendationPriority.CRITICAL


def test_condition_i_return_review(sample_config, sample_as_of_date):
    """Condition I: Return review generated from high return signal."""
    sig = make_signal(
        signal_id="SIG_RET_01",
        signal_type="HIGH_RETURN_HIGH_REVENUE",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension=CrossDomainDimension.SKU.value,
        sku_id="SKU_RET",
        description="High return volume on high revenue SKU",
        observed_metrics={"return_rate": 0.25, "gross_revenue": 50000.0},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.RETURN_REVIEW
    assert rec.action == "Review return drivers and operational handling."


def test_condition_j_return_root_cause_review(sample_config, sample_as_of_date):
    """Condition J: Return root cause review generated from shift signal."""
    sig = make_signal(
        signal_id="SIG_ROOT_01",
        signal_type="RETURN_REASON_SHIFT_ANOMALY",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.HIGH,
        dimension=CrossDomainDimension.SKU.value,
        sku_id="SKU_DEFECT",
        description="Statistically significant shift toward DEFECTIVE reason",
        observed_metrics={"return_reason": "DEFECTIVE", "current_rate": 0.45},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.RETURN_ROOT_CAUSE_REVIEW
    assert rec.action == "Review the identified return reason pattern."


def test_condition_k_margin_review(sample_config, sample_as_of_date):
    """Condition K: Margin review generated from margin signal."""
    sig = make_signal(
        signal_id="SIG_MARG_01",
        signal_type="LOW_MARGIN_HIGH_INVENTORY",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_MARG",
        warehouse_id="WH_01",
        description="Low gross margin coexisting with high inventory capital",
        observed_metrics={"gross_margin": 0.05, "inventory_value": 25000.0},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.MARGIN_REVIEW
    assert rec.action == "Review margin drivers and associated operating conditions."


def test_condition_l_forecast_review(sample_config, sample_as_of_date):
    """Condition L: Forecast review generated from forecast coverage signal."""
    sig = make_signal(
        signal_id="SIG_FC_01",
        signal_type="FORECAST_COVERAGE_EXPOSURE",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_FC",
        warehouse_id="WH_01",
        description="Forward forecast exceeds available stock",
        observed_metrics={"forecast": 500, "inventory_position": 80},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.FORECAST_REVIEW
    assert rec.action == "Review forecast coverage against current inventory."


def test_condition_m_data_quality_review(sample_config, sample_as_of_date):
    """Condition M: Data quality review generated from missing data signal."""
    sig = make_signal(
        signal_id="SIG_DQ_01",
        signal_type="MISSING_UNIT_COST",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        dimension=CrossDomainDimension.SKU.value,
        sku_id="SKU_NO_COST",
        description="Missing unit cost in product catalog",
        observed_metrics={"missing_field": "unit_cost", "entity_id": "SKU_NO_COST"},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 1
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.DATA_QUALITY_REVIEW
    assert rec.action == "Review and remediate missing upstream operational and master data."


def test_condition_n_missing_evidence_handling(sample_config, sample_as_of_date):
    """Condition N: Missing evidence metrics handled gracefully without crashing."""
    sig = make_signal(
        signal_id="SIG_EMPTY_01",
        signal_type="INVENTORY_IMBALANCE",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.LOW,
        dimension=CrossDomainDimension.SKU.value,
        sku_id="SKU_EMPTY",
        description="Signal with empty metrics",
        observed_metrics={},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 1
    assert res.recommendations[0].recommendation_type == RecommendationType.INVENTORY_REVIEW


def test_condition_o_insufficient_data(sample_config, sample_as_of_date):
    """Condition O: Insufficient data yields INSUFFICIENT_DATA confidence provenance."""
    sig = make_signal(
        signal_id="SIG_INSUF_01",
        signal_type="MISSING_FORECAST",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.LOW,
        dimension=CrossDomainDimension.SKU.value,
        sku_id="SKU_INSUF",
        description="Insufficient data for forecasting",
        observed_metrics={"missing_field": "forecast"},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    rec = res.recommendations[0]
    assert rec.confidence == ConfidenceProvenance.INSUFFICIENT_DATA
    assert res.insufficient_data_count >= 1


def test_condition_p_recommendation_priority(sample_config):
    """Condition P: Recommendation priority tiers map deterministically."""
    rule = RULE_SLOW_MOVING_INVENTORY_001
    crit_pri = rule.determine_priority("CRITICAL", 15000.0, sample_config)
    high_pri = rule.determine_priority("HIGH", 3000.0, sample_config)
    med_pri = rule.determine_priority("MEDIUM", 600.0, sample_config)
    low_pri = rule.determine_priority("LOW", 100.0, sample_config)
    info_pri = rule.determine_priority("INFO", 0.0, sample_config)

    assert crit_pri == RecommendationPriority.CRITICAL
    assert high_pri == RecommendationPriority.HIGH
    assert med_pri == RecommendationPriority.MEDIUM
    assert low_pri == RecommendationPriority.LOW
    assert info_pri == RecommendationPriority.INFO


def test_condition_q_deterministic_priority(sample_config):
    """Condition Q: Deterministic priority returns identical output on repeated runs."""
    rule = RULE_REPLENISHMENT_REVIEW_001
    for _ in range(10):
        p = rule.determine_priority("HIGH", 4500.0, sample_config, urgency="HIGH")
        assert p == RecommendationPriority.HIGH


def test_condition_r_approval_required(sample_replenishment, sample_config, sample_as_of_date):
    """Condition R: approval_required is strictly True and frozen."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    rec = res.recommendations[0]
    assert rec.approval_required is True

    # Check immutability / frozen behavior
    with pytest.raises(ValidationError):
        rec.approval_required = False


def test_condition_s_execution_allowed_false(sample_replenishment, sample_config, sample_as_of_date):
    """Condition S: execution_allowed is strictly False and frozen."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    rec = res.recommendations[0]
    assert rec.execution_allowed is False

    with pytest.raises(ValidationError):
        rec.execution_allowed = True


def test_condition_t_recommendation_id_determinism(sample_as_of_date):
    """Condition T: recommendation ID determinism using SHA-256."""
    id1 = generate_recommendation_id("SKU_001:WH_01", RecommendationType.REPLENISHMENT_REVIEW, sample_as_of_date, "RULE_001")
    id2 = generate_recommendation_id("SKU_001:WH_01", RecommendationType.REPLENISHMENT_REVIEW, sample_as_of_date, "RULE_001")
    assert id1 == id2
    assert id1.startswith("REC-")
    assert len(id1) == 20  # REC- (4) + 16 hex = 20 chars


def test_condition_u_duplicate_prevention(sample_replenishment, sample_config, sample_as_of_date):
    """Condition U: Duplicate inputs produce a single deduplicated recommendation."""
    engine = BusinessRecommendationEngine(config=sample_config)
    # Pass identical replenishment recommendation twice
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment, sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    assert res.total_recommendations == 1


def test_condition_v_signal_traceability(sample_config, sample_as_of_date):
    """Condition V: Signal traceability back to source signal IDs."""
    sig = make_signal(
        signal_id="SIG_TRACE_01",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_TR",
        warehouse_id="WH_01",
        description="Trace test signal",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    rec = res.recommendations[0]
    assert "SIG_TRACE_01" in rec.supporting_signal_ids
    assert "SIG_TRACE_01" in rec.traceability.supporting_signal_ids


def test_condition_w_impact_traceability(sample_config, sample_as_of_date):
    """Condition W: Impact traceability back to source impact IDs."""
    sig = make_signal(
        signal_id="SIG_IMP_01",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_IMP",
        warehouse_id="WH_01",
        description="Signal with impact",
        as_of_date=sample_as_of_date.isoformat(),
    )
    imp = make_impact(
        impact_id="IMP_001",
        signal_id="SIG_IMP_01",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        impact_category=ImpactCategory.INVENTORY,
        impact_type=ImpactType.SLOW_MOVING_INVENTORY_EXPOSURE,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_IMP",
        warehouse_id="WH_01",
        exposure_value=12000.0,
        confidence_status=ImpactConfidence.DIRECT_OBSERVED,
        calculation_status=CalculationStatus.CALCULATED,
        description="Slow moving capital",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], impacts=[imp], as_of_date=sample_as_of_date)
    rec = res.recommendations[0]
    assert "IMP_001" in rec.supporting_impact_ids
    assert "IMP_001" in rec.traceability.supporting_impact_ids


def test_condition_x_evidence_creation(sample_config, sample_as_of_date):
    """Condition X: Recommendation evidence items created with source_id and description."""
    sig = make_signal(
        signal_id="SIG_EVID_01",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_EVID",
        warehouse_id="WH_01",
        description="Evidence test signal",
        observed_metrics={"daily_demand": 0.5},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig], as_of_date=sample_as_of_date)
    rec = res.recommendations[0]
    assert len(rec.evidence) >= 1
    assert any(e.evidence_type == RecommendationEvidenceType.SIGNAL for e in rec.evidence)


def test_condition_y_financial_context(sample_replenishment, sample_config, sample_as_of_date):
    """Condition Y: Financial context populated without fabricating missing figures."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    rec = res.recommendations[0]
    assert rec.financial_context is not None
    assert rec.financial_context.proposed_purchase_value == 6500.0
    assert rec.financial_context.currency == "USD"


def test_condition_z_operational_context(sample_replenishment, sample_config, sample_as_of_date):
    """Condition Z: Operational context populated without fabricating values."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    rec = res.recommendations[0]
    assert rec.operational_context is not None
    assert rec.operational_context.inventory_position == 40
    assert rec.operational_context.reorder_point == 180.0
    assert rec.operational_context.recommended_order_qty == 260


def test_condition_aa_confidence_provenance(sample_replenishment, sample_config, sample_as_of_date):
    """Condition AA: Confidence provenance accurately reflects methodology."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    rec = res.recommendations[0]
    assert rec.confidence == ConfidenceProvenance.DETERMINISTIC_DERIVED
    assert "deterministic" in rec.confidence_reason.lower()


def test_condition_ab_rule_id(sample_replenishment, sample_config, sample_as_of_date):
    """Condition AB: Rule ID correctly assigned."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    rec = res.recommendations[0]
    assert rec.rule_id == "RULE_REPLENISHMENT_REVIEW_001"


def test_condition_ac_rule_version(sample_replenishment, sample_config, sample_as_of_date):
    """Condition AC: Rule version is present ('1.0')."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    rec = res.recommendations[0]
    assert rec.rule_version == "1.0"


def test_condition_ad_validity(sample_as_of_date, sample_config):
    """Condition AD: Validity calculation correctly computes expiration date."""
    valid_until = calculate_valid_until(
        sample_as_of_date, RecommendationType.REPLENISHMENT_REVIEW, sample_config
    )
    assert valid_until == sample_as_of_date + timedelta(days=7)

    ret_valid_until = calculate_valid_until(
        sample_as_of_date, RecommendationType.RETURN_REVIEW, sample_config
    )
    assert ret_valid_until == sample_as_of_date + timedelta(days=30)


def test_condition_ae_currency_isolation(sample_replenishment, sample_config, sample_as_of_date):
    """Condition AE: Currency isolation preserves currency tag."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    rec = res.recommendations[0]
    assert rec.currency == "USD"
    assert rec.financial_context.currency == "USD"


def test_condition_af_as_of_date(sample_config, sample_as_of_date):
    """Condition AF: as_of_date filtering and result stamp."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(as_of_date=sample_as_of_date)
    assert res.as_of_date == sample_as_of_date


def test_condition_ag_future_data_protection(sample_config, sample_as_of_date):
    """Condition AG: Future-dated signals are excluded by point-in-time safety."""
    future_sig = make_signal(
        signal_id="SIG_FUT_01",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_FUT",
        warehouse_id="WH_01",
        description="Future signal",
        as_of_date=(sample_as_of_date + timedelta(days=5)).isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[future_sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 0


def test_condition_ah_conflict_detection(sample_config, sample_as_of_date):
    """Condition AH: Conflict detection identifies contradictions."""
    sig_slow = make_signal(
        signal_id="SIG_SLOW",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_CONF",
        warehouse_id="WH_01",
        description="Slow moving inventory",
        as_of_date=sample_as_of_date.isoformat(),
    )
    sig_rep = make_signal(
        signal_id="SIG_REP",
        signal_type="HIGH_REVENUE_REPLENISHMENT_TRIGGER",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_CONF",
        warehouse_id="WH_01",
        description="Replenishment trigger",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig_slow, sig_rep], as_of_date=sample_as_of_date)
    assert res.conflict_count >= 1
    assert any(c.conflict_type == ConflictType.INVENTORY_DEMAND_CONFLICT for c in res.conflicts)


def test_condition_ai_conflicting_recommendations(sample_config, sample_as_of_date):
    """Condition AI: Conflicting recommendations linked in conflict record."""
    sig_slow = make_signal(
        signal_id="SIG_SLOW",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_CONF",
        warehouse_id="WH_01",
        description="Slow moving inventory",
        as_of_date=sample_as_of_date.isoformat(),
    )
    sig_rep = make_signal(
        signal_id="SIG_REP",
        signal_type="HIGH_REVENUE_REPLENISHMENT_TRIGGER",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_CONF",
        warehouse_id="WH_01",
        description="Replenishment trigger",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig_slow, sig_rep], as_of_date=sample_as_of_date)
    conflict = res.conflicts[0]
    assert len(conflict.conflicting_signal_ids) >= 2


def test_condition_aj_human_review_flag(sample_config, sample_as_of_date):
    """Condition AJ: requires_human_review flag is True on recommendations and conflicts."""
    sig_slow = make_signal(
        signal_id="SIG_SLOW",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_CONF",
        warehouse_id="WH_01",
        description="Slow moving inventory",
        as_of_date=sample_as_of_date.isoformat(),
    )
    sig_rep = make_signal(
        signal_id="SIG_REP",
        signal_type="HIGH_REVENUE_REPLENISHMENT_TRIGGER",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_CONF",
        warehouse_id="WH_01",
        description="Replenishment trigger",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig_slow, sig_rep], as_of_date=sample_as_of_date)
    for rec in res.recommendations:
        assert rec.requires_human_review is True
    for c in res.conflicts:
        assert c.requires_human_review is True


def test_condition_ak_high_inventory_replenishment_conflict(sample_config, sample_as_of_date):
    """Condition AK: High inventory + replenishment conflict detected specifically."""
    sig_slow = make_signal(
        signal_id="SIG_SLOW",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_AK",
        warehouse_id="WH_01",
        description="High inventory low demand",
        as_of_date=sample_as_of_date.isoformat(),
    )
    sig_rep = make_signal(
        signal_id="SIG_REP",
        signal_type="HIGH_REVENUE_REPLENISHMENT_TRIGGER",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_AK",
        warehouse_id="WH_01",
        description="Replenishment trigger",
        as_of_date=sample_as_of_date.isoformat(),
    )
    conflicts = RecommendationConflictDetector.detect_conflicts([], [sig_slow, sig_rep], sample_as_of_date)
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == ConflictType.INVENTORY_DEMAND_CONFLICT
    assert conflicts[0].requires_human_review is True


def test_condition_al_stockout_replenishment_linkage(sample_replenishment, sample_config, sample_as_of_date):
    """Condition AL: Stockout recommendation links existing replenishment recommendation."""
    sig_so = make_signal(
        signal_id="SIG_SO",
        signal_type="HIGH_REVENUE_STOCKOUT_EXPOSURE",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.CRITICAL,
        sku_id="SKU_001",
        warehouse_id="WH_01",
        description="Stockout of top SKU",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        signals=[sig_so],
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    so_rec = next(r for r in res.recommendations if r.recommendation_type == RecommendationType.STOCKOUT_REVIEW)
    assert so_rec.operational_context.recommended_order_qty == sample_replenishment.recommended_order_qty
    assert "replenishment order" in so_rec.action.lower()


def test_condition_am_stockout_rebalancing_linkage(sample_warehouse_transfer, sample_config, sample_as_of_date):
    """Condition AM: Stockout recommendation links existing warehouse rebalancing recommendation."""
    sig_so = make_signal(
        signal_id="SIG_SO",
        signal_type="HIGH_REVENUE_STOCKOUT_EXPOSURE",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.CRITICAL,
        sku_id="SKU_001",
        warehouse_id="WH_01",
        description="Stockout of top SKU",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        signals=[sig_so],
        warehouse_transfers=[sample_warehouse_transfer],
        as_of_date=sample_as_of_date,
    )
    so_rec = next(r for r in res.recommendations if r.recommendation_type == RecommendationType.STOCKOUT_REVIEW)
    assert so_rec.operational_context.transfer_quantity == sample_warehouse_transfer.transfer_quantity
    assert "warehouse transfer" in so_rec.action.lower()


def test_condition_an_return_anomaly_linkage(sample_config, sample_as_of_date):
    """Condition AN: Return review links return anomaly signal."""
    sig_ret = make_signal(
        signal_id="SIG_ANOM",
        signal_type="RETURN_ANOMALY_HIGH_VALUE_SKU",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension=CrossDomainDimension.SKU.value,
        sku_id="SKU_ANOM",
        description="Return anomaly on high value SKU",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig_ret], as_of_date=sample_as_of_date)
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.RETURN_REVIEW
    assert "SIG_ANOM" in rec.supporting_signal_ids


def test_condition_ao_margin_signal_linkage(sample_config, sample_as_of_date):
    """Condition AO: Margin review links margin signal."""
    sig_marg = make_signal(
        signal_id="SIG_MARG",
        signal_type="LOW_MARGIN_HIGH_INVENTORY",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_AO",
        warehouse_id="WH_01",
        description="Margin compression",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig_marg], as_of_date=sample_as_of_date)
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.MARGIN_REVIEW
    assert "SIG_MARG" in rec.supporting_signal_ids


def test_condition_ap_forecast_linkage(sample_config, sample_as_of_date):
    """Condition AP: Forecast review links forecast coverage signal."""
    sig_fc = make_signal(
        signal_id="SIG_FC_LINK",
        signal_type="FORECASTED_DEMAND_ABOVE_AVAILABLE_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_AP",
        warehouse_id="WH_01",
        description="Forecasted demand above stock",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig_fc], as_of_date=sample_as_of_date)
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.FORECAST_REVIEW
    assert "SIG_FC_LINK" in rec.supporting_signal_ids


def test_condition_aq_missing_supplier(sample_config, sample_as_of_date):
    """Condition AQ: Missing supplier data handled gracefully."""
    sig_sup = make_signal(
        signal_id="SIG_NO_SUP",
        signal_type="MISSING_SUPPLIER",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        dimension=CrossDomainDimension.SKU.value,
        sku_id="SKU_AQ",
        description="SKU without preferred supplier",
        observed_metrics={"missing_field": "supplier_id"},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig_sup], as_of_date=sample_as_of_date)
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.DATA_QUALITY_REVIEW


def test_condition_ar_missing_unit_cost(sample_config, sample_as_of_date):
    """Condition AR: Missing unit cost handled gracefully."""
    sig_cost = make_signal(
        signal_id="SIG_NO_COST",
        signal_type="MISSING_UNIT_COST",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension=CrossDomainDimension.SKU.value,
        sku_id="SKU_AR",
        description="Missing unit cost in products master",
        observed_metrics={"missing_field": "unit_cost"},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig_cost], as_of_date=sample_as_of_date)
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.DATA_QUALITY_REVIEW


def test_condition_as_missing_forecast(sample_config, sample_as_of_date):
    """Condition AS: Missing forecast handled gracefully."""
    sig_fc = make_signal(
        signal_id="SIG_NO_FC",
        signal_type="MISSING_FORECAST",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.LOW,
        dimension=CrossDomainDimension.SKU.value,
        sku_id="SKU_AS",
        description="Missing forecast for SKU",
        observed_metrics={"missing_field": "forecast"},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig_fc], as_of_date=sample_as_of_date)
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.DATA_QUALITY_REVIEW


def test_condition_at_missing_inventory(sample_config, sample_as_of_date):
    """Condition AT: Missing inventory handled gracefully."""
    sig_inv = make_signal(
        signal_id="SIG_NO_INV",
        signal_type="MISSING_INVENTORY",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension=CrossDomainDimension.SKU.value,
        sku_id="SKU_AT",
        description="Missing inventory record",
        observed_metrics={"missing_field": "inventory_snapshot"},
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig_inv], as_of_date=sample_as_of_date)
    rec = res.recommendations[0]
    assert rec.recommendation_type == RecommendationType.DATA_QUALITY_REVIEW


def test_condition_au_duplicate_signals(sample_config, sample_as_of_date):
    """Condition AU: Duplicate signals deduplicated without multiplying recommendations."""
    sig = make_signal(
        signal_id="SIG_DUP_01",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_AU",
        warehouse_id="WH_01",
        description="Duplicate test",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig, sig], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 1


def test_condition_av_multiple_signals_same_entity(sample_config, sample_as_of_date):
    """Condition AV: Multiple distinct signals for same entity processed cleanly."""
    sig1 = make_signal(
        signal_id="SIG_AV_1",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_AV",
        warehouse_id="WH_01",
        description="Low demand",
        as_of_date=sample_as_of_date.isoformat(),
    )
    sig2 = make_signal(
        signal_id="SIG_AV_2",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_AV",
        warehouse_id="WH_01",
        description="High capital",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig1, sig2], as_of_date=sample_as_of_date)
    assert res.total_recommendations == 2
    types = {r.recommendation_type for r in res.recommendations}
    assert RecommendationType.SLOW_MOVING_INVENTORY_REVIEW in types
    assert RecommendationType.HIGH_VALUE_INVENTORY_REVIEW in types


def test_condition_aw_no_autonomous_execution(sample_replenishment, sample_config, sample_as_of_date):
    """Condition AW: Zero autonomous execution; status is strictly DRAFT."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    for rec in res.recommendations:
        assert rec.status == RecommendationStatus.DRAFT
        assert rec.execution_allowed is False


def test_condition_ax_no_po_creation(sample_po_proposal, sample_config, sample_as_of_date):
    """Condition AX: Purchase order review does not submit or create external POs."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        purchase_order_proposals=[sample_po_proposal],
        as_of_date=sample_as_of_date,
    )
    rec = res.recommendations[0]
    assert rec.status == RecommendationStatus.DRAFT
    assert "draft" in rec.action.lower()


def test_condition_ay_no_transfer_execution(sample_warehouse_transfer, sample_config, sample_as_of_date):
    """Condition AY: Warehouse rebalancing review does not execute transfers."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        warehouse_transfers=[sample_warehouse_transfer],
        as_of_date=sample_as_of_date,
    )
    rec = res.recommendations[0]
    assert rec.status == RecommendationStatus.DRAFT
    assert rec.execution_allowed is False


def test_condition_az_no_pricing_recommendation(sample_config, sample_as_of_date):
    """Condition AZ: No pricing or discount recommendations generated."""
    sig_marg = make_signal(
        signal_id="SIG_AZ",
        signal_type="LOW_MARGIN_HIGH_INVENTORY",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        sku_id="SKU_AZ",
        warehouse_id="WH_01",
        description="Margin compression",
        as_of_date=sample_as_of_date.isoformat(),
    )
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(signals=[sig_marg], as_of_date=sample_as_of_date)
    for rec in res.recommendations:
        assert "discount" not in rec.action.lower()
        assert "price reduction" not in rec.action.lower()
        assert "markdown" not in rec.action.lower()


def test_condition_ba_no_causal_wording():
    """Condition BA: Summaries and rationales use non-causal language."""
    for rule in ALL_RULES:
        desc_lower = rule.description.lower()
        assert "caused by" not in desc_lower
        assert "definitely caused" not in desc_lower


def test_condition_bb_no_ranking(sample_replenishment, sample_config, sample_as_of_date):
    """Condition BB: Result bundle contains aggregate metrics only, no top-10 ranking."""
    engine = BusinessRecommendationEngine(config=sample_config)
    res = engine.generate_recommendations(
        replenishment_recommendations=[sample_replenishment],
        as_of_date=sample_as_of_date,
    )
    d = res.model_dump()
    assert "top_10" not in d
    assert "rank" not in d
    assert "opportunities_ranked" not in d


def test_condition_bc_phase_6e_regression():
    """Condition BC: Phase 6E cross-domain signals continue to function correctly."""
    sig = CrossDomainSignal(
        signal_id="SIG_6E_REG",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension=CrossDomainDimension.SKU_WAREHOUSE.value,
        sku_id="SKU_6E",
        warehouse_id="WH_01",
        description="High revenue SKU with critically low stock",
        as_of_date="2026-06-30",
    )
    assert sig.signal_id == "SIG_6E_REG"
    assert sig.severity == SignalSeverity.HIGH


def test_condition_bd_phase_6f_regression():
    """Condition BD: Phase 6F business impact records continue to function correctly."""
    imp = BusinessImpactRecord(
        impact_id="IMP_6F_REG",
        signal_id="SIG_6E_REG",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        impact_category=ImpactCategory.STOCKOUT,
        impact_type=ImpactType.STOCKOUT_REVENUE_EXPOSURE,
        severity=SignalSeverity.HIGH,
        sku_id="SKU_6E",
        warehouse_id="WH_01",
        exposure_value=4500.0,
        calculation_status=CalculationStatus.CALCULATED,
        confidence_status=ImpactConfidence.DIRECT_OBSERVED,
        calculation_method="direct_calculation",
        description="Stockout lost sales",
        as_of_date="2026-06-30",
    )
    assert imp.impact_id == "IMP_6F_REG"
    assert imp.exposure_value == 4500.0
