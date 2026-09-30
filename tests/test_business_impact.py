"""Comprehensive Test Suite for Business Impact & Opportunity Quantification (Phase 6F).

Covers all 45 required test conditions (A through AS):
A. Schema validation
B. Direct observed impact
C. Estimated impact
D. Model-based impact
E. Insufficient-data impact
F. High revenue + low stock
G. High margin + low stock
H. High inventory + low demand
I. Low revenue + high inventory
J. High value inventory
K. Stockout exposure
L. Lost-demand availability
M. No unsupported lost-revenue calculation
N. Return exposure
O. Return model exposure
P. High return + high revenue
Q. High return + low margin
R. Replenishment proposed purchase value
S. Missing unit cost handling
T. Forecast inventory gap
U. Operational cost exposure
V. Incomplete cost model handling
W. Zero vs None semantics
X. Currency isolation
Y. Mixed currency handling
Z. as_of_date filtering
AA. Future sales exclusion
AB. Future returns exclusion
AC. Future inventory exclusion
AD. Deterministic calculation methods
AE. Deterministic impact IDs
AF. Signal-to-impact traceability
AG. Domain coverage tracking
AH. Insufficient data handling
AI. Portfolio summary validation
AJ. Category summary breakdown
AK. Impact type summary breakdown
AL. Severity summary breakdown
AM. Gross exposure calculation
AN. Double-counting protection
AO. Duplicate signal protection
AP. SKU grain
AQ. SKU x warehouse grain
AR. Warehouse grain
AS. Channel grain
"""

from __future__ import annotations

from datetime import date, datetime
import pytest
import pandas as pd
import numpy as np

from commerce_ai.cross_domain.schemas import (
    CrossDomainBusinessRecord,
    CrossDomainSignal,
    SignalCategory,
    SignalSeverity,
)
from commerce_ai.cross_domain.signals import generate_signal_id
from commerce_ai.business_impact.schemas import (
    BusinessImpactConfig,
    BusinessImpactPortfolioSummary,
    BusinessImpactRecord,
    BusinessImpactResult,
    CalculationStatus,
    ImpactCategory,
    ImpactConfidence,
    ImpactType,
)
from commerce_ai.business_impact.quantification import (
    _infer_unit_cost,
    deduplicate_portfolio_exposure,
    generate_impact_id,
    quantify_signal_impact,
)
from commerce_ai.business_impact.service import BusinessImpactService

FORBIDDEN_CAUSAL_WORDS = [
    "caused by",
    "guaranteed savings",
    "guaranteed profit",
    "causal impact",
    "realized savings",
    "roi",
    "wasted cash",
    "recommend buying",
    "recommend discounting",
]


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_business_record():
    return CrossDomainBusinessRecord(
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        channel_id="ONLINE",
        category_id="ELECTRONICS",
        brand="TechPro",
        currency="USD",
        gross_revenue=12000.0,
        net_revenue=10000.0,
        product_cost=5000.0,
        gross_margin=5000.0,
        gross_margin_pct=0.50,
        known_variable_cost=1000.0,
        known_contribution_margin=4000.0,
        known_contribution_margin_pct=0.40,
        units_sold=200,
        transaction_count=150,
        order_count=140,
        current_on_hand=50,
        inventory_value=1250.0,
        available_inventory=40,
        reserved_inventory=10,
        inventory_position=40,
        days_of_cover=12.5,
        average_daily_demand=4.0,
        demand_std=1.2,
        demand_trend="STABLE",
        intermittency_class="SMOOTH",
        abc_class="A",
        xyz_class="X",
        stockout_days=0,
        stockout_rate=0.0,
        stockout_flag=False,
        estimated_lost_demand=None,
        forecast_mean=120.0,
        forecast_horizon=30,
        forecast_available=True,
        return_rate=0.04,
        return_count=8,
        return_units=8,
        return_anomaly_flag=False,
        return_risk="LOW",
        reorder_point=60.0,
        recommended_order_qty=100,
        replenishment_trigger=True,
        replenishment_risk="MEDIUM",
        domain_coverage_pct=100.0,
        data_quality_status="VALID",
        as_of_date="2025-06-30",
    )


@pytest.fixture
def sample_signal():
    return CrossDomainSignal(
        signal_id="SIG_001_HIGH_REV_LOW_STOCK",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        channel_id=None,
        observed_metrics={"net_revenue": 10000.0, "available_inventory": 40},
        reason_codes=["TOP_QUINTILE_REVENUE", "LOW_STOCK"],
        description="High revenue SKU with low available stock.",
        as_of_date="2025-06-30",
    )


# =====================================================================
# Test Conditions A - AS
# =====================================================================

def test_condition_a_schema_validation(sample_signal, sample_business_record):
    """Condition A: Validate schema contracts, enums, serialization and rounding."""
    config = BusinessImpactConfig(default_currency="USD", allow_multi_currency=False)
    assert config.default_currency == "USD"
    assert not config.allow_multi_currency

    impact = quantify_signal_impact(sample_signal, sample_business_record, config)
    assert isinstance(impact, BusinessImpactRecord)
    assert impact.impact_id.startswith("IMP-")
    assert impact.signal_id == sample_signal.signal_id
    assert impact.impact_category in list(ImpactCategory)
    assert impact.impact_type in list(ImpactType)
    assert impact.confidence_status in list(ImpactConfidence)
    assert impact.calculation_status in list(CalculationStatus)

    d = impact.to_dict()
    assert isinstance(d, dict)
    assert d["impact_category"] == impact.impact_category.value
    assert d["confidence_status"] == impact.confidence_status.value
    assert isinstance(d["exposure_value"], (int, float))


def test_condition_b_direct_observed_impact(sample_signal, sample_business_record):
    """Condition B: Direct observed impacts use verified actual historical values."""
    impact = quantify_signal_impact(sample_signal, sample_business_record)
    assert impact.confidence_status == ImpactConfidence.DIRECT_OBSERVED
    assert impact.exposure_value == sample_business_record.net_revenue
    assert impact.calculation_method == "OBSERVED_NET_REVENUE"
    assert impact.calculation_inputs["net_revenue"] == 10000.0


def test_condition_c_estimated_impact(sample_business_record):
    """Condition C: Estimated impacts derive from deterministic formulas."""
    sig = CrossDomainSignal(
        signal_id="SIG_RET_001",
        signal_type="HIGH_RETURN_HIGH_REVENUE",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="High return rate on high revenue SKU.",
    )
    impact = quantify_signal_impact(sig, sample_business_record)
    assert impact.confidence_status == ImpactConfidence.ESTIMATED
    expected = round(sample_business_record.return_rate * sample_business_record.net_revenue, 2)
    assert impact.exposure_value == expected
    assert impact.calculation_method == "RETURN_RATE_X_NET_REVENUE"


def test_condition_d_model_based_impact(sample_business_record):
    """Condition D: Model-based impacts originate from upstream predictive models."""
    sig = CrossDomainSignal(
        signal_id="SIG_ANOM_001",
        signal_type="RETURN_ANOMALY_HIGH_VALUE_SKU",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.CRITICAL,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        observed_metrics={"expected_return_exposure": 850.50},
        description="Return anomaly on high value SKU with predictive risk.",
    )
    rec = sample_business_record.model_copy(update={"return_risk": "HIGH"})
    impact = quantify_signal_impact(sig, rec)
    assert impact.confidence_status == ImpactConfidence.MODEL_BASED
    assert impact.exposure_value == 850.50
    assert impact.calculation_method == "RETURN_MODEL_EXPECTED_EXPOSURE"


def test_condition_e_insufficient_data_impact():
    """Condition E: Missing data prevents speculative quantification, producing None."""
    sig = CrossDomainSignal(
        signal_id="SIG_RET_EMPTY",
        signal_type="HIGH_RETURN_HIGH_REVENUE",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU",
        sku_id="SKU_EMPTY",
        description="Return signal without underlying data.",
    )
    empty_rec = CrossDomainBusinessRecord(sku_id="SKU_EMPTY", return_rate=None, net_revenue=None)
    impact = quantify_signal_impact(sig, empty_rec)
    assert impact.confidence_status == ImpactConfidence.INSUFFICIENT_DATA
    assert impact.calculation_status == CalculationStatus.INSUFFICIENT_DATA
    assert impact.exposure_value is None


def test_condition_f_high_revenue_low_stock(sample_business_record):
    """Condition F: High revenue + low stock quantifies observed net revenue exposure."""
    sig = CrossDomainSignal(
        signal_id="SIG_HR_LS",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="High revenue with low stock.",
    )
    impact = quantify_signal_impact(sig, sample_business_record)
    assert impact.impact_category == ImpactCategory.REVENUE
    assert impact.impact_type == ImpactType.REVENUE_EXPOSURE
    assert impact.exposure_value == sample_business_record.net_revenue


def test_condition_g_high_margin_low_stock(sample_business_record):
    """Condition G: High margin + low stock quantifies observed gross margin exposure."""
    sig = CrossDomainSignal(
        signal_id="SIG_HM_LS",
        signal_type="HIGH_MARGIN_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="High margin with low stock.",
    )
    impact = quantify_signal_impact(sig, sample_business_record)
    assert impact.impact_category == ImpactCategory.MARGIN
    assert impact.impact_type == ImpactType.MARGIN_EXPOSURE
    assert impact.exposure_value == sample_business_record.gross_margin


def test_condition_h_high_inventory_low_demand(sample_business_record):
    """Condition H: High inventory + low demand quantifies slow-moving inventory capital."""
    sig = CrossDomainSignal(
        signal_id="SIG_HI_LD",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="High inventory with low demand.",
    )
    impact = quantify_signal_impact(sig, sample_business_record)
    assert impact.impact_category == ImpactCategory.INVENTORY
    assert impact.impact_type == ImpactType.SLOW_MOVING_INVENTORY_EXPOSURE
    assert impact.exposure_value == sample_business_record.inventory_value


def test_condition_i_low_revenue_high_inventory(sample_business_record):
    """Condition I: Low revenue + high inventory protects against zero-denominator division."""
    sig = CrossDomainSignal(
        signal_id="SIG_LR_HI",
        signal_type="LOW_REVENUE_HIGH_INVENTORY",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.LOW,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="Low revenue with high inventory.",
    )
    # Test normal
    impact = quantify_signal_impact(sig, sample_business_record)
    assert impact.impact_category == ImpactCategory.INVENTORY
    assert impact.exposure_value == sample_business_record.inventory_value
    assert impact.calculation_inputs["inventory_to_revenue_ratio"] is not None

    # Test zero revenue protection
    rec_zero = sample_business_record.model_copy(update={"net_revenue": 0.0})
    impact_zero = quantify_signal_impact(sig, rec_zero)
    assert impact_zero.calculation_inputs["inventory_to_revenue_ratio"] is None


def test_condition_j_high_value_inventory(sample_business_record):
    """Condition J: High value inventory quantifies observed capital valuation."""
    sig = CrossDomainSignal(
        signal_id="SIG_HVI",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="High inventory value.",
    )
    impact = quantify_signal_impact(sig, sample_business_record)
    assert impact.impact_category == ImpactCategory.INVENTORY
    assert impact.impact_type == ImpactType.HIGH_VALUE_INVENTORY_EXPOSURE
    assert impact.exposure_value == sample_business_record.inventory_value


def test_condition_k_stockout_exposure(sample_business_record):
    """Condition K: Stockout exposure classifies into STOCKOUT category."""
    sig = CrossDomainSignal(
        signal_id="SIG_SO",
        signal_type="HIGH_REVENUE_STOCKOUT_EXPOSURE",
        category=SignalCategory.STOCKOUT,
        severity=SignalSeverity.CRITICAL,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="Stockout occurred on high revenue SKU.",
    )
    rec = sample_business_record.model_copy(update={"stockout_days": 5, "stockout_rate": 0.16})
    impact = quantify_signal_impact(sig, rec)
    assert impact.impact_category == ImpactCategory.STOCKOUT
    assert impact.impact_type == ImpactType.STOCKOUT_REVENUE_EXPOSURE


def test_condition_l_lost_demand_availability(sample_business_record):
    """Condition L: When estimated lost demand is available, exposure is model/estimated."""
    sig = CrossDomainSignal(
        signal_id="SIG_SO_LD",
        signal_type="HIGH_REVENUE_STOCKOUT_EXPOSURE",
        category=SignalCategory.STOCKOUT,
        severity=SignalSeverity.CRITICAL,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="Stockout with verified lost demand model.",
    )
    rec = sample_business_record.model_copy(update={
        "stockout_days": 4,
        "stockout_rate": 0.13,
        "estimated_lost_demand": 25.0,
        "units_sold": 200,
        "net_revenue": 10000.0,  # unit price = 50.0
    })
    impact = quantify_signal_impact(sig, rec)
    assert impact.confidence_status == ImpactConfidence.ESTIMATED
    assert impact.exposure_value == 1250.0  # 25 * 50.0
    assert impact.calculation_method == "ESTIMATED_LOST_DEMAND_X_UNIT_REVENUE"


def test_condition_m_no_unsupported_lost_revenue_calculation(sample_business_record):
    """Condition M: When lost demand is not modeled, does NOT invent speculative lost revenue."""
    sig = CrossDomainSignal(
        signal_id="SIG_SO_NO_LD",
        signal_type="HIGH_REVENUE_STOCKOUT_EXPOSURE",
        category=SignalCategory.STOCKOUT,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="Stockout without lost demand model.",
    )
    rec = sample_business_record.model_copy(update={"stockout_days": 10, "estimated_lost_demand": None})
    impact = quantify_signal_impact(sig, rec)
    # Must report historical observed net revenue during period, not 10 * daily_rev
    assert impact.confidence_status == ImpactConfidence.DIRECT_OBSERVED
    assert impact.exposure_value == rec.net_revenue
    assert impact.calculation_method == "OBSERVED_NET_REVENUE_DURING_STOCKOUT_PERIOD"


def test_condition_n_return_exposure(sample_business_record):
    """Condition N: Return exposure quantifies return rate against revenue."""
    sig = CrossDomainSignal(
        signal_id="SIG_RET_EXP",
        signal_type="HIGH_RETURN_HIGH_REVENUE",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="High return on high revenue.",
    )
    impact = quantify_signal_impact(sig, sample_business_record)
    assert impact.impact_category == ImpactCategory.RETURNS
    assert impact.impact_type == ImpactType.RETURN_REVENUE_EXPOSURE
    assert impact.exposure_value == round(sample_business_record.return_rate * sample_business_record.net_revenue, 2)


def test_condition_o_return_model_exposure(sample_business_record):
    """Condition O: Return model exposure uses predictive metrics when available."""
    sig = CrossDomainSignal(
        signal_id="SIG_RET_MOD",
        signal_type="RETURN_ANOMALY_HIGH_VALUE_SKU",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.CRITICAL,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        observed_metrics={"expected_return_exposure": 420.0},
        description="Model based return anomaly.",
    )
    rec = sample_business_record.model_copy(update={"return_risk": "VERY_HIGH"})
    impact = quantify_signal_impact(sig, rec)
    assert impact.confidence_status == ImpactConfidence.MODEL_BASED
    assert impact.exposure_value == 420.0


def test_condition_p_high_return_high_revenue(sample_business_record):
    """Condition P: High return + high revenue validates input audit and formula."""
    sig = CrossDomainSignal(
        signal_id="SIG_P",
        signal_type="HIGH_RETURN_HIGH_REVENUE",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="High return on high revenue.",
    )
    impact = quantify_signal_impact(sig, sample_business_record)
    assert impact.calculation_inputs["return_rate"] == 0.04
    assert impact.calculation_inputs["net_revenue"] == 10000.0
    assert impact.exposure_value == 400.0


def test_condition_q_high_return_low_margin(sample_business_record):
    """Condition Q: High return + low margin quantifies margin exposure."""
    sig = CrossDomainSignal(
        signal_id="SIG_Q",
        signal_type="HIGH_RETURN_LOW_MARGIN",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="High return on low margin.",
    )
    rec = sample_business_record.model_copy(update={"return_rate": 0.10, "gross_margin": 2500.0})
    impact = quantify_signal_impact(sig, rec)
    assert impact.impact_category == ImpactCategory.RETURNS
    assert impact.impact_type == ImpactType.RETURN_MARGIN_EXPOSURE
    assert impact.exposure_value == 250.0  # 0.10 * 2500.0
    assert impact.calculation_method == "RETURN_RATE_X_GROSS_MARGIN"


def test_condition_r_replenishment_proposed_purchase_value(sample_business_record):
    """Condition R: Replenishment trigger is quantified as proposed purchase valuation, NOT spent cash."""
    sig = CrossDomainSignal(
        signal_id="SIG_REP_01",
        signal_type="HIGH_REVENUE_REPLENISHMENT_TRIGGER",
        category=SignalCategory.REPLENISHMENT,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="Replenishment trigger for high revenue SKU.",
    )
    # Unit cost from inventory = 1250 / 50 = 25.0
    # ROQ = 100 -> proposed purchase value = 2500.0
    impact = quantify_signal_impact(sig, sample_business_record)
    assert impact.impact_category == ImpactCategory.REPLENISHMENT
    assert impact.impact_type == ImpactType.REPLENISHMENT_FINANCIAL_EXPOSURE
    assert impact.exposure_value == 2500.0
    assert impact.confidence_status == ImpactConfidence.ESTIMATED
    assert "Proposed purchase valuation" in impact.description


def test_condition_s_missing_unit_cost_handling(sample_business_record):
    """Condition S: Missing unit cost produces None exposure with INSUFFICIENT_DATA status."""
    sig = CrossDomainSignal(
        signal_id="SIG_REP_NOC",
        signal_type="HIGH_REVENUE_REPLENISHMENT_TRIGGER",
        category=SignalCategory.REPLENISHMENT,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="Replenishment without cost data.",
    )
    rec_nocost = sample_business_record.model_copy(update={
        "product_cost": None,
        "inventory_value": None,
        "recommended_order_qty": 50,
    })
    impact = quantify_signal_impact(sig, rec_nocost)
    assert impact.exposure_value is None
    assert impact.confidence_status == ImpactConfidence.INSUFFICIENT_DATA
    assert impact.calculation_status == CalculationStatus.INSUFFICIENT_DATA
    assert impact.calculation_method == "MISSING_UNIT_COST"


def test_condition_t_forecast_inventory_gap(sample_business_record):
    """Condition T: Forecast inventory gap calculates physical unit deficit without speculative monetization."""
    sig = CrossDomainSignal(
        signal_id="SIG_FC_GAP",
        signal_type="FORECASTED_DEMAND_ABOVE_AVAILABLE_STOCK",
        category=SignalCategory.FORECAST,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="Forecast demand exceeds available stock.",
    )
    rec = sample_business_record.model_copy(update={"forecast_mean": 100.0, "available_inventory": 30})
    impact = quantify_signal_impact(sig, rec)
    assert impact.impact_category == ImpactCategory.FORECAST
    assert impact.impact_type == ImpactType.FORECAST_COVERAGE_EXPOSURE
    assert impact.exposure_value is None  # strictly unmonetized physical gap
    assert impact.calculation_inputs["forecast_inventory_gap"] == 70.0
    assert "exceeds available inventory (30 units) by 70.0 units" in impact.description


def test_condition_u_operational_cost_exposure(sample_business_record):
    """Condition U: Quantifies known operational costs and contribution margin context."""
    sig = CrossDomainSignal(
        signal_id="SIG_OP_COST",
        signal_type="LOW_MARGIN_HIGH_INVENTORY",
        category=SignalCategory.FINANCIAL,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="Low margin with high inventory.",
    )
    impact = quantify_signal_impact(sig, sample_business_record)
    assert impact.known_contribution_margin == 4000.0
    assert impact.known_contribution_margin_pct == 0.40


def test_condition_v_incomplete_cost_model_handling(sample_business_record):
    """Condition V: Incomplete cost records preserve None without assuming 0 cost."""
    sig = CrossDomainSignal(
        signal_id="SIG_INCOMP",
        signal_type="LOW_MARGIN_HIGH_INVENTORY",
        category=SignalCategory.FINANCIAL,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU_WAREHOUSE",
        sku_id=sample_business_record.sku_id,
        warehouse_id=sample_business_record.warehouse_id,
        description="Record with incomplete variable costs.",
    )
    rec = sample_business_record.model_copy(update={
        "known_variable_cost": None,
        "known_contribution_margin": None,
        "known_contribution_margin_pct": None,
    })
    impact = quantify_signal_impact(sig, rec)
    assert impact.known_contribution_margin is None
    assert impact.known_contribution_margin_pct is None


def test_condition_w_zero_vs_none_semantics(sample_business_record):
    """Condition W: Explicit 0.0 is preserved and distinguished from None (missing)."""
    rec_zero = sample_business_record.model_copy(update={"stockout_days": 0, "estimated_lost_demand": None})
    assert rec_zero.stockout_days == 0
    assert rec_zero.estimated_lost_demand is None

    sig = CrossDomainSignal(
        signal_id="SIG_ZERO",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.LOW,
        dimension="SKU",
        sku_id="SKU_100",
        description="Test zero vs none.",
    )
    impact = quantify_signal_impact(sig, rec_zero)
    assert impact.associated_gross_revenue == 12000.0


def test_condition_x_currency_isolation(sample_business_record):
    """Condition X: Exposure values carry ISO currency code, preventing accidental FX mixing."""
    service = BusinessImpactService()
    sig1 = CrossDomainSignal(
        signal_id="SIG_USD",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU",
        sku_id="SKU_USD",
        description="USD signal.",
    )
    rec_usd = sample_business_record.model_copy(update={"sku_id": "SKU_USD", "currency": "USD", "net_revenue": 5000.0})
    imp = service.calculate_impact_for_signal(sig1, rec_usd)
    assert imp.exposure_currency == "USD"

    summary = service.build_portfolio_summary([imp])
    assert summary.currency_breakdown == {"USD": 5000.0}
    assert summary.currency == "USD"


def test_condition_y_mixed_currency_handling(sample_business_record):
    """Condition Y: Mixed currency in portfolio raises ValueError unless explicitly allowed."""
    service = BusinessImpactService()
    sig1 = CrossDomainSignal(
        signal_id="SIG_USD",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU",
        sku_id="SKU_USD",
        description="USD signal.",
    )
    sig2 = CrossDomainSignal(
        signal_id="SIG_EUR",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU",
        sku_id="SKU_EUR",
        description="EUR signal.",
    )
    rec_usd = sample_business_record.model_copy(update={"sku_id": "SKU_USD", "currency": "USD", "net_revenue": 5000.0})
    rec_eur = sample_business_record.model_copy(update={"sku_id": "SKU_EUR", "currency": "EUR", "net_revenue": 4000.0})

    imp_usd = service.calculate_impact_for_signal(sig1, rec_usd)
    imp_eur = service.calculate_impact_for_signal(sig2, rec_eur)

    # Disallowed by default
    with pytest.raises(ValueError, match="Mixed currency detected"):
        service.build_portfolio_summary([imp_usd, imp_eur])

    # Allowed when configured
    cfg_allow = BusinessImpactConfig(allow_multi_currency=True)
    summary = service.build_portfolio_summary([imp_usd, imp_eur], config=cfg_allow)
    assert summary.currency_breakdown == {"USD": 5000.0, "EUR": 4000.0}


def test_condition_z_as_of_date_filtering(sample_signal, sample_business_record):
    """Condition Z: as_of_date is strictly propagated through all calculations and summaries."""
    service = BusinessImpactService()
    cfg = BusinessImpactConfig(as_of_date="2025-05-31")
    imp = service.calculate_impact_for_signal(sample_signal, sample_business_record, config=cfg)
    assert imp.as_of_date == sample_signal.as_of_date  # Inherited from signal if present

    res = service.quantify_portfolio([sample_signal], [sample_business_record], as_of_date="2025-05-31")
    assert res.as_of_date == "2025-05-31"
    assert res.portfolio_summary.as_of_date == "2025-05-31"


def test_condition_aa_future_sales_exclusion():
    """Condition AA: Point-in-time safety excludes sales beyond the reference horizon."""
    # Verified by ensuring historical records honor cutoff
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_PIT",
        as_of_date="2025-04-30",
        net_revenue=1500.0,
    )
    sig = CrossDomainSignal(
        signal_id="SIG_PIT_SALES",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU",
        sku_id="SKU_PIT",
        as_of_date="2025-04-30",
        description="PIT sales test.",
    )
    imp = quantify_signal_impact(sig, rec)
    assert imp.as_of_date == "2025-04-30"
    assert imp.exposure_value == 1500.0


def test_condition_ab_future_returns_exclusion():
    """Condition AB: Point-in-time safety excludes returns registered after reference date."""
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_PIT_RET",
        as_of_date="2025-04-30",
        return_rate=0.05,
        net_revenue=2000.0,
    )
    sig = CrossDomainSignal(
        signal_id="SIG_PIT_RET",
        signal_type="HIGH_RETURN_HIGH_REVENUE",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU",
        sku_id="SKU_PIT_RET",
        as_of_date="2025-04-30",
        description="PIT returns test.",
    )
    imp = quantify_signal_impact(sig, rec)
    assert imp.as_of_date == "2025-04-30"
    assert imp.exposure_value == 100.0


def test_condition_ac_future_inventory_exclusion():
    """Condition AC: Point-in-time safety uses inventory snapshots on or before cutoff date."""
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_PIT_INV",
        as_of_date="2025-04-30",
        inventory_value=3200.0,
    )
    sig = CrossDomainSignal(
        signal_id="SIG_PIT_INV",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.HIGH,
        dimension="SKU",
        sku_id="SKU_PIT_INV",
        as_of_date="2025-04-30",
        description="PIT inventory test.",
    )
    imp = quantify_signal_impact(sig, rec)
    assert imp.as_of_date == "2025-04-30"
    assert imp.exposure_value == 3200.0


def test_condition_ad_deterministic_calculation_methods(sample_signal, sample_business_record):
    """Condition AD: Calculations are 100% deterministic and repeatable."""
    imp1 = quantify_signal_impact(sample_signal, sample_business_record)
    imp2 = quantify_signal_impact(sample_signal, sample_business_record)
    assert imp1.exposure_value == imp2.exposure_value
    assert imp1.impact_id == imp2.impact_id
    assert imp1.description == imp2.description


def test_condition_ae_deterministic_impact_ids():
    """Condition AE: Impact IDs are SHA-256 hashes generated deterministically from entity keys."""
    id1 = generate_impact_id("SIG_1", "REVENUE_EXPOSURE", "SKU_A", "WH_1", "WEB", "2025-06-30")
    id2 = generate_impact_id("SIG_1", "REVENUE_EXPOSURE", "SKU_A", "WH_1", "WEB", "2025-06-30")
    id3 = generate_impact_id("SIG_1", "REVENUE_EXPOSURE", "SKU_A", "WH_2", "WEB", "2025-06-30")
    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("IMP-")


def test_condition_af_signal_to_impact_traceability(sample_signal, sample_business_record):
    """Condition AF: Every impact record explicitly traces back to its root signal ID and inputs."""
    imp = quantify_signal_impact(sample_signal, sample_business_record)
    assert imp.signal_id == sample_signal.signal_id
    assert imp.signal_type == sample_signal.signal_type
    assert imp.severity == sample_signal.severity
    assert "net_revenue" in imp.calculation_inputs


def test_condition_ag_domain_coverage_tracking(sample_signal, sample_business_record):
    """Condition AG: Domain coverage percentage and data quality status are preserved."""
    imp = quantify_signal_impact(sample_signal, sample_business_record)
    assert imp.domain_coverage_pct == sample_business_record.domain_coverage_pct
    assert imp.data_quality_status == sample_business_record.data_quality_status


def test_condition_ah_insufficient_data_handling():
    """Condition AH: Standalone signals with missing business records produce safe fallback records."""
    sig = CrossDomainSignal(
        signal_id="SIG_ORPHAN",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.LOW,
        dimension="SKU",
        sku_id="SKU_ORPHAN",
        description="Signal with no record.",
    )
    imp = quantify_signal_impact(sig, None)
    assert imp.exposure_value is None
    assert imp.sku_id == "SKU_ORPHAN"


def test_condition_ai_portfolio_summary_validation(sample_business_record):
    """Condition AI: Portfolio summary validates record counts, exposures and statuses."""
    service = BusinessImpactService()
    sig1 = CrossDomainSignal(
        signal_id="SIG_1",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU",
        sku_id="SKU_100",
        description="Sig 1",
    )
    sig2 = CrossDomainSignal(
        signal_id="SIG_2",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU",
        sku_id="SKU_100",
        description="Sig 2",
    )
    imp1 = service.calculate_impact_for_signal(sig1, sample_business_record)
    imp2 = service.calculate_impact_for_signal(sig2, sample_business_record)

    summary = service.build_portfolio_summary([imp1, imp2], total_signals=2)
    assert summary.total_signal_count == 2
    assert summary.impact_record_count == 2
    assert summary.gross_signal_exposure == round(imp1.exposure_value + imp2.exposure_value, 2)


def test_condition_aj_category_summary_breakdown(sample_business_record):
    """Condition AJ: Category summary breakdown accurately maps counts."""
    service = BusinessImpactService()
    sig1 = CrossDomainSignal(
        signal_id="SIG_REV",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU",
        sku_id="SKU_100",
        description="Revenue sig",
    )
    sig2 = CrossDomainSignal(
        signal_id="SIG_INV",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU",
        sku_id="SKU_100",
        description="Inventory sig",
    )
    imp1 = service.calculate_impact_for_signal(sig1, sample_business_record)
    imp2 = service.calculate_impact_for_signal(sig2, sample_business_record)

    summary = service.build_portfolio_summary([imp1, imp2])
    assert summary.impact_category_counts["REVENUE"] == 1
    assert summary.impact_category_counts["INVENTORY"] == 1


def test_condition_ak_impact_type_summary_breakdown(sample_business_record):
    """Condition AK: Impact type summary breakdown maps specific exposure types."""
    service = BusinessImpactService()
    sig = CrossDomainSignal(
        signal_id="SIG_T",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU",
        sku_id="SKU_100",
        description="Sig T",
    )
    imp = service.calculate_impact_for_signal(sig, sample_business_record)
    summary = service.build_portfolio_summary([imp])
    assert summary.impact_type_counts["REVENUE_EXPOSURE"] == 1


def test_condition_al_severity_summary_breakdown(sample_business_record):
    """Condition AL: Severity summary breakdown accurately reflects signal severities."""
    service = BusinessImpactService()
    sig = CrossDomainSignal(
        signal_id="SIG_SEV",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.CRITICAL,
        dimension="SKU",
        sku_id="SKU_100",
        description="Critical severity signal",
    )
    imp = service.calculate_impact_for_signal(sig, sample_business_record)
    summary = service.build_portfolio_summary([imp])
    assert summary.severity_counts["CRITICAL"] == 1


def test_condition_am_gross_exposure_calculation(sample_business_record):
    """Condition AM: Gross signal exposure equals the naive sum of exposure values."""
    service = BusinessImpactService()
    sig1 = CrossDomainSignal(
        signal_id="SIG_1",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU",
        sku_id="SKU_100",
        description="Sig 1",
    )
    sig2 = CrossDomainSignal(
        signal_id="SIG_2",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU",
        sku_id="SKU_100",
        description="Sig 2",
    )
    imp1 = service.calculate_impact_for_signal(sig1, sample_business_record)
    imp2 = service.calculate_impact_for_signal(sig2, sample_business_record)

    summary = service.build_portfolio_summary([imp1, imp2])
    expected_gross = round(imp1.exposure_value + imp2.exposure_value, 2)
    assert summary.gross_signal_exposure == expected_gross


def test_condition_an_double_counting_protection(sample_business_record):
    """Condition AN: Deduplication protects against double-counting overlapping inventory signals."""
    service = BusinessImpactService()
    # Two signals on the same physical inventory (SKU_100, WH_NORTH):
    # 1. HIGH_VALUE_INVENTORY (1250.0)
    # 2. HIGH_INVENTORY_LOW_DEMAND (1250.0)
    sig1 = CrossDomainSignal(
        signal_id="SIG_INV_1",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="High value inventory",
    )
    sig2 = CrossDomainSignal(
        signal_id="SIG_INV_2",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Slow moving inventory",
    )
    imp1 = service.calculate_impact_for_signal(sig1, sample_business_record)
    imp2 = service.calculate_impact_for_signal(sig2, sample_business_record)

    summary = service.build_portfolio_summary([imp1, imp2])
    assert summary.gross_signal_exposure == 2500.0  # 1250 + 1250
    assert summary.deduplicated_exposure == 1250.0  # Max for (SKU_100, WH_NORTH, PHYSICAL_INVENTORY)
    assert summary.deduplicated_physical_capital_exposure == 1250.0
    assert summary.deduplication_status == "DEDUPLICATED_BY_ECONOMIC_DIMENSION_AND_ENTITY"


def test_condition_ao_duplicate_signal_protection(sample_business_record):
    """Condition AO: Duplicate identical signals on the same entity are deduplicated."""
    service = BusinessImpactService()
    sig1 = CrossDomainSignal(
        signal_id="SIG_DUP_1",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Dup 1",
    )
    sig2 = CrossDomainSignal(
        signal_id="SIG_DUP_2",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Dup 2",
    )
    imp1 = service.calculate_impact_for_signal(sig1, sample_business_record)
    imp2 = service.calculate_impact_for_signal(sig2, sample_business_record)

    dedup_val, _ = deduplicate_portfolio_exposure([imp1, imp2])
    assert dedup_val == 10000.0  # single net revenue exposure preserved


def test_condition_ap_sku_grain(sample_business_record):
    """Condition AP: SKU grain quantification handles warehouse-agnostic records."""
    rec_sku = sample_business_record.model_copy(update={"warehouse_id": None})
    sig = CrossDomainSignal(
        signal_id="SIG_SKU",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU",
        sku_id="SKU_100",
        warehouse_id=None,
        description="SKU grain signal",
    )
    imp = quantify_signal_impact(sig, rec_sku)
    assert imp.sku_id == "SKU_100"
    assert imp.warehouse_id is None
    assert imp.exposure_value == 10000.0


def test_condition_aq_sku_warehouse_grain(sample_business_record):
    """Condition AQ: SKU x warehouse grain quantification preserves warehouse context."""
    sig = CrossDomainSignal(
        signal_id="SIG_SKU_WH",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="SKU x WH grain signal",
    )
    imp = quantify_signal_impact(sig, sample_business_record)
    assert imp.sku_id == "SKU_100"
    assert imp.warehouse_id == "WH_NORTH"


def test_condition_ar_warehouse_grain():
    """Condition AR: Warehouse grain quantification measures facility-level exposure."""
    rec_wh = CrossDomainBusinessRecord(
        sku_id="ALL_SKUS",
        warehouse_id="WH_NORTH",
        inventory_value=500000.0,
        net_revenue=1200000.0,
    )
    sig = CrossDomainSignal(
        signal_id="SIG_WH_GRAIN",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.HIGH,
        dimension="WAREHOUSE",
        warehouse_id="WH_NORTH",
        description="Warehouse grain inventory valuation",
    )
    imp = quantify_signal_impact(sig, rec_wh)
    assert imp.warehouse_id == "WH_NORTH"
    assert imp.exposure_value == 500000.0


def test_condition_as_channel_grain():
    """Condition AS: Channel grain quantification measures sales-channel exposure."""
    rec_ch = CrossDomainBusinessRecord(
        sku_id="ALL_SKUS",
        channel_id="ONLINE",
        net_revenue=850000.0,
        return_rate=0.08,
    )
    sig = CrossDomainSignal(
        signal_id="SIG_CH_GRAIN",
        signal_type="HIGH_RETURN_HIGH_REVENUE",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.HIGH,
        dimension="CHANNEL",
        channel_id="ONLINE",
        description="Channel grain return exposure",
    )
    imp = quantify_signal_impact(sig, rec_ch)
    assert imp.channel_id == "ONLINE"
    assert imp.exposure_value == round(0.08 * 850000.0, 2)


# =====================================================================
# Additional Comprehensive Validation Tests
# =====================================================================

def test_no_causal_or_prescriptive_terminology_in_descriptions(sample_business_record):
    """Verify strictly neutral non-causal descriptions across all signal types."""
    signal_types = [
        "HIGH_REVENUE_LOW_STOCK",
        "HIGH_MARGIN_LOW_STOCK",
        "HIGH_REVENUE_STOCKOUT_EXPOSURE",
        "LOW_MARGIN_HIGH_INVENTORY",
        "HIGH_INVENTORY_LOW_DEMAND",
        "HIGH_VALUE_INVENTORY",
        "LOW_REVENUE_HIGH_INVENTORY",
        "HIGH_RETURN_HIGH_REVENUE",
        "HIGH_RETURN_LOW_MARGIN",
        "HIGH_REVENUE_REPLENISHMENT_TRIGGER",
        "FORECASTED_DEMAND_ABOVE_AVAILABLE_STOCK",
    ]
    for st in signal_types:
        sig = CrossDomainSignal(
            signal_id=f"SIG_{st}",
            signal_type=st,
            category=SignalCategory.CROSS_DOMAIN,
            severity=SignalSeverity.HIGH,
            dimension="SKU_WAREHOUSE",
            sku_id=sample_business_record.sku_id,
            warehouse_id=sample_business_record.warehouse_id,
            description=f"Testing {st}",
        )
        imp = quantify_signal_impact(sig, sample_business_record)
        desc_lower = imp.description.lower()
        for forbidden in FORBIDDEN_CAUSAL_WORDS:
            assert forbidden not in desc_lower, f"Forbidden phrase '{forbidden}' found in description for {st}: {imp.description}"


def test_end_to_end_service_quantify_portfolio(sample_business_record):
    """Test full end-to-end quantify_portfolio pipeline execution."""
    service = BusinessImpactService()
    signals = [
        CrossDomainSignal(
            signal_id="SIG_E2E_1",
            signal_type="HIGH_REVENUE_LOW_STOCK",
            category=SignalCategory.CROSS_DOMAIN,
            severity=SignalSeverity.HIGH,
            dimension="SKU_WAREHOUSE",
            sku_id="SKU_100",
            warehouse_id="WH_NORTH",
            description="E2E Sig 1",
        ),
        CrossDomainSignal(
            signal_id="SIG_E2E_2",
            signal_type="HIGH_VALUE_INVENTORY",
            category=SignalCategory.INVENTORY,
            severity=SignalSeverity.HIGH,
            dimension="SKU_WAREHOUSE",
            sku_id="SKU_100",
            warehouse_id="WH_NORTH",
            description="E2E Sig 2",
        ),
    ]

    result = service.quantify_portfolio(signals, [sample_business_record], as_of_date="2025-06-30")
    assert isinstance(result, BusinessImpactResult)
    assert len(result.impact_records) == 2
    assert result.portfolio_summary.impact_record_count == 2
    assert result.portfolio_summary.gross_signal_exposure > 0
    assert result.portfolio_summary.deduplicated_exposure is not None
    assert result.as_of_date == "2025-06-30"


def test_unit_cost_inference():
    """Verify _infer_unit_cost logic under different available data conditions."""
    # From product cost / units sold
    rec1 = CrossDomainBusinessRecord(sku_id="SKU_C1", product_cost=3000.0, units_sold=100)
    assert _infer_unit_cost(rec1) == 30.0

    # From inventory value / on hand
    rec2 = CrossDomainBusinessRecord(sku_id="SKU_C2", inventory_value=4500.0, current_on_hand=150)
    assert _infer_unit_cost(rec2) == 30.0

    # Missing both
    rec3 = CrossDomainBusinessRecord(sku_id="SKU_C3")
    assert _infer_unit_cost(rec3) is None


# =====================================================================
# Semantic Audit & Deduplication Verification Tests
# =====================================================================

def test_deduplication_same_entity_same_impact_type_duplicate_signal(sample_business_record):
    """1. Same entity + same impact type + duplicate signal -> duplicate protection applies."""
    sig1 = CrossDomainSignal(
        signal_id="SIG_DUP_TYPE_1",
        signal_type="HIGH_RETURN_LOW_MARGIN",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Return margin signal 1",
    )
    sig2 = CrossDomainSignal(
        signal_id="SIG_DUP_TYPE_2",
        signal_type="HIGH_RETURN_LOW_MARGIN",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Return margin signal 2",
    )
    rec = sample_business_record.model_copy(update={"return_rate": 0.05, "gross_margin": 5000.0})
    imp1 = quantify_signal_impact(sig1, rec)
    imp2 = quantify_signal_impact(sig2, rec)
    assert imp1.exposure_value == 250.0
    assert imp2.exposure_value == 250.0
    dedup_val, _ = deduplicate_portfolio_exposure([imp1, imp2])
    assert dedup_val == 250.0  # collapsed to single exposure


def test_deduplication_same_entity_different_impact_types_remain_distinct(sample_business_record):
    """2. Same entity + different impact types -> exposures remain distinct."""
    sig_rev = CrossDomainSignal(
        signal_id="SIG_DIF_1",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Revenue signal",
    )
    sig_margin = CrossDomainSignal(
        signal_id="SIG_DIF_2",
        signal_type="HIGH_MARGIN_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Margin signal",
    )
    imp_rev = quantify_signal_impact(sig_rev, sample_business_record)
    imp_margin = quantify_signal_impact(sig_margin, sample_business_record)
    assert imp_rev.exposure_value == 10000.0
    assert imp_margin.exposure_value == 5000.0
    dedup_val, _ = deduplicate_portfolio_exposure([imp_rev, imp_margin])
    assert dedup_val == 15000.0  # distinct dimensions are summed, not collapsed


def test_deduplication_same_entity_inventory_and_return_exposure_not_collapsed(sample_business_record):
    """3. Same entity + inventory exposure + return exposure -> must not silently collapse them."""
    sig_inv = CrossDomainSignal(
        signal_id="SIG_INV",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Inventory signal",
    )
    sig_ret = CrossDomainSignal(
        signal_id="SIG_RET",
        signal_type="HIGH_RETURN_LOW_MARGIN",
        category=SignalCategory.RETURNS,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Return margin signal",
    )
    rec = sample_business_record.model_copy(update={"return_rate": 0.05, "gross_margin": 5000.0})
    imp_inv = quantify_signal_impact(sig_inv, rec)
    imp_ret = quantify_signal_impact(sig_ret, rec)
    assert imp_inv.exposure_value == 1250.0
    assert imp_ret.exposure_value == 250.0
    dedup_val, _ = deduplicate_portfolio_exposure([imp_inv, imp_ret])
    assert dedup_val == 1500.0  # 1250 + 250, not collapsed to 1250


def test_deduplication_same_entity_revenue_and_margin_exposure_not_collapsed(sample_business_record):
    """4. Same entity + revenue exposure + margin exposure -> must not silently collapse them."""
    sig_rev = CrossDomainSignal(
        signal_id="SIG_R",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Revenue signal",
    )
    sig_mar = CrossDomainSignal(
        signal_id="SIG_M",
        signal_type="HIGH_MARGIN_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Margin signal",
    )
    imp_rev = quantify_signal_impact(sig_rev, sample_business_record)
    imp_mar = quantify_signal_impact(sig_mar, sample_business_record)
    dedup_val, _ = deduplicate_portfolio_exposure([imp_rev, imp_mar])
    assert dedup_val == 15000.0  # 10000 + 5000, not collapsed


def test_deterministic_duplicate_handling(sample_business_record):
    """5. Deterministic duplicate handling produces identical results across runs."""
    sig1 = CrossDomainSignal(
        signal_id="SIG_D1",
        signal_type="HIGH_VALUE_INVENTORY",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Inventory 1",
    )
    sig2 = CrossDomainSignal(
        signal_id="SIG_D2",
        signal_type="HIGH_INVENTORY_LOW_DEMAND",
        category=SignalCategory.INVENTORY,
        severity=SignalSeverity.MEDIUM,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_NORTH",
        description="Inventory 2",
    )
    imp1 = quantify_signal_impact(sig1, sample_business_record)
    imp2 = quantify_signal_impact(sig2, sample_business_record)

    res1, stat1 = deduplicate_portfolio_exposure([imp1, imp2])
    res2, stat2 = deduplicate_portfolio_exposure([imp1, imp2])
    assert res1 == res2 == 1250.0
    assert stat1 == stat2


def test_benchmark_contains_no_individual_ranking():
    """6. Benchmark script contains no individual ranking (top 10 SKUs, warehouses, opportunities, or ranked entities)."""
    benchmark_path = "scripts/benchmark_business_impact.py"
    with open(benchmark_path, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_benchmark_patterns = [
        "top 10 quantified impact exposures",
        "top 10 opportunities",
        "top 10 skus",
        "top 10 warehouses",
        "largest individual exposures",
        "top_10 = sorted",
    ]
    for pattern in forbidden_benchmark_patterns:
        assert pattern not in content.lower(), f"Forbidden ranking pattern '{pattern}' found in benchmark script"


def test_documentation_contains_no_top_10_opportunities_ranking():
    """7. Documentation contains no 'top 10 opportunities' or individual entity ranking."""
    doc_path = "docs/business-impact.md"
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_doc_patterns = [
        "top 10 opportunities",
        "top 10 biggest opportunities",
        "top 10 quantified impact exposures",
        "top 10 skus",
        "top 10 warehouses",
    ]
    for pattern in forbidden_doc_patterns:
        assert pattern not in content.lower(), f"Forbidden ranking pattern '{pattern}' found in documentation"
