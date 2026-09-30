"""Comprehensive Test Suite for Cross-Domain Business Intelligence (Phase 6E).

Covers all 40 required test conditions (A through AN):
A. schema validation
B. SKU × warehouse grain
C. SKU grain
D. channel grain
E. warehouse grain
F. financial integration
G. inventory integration
H. demand integration
I. forecast integration
J. stockout integration
K. returns integration
L. replenishment integration
M. high revenue + low stock
N. high margin + low stock
O. low margin + high inventory
P. high inventory + low demand
Q. high return + high revenue
R. high return + low margin
S. stockout exposure
T. forecast coverage
U. replenishment financial context
V. warehouse aggregation
W. domain coverage
X. missing forecast
Y. missing returns
Z. missing inventory
AA. missing financial data
AB. currency isolation
AC. as_of_date filtering
AD. future returns excluded
AE. future sales excluded
AF. deterministic signal IDs
AG. deterministic severity
AH. configurable thresholds
AI. minimum sample size
AJ. no causal wording in generated descriptions
AK. regression against Phase 6A
AL. regression against Phase 6B
AM. regression against Phase 6C
AN. regression against Phase 6D
"""

from __future__ import annotations

from datetime import date, datetime
import pytest
import pandas as pd
import numpy as np

from commerce_ai.cross_domain.schemas import (
    CrossDomainBusinessRecord,
    CrossDomainDimension,
    CrossDomainIntelligenceConfig,
    CrossDomainIntelligenceResult,
    CrossDomainPortfolioSummary,
    CrossDomainSignal,
    SignalCategory,
    SignalSeverity,
)
from commerce_ai.cross_domain.signals import (
    FORBIDDEN_CAUSAL_WORDS,
    generate_cross_domain_signals,
    generate_signal_id,
)
from commerce_ai.cross_domain.service import CrossDomainIntelligenceService

# Phase 6A, 6B, 6C, 6D regression imports
from commerce_ai.financial import (
    FinancialIntelligenceService,
    UnitEconomicsService,
    ProfitabilityAttributionService,
    OperationalEconomicsService,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_products_df():
    return pd.DataFrame([
        {"sku_id": "SKU_001", "name": "Widget A", "category": "Electronics", "brand": "BrandX", "unit_cost": 20.0, "unit_price": 50.0},
        {"sku_id": "SKU_002", "name": "Widget B", "category": "Apparel", "brand": "BrandY", "unit_cost": 8.0, "unit_price": 10.0},
    ])


@pytest.fixture
def sample_sales_df():
    return pd.DataFrame([
        {"date": "2025-05-01", "order_id": "ORD_1", "sku_id": "SKU_001", "warehouse_id": "WH_1", "channel_id": "WEB", "quantity": 10, "unit_price": 50.0, "revenue": 500.0, "currency": "USD"},
        {"date": "2025-05-02", "order_id": "ORD_2", "sku_id": "SKU_001", "warehouse_id": "WH_1", "channel_id": "WEB", "quantity": 15, "unit_price": 50.0, "revenue": 750.0, "currency": "USD"},
        {"date": "2025-05-03", "order_id": "ORD_3", "sku_id": "SKU_001", "warehouse_id": "WH_2", "channel_id": "STORE", "quantity": 5, "unit_price": 50.0, "revenue": 250.0, "currency": "USD"},
        {"date": "2025-05-04", "order_id": "ORD_4", "sku_id": "SKU_002", "warehouse_id": "WH_1", "channel_id": "WEB", "quantity": 20, "unit_price": 10.0, "revenue": 200.0, "currency": "USD"},
    ])


@pytest.fixture
def sample_inventory_df():
    return pd.DataFrame([
        {"snapshot_date": "2025-05-05", "sku_id": "SKU_001", "warehouse_id": "WH_1", "available_qty": 5, "reserved_qty": 2, "on_hand": 7},
        {"snapshot_date": "2025-05-05", "sku_id": "SKU_001", "warehouse_id": "WH_2", "available_qty": 50, "reserved_qty": 0, "on_hand": 50},
        {"snapshot_date": "2025-05-05", "sku_id": "SKU_002", "warehouse_id": "WH_1", "available_qty": 200, "reserved_qty": 10, "on_hand": 210},
    ])


@pytest.fixture
def sample_returns_df():
    return pd.DataFrame([
        {"return_date": "2025-05-03", "return_id": "RET_1", "order_id": "ORD_1", "sku_id": "SKU_001", "warehouse_id": "WH_1", "channel_id": "WEB", "quantity": 3, "reason": "DEFECTIVE"},
        {"return_date": "2025-05-04", "return_id": "RET_2", "order_id": "ORD_2", "sku_id": "SKU_001", "warehouse_id": "WH_1", "channel_id": "WEB", "quantity": 2, "reason": "WRONG_SIZE"},
    ])


@pytest.fixture
def sample_forecasts_df():
    return pd.DataFrame([
        {"sku_id": "SKU_001", "warehouse_id": "WH_1", "forecast_mean": 25.0, "forecast_horizon": 14, "forecast_model": "LightGBM"},
        {"sku_id": "SKU_001", "warehouse_id": "WH_2", "forecast_mean": 10.0, "forecast_horizon": 14, "forecast_model": "LightGBM"},
        {"sku_id": "SKU_002", "warehouse_id": "WH_1", "forecast_mean": 15.0, "forecast_horizon": 14, "forecast_model": "LightGBM"},
    ])


@pytest.fixture
def sample_stockouts_df():
    return pd.DataFrame([
        {"date": "2025-05-01", "sku_id": "SKU_001", "warehouse_id": "WH_1", "is_stockout": True},
        {"date": "2025-05-02", "sku_id": "SKU_001", "warehouse_id": "WH_1", "is_stockout": False},
    ])


# =====================================================================
# Tests: A through AN
# =====================================================================

def test_condition_a_schema_validation():
    """A. schema validation."""
    record = CrossDomainBusinessRecord(
        sku_id="SKU_100",
        warehouse_id="WH_01",
        net_revenue=1500.0,
        gross_margin=600.0,
        units_sold=30,
        available_inventory=12,
        financial_available=True,
        inventory_available=True,
    )
    assert record.sku_id == "SKU_100"
    assert record.net_revenue == 1500.0
    d = record.to_dict()
    assert isinstance(d, dict)
    assert d["sku_id"] == "SKU_100"

    sig = CrossDomainSignal(
        signal_id="SIG-1234567890abcdef",
        signal_type="HIGH_REVENUE_LOW_STOCK",
        category=SignalCategory.CROSS_DOMAIN,
        severity=SignalSeverity.HIGH,
        dimension="SKU_WAREHOUSE",
        sku_id="SKU_100",
        warehouse_id="WH_01",
        description="Observed high revenue coincides with low inventory.",
    )
    assert sig.category == SignalCategory.CROSS_DOMAIN
    assert sig.severity == SignalSeverity.HIGH
    sig_dict = sig.to_dict()
    assert sig_dict["severity"] == "HIGH"

    cfg = CrossDomainIntelligenceConfig()
    assert cfg.low_stock_threshold == 10
    assert cfg.high_revenue_percentile == 0.80


def test_condition_b_sku_warehouse_grain(sample_sales_df, sample_inventory_df, sample_products_df):
    """B. SKU × warehouse grain."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
    )
    assert len(records) >= 3
    # Check natural grain keys
    for r in records:
        assert r.sku_id is not None
        assert r.warehouse_id is not None
        assert ":" not in r.sku_id  # Clean separate dimension keys


def test_condition_c_sku_grain(sample_sales_df, sample_inventory_df, sample_products_df):
    """C. SKU grain."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
    )
    sku_records = service.build_sku_view(records)
    assert len(sku_records) == 2
    sku_1 = next(r for r in sku_records if r.sku_id == "SKU_001")
    assert sku_1.warehouse_id is None
    # SKU_001 total units: 10 + 15 in WH_1 + 5 in WH_2 = 30
    assert sku_1.units_sold == 30
    # Total revenue: 500 + 750 + 250 = 1500
    assert sku_1.net_revenue == 1500.0


def test_condition_d_channel_grain(sample_sales_df, sample_products_df, sample_returns_df):
    """D. channel grain."""
    service = CrossDomainIntelligenceService()
    channel_records = service.build_channel_view(
        sales=sample_sales_df,
        returns=sample_returns_df,
        products=sample_products_df,
    )
    assert len(channel_records) == 2  # WEB and STORE
    web = next(r for r in channel_records if r.channel_id == "WEB")
    assert web.channel_id == "WEB"
    assert web.units_sold == 45  # 10 + 15 + 20
    assert web.inventory_available is False  # Inventory is not native to channel grain


def test_condition_e_warehouse_grain(sample_sales_df, sample_inventory_df, sample_products_df):
    """E. warehouse grain."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
    )
    wh_records = service.build_warehouse_view(records)
    assert len(wh_records) == 2  # WH_1 and WH_2
    wh_1 = next(r for r in wh_records if r.warehouse_id == "WH_1")
    assert wh_1.warehouse_id == "WH_1"
    assert wh_1.net_revenue == 1450.0  # 500 + 750 + 200


def test_condition_f_financial_integration(sample_sales_df, sample_inventory_df, sample_products_df):
    """F. financial integration."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    assert rec.net_revenue == 1250.0  # 500 + 750
    # 25 units * 20.0 unit_cost = 500.0 COGS
    assert rec.product_cost == 500.0
    assert rec.gross_margin == 750.0
    assert pytest.approx(rec.gross_margin_pct, 0.001) == 0.60


def test_condition_g_inventory_integration(sample_sales_df, sample_inventory_df, sample_products_df):
    """G. inventory integration."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    assert rec.available_inventory == 5
    assert rec.reserved_inventory == 2
    assert rec.current_on_hand == 7
    # 7 units * 20.0 = 140.0 inventory value
    assert rec.inventory_value == 140.0


def test_condition_h_demand_integration(sample_sales_df, sample_inventory_df, sample_products_df):
    """H. demand integration."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    assert rec.average_daily_demand is not None
    assert rec.demand_available is True
    assert rec.abc_class in {"A", "B", "C"}
    assert rec.xyz_class in {"X", "Y", "Z"}


def test_condition_i_forecast_integration(sample_sales_df, sample_inventory_df, sample_products_df, sample_forecasts_df):
    """I. forecast integration."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
        forecasts=sample_forecasts_df,
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    assert rec.forecast_available is True
    assert rec.forecast_mean == 25.0
    assert rec.forecast_horizon == 14
    assert rec.forecast_model == "LightGBM"


def test_condition_j_stockout_integration(sample_sales_df, sample_inventory_df, sample_products_df, sample_stockouts_df):
    """J. stockout integration."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
        stockouts=sample_stockouts_df,
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    assert rec.stockout_days == 1
    assert rec.stockout_rate == 0.50
    assert rec.stockout_flag is True


def test_condition_k_returns_integration(sample_sales_df, sample_inventory_df, sample_products_df, sample_returns_df):
    """K. returns integration."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
        returns=sample_returns_df,
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    assert rec.returns_available is True
    assert rec.return_units == 5  # 3 + 2
    assert rec.return_count == 2
    # 5 returns / 25 sold = 0.20 (20%)
    assert pytest.approx(rec.return_rate, 0.001) == 0.20


def test_condition_l_replenishment_integration(sample_sales_df, sample_inventory_df, sample_products_df):
    """L. replenishment integration."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    assert rec.replenishment_available is True
    assert rec.reorder_point is not None
    assert rec.replenishment_trigger is not None


def test_condition_m_high_revenue_low_stock():
    """M. high revenue + low stock."""
    service = CrossDomainIntelligenceService()
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_TOP",
        warehouse_id="WH_1",
        net_revenue=50000.0,
        available_inventory=2,
        financial_available=True,
        inventory_available=True,
    )
    cfg = CrossDomainIntelligenceConfig(low_stock_threshold=5)
    signals = service.build_cross_domain_signals([rec], config=cfg)
    sig_types = [s.signal_type for s in signals]
    assert "HIGH_REVENUE_LOW_STOCK" in sig_types
    sig = next(s for s in signals if s.signal_type == "HIGH_REVENUE_LOW_STOCK")
    assert sig.severity in {SignalSeverity.HIGH, SignalSeverity.CRITICAL}


def test_condition_n_high_margin_low_stock():
    """N. high margin + low stock."""
    service = CrossDomainIntelligenceService()
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_TOP",
        warehouse_id="WH_1",
        gross_margin=25000.0,
        available_inventory=3,
        financial_available=True,
        inventory_available=True,
    )
    cfg = CrossDomainIntelligenceConfig(low_stock_threshold=5)
    signals = service.build_cross_domain_signals([rec], config=cfg)
    sig_types = [s.signal_type for s in signals]
    assert "HIGH_MARGIN_LOW_STOCK" in sig_types


def test_condition_o_low_margin_high_inventory():
    """O. low margin + high inventory."""
    service = CrossDomainIntelligenceService()
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_LOW_M",
        warehouse_id="WH_1",
        gross_margin_pct=0.08,
        inventory_value=12000.0,
        current_on_hand=500,
        financial_available=True,
        inventory_available=True,
    )
    signals = service.build_cross_domain_signals([rec])
    sig_types = [s.signal_type for s in signals]
    assert "LOW_MARGIN_HIGH_INVENTORY" in sig_types


def test_condition_p_high_inventory_low_demand():
    """P. high inventory + low demand."""
    service = CrossDomainIntelligenceService()
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_SLOW",
        warehouse_id="WH_1",
        current_on_hand=400,
        average_daily_demand=0.2,
        abc_class="C",
        inventory_available=True,
        demand_available=True,
    )
    signals = service.build_cross_domain_signals([rec])
    sig_types = [s.signal_type for s in signals]
    assert "HIGH_INVENTORY_LOW_DEMAND" in sig_types


def test_condition_q_high_return_high_revenue():
    """Q. high return + high revenue."""
    service = CrossDomainIntelligenceService()
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_RET",
        warehouse_id="WH_1",
        net_revenue=8000.0,
        return_rate=0.22,
        units_sold=100,
        order_count=50,
        financial_available=True,
        returns_available=True,
    )
    signals = service.build_cross_domain_signals([rec])
    sig_types = [s.signal_type for s in signals]
    assert "HIGH_RETURN_HIGH_REVENUE" in sig_types


def test_condition_r_high_return_low_margin():
    """R. high return + low margin."""
    service = CrossDomainIntelligenceService()
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_RET_LM",
        warehouse_id="WH_1",
        gross_margin_pct=0.12,
        return_rate=0.18,
        units_sold=80,
        order_count=40,
        financial_available=True,
        returns_available=True,
    )
    signals = service.build_cross_domain_signals([rec])
    sig_types = [s.signal_type for s in signals]
    assert "HIGH_RETURN_LOW_MARGIN" in sig_types


def test_condition_s_stockout_exposure():
    """S. stockout exposure."""
    service = CrossDomainIntelligenceService()
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_SO",
        warehouse_id="WH_1",
        net_revenue=9000.0,
        stockout_days=7,
        stockout_rate=0.25,
        stockout_flag=True,
        financial_available=True,
    )
    signals = service.build_cross_domain_signals([rec])
    sig_types = [s.signal_type for s in signals]
    assert "HIGH_REVENUE_STOCKOUT_EXPOSURE" in sig_types


def test_condition_t_forecast_coverage():
    """T. forecast coverage."""
    service = CrossDomainIntelligenceService()
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_FC",
        warehouse_id="WH_1",
        forecast_mean=100.0,
        available_inventory=20,
        forecast_available=True,
        inventory_available=True,
    )
    signals = service.build_cross_domain_signals([rec])
    sig_types = [s.signal_type for s in signals]
    assert "FORECASTED_DEMAND_ABOVE_AVAILABLE_STOCK" in sig_types


def test_condition_u_replenishment_financial_context():
    """U. replenishment financial context."""
    service = CrossDomainIntelligenceService()
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_REP",
        warehouse_id="WH_1",
        net_revenue=15000.0,
        gross_margin=7000.0,
        replenishment_trigger=True,
        financial_available=True,
        replenishment_available=True,
    )
    signals = service.build_cross_domain_signals([rec])
    sig_types = [s.signal_type for s in signals]
    assert "HIGH_REVENUE_REPLENISHMENT_TRIGGER" in sig_types
    assert "HIGH_MARGIN_REPLENISHMENT_TRIGGER" in sig_types


def test_condition_v_warehouse_aggregation(sample_sales_df, sample_inventory_df, sample_products_df):
    """V. warehouse aggregation."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
    )
    wh_records = service.build_warehouse_view(records)
    for wh in wh_records:
        assert wh.units_sold is not None
        assert wh.units_sold >= 0
        assert wh.gross_margin is not None
        assert wh.gross_margin_pct is not None


def test_condition_w_domain_coverage(sample_sales_df, sample_inventory_df, sample_products_df, sample_returns_df, sample_forecasts_df, sample_stockouts_df):
    """W. domain coverage with all domains present."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
        returns=sample_returns_df,
        forecasts=sample_forecasts_df,
        stockouts=sample_stockouts_df,
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    assert rec.financial_available is True
    assert rec.inventory_available is True
    assert rec.demand_available is True
    assert rec.forecast_available is True
    assert rec.returns_available is True
    assert rec.replenishment_available is True
    assert rec.missing_domain_count == 0
    assert rec.domain_coverage_pct == 100.0


def test_condition_x_missing_forecast(sample_sales_df, sample_inventory_df, sample_products_df, sample_returns_df):
    """X. missing forecast."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
        returns=sample_returns_df,
        forecasts=None,
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    assert rec.forecast_available is False
    assert rec.missing_domain_count == 1
    assert pytest.approx(rec.domain_coverage_pct, 0.01) == 83.33


def test_condition_y_missing_returns(sample_sales_df, sample_inventory_df, sample_products_df):
    """Y. missing returns."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
        returns=None,
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    assert rec.returns_available is False
    assert rec.return_units == 0


def test_condition_z_missing_inventory(sample_sales_df, sample_products_df):
    """Z. missing inventory."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=None,
        products=sample_products_df,
    )
    for r in records:
        assert r.inventory_available is False
        assert r.current_on_hand is None


def test_condition_aa_missing_financial_data(sample_inventory_df, sample_products_df):
    """AA. missing financial data."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=None,
        inventory=sample_inventory_df,
        products=sample_products_df,
    )
    for r in records:
        assert r.financial_available is False
        assert r.net_revenue is None
        assert r.units_sold == 0


def test_condition_ab_currency_isolation(sample_inventory_df, sample_products_df):
    """AB. currency isolation."""
    mixed_sales = pd.DataFrame([
        {"date": "2025-05-01", "order_id": "O1", "sku_id": "SKU_001", "warehouse_id": "WH_1", "quantity": 5, "revenue": 100.0, "currency": "USD"},
        {"date": "2025-05-02", "order_id": "O2", "sku_id": "SKU_001", "warehouse_id": "WH_1", "quantity": 5, "revenue": 8000.0, "currency": "INR"},
    ])
    service = CrossDomainIntelligenceService()
    with pytest.raises(ValueError, match="Mixed currency detected"):
        service.build_sku_warehouse_view(
            sales=mixed_sales,
            inventory=sample_inventory_df,
            products=sample_products_df,
            config=CrossDomainIntelligenceConfig(allow_multi_currency=False),
        )


def test_condition_ac_as_of_date_filtering(sample_sales_df, sample_inventory_df, sample_products_df):
    """AC. as_of_date filtering."""
    service = CrossDomainIntelligenceService()
    # Filter up to May 2nd (exclude May 3 and May 4)
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
        as_of_date="2025-05-02",
    )
    wh2_rec = next((r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_2"), None)
    # On May 2nd, no sales had occurred at WH_2 yet
    assert wh2_rec is not None
    assert wh2_rec.units_sold == 0
    assert wh2_rec.net_revenue is None


def test_condition_ad_future_returns_excluded(sample_sales_df, sample_inventory_df, sample_products_df, sample_returns_df):
    """AD. future returns excluded."""
    service = CrossDomainIntelligenceService()
    # As of May 3rd, the return on May 4th should be excluded
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
        returns=sample_returns_df,
        as_of_date="2025-05-03",
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    # Only return on May 3rd (qty 3), May 4th (qty 2) excluded
    assert rec.return_units == 3


def test_condition_ae_future_sales_excluded(sample_sales_df, sample_inventory_df, sample_products_df):
    """AE. future sales excluded."""
    service = CrossDomainIntelligenceService()
    records = service.build_sku_warehouse_view(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
        as_of_date="2025-05-01",
    )
    rec = next(r for r in records if r.sku_id == "SKU_001" and r.warehouse_id == "WH_1")
    # Only May 1 sales (10 units), May 2 (15 units) excluded
    assert rec.units_sold == 10
    assert rec.net_revenue == 500.0


def test_condition_af_deterministic_signal_ids():
    """AF. deterministic signal IDs."""
    id1 = generate_signal_id("HIGH_REVENUE_LOW_STOCK", "SKU_WAREHOUSE", "SKU_1", "WH_1", None, "2025-05-01")
    id2 = generate_signal_id("HIGH_REVENUE_LOW_STOCK", "SKU_WAREHOUSE", "SKU_1", "WH_1", None, "2025-05-01")
    assert id1 == id2
    assert id1.startswith("SIG-")


def test_condition_ag_deterministic_severity():
    """AG. deterministic severity."""
    service = CrossDomainIntelligenceService()
    # Zero stock -> CRITICAL
    rec_crit = CrossDomainBusinessRecord(
        sku_id="SKU_1", warehouse_id="WH_1", net_revenue=10000.0, available_inventory=0, financial_available=True, inventory_available=True
    )
    sigs_crit = service.build_cross_domain_signals([rec_crit])
    sig = next(s for s in sigs_crit if s.signal_type == "HIGH_REVENUE_LOW_STOCK")
    assert sig.severity == SignalSeverity.CRITICAL

    # Positive stock below threshold -> HIGH
    rec_high = CrossDomainBusinessRecord(
        sku_id="SKU_1", warehouse_id="WH_1", net_revenue=10000.0, available_inventory=4, financial_available=True, inventory_available=True
    )
    sigs_high = service.build_cross_domain_signals([rec_high])
    sig_h = next(s for s in sigs_high if s.signal_type == "HIGH_REVENUE_LOW_STOCK")
    assert sig_h.severity == SignalSeverity.HIGH


def test_condition_ah_configurable_thresholds():
    """AH. configurable thresholds."""
    service = CrossDomainIntelligenceService()
    rec = CrossDomainBusinessRecord(
        sku_id="SKU_1", warehouse_id="WH_1", net_revenue=5000.0, available_inventory=8, financial_available=True, inventory_available=True
    )
    # Default low_stock_threshold is 10 -> triggers signal
    sigs_default = service.build_cross_domain_signals([rec], config=CrossDomainIntelligenceConfig(low_stock_threshold=10))
    assert any(s.signal_type == "HIGH_REVENUE_LOW_STOCK" for s in sigs_default)

    # Restrictive threshold 5 -> does not trigger signal (8 > 5)
    sigs_custom = service.build_cross_domain_signals([rec], config=CrossDomainIntelligenceConfig(low_stock_threshold=5))
    assert not any(s.signal_type == "HIGH_REVENUE_LOW_STOCK" for s in sigs_custom)


def test_condition_ai_minimum_sample_size():
    """AI. minimum sample size."""
    service = CrossDomainIntelligenceService()
    # Low sales sample (units_sold=2 < min_sample 10)
    rec_low_sample = CrossDomainBusinessRecord(
        sku_id="SKU_1", warehouse_id="WH_1", net_revenue=8000.0, return_rate=0.50, units_sold=2, order_count=1, financial_available=True, returns_available=True
    )
    sigs = service.build_cross_domain_signals([rec_low_sample], config=CrossDomainIntelligenceConfig(min_sample_size_units=10))
    # Return rate signals should be suppressed due to insufficient sample
    assert not any(s.signal_type == "HIGH_RETURN_HIGH_REVENUE" for s in sigs)


def test_condition_aj_no_causal_wording_in_generated_descriptions(
    sample_sales_df, sample_inventory_df, sample_products_df, sample_returns_df, sample_forecasts_df, sample_stockouts_df
):
    """AJ. no causal wording in generated descriptions."""
    service = CrossDomainIntelligenceService()
    res = service.analyze(
        sales=sample_sales_df,
        inventory=sample_inventory_df,
        products=sample_products_df,
        returns=sample_returns_df,
        forecasts=sample_forecasts_df,
        stockouts=sample_stockouts_df,
    )
    assert len(res.signals) > 0
    for s in res.signals:
        desc_lower = s.description.lower()
        for forbidden in FORBIDDEN_CAUSAL_WORDS:
            assert forbidden not in desc_lower, f"Forbidden word '{forbidden}' found in description: '{s.description}'"


def test_condition_ak_regression_against_phase_6a(sample_sales_df, sample_products_df):
    """AK. regression against Phase 6A."""
    service_6a = FinancialIntelligenceService()
    result = service_6a.analyze(sales=sample_sales_df, products=sample_products_df)
    assert result.portfolio_summary.total_net_revenue > 0
    assert result.portfolio_summary.total_gross_margin > 0


def test_condition_al_regression_against_phase_6b(sample_sales_df, sample_products_df):
    """AL. regression against Phase 6B."""
    service_6b = UnitEconomicsService()
    res = service_6b.calculate_unit_economics(sales=sample_sales_df, products=sample_products_df)
    assert res.portfolio_summary.total_net_revenue > 0
    assert res.portfolio_summary.total_gross_margin > 0


def test_condition_am_regression_against_phase_6c(sample_sales_df, sample_products_df):
    """AM. regression against Phase 6C."""
    service_6c = ProfitabilityAttributionService()
    res = service_6c.compute_profitability_attribution(sales_df=sample_sales_df, products_df=sample_products_df)
    assert res.portfolio_attribution.net_revenue > 0


def test_condition_an_regression_against_phase_6d(sample_sales_df, sample_products_df):
    """AN. regression against Phase 6D."""
    service_6d = OperationalEconomicsService()
    df = service_6d.calculate_transaction_economics(sales=sample_sales_df, products=sample_products_df)
    assert not df.empty
    assert "final_contribution_margin" in df.columns
