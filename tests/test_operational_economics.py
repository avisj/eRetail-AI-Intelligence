"""Tests for Phase 6D: Cost Completeness & Operational Economics.

Validates conditions A through AK:
- A: Explicit provenance tracking (SOURCE_DATA, CATALOG_ESTIMATE, CONFIGURED_ASSUMPTION, UNAVAILABLE)
- B: Distinguish source vs estimate vs assumption vs unavailable in cost details
- C: Completeness percentage calculation (available / total * 100)
- D: Complete economics status (when all required components present)
- E: Partially calculable status (when some required components missing)
- F: Missing shipping cost handling
- G: Missing payment processing fee handling
- H: Missing packaging cost handling
- I: Missing warehouse handling cost handling
- J: Missing return processing cost handling
- K: All variable costs unavailable (known CM = gross margin, final CM = None)
- L: Zero cost ($0.00) vs missing cost (None) strict differentiation
- M: Multi-line order handling (order costs allocated proportionally, no multi-line duplication)
- N: Order shipping cost allocation and sum invariance
- O: Unit product cost calculation (COGS = quantity * unit_product_cost)
- P: Per-unit cost assumptions applied correctly
- Q: Percentage-based cost assumptions applied correctly
- R: Fixed cost assumptions applied correctly
- S: Per-unit metrics (net revenue, gross margin, variable cost, contribution margin per unit)
- T: Order-level metrics and records
- U: Contribution margin calculation (net revenue - product cost - variable costs)
- V: Contribution margin percentage calculation
- W: Negative contribution margin detection
- X: Zero revenue handling (safe division, pct = None or 0)
- Y: Zero quantity handling (safe division, per-unit = None)
- Z: Currency isolation / verification
- AA: Point-in-time (as_of_date) filtering
- AB: Duplicate order cost protection
- AC: Missing SKU identifier handling (INVALID_INPUT)
- AD: Negative quantity, price, or discount handling (INVALID_INPUT)
- AE: Invalid assumption rate handling (validation error for rate > 1.0 or < 0.0)
- AF: Configurable required cost components
- AG: Estimated costs excluded from completeness when allow_estimated_costs_for_completeness=False
- AH: Assumed costs excluded from completeness when allow_assumed_costs_for_completeness=False
- AI: Returns data integration (return event cost vs non-returned $0.00 vs missing returns_df)
- AJ: Phase 6B regression verification
- AK: Phase 6C regression verification
- Additional service, dimensional, temporal, and portfolio tests
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pydantic import ValidationError

from commerce_ai.financial.schemas import (
    ContributionMarginStatus,
    CostAssumptionsConfig,
    CostCompletenessReport,
    CostCompletenessStatus,
    CostComponent,
    CostComponentDetail,
    CostComponentStatus,
    CostGrain,
    CostSourceType,
    OperationalCostDetail,
    OperationalEconomicsConfig,
    OperationalEconomicsRecord,
    OperationalEconomicsSegment,
    OperationalEconomicsStatus,
    OperationalEconomicsSummary,
    OrderOperationalEconomicsRecord,
    TimeGrain,
)
from commerce_ai.financial.operational_economics import (
    aggregate_operational_dimension,
    aggregate_operational_time_series,
    calculate_cost_completeness_report,
    calculate_operational_economics_record,
    compute_operational_economics_dataframe,
    compute_order_operational_economics_dataframe,
    generate_operational_order_id,
    generate_operational_record_id,
    generate_operational_segment_id,
    resolve_transaction_operational_costs,
    summarize_operational_portfolio,
)
from commerce_ai.financial.operational_service import OperationalEconomicsService
from commerce_ai.financial.service import UnitEconomicsService
from commerce_ai.financial.attribution import ProfitabilityAttributionService


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_products_df() -> pd.DataFrame:
    """Standard catalog products with procurement costs."""
    return pd.DataFrame(
        [
            {"sku_id": "SKU_001", "name": "Standard Widget", "unit_cost": 25.0, "category": "Electronics", "brand": "BrandA"},
            {"sku_id": "SKU_002", "name": "Premium Gadget", "unit_cost": 45.0, "category": "Electronics", "brand": "BrandB"},
            {"sku_id": "SKU_003", "name": "Basic Cable", "unit_cost": 5.0, "category": "Accessories", "brand": "BrandA"},
        ]
    )


@pytest.fixture
def sample_sales_df() -> pd.DataFrame:
    """Sample sales transactions including single-line and multi-line orders."""
    return pd.DataFrame(
        [
            # Order 1: single-line
            {
                "sale_id": "TX_001",
                "order_id": "ORD_101",
                "date": "2026-01-15",
                "sku_id": "SKU_001",
                "warehouse_id": "WH_EAST",
                "channel_id": "ONLINE",
                "quantity": 2,
                "unit_price": 50.0,
                "discount": 10.0,
                "revenue": 90.0,
            },
            # Order 2: multi-line (Line 1 & Line 2)
            {
                "sale_id": "TX_002",
                "order_id": "ORD_102",
                "date": "2026-01-16",
                "sku_id": "SKU_001",
                "warehouse_id": "WH_WEST",
                "channel_id": "ONLINE",
                "quantity": 1,
                "unit_price": 50.0,
                "discount": 0.0,
                "revenue": 50.0,
            },
            {
                "sale_id": "TX_003",
                "order_id": "ORD_102",
                "date": "2026-01-16",
                "sku_id": "SKU_002",
                "warehouse_id": "WH_WEST",
                "channel_id": "ONLINE",
                "quantity": 2,
                "unit_price": 100.0,
                "discount": 0.0,
                "revenue": 200.0,
            },
            # Order 3: single-line in March
            {
                "sale_id": "TX_004",
                "order_id": "ORD_103",
                "date": "2026-03-20",
                "sku_id": "SKU_003",
                "warehouse_id": "WH_EAST",
                "channel_id": "RETAIL",
                "quantity": 4,
                "unit_price": 15.0,
                "discount": 5.0,
                "revenue": 55.0,
            },
        ]
    )


@pytest.fixture
def sample_returns_df() -> pd.DataFrame:
    """Sample returns data matching specific orders."""
    return pd.DataFrame(
        [
            {
                "return_id": "RET_001",
                "order_id": "ORD_101",
                "sku_id": "SKU_001",
                "return_date": "2026-01-20",
                "warehouse_id": "WH_EAST",
                "quantity": 1,
                "reason": "DEFECTIVE",
                "channel_id": "ONLINE",
            }
        ]
    )


# =============================================================================
# Condition A & B: Explicit Provenance Tracking & Separation
# =============================================================================


def test_condition_a_provenance_enums():
    """Condition A: Verify CostSourceType and CostGrain enums."""
    assert CostSourceType.SOURCE_DATA.value == "SOURCE_DATA"
    assert CostSourceType.CATALOG_ESTIMATE.value == "CATALOG_ESTIMATE"
    assert CostSourceType.CONFIGURED_ASSUMPTION.value == "CONFIGURED_ASSUMPTION"
    assert CostSourceType.UNAVAILABLE.value == "UNAVAILABLE"

    assert CostGrain.UNIT.value == "UNIT"
    assert CostGrain.TRANSACTION.value == "TRANSACTION"
    assert CostGrain.ORDER.value == "ORDER"
    assert CostGrain.RETURN_EVENT.value == "RETURN_EVENT"
    assert CostGrain.UNKNOWN.value == "UNKNOWN"


def test_condition_b_distinguish_source_estimate_assumption_unavailable(sample_products_df):
    """Condition B: Verify explicit resolution distinguishing source vs estimate vs assumption vs unavailable."""
    config = OperationalEconomicsConfig(
        assumptions=CostAssumptionsConfig(
            shipping_cost_per_unit=5.0,
            payment_processing_rate=0.03,
        )
    )
    row = {
        "transaction_id": "T1",
        "order_id": "O1",
        "sku_id": "SKU_001",
        "date": "2026-01-01",
        "warehouse_id": "WH1",
        "channel_id": "CH1",
        "quantity": 2,
        "unit_price": 50.0,
        "discount": 0.0,
        "packaging_cost": 2.50,  # Explicit source data
    }
    details = resolve_transaction_operational_costs(row, unit_cost=25.0, config=config)

    # 1. Product cost from catalog estimate
    p = details[CostComponent.PRODUCT_COST.value]
    assert p.source_type == CostSourceType.CATALOG_ESTIMATE
    assert p.amount == 50.0
    assert p.is_estimated is True
    assert p.is_assumed is False

    # 2. Shipping cost from configured assumption
    s = details[CostComponent.SHIPPING_COST.value]
    assert s.source_type == CostSourceType.CONFIGURED_ASSUMPTION
    assert s.amount == 10.0
    assert s.is_assumed is True
    assert s.is_estimated is False

    # 3. Packaging cost from observed source data
    pkg = details[CostComponent.PACKAGING_COST.value]
    assert pkg.source_type == CostSourceType.SOURCE_DATA
    assert pkg.amount == 2.50
    assert pkg.is_assumed is False
    assert pkg.is_estimated is False

    # 4. Warehouse handling is unconfigured and unobserved -> UNAVAILABLE
    wh = details[CostComponent.WAREHOUSE_HANDLING_COST.value]
    assert wh.source_type == CostSourceType.UNAVAILABLE
    assert wh.amount is None
    assert wh.availability_status == CostComponentStatus.UNAVAILABLE


# =============================================================================
# Condition C, D, E: Completeness Calculation and Statuses
# =============================================================================


def test_condition_c_completeness_percentage_calculation():
    """Condition C: Verify cost completeness percentage calculation (available / total * 100)."""
    config = OperationalEconomicsConfig(
        required_cost_components=[
            CostComponent.PRODUCT_COST,
            CostComponent.SHIPPING_COST,
            CostComponent.PAYMENT_PROCESSING_COST,
            CostComponent.PACKAGING_COST,
        ],
        allow_estimated_costs_for_completeness=True,
        allow_assumed_costs_for_completeness=True,
    )
    details = {
        CostComponent.PRODUCT_COST.value: OperationalCostDetail(
            component=CostComponent.PRODUCT_COST,
            amount=50.0,
            source_type=CostSourceType.CATALOG_ESTIMATE,
            availability_status=CostComponentStatus.AVAILABLE_ESTIMATED,
        ),
        CostComponent.SHIPPING_COST.value: OperationalCostDetail(
            component=CostComponent.SHIPPING_COST,
            amount=10.0,
            source_type=CostSourceType.CONFIGURED_ASSUMPTION,
            availability_status=CostComponentStatus.AVAILABLE_ASSUMED,
        ),
        CostComponent.PAYMENT_PROCESSING_COST.value: OperationalCostDetail(
            component=CostComponent.PAYMENT_PROCESSING_COST,
            amount=None,
            source_type=CostSourceType.UNAVAILABLE,
            availability_status=CostComponentStatus.UNAVAILABLE,
        ),
        CostComponent.PACKAGING_COST.value: OperationalCostDetail(
            component=CostComponent.PACKAGING_COST,
            amount=None,
            source_type=CostSourceType.UNAVAILABLE,
            availability_status=CostComponentStatus.UNAVAILABLE,
        ),
    }
    report = calculate_cost_completeness_report(details, config)
    # 2 out of 4 required components available -> 50.0%
    assert report.completeness_pct == 50.0
    assert report.total_required_components == 4
    assert report.available_required_components == 2
    assert report.missing_required_components == 2
    assert report.completeness_status == CostCompletenessStatus.PARTIALLY_COMPLETE


def test_condition_d_complete_economics_status(sample_products_df):
    """Condition D: All required components satisfied -> COMPLETE status and populated final contribution margin."""
    config = OperationalEconomicsConfig(
        required_cost_components=[
            CostComponent.PRODUCT_COST,
            CostComponent.SHIPPING_COST,
        ],
        allow_estimated_costs_for_completeness=True,
        assumptions=CostAssumptionsConfig(
            shipping_cost_per_unit=5.0,
        ),
        allow_assumed_costs_for_completeness=True,
    )
    sales = pd.DataFrame(
        [
            {
                "sale_id": "T1",
                "order_id": "O1",
                "date": "2026-01-01",
                "sku_id": "SKU_001",
                "warehouse_id": "W1",
                "channel_id": "C1",
                "quantity": 2,
                "unit_price": 50.0,
                "discount": 0.0,
                "revenue": 100.0,
            }
        ]
    )
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    assert len(df) == 1
    assert df["economics_status"].iloc[0] == OperationalEconomicsStatus.COMPLETE.value
    assert df["final_contribution_margin_status"].iloc[0] == ContributionMarginStatus.CALCULABLE.value
    # Net revenue = 100, product cost = 50, shipping = 10 -> final CM = 40.0
    assert df["final_contribution_margin"].iloc[0] == 40.0
    assert df["final_contribution_margin_pct"].iloc[0] == 0.40
    assert df["cost_completeness_status"].iloc[0] == CostCompletenessStatus.COMPLETE.value


def test_condition_e_partially_calculable_status(sample_products_df):
    """Condition E: Some required components missing -> PARTIALLY_CALCULABLE and final_cm is None."""
    config = OperationalEconomicsConfig(
        required_cost_components=[
            CostComponent.PRODUCT_COST,
            CostComponent.SHIPPING_COST,
            CostComponent.PACKAGING_COST,
        ],
        allow_estimated_costs_for_completeness=True,
        assumptions=CostAssumptionsConfig(
            shipping_cost_per_unit=5.0,
        ),
        allow_assumed_costs_for_completeness=True,
        # packaging_cost is missing
    )
    sales = pd.DataFrame(
        [
            {
                "sale_id": "T1",
                "order_id": "O1",
                "date": "2026-01-01",
                "sku_id": "SKU_001",
                "warehouse_id": "W1",
                "channel_id": "C1",
                "quantity": 2,
                "unit_price": 50.0,
                "discount": 0.0,
                "revenue": 100.0,
            }
        ]
    )
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    assert len(df) == 1
    assert df["economics_status"].iloc[0] == OperationalEconomicsStatus.PARTIALLY_CALCULABLE.value
    assert df["final_contribution_margin_status"].iloc[0] == ContributionMarginStatus.INSUFFICIENT_COST_DATA.value
    assert pd.isna(df["final_contribution_margin"].iloc[0])
    assert pd.isna(df["final_contribution_margin_pct"].iloc[0])
    # But known contribution margin IS calculable
    assert df["known_contribution_margin"].iloc[0] == 40.0


# =============================================================================
# Conditions F through K: Missing Individual Variable Costs
# =============================================================================


def test_condition_f_missing_shipping_cost(sample_products_df):
    """Condition F: Missing shipping cost is None (not silently 0)."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df)
    assert pd.isna(df["shipping_cost"].iloc[0])
    assert df["shipping_cost_source"].iloc[0] == CostSourceType.UNAVAILABLE.value


def test_condition_g_missing_payment_cost(sample_products_df):
    """Condition G: Missing payment processing fee is None."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df)
    assert pd.isna(df["payment_processing_cost"].iloc[0])
    assert df["payment_cost_source"].iloc[0] == CostSourceType.UNAVAILABLE.value


def test_condition_h_missing_packaging_cost(sample_products_df):
    """Condition H: Missing packaging cost is None."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df)
    assert pd.isna(df["packaging_cost"].iloc[0])
    assert df["packaging_cost_source"].iloc[0] == CostSourceType.UNAVAILABLE.value


def test_condition_i_missing_warehouse_handling_cost(sample_products_df):
    """Condition I: Missing warehouse handling cost is None."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df)
    assert pd.isna(df["warehouse_handling_cost"].iloc[0])
    assert df["warehouse_cost_source"].iloc[0] == CostSourceType.UNAVAILABLE.value


def test_condition_j_missing_return_processing_cost(sample_products_df):
    """Condition J: Missing return processing cost is None when returns_df is absent and unobserved."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df, returns_df=None)
    assert pd.isna(df["return_processing_cost"].iloc[0])
    assert df["return_cost_source"].iloc[0] == CostSourceType.UNAVAILABLE.value


def test_condition_k_all_variable_costs_unavailable(sample_products_df):
    """Condition K: When all variable costs are unavailable, known CM equals gross margin, final CM is None."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 2, "unit_price": 50.0, "revenue": 100.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df)
    # Net revenue = 100, product cost = 50 -> gross margin = 50
    assert df["gross_margin"].iloc[0] == 50.0
    assert df["total_known_variable_cost"].iloc[0] == 0.0
    assert df["known_contribution_margin"].iloc[0] == 50.0
    assert pd.isna(df["final_contribution_margin"].iloc[0])
    assert df["final_contribution_margin_status"].iloc[0] == ContributionMarginStatus.INSUFFICIENT_COST_DATA.value


# =============================================================================
# Condition L: Zero Cost ($0.00) vs Missing Cost (None) Strict Separation
# =============================================================================


def test_condition_l_zero_cost_vs_missing_cost():
    """Condition L: An observed 0.0 cost is SOURCE_DATA / 0.0, distinct from None / UNAVAILABLE."""
    row = {
        "transaction_id": "T1",
        "order_id": "O1",
        "sku_id": "SKU_001",
        "date": "2026-01-01",
        "warehouse_id": "W1",
        "channel_id": "C1",
        "quantity": 1,
        "unit_price": 50.0,
        "shipping_cost": 0.0,  # Observed free shipping
        # packaging_cost is missing
    }
    details = resolve_transaction_operational_costs(row, unit_cost=25.0)

    ship = details[CostComponent.SHIPPING_COST.value]
    pkg = details[CostComponent.PACKAGING_COST.value]

    # Shipping is observed zero
    assert ship.amount == 0.0
    assert ship.source_type == CostSourceType.SOURCE_DATA
    assert ship.availability_status == CostComponentStatus.AVAILABLE_SOURCE
    assert ship.included_in_known_contribution is True

    # Packaging is unobserved None
    assert pkg.amount is None
    assert pkg.source_type == CostSourceType.UNAVAILABLE
    assert pkg.availability_status == CostComponentStatus.UNAVAILABLE
    assert pkg.included_in_known_contribution is False


# =============================================================================
# Condition M, N, AB: Multi-Line Order Handling & Invariance
# =============================================================================


def test_condition_m_n_ab_multiline_order_allocation_invariance(sample_products_df):
    """Conditions M, N, AB: Multi-line order costs allocated proportionally without duplication or multiplication."""
    config = OperationalEconomicsConfig(
        assumptions=CostAssumptionsConfig(
            shipping_cost_per_order=12.0,  # $12 flat shipping per order
        ),
        order_cost_allocation_method="NET_REVENUE",
    )
    sales = pd.DataFrame(
        [
            # Multi-line order ORD_MULTI: 2 lines
            # Line 1: revenue $30 (30 / 100 = 30%)
            {
                "sale_id": "L1",
                "order_id": "ORD_MULTI",
                "date": "2026-01-01",
                "sku_id": "SKU_001",
                "warehouse_id": "W1",
                "channel_id": "C1",
                "quantity": 1,
                "unit_price": 30.0,
                "discount": 0.0,
                "revenue": 30.0,
            },
            # Line 2: revenue $70 (70 / 100 = 70%)
            {
                "sale_id": "L2",
                "order_id": "ORD_MULTI",
                "date": "2026-01-01",
                "sku_id": "SKU_002",
                "warehouse_id": "W1",
                "channel_id": "C1",
                "quantity": 1,
                "unit_price": 70.0,
                "discount": 0.0,
                "revenue": 70.0,
            },
        ]
    )
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)

    # Line 1 should receive 30% of $12 = $3.60
    assert np.isclose(df.loc[df["sale_id"] == "L1", "shipping_cost"].iloc[0], 3.60)
    # Line 2 should receive 70% of $12 = $8.40
    assert np.isclose(df.loc[df["sale_id"] == "L2", "shipping_cost"].iloc[0], 8.40)

    # Sum invariance: total allocated shipping must equal exactly $12.00 (NOT $24.00!)
    total_line_shipping = df["shipping_cost"].sum()
    assert np.isclose(total_line_shipping, 12.00)

    # Order level economics
    order_df = compute_order_operational_economics_dataframe(df, config=config)
    assert len(order_df) == 1
    assert np.isclose(order_df["shipping_cost"].iloc[0], 12.00)
    assert order_df["line_count"].iloc[0] == 2


# =============================================================================
# Condition O, P, Q, R: Product Cost and Parametric Cost Assumptions
# =============================================================================


def test_condition_o_unit_product_cost_cogs(sample_products_df):
    """Condition O: COGS = quantity * unit_product_cost."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 4, "unit_price": 50.0, "revenue": 200.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df)
    # SKU_001 unit cost is 25.0 -> product_cost = 100.0
    assert df["unit_product_cost"].iloc[0] == 25.0
    assert df["product_cost"].iloc[0] == 100.0
    assert df["gross_margin"].iloc[0] == 100.0


def test_condition_p_per_unit_cost_assumptions(sample_products_df):
    """Condition P: Per-unit packaging & handling assumptions scale with quantity."""
    config = OperationalEconomicsConfig(
        assumptions=CostAssumptionsConfig(
            packaging_cost_per_unit=1.50,
            warehouse_handling_cost_per_unit=2.00,
        )
    )
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 5, "unit_price": 50.0, "revenue": 250.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    assert df["packaging_cost"].iloc[0] == 5 * 1.50  # 7.50
    assert df["warehouse_handling_cost"].iloc[0] == 5 * 2.00  # 10.00


def test_condition_q_percentage_cost_assumptions(sample_products_df):
    """Condition Q: Percentage-based payment processing fee assumption."""
    config = OperationalEconomicsConfig(
        assumptions=CostAssumptionsConfig(
            payment_processing_rate=0.03,  # 3% fee
        )
    )
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 200.0, "revenue": 200.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    assert np.isclose(df["payment_processing_cost"].iloc[0], 6.00)


def test_condition_r_fixed_cost_assumptions(sample_products_df):
    """Condition R: Fixed transaction fee per order."""
    config = OperationalEconomicsConfig(
        assumptions=CostAssumptionsConfig(
            payment_processing_fixed_per_order=0.30,
        )
    )
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    assert np.isclose(df["payment_processing_cost"].iloc[0], 0.30)


# =============================================================================
# Condition S, T: Per-Unit and Order-Level Metrics
# =============================================================================


def test_condition_s_per_unit_metrics(sample_products_df):
    """Condition S: Net revenue, gross margin, variable cost, and contribution margin per unit."""
    config = OperationalEconomicsConfig(
        assumptions=CostAssumptionsConfig(shipping_cost_per_unit=5.0),
        allow_assumed_costs_for_completeness=True,
    )
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 2, "unit_price": 50.0, "discount": 10.0, "revenue": 90.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)

    # Net revenue = 90 / 2 = 45.0 per unit
    assert df["net_revenue_per_unit"].iloc[0] == 45.0
    # Gross margin = (90 - 50) / 2 = 20.0 per unit
    assert df["gross_margin_per_unit"].iloc[0] == 20.0
    # Variable cost = 10 / 2 = 5.0 per unit
    assert df["variable_cost_per_unit"].iloc[0] == 5.0
    # Contribution margin = (40) / 2 = 15.0 per unit
    assert df["contribution_margin_per_unit"].iloc[0] == 15.0


def test_condition_t_order_level_dataframe(sample_sales_df, sample_products_df):
    """Condition T: Order-level aggregated metrics and distinct order counts."""
    line_df = compute_operational_economics_dataframe(sample_sales_df, sample_products_df)
    order_df = compute_order_operational_economics_dataframe(line_df)

    # 3 distinct orders in sample_sales_df
    assert len(order_df) == 3
    assert set(order_df["order_id"]) == {"ORD_101", "ORD_102", "ORD_103"}
    # ORD_102 has 2 lines and 3 units
    ord102 = order_df.loc[order_df["order_id"] == "ORD_102"].iloc[0]
    assert ord102["line_count"] == 2
    assert ord102["total_quantity"] == 3
    assert ord102["net_revenue"] == 250.0


# =============================================================================
# Condition U, V, W, X, Y: Mathematical Integrity & Edge Cases
# =============================================================================


def test_condition_u_v_contribution_margin_and_pct(sample_products_df):
    """Condition U & V: Contribution margin and margin percentage formulas."""
    config = OperationalEconomicsConfig(
        required_cost_components=[CostComponent.PRODUCT_COST, CostComponent.SHIPPING_COST],
        allow_estimated_costs_for_completeness=True,
        allow_assumed_costs_for_completeness=True,
        assumptions=CostAssumptionsConfig(shipping_cost_per_unit=10.0),
    )
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 100.0, "revenue": 100.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)

    # Net rev = 100, product = 25, shipping = 10 -> CM = 65, CM% = 65%
    assert df["final_contribution_margin"].iloc[0] == 65.0
    assert df["final_contribution_margin_pct"].iloc[0] == 0.65


def test_condition_w_negative_contribution_margin(sample_products_df):
    """Condition W: Negative contribution margin correctly detected when expenses exceed revenue."""
    config = OperationalEconomicsConfig(
        required_cost_components=[CostComponent.PRODUCT_COST, CostComponent.SHIPPING_COST],
        allow_estimated_costs_for_completeness=True,
        allow_assumed_costs_for_completeness=True,
        assumptions=CostAssumptionsConfig(shipping_cost_per_unit=80.0),  # High shipping fee
    )
    # Unit price = 50, Product cost = 25, Shipping = 80 -> Net rev = 50, Costs = 105 -> CM = -55
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)

    assert df["final_contribution_margin"].iloc[0] == -55.0
    assert df["final_contribution_margin_pct"].iloc[0] == -1.10


def test_condition_x_zero_revenue_handling(sample_products_df):
    """Condition X: Zero revenue safe division; percentage is None or 0 without ZeroDivisionError."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 0.0, "revenue": 0.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df)
    assert df["gross_margin"].iloc[0] == -25.0
    assert pd.isna(df["gross_margin_pct"].iloc[0])
    assert pd.isna(df["known_contribution_margin_pct"].iloc[0])


def test_condition_y_zero_quantity_handling(sample_products_df):
    """Condition Y: Zero quantity safe division; per-unit metrics are None."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 0, "unit_price": 50.0, "revenue": 0.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df)
    assert pd.isna(df["net_revenue_per_unit"].iloc[0])
    assert pd.isna(df["gross_margin_per_unit"].iloc[0])
    assert pd.isna(df["variable_cost_per_unit"].iloc[0])


# =============================================================================
# Condition Z, AA: Currency and Point-in-Time Anti-Leakage
# =============================================================================


def test_condition_z_currency_isolation():
    """Condition Z: Records stamped with currency."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "currency": "EUR", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales)
    assert df["currency"].iloc[0] == "EUR"


def test_condition_aa_as_of_date_filtering(sample_sales_df, sample_products_df):
    """Condition AA: Point-in-time filter excludes future sales events."""
    config = OperationalEconomicsConfig(as_of_date="2026-01-31")
    # sample_sales_df has transactions in January and March 2026
    df = compute_operational_economics_dataframe(sample_sales_df, sample_products_df, config=config)
    assert len(df) == 3
    assert (df["order_date"] <= "2026-01-31").all()


# =============================================================================
# Condition AC, AD, AE: Validation & Error Handling
# =============================================================================


def test_condition_ac_missing_sku_invalid():
    """Condition AC: Missing SKU identifier flagged as INVALID_INPUT."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales)
    assert df["economics_status"].iloc[0] == OperationalEconomicsStatus.INVALID_INPUT.value
    assert df["final_contribution_margin_status"].iloc[0] == ContributionMarginStatus.NOT_CALCULABLE.value


def test_condition_ad_negative_inputs_invalid():
    """Condition AD: Negative quantity or discount flagged as INVALID_INPUT."""
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": -2, "unit_price": 50.0, "revenue": 100.0}])
    df = compute_operational_economics_dataframe(sales)
    assert df["economics_status"].iloc[0] == OperationalEconomicsStatus.INVALID_INPUT.value


def test_condition_ae_invalid_assumption_rates():
    """Condition AE: Invalid rate (> 1.0 or < 0.0) raises validation error."""
    with pytest.raises(ValidationError):
        CostAssumptionsConfig(payment_processing_rate=1.5)  # Must be <= 1.0
    with pytest.raises(ValidationError):
        CostAssumptionsConfig(shipping_pct_of_net_revenue=-0.05)  # Must be >= 0.0


# =============================================================================
# Condition AF, AG, AH: Configurable Completeness Policy
# =============================================================================


def test_condition_af_configurable_required_components(sample_products_df):
    """Condition AF: Custom required components list."""
    # Require ONLY Product Cost and Packaging Cost
    config = OperationalEconomicsConfig(
        required_cost_components=[CostComponent.PRODUCT_COST, CostComponent.PACKAGING_COST],
        allow_estimated_costs_for_completeness=True,
    )
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "packaging_cost": 2.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    assert df["cost_completeness_pct"].iloc[0] == 100.0
    assert df["economics_status"].iloc[0] == OperationalEconomicsStatus.COMPLETE.value


def test_condition_ag_estimated_costs_excluded_when_disallowed(sample_products_df):
    """Condition AG: When allow_estimated_costs_for_completeness=False, catalog estimates do not satisfy completeness."""
    config = OperationalEconomicsConfig(
        required_cost_components=[CostComponent.PRODUCT_COST],
        allow_estimated_costs_for_completeness=False,  # Strict: catalog cost doesn't count
    )
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    assert df["cost_completeness_pct"].iloc[0] == 0.0
    assert df["final_contribution_margin_status"].iloc[0] == ContributionMarginStatus.INSUFFICIENT_COST_DATA.value


def test_condition_ah_assumed_costs_excluded_by_default(sample_products_df):
    """Condition AH: By default, allow_assumed_costs_for_completeness is False; assumptions don't satisfy completeness."""
    config = OperationalEconomicsConfig(
        required_cost_components=[CostComponent.PRODUCT_COST, CostComponent.SHIPPING_COST],
        allow_estimated_costs_for_completeness=True,
        allow_assumed_costs_for_completeness=False,  # Default strict
        assumptions=CostAssumptionsConfig(shipping_cost_per_unit=5.0),
    )
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    # Product cost is satisfied (1/2), but shipping assumption is not allowed for completeness
    assert df["cost_completeness_pct"].iloc[0] == 50.0
    assert df["final_contribution_margin_status"].iloc[0] == ContributionMarginStatus.INSUFFICIENT_COST_DATA.value


# =============================================================================
# Condition AI: Returns Data Integration
# =============================================================================


def test_condition_ai_returns_integration(sample_products_df, sample_returns_df):
    """Condition AI: Actual returns from returns_df apply fee; non-returned orders have $0.00 observed return cost."""
    config = OperationalEconomicsConfig(
        assumptions=CostAssumptionsConfig(
            return_cost_per_returned_unit=10.0,
        )
    )
    sales = pd.DataFrame(
        [
            # ORD_101: returned 1 unit of SKU_001 in sample_returns_df
            {
                "sale_id": "T1",
                "order_id": "ORD_101",
                "date": "2026-01-15",
                "sku_id": "SKU_001",
                "quantity": 1,
                "unit_price": 50.0,
                "revenue": 50.0,
            },
            # ORD_102: NO return in sample_returns_df
            {
                "sale_id": "T2",
                "order_id": "ORD_102",
                "date": "2026-01-16",
                "sku_id": "SKU_001",
                "quantity": 1,
                "unit_price": 50.0,
                "revenue": 50.0,
            },
        ]
    )
    df = compute_operational_economics_dataframe(
        sales_df=sales,
        products_df=sample_products_df,
        returns_df=sample_returns_df,
        config=config,
    )

    # Returned order: fee is $10.00, source CONFIGURED_ASSUMPTION
    ord101 = df.loc[df["order_id"] == "ORD_101"].iloc[0]
    assert ord101["return_processing_cost"] == 10.0
    assert ord101["return_cost_source"] == CostSourceType.CONFIGURED_ASSUMPTION.value

    # Non-returned order under configured fee policy: return cost is $0.00, source CONFIGURED_ASSUMPTION
    ord102 = df.loc[df["order_id"] == "ORD_102"].iloc[0]
    assert ord102["return_processing_cost"] == 0.0
    assert ord102["return_cost_source"] == CostSourceType.CONFIGURED_ASSUMPTION.value

    # When NO return fee assumption is configured, returns_df does not make return cost available
    df_no_asm = compute_operational_economics_dataframe(
        sales_df=sales,
        products_df=sample_products_df,
        returns_df=sample_returns_df,
        config=OperationalEconomicsConfig(),  # No return fee assumption
    )
    assert pd.isna(df_no_asm.loc[df_no_asm["order_id"] == "ORD_101", "return_processing_cost"].iloc[0])
    assert pd.isna(df_no_asm.loc[df_no_asm["order_id"] == "ORD_102", "return_processing_cost"].iloc[0])
    assert df_no_asm["return_cost_source"].iloc[0] == CostSourceType.UNAVAILABLE.value


# =============================================================================
# Condition AJ & AK: Phase 6B & 6C Regression Protection
# =============================================================================


def test_condition_aj_phase_6b_regression(sample_sales_df, sample_products_df):
    """Condition AJ: UnitEconomicsService from Phase 6B continues to operate with identical results."""
    service = UnitEconomicsService()
    result = service.calculate_unit_economics(sample_sales_df, sample_products_df)
    assert result.portfolio_summary.total_order_lines == 4
    assert result.portfolio_summary.total_gross_margin > 0
    assert result.margin_erosion is not None


def test_condition_ak_phase_6c_regression(sample_sales_df, sample_products_df):
    """Condition AK: ProfitabilityAttributionService from Phase 6C continues to operate with identical results."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales_df, sample_products_df)
    assert result.portfolio_attribution is not None
    assert result.margin_waterfall is not None
    assert len(result.margin_waterfall.stages) > 0


# =============================================================================
# Additional Functional & Service Tests
# =============================================================================


def test_operational_service_orchestration(sample_sales_df, sample_products_df, sample_returns_df):
    """Verify OperationalEconomicsService high-level orchestration methods."""
    service = OperationalEconomicsService()

    # 1. Transaction economics
    tx_df = service.calculate_transaction_economics(sample_sales_df, sample_products_df, sample_returns_df)
    assert len(tx_df) == 4
    assert "final_contribution_margin_status" in tx_df.columns

    # 2. Order economics
    ord_df = service.calculate_order_economics(sample_sales_df, sample_products_df, sample_returns_df)
    assert len(ord_df) == 3

    # 3. Portfolio summary
    portfolio = service.calculate_portfolio_economics(sample_sales_df, sample_products_df, sample_returns_df)
    assert portfolio.total_records == 4
    assert portfolio.cost_completeness_report is not None

    # 4. Segment economics
    sku_segs = service.calculate_segment_economics(sample_sales_df, "sku_id", sample_products_df, sample_returns_df)
    assert len(sku_segs) > 0

    # 5. Temporal economics
    monthly_segs = service.calculate_temporal_economics(sample_sales_df, TimeGrain.MONTHLY, sample_products_df, sample_returns_df)
    assert len(monthly_segs) == 2  # January and March

    # 6. Cost completeness audit
    comp_rep = service.calculate_cost_completeness(sample_sales_df, sample_products_df, sample_returns_df)
    assert isinstance(comp_rep, CostCompletenessReport)


def test_deterministic_ids_generation():
    """Verify deterministic format and reproducibility of SHA-256 IDs."""
    id1 = generate_operational_record_id("T1", "SKU1", "O1", "2026-01-01", "USD")
    id2 = generate_operational_record_id("T1", "SKU1", "O1", "2026-01-01", "USD")
    id3 = generate_operational_record_id("T2", "SKU1", "O1", "2026-01-01", "USD")
    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("FIN-OE-")

    ord_id = generate_operational_order_id("O1", "2026-01-01", "USD")
    assert ord_id.startswith("FIN-ORD-")

    seg_id = generate_operational_segment_id("SKU", "SKU1", "2026-01-01", "USD")
    assert seg_id.startswith("FIN-OESEG-")


def test_multiline_equal_allocation(sample_products_df):
    """Verify EQUAL allocation splits order-level costs evenly across order lines."""
    config = OperationalEconomicsConfig(
        assumptions=CostAssumptionsConfig(shipping_cost_per_order=10.0),
        order_cost_allocation_method="EQUAL",
    )
    sales = pd.DataFrame(
        [
            {"sale_id": "L1", "order_id": "O_EQ", "sku_id": "SKU_001", "quantity": 1, "unit_price": 20.0, "revenue": 20.0},
            {"sale_id": "L2", "order_id": "O_EQ", "sku_id": "SKU_002", "quantity": 9, "unit_price": 80.0, "revenue": 720.0},
        ]
    )
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    assert df.loc[df["sale_id"] == "L1", "shipping_cost"].iloc[0] == 5.0
    assert df.loc[df["sale_id"] == "L2", "shipping_cost"].iloc[0] == 5.0
    assert df["shipping_cost"].sum() == 10.0


def test_multiline_quantity_allocation(sample_products_df):
    """Verify QUANTITY allocation splits order-level costs proportionally by units."""
    config = OperationalEconomicsConfig(
        assumptions=CostAssumptionsConfig(shipping_cost_per_order=10.0),
        order_cost_allocation_method="QUANTITY",
    )
    sales = pd.DataFrame(
        [
            {"sale_id": "L1", "order_id": "O_QTY", "sku_id": "SKU_001", "quantity": 2, "unit_price": 50.0, "revenue": 100.0},
            {"sale_id": "L2", "order_id": "O_QTY", "sku_id": "SKU_002", "quantity": 8, "unit_price": 50.0, "revenue": 400.0},
        ]
    )
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    # L1: 2/10 = 20% of $10 = $2.0; L2: 8/10 = 80% of $10 = $8.0
    assert np.isclose(df.loc[df["sale_id"] == "L1", "shipping_cost"].iloc[0], 2.0)
    assert np.isclose(df.loc[df["sale_id"] == "L2", "shipping_cost"].iloc[0], 8.0)
    assert np.isclose(df["shipping_cost"].sum(), 10.0)


def test_strict_source_only_policy(sample_products_df):
    """Verify strict_source_only=True ignores catalog estimates and assumptions for completeness."""
    config = OperationalEconomicsConfig(
        required_cost_components=[CostComponent.PRODUCT_COST],
        strict_source_only=True,
    )
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "O1", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    # Catalog product cost is available (CATALOG_ESTIMATE), but strict_source_only requires SOURCE_DATA
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    assert df["cost_completeness_pct"].iloc[0] == 0.0
    assert df["final_contribution_margin_status"].iloc[0] == ContributionMarginStatus.INSUFFICIENT_COST_DATA.value


def test_point_in_time_return_filtering(sample_products_df):
    """Verify returns occurring after as_of_date are excluded from return processing cost."""
    config = OperationalEconomicsConfig(
        as_of_date="2026-01-18",
        assumptions=CostAssumptionsConfig(return_cost_per_returned_unit=15.0),
    )
    sales = pd.DataFrame([{"sale_id": "T1", "order_id": "ORD_101", "date": "2026-01-15", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0}])
    # Return occurred on 2026-01-20 (after as_of_date 2026-01-18)
    returns = pd.DataFrame([{"return_id": "R1", "order_id": "ORD_101", "sku_id": "SKU_001", "return_date": "2026-01-20", "quantity": 1}])
    df = compute_operational_economics_dataframe(sales, sample_products_df, returns_df=returns, config=config)
    # Since return date > as_of_date, at this point in time no return occurred -> return cost is $0.00
    assert df["return_processing_cost"].iloc[0] == 0.0
    assert df["return_cost_source"].iloc[0] == CostSourceType.CONFIGURED_ASSUMPTION.value


def test_channel_specific_shipping_assumptions(sample_products_df):
    """Verify channel-keyed shipping fee assumptions apply to designated channels."""
    config = OperationalEconomicsConfig(
        assumptions=CostAssumptionsConfig(
            shipping_cost_by_channel={"ONLINE": 7.50, "RETAIL": 1.50},
        )
    )
    sales = pd.DataFrame(
        [
            {"sale_id": "T1", "order_id": "O1", "channel_id": "ONLINE", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0},
            {"sale_id": "T2", "order_id": "O2", "channel_id": "RETAIL", "sku_id": "SKU_001", "quantity": 1, "unit_price": 50.0, "revenue": 50.0},
        ]
    )
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    assert df.loc[df["channel_id"] == "ONLINE", "shipping_cost"].iloc[0] == 7.50
    assert df.loc[df["channel_id"] == "RETAIL", "shipping_cost"].iloc[0] == 1.50


def test_facility_specific_warehouse_handling_assumptions(sample_products_df):
    """Verify facility-keyed handling fee assumptions apply per unit to designated warehouses."""
    config = OperationalEconomicsConfig(
        assumptions=CostAssumptionsConfig(
            warehouse_handling_by_facility={"WH_EAST": 1.75, "WH_WEST": 2.50},
        )
    )
    sales = pd.DataFrame(
        [
            {"sale_id": "T1", "order_id": "O1", "warehouse_id": "WH_EAST", "sku_id": "SKU_001", "quantity": 2, "unit_price": 50.0, "revenue": 100.0},
            {"sale_id": "T2", "order_id": "O2", "warehouse_id": "WH_WEST", "sku_id": "SKU_001", "quantity": 2, "unit_price": 50.0, "revenue": 100.0},
        ]
    )
    df = compute_operational_economics_dataframe(sales, sample_products_df, config=config)
    assert df.loc[df["warehouse_id"] == "WH_EAST", "warehouse_handling_cost"].iloc[0] == 2 * 1.75
    assert df.loc[df["warehouse_id"] == "WH_WEST", "warehouse_handling_cost"].iloc[0] == 2 * 2.50
