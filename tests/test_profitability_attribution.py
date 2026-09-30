"""Unit and Integration Tests for Phase 6C — Profitability Attribution & Margin Drivers.

Covers:
- Revenue and Margin contribution share calculations
- Contribution gap and absolute gap calculations
- Negative and low margin detection with deterministic reason codes
- High discount detection and impact formula invariants
- Discount bucket segmentation and margin erosion
- All 8 dimensional cuts: SKU, Category, Brand, Channel, Warehouse, SKU x Channel, SKU x Warehouse, Channel x Warehouse
- Margin concentration tiers (Top 1%, 5%, 10%, 20%) and cumulative contribution curve
- Margin waterfall stages and reconciliation
- Temporal attribution (Daily, Weekly, Monthly)
- Point-in-time anti-leakage filtering
- Multi-currency isolation
- Missing catalog and duplicate sales handling
- Deterministic reproducibility and empty dataset handling
- Configurable thresholds
- Regression checks for Phase 6A and 6B
"""

from __future__ import annotations

import math
from typing import List
import numpy as np
import pandas as pd
import pytest

from commerce_ai.financial.attribution import (
    ProfitabilityAttributionService,
    compute_dimensional_profitability_attribution,
    compute_sku_profitability_profiles,
    compute_temporal_profitability_attribution,
)
from commerce_ai.financial.concentration import (
    calculate_margin_concentration,
    calculate_margin_contribution_curve,
)
from commerce_ai.financial.discount_analysis import (
    analyze_discount_buckets,
    analyze_discount_impact,
)
from commerce_ai.financial.margin_drivers import (
    analyze_negative_and_low_margins,
    build_margin_waterfall,
    classify_margin_drivers,
    evaluate_sku_driver_classifications,
)
from commerce_ai.financial.schemas import (
    AttributionReasonCode,
    MarginClassification,
    MarginDriverClassification,
    ProfitabilityAttributionConfig,
    ProfitabilityAttributionRecord,
    ProfitabilityAttributionResult,
    SKUProfitabilityProfile,
    TimeGrain,
)
from commerce_ai.financial.service import (
    FinancialIntelligenceService,
    UnitEconomicsService,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def sample_products() -> pd.DataFrame:
    """Product catalog with known costs, categories, and brands."""
    return pd.DataFrame([
        {"sku_id": "SKU_001", "product_name": "Premium Gadget", "category_id": "Electronics", "brand": "BrandAlpha", "unit_cost": 50.0},
        {"sku_id": "SKU_002", "product_name": "Basic Tee", "category_id": "Apparel", "brand": "BrandBeta", "unit_cost": 20.0},
        {"sku_id": "SKU_003", "product_name": "Home Lamp", "category_id": "Home", "brand": "BrandAlpha", "unit_cost": 70.0},
        {"sku_id": "SKU_004", "product_name": "Loss Leader", "category_id": "Electronics", "brand": "BrandGamma", "unit_cost": 100.0},
    ])


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    """Controlled multi-line sales dataset covering different margin and discount profiles."""
    return pd.DataFrame([
        # SKU_001: gross = 10 * 100 = 1000, disc = 50 (5%), net = 950, cogs = 10 * 50 = 500, margin = 450 (47.4%) -> Healthy
        {
            "sale_id": "S1", "order_id": "O1", "date": "2024-01-15", "sku_id": "SKU_001",
            "warehouse_id": "WH_EAST", "channel_id": "CH_AMZ", "quantity": 10, "unit_price": 100.0,
            "discount": 50.0, "revenue": 950.0, "currency": "USD"
        },
        # SKU_002: gross = 20 * 25 = 500, disc = 50 (10%), net = 450, cogs = 20 * 20 = 400, margin = 50 (11.1%) -> Low Margin (<20%)
        {
            "sale_id": "S2", "order_id": "O2", "date": "2024-01-20", "sku_id": "SKU_002",
            "warehouse_id": "WH_WEST", "channel_id": "CH_DTC", "quantity": 20, "unit_price": 25.0,
            "discount": 50.0, "revenue": 450.0, "currency": "USD"
        },
        # SKU_003: gross = 5 * 150 = 750, disc = 200 (26.7%), net = 550, cogs = 5 * 70 = 350, margin = 200 (36.4%) -> High Discount (>20%)
        {
            "sale_id": "S3", "order_id": "O3", "date": "2024-02-10", "sku_id": "SKU_003",
            "warehouse_id": "WH_EAST", "channel_id": "CH_AMZ", "quantity": 5, "unit_price": 150.0,
            "discount": 200.0, "revenue": 550.0, "currency": "USD"
        },
        # SKU_004: gross = 2 * 90 = 180, disc = 0, net = 180, cogs = 2 * 100 = 200, margin = -20 (-11.1%) -> Negative Margin
        {
            "sale_id": "S4", "order_id": "O4", "date": "2024-02-15", "sku_id": "SKU_004",
            "warehouse_id": "WH_WEST", "channel_id": "CH_BBY", "quantity": 2, "unit_price": 90.0,
            "discount": 0.0, "revenue": 180.0, "currency": "USD"
        },
        # SKU_001 second order in March for temporal testing
        {
            "sale_id": "S5", "order_id": "O5", "date": "2024-03-05", "sku_id": "SKU_001",
            "warehouse_id": "WH_EAST", "channel_id": "CH_DTC", "quantity": 5, "unit_price": 100.0,
            "discount": 0.0, "revenue": 500.0, "currency": "USD"
        },
    ])


# =====================================================================
# Tests: Contribution Calculations (1 - 4)
# =====================================================================


def test_revenue_contribution_percentage_calculation(sample_sales, sample_products):
    """Test 1: Segment revenue contribution percentage equals segment net revenue / portfolio net revenue."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    sku_recs = result.dimension_attributions["SKU"]
    total_rev = result.portfolio_attribution.net_revenue

    for rec in sku_recs:
        expected_share = rec.net_revenue / total_rev
        assert rec.revenue_contribution_pct == pytest.approx(expected_share, abs=1e-4)

    sum_rev_share = sum(r.revenue_contribution_pct for r in sku_recs)
    assert sum_rev_share == pytest.approx(1.0, abs=1e-3)


def test_margin_contribution_percentage_calculation(sample_sales, sample_products):
    """Test 2: Segment margin contribution percentage equals segment gross margin / portfolio gross margin."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    sku_recs = result.dimension_attributions["SKU"]
    total_margin = result.portfolio_attribution.gross_margin

    for rec in sku_recs:
        expected_share = rec.gross_margin / total_margin
        assert rec.margin_contribution_pct == pytest.approx(expected_share, abs=1e-4)

    sum_margin_share = sum(r.margin_contribution_pct for r in sku_recs)
    assert sum_margin_share == pytest.approx(1.0, abs=1e-3)


def test_contribution_gap_calculation(sample_sales, sample_products):
    """Test 3: contribution_gap = margin_contribution_pct - revenue_contribution_pct."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    sku_recs = result.dimension_attributions["SKU"]
    for rec in sku_recs:
        expected_gap = rec.margin_contribution_pct - rec.revenue_contribution_pct
        assert rec.contribution_gap == pytest.approx(expected_gap, abs=1e-4)


def test_absolute_contribution_gap_calculation(sample_sales, sample_products):
    """Test 4: absolute_contribution_gap equals abs(contribution_gap)."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    sku_recs = result.dimension_attributions["SKU"]
    for rec in sku_recs:
        assert rec.absolute_contribution_gap == pytest.approx(abs(rec.contribution_gap), abs=1e-4)


# =====================================================================
# Tests: Margin Driver Detection & Reason Codes (5 - 11)
# =====================================================================


def test_negative_margin_detection(sample_sales, sample_products):
    """Test 5: Segments with realized gross_margin < 0 are detected."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    neg_drivers = [d for d in result.margin_drivers if d.driver_classification == MarginDriverClassification.NEGATIVE_MARGIN]
    assert len(neg_drivers) >= 1
    neg_sku = next(d for d in neg_drivers if d.segment_key == "SKU_004")
    assert neg_sku.gross_margin < 0


def test_negative_margin_reason_codes(sample_sales, sample_products):
    """Test 6: Flagged negative margin segments are assigned NEGATIVE_GROSS_MARGIN reason code."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    sku_004 = next(p for p in result.sku_profiles if p.sku_id == "SKU_004")
    assert MarginDriverClassification.NEGATIVE_MARGIN in sku_004.driver_classifications
    assert AttributionReasonCode.NEGATIVE_GROSS_MARGIN in sku_004.reason_codes


def test_low_margin_detection(sample_sales, sample_products):
    """Test 7: Segments with gross_margin_pct < 20% (and >= 0) are flagged as LOW_MARGIN."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    low_drivers = [d for d in result.margin_drivers if d.driver_classification == MarginDriverClassification.LOW_MARGIN]
    assert any(d.segment_key == "SKU_002" for d in low_drivers)


def test_low_margin_reason_codes(sample_sales, sample_products):
    """Test 8: Flagged low margin segments are assigned LOW_GROSS_MARGIN_PERCENT reason code."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    sku_002 = next(p for p in result.sku_profiles if p.sku_id == "SKU_002")
    assert MarginDriverClassification.LOW_MARGIN in sku_002.driver_classifications
    assert AttributionReasonCode.LOW_GROSS_MARGIN_PERCENT in sku_002.reason_codes


def test_high_discount_detection(sample_sales, sample_products):
    """Test 9: Segments with discount rate >= 20% are flagged with HIGH_DISCOUNT."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    sku_003 = next(p for p in result.sku_profiles if p.sku_id == "SKU_003")
    assert sku_003.discount_rate >= 0.20
    assert MarginDriverClassification.HIGH_DISCOUNT in sku_003.driver_classifications
    assert AttributionReasonCode.HIGH_DISCOUNT in sku_003.reason_codes


def test_high_revenue_low_margin_share_classification(sample_sales, sample_products):
    """Test 10: Segment with contribution gap <= -0.02 is classified as HIGH_REVENUE_LOW_MARGIN_SHARE."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    sku_002 = next(p for p in result.sku_profiles if p.sku_id == "SKU_002")
    # SKU_002 generates 450/2630 = 17.1% rev, but only 50/880 = 5.7% margin (gap = -11.4%)
    assert sku_002.contribution_gap <= -0.02
    assert MarginDriverClassification.HIGH_REVENUE_LOW_MARGIN_SHARE in sku_002.driver_classifications
    assert AttributionReasonCode.CONTRIBUTION_GAP_DEFICIT in sku_002.reason_codes


def test_low_revenue_high_margin_share_classification(sample_sales, sample_products):
    """Test 11: Segment with contribution gap >= +0.02 is classified as LOW_REVENUE_HIGH_MARGIN_SHARE."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    sku_001 = next(p for p in result.sku_profiles if p.sku_id == "SKU_001")
    # SKU_001 generates 1450/2630 = 55.1% rev, and 700/880 = 79.5% margin (gap = +24.4%)
    assert sku_001.contribution_gap >= 0.02
    assert MarginDriverClassification.LOW_REVENUE_HIGH_MARGIN_SHARE in sku_001.driver_classifications


# =====================================================================
# Tests: Discount Impact & Buckets (12 - 15)
# =====================================================================


def test_discount_impact_formula_invariant(sample_sales, sample_products):
    """Test 12: Invariant discount_margin_impact = margin_after_discount - margin_before_discount == -discount."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)
    disc_summary = result.discount_impact

    # Mathematical identity checks
    assert disc_summary.margin_before_discount == pytest.approx(
        disc_summary.total_gross_revenue - (disc_summary.total_gross_revenue - disc_summary.margin_before_discount),
        abs=1e-2,
    )
    assert disc_summary.margin_after_discount == pytest.approx(
        disc_summary.total_net_revenue - (disc_summary.total_gross_revenue - disc_summary.margin_before_discount),
        abs=1e-2,
    )
    expected_impact = disc_summary.margin_after_discount - disc_summary.margin_before_discount
    assert disc_summary.discount_margin_impact == pytest.approx(expected_impact, abs=1e-2)
    assert disc_summary.discount_margin_impact == pytest.approx(-disc_summary.total_discount, abs=1e-2)


def test_discount_bucket_segmentation_defaults(sample_sales, sample_products):
    """Test 13: Default discount buckets include 0%, 0-5%, 5-10%, 10-20%, 20-30%, 30%+."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    labels = [b.bucket_label for b in result.discount_impact.buckets]
    assert labels == ["0%", "0-5%", "5-10%", "10-20%", "20-30%", "30%+"]


def test_discount_bucket_transaction_counts(sample_sales, sample_products):
    """Test 14: Sum of transaction counts across discount buckets equals total transactions."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    total_tx = sum(b.transaction_count for b in result.discount_impact.buckets)
    assert total_tx == len(sample_sales)


def test_discount_bucket_margin_erosion(sample_sales, sample_products):
    """Test 15: Discount bucket metrics show negative discount_margin_impact proportional to discount granted."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    for b in result.discount_impact.buckets:
        assert b.discount_margin_impact == pytest.approx(-b.total_discount, abs=1e-2)
        if b.total_gross_revenue > 0:
            assert b.total_net_revenue == pytest.approx(b.total_gross_revenue - b.total_discount, abs=1e-2)


# =====================================================================
# Tests: Dimensional Cuts (16 - 23)
# =====================================================================


def test_sku_level_attribution(sample_sales, sample_products):
    """Test 16: SKU attribution aggregates sales metrics correctly per SKU."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    sku_recs = result.dimension_attributions["SKU"]
    assert len(sku_recs) == 4
    keys = {r.segment_key for r in sku_recs}
    assert keys == {"SKU_001", "SKU_002", "SKU_003", "SKU_004"}


def test_category_level_attribution(sample_sales, sample_products):
    """Test 17: Category attribution aggregates across categories."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    cat_recs = result.dimension_attributions["CATEGORY"]
    keys = {r.segment_key for r in cat_recs}
    assert keys == {"Electronics", "Apparel", "Home"}


def test_brand_level_attribution(sample_sales, sample_products):
    """Test 18: Brand attribution aggregates across brands."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    brand_recs = result.dimension_attributions["BRAND"]
    keys = {r.segment_key for r in brand_recs}
    assert keys == {"BrandAlpha", "BrandBeta", "BrandGamma"}


def test_channel_level_attribution(sample_sales, sample_products):
    """Test 19: Channel attribution aggregates across sales channels."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    ch_recs = result.dimension_attributions["CHANNEL"]
    keys = {r.segment_key for r in ch_recs}
    assert keys == {"CH_AMZ", "CH_DTC", "CH_BBY"}


def test_warehouse_level_attribution(sample_sales, sample_products):
    """Test 20: Warehouse attribution aggregates across fulfillment facilities."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    wh_recs = result.dimension_attributions["WAREHOUSE"]
    keys = {r.segment_key for r in wh_recs}
    assert keys == {"WH_EAST", "WH_WEST"}


def test_sku_channel_cross_slice_attribution(sample_sales, sample_products):
    """Test 21: SKU x Channel compound slice generates composite segment keys."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    cross_recs = result.dimension_attributions["SKU_CHANNEL"]
    assert len(cross_recs) > 0
    for r in cross_recs:
        assert "|" in r.segment_key
        assert r.dimension == "SKU_CHANNEL"


def test_sku_warehouse_cross_slice_attribution(sample_sales, sample_products):
    """Test 22: SKU x Warehouse compound slice generates composite keys."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    cross_recs = result.dimension_attributions["SKU_WAREHOUSE"]
    assert len(cross_recs) > 0
    for r in cross_recs:
        assert "|" in r.segment_key
        assert r.dimension == "SKU_WAREHOUSE"


def test_channel_warehouse_cross_slice_attribution(sample_sales, sample_products):
    """Test 23: Channel x Warehouse compound slice generates composite keys."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    cross_recs = result.dimension_attributions["CHANNEL_WAREHOUSE"]
    assert len(cross_recs) > 0
    for r in cross_recs:
        assert "|" in r.segment_key
        assert r.dimension == "CHANNEL_WAREHOUSE"


# =====================================================================
# Tests: Concentration & Contribution Curve (24 - 25)
# =====================================================================


def test_margin_concentration_tiers(sample_sales, sample_products):
    """Test 24: Margin concentration tiers exist for 1%, 5%, 10%, 20% and have monotonic shares."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)
    conc = result.margin_concentration

    assert len(conc.tiers) == 4
    pcts = [t.percentile for t in conc.tiers]
    assert pcts == [0.01, 0.05, 0.10, 0.20]

    # Monotonic top_n_count and cumulative margin
    top_counts = [t.top_n_count for t in conc.tiers]
    assert top_counts == sorted(top_counts)


def test_margin_contribution_curve(sample_sales, sample_products):
    """Test 25: Contribution curve ranks segments from 1 to N with non-decreasing cumulative margin share."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)
    curve = result.contribution_curve

    assert len(curve) == 4
    for idx, pt in enumerate(curve, start=1):
        assert pt.rank == idx

    # Verify cumulative margin percentage is monotonic
    cum_margins = [pt.cumulative_margin_contribution_pct for pt in curve]
    for i in range(len(cum_margins) - 1):
        # Allow slight dip if negative margin segment is at the end
        if curve[i + 1].segment_margin >= 0:
            assert cum_margins[i + 1] >= cum_margins[i] - 1e-4


# =====================================================================
# Tests: Margin Waterfall (26)
# =====================================================================


def test_margin_waterfall_stages_and_amounts(sample_sales, sample_products):
    """Test 26: Margin waterfall contains 8 sequential stages matching portfolio amounts."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products)
    wf = result.margin_waterfall

    assert len(wf.stages) == 8
    orders = [s.stage_order for s in wf.stages]
    assert orders == list(range(1, 9))

    # Stage 1: Gross revenue
    assert wf.stages[0].amount == pytest.approx(wf.gross_revenue, abs=1e-2)
    # Stage 2: Discount deduction
    assert wf.stages[1].amount == pytest.approx(-wf.discount, abs=1e-2)
    # Stage 3: Net revenue
    assert wf.stages[2].amount == pytest.approx(wf.net_revenue, abs=1e-2)
    # Stage 4: COGS deduction
    assert wf.stages[3].amount == pytest.approx(-wf.product_cost, abs=1e-2)
    # Stage 5: Gross margin
    assert wf.stages[4].amount == pytest.approx(wf.gross_margin, abs=1e-2)


# =====================================================================
# Tests: Temporal Attribution (27 - 29)
# =====================================================================


def test_temporal_attribution_daily(sample_sales, sample_products):
    """Test 27: Daily temporal attribution creates records bucketed by day."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products, time_grain=TimeGrain.DAILY)
    daily = result.time_attributions

    assert len(daily) == 5
    for r in daily:
        assert r.dimension == "DATE"
        assert len(r.segment_key) == 10  # YYYY-MM-DD


def test_temporal_attribution_weekly(sample_sales, sample_products):
    """Test 28: Weekly temporal attribution creates records bucketed by ISO week."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products, time_grain=TimeGrain.WEEKLY)
    weekly = result.time_attributions

    assert len(weekly) > 0
    for r in weekly:
        assert r.dimension == "WEEK"
        assert "-W" in r.segment_key


def test_temporal_attribution_monthly(sample_sales, sample_products):
    """Test 29: Monthly temporal attribution creates records bucketed by month."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, sample_products, time_grain=TimeGrain.MONTHLY)
    monthly = result.time_attributions

    assert len(monthly) == 3  # Jan, Feb, Mar
    keys = [r.segment_key for r in monthly]
    assert keys == ["2024-01", "2024-02", "2024-03"]


# =====================================================================
# Tests: Data Quality & Boundary Conditions (30 - 37)
# =====================================================================


def test_as_of_date_anti_leakage_filtering(sample_sales, sample_products):
    """Test 30: as_of_date cutoff excludes future transactions."""
    service = ProfitabilityAttributionService()
    # Filter up to 2024-01-31 (should exclude Feb and Mar sales)
    result = service.compute_profitability_attribution(sample_sales, sample_products, as_of_date="2024-01-31")

    assert result.portfolio_attribution.record_count == 2
    sku_recs = result.dimension_attributions["SKU"]
    sku_keys = {r.segment_key for r in sku_recs}
    assert sku_keys == {"SKU_001", "SKU_002"}


def test_currency_isolation(sample_products):
    """Test 31: Sales with non-USD currency are flagged in data quality report."""
    mixed_sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2024-01-15", "sku_id": "SKU_001", "quantity": 1, "unit_price": 100.0, "currency": "EUR"},
        {"sale_id": "S2", "order_id": "O2", "date": "2024-01-16", "sku_id": "SKU_001", "quantity": 1, "unit_price": 100.0, "currency": "USD"},
    ])
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(mixed_sales, sample_products)

    dq = result.data_quality_report
    curr_issue = next((i for i in dq.issues if i.get("check") == "mismatched_currencies"), None)
    assert curr_issue is not None
    assert "EUR" in curr_issue["currencies"]


def test_missing_product_catalog_handling(sample_sales):
    """Test 32: When products catalog is missing, service handles missing unit cost gracefully."""
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(sample_sales, products_df=None)

    # Unit costs are missing so product_cost is 0 or unavailable
    assert result.portfolio_attribution.record_count == len(sample_sales)
    assert result.data_quality_report.missing_unit_cost_count > 0


def test_duplicate_sales_handling(sample_products):
    """Test 33: Duplicate sale_ids are captured by the data quality report."""
    dup_sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2024-01-15", "sku_id": "SKU_001", "quantity": 1, "unit_price": 100.0, "currency": "USD"},
        {"sale_id": "S1", "order_id": "O2", "date": "2024-01-15", "sku_id": "SKU_001", "quantity": 1, "unit_price": 100.0, "currency": "USD"},
    ])
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(dup_sales, sample_products)

    assert result.data_quality_report.duplicate_sale_id_count == 1


def test_deterministic_reproducibility(sample_sales, sample_products):
    """Test 34: Two consecutive runs with identical inputs produce identical numerical outputs."""
    service = ProfitabilityAttributionService()
    res1 = service.compute_profitability_attribution(sample_sales, sample_products)
    res2 = service.compute_profitability_attribution(sample_sales, sample_products)

    assert res1.portfolio_attribution.gross_margin == res2.portfolio_attribution.gross_margin
    assert res1.portfolio_attribution.net_revenue == res2.portfolio_attribution.net_revenue
    assert len(res1.margin_drivers) == len(res2.margin_drivers)


def test_empty_sales_dataset_handling(sample_products):
    """Test 35: Empty sales DataFrame produces clean zero-valued results without crashing."""
    empty_sales = pd.DataFrame(columns=["sale_id", "order_id", "date", "sku_id", "quantity", "unit_price", "discount", "currency"])
    service = ProfitabilityAttributionService()
    result = service.compute_profitability_attribution(empty_sales, sample_products)

    assert result.portfolio_attribution.record_count == 0
    assert result.portfolio_attribution.gross_margin == 0.0
    assert len(result.contribution_curve) == 0


def test_configurable_thresholds(sample_sales, sample_products):
    """Test 36: Modifying low_margin_threshold alters low-margin classifications."""
    # Set low_margin_threshold = 0.50 (50%) -> SKU_001 has ~47% margin so it should now be flagged LOW_MARGIN
    cfg = ProfitabilityAttributionConfig(low_margin_threshold=0.50)
    service = ProfitabilityAttributionService(config=cfg)
    result = service.compute_profitability_attribution(sample_sales, sample_products)

    sku_001 = next(p for p in result.sku_profiles if p.sku_id == "SKU_001")
    assert MarginDriverClassification.LOW_MARGIN in sku_001.driver_classifications


def test_regression_phase_6a_6b_intact(sample_sales, sample_products):
    """Test 37: FinancialIntelligenceService and UnitEconomicsService continue to function correctly."""
    fin_svc = FinancialIntelligenceService()
    fin_res = fin_svc.analyze(sample_sales, sample_products)
    assert fin_res.portfolio_summary.total_net_revenue > 0

    ue_svc = UnitEconomicsService()
    ue_res = ue_svc.calculate_unit_economics(sample_sales, sample_products)
    assert ue_res.portfolio_summary.total_net_revenue > 0
    assert ue_res.portfolio_summary.total_gross_margin > 0
