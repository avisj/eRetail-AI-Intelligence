"""Unit Tests for Supplier and SKU Lead Time Profiling."""

import pandas as pd
import pytest

from commerce_ai.inventory.lead_time import (
    LeadTimeConfig,
    calculate_po_lead_times,
    profile_supplier_lead_times,
    profile_sku_lead_times,
)


@pytest.fixture
def delivered_purchases():
    return pd.DataFrame([
        # Supplier 1: 4 deliveries (lead times: 10, 12, 14, 10 days)
        {"purchase_order_id": "PO_1", "supplier_id": "SUPP_1", "sku_id": "SKU_A", "order_date": "2024-01-01", "expected_delivery_date": "2024-01-12", "actual_delivery_date": "2024-01-11"}, # 10d, on-time
        {"purchase_order_id": "PO_2", "supplier_id": "SUPP_1", "sku_id": "SKU_A", "order_date": "2024-01-10", "expected_delivery_date": "2024-01-20", "actual_delivery_date": "2024-01-22"}, # 12d, late
        {"purchase_order_id": "PO_3", "supplier_id": "SUPP_1", "sku_id": "SKU_A", "order_date": "2024-01-20", "expected_delivery_date": "2024-02-05", "actual_delivery_date": "2024-02-03"}, # 14d, on-time
        {"purchase_order_id": "PO_4", "supplier_id": "SUPP_1", "sku_id": "SKU_A", "order_date": "2024-02-01", "expected_delivery_date": "2024-02-15", "actual_delivery_date": "2024-02-11"}, # 10d, on-time
        # Supplier 2: 1 delivery (insufficient for min_history=3)
        {"purchase_order_id": "PO_5", "supplier_id": "SUPP_2", "sku_id": "SKU_B", "order_date": "2024-01-05", "expected_delivery_date": "2024-01-15", "actual_delivery_date": "2024-01-20"}, # 15d, late
        # Undelivered PO (should be excluded)
        {"purchase_order_id": "PO_6", "supplier_id": "SUPP_2", "sku_id": "SKU_B", "order_date": "2024-02-01", "expected_delivery_date": "2024-02-15", "actual_delivery_date": None},
    ])


@pytest.fixture
def supplier_catalog():
    return pd.DataFrame([
        {"supplier_id": "SUPP_1", "supplier_name": "Supplier One", "average_lead_time_days": 10, "minimum_order_quantity": 50},
        {"supplier_id": "SUPP_2", "supplier_name": "Supplier Two", "average_lead_time_days": 21, "minimum_order_quantity": 100},
        {"supplier_id": "SUPP_3", "supplier_name": "Supplier Three", "average_lead_time_days": 7, "minimum_order_quantity": 25},
    ])


@pytest.fixture
def product_catalog():
    return pd.DataFrame([
        {"sku_id": "SKU_A", "product_name": "Item A", "preferred_supplier_id": "SUPP_1"},
        {"sku_id": "SKU_B", "product_name": "Item B", "preferred_supplier_id": "SUPP_2"},
        {"sku_id": "SKU_C", "product_name": "Item C", "preferred_supplier_id": "SUPP_3"},
    ])


class TestLeadTimeProfiler:
    def test_calculate_po_lead_times(self, delivered_purchases):
        res = calculate_po_lead_times(delivered_purchases)
        # 5 delivered orders (PO_6 is not delivered)
        assert len(res) == 5
        po1 = res[res["purchase_order_id"] == "PO_1"].iloc[0]
        assert po1["actual_lead_time_days"] == 10.0
        assert bool(po1["is_on_time"]) is True

        po2 = res[res["purchase_order_id"] == "PO_2"].iloc[0]
        assert po2["actual_lead_time_days"] == 12.0
        assert bool(po2["is_on_time"]) is False

    def test_profile_supplier_lead_times_with_fallback(self, delivered_purchases, supplier_catalog):
        cfg = LeadTimeConfig(min_history=3)
        profiles = profile_supplier_lead_times(delivered_purchases, supplier_catalog, config=cfg)

        # SUPP_1 has 4 observations >= min_history(3) -> empirical
        supp1 = profiles["SUPP_1"]
        assert supp1.sample_size == 4
        assert supp1.is_fallback is False
        assert supp1.mean_lead_time_days == (10 + 12 + 14 + 10) / 4.0  # 11.5
        assert supp1.on_time_delivery_rate == 0.75  # 3 of 4 on-time

        # SUPP_2 has 1 observation < min_history(3) -> fallback to catalog average 21
        supp2 = profiles["SUPP_2"]
        assert supp2.sample_size == 1
        assert supp2.is_fallback is True
        assert supp2.mean_lead_time_days == 21.0

        # SUPP_3 has 0 observations -> fallback to catalog average 7
        supp3 = profiles["SUPP_3"]
        assert supp3.sample_size == 0
        assert supp3.is_fallback is True
        assert supp3.mean_lead_time_days == 7.0

    def test_profile_sku_lead_times_hierarchy(self, delivered_purchases, product_catalog, supplier_catalog):
        cfg = LeadTimeConfig(min_history=3)
        sku_profiles = profile_sku_lead_times(
            purchases=delivered_purchases,
            products=product_catalog,
            suppliers=supplier_catalog,
            config=cfg,
        )

        # SKU_A has 4 POs -> empirical
        assert sku_profiles["SKU_A"].sample_size == 4
        assert sku_profiles["SKU_A"].is_fallback is False
        assert sku_profiles["SKU_A"].mean_lead_time_days == 11.5

        # SKU_B has 1 PO -> fallback to supplier SUPP_2 catalog average (21)
        assert sku_profiles["SKU_B"].sample_size == 1
        assert sku_profiles["SKU_B"].is_fallback is True
        assert sku_profiles["SKU_B"].mean_lead_time_days == 21.0

        # SKU_C has 0 POs -> fallback to supplier SUPP_3 catalog average (7)
        assert sku_profiles["SKU_C"].sample_size == 0
        assert sku_profiles["SKU_C"].is_fallback is True
        assert sku_profiles["SKU_C"].mean_lead_time_days == 7.0

    def test_as_of_date_lead_time_filtering(self, supplier_catalog, product_catalog):
        """Fix 3: Completed POs delivered after as_of_date must not leak into empirical lead time."""
        as_of = "2024-01-15"
        po_df = pd.DataFrame([
            # 1. Delivered before as_of_date -> included (10 days)
            {"purchase_order_id": "PO_PAST_1", "supplier_id": "SUPP_1", "sku_id": "SKU_A", "order_date": "2024-01-01", "expected_delivery_date": "2024-01-12", "actual_delivery_date": "2024-01-10"},
            # 2. Delivered on as_of_date -> included (10 days)
            {"purchase_order_id": "PO_PAST_2", "supplier_id": "SUPP_1", "sku_id": "SKU_A", "order_date": "2024-01-05", "expected_delivery_date": "2024-01-15", "actual_delivery_date": "2024-01-15"},
            # 3. Delivered before as_of_date -> included (8 days)
            {"purchase_order_id": "PO_PAST_3", "supplier_id": "SUPP_1", "sku_id": "SKU_A", "order_date": "2024-01-02", "expected_delivery_date": "2024-01-11", "actual_delivery_date": "2024-01-10"},
            # 4. Ordered before as_of_date, but delivered AFTER as_of_date -> MUST BE EXCLUDED from empirical lead time
            {"purchase_order_id": "PO_FUTURE_DELIVERY", "supplier_id": "SUPP_1", "sku_id": "SKU_A", "order_date": "2024-01-08", "expected_delivery_date": "2024-01-20", "actual_delivery_date": "2024-01-25"},
            # 5. Ordered after as_of_date -> MUST BE EXCLUDED
            {"purchase_order_id": "PO_FUTURE_ORDER", "supplier_id": "SUPP_1", "sku_id": "SKU_A", "order_date": "2024-01-20", "expected_delivery_date": "2024-02-01", "actual_delivery_date": "2024-01-30"},
        ])

        # calculate_po_lead_times filtering
        valid_pos = calculate_po_lead_times(po_df, as_of_date=as_of)
        assert len(valid_pos) == 3
        assert set(valid_pos["purchase_order_id"]) == {"PO_PAST_1", "PO_PAST_2", "PO_PAST_3"}

        # profile_sku_lead_times filtering
        cfg = LeadTimeConfig(min_history=3)
        sku_profiles = profile_sku_lead_times(
            purchases=po_df,
            products=product_catalog,
            suppliers=supplier_catalog,
            config=cfg,
            as_of_date=as_of,
        )
        profile_a = sku_profiles["SKU_A"]
        assert profile_a.sample_size == 3
        # Mean of 9, 10, 8 = 27 / 3 = 9.0
        assert pytest.approx(profile_a.mean_lead_time_days, rel=1e-3) == (9 + 10 + 8) / 3.0
        assert profile_a.is_fallback is False

