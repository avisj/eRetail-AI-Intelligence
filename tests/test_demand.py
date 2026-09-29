"""Tests for Demand Reconstruction Engine."""

import pandas as pd
import pytest

from commerce_ai.analytics.demand import build_daily_demand


@pytest.fixture
def sample_sales_and_inventory():
    """Create minimal sales and inventory data for testing demand grid reconstruction."""
    sales = pd.DataFrame([
        # Day 1: SKU_1 at WH_1 sells 10
        {"date": "2025-01-01", "sku_id": "SKU_1", "warehouse_id": "WH_1", "quantity": 10, "revenue": 100.0},
        # Day 3: SKU_1 at WH_1 sells 15 (Day 2 has zero sales!)
        {"date": "2025-01-03", "sku_id": "SKU_1", "warehouse_id": "WH_1", "quantity": 15, "revenue": 150.0},
        # Day 2: SKU_2 at WH_2 sells 5
        {"date": "2025-01-02", "sku_id": "SKU_2", "warehouse_id": "WH_2", "quantity": 5, "revenue": 50.0},
    ])

    inventory = pd.DataFrame([
        # Day 1 snapshots
        {"snapshot_date": "2025-01-01", "sku_id": "SKU_1", "warehouse_id": "WH_1", "available_qty": 50, "reserved_qty": 0, "in_transit_qty": 0, "damaged_qty": 0},
        {"snapshot_date": "2025-01-01", "sku_id": "SKU_1", "warehouse_id": "WH_2", "available_qty": 20, "reserved_qty": 0, "in_transit_qty": 0, "damaged_qty": 0},
        {"snapshot_date": "2025-01-01", "sku_id": "SKU_2", "warehouse_id": "WH_1", "available_qty": 30, "reserved_qty": 0, "in_transit_qty": 0, "damaged_qty": 0},
        {"snapshot_date": "2025-01-01", "sku_id": "SKU_2", "warehouse_id": "WH_2", "available_qty": 40, "reserved_qty": 0, "in_transit_qty": 0, "damaged_qty": 0},
    ])

    return sales, inventory


class TestDemandReconstruction:
    def test_grid_completeness_and_zero_filling(self, sample_sales_and_inventory):
        sales, inventory = sample_sales_and_inventory
        daily_demand = build_daily_demand(sales, inventory)

        # 3 days (Jan 1, 2, 3) x 2 SKUs x 2 WHs = 12 total rows
        assert len(daily_demand) == 12

        # Verify Day 2 for SKU_1 at WH_1 has 0 units_sold and 0.0 revenue
        day_2_sku1_wh1 = daily_demand[
            (daily_demand["date"] == "2025-01-02")
            & (daily_demand["sku_id"] == "SKU_1")
            & (daily_demand["warehouse_id"] == "WH_1")
        ]
        assert len(day_2_sku1_wh1) == 1
        assert day_2_sku1_wh1["units_sold"].iloc[0] == 0
        assert day_2_sku1_wh1["revenue"].iloc[0] == 0.0

    def test_inventory_forward_fill(self, sample_sales_and_inventory):
        sales, inventory = sample_sales_and_inventory
        daily_demand = build_daily_demand(sales, inventory)

        # SKU_1 at WH_1 had snapshot of 50 on Jan 1; on Jan 2 and Jan 3 it should be forward-filled as 50
        series = daily_demand[(daily_demand["sku_id"] == "SKU_1") & (daily_demand["warehouse_id"] == "WH_1")]
        assert all(series["available_qty"] == 50)

    def test_filtering_by_sku_and_warehouse(self, sample_sales_and_inventory):
        sales, inventory = sample_sales_and_inventory
        daily_demand = build_daily_demand(
            sales,
            inventory,
            skus=["SKU_1"],
            warehouse_ids=["WH_1"],
        )
        # 3 days x 1 SKU x 1 WH = 3 rows
        assert len(daily_demand) == 3
        assert set(daily_demand["sku_id"]) == {"SKU_1"}
        assert set(daily_demand["warehouse_id"]) == {"WH_1"}
