"""Unit Tests for Inventory Position Accounting."""

from datetime import date
import pandas as pd
import pytest

from commerce_ai.inventory.position import (
    calculate_inventory_positions,
    extract_position_records,
)
from commerce_ai.inventory.schemas import InventoryPositionRecord


@pytest.fixture
def sample_snapshots():
    return pd.DataFrame([
        {
            "snapshot_date": "2024-01-01",
            "sku_id": "SKU_001",
            "warehouse_id": "WH_01",
            "available_qty": 50,
            "reserved_qty": 10,
            "in_transit_qty": 20,
            "damaged_qty": 2,
        },
        {
            "snapshot_date": "2024-01-02",
            "sku_id": "SKU_001",
            "warehouse_id": "WH_01",
            "available_qty": 45,
            "reserved_qty": 5,
            "in_transit_qty": 20,
            "damaged_qty": 2,
        },
        {
            "snapshot_date": "2024-01-02",
            "sku_id": "SKU_002",
            "warehouse_id": "WH_01",
            "available_qty": 100,
            "reserved_qty": 0,
            "in_transit_qty": 0,
            "damaged_qty": 0,
        },
    ])


@pytest.fixture
def sample_purchases():
    return pd.DataFrame([
        {
            "purchase_order_id": "PO_101",
            "sku_id": "SKU_001",
            "warehouse_id": "WH_01",
            "quantity": 30,
            "status": "IN_TRANSIT",
            "order_date": "2024-01-01",
            "expected_delivery_date": "2024-01-10",
            "actual_delivery_date": None,
        },
        {
            "purchase_order_id": "PO_102",
            "sku_id": "SKU_001",
            "warehouse_id": "WH_01",
            "quantity": 50,
            "status": "DELIVERED",
            "order_date": "2023-12-15",
            "expected_delivery_date": "2023-12-25",
            "actual_delivery_date": "2023-12-24",
        },
        {
            "purchase_order_id": "PO_103",
            "sku_id": "SKU_001",
            "warehouse_id": "WH_01",
            "quantity": 25,
            "status": "CANCELLED",
            "order_date": "2023-12-20",
            "expected_delivery_date": "2024-01-05",
            "actual_delivery_date": None,
        },
    ])


class TestInventoryPosition:
    def test_basic_inventory_position_with_open_pos(self, sample_snapshots, sample_purchases):
        res = calculate_inventory_positions(
            snapshots=sample_snapshots,
            purchases=sample_purchases,
            as_of_date="2024-01-02",
        )

        assert len(res) == 2
        sku1 = res[res["sku_id"] == "SKU_001"].iloc[0]

        # Latest snapshot on 2024-01-02: available=45, reserved=5
        assert sku1["on_hand"] == 45
        assert sku1["reserved"] == 5
        # Open PO: only PO_101 (qty=30). PO_102 is DELIVERED, PO_103 is CANCELLED
        assert sku1["on_order"] == 30
        assert sku1["damaged"] == 2
        # Net position = on_hand (45) + on_order (30) - reserved (5) = 70
        assert sku1["net_position"] == 70
        # Total physical = on_hand (45) + reserved (5) + damaged (2) = 52
        assert sku1["total_physical"] == 52

        sku2 = res[res["sku_id"] == "SKU_002"].iloc[0]
        assert sku2["on_hand"] == 100
        assert sku2["on_order"] == 0
        assert sku2["net_position"] == 100

    def test_fallback_to_snapshot_in_transit_when_no_purchases(self, sample_snapshots):
        res = calculate_inventory_positions(
            snapshots=sample_snapshots,
            purchases=None,
            as_of_date="2024-01-02",
        )
        sku1 = res[res["sku_id"] == "SKU_001"].iloc[0]
        # In snapshot 2024-01-02, in_transit_qty is 20
        assert sku1["on_order"] == 20
        assert sku1["net_position"] == 45 + 20 - 5

    def test_as_of_date_filtering(self, sample_snapshots, sample_purchases):
        # As of 2024-01-01: SKU_002 did not exist yet in snapshots
        res = calculate_inventory_positions(
            snapshots=sample_snapshots,
            purchases=sample_purchases,
            as_of_date="2024-01-01",
        )
        assert len(res) == 1
        assert res.iloc[0]["sku_id"] == "SKU_001"
        assert res.iloc[0]["on_hand"] == 50
        assert res.iloc[0]["reserved"] == 10

    def test_empty_and_missing_inputs(self):
        empty_res = calculate_inventory_positions(pd.DataFrame())
        assert empty_res.empty

        with pytest.raises(ValueError, match="must contain 'sku_id' and 'warehouse_id'"):
            calculate_inventory_positions(pd.DataFrame({"available_qty": [10]}))

        with pytest.raises(ValueError, match="must contain 'available_qty'"):
            calculate_inventory_positions(pd.DataFrame({"sku_id": ["S1"], "warehouse_id": ["W1"]}))

    def test_extract_position_records(self, sample_snapshots):
        res = calculate_inventory_positions(sample_snapshots)
        records = extract_position_records(res)
        assert len(records) == 2
        assert isinstance(records[0], InventoryPositionRecord)
        assert records[0].net_position == records[0].on_hand + records[0].on_order - records[0].reserved
