"""Tests for Pydantic Data Contract Schemas."""

from datetime import date
import pytest
from pydantic import ValidationError

from commerce_ai.data.schemas import (
    SaleRecord,
    InventorySnapshot,
    Product,
    Warehouse,
    PurchaseOrder,
    ReturnRecord,
    Channel,
    Supplier,
)


class TestSaleRecordSchema:
    def test_valid_sale_record(self):
        sale = SaleRecord(
            sale_id="SALE_001",
            order_id="ORD_1001",
            date=date(2025, 1, 15),
            sku_id="SKU_001",
            warehouse_id="WH_01",
            channel_id="CH_AMZ",
            quantity=2,
            unit_price=29.99,
            discount=5.0,
            revenue=54.98,
            currency="USD",
        )
        assert sale.sale_id == "SALE_001"
        assert sale.quantity == 2
        assert sale.revenue == 54.98
        assert sale.currency == "USD"

    def test_extra_fields_preserved(self):
        sale = SaleRecord(
            sale_id="SALE_002",
            order_id="ORD_1002",
            date=date(2025, 1, 15),
            sku_id="SKU_001",
            warehouse_id="WH_01",
            channel_id="CH_AMZ",
            quantity=1,
            unit_price=10.0,
            revenue=10.0,
            custom_tracking_code="TRACK_XYZ",
        )
        assert getattr(sale, "custom_tracking_code") == "TRACK_XYZ"

    def test_invalid_negative_quantity(self):
        with pytest.raises(ValidationError):
            SaleRecord(
                sale_id="SALE_003",
                order_id="ORD_1003",
                date=date(2025, 1, 15),
                sku_id="SKU_001",
                warehouse_id="WH_01",
                channel_id="CH_AMZ",
                quantity=-5,
                unit_price=10.0,
                revenue=10.0,
            )

    def test_invalid_negative_price(self):
        with pytest.raises(ValidationError):
            SaleRecord(
                sale_id="SALE_004",
                order_id="ORD_1004",
                date=date(2025, 1, 15),
                sku_id="SKU_001",
                warehouse_id="WH_01",
                channel_id="CH_AMZ",
                quantity=1,
                unit_price=-10.0,
                revenue=10.0,
            )

    def test_empty_id_rejected(self):
        with pytest.raises(ValidationError):
            SaleRecord(
                sale_id="   ",
                order_id="ORD_1005",
                date=date(2025, 1, 15),
                sku_id="SKU_001",
                warehouse_id="WH_01",
                channel_id="CH_AMZ",
                quantity=1,
                unit_price=10.0,
                revenue=10.0,
            )


class TestInventorySnapshotSchema:
    def test_valid_inventory_snapshot(self):
        inv = InventorySnapshot(
            snapshot_date=date(2025, 2, 1),
            sku_id="SKU_001",
            warehouse_id="WH_01",
            available_qty=150,
            reserved_qty=10,
            in_transit_qty=50,
            damaged_qty=2,
        )
        assert inv.available_qty == 150
        assert inv.total_physical_qty == 162

    def test_negative_available_qty_rejected(self):
        with pytest.raises(ValidationError):
            InventorySnapshot(
                snapshot_date=date(2025, 2, 1),
                sku_id="SKU_001",
                warehouse_id="WH_01",
                available_qty=-10,
            )


class TestProductSchema:
    def test_valid_product(self):
        prod = Product(
            sku_id="SKU_ELEC_101",
            product_name="Pro Wireless Mouse",
            category_id="Electronics",
            brand="LogiTech",
            unit_cost=15.50,
            selling_price=39.99,
        )
        assert prod.sku_id == "SKU_ELEC_101"
        assert prod.status == "ACTIVE"

    def test_negative_cost_rejected(self):
        with pytest.raises(ValidationError):
            Product(
                sku_id="SKU_002",
                product_name="Test Product",
                category_id="Home",
                unit_cost=-5.0,
                selling_price=15.0,
            )


class TestWarehouseSchema:
    def test_valid_warehouse(self):
        wh = Warehouse(
            warehouse_id="WH_MIDWEST",
            warehouse_name="Midwest Mega Hub",
            city="Chicago",
            state="IL",
            country="USA",
            capacity_units=750000,
        )
        assert wh.warehouse_id == "WH_MIDWEST"
        assert wh.capacity_units == 750000

    def test_optional_capacity(self):
        wh = Warehouse(
            warehouse_id="WH_3PL",
            warehouse_name="Partner 3PL Depot",
            city="Dallas",
            state="TX",
            country="USA",
        )
        assert wh.capacity_units is None


class TestPurchaseOrderSchema:
    def test_valid_purchase_order(self):
        po = PurchaseOrder(
            purchase_order_id="PO_9901",
            order_date=date(2025, 3, 1),
            sku_id="SKU_001",
            supplier_id="SUPP_01",
            warehouse_id="WH_01",
            quantity=500,
            unit_cost=12.50,
            expected_delivery_date=date(2025, 3, 15),
            actual_delivery_date=date(2025, 3, 14),
            status="DELIVERED",
        )
        assert po.status == "DELIVERED"

    def test_invalid_expected_delivery_before_order_date(self):
        with pytest.raises(ValidationError):
            PurchaseOrder(
                purchase_order_id="PO_9902",
                order_date=date(2025, 3, 10),
                sku_id="SKU_001",
                supplier_id="SUPP_01",
                warehouse_id="WH_01",
                quantity=100,
                unit_cost=10.0,
                expected_delivery_date=date(2025, 3, 5),  # Earlier than order_date
            )


class TestReturnRecordSchema:
    def test_valid_return(self):
        ret = ReturnRecord(
            return_id="RET_001",
            order_id="ORD_1001",
            return_date=date(2025, 1, 20),
            sku_id="SKU_001",
            warehouse_id="WH_01",
            quantity=1,
            reason="Wrong Size",
            channel_id="CH_AMZ",
        )
        assert ret.return_id == "RET_001"
        assert ret.quantity == 1


class TestChannelSchema:
    def test_valid_channel(self):
        ch = Channel(
            channel_id="CH_AMZ",
            channel_name="Amazon US",
            channel_type="Marketplace",
        )
        assert ch.channel_id == "CH_AMZ"


class TestSupplierSchema:
    def test_valid_supplier(self):
        supp = Supplier(
            supplier_id="SUPP_01",
            supplier_name="Global Tech Imports",
            average_lead_time_days=14,
            minimum_order_quantity=50,
        )
        assert supp.average_lead_time_days == 14

    def test_invalid_lead_time_zero(self):
        with pytest.raises(ValidationError):
            Supplier(
                supplier_id="SUPP_02",
                supplier_name="Invalid Supplier",
                average_lead_time_days=0,
                minimum_order_quantity=10,
            )
