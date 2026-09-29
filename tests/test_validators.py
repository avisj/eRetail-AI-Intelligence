"""Tests for Data Validation and Quality Reporting."""

import pandas as pd
import pytest

from commerce_ai.data.validators import (
    CommerceDataValidator,
    generate_data_quality_report,
)


@pytest.fixture
def valid_dataset_bundle():
    """Provides a valid, referentially-consistent bundle of DataFrames."""
    channels_df = pd.DataFrame([
        {"channel_id": "CH_01", "channel_name": "Webstore", "channel_type": "Direct"}
    ])
    warehouses_df = pd.DataFrame([
        {"warehouse_id": "WH_01", "warehouse_name": "Central Hub", "city": "Dallas", "state": "TX", "country": "USA", "capacity_units": 100000}
    ])
    suppliers_df = pd.DataFrame([
        {"supplier_id": "SUPP_01", "supplier_name": "Apex Vendor", "average_lead_time_days": 10, "minimum_order_quantity": 50}
    ])
    products_df = pd.DataFrame([
        {"sku_id": "SKU_01", "product_name": "Wireless Keyboard", "category_id": "Electronics", "unit_cost": 20.0, "selling_price": 45.0, "currency": "USD"}
    ])
    sales_df = pd.DataFrame([
        {"sale_id": "SALE_01", "order_id": "ORD_01", "date": "2025-01-10", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_01", "quantity": 2, "unit_price": 45.0, "discount": 0.0, "revenue": 90.0, "currency": "USD"}
    ])
    inventory_df = pd.DataFrame([
        {"snapshot_date": "2025-01-10", "sku_id": "SKU_01", "warehouse_id": "WH_01", "available_qty": 50, "reserved_qty": 5, "in_transit_qty": 20, "damaged_qty": 0}
    ])
    purchases_df = pd.DataFrame([
        {"purchase_order_id": "PO_01", "order_date": "2025-01-01", "sku_id": "SKU_01", "supplier_id": "SUPP_01", "warehouse_id": "WH_01", "quantity": 100, "unit_cost": 20.0, "expected_delivery_date": "2025-01-12", "status": "DELIVERED"}
    ])
    returns_df = pd.DataFrame([
        {"return_id": "RET_01", "order_id": "ORD_01", "return_date": "2025-01-15", "sku_id": "SKU_01", "warehouse_id": "WH_01", "quantity": 1, "reason": "Defective", "channel_id": "CH_01"}
    ])

    return {
        "channels": channels_df,
        "warehouses": warehouses_df,
        "suppliers": suppliers_df,
        "products": products_df,
        "sales": sales_df,
        "inventory": inventory_df,
        "purchases": purchases_df,
        "returns": returns_df,
    }


class TestCommerceDataValidator:
    def test_valid_bundle_passes(self, valid_dataset_bundle):
        validator = CommerceDataValidator()
        res = validator.validate_all(valid_dataset_bundle)
        assert res.valid is True
        assert len(res.errors) == 0

    def test_missing_required_column(self, valid_dataset_bundle):
        validator = CommerceDataValidator()
        corrupt_sales = valid_dataset_bundle["sales"].drop(columns=["revenue"])
        res = validator.validate_entity("sales", corrupt_sales)
        assert res.valid is False
        assert any("Missing required columns" in e.issue for e in res.errors)

    def test_duplicate_primary_key_detected(self, valid_dataset_bundle):
        validator = CommerceDataValidator()
        # Duplicate product sku_id
        dup_products = pd.concat([valid_dataset_bundle["products"], valid_dataset_bundle["products"]], ignore_index=True)
        res = validator.validate_entity("products", dup_products)
        assert res.valid is False
        assert any("duplicate primary key" in e.issue.lower() for e in res.errors)

    def test_negative_quantity_rejected(self, valid_dataset_bundle):
        validator = CommerceDataValidator()
        corrupt_sales = valid_dataset_bundle["sales"].copy()
        corrupt_sales.loc[0, "quantity"] = -3
        res = validator.validate_entity("sales", corrupt_sales)
        assert res.valid is False
        assert any("non-positive quantity" in e.issue for e in res.errors)

    def test_referential_integrity_catches_orphan_sku(self, valid_dataset_bundle):
        validator = CommerceDataValidator()
        corrupt_sales = valid_dataset_bundle["sales"].copy()
        corrupt_sales.loc[0, "sku_id"] = "SKU_ORPHAN_999"
        corrupt_bundle = dict(valid_dataset_bundle)
        corrupt_bundle["sales"] = corrupt_sales

        res = validator.validate_relational_integrity(corrupt_bundle)
        assert res.valid is False
        assert any("orphan 'sku_id'" in e.issue for e in res.errors)

    def test_data_quality_report_metrics(self, valid_dataset_bundle):
        report = generate_data_quality_report(valid_dataset_bundle)
        assert report["status"] == "VALID"
        assert report["total_errors"] == 0
        assert report["dimensions"]["unique_skus"] == 1
        assert report["dimensions"]["unique_warehouses"] == 1
        assert report["dimensions"]["unique_channels"] == 1
        assert report["sales_summary"]["total_revenue"] == 90.0
        assert report["sales_summary"]["total_sales_quantity"] == 2
