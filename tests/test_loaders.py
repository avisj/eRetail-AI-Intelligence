"""Tests for CSV Data Loader."""

from pathlib import Path
import pandas as pd
import pytest

from commerce_ai.data.loaders import (
    CommerceDataLoader,
    DataLoadError,
    SchemaValidationError,
    load_csv,
    load_dataset,
)


@pytest.fixture
def temp_csv_dir(tmp_path):
    """Creates a temporary directory with valid CSV files."""
    channels_csv = tmp_path / "channels.csv"
    channels_csv.write_text("channel_id,channel_name,channel_type\nCH_01,Amazon,Marketplace\n")

    products_csv = tmp_path / "products.csv"
    products_csv.write_text("sku_id,product_name,category_id,unit_cost,selling_price,currency\nSKU_01,Keyboard,Electronics,15.0,30.0,USD\n")

    warehouses_csv = tmp_path / "warehouses.csv"
    warehouses_csv.write_text("warehouse_id,warehouse_name,city,state,country\nWH_01,Main Depot,Dallas,TX,USA\n")

    suppliers_csv = tmp_path / "suppliers.csv"
    suppliers_csv.write_text("supplier_id,supplier_name,average_lead_time_days,minimum_order_quantity\nSUPP_01,Acme,10,50\n")

    sales_csv = tmp_path / "sales.csv"
    sales_csv.write_text("sale_id,order_id,date,sku_id,warehouse_id,channel_id,quantity,unit_price,discount,revenue,currency\nSALE_01,ORD_01,2025-01-10,SKU_01,WH_01,CH_01,2,30.0,0.0,60.0,USD\n")

    return tmp_path


class TestCommerceDataLoader:
    def test_load_existing_csv(self, temp_csv_dir):
        file_path = temp_csv_dir / "products.csv"
        df = load_csv(file_path, entity_name="products", validate=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["sku_id"].iloc[0] == "SKU_01"

    def test_load_non_existent_file_raises_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "non_existent.csv")

    def test_load_empty_file_raises_error(self, tmp_path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        with pytest.raises(DataLoadError):
            load_csv(empty_csv, entity_name="products", validate=False)

    def test_schema_validation_error_on_corrupt_data(self, tmp_path):
        bad_csv = tmp_path / "sales.csv"
        # Missing required revenue and quantity
        bad_csv.write_text("sale_id,order_id,date\nSALE_01,ORD_01,2025-01-01\n")
        with pytest.raises(SchemaValidationError):
            load_csv(bad_csv, entity_name="sales", validate=True)

    def test_load_all_from_directory(self, temp_csv_dir):
        loader = CommerceDataLoader()
        datasets = loader.load_all(temp_csv_dir, validate=True, validate_relational=False)
        assert "products" in datasets
        assert "channels" in datasets
        assert "sales" in datasets
        assert len(datasets["sales"]) == 1
