"""Unit and Integration Tests for Return Prediction Dataset & Label Engineering (Phase 5C-1).

Validates all 28 required specifications:
1. Correct prediction grain (order line: sale_id / (order_id, sku_id))
2. Positive return label (target_returned = 1)
3. Negative return label (target_returned = 0)
4. Partial return tracking (target_returned = 1, target_return_quantity = 3)
5. Return-window handling (within window = 1, outside window = 0)
6. Current-order return leakage protection (no return info in X)
7. Future return leakage protection
8. Historical feature point-in-time filtering (strictly < order_date)
9. SKU historical return rate calculation
10. Channel historical return rate calculation
11. Warehouse historical return rate calculation
12. SKU x Channel historical rate calculation
13. SKU x Warehouse historical rate calculation
14. Cold-start SKU handling and indicator
15. Cold-start channel handling and indicator
16. Cold-start combination handling and indicator
17. Missing product referential integrity audit
18. Missing warehouse referential integrity audit
19. Missing channel referential integrity audit
20. Invalid sales quantities audit
21. Impossible return quantity (return_qty > sold_qty) audit
22. Duplicate prediction rows audit
23. Temporal chronological split (no overlap, no future in train)
24. Explicit feature / target separation (X, y, metadata)
25. Forbidden leakage columns detection in verify_no_leakage
26. Deterministic dataset generation across repeated runs
27. Empty dataset handling
28. Target imbalance reporting
"""

from __future__ import annotations

import pandas as pd
import pytest

from commerce_ai.returns.schemas import (
    ReturnPredictionConfig,
    ReturnPredictionRecord,
)
from commerce_ai.returns.prediction_dataset import (
    FORBIDDEN_LEAKAGE_COLUMNS,
    ReturnPredictionDataset,
    ReturnPredictionDatasetBuilder,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def sample_products() -> pd.DataFrame:
    return pd.DataFrame([
        {"sku_id": "SKU_01", "product_name": "Pro Headphones", "category_id": "Electronics", "brand": "AudioTech", "unit_cost": 50.0, "selling_price": 100.0, "velocity_tier": "HIGH"},
        {"sku_id": "SKU_02", "product_name": "Cotton T-Shirt", "category_id": "Fashion", "brand": "StyleCo", "unit_cost": 10.0, "selling_price": 25.0, "velocity_tier": "MEDIUM"},
    ])


@pytest.fixture
def sample_warehouses() -> pd.DataFrame:
    return pd.DataFrame([
        {"warehouse_id": "WH_01", "warehouse_name": "North Hub", "city": "Chicago", "state": "IL", "country": "USA"},
        {"warehouse_id": "WH_02", "warehouse_name": "South Hub", "city": "Dallas", "state": "TX", "country": "USA"},
    ])


@pytest.fixture
def sample_channels() -> pd.DataFrame:
    return pd.DataFrame([
        {"channel_id": "CH_WEB", "channel_name": "Webstore", "channel_type": "Direct"},
        {"channel_id": "CH_APP", "channel_name": "Mobile App", "channel_type": "Direct"},
    ])


@pytest.fixture
def default_config() -> ReturnPredictionConfig:
    return ReturnPredictionConfig(
        return_window_days=30,
        prediction_cutoff_policy="include_all",
        temporal_train_ratio=0.70,
        temporal_validation_ratio=0.15,
        temporal_test_ratio=0.15,
    )


# =====================================================================
# Tests 1-5: Prediction Grain & Label Construction
# =====================================================================


def test_correct_prediction_grain(default_config, sample_products, sample_warehouses, sample_channels):
    """Each prediction row represents an order line with unique sale_id and (order_id, sku_id)."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-05", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 2, "unit_price": 100.0, "revenue": 200.0},
        {"sale_id": "S2", "order_id": "O1", "date": "2026-01-05", "sku_id": "SKU_02", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 1, "unit_price": 25.0, "revenue": 25.0},
    ])
    returns = pd.DataFrame()

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns, products=sample_products, warehouses=sample_warehouses, channels=sample_channels)

    assert len(dataset.X) == 2
    assert len(dataset.metadata) == 2
    assert list(dataset.metadata["sale_id"]) == ["S1", "S2"]
    assert list(dataset.metadata["sku_id"]) == ["SKU_01", "SKU_02"]
    assert dataset.metadata.iloc[0]["prediction_id"] == "PRED-S1"


def test_positive_return_label(default_config):
    """Sale line with return occurring within return window receives target_returned = 1."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 2, "unit_price": 50.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-10", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 2},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    assert len(dataset.y) == 1
    assert dataset.y.iloc[0] == 1
    assert dataset.metadata.iloc[0]["target_return_quantity"] == 2


def test_negative_return_label(default_config):
    """Sale line without return receives target_returned = 0."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 2, "unit_price": 50.0},
    ])
    returns = pd.DataFrame()

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    assert len(dataset.y) == 1
    assert dataset.y.iloc[0] == 0
    assert dataset.metadata.iloc[0]["target_return_quantity"] == 0


def test_partial_return(default_config):
    """Partial return (sold 10, returned 3) flags target_returned = 1 and records target_return_quantity = 3."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 10, "unit_price": 20.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-12", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 3},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    assert dataset.y.iloc[0] == 1
    assert dataset.metadata.iloc[0]["target_return_quantity"] == 3


def test_return_window_handling():
    """Return within 30-day window is labeled 1; return on day 45 is outside window and labeled 0."""
    cfg = ReturnPredictionConfig(return_window_days=30, prediction_cutoff_policy="include_all")
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 1, "unit_price": 100.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-01", "sku_id": "SKU_02", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 1, "unit_price": 50.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-20", "sku_id": "SKU_01", "quantity": 1},  # 19 days -> in window
        {"return_id": "R2", "order_id": "O2", "return_date": "2026-02-15", "sku_id": "SKU_02", "quantity": 1},  # 45 days -> outside window
    ])

    builder = ReturnPredictionDatasetBuilder(config=cfg)
    dataset = builder.build(sales=sales, returns=returns)

    res_df = dataset.metadata.copy()
    res_df["y"] = dataset.y.values

    assert res_df[res_df["sale_id"] == "S1"]["y"].iloc[0] == 1
    assert res_df[res_df["sale_id"] == "S2"]["y"].iloc[0] == 0


# =====================================================================
# Tests 6-8: Strict Anti-Leakage & Point-in-Time Protections
# =====================================================================


def test_current_order_return_leakage(default_config):
    """Features X strictly excludes return_date, return_reason, returned_quantity, refund_amount, etc."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 1, "unit_price": 100.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-05", "sku_id": "SKU_01", "quantity": 1, "reason": "Defective Item"},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    # Check forbidden leakage columns
    for forbidden in FORBIDDEN_LEAKAGE_COLUMNS:
        assert forbidden not in dataset.X.columns, f"Leakage detected! {forbidden} is inside feature matrix X"


def test_future_return_leakage(default_config):
    """Returns occurring on or after order date D do NOT leak into historical rate of an order on date D."""
    # Day 1: Order 1 sold, returned on Day 5
    # Day 3: Order 2 sold. On Day 3, Order 1 return (processed Day 5) was NOT yet known!
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 100, "unit_price": 10.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-03", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 10, "unit_price": 10.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-05", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 20},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    # For S2 on Jan 3: Return R1 happened on Jan 5, so prior returned units must be 0!
    s2_row = dataset.X.iloc[1]
    assert s2_row["hist_sku_returned_units"] == 0
    assert s2_row["hist_sku_return_rate"] == 0.0


def test_historical_feature_point_in_time_filtering(default_config):
    """Historical return rate on Day 6 strictly incorporates returns processed on or before Day 5."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 100, "unit_price": 10.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-06", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 10, "unit_price": 10.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-04", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 15},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    # For S2 on Jan 6: Return on Jan 4 is known! 15 / 100 = 0.15
    s2_row = dataset.X.iloc[1]
    assert s2_row["hist_sku_sold_units"] == 100
    assert s2_row["hist_sku_returned_units"] == 15
    assert s2_row["hist_sku_return_rate"] == 0.15


# =====================================================================
# Tests 9-13: Multi-Dimensional Historical Return Features
# =====================================================================


def test_sku_historical_return_rate(default_config):
    """SKU historical return rate accurately calculated from prior sales and returns."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 200, "unit_price": 10.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-10", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 50, "unit_price": 10.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-05", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 20},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    assert dataset.X.iloc[1]["hist_sku_return_rate"] == 0.10  # 20 / 200


def test_channel_historical_return_rate(default_config):
    """Channel historical return rate accurately calculated from prior channel sales and returns."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_APP", "quantity": 100, "unit_price": 10.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-10", "sku_id": "SKU_02", "warehouse_id": "WH_01", "channel_id": "CH_APP", "quantity": 20, "unit_price": 10.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-05", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_APP", "quantity": 12},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    assert dataset.X.iloc[1]["hist_channel_return_rate"] == 0.12  # 12 / 100


def test_warehouse_historical_return_rate(default_config):
    """Warehouse historical return rate accurately calculated from prior warehouse sales and returns."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_SOUTH", "channel_id": "CH_WEB", "quantity": 100, "unit_price": 10.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-10", "sku_id": "SKU_02", "warehouse_id": "WH_SOUTH", "channel_id": "CH_WEB", "quantity": 20, "unit_price": 10.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-05", "sku_id": "SKU_01", "warehouse_id": "WH_SOUTH", "channel_id": "CH_WEB", "quantity": 8},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    assert dataset.X.iloc[1]["hist_warehouse_return_rate"] == 0.08  # 8 / 100


def test_sku_channel_historical_rate(default_config):
    """SKU x Channel interaction historical return rate computed correctly."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_APP", "quantity": 100, "unit_price": 10.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-10", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_APP", "quantity": 20, "unit_price": 10.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-05", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_APP", "quantity": 14},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    assert dataset.X.iloc[1]["hist_sku_channel_return_rate"] == 0.14


def test_sku_warehouse_historical_rate(default_config):
    """SKU x Warehouse interaction historical return rate computed correctly."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_WEST", "channel_id": "CH_WEB", "quantity": 100, "unit_price": 10.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-10", "sku_id": "SKU_01", "warehouse_id": "WH_WEST", "channel_id": "CH_WEB", "quantity": 20, "unit_price": 10.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-05", "sku_id": "SKU_01", "warehouse_id": "WH_WEST", "channel_id": "CH_WEB", "quantity": 7},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    assert dataset.X.iloc[1]["hist_sku_warehouse_return_rate"] == 0.07


# =====================================================================
# Tests 14-16: Cold Start Handling & Fallback Hierarchy
# =====================================================================


def test_cold_start_sku(default_config):
    """New SKU with no prior history has is_cold_start_sku = 1 and falls back gracefully."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_NEW", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 5, "unit_price": 50.0},
    ])
    returns = pd.DataFrame()

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    assert dataset.X.iloc[0]["is_cold_start_sku"] == 1
    assert dataset.X.iloc[0]["hist_sku_sold_units"] == 0
    assert dataset.X.iloc[0]["hist_sku_return_rate"] == 0.0


def test_cold_start_channel(default_config):
    """New sales channel with no prior history has is_cold_start_channel = 1."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_NEW_TIKTOK", "quantity": 2, "unit_price": 30.0},
    ])
    returns = pd.DataFrame()

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    assert dataset.X.iloc[0]["is_cold_start_channel"] == 1


def test_cold_start_combination(default_config):
    """New (SKU, Channel) combination falls back to SKU rate if SKU has history."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 100, "unit_price": 10.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-10", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_NEW", "quantity": 5, "unit_price": 10.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-05", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 10},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    # S2 is on CH_NEW (0 prior combination sales) -> falls back to SKU_01 prior rate (10 / 100 = 0.10)
    s2_row = dataset.X.iloc[1]
    assert s2_row["is_cold_start_combination"] == 1
    assert s2_row["hist_sku_channel_return_rate"] == 0.10


# =====================================================================
# Tests 17-22: Data Quality & Referential Auditing
# =====================================================================


def test_missing_product(default_config, sample_products):
    """Sales row with unknown SKU is flagged in quality report."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_UNKNOWN", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 1, "unit_price": 10.0},
    ])
    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=pd.DataFrame(), products=sample_products)

    assert dataset.quality_report.unknown_sku_count == 1
    assert dataset.quality_report.is_clean is False


def test_missing_warehouse(default_config, sample_warehouses):
    """Sales row with unknown warehouse is flagged in quality report."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_UNKNOWN", "channel_id": "CH_WEB", "quantity": 1, "unit_price": 10.0},
    ])
    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=pd.DataFrame(), warehouses=sample_warehouses)

    assert dataset.quality_report.unknown_warehouse_count == 1
    assert dataset.quality_report.is_clean is False


def test_missing_channel(default_config, sample_channels):
    """Sales row with unknown channel is flagged in quality report."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_UNKNOWN", "quantity": 1, "unit_price": 10.0},
    ])
    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=pd.DataFrame(), channels=sample_channels)

    assert dataset.quality_report.unknown_channel_count == 1
    assert dataset.quality_report.is_clean is False


def test_invalid_quantities(default_config):
    """Sales with zero or negative quantity are rejected and audited."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": -5, "unit_price": 10.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 2, "unit_price": 10.0},
    ])
    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=pd.DataFrame())

    assert dataset.quality_report.invalid_quantity_count == 1
    assert len(dataset.X) == 1
    assert dataset.metadata.iloc[0]["sale_id"] == "S2"


def test_return_quantity_greater_than_sold_quantity(default_config):
    """Returns with quantity > sold quantity are audited as impossible."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 2, "unit_price": 10.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-05", "sku_id": "SKU_01", "quantity": 10},  # 10 > 2
    ])
    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    assert dataset.quality_report.impossible_return_quantity_count == 1
    assert dataset.metadata.iloc[0]["target_return_quantity"] == 2  # Capped at sold quantity


def test_duplicate_prediction_rows(default_config):
    """Duplicate sale_ids in input sales are reported."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 2, "unit_price": 10.0},
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 2, "unit_price": 10.0},
    ])
    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=pd.DataFrame())

    assert dataset.quality_report.duplicate_prediction_rows == 1


# =====================================================================
# Tests 23-28: Splits, Isolation, Determinism, Imbalance
# =====================================================================


def test_temporal_split():
    """Dataset is split chronologically into train, val, and test without index or temporal overlap."""
    cfg = ReturnPredictionConfig(
        return_window_days=5,
        prediction_cutoff_policy="include_all",
        temporal_train_ratio=0.50,
        temporal_validation_ratio=0.25,
        temporal_test_ratio=0.25,
    )
    sales = pd.DataFrame([
        {"sale_id": f"S_{i}", "order_id": f"O_{i}", "date": f"2026-01-{i+1:02d}", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 2, "unit_price": 10.0}
        for i in range(20)
    ])
    builder = ReturnPredictionDatasetBuilder(config=cfg)
    dataset = builder.build(sales=sales, returns=pd.DataFrame())

    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_val_data()
    X_test, y_test = dataset.get_test_data()

    assert len(X_train) == 10
    assert len(X_val) == 5
    assert len(X_test) == 5
    assert len(X_train) + len(X_val) + len(X_test) == 20

    # Verify disjointness
    train_s = set(dataset.train_indices)
    val_s = set(dataset.val_indices)
    test_s = set(dataset.test_indices)
    assert not (train_s & val_s)
    assert not (train_s & test_s)
    assert not (val_s & test_s)


def test_feature_target_separation(default_config):
    """Features X, target y, and metadata are strictly separated."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 2, "unit_price": 10.0},
    ])
    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=pd.DataFrame())

    assert "target_returned" not in dataset.X.columns
    assert "sale_id" not in dataset.X.columns
    assert "order_id" not in dataset.X.columns
    assert dataset.y.name == "target_returned"
    assert "sale_id" in dataset.metadata.columns


def test_forbidden_leakage_columns(default_config):
    """Manual injection of forbidden leakage column into X triggers ValueError."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 2, "unit_price": 10.0},
    ])
    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=pd.DataFrame())

    # Tamper with X to inject a forbidden column
    dataset.X["return_reason"] = "Defective Item"
    with pytest.raises(ValueError, match="Target leakage detected"):
        dataset.verify_no_leakage()


def test_deterministic_dataset_generation(default_config, sample_products):
    """Two identical build calls produce 100% invariant features and targets."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 5, "unit_price": 100.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-02", "sku_id": "SKU_02", "warehouse_id": "WH_02", "channel_id": "CH_APP", "quantity": 2, "unit_price": 25.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-05", "sku_id": "SKU_01", "quantity": 2},
    ])

    b1 = ReturnPredictionDatasetBuilder(config=default_config)
    d1 = b1.build(sales=sales, returns=returns, products=sample_products)

    b2 = ReturnPredictionDatasetBuilder(config=default_config)
    d2 = b2.build(sales=sales, returns=returns, products=sample_products)

    pd.testing.assert_frame_equal(d1.X, d2.X)
    pd.testing.assert_series_equal(d1.y, d2.y)
    pd.testing.assert_frame_equal(d1.metadata, d2.metadata)


def test_empty_dataset(default_config):
    """Empty sales DataFrame is handled gracefully and returns empty dataset."""
    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=pd.DataFrame(), returns=pd.DataFrame())

    assert dataset.X.empty
    assert dataset.y.empty
    assert dataset.metadata.empty
    assert dataset.quality_report.total_input_sales == 0


def test_target_imbalance_reporting(default_config):
    """Target imbalance correctly counts positive and negative cases."""
    sales = pd.DataFrame([
        {"sale_id": "S1", "order_id": "O1", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 1, "unit_price": 10.0},
        {"sale_id": "S2", "order_id": "O2", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 1, "unit_price": 10.0},
        {"sale_id": "S3", "order_id": "O3", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 1, "unit_price": 10.0},
        {"sale_id": "S4", "order_id": "O4", "date": "2026-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_WEB", "quantity": 1, "unit_price": 10.0},
    ])
    returns = pd.DataFrame([
        {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-05", "sku_id": "SKU_01", "quantity": 1},
    ])

    builder = ReturnPredictionDatasetBuilder(config=default_config)
    dataset = builder.build(sales=sales, returns=returns)

    stats = dataset.get_imbalance_stats()
    assert stats["positive_count"] == 1
    assert stats["negative_count"] == 3
    assert stats["total_count"] == 4
    assert stats["positive_rate"] == 0.25
