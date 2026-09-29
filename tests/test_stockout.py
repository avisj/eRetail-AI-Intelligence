"""Tests for Stockout Detection and Demand Masking."""

import pandas as pd
import pytest

from commerce_ai.analytics.stockout import (
    StockoutConfig,
    detect_stockout_periods,
    mask_stockout_demand,
)


@pytest.fixture
def sample_stockout_series():
    """Generates a 6-day trajectory with deliberate stockouts."""
    dates = pd.date_range("2025-01-01", periods=6, freq="D")
    return pd.DataFrame({
        "date": dates,
        "sku_id": ["SKU_A"] * 6,
        "warehouse_id": ["WH_1"] * 6,
        "units_sold": [10, 8, 2, 0, 0, 12],
        "available_qty": [100, 50, 2, 0, 0, 80],
    })


class TestStockoutDetection:
    def test_stockout_detection_thresholds(self, sample_stockout_series):
        cfg = StockoutConfig(stockout_threshold=0, low_stock_threshold=5)
        detected = detect_stockout_periods(sample_stockout_series, config=cfg)

        # Day 1 & 2: Normal stock
        assert not detected["is_stockout"].iloc[0]
        assert not detected["is_low_stock"].iloc[0]

        # Day 3: available_qty = 2 -> low stock but not stockout
        assert not detected["is_stockout"].iloc[2]
        assert detected["is_low_stock"].iloc[2]

        # Day 4 & 5: available_qty = 0 -> stockout
        assert detected["is_stockout"].iloc[3]
        assert detected["is_stockout"].iloc[4]

        # Stockout days running count
        assert detected["stockout_days"].iloc[3] == 1
        assert detected["stockout_days"].iloc[4] == 2
        assert detected["stockout_days"].iloc[5] == 0  # Restocked

        # Event ID should be consistent across Day 4 and 5
        event_id_day4 = detected["stockout_event_id"].iloc[3]
        event_id_day5 = detected["stockout_event_id"].iloc[4]
        assert event_id_day4 != ""
        assert event_id_day4 == event_id_day5

    def test_demand_masking_flags(self, sample_stockout_series):
        cfg = StockoutConfig(stockout_threshold=0, low_stock_threshold=5, mask_low_stock_exhausted=True)
        masked = mask_stockout_demand(sample_stockout_series, config=cfg)

        # Day 1, 2, 6: clean observations eligible for training
        assert bool(masked["forecast_training_eligible"].iloc[0]) is True
        assert bool(masked["forecast_training_eligible"].iloc[1]) is True
        assert bool(masked["forecast_training_eligible"].iloc[5]) is True

        # Day 3: Low stock (2) was completely exhausted by sales (2) -> demand constrained
        assert bool(masked["is_demand_constrained"].iloc[2]) is True
        assert bool(masked["forecast_training_eligible"].iloc[2]) is False

        # Day 4 & 5: Stockout -> constrained, ineligible
        assert bool(masked["is_demand_constrained"].iloc[3]) is True
        assert bool(masked["forecast_training_eligible"].iloc[3]) is False
        assert bool(masked["is_demand_constrained"].iloc[4]) is True
        assert bool(masked["forecast_training_eligible"].iloc[4]) is False
