"""Comprehensive Unit and Integration Tests for Returns Intelligence Foundation (Phase 5A).

Validates all 25 required edge cases, data quality audits, anti-leakage guards,
dimensional analytics, sample size thresholds, and synthetic end-to-end integration.
"""

from __future__ import annotations

from datetime import date
import pytest
import pandas as pd

from commerce_ai.data.generators import GeneratorConfig, SyntheticDataGenerator
from commerce_ai.returns.schemas import (
    InvestigationFlag,
    ReturnsConfig,
    ReturnsDataQualityReport,
    ReturnsIntelligenceResult,
)
from commerce_ai.returns.analytics import (
    assess_returns_data_quality,
    enrich_returns_with_monetary_value,
    filter_by_as_of_date,
)
from commerce_ai.returns.service import ReturnsIntelligenceService


class TestReturnsIntelligenceFoundation:
    """Test suite covering Phase 5A Returns Intelligence."""

    # 1. Edge Case: Normal returns
    def test_normal_returns(self):
        """Standard sales and returns calculate exact unit return rate."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "unit_price": 20.0, "revenue": 2000.0}
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 10, "reason": "Defective Item"}
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        assert result.summary.total_sold_units == 100
        assert result.summary.total_returned_units == 10
        assert result.summary.overall_unit_return_rate == 0.10
        assert len(result.sku_metrics) == 1
        assert result.sku_metrics[0].return_rate == 0.10
        assert result.sku_metrics[0].returned_units == 10
        assert result.sku_metrics[0].sold_units == 100

    # 2. Edge Case: Zero returns
    def test_zero_returns(self):
        """Sales exist with zero returns; rate is 0.0 and NO_RETURN_DATA flag raised."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 50, "unit_price": 10.0, "revenue": 500.0}
        ])
        returns = pd.DataFrame(columns=["return_id", "order_id", "return_date", "sku_id", "warehouse_id", "channel_id", "quantity", "reason"])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        assert result.summary.total_returned_units == 0
        assert result.summary.overall_unit_return_rate == 0.0
        assert len(result.sku_metrics) == 0  # Empty returns dataset returns 0 SKU records

    # 3. Edge Case: Zero sales (return from previous untracked period)
    def test_zero_sales(self):
        """Returns occur for a SKU with 0 sales in current window; rate is None and ZERO_SALES_RECORDED flag set."""
        sales = pd.DataFrame(columns=["sale_id", "order_id", "date", "sku_id", "warehouse_id", "channel_id", "quantity", "unit_price", "revenue"])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 5, "reason": "Defective Item"}
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        assert result.summary.total_sold_units == 0
        assert result.summary.total_returned_units == 5
        assert result.summary.overall_unit_return_rate is None
        assert len(result.sku_metrics) == 1
        sku_m = result.sku_metrics[0]
        assert sku_m.return_rate is None
        assert InvestigationFlag.ZERO_SALES_RECORDED.value in sku_m.investigation_flags

    # 4. Edge Case: Missing SKU in data quality
    def test_missing_sku_in_data_quality(self):
        """Audit detects records with missing or empty SKU."""
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": None, "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 5, "reason": "Defective Item"},
            {"return_id": "R2", "order_id": "O2", "return_date": "2026-03-06", "sku_id": "  ", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 2, "reason": "Defective Item"},
        ])
        report = assess_returns_data_quality(returns)
        assert report.missing_sku_count == 2
        assert not report.is_clean

    # 5. Edge Case: Missing date in data quality
    def test_missing_date_in_data_quality(self):
        """Audit detects records with null return date."""
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": None, "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 5, "reason": "Defective Item"}
        ])
        report = assess_returns_data_quality(returns)
        assert report.missing_date_count == 1
        assert not report.is_clean

    # 6. Edge Case: Invalid negative or zero quantity
    def test_invalid_negative_quantity(self):
        """Audit detects non-positive return quantities."""
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": -5, "reason": "Defective Item"},
            {"return_id": "R2", "order_id": "O2", "return_date": "2026-03-02", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 0, "reason": "Defective Item"},
        ])
        report = assess_returns_data_quality(returns)
        assert report.invalid_quantity_count == 2
        assert not report.is_clean

    # 7. Edge Case: Multiple SKUs
    def test_multiple_skus(self):
        """SKU metrics are isolated and ranked by returned units."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 200, "revenue": 2000.0},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-03-01", "sku_id": "SKU-2", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0},
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 20, "reason": "Wrong Size"},
            {"return_id": "R2", "order_id": "O2", "return_date": "2026-03-05", "sku_id": "SKU-2", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 25, "reason": "Defective Item"},
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        assert len(result.sku_metrics) == 2
        # SKU-2 has 25 returned units, should be sorted first
        assert result.sku_metrics[0].sku_id == "SKU-2"
        assert result.sku_metrics[0].return_rate == 0.25
        assert result.sku_metrics[1].sku_id == "SKU-1"
        assert result.sku_metrics[1].return_rate == 0.10

    # 8. Edge Case: Multiple warehouses
    def test_multiple_warehouses(self):
        """Warehouse dimension metrics report factual rates without causal inference."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 300, "revenue": 3000.0},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-2", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0},
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 30, "reason": "Defective Item"},
            {"return_id": "R2", "order_id": "O2", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-2", "channel_id": "CH-1", "quantity": 20, "reason": "Defective Item"},
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        wh_map = {m.warehouse_id: m for m in result.warehouse_metrics}
        assert wh_map["WH-1"].return_rate == 0.10
        assert wh_map["WH-2"].return_rate == 0.20

    # 9. Edge Case: Multiple channels
    def test_multiple_channels(self):
        """Channel dimension comparison is factual and neutral."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-ONLINE", "quantity": 500, "revenue": 5000.0},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-RETAIL", "quantity": 200, "revenue": 2000.0},
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-ONLINE", "quantity": 25, "reason": "Customer Regret"},
            {"return_id": "R2", "order_id": "O2", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-RETAIL", "quantity": 40, "reason": "Defective Item"},
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        ch_map = {m.channel_id: m for m in result.channel_metrics}
        assert ch_map["CH-ONLINE"].return_rate == 0.05
        assert ch_map["CH-RETAIL"].return_rate == 0.20

    # 10. Edge Case: Multiple return reasons
    def test_multiple_return_reasons(self):
        """Return reasons breakdown calculates correct shares summing to 100%."""
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 50, "reason": "Defective Item"},
            {"return_id": "R2", "order_id": "O2", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 30, "reason": "Wrong Size/Color"},
            {"return_id": "R3", "order_id": "O3", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 20, "reason": "Customer Regret"},
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns)

        assert len(result.reason_metrics) == 3
        # Top reason is Defective Item with 50 units (50%)
        assert result.reason_metrics[0].reason == "Defective Item"
        assert result.reason_metrics[0].returned_units == 50
        assert result.reason_metrics[0].percentage_of_units == 50.0

        total_unit_pct = sum(r.percentage_of_units for r in result.reason_metrics)
        assert round(total_unit_pct, 1) == 100.0

    # 11. Edge Case: Explicit Return-Rate Calculations (Unit, Order, Revenue)
    def test_return_rate_calculation(self):
        """Accurately calculates unit_return_rate, order_return_rate, and revenue_return_rate."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "unit_price": 50.0, "revenue": 5000.0},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "unit_price": 50.0, "revenue": 5000.0},
        ])
        # O1 returned 20 units ($1000), O2 has no returns
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 20, "reason": "Defective Item"}
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        sku_m = result.sku_metrics[0]
        # Unit rate = 20 / 200 = 0.10
        assert sku_m.return_rate == 0.10
        # Order rate = 1 returned order / 2 total orders = 0.50
        assert sku_m.order_return_rate == 0.50
        # Revenue rate = 1000 / 10000 = 0.10
        assert sku_m.revenue_return_rate == 0.10

    # 12. Edge Case: Insufficient sample size guard
    def test_insufficient_sample(self):
        """SKUs below min_sold_units_threshold are marked INSUFFICIENT_SAMPLE and not flagged HIGH_RETURN_RATE."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-TINY", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 10, "revenue": 200.0}
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-TINY", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 3, "reason": "Defective Item"}
        ])

        # 3 out of 10 is 30% return rate, but sample is only 10 (default threshold is 30)
        service = ReturnsIntelligenceService(config=ReturnsConfig(min_sold_units_threshold=30, high_return_rate_threshold=0.15))
        result = service.analyze(returns=returns, sales=sales)

        sku_m = result.sku_metrics[0]
        assert not sku_m.is_sufficient_sample
        assert InvestigationFlag.INSUFFICIENT_SAMPLE.value in sku_m.investigation_flags
        # Must NOT trigger HIGH_RETURN_RATE due to sample guard
        assert InvestigationFlag.HIGH_RETURN_RATE.value not in sku_m.investigation_flags

    # 13. Edge Case: Missing monetary value
    def test_missing_monetary_value(self):
        """When sales revenue and product prices are absent, return_value is None without crashing."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100}
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 10, "reason": "Defective Item"}
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        sku_m = result.sku_metrics[0]
        assert sku_m.return_value is None
        assert sku_m.revenue_return_rate is None

    # 14. Edge Case: Time aggregation
    def test_time_aggregation(self):
        """Aggregates returns and sales into monthly buckets."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-01-15", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-02-15", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 200, "revenue": 2000.0},
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-01-20", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 10, "reason": "Defective Item"},
            {"return_id": "R2", "order_id": "O2", "return_date": "2026-02-20", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 30, "reason": "Defective Item"},
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales, time_series_freq="M")

        assert len(result.time_series) == 2
        assert result.time_series[0].period == "2026-01"
        assert result.time_series[0].return_rate == 0.10
        assert result.time_series[1].period == "2026-02"
        assert result.time_series[1].return_rate == 0.15

    # 15. Edge Case: Previous-period comparison & trend stability
    def test_previous_period_comparison(self):
        """Detects return rate increase vs prior period and tags RETURN_RATE_INCREASING."""
        sales = pd.DataFrame([
            # Period 1 (prior 30 days: Jan 31 - Mar 01)
            {"sale_id": "S1", "order_id": "O1", "date": "2026-02-10", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0},
            # Period 2 (current 30 days: Mar 01 - Mar 31)
            {"sale_id": "S2", "order_id": "O2", "date": "2026-03-10", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0},
        ])
        returns = pd.DataFrame([
            # Period 1: 10 returns (rate = 0.10)
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-02-15", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 10, "reason": "Defective Item"},
            # Period 2: 25 returns (rate = 0.25, delta = +0.15)
            {"return_id": "R2", "order_id": "O2", "return_date": "2026-03-15", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 25, "reason": "Defective Item"},
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(
            returns=returns,
            sales=sales,
            as_of_date="2026-03-31",
            previous_period_days=30,
        )

        sku_m = result.sku_metrics[0]
        assert sku_m.return_rate == 0.25
        assert sku_m.return_rate_change == 0.15
        assert sku_m.trend == "INCREASING"
        assert InvestigationFlag.RETURN_RATE_INCREASING.value in sku_m.investigation_flags

    # 16. Edge Case: SKU × Channel cross-tabulation
    def test_sku_x_channel(self):
        """Cross-tabulation exposes channel-specific return spike for a SKU."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-DIRECT", "quantity": 200, "revenue": 2000.0},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-MARKETPLACE", "quantity": 200, "revenue": 2000.0},
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-DIRECT", "quantity": 10, "reason": "Defective Item"},
            {"return_id": "R2", "order_id": "O2", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-MARKETPLACE", "quantity": 50, "reason": "Defective Item"},
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        sc_map = {m.key: m for m in result.sku_channel_metrics}
        assert sc_map["SKU-1:CH-DIRECT"].return_rate == 0.05
        assert sc_map["SKU-1:CH-MARKETPLACE"].return_rate == 0.25

    # 17. Edge Case: SKU × Warehouse cross-tabulation
    def test_sku_x_warehouse(self):
        """Cross-tabulation exposes warehouse-specific return pattern."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-A", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-B", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0},
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-A", "channel_id": "CH-1", "quantity": 5, "reason": "Defective Item"},
            {"return_id": "R2", "order_id": "O2", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-B", "channel_id": "CH-1", "quantity": 25, "reason": "Defective Item"},
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        sw_map = {m.key: m for m in result.sku_warehouse_metrics}
        assert sw_map["SKU-1:WH-A"].return_rate == 0.05
        assert sw_map["SKU-1:WH-B"].return_rate == 0.25

    # 18. Edge Case: Anti-leakage as_of_date filtering
    def test_as_of_date_filtering(self):
        """Events occurring strictly after as_of_date are discarded to prevent lookahead leakage."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0},
            {"sale_id": "S2", "order_id": "O2", "date": "2026-03-25", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0},
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 10, "reason": "Defective Item"},
            {"return_id": "R2", "order_id": "O2", "return_date": "2026-03-28", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 20, "reason": "Defective Item"},
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales, as_of_date="2026-03-10")

        # S2 and R2 are past Mar 10 and must not contribute
        assert result.summary.total_sold_units == 100
        assert result.summary.total_returned_units == 10
        assert result.summary.overall_unit_return_rate == 0.10

    # 19. Edge Case: Deterministic output
    def test_deterministic_output(self):
        """Identical inputs produce identical metrics across successive evaluations."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0}
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 15, "reason": "Defective Item"}
        ])

        service = ReturnsIntelligenceService()
        res1 = service.analyze(returns=returns, sales=sales)
        res2 = service.analyze(returns=returns, sales=sales)

        assert res1.summary.to_dict() == res2.summary.to_dict()
        assert [m.to_dict() for m in res1.sku_metrics] == [m.to_dict() for m in res2.sku_metrics]

    # 20. Edge Case: Completely empty returns dataset
    def test_empty_returns_dataset(self):
        """Evaluates gracefully when returns dataframe is empty."""
        returns = pd.DataFrame()
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0}
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        assert result.summary.total_returned_units == 0
        assert result.summary.overall_unit_return_rate == 0.0
        assert len(result.sku_metrics) == 0

    # 21. Edge Case: Unknown SKU referential integrity
    def test_unknown_sku_in_data_quality(self):
        """Data quality detects returns containing unknown SKUs not in products master."""
        products = pd.DataFrame([{"sku_id": "SKU-VALID"}])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-GHOST", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 5, "reason": "Defective Item"}
        ])

        report = assess_returns_data_quality(returns_df=returns, products_df=products)
        assert report.unknown_sku_count == 1
        assert not report.is_clean

    # 22. Edge Case: Unknown warehouse referential integrity
    def test_unknown_warehouse_in_data_quality(self):
        """Data quality detects returns referencing unknown warehouses."""
        warehouses = pd.DataFrame([{"warehouse_id": "WH-VALID"}])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-GHOST", "channel_id": "CH-1", "quantity": 5, "reason": "Defective Item"}
        ])

        report = assess_returns_data_quality(returns_df=returns, warehouses_df=warehouses)
        assert report.unknown_warehouse_count == 1
        assert not report.is_clean

    # 23. Edge Case: Unknown channel referential integrity
    def test_unknown_channel_in_data_quality(self):
        """Data quality detects returns referencing unknown channels."""
        channels = pd.DataFrame([{"channel_id": "CH-VALID"}])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-GHOST", "quantity": 5, "reason": "Defective Item"}
        ])

        report = assess_returns_data_quality(returns_df=returns, channels_df=channels)
        assert report.unknown_channel_count == 1
        assert not report.is_clean

    # 24. Edge Case: Duplicate return_id detection
    def test_duplicate_detection(self):
        """Data quality detects duplicate primary keys in returns."""
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 5, "reason": "Defective Item"},
            {"return_id": "R1", "order_id": "O2", "return_date": "2026-03-06", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 3, "reason": "Defective Item"},
        ])

        report = assess_returns_data_quality(returns_df=returns)
        assert report.duplicate_record_count == 1
        assert not report.is_clean

    # 25. Edge Case: Configurable thresholds
    def test_configurable_thresholds(self):
        """Custom configuration correctly adjusts high return rate and volume triggers."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0}
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 10, "reason": "Defective Item"}
        ])

        # 10% rate is NOT high under default 15%, but IS high under custom 8%
        cfg_custom = ReturnsConfig(high_return_rate_threshold=0.08, min_sold_units_threshold=30)
        service = ReturnsIntelligenceService(config=cfg_custom)
        result = service.analyze(returns=returns, sales=sales)

        sku_m = result.sku_metrics[0]
        assert InvestigationFlag.HIGH_RETURN_RATE.value in sku_m.investigation_flags

    # 26. Helper: DataFrame export methods
    def test_dataframe_exports(self):
        """Verify export methods produce valid non-empty DataFrames."""
        sales = pd.DataFrame([
            {"sale_id": "S1", "order_id": "O1", "date": "2026-03-01", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 100, "revenue": 1000.0}
        ])
        returns = pd.DataFrame([
            {"return_id": "R1", "order_id": "O1", "return_date": "2026-03-05", "sku_id": "SKU-1", "warehouse_id": "WH-1", "channel_id": "CH-1", "quantity": 10, "reason": "Defective Item"}
        ])

        service = ReturnsIntelligenceService()
        result = service.analyze(returns=returns, sales=sales)

        df_sku = result.sku_to_dataframe()
        assert not df_sku.empty
        assert "return_rate" in df_sku.columns

        df_ch = result.channel_to_dataframe()
        assert not df_ch.empty
        assert "channel_id" in df_ch.columns

        df_wh = result.warehouse_to_dataframe()
        assert not df_wh.empty
        assert "warehouse_id" in df_wh.columns

        df_reasons = result.reasons_to_dataframe()
        assert not df_reasons.empty
        assert "percentage_of_units" in df_reasons.columns

        df_ts = result.time_series_to_dataframe()
        assert not df_ts.empty
        assert "period" in df_ts.columns

    # 27. End-to-end integration with SyntheticDataGenerator
    def test_end_to_end_synthetic_data_integration(self):
        """Runs ReturnsIntelligenceService against coupled datasets from SyntheticDataGenerator."""
        gen_cfg = GeneratorConfig(
            number_of_skus=10,
            number_of_warehouses=2,
            number_of_channels=2,
            number_of_suppliers=3,
            historical_days=60,
            random_seed=123,
        )
        generator = SyntheticDataGenerator(config=gen_cfg)
        datasets = generator.generate_all()

        service = ReturnsIntelligenceService()
        result = service.analyze(
            returns=datasets["returns"],
            sales=datasets["sales"],
            products=datasets["products"],
            warehouses=datasets["warehouses"],
            channels=datasets["channels"],
        )

        assert result.summary.total_sold_units > 0
        assert result.summary.total_returned_units > 0
        assert result.summary.overall_unit_return_rate is not None
        assert len(result.sku_metrics) > 0
        assert len(result.reason_metrics) > 0
        assert len(result.time_series) > 0
        assert result.data_quality.is_clean
