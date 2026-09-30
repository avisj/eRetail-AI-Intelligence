"""Unit Tests for Inventory Risk, Depletion Trajectory, and Health Assessment."""

import numpy as np
import pytest

from commerce_ai.inventory.risk import (
    RiskThresholdConfig,
    simulate_depletion_curve,
    compute_stockout_hazard_score,
    assess_sku_inventory_risk,
)


class TestInventoryRisk:
    def test_simulate_depletion_curve(self):
        forecast = [5.0, 5.0, 5.0, 5.0, 5.0]
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]

        # Exact runout on day 3: 15 units / 5 per day = 3 days
        remaining, runout_dt, days = simulate_depletion_curve(15, forecast, dates)
        assert days == 3.0
        assert runout_dt == "2024-01-03"
        assert remaining[-1] == 0.0

        # Fractional runout: 12 units -> Day 1 (5), Day 2 (5), Day 3 needs 2 of 5 = 2.4 days
        _, runout_dt2, days2 = simulate_depletion_curve(12, forecast, dates)
        assert pytest.approx(days2, rel=1e-3) == 2.4
        assert runout_dt2 == "2024-01-03"

        # No runout in horizon: 50 units on hand
        rem_high, runout_dt_none, days_none = simulate_depletion_curve(50, forecast, dates)
        assert days_none is None
        assert runout_dt_none is None
        assert rem_high[-1] == 25.0

        # Depleted at start
        _, runout_zero, days_zero = simulate_depletion_curve(0, forecast, dates)
        assert days_zero == 0.0
        assert runout_zero == "2024-01-01"

    def test_stockout_hazard_score(self):
        # 1. Zero net position -> 100% hazard (1.0)
        assert compute_stockout_hazard_score(0, daily_demand_mean=5.0, daily_demand_std=1.0, lead_time_mean_days=7.0) == 1.0

        # 2. Net position == expected lead time demand (5 * 7 = 35) -> 50% hazard (0.50)
        p_mid = compute_stockout_hazard_score(35.0, daily_demand_mean=5.0, daily_demand_std=1.0, lead_time_mean_days=7.0)
        assert pytest.approx(p_mid, rel=1e-2) == 0.50

        # 3. High buffer >> lead time demand -> near 0% hazard
        p_safe = compute_stockout_hazard_score(150.0, daily_demand_mean=5.0, daily_demand_std=1.0, lead_time_mean_days=7.0)
        assert p_safe < 0.01

    def test_risk_classification_categories(self):
        daily_fc = [10.0] * 30
        cfg = RiskThresholdConfig(critical_dos_threshold=3.0, overstock_dos_threshold=60.0)

        # 1. CRITICAL_STOCKOUT: on_hand = 0
        r_crit1 = assess_sku_inventory_risk(
            sku_id="SKU_1",
            warehouse_id="WH_1",
            on_hand=0,
            net_position=0,
            daily_forecast=daily_fc,
            lead_time_mean_days=10.0,
            safety_stock=20.0,
            reorder_point=120.0,
            config=cfg,
        )
        assert r_crit1.risk_category == "CRITICAL_STOCKOUT"
        assert r_crit1.stockout_risk_score == 1.0

        # 2. CRITICAL_STOCKOUT: on_hand = 20 (DOS = 2.0 < Lead Time 10.0)
        r_crit2 = assess_sku_inventory_risk(
            sku_id="SKU_1",
            warehouse_id="WH_1",
            on_hand=20,
            net_position=20,
            daily_forecast=daily_fc,
            lead_time_mean_days=10.0,
            safety_stock=20.0,
            reorder_point=120.0,
            config=cfg,
        )
        assert r_crit2.risk_category == "CRITICAL_STOCKOUT"

        # 3. UNDERSTOCK: on_hand = 100, but net_position (100) < ROP (120)
        # DOS = 10.0 >= Lead Time 10.0, so not critical, but needs replenishment
        r_under = assess_sku_inventory_risk(
            sku_id="SKU_1",
            warehouse_id="WH_1",
            on_hand=100,
            net_position=100,
            daily_forecast=daily_fc,
            lead_time_mean_days=10.0,
            safety_stock=20.0,
            reorder_point=120.0,
            config=cfg,
        )
        assert r_under.risk_category == "UNDERSTOCK"

        # 4. HEALTHY: on_hand = 150, net_position = 150 >= ROP (120), DOS = 15 <= 60
        r_healthy = assess_sku_inventory_risk(
            sku_id="SKU_1",
            warehouse_id="WH_1",
            on_hand=150,
            net_position=150,
            daily_forecast=daily_fc,
            lead_time_mean_days=10.0,
            safety_stock=20.0,
            reorder_point=120.0,
            config=cfg,
        )
        assert r_healthy.risk_category == "HEALTHY"

        # 5. OVERSTOCK: on_hand = 800, DOS = 80 > 60
        r_over = assess_sku_inventory_risk(
            sku_id="SKU_1",
            warehouse_id="WH_1",
            on_hand=800,
            net_position=800,
            daily_forecast=daily_fc,
            lead_time_mean_days=10.0,
            safety_stock=20.0,
            reorder_point=120.0,
            target_stock_level=200.0,
            unit_cost=15.0,
            config=cfg,
        )
        assert r_over.risk_category == "OVERSTOCK"
        assert r_over.excess_units == 600
        assert r_over.excess_capital == 600 * 15.0

        # 6. DEAD_STOCK: on_hand = 50, but daily forecast is 0.0
        r_dead = assess_sku_inventory_risk(
            sku_id="SKU_1",
            warehouse_id="WH_1",
            on_hand=50,
            net_position=50,
            daily_forecast=[0.0] * 30,
            lead_time_mean_days=10.0,
            unit_cost=10.0,
            config=cfg,
        )
        assert r_dead.risk_category == "DEAD_STOCK"
        assert r_dead.excess_capital == 50 * 10.0

    def test_evidence_transparency(self):
        r = assess_sku_inventory_risk(
            sku_id="SKU_TEST",
            warehouse_id="WH_1",
            on_hand=50,
            net_position=50,
            daily_forecast=[5.0] * 14,
            lead_time_mean_days=7.0,
            safety_stock=10.0,
            reorder_point=45.0,
            unit_cost=25.0,
            selling_price=40.0,
        )
        # Check underlying metrics exist and match
        assert "daily_demand_forecast_mean" in r.underlying_metrics
        assert "lead_time_mean_days" in r.underlying_metrics
        assert "stockout_probability" in r.underlying_metrics
        assert "days_of_supply" in r.underlying_metrics
        assert r.underlying_metrics["daily_demand_forecast_mean"] == 5.0
        assert r.underlying_metrics["lead_time_mean_days"] == 7.0
        assert r.underlying_metrics["unit_cost"] == 25.0

    def test_dormant_zero_stock_healthy(self):
        """Fix 1: Dormant SKU with zero stock is HEALTHY, not CRITICAL_STOCKOUT."""
        cfg = RiskThresholdConfig(min_demand_threshold=0.01)

        # 1. Dormant demand (0.0) + zero stock (0) -> HEALTHY
        r_dormant_zero = assess_sku_inventory_risk(
            sku_id="SKU_DORMANT_0",
            warehouse_id="WH_1",
            on_hand=0,
            net_position=0,
            daily_forecast=[0.0] * 30,
            lead_time_mean_days=10.0,
            config=cfg,
        )
        assert r_dormant_zero.risk_category == "HEALTHY"
        assert r_dormant_zero.stockout_risk_score == 0.0

        # 2. Dormant demand (0.0) + negative stock (-5) -> HEALTHY
        r_dormant_neg = assess_sku_inventory_risk(
            sku_id="SKU_DORMANT_NEG",
            warehouse_id="WH_1",
            on_hand=-5,
            net_position=-5,
            daily_forecast=[0.0] * 30,
            lead_time_mean_days=10.0,
            config=cfg,
        )
        assert r_dormant_neg.risk_category == "HEALTHY"

        # 3. Dormant demand (0.0) + positive stock (10) -> DEAD_STOCK
        r_dormant_pos = assess_sku_inventory_risk(
            sku_id="SKU_DORMANT_POS",
            warehouse_id="WH_1",
            on_hand=10,
            net_position=10,
            daily_forecast=[0.0] * 30,
            lead_time_mean_days=10.0,
            unit_cost=20.0,
            config=cfg,
        )
        assert r_dormant_pos.risk_category == "DEAD_STOCK"
        assert r_dormant_pos.excess_capital == 200.0

        # 4. Non-dormant demand (5.0) + zero stock (0) -> CRITICAL_STOCKOUT
        r_active_zero = assess_sku_inventory_risk(
            sku_id="SKU_ACTIVE_0",
            warehouse_id="WH_1",
            on_hand=0,
            net_position=0,
            daily_forecast=[5.0] * 30,
            lead_time_mean_days=10.0,
            config=cfg,
        )
        assert r_active_zero.risk_category == "CRITICAL_STOCKOUT"
        assert r_active_zero.stockout_risk_score == 1.0

    def test_stockout_hazard_with_flat_forecast_and_demand_std(self):
        """Fix 2: Flat forecast with explicit daily_demand_std avoids collapsing hazard curve."""
        # Flat forecast of 10.0 units/day over 10 days -> Lead time demand mean = 100
        daily_fc = [10.0] * 10
        # If daily_demand_std = 3.0, lead time variance sigma_D * sqrt(10) ~ 9.487
        # Net position = 90 (below expected demand of 100)
        # Z = (90 - 100) / 9.487 = -1.054 -> Hazard = 1 - Phi(-1.054) ~ 0.854
        r = assess_sku_inventory_risk(
            sku_id="SKU_FLAT",
            warehouse_id="WH_1",
            on_hand=90,
            net_position=90,
            daily_forecast=daily_fc,
            lead_time_mean_days=10.0,
            lead_time_std_days=0.0,
            daily_demand_std=3.0,
        )
        assert 0.70 < r.stockout_risk_score < 0.95
        assert r.underlying_metrics["daily_demand_forecast_std"] == 3.0

