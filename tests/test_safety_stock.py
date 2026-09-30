"""Unit Tests for Deterministic Safety Stock, ROP, and Policy Configuration."""

import math
import pytest
from scipy import stats

from commerce_ai.inventory.safety_stock import (
    ServiceLevelPolicy,
    service_level_to_z_score,
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_optional_eoq,
    calculate_safety_stock_result,
)


class TestSafetyStock:
    def test_z_score_calculation(self):
        # 95% -> ~1.6449
        z_95 = service_level_to_z_score(0.95)
        assert pytest.approx(z_95, rel=1e-3) == 1.6449

        # 99% -> ~2.3263
        z_99 = service_level_to_z_score(0.99)
        assert pytest.approx(z_99, rel=1e-3) == 2.3263

        # Invalid bounds
        with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
            service_level_to_z_score(0.0)

        with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
            service_level_to_z_score(1.0)

        with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
            service_level_to_z_score(-0.5)

    def test_custom_z_mapping(self):
        # Custom discrete table
        custom_table = {0.95: 1.65, 0.99: 2.33}
        assert service_level_to_z_score(0.95, custom_mapping=custom_table) == 1.65

        # Custom callable
        assert service_level_to_z_score(0.90, custom_mapping=lambda sl: 1.28) == 1.28

    def test_safety_stock_analytical_formula(self):
        # D = 10, sigma_D = 2, L = 9, sigma_L = 0, SL = 0.95
        # SS = Z * sqrt(9 * 4) = 1.64485 * 6 = 9.869
        ss, z = calculate_safety_stock(
            daily_demand_mean=10.0,
            daily_demand_std=2.0,
            lead_time_mean_days=9.0,
            lead_time_std_days=0.0,
            service_level=0.95,
        )
        assert pytest.approx(ss, rel=1e-3) == 1.644853 * 6.0

        # With lead time uncertainty: sigma_L = 1
        # Variance term: 9 * 4 + 100 * 1 = 36 + 100 = 136
        # SS = Z * sqrt(136) = 1.64485 * 11.6619 = 19.182
        ss_dual, _ = calculate_safety_stock(
            daily_demand_mean=10.0,
            daily_demand_std=2.0,
            lead_time_mean_days=9.0,
            lead_time_std_days=1.0,
            service_level=0.95,
        )
        expected_ss = z * math.sqrt(136.0)
        assert pytest.approx(ss_dual, rel=1e-3) == expected_ss

    def test_reorder_point_formula(self):
        # ROP = (D * L) + SS = 10 * 7 + 15 = 85
        rop = calculate_reorder_point(daily_demand_mean=10.0, lead_time_mean_days=7.0, safety_stock=15.0)
        assert rop == 85.0

    def test_validations_and_edge_cases(self):
        # Zero demand
        ss_zero, _ = calculate_safety_stock(0.0, 0.0, 10.0, 1.0)
        assert ss_zero == 0.0
        assert calculate_reorder_point(0.0, 10.0, 0.0) == 0.0

        # Zero lead time
        ss_no_lt, _ = calculate_safety_stock(10.0, 2.0, 0.0, 0.0)
        assert ss_no_lt == 0.0

        # Zero variance
        ss_det, _ = calculate_safety_stock(10.0, 0.0, 5.0, 0.0)
        assert ss_det == 0.0
        assert calculate_reorder_point(10.0, 5.0, ss_det) == 50.0

        # Negative inputs
        with pytest.raises(ValueError, match="Daily demand mean cannot be negative"):
            calculate_safety_stock(-1.0, 2.0, 5.0)

        with pytest.raises(ValueError, match="Daily demand std cannot be negative"):
            calculate_safety_stock(10.0, -2.0, 5.0)

        with pytest.raises(ValueError, match="Lead time mean cannot be negative"):
            calculate_safety_stock(10.0, 2.0, -5.0)

    def test_service_level_policy_resolution(self):
        policy = ServiceLevelPolicy(
            default_service_level=0.95,
            sku_overrides={"SKU_SPECIAL": 0.999},
        )
        # 1. SKU override
        assert policy.get_service_level("SKU_SPECIAL", segment="CZ") == 0.999
        # 2. Segment lookup
        assert policy.get_service_level("SKU_001", segment="AX") == 0.99
        assert policy.get_service_level("SKU_002", segment="CZ") == 0.85
        # 3. Default fallback
        assert policy.get_service_level("SKU_003", segment="UNKNOWN") == 0.95

    def test_optional_eoq_behavior(self):
        # Incomplete inputs -> strictly returns None without guessing
        assert calculate_optional_eoq(annual_demand=1000, order_cost=None) is None
        assert calculate_optional_eoq(annual_demand=None, order_cost=50) is None
        assert calculate_optional_eoq(annual_demand=1000, order_cost=50, holding_cost_per_unit_per_year=None) is None

        # Direct holding cost: D = 1000, S = 50, H = 2 -> EOQ = sqrt(2 * 1000 * 50 / 2) = sqrt(50000) = ~223.6
        eoq = calculate_optional_eoq(annual_demand=1000.0, order_cost=50.0, holding_cost_per_unit_per_year=2.0)
        assert eoq is not None
        assert pytest.approx(eoq, rel=1e-3) == math.sqrt(50000.0)

        # Derived holding cost: unit_cost = 20, rate = 0.10 -> H = 2.0
        eoq_derived = calculate_optional_eoq(
            annual_demand=1000.0,
            order_cost=50.0,
            unit_cost=20.0,
            annual_holding_rate=0.10,
        )
        assert pytest.approx(eoq_derived, rel=1e-3) == math.sqrt(50000.0)

    def test_safety_stock_result_bundle(self):
        res = calculate_safety_stock_result(
            sku_id="SKU_1",
            warehouse_id="WH_1",
            daily_demand_mean=5.0,
            daily_demand_std=1.5,
            lead_time_mean_days=10.0,
            lead_time_std_days=1.0,
            service_level=0.95,
            demand_rate_source="FORECAST",
        )
        assert res.sku_id == "SKU_1"
        assert res.reorder_point > res.safety_stock
        assert res.target_stock_level is not None
        assert res.target_stock_level > res.reorder_point
        assert res.demand_rate_source == "FORECAST"
        d = res.to_dict()
        assert "safety_stock" in d
        assert "reorder_point" in d
        assert d["demand_rate_source"] == "FORECAST"

    def test_safety_stock_invariants_and_variance_preservation(self):
        """Fix 4: Safety stock analytical formula preserves variance and supports rate source tagging."""
        # 1. Zero lead time variance reduction: SS = Z * sigma_D * sqrt(L)
        # Z for 0.95 = 1.64485, sigma_D = 4, L = 16 -> SS = 1.64485 * 4 * 4 = 26.3176
        ss, z = calculate_safety_stock(
            daily_demand_mean=10.0,
            daily_demand_std=4.0,
            lead_time_mean_days=16.0,
            lead_time_std_days=0.0,
            service_level=0.95,
        )
        assert pytest.approx(ss, rel=1e-3) == z * 4.0 * 4.0

        # 2. Historical fallback rate source tagging
        res_hist = calculate_safety_stock_result(
            sku_id="SKU_2",
            warehouse_id="WH_1",
            daily_demand_mean=12.0,
            daily_demand_std=2.5,
            lead_time_mean_days=7.0,
            demand_rate_source="HISTORICAL_FALLBACK",
        )
        assert res_hist.demand_rate_source == "HISTORICAL_FALLBACK"
        assert res_hist.to_dict()["demand_rate_source"] == "HISTORICAL_FALLBACK"

