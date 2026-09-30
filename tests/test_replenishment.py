"""Unit and Integration Tests for Replenishment Solver (Phase 4B-1)."""

import math
import pandas as pd
import pytest
from pydantic import ValidationError

from commerce_ai.inventory.service import InventoryAnalysisResult
from commerce_ai.recommendations.schemas import (
    UrgencyLevel,
    ReplenishmentConstraint,
    ReplenishmentConfig,
    ReplenishmentRecommendation,
    ReplenishmentResult,
)
from commerce_ai.recommendations.replenishment import (
    determine_urgency,
    generate_deterministic_rationale,
    solve_entity_replenishment,
    ReplenishmentSolver,
)


class TestReplenishmentSolver:
    """Test suite covering deterministic replenishment sizing, constraints, and edge cases."""

    def test_healthy_sku_above_rop(self):
        """SKU with net position strictly greater than ROP requires no replenishment."""
        rec = solve_entity_replenishment(
            sku_id="SKU_HEALTHY",
            warehouse_id="WH_1",
            on_hand=150,
            reserved=10,
            on_order=20,  # net_pos = 160
            reorder_point=100.0,
            target_stock_level=200.0,
            forecast_daily_demand=10.0,
            lead_time_days=7.0,
            risk_category="HEALTHY",
            unit_cost=15.0,
        )
        assert rec.recommendation_required is False
        assert rec.recommended_order_qty == 0
        assert rec.shortfall_qty == 0
        assert rec.required_qty == 0
        assert rec.urgency == UrgencyLevel.LOW.value
        assert rec.status == "HEALTHY_ABOVE_ROP"
        assert "exceeds reorder point" in rec.rationale

    def test_exact_rop_triggers_replenishment(self):
        """SKU with net position exactly equal to ROP triggers replenishment candidate."""
        rec = solve_entity_replenishment(
            sku_id="SKU_EXACT_ROP",
            warehouse_id="WH_1",
            on_hand=100,
            reserved=0,
            on_order=0,  # net_pos = 100
            reorder_point=100.0,
            target_stock_level=250.0,
            forecast_daily_demand=10.0,
            lead_time_days=7.0,
            risk_category="UNDERSTOCK",
            unit_cost=20.0,
        )
        assert rec.recommendation_required is True
        assert rec.net_inventory_position == 100
        assert rec.shortfall_qty == 0  # max(0, 100 - 100) = 0
        assert rec.required_qty == 150  # 250 - 100 = 150
        assert rec.recommended_order_qty == 150
        assert rec.estimated_order_cost == 150 * 20.0
        assert rec.urgency in [UrgencyLevel.HIGH.value, UrgencyLevel.MEDIUM.value]

    def test_below_rop_and_shortfall_distinction(self):
        """Verify distinction between shortfall_qty, required_qty, and recommended_order_qty."""
        # Net position = 40, ROP = 100, Target Stock = 200
        # shortfall = 100 - 40 = 60
        # required_qty = 200 - 40 = 160
        rec = solve_entity_replenishment(
            sku_id="SKU_BELOW_ROP",
            warehouse_id="WH_1",
            on_hand=40,
            reserved=0,
            on_order=0,
            reorder_point=100.0,
            target_stock_level=200.0,
            forecast_daily_demand=10.0,
            lead_time_days=7.0,
            risk_category="UNDERSTOCK",
            unit_cost=10.0,
        )
        assert rec.recommendation_required is True
        assert rec.net_inventory_position == 40
        assert rec.shortfall_qty == 60
        assert rec.required_qty == 160
        assert rec.recommended_order_qty == 160
        assert rec.shortfall_qty != rec.required_qty
        assert rec.estimated_order_cost == 1600.0

    def test_zero_inventory_and_critical_urgency(self):
        """Zero on-hand and net position with active demand triggers CRITICAL urgency."""
        rec = solve_entity_replenishment(
            sku_id="SKU_ZERO",
            warehouse_id="WH_1",
            on_hand=0,
            reserved=0,
            on_order=0,
            reorder_point=50.0,
            target_stock_level=120.0,
            forecast_daily_demand=5.0,
            lead_time_days=7.0,
            risk_category="CRITICAL_STOCKOUT",
            unit_cost=10.0,
        )
        assert rec.recommendation_required is True
        assert rec.urgency == UrgencyLevel.CRITICAL.value
        assert rec.net_inventory_position == 0
        assert rec.required_qty == 120
        assert rec.recommended_order_qty == 120
        assert "CRITICAL" in rec.rationale

    def test_negative_net_inventory(self):
        """Negative net position (reserved > on_hand) correctly adds backorders to required qty."""
        # on_hand = 10, reserved = 25, on_order = 0 -> net_position = -15
        # Target stock = 100 -> required_qty = 100 - (-15) = 115
        rec = solve_entity_replenishment(
            sku_id="SKU_NEGATIVE",
            warehouse_id="WH_1",
            on_hand=10,
            reserved=25,
            on_order=0,
            reorder_point=50.0,
            target_stock_level=100.0,
            forecast_daily_demand=5.0,
            lead_time_days=10.0,
            risk_category="CRITICAL_STOCKOUT",
            unit_cost=12.0,
        )
        assert rec.net_inventory_position == -15
        assert rec.recommendation_required is True
        assert rec.urgency == UrgencyLevel.CRITICAL.value
        assert rec.shortfall_qty == 65  # 50 - (-15)
        assert rec.required_qty == 115  # 100 - (-15)
        assert rec.recommended_order_qty == 115
        assert rec.estimated_order_cost == 115 * 12.0

    def test_dormant_sku_and_zero_forecast_suppression(self):
        """Dormant or dead-stock items must never generate automatic replenishment recommendations."""
        # 1. Zero forecast demand with zero stock
        rec_zero_fc = solve_entity_replenishment(
            sku_id="SKU_DORMANT_0",
            warehouse_id="WH_1",
            on_hand=0,
            reserved=0,
            on_order=0,
            reorder_point=10.0,
            target_stock_level=50.0,
            forecast_daily_demand=0.0,
            lead_time_days=14.0,
            risk_category="HEALTHY",
        )
        assert rec_zero_fc.recommendation_required is False
        assert rec_zero_fc.recommended_order_qty == 0
        assert rec_zero_fc.urgency == UrgencyLevel.LOW.value
        assert rec_zero_fc.status == "DORMANT_SUPPRESSED"
        assert "dormant forecast demand" in rec_zero_fc.rationale

        # 2. Dead stock item with positive inventory
        rec_dead = solve_entity_replenishment(
            sku_id="SKU_DEAD",
            warehouse_id="WH_1",
            on_hand=30,
            reorder_point=10.0,
            target_stock_level=50.0,
            forecast_daily_demand=0.0,
            risk_category="DEAD_STOCK",
        )
        assert rec_dead.recommendation_required is False
        assert rec_dead.recommended_order_qty == 0
        assert rec_dead.status == "DORMANT_SUPPRESSED"

    def test_missing_supplier_and_missing_moq(self):
        """Gracefully handle missing supplier and missing MOQ without crashing."""
        rec = solve_entity_replenishment(
            sku_id="SKU_NO_SUPP",
            warehouse_id="WH_1",
            on_hand=20,
            reorder_point=50.0,
            target_stock_level=100.0,
            forecast_daily_demand=5.0,
            supplier_id=None,
            minimum_order_quantity=None,
            unit_cost=10.0,
        )
        assert rec.recommendation_required is True
        assert rec.supplier_id is None
        assert rec.minimum_order_quantity is None
        assert rec.required_qty == 80
        assert rec.recommended_order_qty == 80
        assert len(rec.constraints_applied) == 0

    def test_moq_enforcement(self):
        """Required quantity below MOQ is raised to MOQ."""
        # Required = 150 - 70 = 80 units; MOQ = 100 units -> Recommended = 100 units
        rec = solve_entity_replenishment(
            sku_id="SKU_MOQ",
            warehouse_id="WH_1",
            on_hand=70,
            reorder_point=100.0,
            target_stock_level=150.0,
            forecast_daily_demand=10.0,
            minimum_order_quantity=100,
            unit_cost=5.0,
        )
        assert rec.required_qty == 80
        assert rec.recommended_order_qty == 100
        assert ReplenishmentConstraint.MOQ_APPLIED.value in rec.constraints_applied
        assert rec.estimated_order_cost == 100 * 5.0
        assert "supplier MOQ of 100 units" in rec.rationale

    def test_pack_size_rounding(self):
        """Order quantity is rounded UP to the nearest valid pack/lot multiple."""
        # Required = 300 - 70 = 230 units; Pack size = 50 -> 250 units
        rec = solve_entity_replenishment(
            sku_id="SKU_PACK",
            warehouse_id="WH_1",
            on_hand=70,
            reorder_point=100.0,
            target_stock_level=300.0,
            forecast_daily_demand=10.0,
            pack_size=50,
            unit_cost=10.0,
        )
        assert rec.required_qty == 230
        assert rec.recommended_order_qty == 250
        assert ReplenishmentConstraint.PACK_SIZE_ROUNDED.value in rec.constraints_applied
        assert "pack size multiple of 50 units" in rec.rationale

    def test_combined_moq_and_pack_size_rounding(self):
        """Simultaneous MOQ and pack size rounding are evaluated in proper order."""
        # Required = 40 units; MOQ = 100 units; Pack size = 60 units
        # After MOQ: 100 units -> Nearest multiple of 60 >= 100 is 120 units
        rec = solve_entity_replenishment(
            sku_id="SKU_COMBO",
            warehouse_id="WH_1",
            on_hand=10,
            reorder_point=30.0,
            target_stock_level=50.0,
            forecast_daily_demand=5.0,
            minimum_order_quantity=100,
            pack_size=60,
            unit_cost=10.0,
        )
        assert rec.required_qty == 40
        assert rec.recommended_order_qty == 120
        assert ReplenishmentConstraint.MOQ_APPLIED.value in rec.constraints_applied
        assert ReplenishmentConstraint.PACK_SIZE_ROUNDED.value in rec.constraints_applied
        assert rec.estimated_order_cost == 1200.0

    def test_maximum_order_guards(self):
        """Verify maximum_order_quantity, maximum_inventory_days, and maximum_inventory_value caps."""
        # 1. Maximum order quantity cap
        cfg_max_qty = ReplenishmentConfig(maximum_order_quantity=150)
        rec_max_qty = solve_entity_replenishment(
            sku_id="SKU_CAP1",
            warehouse_id="WH_1",
            on_hand=0,
            reorder_point=100.0,
            target_stock_level=300.0,
            forecast_daily_demand=10.0,
            config=cfg_max_qty,
        )
        assert rec_max_qty.required_qty == 300
        assert rec_max_qty.recommended_order_qty == 150
        assert ReplenishmentConstraint.MAX_ORDER_QTY_CAPPED.value in rec_max_qty.constraints_applied

        # 2. Maximum inventory days cap
        # Net pos = 50, Daily demand = 10 -> Cap at 15 days = 150 units max total
        # Max order = 150 - 50 = 100 units. Target stock calls for 200 (deficit 150).
        cfg_max_days = ReplenishmentConfig(maximum_inventory_days=15.0)
        rec_max_days = solve_entity_replenishment(
            sku_id="SKU_CAP2",
            warehouse_id="WH_1",
            on_hand=50,
            reorder_point=100.0,
            target_stock_level=200.0,
            forecast_daily_demand=10.0,
            config=cfg_max_days,
        )
        assert rec_max_days.recommended_order_qty == 100
        assert ReplenishmentConstraint.MAX_INVENTORY_DAYS_CAPPED.value in rec_max_days.constraints_applied

        # 3. Maximum inventory value cap
        # Unit cost = 20.0, Net pos = 50. Max inventory value = $2000 -> Max total units = 100.
        # Max order = 100 - 50 = 50 units. Required = 150.
        cfg_max_val = ReplenishmentConfig(maximum_inventory_value=2000.0)
        rec_max_val = solve_entity_replenishment(
            sku_id="SKU_CAP3",
            warehouse_id="WH_1",
            on_hand=50,
            reorder_point=100.0,
            target_stock_level=200.0,
            forecast_daily_demand=10.0,
            unit_cost=20.0,
            config=cfg_max_val,
        )
        assert rec_max_val.recommended_order_qty == 50
        assert ReplenishmentConstraint.MAX_INVENTORY_VALUE_CAPPED.value in rec_max_val.constraints_applied

    def test_missing_target_stock_level(self):
        """Missing target stock level returns INSUFFICIENT_POLICY_INPUTS without inventing a quantity."""
        rec = solve_entity_replenishment(
            sku_id="SKU_NO_TARGET",
            warehouse_id="WH_1",
            on_hand=20,
            reorder_point=50.0,
            target_stock_level=None,
            forecast_daily_demand=5.0,
            lead_time_days=7.0,
        )
        assert rec.recommendation_required is True
        assert rec.recommended_order_qty == 0
        assert rec.status == "INSUFFICIENT_POLICY_INPUTS"
        assert "target stock level is unavailable" in rec.rationale

    def test_missing_unit_cost(self):
        """Missing unit cost sets estimated_order_cost to None rather than guessing or defaulting to zero."""
        rec = solve_entity_replenishment(
            sku_id="SKU_NO_COST",
            warehouse_id="WH_1",
            on_hand=20,
            reorder_point=50.0,
            target_stock_level=100.0,
            forecast_daily_demand=5.0,
            unit_cost=None,
        )
        assert rec.recommendation_required is True
        assert rec.recommended_order_qty == 80
        assert rec.unit_cost is None
        assert rec.estimated_order_cost is None

    def test_schema_validations_and_negative_rejection(self):
        """Pydantic model rejects negative quantities for physical constraints."""
        with pytest.raises(ValidationError):
            ReplenishmentRecommendation(
                sku_id="SKU_INV",
                warehouse_id="WH_1",
                recommendation_required=True,
                urgency="INVALID_TIER",  # Invalid urgency tier
                risk_category="UNDERSTOCK",
                on_hand=-10,  # Negative on_hand rejected
                net_inventory_position=0,
                forecast_daily_demand=5.0,
                lead_time_days=7.0,
                reorder_point=50.0,
                shortfall_qty=0,
                required_qty=0,
                recommended_order_qty=0,
                rationale="test",
            )

    def test_urgency_tier_classification(self):
        """Verify deterministic transitions between CRITICAL, HIGH, MEDIUM, and LOW urgency."""
        # 1. CRITICAL: Net position <= 0 with active demand
        u1 = determine_urgency(
            net_position=0,
            reorder_point=50.0,
            forecast_daily_demand=5.0,
            lead_time_days=10.0,
            risk_category="CRITICAL_STOCKOUT",
        )
        assert u1 == UrgencyLevel.CRITICAL.value

        # 2. HIGH: Net position <= ROP and days_to_runout <= lead_time_days
        u2 = determine_urgency(
            net_position=30,
            reorder_point=50.0,
            forecast_daily_demand=5.0,
            lead_time_days=10.0,
            risk_category="UNDERSTOCK",
            days_to_runout=6.0,  # runout in 6 days < lead time 10 days
        )
        assert u2 == UrgencyLevel.HIGH.value

        # 3. MEDIUM: Net position <= ROP and days_to_runout > lead_time_days
        u3 = determine_urgency(
            net_position=40,
            reorder_point=50.0,
            forecast_daily_demand=2.0,
            lead_time_days=10.0,
            risk_category="UNDERSTOCK",
            days_to_runout=20.0,  # runout in 20 days > lead time 10 days
        )
        assert u3 == UrgencyLevel.MEDIUM.value

        # 4. LOW: Net position > ROP
        u4 = determine_urgency(
            net_position=80,
            reorder_point=50.0,
            forecast_daily_demand=2.0,
            lead_time_days=10.0,
            risk_category="HEALTHY",
        )
        assert u4 == UrgencyLevel.LOW.value

    def test_optional_eoq_usage(self):
        """When use_eoq is enabled and EOQ is available, batch size uses EOQ."""
        cfg_eoq = ReplenishmentConfig(use_eoq=True)
        # Required = 150 - 100 = 50 units. EOQ = 120 units -> order_qty = 120
        rec = solve_entity_replenishment(
            sku_id="SKU_EOQ",
            warehouse_id="WH_1",
            on_hand=100,
            reorder_point=110.0,
            target_stock_level=150.0,
            forecast_daily_demand=10.0,
            eoq=120.0,
            config=cfg_eoq,
        )
        assert rec.required_qty == 50
        assert rec.recommended_order_qty == 120

    def test_portfolio_solve_multi_sku_multi_warehouse(self):
        """Test ReplenishmentSolver.solve across multi-SKU, multi-warehouse portfolio."""
        positions_df = pd.DataFrame([
            {"sku_id": "SKU_001", "warehouse_id": "WH_01", "on_hand": 20, "reserved": 0, "on_order": 0, "net_position": 20},
            {"sku_id": "SKU_001", "warehouse_id": "WH_02", "on_hand": 150, "reserved": 0, "on_order": 0, "net_position": 150},
            {"sku_id": "SKU_002", "warehouse_id": "WH_01", "on_hand": 0, "reserved": 0, "on_order": 0, "net_position": 0},
            {"sku_id": "SKU_DORMANT", "warehouse_id": "WH_01", "on_hand": 0, "reserved": 0, "on_order": 0, "net_position": 0},
        ])

        safety_stocks_df = pd.DataFrame([
            {"sku_id": "SKU_001", "warehouse_id": "WH_01", "reorder_point": 60.0, "target_stock_level": 120.0, "daily_demand_mean": 5.0, "lead_time_days": 10.0},
            {"sku_id": "SKU_001", "warehouse_id": "WH_02", "reorder_point": 60.0, "target_stock_level": 120.0, "daily_demand_mean": 5.0, "lead_time_days": 10.0},
            {"sku_id": "SKU_002", "warehouse_id": "WH_01", "reorder_point": 40.0, "target_stock_level": 100.0, "daily_demand_mean": 8.0, "lead_time_days": 7.0},
            {"sku_id": "SKU_DORMANT", "warehouse_id": "WH_01", "reorder_point": 10.0, "target_stock_level": 50.0, "daily_demand_mean": 0.0, "lead_time_days": 14.0},
        ])

        risks_df = pd.DataFrame([
            {"sku_id": "SKU_001", "warehouse_id": "WH_01", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0, "days_to_runout": 4.0},
            {"sku_id": "SKU_001", "warehouse_id": "WH_02", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0, "days_to_runout": 30.0},
            {"sku_id": "SKU_002", "warehouse_id": "WH_01", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 8.0, "days_to_runout": 0.0},
            {"sku_id": "SKU_DORMANT", "warehouse_id": "WH_01", "risk_category": "HEALTHY", "mean_daily_forecast": 0.0, "days_to_runout": None},
        ])

        lead_times_df = pd.DataFrame([
            {"sku_id": "SKU_001", "supplier_id": "SUPP_A", "mean_lead_time_days": 10.0},
            {"sku_id": "SKU_002", "supplier_id": "SUPP_B", "mean_lead_time_days": 7.0},
            {"sku_id": "SKU_DORMANT", "supplier_id": "SUPP_C", "mean_lead_time_days": 14.0},
        ])

        products_df = pd.DataFrame([
            {"sku_id": "SKU_001", "preferred_supplier_id": "SUPP_A", "unit_cost": 25.0, "pack_size": 20},
            {"sku_id": "SKU_002", "preferred_supplier_id": "SUPP_B", "unit_cost": 50.0, "pack_size": 10},
            {"sku_id": "SKU_DORMANT", "preferred_supplier_id": "SUPP_C", "unit_cost": 10.0, "pack_size": 1},
        ])

        suppliers_df = pd.DataFrame([
            {"supplier_id": "SUPP_A", "minimum_order_quantity": 50},
            {"supplier_id": "SUPP_B", "minimum_order_quantity": 100},
            {"supplier_id": "SUPP_C", "minimum_order_quantity": 10},
        ])

        solver = ReplenishmentSolver()
        result: ReplenishmentResult = solver.solve(
            positions=positions_df,
            safety_stocks=safety_stocks_df,
            risks=risks_df,
            lead_times=lead_times_df,
            products=products_df,
            suppliers=suppliers_df,
        )

        assert len(result.recommendations) == 4
        assert len(result.actionable_recommendations) == 2
        assert len(result.non_recommendations) == 2
        assert len(result.excluded_dormant) == 1

        # Check SKU_001 @ WH_01: net_pos 20 <= ROP 60 -> required = 100 -> pack_size 20 -> 100 >= MOQ 50
        rec_s1_w1 = [r for r in result.recommendations if r.sku_id == "SKU_001" and r.warehouse_id == "WH_01"][0]
        assert rec_s1_w1.recommendation_required is True
        assert rec_s1_w1.recommended_order_qty == 100
        assert rec_s1_w1.estimated_order_cost == 100 * 25.0
        assert rec_s1_w1.urgency == UrgencyLevel.HIGH.value  # runout in 4d <= lead time 10d

        # Check SKU_001 @ WH_02: net_pos 150 > ROP 60 -> No recommendation
        rec_s1_w2 = [r for r in result.recommendations if r.sku_id == "SKU_001" and r.warehouse_id == "WH_02"][0]
        assert rec_s1_w2.recommendation_required is False
        assert rec_s1_w2.recommended_order_qty == 0

        # Check SKU_002 @ WH_01: net_pos 0 -> required = 100 -> MOQ 100, pack 10 -> 100 units
        rec_s2_w1 = [r for r in result.recommendations if r.sku_id == "SKU_002" and r.warehouse_id == "WH_01"][0]
        assert rec_s2_w1.recommendation_required is True
        assert rec_s2_w1.urgency == UrgencyLevel.CRITICAL.value
        assert rec_s2_w1.recommended_order_qty == 100
        assert rec_s2_w1.estimated_order_cost == 100 * 50.0

        # Check SKU_DORMANT: dormant demand -> suppressed
        rec_dormant = [r for r in result.recommendations if r.sku_id == "SKU_DORMANT"][0]
        assert rec_dormant.recommendation_required is False
        assert rec_dormant.status == "DORMANT_SUPPRESSED"

        # Check Summary
        summary = result.summary
        assert summary["total_entities_evaluated"] == 4
        assert summary["replenishment_recommended_count"] == 2
        assert summary["dormant_excluded_count"] == 1
        assert summary["total_recommended_units"] == 200
        assert summary["total_estimated_spend"] == (100 * 25.0) + (100 * 50.0)

        # DataFrame export
        df = result.to_dataframe()
        assert len(df) == 4
        assert "recommended_order_qty" in df.columns
        assert "rationale" in df.columns

    def test_end_to_end_from_inventory_service_result(self):
        """Verify solver directly consumes Phase 4A InventoryAnalysisResult object."""
        analysis = InventoryAnalysisResult(
            positions=pd.DataFrame([
                {"sku_id": "SKU_X", "warehouse_id": "WH_A", "on_hand": 15, "reserved": 5, "on_order": 0, "net_position": 10},
            ]),
            safety_stocks=pd.DataFrame([
                {"sku_id": "SKU_X", "warehouse_id": "WH_A", "reorder_point": 30.0, "target_stock_level": 70.0, "daily_demand_mean": 4.0, "lead_time_days": 5.0},
            ]),
            risks=pd.DataFrame([
                {"sku_id": "SKU_X", "warehouse_id": "WH_A", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 4.0, "days_to_runout": 3.0},
            ]),
            lead_times=pd.DataFrame([
                {"sku_id": "SKU_X", "supplier_id": "SUPP_1", "mean_lead_time_days": 5.0},
            ]),
            summary={"total_series_analyzed": 1},
        )

        solver = ReplenishmentSolver()
        res = solver.solve(inventory_analysis=analysis)

        assert len(res.recommendations) == 1
        rec = res.recommendations[0]
        assert rec.sku_id == "SKU_X"
        assert rec.warehouse_id == "WH_A"
        assert rec.recommendation_required is True
        # Net pos = 10 <= ROP 30 -> required = 70 - 10 = 60
        assert rec.required_qty == 60
        assert rec.recommended_order_qty == 60
        assert rec.shortfall_qty == 20
        # days_to_runout (3.0) <= lead_time (5.0) -> HIGH
        assert rec.urgency == UrgencyLevel.HIGH.value

