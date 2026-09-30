"""Unit and Integration Tests for Warehouse Rebalancing Engine (Phase 4C).

Validates all 26 required edge cases, constraints, source protection,
multi-destination allocation, deterministic rationales, and end-to-end integration.
"""

from __future__ import annotations

from datetime import date
import pytest
import pandas as pd
from pydantic import ValidationError

from commerce_ai.inventory.service import InventoryAnalysisResult, InventoryService
from commerce_ai.recommendations.schemas import (
    CostDataStatus,
    ProposalStatus,
    TransferConstraint,
    TransferPriority,
    WarehouseRebalancingConfig,
    WarehouseRebalancingResult,
    WarehouseTransferRecommendation,
)
from commerce_ai.recommendations.warehouse_rebalancing import (
    determine_transfer_priority,
    generate_deterministic_transfer_id,
    generate_transfer_rationale,
    WarehouseRebalancingService,
)


class TestWarehouseRebalancingEngine:
    """Complete test suite verifying Phase 4C Warehouse Rebalancing Engine."""

    # 1. Edge Case: No transfer needed
    def test_no_transfer_needed(self):
        """All warehouses healthy above ROP -> 0 recommendations."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 500, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 400, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 200.0, "target_stock_level": 400.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 150.0, "target_stock_level": 300.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "HEALTHY", "mean_daily_forecast": 10.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "HEALTHY", "mean_daily_forecast": 8.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 0
        assert result.total_transfer_quantity == 0.0
        assert result.cost_data_status == CostDataStatus.NO_COST_DATA.value

    # 2. Edge Case: One source -> one destination
    def test_one_source_one_destination(self):
        """Direct single-source single-destination transfer meeting requirement."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 20, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 50.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 80.0, "target_stock_level": 150.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        assert rec.source_warehouse_id == "WH-B"
        assert rec.destination_warehouse_id == "WH-A"
        assert rec.transfer_quantity == 80.0  # target 100 - net 20 = 80
        assert rec.source_transferable_surplus == 120.0  # net 200 - rop 80 = 120
        assert result.source_inventory_remaining["SKU-1"]["WH-B"] == 40.0
        assert len(result.unmet_destination_demand) == 0

    # 3. Edge Case: Multiple source warehouses
    def test_multiple_source_warehouses(self):
        """Destination with large deficit satisfied by two source warehouses in surplus order."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 10, "reserved": 0, "on_order": 0},  # needs 90
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 150, "reserved": 0, "on_order": 0},  # surplus 50
            {"sku_id": "SKU-1", "warehouse_id": "WH-C", "on_hand": 160, "reserved": 0, "on_order": 0},  # surplus 60
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 50.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 100.0, "target_stock_level": 120.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-C", "reorder_point": 100.0, "target_stock_level": 120.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-C", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 2
        # WH-C has higher surplus (60 > 50), so WH-C supplies first
        assert result.recommendations[0].source_warehouse_id == "WH-C"
        assert result.recommendations[0].transfer_quantity == 60.0
        # WH-B supplies remaining 30 units
        assert result.recommendations[1].source_warehouse_id == "WH-B"
        assert result.recommendations[1].transfer_quantity == 30.0
        assert result.total_transfer_quantity == 90.0
        assert result.source_inventory_remaining["SKU-1"]["WH-C"] == 0.0
        assert result.source_inventory_remaining["SKU-1"]["WH-B"] == 20.0

    # 4. Edge Case: Multiple destination warehouses
    def test_multiple_destination_warehouses(self):
        """Single source supplies multiple destinations without conflict."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 50, "reserved": 0, "on_order": 0},  # needs 50
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 60, "reserved": 0, "on_order": 0},  # needs 40
            {"sku_id": "SKU-1", "warehouse_id": "WH-C", "on_hand": 300, "reserved": 0, "on_order": 0}, # surplus 200
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 80.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 80.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-C", "reorder_point": 100.0, "target_stock_level": 150.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-C", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 2
        dest_map = {r.destination_warehouse_id: r.transfer_quantity for r in result.recommendations}
        assert dest_map["WH-A"] == 50.0
        assert dest_map["WH-B"] == 40.0
        assert result.total_transfer_quantity == 90.0
        assert result.source_inventory_remaining["SKU-1"]["WH-C"] == 110.0

    # 5. Edge Case: Source surplus insufficient
    def test_source_surplus_insufficient(self):
        """Source surplus is less than destination requirement; transfer capped at surplus."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 0, "reserved": 0, "on_order": 0},   # needs 100
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 140, "reserved": 0, "on_order": 0}, # surplus 40
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 50.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 100.0, "target_stock_level": 120.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 1
        assert result.recommendations[0].transfer_quantity == 40.0
        assert len(result.unmet_destination_demand) == 1
        unmet = result.unmet_destination_demand[0]
        assert unmet["destination_warehouse_id"] == "WH-A"
        assert unmet["unmet_quantity"] == 60.0
        assert unmet["requested_quantity"] == 100.0

    # 6. Edge Case: Destination requirement greater than total network surplus
    def test_destination_requirement_greater_than_total_network_surplus(self):
        """All network surplus exhausted; unmet balance recorded."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 0, "reserved": 0, "on_order": 0},   # needs 200
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 150, "reserved": 0, "on_order": 0}, # surplus 50
            {"sku_id": "SKU-1", "warehouse_id": "WH-C", "on_hand": 130, "reserved": 0, "on_order": 0}, # surplus 30
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 100.0, "target_stock_level": 200.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 100.0, "target_stock_level": 120.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-C", "reorder_point": 100.0, "target_stock_level": 120.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-C", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert result.total_transfer_quantity == 80.0
        assert len(result.unmet_destination_demand) == 1
        assert result.unmet_destination_demand[0]["unmet_quantity"] == 120.0

    # 7. Edge Case: Source protected at ROP (Section 13)
    def test_source_protected_at_rop(self):
        """Source on_hand=500, ROP=300 -> surplus=200. Destination needs 400. Transfer must be 200."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "on_hand": 0, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "on_hand": 500, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "reorder_point": 150.0, "target_stock_level": 400.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "reorder_point": 300.0, "target_stock_level": 450.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        assert rec.transfer_quantity == 200.0
        # Source inventory remaining surplus is 0, so net position after transfer is exactly 300 (ROP)
        assert result.source_inventory_remaining["SKU-1"]["WH-SRC"] == 0.0
        assert rec.source_net_inventory_position - rec.transfer_quantity == 300.0

    # 8. Edge Case: Multiple destinations competing for same source (Section 15)
    def test_multiple_destinations_competing_for_same_source(self):
        """Source surplus=200. Dest A needs 150, Dest B needs 150. Total allocated cannot exceed 200."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 0, "reserved": 0, "on_order": 0},   # needs 150
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 0, "reserved": 0, "on_order": 0},   # needs 150
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "on_hand": 500, "reserved": 0, "on_order": 0}, # surplus 200
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 100.0, "target_stock_level": 150.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 100.0, "target_stock_level": 150.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "reorder_point": 300.0, "target_stock_level": 400.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        # Total transferred must NOT exceed 200
        assert result.total_transfer_quantity == 200.0
        # WH-A (CRITICAL) receives 150, WH-B (UNDERSTOCK) receives remaining 50
        rec_map = {r.destination_warehouse_id: r.transfer_quantity for r in result.recommendations}
        assert rec_map["WH-A"] == 150.0
        assert rec_map["WH-B"] == 50.0
        # WH-B has unmet need of 100
        assert len(result.unmet_destination_demand) == 1
        assert result.unmet_destination_demand[0]["destination_warehouse_id"] == "WH-B"
        assert result.unmet_destination_demand[0]["unmet_quantity"] == 100.0

    # 9. Edge Case: Critical destination gets priority over High/Medium
    def test_critical_destination_gets_priority(self):
        """CRITICAL gets first claim on scarce surplus over MEDIUM even if MEDIUM has larger deficit."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-MED", "on_hand": 50, "reserved": 0, "on_order": 0},  # needs 150, MEDIUM
            {"sku_id": "SKU-1", "warehouse_id": "WH-CRIT", "on_hand": 0, "reserved": 0, "on_order": 0},  # needs 50, CRITICAL
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "on_hand": 150, "reserved": 0, "on_order": 0}, # surplus 50
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-MED", "reorder_point": 100.0, "target_stock_level": 200.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-CRIT", "reorder_point": 100.0, "target_stock_level": 50.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "reorder_point": 100.0, "target_stock_level": 120.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-MED", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0}, # below ROP -> MEDIUM
            {"sku_id": "SKU-1", "warehouse_id": "WH-CRIT", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 1
        assert result.recommendations[0].destination_warehouse_id == "WH-CRIT"
        assert result.recommendations[0].transfer_quantity == 50.0
        assert len(result.unmet_destination_demand) == 1
        assert result.unmet_destination_demand[0]["destination_warehouse_id"] == "WH-MED"

    # 10. Edge Case: Dormant destination excluded
    def test_dormant_destination_excluded(self):
        """Destination below threshold demand (<=0.01) is placed in excluded_dormant."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DORM", "on_hand": 0, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DORM", "reorder_point": 50.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "reorder_point": 80.0, "target_stock_level": 100.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DORM", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 0.005},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 0
        assert len(result.excluded_dormant) == 1
        assert result.excluded_dormant[0]["warehouse_id"] == "WH-DORM"
        assert result.excluded_dormant[0]["reason"] == "DORMANT_DEMAND"

    # 11. Edge Case: Dead-stock destination excluded
    def test_dead_stock_destination_excluded(self):
        """Destination with risk DEAD_STOCK is excluded from transfers."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEAD", "on_hand": 5, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEAD", "reorder_point": 20.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "reorder_point": 80.0, "target_stock_level": 100.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEAD", "risk_category": "DEAD_STOCK", "mean_daily_forecast": 0.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 0
        assert len(result.excluded_dormant) == 1
        assert result.excluded_dormant[0]["reason"] == "DEAD_STOCK"

    # 12. Edge Case: Zero demand destination excluded
    def test_zero_demand_destination_excluded(self):
        """Active zero demand warehouse does not generate transfer recommendation."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-ZERO", "on_hand": 0, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-ZERO", "reorder_point": 10.0, "target_stock_level": 50.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "reorder_point": 80.0, "target_stock_level": 100.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-ZERO", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 0.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 0
        assert len(result.excluded_dormant) == 1

    # 13. Edge Case: Missing target stock
    def test_missing_target_stock(self):
        """Missing target_stock_level records in insufficient_policy_inputs, no arbitrary transfer."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "on_hand": 10, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "reorder_point": 50.0, "target_stock_level": None},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "reorder_point": 80.0, "target_stock_level": 100.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 0
        assert len(result.insufficient_policy_inputs) == 1
        assert result.insufficient_policy_inputs[0]["reason"] == "MISSING_TARGET_STOCK"

    # 14. Edge Case: Missing reorder point
    def test_missing_reorder_point(self):
        """Missing reorder_point records in insufficient_policy_inputs, no arbitrary transfer."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "on_hand": 10, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "reorder_point": None, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "reorder_point": 80.0, "target_stock_level": 100.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 0
        assert len(result.insufficient_policy_inputs) == 1
        assert result.insufficient_policy_inputs[0]["reason"] == "MISSING_REORDER_POINT"

    # 15. Edge Case: Missing transfer cost
    def test_missing_transfer_cost(self):
        """When transfer cost is unavailable, estimated_transfer_cost=None and status NO_COST_DATA."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 20, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 50.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 80.0, "target_stock_level": 150.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 1
        assert result.recommendations[0].estimated_transfer_cost is None
        assert result.estimated_total_transfer_cost is None
        assert result.cost_data_status == CostDataStatus.NO_COST_DATA.value

    # 16. Edge Case: Transfer cost calculation
    def test_transfer_cost_calculation(self):
        """Valid unit transfer cost computes exact cost and COMPLETE_COST_DATA."""
        cfg = WarehouseRebalancingConfig(
            transfer_costs_per_unit={("WH-B", "WH-A"): 2.50}
        )
        service = WarehouseRebalancingService(config=cfg)
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 20, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 50.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 80.0, "target_stock_level": 150.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        assert rec.transfer_quantity == 80.0
        assert rec.estimated_transfer_cost == 200.0  # 80 * 2.50
        assert result.estimated_total_transfer_cost == 200.0
        assert result.cost_data_status == CostDataStatus.COMPLETE_COST_DATA.value

    # 17. Edge Case: Missing distance
    def test_missing_distance(self):
        """Missing distance does not invent geography; falls back to deterministic warehouse_id tie-breaker."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "on_hand": 20, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-Z", "on_hand": 200, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-A", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "reorder_point": 50.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-Z", "reorder_point": 100.0, "target_stock_level": 150.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-A", "reorder_point": 100.0, "target_stock_level": 150.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-Z", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-A", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        # Both sources have 100 surplus. Tie-breaker selects WH-SRC-A alphabetically
        assert len(result.recommendations) == 1
        assert result.recommendations[0].source_warehouse_id == "WH-SRC-A"
        assert result.recommendations[0].distance_km is None

    # 18. Edge Case: Deterministic source selection with distance
    def test_deterministic_source_selection(self):
        """When surplus is equal, shorter distance source is selected over longer distance."""
        cfg = WarehouseRebalancingConfig(
            warehouse_distances_km={
                ("WH-SRC-FAR", "WH-DEST"): 500.0,
                ("WH-SRC-NEAR", "WH-DEST"): 60.0,
            }
        )
        service = WarehouseRebalancingService(config=cfg)
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "on_hand": 20, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-FAR", "on_hand": 200, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-NEAR", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "reorder_point": 50.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-FAR", "reorder_point": 100.0, "target_stock_level": 150.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-NEAR", "reorder_point": 100.0, "target_stock_level": 150.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-DEST", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-FAR", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC-NEAR", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        assert rec.source_warehouse_id == "WH-SRC-NEAR"
        assert rec.distance_km == 60.0

    # 19. Edge Case: Deterministic recommendation IDs
    def test_deterministic_recommendation_ids(self):
        """Same parameters generate byte-for-byte identical recommendation IDs."""
        id1 = generate_deterministic_transfer_id("SKU-1", "WH-B", "WH-A", 80.0, as_of_date="2026-03-30")
        id2 = generate_deterministic_transfer_id("SKU-1", "WH-B", "WH-A", 80.0, as_of_date="2026-03-30")
        id3 = generate_deterministic_transfer_id("SKU-1", "WH-B", "WH-A", 75.0, as_of_date="2026-03-30")
        assert id1 == id2
        assert id1 != id3
        assert id1.startswith("XFER_SKU-1_WH-B_WH-A_")

    # 20. Edge Case: No negative transfer quantities
    def test_no_negative_transfer_quantities(self):
        """Schema rejects negative or zero transfer quantity."""
        with pytest.raises(ValidationError):
            WarehouseTransferRecommendation(
                recommendation_id="XFER-1",
                sku_id="SKU-1",
                source_warehouse_id="WH-B",
                destination_warehouse_id="WH-A",
                transfer_quantity=-10.0,  # Negative
                source_net_inventory_position=200.0,
                destination_net_inventory_position=20.0,
                source_reorder_point=100.0,
                destination_reorder_point=50.0,
                source_transferable_surplus=100.0,
                destination_required_qty=80.0,
                destination_risk_category="UNDERSTOCK",
                priority="HIGH",
                forecast_daily_demand=5.0,
                rationale="Test",
            )

        with pytest.raises(ValidationError):
            WarehouseTransferRecommendation(
                recommendation_id="XFER-1",
                sku_id="SKU-1",
                source_warehouse_id="WH-B",
                destination_warehouse_id="WH-A",
                transfer_quantity=0.0,  # Zero
                source_net_inventory_position=200.0,
                destination_net_inventory_position=20.0,
                source_reorder_point=100.0,
                destination_reorder_point=50.0,
                source_transferable_surplus=100.0,
                destination_required_qty=80.0,
                destination_risk_category="UNDERSTOCK",
                priority="HIGH",
                forecast_daily_demand=5.0,
                rationale="Test",
            )

    # 21. Edge Case: No source inventory double allocation
    def test_no_source_inventory_double_allocation(self):
        """Source surplus of 100 cannot be allocated to 3 destinations requesting 50 each."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-1", "on_hand": 0, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-2", "on_hand": 0, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-3", "on_hand": 0, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "on_hand": 200, "reserved": 0, "on_order": 0}, # surplus 100
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-1", "reorder_point": 50.0, "target_stock_level": 50.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-2", "reorder_point": 50.0, "target_stock_level": 50.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-3", "reorder_point": 50.0, "target_stock_level": 50.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "reorder_point": 100.0, "target_stock_level": 120.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-1", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-2", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-3", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-SRC", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        # Total transferred must strictly be 100.0
        assert result.total_transfer_quantity == 100.0
        assert result.source_inventory_remaining["SKU-1"]["WH-SRC"] == 0.0
        assert len(result.unmet_destination_demand) == 1
        assert result.unmet_destination_demand[0]["unmet_quantity"] == 50.0

    # 22. Edge Case: Multiple SKUs
    def test_multiple_skus(self):
        """SKU boundaries are strictly respected; no cross-SKU transfers."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            # SKU-A
            {"sku_id": "SKU-A", "warehouse_id": "WH-1", "on_hand": 10, "reserved": 0, "on_order": 0}, # needs 40
            {"sku_id": "SKU-A", "warehouse_id": "WH-2", "on_hand": 100, "reserved": 0, "on_order": 0}, # surplus 60
            # SKU-B
            {"sku_id": "SKU-B", "warehouse_id": "WH-1", "on_hand": 200, "reserved": 0, "on_order": 0}, # surplus 100
            {"sku_id": "SKU-B", "warehouse_id": "WH-2", "on_hand": 20, "reserved": 0, "on_order": 0}, # needs 50
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-A", "warehouse_id": "WH-1", "reorder_point": 30.0, "target_stock_level": 50.0},
            {"sku_id": "SKU-A", "warehouse_id": "WH-2", "reorder_point": 40.0, "target_stock_level": 80.0},
            {"sku_id": "SKU-B", "warehouse_id": "WH-1", "reorder_point": 100.0, "target_stock_level": 150.0},
            {"sku_id": "SKU-B", "warehouse_id": "WH-2", "reorder_point": 40.0, "target_stock_level": 70.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-A", "warehouse_id": "WH-1", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-A", "warehouse_id": "WH-2", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-B", "warehouse_id": "WH-1", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-B", "warehouse_id": "WH-2", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 2
        sku_a_rec = next(r for r in result.recommendations if r.sku_id == "SKU-A")
        sku_b_rec = next(r for r in result.recommendations if r.sku_id == "SKU-B")

        assert sku_a_rec.source_warehouse_id == "WH-2" and sku_a_rec.destination_warehouse_id == "WH-1"
        assert sku_a_rec.transfer_quantity == 40.0

        assert sku_b_rec.source_warehouse_id == "WH-1" and sku_b_rec.destination_warehouse_id == "WH-2"
        assert sku_b_rec.transfer_quantity == 50.0

    # 23. Edge Case: Section 14 Multi-warehouse example
    def test_section_14_multi_warehouse_example(self):
        """Exact verification of Section 14 specification scenario."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-100", "warehouse_id": "WH-A", "on_hand": 20, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-100", "warehouse_id": "WH-B", "on_hand": 500, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-100", "warehouse_id": "WH-C", "on_hand": 280, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-100", "warehouse_id": "WH-A", "reorder_point": 150.0, "target_stock_level": 300.0},
            {"sku_id": "SKU-100", "warehouse_id": "WH-B", "reorder_point": 200.0, "target_stock_level": 350.0},
            {"sku_id": "SKU-100", "warehouse_id": "WH-C", "reorder_point": 200.0, "target_stock_level": 300.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-100", "warehouse_id": "WH-A", "risk_category": "CRITICAL_STOCKOUT", "mean_daily_forecast": 10.0},
            {"sku_id": "SKU-100", "warehouse_id": "WH-B", "risk_category": "HEALTHY", "mean_daily_forecast": 10.0},
            {"sku_id": "SKU-100", "warehouse_id": "WH-C", "risk_category": "HEALTHY", "mean_daily_forecast": 10.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        assert rec.source_warehouse_id == "WH-B"
        assert rec.destination_warehouse_id == "WH-A"
        assert rec.transfer_quantity == 280.0
        assert rec.source_transferable_surplus == 300.0
        # WH-B remaining surplus is 20 (leaving net position at 220, above ROP of 200)
        assert result.source_inventory_remaining["SKU-100"]["WH-B"] == 20.0
        # WH-C surplus remained untouched at 80
        assert result.source_inventory_remaining["SKU-100"]["WH-C"] == 80.0

    # 24. Edge Case: Approval flag always True
    def test_approval_flag_always_true(self):
        """Recommendations strictly require human approval."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 20, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 50.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 80.0, "target_stock_level": 150.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        for r in result.recommendations:
            assert r.approval_required is True

        with pytest.raises(ValidationError):
            WarehouseTransferRecommendation(
                recommendation_id="XFER-1",
                sku_id="SKU-1",
                source_warehouse_id="WH-B",
                destination_warehouse_id="WH-A",
                transfer_quantity=80.0,
                source_net_inventory_position=200.0,
                destination_net_inventory_position=20.0,
                source_reorder_point=80.0,
                destination_reorder_point=50.0,
                source_transferable_surplus=120.0,
                destination_required_qty=80.0,
                destination_risk_category="UNDERSTOCK",
                priority="HIGH",
                forecast_daily_demand=5.0,
                rationale="Test",
                approval_required=False,  # Rejection
            )

    # 25. Edge Case: Status always DRAFT
    def test_status_always_draft(self):
        """Status must strictly be DRAFT in Phase 4C."""
        service = WarehouseRebalancingService()
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 20, "reserved": 0, "on_order": 0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 200, "reserved": 0, "on_order": 0},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 50.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 80.0, "target_stock_level": 150.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        for r in result.recommendations:
            assert r.status == ProposalStatus.DRAFT.value

        with pytest.raises(ValidationError):
            WarehouseTransferRecommendation(
                recommendation_id="XFER-1",
                sku_id="SKU-1",
                source_warehouse_id="WH-B",
                destination_warehouse_id="WH-A",
                transfer_quantity=80.0,
                source_net_inventory_position=200.0,
                destination_net_inventory_position=20.0,
                source_reorder_point=80.0,
                destination_reorder_point=50.0,
                source_transferable_surplus=120.0,
                destination_required_qty=80.0,
                destination_risk_category="UNDERSTOCK",
                priority="HIGH",
                forecast_daily_demand=5.0,
                rationale="Test",
                status="APPROVED",  # Rejection
            )

    # 26. Edge Case: Deterministic rationale
    def test_deterministic_rationale(self):
        """Rationale explains positions, target, ROP, transferable units, and constraints without LLM."""
        rationale = generate_transfer_rationale(
            sku_id="SKU-100",
            source_warehouse_id="WH-B",
            destination_warehouse_id="WH-A",
            destination_net_pos=20.0,
            destination_target=300.0,
            destination_risk="CRITICAL_STOCKOUT",
            source_net_pos=500.0,
            source_rop=200.0,
            source_surplus=300.0,
            transfer_qty=280.0,
            constraints_applied=["TRANSFER_PACK_SIZE_ROUNDED"],
        )
        assert "Warehouse WH-A has a net inventory position of 20 units" in rationale
        assert "target stock level of 300 units and is classified as CRITICAL_STOCKOUT" in rationale
        assert "Warehouse WH-B has 500 units against a reorder point of 200 units" in rationale
        assert "leaving 300 transferable units" in rationale
        assert "transfer of 280 units is recommended" in rationale
        assert "keeping WH-B at or above its reorder point" in rationale
        assert "TRANSFER_PACK_SIZE_ROUNDED" in rationale

    # 27. Constraints: Pack size rounding, min/max transfer quantities
    def test_transfer_constraints_applied(self):
        """Pack size rounds down without violating surplus; caps at max quantity."""
        cfg = WarehouseRebalancingConfig(
            transfer_pack_size=10,
            maximum_transfer_quantity=50,
        )
        service = WarehouseRebalancingService(config=cfg)
        positions = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "on_hand": 10, "reserved": 0, "on_order": 0},  # needs 90
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "on_hand": 200, "reserved": 0, "on_order": 0}, # surplus 120
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "reorder_point": 50.0, "target_stock_level": 100.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "reorder_point": 80.0, "target_stock_level": 150.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-1", "warehouse_id": "WH-A", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 5.0},
            {"sku_id": "SKU-1", "warehouse_id": "WH-B", "risk_category": "HEALTHY", "mean_daily_forecast": 5.0},
        ])

        result = service.generate(positions=positions, safety_stocks=safety_stocks, risks=risks)
        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        # Needs 90, surplus 120, capped at max 50
        assert rec.transfer_quantity == 50.0
        assert TransferConstraint.MAX_TRANSFER_QTY_CAPPED.value in rec.constraints_applied

    # 28. End-to-end integration consuming InventoryAnalysisResult
    def test_end_to_end_from_inventory_service_result(self):
        """End-to-end test feeding InventoryAnalysisResult bundle into WarehouseRebalancingService."""
        positions = pd.DataFrame([
            {"sku_id": "SKU-INT", "warehouse_id": "WH-EAST", "on_hand": 20, "reserved": 0, "on_order": 0, "net_position": 20},
            {"sku_id": "SKU-INT", "warehouse_id": "WH-WEST", "on_hand": 300, "reserved": 0, "on_order": 0, "net_position": 300},
        ])
        safety_stocks = pd.DataFrame([
            {"sku_id": "SKU-INT", "warehouse_id": "WH-EAST", "reorder_point": 60.0, "target_stock_level": 120.0},
            {"sku_id": "SKU-INT", "warehouse_id": "WH-WEST", "reorder_point": 100.0, "target_stock_level": 150.0},
        ])
        risks = pd.DataFrame([
            {"sku_id": "SKU-INT", "warehouse_id": "WH-EAST", "risk_category": "UNDERSTOCK", "mean_daily_forecast": 6.0},
            {"sku_id": "SKU-INT", "warehouse_id": "WH-WEST", "risk_category": "HEALTHY", "mean_daily_forecast": 6.0},
        ])
        lead_times = pd.DataFrame([
            {"sku_id": "SKU-INT", "mean_lead_time_days": 10.0}
        ])

        analysis_bundle = InventoryAnalysisResult(
            positions=positions,
            safety_stocks=safety_stocks,
            risks=risks,
            lead_times=lead_times,
            summary={"test": True},
        )

        service = WarehouseRebalancingService()
        result = service.generate(inventory_analysis=analysis_bundle, as_of_date="2026-03-30")
        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        assert rec.source_warehouse_id == "WH-WEST"
        assert rec.destination_warehouse_id == "WH-EAST"
        assert rec.transfer_quantity == 100.0  # target 120 - 20 = 100
        assert rec.priority == "HIGH"

        # Verify DataFrame export
        df = result.to_dataframe()
        assert not df.empty
        assert "recommendation_id" in df.columns
        assert "transfer_quantity" in df.columns
