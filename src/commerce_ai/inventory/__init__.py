"""Inventory Intelligence Engine (Phase 4A).

Provides:
- Inventory position accounting (on-hand, on-order, reserved, net position)
- Empirical supplier and SKU lead-time profiling with fallback guarantees
- Deterministic dual-uncertainty safety stock and reorder point (ROP) calculation
- Forward-looking depletion trajectory and inventory risk assessment (DOS, runout date)
- Unified InventoryService orchestrating the end-to-end intelligence workflow
"""

from commerce_ai.inventory.schemas import (
    InventoryPositionRecord,
    LeadTimeMetrics,
    SafetyStockResult,
    InventoryRiskRecord,
)
from commerce_ai.inventory.position import (
    calculate_inventory_positions,
    extract_position_records,
)
from commerce_ai.inventory.lead_time import (
    LeadTimeConfig,
    calculate_po_lead_times,
    profile_supplier_lead_times,
    profile_sku_lead_times,
)
from commerce_ai.inventory.safety_stock import (
    ServiceLevelPolicy,
    DEFAULT_ABC_XYZ_SERVICE_LEVELS,
    service_level_to_z_score,
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_optional_eoq,
    calculate_safety_stock_result,
)
from commerce_ai.inventory.risk import (
    RiskThresholdConfig,
    simulate_depletion_curve,
    compute_stockout_hazard_score,
    assess_sku_inventory_risk,
)
from commerce_ai.inventory.service import (
    InventoryAnalysisResult,
    InventoryService,
)

__all__ = [
    "InventoryPositionRecord",
    "LeadTimeMetrics",
    "SafetyStockResult",
    "InventoryRiskRecord",
    "calculate_inventory_positions",
    "extract_position_records",
    "LeadTimeConfig",
    "calculate_po_lead_times",
    "profile_supplier_lead_times",
    "profile_sku_lead_times",
    "ServiceLevelPolicy",
    "DEFAULT_ABC_XYZ_SERVICE_LEVELS",
    "service_level_to_z_score",
    "calculate_safety_stock",
    "calculate_reorder_point",
    "calculate_optional_eoq",
    "calculate_safety_stock_result",
    "RiskThresholdConfig",
    "simulate_depletion_curve",
    "compute_stockout_hazard_score",
    "assess_sku_inventory_risk",
    "InventoryAnalysisResult",
    "InventoryService",
]
