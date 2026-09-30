"""Deterministic Replenishment Solver (Phase 4B-1).

Calculates deterministic, explainable replenishment recommendations for every
SKU-warehouse combination by consuming Phase 4A inventory intelligence outputs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from commerce_ai.inventory.service import InventoryAnalysisResult
from commerce_ai.recommendations.schemas import (
    ReplenishmentConfig,
    ReplenishmentConstraint,
    ReplenishmentRecommendation,
    ReplenishmentResult,
    UrgencyLevel,
)


def determine_urgency(
    net_position: int,
    reorder_point: float,
    forecast_daily_demand: float,
    lead_time_days: float,
    risk_category: str,
    days_to_runout: Optional[float] = None,
    min_demand_threshold: float = 0.01,
) -> str:
    """Classify replenishment urgency into deterministic tiers.

    Rules:
    - CRITICAL: Active demand and (net position <= 0 OR risk_category == 'CRITICAL_STOCKOUT')
    - HIGH: net position <= reorder_point AND projected runout occurs before/equal to lead-time coverage
    - MEDIUM: net position <= reorder_point (standard replenishment candidate)
    - LOW: net position > reorder_point OR dormant demand
    """
    is_active = forecast_daily_demand > min_demand_threshold

    # 1. Critical stockout or physical exhaustion with active demand
    if is_active and (net_position <= 0 or risk_category == "CRITICAL_STOCKOUT"):
        return UrgencyLevel.CRITICAL.value

    # 2. High urgency: Runout arrives before normal supplier lead-time delivery
    if net_position <= reorder_point and is_active:
        if days_to_runout is not None and days_to_runout <= max(1.0, lead_time_days):
            return UrgencyLevel.HIGH.value
        return UrgencyLevel.MEDIUM.value

    # 3. Low urgency: Healthy stock, overstock, or dormant items
    return UrgencyLevel.LOW.value


def generate_deterministic_rationale(
    sku_id: str,
    warehouse_id: str,
    net_position: int,
    reorder_point: float,
    target_stock_level: Optional[float],
    forecast_daily_demand: float,
    lead_time_days: float,
    required_qty: int,
    recommended_order_qty: int,
    constraints_applied: List[str],
    urgency: str,
    risk_category: str,
    status: str,
    moq: Optional[int] = None,
    pack_size: Optional[int] = None,
    days_to_runout: Optional[float] = None,
) -> str:
    """Build an explainable, deterministic natural-language rationale without LLM generation."""
    # 1. Dormant or dead-stock items
    if status == "DORMANT_SUPPRESSED":
        return (
            f"Item {sku_id} at warehouse {warehouse_id} has dormant forecast demand "
            f"({forecast_daily_demand:.2f} units/day) with risk category '{risk_category}'. "
            f"Replenishment is suppressed to avoid dead stock and working capital lockup."
        )

    # 2. Healthy stock above ROP
    if status == "HEALTHY_ABOVE_ROP":
        return (
            f"Net inventory position ({net_position} units) exceeds reorder point ({reorder_point:.1f} units). "
            f"Daily demand is {forecast_daily_demand:.2f} units/day over {lead_time_days:.1f} days lead time. "
            f"No replenishment order required."
        )

    # 3. Target stock missing / insufficient policy inputs
    if status == "INSUFFICIENT_POLICY_INPUTS":
        return (
            f"Net inventory position ({net_position} units) is at or below reorder point ({reorder_point:.1f} units), "
            f"but target stock level is unavailable. Insufficient policy inputs to compute order quantity."
        )

    # 4. Active replenishment recommendation
    parts = [
        f"Net inventory position is {net_position} units, at or below reorder point of {reorder_point:.1f} units.",
        f"Forecast demand is {forecast_daily_demand:.2f} units/day with supplier lead time of {lead_time_days:.1f} days.",
    ]

    if target_stock_level is not None:
        parts.append(f"Target stock level is {target_stock_level:.1f} units (unconstrained required: {required_qty} units).")

    # Describe adjustments made
    adjustment_notes = []
    if ReplenishmentConstraint.MOQ_APPLIED.value in constraints_applied and moq is not None:
        adjustment_notes.append(f"supplier MOQ of {moq} units")
    if ReplenishmentConstraint.PACK_SIZE_ROUNDED.value in constraints_applied and pack_size is not None:
        adjustment_notes.append(f"pack size multiple of {pack_size} units")
    if ReplenishmentConstraint.MAX_ORDER_QTY_CAPPED.value in constraints_applied:
        adjustment_notes.append("maximum order quantity cap")
    if ReplenishmentConstraint.MAX_INVENTORY_DAYS_CAPPED.value in constraints_applied:
        adjustment_notes.append("maximum inventory days cap")
    if ReplenishmentConstraint.MAX_INVENTORY_VALUE_CAPPED.value in constraints_applied:
        adjustment_notes.append("maximum inventory value cap")

    if adjustment_notes:
        parts.append(
            f"Recommended order quantity is {recommended_order_qty} units after applying {', '.join(adjustment_notes)}."
        )
    else:
        parts.append(f"Recommended order quantity is {recommended_order_qty} units to restore target coverage.")

    # Urgency caveat
    if urgency == UrgencyLevel.CRITICAL.value:
        parts.append("Urgency is CRITICAL due to active stockout hazard or depleted stock.")
    elif urgency == UrgencyLevel.HIGH.value:
        runout_text = f"in {days_to_runout:.1f} days" if days_to_runout is not None else "imminently"
        parts.append(f"Urgency is HIGH with projected stockout {runout_text}, before normal lead time ({lead_time_days:.1f} days).")

    return " ".join(parts)


def solve_entity_replenishment(
    sku_id: str,
    warehouse_id: str,
    on_hand: int,
    reserved: int = 0,
    on_order: int = 0,
    net_inventory_position: Optional[int] = None,
    reorder_point: float = 0.0,
    target_stock_level: Optional[float] = None,
    risk_category: str = "HEALTHY",
    forecast_daily_demand: float = 0.0,
    lead_time_days: float = 14.0,
    supplier_id: Optional[str] = None,
    minimum_order_quantity: Optional[int] = None,
    pack_size: Optional[int] = None,
    unit_cost: Optional[float] = None,
    projected_runout_date: Optional[str] = None,
    days_to_runout: Optional[float] = None,
    eoq: Optional[float] = None,
    config: Optional[ReplenishmentConfig] = None,
) -> ReplenishmentRecommendation:
    """Solve replenishment requirements for a single SKU × warehouse entity.

    Applies deterministic trigger evaluation, target stock deficit sizing,
    MOQ enforcement, pack-size rounding, and maximum safety caps.
    """
    cfg = config or ReplenishmentConfig()

    # Calculate or normalize net position
    net_pos = (
        int(on_hand + on_order - reserved)
        if net_inventory_position is None
        else int(net_inventory_position)
    )

    # 1. Dormant Demand & Dead Stock Suppression
    is_dormant = (forecast_daily_demand <= cfg.min_demand_threshold) or (risk_category == "DEAD_STOCK")
    if is_dormant and not cfg.allow_zero_stock_dormant_reorder:
        rationale = generate_deterministic_rationale(
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            net_position=net_pos,
            reorder_point=reorder_point,
            target_stock_level=target_stock_level,
            forecast_daily_demand=forecast_daily_demand,
            lead_time_days=lead_time_days,
            required_qty=0,
            recommended_order_qty=0,
            constraints_applied=[],
            urgency=UrgencyLevel.LOW.value,
            risk_category=risk_category,
            status="DORMANT_SUPPRESSED",
        )
        return ReplenishmentRecommendation(
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            supplier_id=supplier_id,
            recommendation_required=False,
            urgency=UrgencyLevel.LOW.value,
            risk_category=risk_category,
            on_hand=on_hand,
            reserved=reserved,
            on_order=on_order,
            net_inventory_position=net_pos,
            forecast_daily_demand=forecast_daily_demand,
            lead_time_days=lead_time_days,
            reorder_point=reorder_point,
            target_stock_level=target_stock_level,
            shortfall_qty=0,
            required_qty=0,
            minimum_order_quantity=minimum_order_quantity,
            pack_size=pack_size,
            recommended_order_qty=0,
            unit_cost=unit_cost,
            estimated_order_cost=0.0 if unit_cost is not None else None,
            projected_runout_date=projected_runout_date,
            rationale=rationale,
            constraints_applied=[],
            status="DORMANT_SUPPRESSED",
        )

    # 2. Shortfall Calculation (Diagnostic metric)
    shortfall_qty = max(0, int(math.ceil(reorder_point - net_pos)))

    # 3. Replenishment Trigger Evaluation
    if net_pos > reorder_point:
        rationale = generate_deterministic_rationale(
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            net_position=net_pos,
            reorder_point=reorder_point,
            target_stock_level=target_stock_level,
            forecast_daily_demand=forecast_daily_demand,
            lead_time_days=lead_time_days,
            required_qty=0,
            recommended_order_qty=0,
            constraints_applied=[],
            urgency=UrgencyLevel.LOW.value,
            risk_category=risk_category,
            status="HEALTHY_ABOVE_ROP",
        )
        return ReplenishmentRecommendation(
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            supplier_id=supplier_id,
            recommendation_required=False,
            urgency=UrgencyLevel.LOW.value,
            risk_category=risk_category,
            on_hand=on_hand,
            reserved=reserved,
            on_order=on_order,
            net_inventory_position=net_pos,
            forecast_daily_demand=forecast_daily_demand,
            lead_time_days=lead_time_days,
            reorder_point=reorder_point,
            target_stock_level=target_stock_level,
            shortfall_qty=0,
            required_qty=0,
            minimum_order_quantity=minimum_order_quantity,
            pack_size=pack_size,
            recommended_order_qty=0,
            unit_cost=unit_cost,
            estimated_order_cost=0.0 if unit_cost is not None else None,
            projected_runout_date=projected_runout_date,
            rationale=rationale,
            constraints_applied=[],
            status="HEALTHY_ABOVE_ROP",
        )

    # 4. Triggered Candidate (net_pos <= reorder_point)
    urgency = determine_urgency(
        net_position=net_pos,
        reorder_point=reorder_point,
        forecast_daily_demand=forecast_daily_demand,
        lead_time_days=lead_time_days,
        risk_category=risk_category,
        days_to_runout=days_to_runout,
        min_demand_threshold=cfg.min_demand_threshold,
    )

    # Check target stock availability
    if target_stock_level is None:
        rationale = generate_deterministic_rationale(
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            net_position=net_pos,
            reorder_point=reorder_point,
            target_stock_level=None,
            forecast_daily_demand=forecast_daily_demand,
            lead_time_days=lead_time_days,
            required_qty=0,
            recommended_order_qty=0,
            constraints_applied=[],
            urgency=urgency,
            risk_category=risk_category,
            status="INSUFFICIENT_POLICY_INPUTS",
        )
        return ReplenishmentRecommendation(
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            supplier_id=supplier_id,
            recommendation_required=True,
            urgency=urgency,
            risk_category=risk_category,
            on_hand=on_hand,
            reserved=reserved,
            on_order=on_order,
            net_inventory_position=net_pos,
            forecast_daily_demand=forecast_daily_demand,
            lead_time_days=lead_time_days,
            reorder_point=reorder_point,
            target_stock_level=None,
            shortfall_qty=shortfall_qty,
            required_qty=0,
            minimum_order_quantity=minimum_order_quantity,
            pack_size=pack_size,
            recommended_order_qty=0,
            unit_cost=unit_cost,
            estimated_order_cost=None,
            projected_runout_date=projected_runout_date,
            rationale=rationale,
            constraints_applied=[],
            status="INSUFFICIENT_POLICY_INPUTS",
        )

    # Base required quantity calculation
    required_qty = max(0, int(math.ceil(target_stock_level - net_pos)))
    if required_qty <= 0:
        rationale = (
            f"Net inventory position ({net_pos} units) is at or below reorder point ({reorder_point:.1f} units), "
            f"but already covers target stock level ({target_stock_level:.1f} units). No replenishment order required."
        )
        return ReplenishmentRecommendation(
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            supplier_id=supplier_id,
            recommendation_required=False,
            urgency=UrgencyLevel.LOW.value,
            risk_category=risk_category,
            on_hand=on_hand,
            reserved=reserved,
            on_order=on_order,
            net_inventory_position=net_pos,
            forecast_daily_demand=forecast_daily_demand,
            lead_time_days=lead_time_days,
            reorder_point=reorder_point,
            target_stock_level=target_stock_level,
            shortfall_qty=shortfall_qty,
            required_qty=0,
            minimum_order_quantity=minimum_order_quantity,
            pack_size=pack_size,
            recommended_order_qty=0,
            unit_cost=unit_cost,
            estimated_order_cost=0.0 if unit_cost is not None else None,
            projected_runout_date=projected_runout_date,
            rationale=rationale,
            constraints_applied=[],
            status="COMPLETED",
        )

    # Optional EOQ override if explicitly enabled and valid
    order_qty = required_qty
    if cfg.use_eoq and eoq is not None and eoq > 0:
        eoq_int = int(math.ceil(eoq))
        order_qty = max(required_qty, eoq_int)

    constraints_applied: List[str] = []

    # 5. MOQ Enforcement
    eff_moq = cfg.sku_moq_overrides.get(sku_id, minimum_order_quantity)
    if eff_moq is not None and eff_moq > 1:
        if order_qty < eff_moq:
            order_qty = eff_moq
            constraints_applied.append(ReplenishmentConstraint.MOQ_APPLIED.value)

    # 6. Pack / Lot Size Rounding
    eff_pack_size = cfg.sku_pack_sizes.get(sku_id, pack_size or cfg.default_pack_size)
    if eff_pack_size is not None and eff_pack_size > 1:
        remainder = order_qty % eff_pack_size
        if remainder != 0:
            order_qty = int(math.ceil(order_qty / eff_pack_size) * eff_pack_size)
            constraints_applied.append(ReplenishmentConstraint.PACK_SIZE_ROUNDED.value)

    # 7. Maximum Order & Safety Guard Capping
    # A. Maximum order quantity cap
    if cfg.maximum_order_quantity is not None and cfg.maximum_order_quantity > 0:
        if order_qty > cfg.maximum_order_quantity:
            order_qty = cfg.maximum_order_quantity
            constraints_applied.append(ReplenishmentConstraint.MAX_ORDER_QTY_CAPPED.value)

    # B. Maximum inventory days cap
    if cfg.maximum_inventory_days is not None and cfg.maximum_inventory_days > 0.0 and forecast_daily_demand > 0.0:
        max_inv_units = int(math.floor(cfg.maximum_inventory_days * forecast_daily_demand))
        max_allowed_order = max(0, max_inv_units - net_pos)
        if order_qty > max_allowed_order:
            order_qty = max_allowed_order
            constraints_applied.append(ReplenishmentConstraint.MAX_INVENTORY_DAYS_CAPPED.value)

    # C. Maximum inventory value cap
    if cfg.maximum_inventory_value is not None and cfg.maximum_inventory_value > 0.0 and unit_cost is not None and unit_cost > 0.0:
        max_val_units = int(math.floor(cfg.maximum_inventory_value / unit_cost))
        max_allowed_by_value = max(0, max_val_units - net_pos)
        if order_qty > max_allowed_by_value:
            order_qty = max_allowed_by_value
            constraints_applied.append(ReplenishmentConstraint.MAX_INVENTORY_VALUE_CAPPED.value)

    final_recommended_qty = max(0, int(order_qty))

    # 8. Financial Calculation
    if unit_cost is not None and unit_cost >= 0.0:
        estimated_cost = round(float(final_recommended_qty * unit_cost), 2)
    else:
        estimated_cost = None

    # 9. Rationale Generation
    rationale = generate_deterministic_rationale(
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        net_position=net_pos,
        reorder_point=reorder_point,
        target_stock_level=target_stock_level,
        forecast_daily_demand=forecast_daily_demand,
        lead_time_days=lead_time_days,
        required_qty=required_qty,
        recommended_order_qty=final_recommended_qty,
        constraints_applied=constraints_applied,
        urgency=urgency,
        risk_category=risk_category,
        status="COMPLETED",
        moq=eff_moq,
        pack_size=eff_pack_size,
        days_to_runout=days_to_runout,
    )

    return ReplenishmentRecommendation(
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        supplier_id=supplier_id,
        recommendation_required=True,
        urgency=urgency,
        risk_category=risk_category,
        on_hand=on_hand,
        reserved=reserved,
        on_order=on_order,
        net_inventory_position=net_pos,
        forecast_daily_demand=forecast_daily_demand,
        lead_time_days=lead_time_days,
        reorder_point=reorder_point,
        target_stock_level=target_stock_level,
        shortfall_qty=shortfall_qty,
        required_qty=required_qty,
        minimum_order_quantity=eff_moq,
        pack_size=eff_pack_size,
        recommended_order_qty=final_recommended_qty,
        unit_cost=unit_cost,
        estimated_order_cost=estimated_cost,
        projected_runout_date=projected_runout_date,
        rationale=rationale,
        constraints_applied=constraints_applied,
        status="COMPLETED",
    )


class ReplenishmentSolver:
    """High-level service orchestrating portfolio-wide replenishment determination.

    Consumes Phase 4A outputs without recalculating positions, forecasts, or risks.
    """

    def __init__(self, config: Optional[ReplenishmentConfig] = None):
        self.config = config or ReplenishmentConfig()

    def solve(
        self,
        inventory_analysis: Optional[InventoryAnalysisResult] = None,
        positions: Optional[pd.DataFrame] = None,
        safety_stocks: Optional[pd.DataFrame] = None,
        risks: Optional[pd.DataFrame] = None,
        lead_times: Optional[pd.DataFrame] = None,
        products: Optional[pd.DataFrame] = None,
        suppliers: Optional[pd.DataFrame] = None,
        pack_sizes: Optional[Dict[str, int]] = None,
    ) -> ReplenishmentResult:
        """Process portfolio inventory intelligence outputs and generate recommendations.

        Args:
            inventory_analysis: Consolidated result bundle from InventoryService.
            positions: Optional DataFrame of inventory positions (if inventory_analysis omitted).
            safety_stocks: Optional DataFrame of safety stock & ROP (if inventory_analysis omitted).
            risks: Optional DataFrame of inventory risks (if inventory_analysis omitted).
            lead_times: Optional DataFrame of lead-time profiles (if inventory_analysis omitted).
            products: Optional product master (for unit_cost, preferred_supplier_id, pack_size).
            suppliers: Optional supplier master (for minimum_order_quantity).
            pack_sizes: Optional dictionary mapping sku_id -> pack_size integer.

        Returns:
            ReplenishmentResult containing all recommendations and portfolio summary metrics.
        """
        # 1. Resolve source DataFrames
        pos_df = inventory_analysis.positions if inventory_analysis is not None else (positions if positions is not None else pd.DataFrame())
        ss_df = inventory_analysis.safety_stocks if inventory_analysis is not None else (safety_stocks if safety_stocks is not None else pd.DataFrame())
        risk_df = inventory_analysis.risks if inventory_analysis is not None else (risks if risks is not None else pd.DataFrame())
        lt_df = inventory_analysis.lead_times if inventory_analysis is not None else (lead_times if lead_times is not None else pd.DataFrame())

        if pos_df.empty:
            return ReplenishmentResult(recommendations=[], summary={"total_entities_evaluated": 0})

        # 2. Build metadata lookup tables
        supplier_moqs: Dict[str, int] = {}
        if suppliers is not None and not suppliers.empty:
            if "supplier_id" in suppliers.columns and "minimum_order_quantity" in suppliers.columns:
                supplier_moqs = suppliers.set_index("supplier_id")["minimum_order_quantity"].dropna().to_dict()

        product_suppliers: Dict[str, str] = {}
        product_costs: Dict[str, float] = {}
        product_packs: Dict[str, int] = {}
        if products is not None and not products.empty:
            if "sku_id" in products.columns and "preferred_supplier_id" in products.columns:
                product_suppliers = products.set_index("sku_id")["preferred_supplier_id"].dropna().to_dict()
            if "sku_id" in products.columns and "unit_cost" in products.columns:
                product_costs = products.set_index("sku_id")["unit_cost"].dropna().to_dict()
            if "sku_id" in products.columns and "pack_size" in products.columns:
                product_packs = products.set_index("sku_id")["pack_size"].dropna().to_dict()

        # Merge manual pack sizes
        if pack_sizes:
            product_packs.update(pack_sizes)

        # Build indexed maps for Phase 4A tables
        ss_map: Dict[tuple, Dict[str, Any]] = {}
        if not ss_df.empty and "sku_id" in ss_df.columns and "warehouse_id" in ss_df.columns:
            for _, row in ss_df.iterrows():
                ss_map[(str(row["sku_id"]), str(row["warehouse_id"]))] = row.to_dict()

        risk_map: Dict[tuple, Dict[str, Any]] = {}
        if not risk_df.empty and "sku_id" in risk_df.columns and "warehouse_id" in risk_df.columns:
            for _, row in risk_df.iterrows():
                risk_map[(str(row["sku_id"]), str(row["warehouse_id"]))] = row.to_dict()

        lt_map: Dict[str, Dict[str, Any]] = {}
        if not lt_df.empty and "sku_id" in lt_df.columns:
            for _, row in lt_df.iterrows():
                lt_map[str(row["sku_id"])] = row.to_dict()

        # 3. Solve for every position entity
        recommendations: List[ReplenishmentRecommendation] = []
        for _, pos_row in pos_df.iterrows():
            s_id = str(pos_row["sku_id"])
            w_id = str(pos_row["warehouse_id"])
            entity_key = (s_id, w_id)

            on_hand = int(pos_row.get("on_hand", 0))
            reserved = int(pos_row.get("reserved", 0))
            on_order = int(pos_row.get("on_order", 0))
            net_pos = int(pos_row.get("net_position", on_hand + on_order - reserved))

            # Retrieve Phase 4A metrics
            ss_data = ss_map.get(entity_key, {})
            risk_data = risk_map.get(entity_key, {})
            lt_data = lt_map.get(s_id, {})

            rop = float(ss_data.get("reorder_point", 0.0))
            target_stock = ss_data.get("target_stock_level")
            target_stock = float(target_stock) if target_stock is not None and not pd.isna(target_stock) else None
            eoq_val = ss_data.get("eoq")
            eoq_val = float(eoq_val) if eoq_val is not None and not pd.isna(eoq_val) else None

            # Daily demand & lead time
            fc_demand = float(
                risk_data.get("mean_daily_forecast", ss_data.get("daily_demand_mean", 0.0))
            )
            lt_days = float(
                ss_data.get("lead_time_days", lt_data.get("mean_lead_time_days", 14.0))
            )
            risk_cat = str(risk_data.get("risk_category", "HEALTHY"))
            runout_dt = risk_data.get("runout_date")
            runout_dt = str(runout_dt) if runout_dt is not None and not pd.isna(runout_dt) else None
            days_to_runout = risk_data.get("days_to_runout")
            days_to_runout = float(days_to_runout) if days_to_runout is not None and not pd.isna(days_to_runout) else None

            # Supplier & master lookups
            supp_id = str(lt_data.get("supplier_id") or product_suppliers.get(s_id) or "")
            supp_id = supp_id if supp_id else None
            moq = int(supplier_moqs.get(supp_id, 1)) if supp_id else None
            cost = product_costs.get(s_id)
            cost = float(cost) if cost is not None and not pd.isna(cost) else None
            p_size = product_packs.get(s_id)

            rec = solve_entity_replenishment(
                sku_id=s_id,
                warehouse_id=w_id,
                on_hand=on_hand,
                reserved=reserved,
                on_order=on_order,
                net_inventory_position=net_pos,
                reorder_point=rop,
                target_stock_level=target_stock,
                risk_category=risk_cat,
                forecast_daily_demand=fc_demand,
                lead_time_days=lt_days,
                supplier_id=supp_id,
                minimum_order_quantity=moq,
                pack_size=p_size,
                unit_cost=cost,
                projected_runout_date=runout_dt,
                days_to_runout=days_to_runout,
                eoq=eoq_val,
                config=self.config,
            )
            recommendations.append(rec)

        # 4. Compute Portfolio Summary Metrics
        actionable = [r for r in recommendations if r.recommendation_required and r.recommended_order_qty > 0]
        non_rec = [r for r in recommendations if not r.recommendation_required]
        dormant = [r for r in recommendations if r.status == "DORMANT_SUPPRESSED"]

        total_units = sum(r.recommended_order_qty for r in actionable)
        total_spend = sum(r.estimated_order_cost for r in actionable if r.estimated_order_cost is not None)

        urgency_counts = {
            UrgencyLevel.CRITICAL.value: sum(1 for r in actionable if r.urgency == UrgencyLevel.CRITICAL.value),
            UrgencyLevel.HIGH.value: sum(1 for r in actionable if r.urgency == UrgencyLevel.HIGH.value),
            UrgencyLevel.MEDIUM.value: sum(1 for r in actionable if r.urgency == UrgencyLevel.MEDIUM.value),
            UrgencyLevel.LOW.value: sum(1 for r in recommendations if r.urgency == UrgencyLevel.LOW.value),
        }

        constraints_applied_counts: Dict[str, int] = {}
        for r in actionable:
            for c in r.constraints_applied:
                constraints_applied_counts[c] = constraints_applied_counts.get(c, 0) + 1

        summary = {
            "total_entities_evaluated": len(recommendations),
            "replenishment_recommended_count": len(actionable),
            "non_recommended_count": len(non_rec),
            "dormant_excluded_count": len(dormant),
            "total_recommended_units": total_units,
            "total_estimated_spend": round(total_spend, 2),
            "urgency_breakdown": urgency_counts,
            "constraints_applied_counts": constraints_applied_counts,
        }

        return ReplenishmentResult(recommendations=recommendations, summary=summary)
