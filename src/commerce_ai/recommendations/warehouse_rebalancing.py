"""Warehouse Rebalancing Engine (Phase 4C).

Identifies inter-warehouse inventory rebalancing transfer recommendations to satisfy
shortages from network surplus without purchasing additional stock.
"""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from commerce_ai.inventory.service import InventoryAnalysisResult
from commerce_ai.recommendations.schemas import (
    CostDataStatus,
    ProposalStatus,
    TransferConstraint,
    TransferPriority,
    WarehouseRebalancingConfig,
    WarehouseRebalancingResult,
    WarehouseTransferRecommendation,
)

# Relative ranking for deterministic destination priority and competing need allocation
TRANSFER_PRIORITY_PRECEDENCE = {
    TransferPriority.CRITICAL.value: 4,
    TransferPriority.HIGH.value: 3,
    TransferPriority.MEDIUM.value: 2,
    TransferPriority.LOW.value: 1,
}


def determine_transfer_priority(
    net_position: float,
    reorder_point: float,
    risk_category: str,
    days_to_runout: Optional[float] = None,
    transfer_lead_time_days: Optional[float] = None,
) -> str:
    """Classify destination transfer priority into deterministic tiers.

    Rules:
    - CRITICAL: destination risk is CRITICAL_STOCKOUT OR destination net position <= 0
    - HIGH: destination is UNDERSTOCK (or projected runout is within transfer lead-time coverage)
    - MEDIUM: destination is below ROP
    - LOW: no transfer required / healthy
    """
    cat_upper = risk_category.strip().upper()

    if net_position <= 0 or cat_upper == "CRITICAL_STOCKOUT":
        return TransferPriority.CRITICAL.value

    if cat_upper == "UNDERSTOCK":
        return TransferPriority.HIGH.value

    if days_to_runout is not None and transfer_lead_time_days is not None:
        if days_to_runout <= transfer_lead_time_days:
            return TransferPriority.HIGH.value

    if net_position <= reorder_point:
        return TransferPriority.MEDIUM.value

    return TransferPriority.LOW.value


def generate_deterministic_transfer_id(
    sku_id: str,
    source_warehouse_id: str,
    destination_warehouse_id: str,
    transfer_quantity: float,
    as_of_date: Optional[Union[str, date, datetime]] = None,
) -> str:
    """Generate a reproducible, deterministic recommendation ID without random UUIDs.

    Derives a stable cryptographic SHA-256 digest over the SKU, source, destination,
    transfer quantity, and proposal reference date.
    """
    date_str = str(as_of_date) if as_of_date is not None else "NODATE"
    qty_str = f"{transfer_quantity:.2f}"
    seed = f"{sku_id}|{source_warehouse_id}|{destination_warehouse_id}|{qty_str}|{date_str}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8].upper()
    return f"XFER_{sku_id}_{source_warehouse_id}_{destination_warehouse_id}_{digest}"


def generate_transfer_rationale(
    sku_id: str,
    source_warehouse_id: str,
    destination_warehouse_id: str,
    destination_net_pos: float,
    destination_target: float,
    destination_risk: str,
    source_net_pos: float,
    source_rop: float,
    source_surplus: float,
    transfer_qty: float,
    constraints_applied: Optional[List[str]] = None,
) -> str:
    """Build an explainable, deterministic natural-language rationale without LLM generation."""
    rationale = (
        f"Warehouse {destination_warehouse_id} has a net inventory position of {destination_net_pos:.0f} units "
        f"against a target stock level of {destination_target:.0f} units and is classified as {destination_risk}. "
        f"Warehouse {source_warehouse_id} has {source_net_pos:.0f} units against a reorder point of {source_rop:.0f} units, "
        f"leaving {source_surplus:.0f} transferable units. A transfer of {transfer_qty:.0f} units is recommended "
        f"to cover {destination_warehouse_id}'s requirement while keeping {source_warehouse_id} at or above its reorder point."
    )
    if constraints_applied:
        rationale += f" Constraints applied: {', '.join(constraints_applied)}."
    return rationale


def lookup_route_distance(
    cfg: WarehouseRebalancingConfig,
    routes_map: Dict[Tuple[str, str], Dict[str, Any]],
    src: str,
    dest: str,
) -> Optional[float]:
    """Retrieve route distance in km from route table or configuration map."""
    key = (src, dest)
    if key in routes_map and routes_map[key].get("distance_km") is not None:
        return float(routes_map[key]["distance_km"])
    if key in cfg.warehouse_distances_km:
        return float(cfg.warehouse_distances_km[key])
    str_key = f"{src}:{dest}"
    if str_key in cfg.warehouse_distances_km:
        return float(cfg.warehouse_distances_km[str_key])
    return None


def lookup_route_lead_time(
    cfg: WarehouseRebalancingConfig,
    routes_map: Dict[Tuple[str, str], Dict[str, Any]],
    src: str,
    dest: str,
) -> Optional[float]:
    """Retrieve route transit lead time in days from route table or configuration map."""
    key = (src, dest)
    if key in routes_map and routes_map[key].get("transfer_lead_time_days") is not None:
        return float(routes_map[key]["transfer_lead_time_days"])
    if key in cfg.warehouse_transfer_lead_times:
        return float(cfg.warehouse_transfer_lead_times[key])
    str_key = f"{src}:{dest}"
    if str_key in cfg.warehouse_transfer_lead_times:
        return float(cfg.warehouse_transfer_lead_times[str_key])
    return None


def lookup_route_cost(
    cfg: WarehouseRebalancingConfig,
    routes_map: Dict[Tuple[str, str], Dict[str, Any]],
    src: str,
    dest: str,
) -> Optional[float]:
    """Retrieve per-unit transfer cost from route table or configuration map."""
    key = (src, dest)
    if key in routes_map and routes_map[key].get("transfer_cost_per_unit") is not None:
        return float(routes_map[key]["transfer_cost_per_unit"])
    if key in cfg.transfer_costs_per_unit:
        return float(cfg.transfer_costs_per_unit[key])
    str_key = f"{src}:{dest}"
    if str_key in cfg.transfer_costs_per_unit:
        return float(cfg.transfer_costs_per_unit[str_key])
    if cfg.default_transfer_cost_per_unit is not None:
        return float(cfg.default_transfer_cost_per_unit)
    return None


class WarehouseRebalancingService:
    """Deterministic, explainable warehouse rebalancing engine."""

    def __init__(self, config: Optional[WarehouseRebalancingConfig] = None):
        self.config = config or WarehouseRebalancingConfig()

    def rebalance(
        self,
        inventory_analysis: Optional[InventoryAnalysisResult] = None,
        positions: Optional[pd.DataFrame] = None,
        safety_stocks: Optional[pd.DataFrame] = None,
        risks: Optional[pd.DataFrame] = None,
        routes: Optional[pd.DataFrame] = None,
        target_stock_overrides: Optional[Dict[Tuple[str, str], float]] = None,
        reorder_point_overrides: Optional[Dict[Tuple[str, str], float]] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[WarehouseRebalancingConfig] = None,
    ) -> WarehouseRebalancingResult:
        """Alias for generate()."""
        return self.generate(
            inventory_analysis=inventory_analysis,
            positions=positions,
            safety_stocks=safety_stocks,
            risks=risks,
            routes=routes,
            target_stock_overrides=target_stock_overrides,
            reorder_point_overrides=reorder_point_overrides,
            as_of_date=as_of_date,
            config=config,
        )

    def generate(
        self,
        inventory_analysis: Optional[InventoryAnalysisResult] = None,
        positions: Optional[pd.DataFrame] = None,
        safety_stocks: Optional[pd.DataFrame] = None,
        risks: Optional[pd.DataFrame] = None,
        routes: Optional[pd.DataFrame] = None,
        target_stock_overrides: Optional[Dict[Tuple[str, str], float]] = None,
        reorder_point_overrides: Optional[Dict[Tuple[str, str], float]] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[WarehouseRebalancingConfig] = None,
    ) -> WarehouseRebalancingResult:
        """Process portfolio inventory intelligence outputs and generate transfer recommendations.

        Args:
            inventory_analysis: Consolidated result bundle from Phase 4A InventoryService.
            positions: Optional DataFrame of inventory positions (if inventory_analysis omitted).
            safety_stocks: Optional DataFrame of safety stock & ROP (if inventory_analysis omitted).
            risks: Optional DataFrame of inventory risks (if inventory_analysis omitted).
            routes: Optional DataFrame defining transfer routes, distances, transit days, and costs.
            target_stock_overrides: Optional mapping (sku_id, warehouse_id) -> target_stock_level.
            reorder_point_overrides: Optional mapping (sku_id, warehouse_id) -> reorder_point.
            as_of_date: Optional reference evaluation date for deterministic ID generation.
            config: Optional WarehouseRebalancingConfig overriding instance configuration.

        Returns:
            WarehouseRebalancingResult containing recommendations, exclusions, unmet demand, and audit logs.
        """
        cfg = config or self.config

        # 1. Resolve source DataFrames
        pos_df = (
            inventory_analysis.positions
            if inventory_analysis is not None
            else (positions if positions is not None else pd.DataFrame())
        )
        ss_df = (
            inventory_analysis.safety_stocks
            if inventory_analysis is not None
            else (safety_stocks if safety_stocks is not None else pd.DataFrame())
        )
        risk_df = (
            inventory_analysis.risks
            if inventory_analysis is not None
            else (risks if risks is not None else pd.DataFrame())
        )

        if pos_df.empty:
            return WarehouseRebalancingResult(
                recommendations=[],
                excluded_dormant=[],
                insufficient_policy_inputs=[],
                unmet_destination_demand=[],
                source_inventory_remaining={},
                total_transfer_quantity=0.0,
                recommendation_count=0,
                estimated_total_transfer_cost=None,
                cost_data_status=CostDataStatus.NO_COST_DATA.value,
                summary={"sku_count_evaluated": 0, "total_entities_evaluated": 0},
            )

        # 2. Build lookup maps for Phase 4A metrics and route parameters
        ss_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if not ss_df.empty and "sku_id" in ss_df.columns and "warehouse_id" in ss_df.columns:
            for _, row in ss_df.iterrows():
                ss_map[(str(row["sku_id"]), str(row["warehouse_id"]))] = row.to_dict()

        risk_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if not risk_df.empty and "sku_id" in risk_df.columns and "warehouse_id" in risk_df.columns:
            for _, row in risk_df.iterrows():
                risk_map[(str(row["sku_id"]), str(row["warehouse_id"]))] = row.to_dict()

        routes_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if routes is not None and not routes.empty:
            for _, rrow in routes.iterrows():
                if "source_warehouse_id" in rrow and "destination_warehouse_id" in rrow:
                    s_w = str(rrow["source_warehouse_id"])
                    d_w = str(rrow["destination_warehouse_id"])
                    routes_map[(s_w, d_w)] = rrow.to_dict()

        # 3. Group inventory positions by SKU
        recommendations: List[WarehouseTransferRecommendation] = []
        excluded_dormant: List[Dict[str, Any]] = []
        insufficient_policy_inputs: List[Dict[str, Any]] = []
        unmet_destination_demand: List[Dict[str, Any]] = []
        source_inventory_remaining: Dict[str, Dict[str, float]] = {}

        unique_skus = pos_df["sku_id"].astype(str).unique()

        for sku_id in unique_skus:
            sku_pos_df = pos_df[pos_df["sku_id"].astype(str) == sku_id]

            destination_candidates: List[Dict[str, Any]] = []
            source_candidates: List[Dict[str, Any]] = []
            source_inventory_remaining[sku_id] = {}

            # Parse each warehouse position for this SKU
            for _, pos_row in sku_pos_df.iterrows():
                w_id = str(pos_row["warehouse_id"])
                entity_key = (sku_id, w_id)

                on_hand = float(pos_row.get("on_hand", 0))
                reserved = float(pos_row.get("reserved", 0))
                on_order = float(pos_row.get("on_order", 0))
                net_pos = float(pos_row.get("net_position", on_hand + on_order - reserved))

                ss_data = ss_map.get(entity_key, {})
                risk_data = risk_map.get(entity_key, {})

                # Resolve ROP
                rop: Optional[float] = None
                if reorder_point_overrides and entity_key in reorder_point_overrides:
                    rop = float(reorder_point_overrides[entity_key])
                else:
                    rop_raw = ss_data.get("reorder_point", risk_data.get("reorder_point"))
                    if rop_raw is not None and not pd.isna(rop_raw):
                        rop = float(rop_raw)

                # Resolve Target Stock Level
                target_stock: Optional[float] = None
                if target_stock_overrides and entity_key in target_stock_overrides:
                    target_stock = float(target_stock_overrides[entity_key])
                else:
                    t_raw = ss_data.get("target_stock_level")
                    if t_raw is not None and not pd.isna(t_raw):
                        target_stock = float(t_raw)

                risk_cat = str(risk_data.get("risk_category", "HEALTHY")).strip().upper()
                fc_demand = float(
                    risk_data.get("mean_daily_forecast", ss_data.get("daily_demand_mean", 0.0))
                )
                runout_raw = risk_data.get("days_to_runout")
                days_to_runout = float(runout_raw) if runout_raw is not None and not pd.isna(runout_raw) else None

                # A. Destination Candidate Evaluation
                is_dormant = fc_demand <= cfg.min_demand_threshold
                is_dead_stock = risk_cat == "DEAD_STOCK"

                # Check if this warehouse has an inventory shortage/risk
                has_shortage = False
                if rop is not None and net_pos <= rop:
                    has_shortage = True
                elif risk_cat in ("CRITICAL_STOCKOUT", "UNDERSTOCK"):
                    has_shortage = True

                if has_shortage:
                    if is_dormant or is_dead_stock:
                        excluded_dormant.append({
                            "sku_id": sku_id,
                            "warehouse_id": w_id,
                            "reason": "DEAD_STOCK" if is_dead_stock else "DORMANT_DEMAND",
                            "forecast_daily_demand": fc_demand,
                            "risk_category": risk_cat,
                            "net_inventory_position": net_pos,
                        })
                    else:
                        # Active destination needing stock: verify policy inputs
                        if rop is None:
                            insufficient_policy_inputs.append({
                                "sku_id": sku_id,
                                "warehouse_id": w_id,
                                "reason": "MISSING_REORDER_POINT",
                                "net_inventory_position": net_pos,
                                "target_stock_level": target_stock,
                                "risk_category": risk_cat,
                            })
                        elif target_stock is None:
                            insufficient_policy_inputs.append({
                                "sku_id": sku_id,
                                "warehouse_id": w_id,
                                "reason": "MISSING_TARGET_STOCK",
                                "net_inventory_position": net_pos,
                                "reorder_point": rop,
                                "risk_category": risk_cat,
                            })
                        else:
                            # Requirement is based on target_stock_level - net_inventory_position
                            req_qty = max(0.0, float(target_stock - net_pos))
                            if req_qty > 0.0:
                                priority = determine_transfer_priority(
                                    net_position=net_pos,
                                    reorder_point=rop,
                                    risk_category=risk_cat,
                                    days_to_runout=days_to_runout,
                                )
                                destination_candidates.append({
                                    "sku_id": sku_id,
                                    "warehouse_id": w_id,
                                    "net_position": net_pos,
                                    "reorder_point": rop,
                                    "target_stock_level": target_stock,
                                    "risk_category": risk_cat,
                                    "required_qty": req_qty,
                                    "remaining_need": req_qty,
                                    "priority": priority,
                                    "forecast_daily_demand": fc_demand,
                                    "days_to_runout": days_to_runout,
                                })

                # B. Source Candidate Evaluation
                # A warehouse can become a source when it has transferable surplus above protected_stock (ROP)
                if rop is not None and net_pos > rop:
                    surplus = max(0.0, float(net_pos - rop))
                    if surplus > 0.0:
                        source_candidates.append({
                            "sku_id": sku_id,
                            "warehouse_id": w_id,
                            "net_position": net_pos,
                            "reorder_point": rop,
                            "target_stock_level": target_stock,
                            "transferable_surplus": surplus,
                            "remaining_surplus": surplus,
                        })
                        source_inventory_remaining[sku_id][w_id] = round(surplus, 2)

            # 4. Perform Matching and Allocation for this SKU
            if not destination_candidates:
                continue

            if not source_candidates:
                # Inventory required, but no warehouse in the network has surplus
                for d in destination_candidates:
                    unmet_destination_demand.append({
                        "sku_id": sku_id,
                        "destination_warehouse_id": d["warehouse_id"],
                        "unmet_quantity": round(d["required_qty"], 2),
                        "requested_quantity": round(d["required_qty"], 2),
                        "priority": d["priority"],
                        "risk_category": d["risk_category"],
                    })
                continue

            # Sort Destinations:
            # 1. Priority (CRITICAL > HIGH > MEDIUM > LOW)
            # 2. Deficit / required quantity (descending)
            # 3. Deterministic warehouse_id (ascending)
            destination_candidates.sort(
                key=lambda d: (
                    -TRANSFER_PRIORITY_PRECEDENCE.get(d["priority"], 1),
                    -d["required_qty"],
                    d["warehouse_id"],
                )
            )

            # Allocate surplus to destinations
            for dest in destination_candidates:
                dest_wh = dest["warehouse_id"]
                used_sources_for_dest: set[str] = set()

                while dest["remaining_need"] > 0:
                    avail_sources = [
                        s for s in source_candidates
                        if s["warehouse_id"] != dest_wh
                        and s["warehouse_id"] not in used_sources_for_dest
                        and s["remaining_surplus"] > 0
                    ]
                    if not avail_sources:
                        break

                    # Sort available sources:
                    # 1. Highest remaining transferable surplus (descending)
                    # 2. Shortest transfer distance if available (ascending; missing distance treated as infinity)
                    # 3. Deterministic warehouse_id (ascending)
                    def source_sort_key(s: Dict[str, Any]):
                        s_wh = s["warehouse_id"]
                        dist = lookup_route_distance(cfg, routes_map, s_wh, dest_wh)
                        dist_val = dist if dist is not None else float("inf")
                        return (-s["remaining_surplus"], dist_val, s_wh)

                    avail_sources.sort(key=source_sort_key)
                    chosen_source = avail_sources[0]
                    src_wh = chosen_source["warehouse_id"]
                    surplus_avail = chosen_source["remaining_surplus"]

                    # Base transfer quantity
                    base_transfer = min(surplus_avail, dest["remaining_need"])

                    # Apply Transfer Multiples & Constraints
                    eff_pack_size = cfg.sku_transfer_pack_sizes.get(sku_id, cfg.transfer_pack_size)
                    eff_min_qty = cfg.sku_min_transfer_quantities.get(sku_id, cfg.minimum_transfer_quantity)
                    eff_max_qty = cfg.sku_max_transfer_quantities.get(sku_id, cfg.maximum_transfer_quantity)

                    constraints_applied: List[str] = []
                    transfer_qty = base_transfer

                    # 1. Maximum transfer quantity cap
                    if eff_max_qty is not None and eff_max_qty > 0:
                        if transfer_qty > eff_max_qty:
                            transfer_qty = float(eff_max_qty)
                            constraints_applied.append(TransferConstraint.MAX_TRANSFER_QTY_CAPPED.value)

                    # 2. Transfer pack size rounding (round down to never exceed surplus or destination need)
                    if eff_pack_size is not None and eff_pack_size > 1:
                        remainder = transfer_qty % eff_pack_size
                        if remainder != 0:
                            transfer_qty = float(math.floor(transfer_qty / eff_pack_size) * eff_pack_size)
                            constraints_applied.append(TransferConstraint.TRANSFER_PACK_SIZE_ROUNDED.value)

                    # 3. Minimum transfer quantity
                    if eff_min_qty is not None and eff_min_qty > 0:
                        if transfer_qty < eff_min_qty:
                            if surplus_avail >= eff_min_qty and dest["remaining_need"] >= eff_min_qty:
                                transfer_qty = float(eff_min_qty)
                                constraints_applied.append(TransferConstraint.MIN_TRANSFER_QTY_APPLIED.value)
                            else:
                                transfer_qty = 0.0

                    used_sources_for_dest.add(src_wh)

                    transfer_qty = max(0.0, float(transfer_qty))
                    if transfer_qty <= 0.0:
                        # Cannot transfer from this source under constraints without violating surplus/need
                        continue

                    # Deduct from source surplus and destination need
                    chosen_source["remaining_surplus"] -= transfer_qty
                    dest["remaining_need"] -= transfer_qty
                    source_inventory_remaining[sku_id][src_wh] = round(chosen_source["remaining_surplus"], 2)

                    # Lookup route distance, transit lead time, and per-unit transfer cost
                    dist_km = lookup_route_distance(cfg, routes_map, src_wh, dest_wh)
                    lt_days = lookup_route_lead_time(cfg, routes_map, src_wh, dest_wh)
                    cost_per_unit = lookup_route_cost(cfg, routes_map, src_wh, dest_wh)

                    if cost_per_unit is not None and cost_per_unit >= 0.0:
                        estimated_cost = round(transfer_qty * cost_per_unit, 2)
                    else:
                        estimated_cost = None

                    # Build deterministic ID and explainable rationale
                    rec_id = generate_deterministic_transfer_id(
                        sku_id=sku_id,
                        source_warehouse_id=src_wh,
                        destination_warehouse_id=dest_wh,
                        transfer_quantity=transfer_qty,
                        as_of_date=as_of_date,
                    )

                    rationale = generate_transfer_rationale(
                        sku_id=sku_id,
                        source_warehouse_id=src_wh,
                        destination_warehouse_id=dest_wh,
                        destination_net_pos=dest["net_position"],
                        destination_target=dest["target_stock_level"],
                        destination_risk=dest["risk_category"],
                        source_net_pos=chosen_source["net_position"],
                        source_rop=chosen_source["reorder_point"],
                        source_surplus=chosen_source["transferable_surplus"],
                        transfer_qty=transfer_qty,
                        constraints_applied=constraints_applied,
                    )

                    recommendation = WarehouseTransferRecommendation(
                        recommendation_id=rec_id,
                        sku_id=sku_id,
                        source_warehouse_id=src_wh,
                        destination_warehouse_id=dest_wh,
                        transfer_quantity=round(transfer_qty, 2),
                        source_net_inventory_position=round(chosen_source["net_position"], 2),
                        destination_net_inventory_position=round(dest["net_position"], 2),
                        source_reorder_point=round(chosen_source["reorder_point"], 2),
                        destination_reorder_point=round(dest["reorder_point"], 2),
                        source_transferable_surplus=round(chosen_source["transferable_surplus"], 2),
                        destination_required_qty=round(dest["required_qty"], 2),
                        destination_risk_category=dest["risk_category"],
                        priority=dest["priority"],
                        forecast_daily_demand=round(dest["forecast_daily_demand"], 4),
                        estimated_transfer_cost=estimated_cost,
                        distance_km=round(dist_km, 2) if dist_km is not None else None,
                        transfer_lead_time_days=round(lt_days, 2) if lt_days is not None else None,
                        rationale=rationale,
                        constraints_applied=constraints_applied,
                        approval_required=True,
                        status=ProposalStatus.DRAFT.value,
                    )
                    recommendations.append(recommendation)

                # Record any remaining unmet requirement
                if dest["remaining_need"] > 0:
                    unmet_destination_demand.append({
                        "sku_id": sku_id,
                        "destination_warehouse_id": dest_wh,
                        "unmet_quantity": round(dest["remaining_need"], 2),
                        "requested_quantity": round(dest["required_qty"], 2),
                        "priority": dest["priority"],
                        "risk_category": dest["risk_category"],
                    })

        # 5. Calculate portfolio aggregates and cost completeness status
        total_transfer_qty = sum(r.transfer_quantity for r in recommendations)
        rec_count = len(recommendations)

        if rec_count == 0:
            estimated_total_cost = None
            cost_status = CostDataStatus.NO_COST_DATA.value
        else:
            cost_flags = [r.estimated_transfer_cost is not None for r in recommendations]
            if all(cost_flags):
                cost_status = CostDataStatus.COMPLETE_COST_DATA.value
                estimated_total_cost = round(
                    sum(r.estimated_transfer_cost for r in recommendations if r.estimated_transfer_cost is not None), 2
                )
            elif not any(cost_flags):
                cost_status = CostDataStatus.NO_COST_DATA.value
                estimated_total_cost = None
            else:
                cost_status = CostDataStatus.PARTIAL_COST_DATA.value
                estimated_total_cost = round(
                    sum(r.estimated_transfer_cost for r in recommendations if r.estimated_transfer_cost is not None), 2
                )

        return WarehouseRebalancingResult(
            recommendations=recommendations,
            excluded_dormant=excluded_dormant,
            insufficient_policy_inputs=insufficient_policy_inputs,
            unmet_destination_demand=unmet_destination_demand,
            source_inventory_remaining=source_inventory_remaining,
            total_transfer_quantity=round(total_transfer_qty, 2),
            recommendation_count=rec_count,
            estimated_total_transfer_cost=estimated_total_cost,
            cost_data_status=cost_status,
            summary={
                "sku_count_evaluated": len(unique_skus),
                "recommendation_count": rec_count,
                "total_transfer_quantity": round(total_transfer_qty, 2),
                "unmet_destination_count": len(unmet_destination_demand),
                "excluded_dormant_count": len(excluded_dormant),
                "insufficient_policy_inputs_count": len(insufficient_policy_inputs),
            },
        )
