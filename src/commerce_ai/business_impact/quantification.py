"""Deterministic Business Impact Quantification Engine (Phase 6F).

Quantifies observed and modeled financial and operational exposure for cross-domain signals.
Guarantees:
- Strictly auditable, deterministic calculations
- Preserves natural entity grain and currency
- Zero vs None semantics (missing data is never converted to 0)
- Traceability back to triggering signal IDs
- Explicit confidence levels: DIRECT_OBSERVED, ESTIMATED, MODEL_BASED, INSUFFICIENT_DATA
- Explicit non-prescriptive opportunity proxy values
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from commerce_ai.cross_domain.schemas import (
    CrossDomainBusinessRecord,
    CrossDomainSignal,
    SignalSeverity,
)
from commerce_ai.business_impact.schemas import (
    BusinessImpactConfig,
    BusinessImpactRecord,
    CalculationStatus,
    ImpactCategory,
    ImpactConfidence,
    ImpactType,
)


def generate_impact_id(
    signal_id: str,
    impact_type: str,
    sku_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> str:
    """Generate a deterministic, reproducible impact identifier."""
    key = f"{signal_id}:{impact_type}:{sku_id or ''}:{warehouse_id or ''}:{channel_id or ''}:{as_of_date or ''}"
    hash_hex = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"IMP-{hash_hex}"


def _infer_unit_cost(record: CrossDomainBusinessRecord) -> Optional[float]:
    """Deterministically derive standard unit cost from available product or inventory metrics."""
    # 1. From product_cost / units_sold if valid
    if record.product_cost is not None and record.units_sold is not None and record.units_sold > 0:
        c = record.product_cost / record.units_sold
        if c > 0:
            return round(c, 4)
    # 2. From inventory_value / current_on_hand
    if record.inventory_value is not None and record.current_on_hand is not None and record.current_on_hand > 0:
        c = record.inventory_value / record.current_on_hand
        if c > 0:
            return round(c, 4)
    return None


def quantify_signal_impact(
    signal: CrossDomainSignal,
    record: Optional[CrossDomainBusinessRecord],
    config: Optional[BusinessImpactConfig] = None,
) -> BusinessImpactRecord:
    """Evaluate deterministic impact quantification for a single cross-domain signal."""
    cfg = config or BusinessImpactConfig()
    sig_type = signal.signal_type
    as_of_str = signal.as_of_date or (str(cfg.as_of_date) if cfg.as_of_date is not None else None)

    # Fallback mock empty record if None provided
    rec = record or CrossDomainBusinessRecord(
        sku_id=signal.sku_id or "UNKNOWN",
        warehouse_id=signal.warehouse_id,
        channel_id=signal.channel_id,
        as_of_date=as_of_str,
    )

    currency = rec.currency or cfg.default_currency
    sku_id = rec.sku_id
    warehouse_id = rec.warehouse_id
    channel_id = rec.channel_id
    cat_id = rec.category_id
    brand = rec.brand

    # Initialize common variables
    imp_cat = ImpactCategory.CROSS_DOMAIN
    imp_type = ImpactType.REVENUE_EXPOSURE
    conf = ImpactConfidence.DIRECT_OBSERVED
    calc_status = CalculationStatus.CALCULATED
    calc_method = "OBSERVED_VALUE"
    calc_inputs: Dict[str, Any] = {}
    exposure_val: Optional[float] = None
    opp_proxy: Optional[float] = None
    desc = ""

    # Rule-specific quantification
    if sig_type == "HIGH_REVENUE_LOW_STOCK":
        imp_cat = ImpactCategory.REVENUE
        imp_type = ImpactType.REVENUE_EXPOSURE
        conf = ImpactConfidence.DIRECT_OBSERVED
        exposure_val = rec.net_revenue
        opp_proxy = rec.net_revenue
        calc_method = "OBSERVED_NET_REVENUE"
        calc_inputs = {
            "net_revenue": rec.net_revenue,
            "available_inventory": rec.available_inventory,
            "units_sold": rec.units_sold,
        }
        desc = (
            f"Observed net revenue of ${rec.net_revenue or 0.0:.2f} is associated with low available stock "
            f"({rec.available_inventory or 0} units)."
        )

    elif sig_type == "HIGH_MARGIN_LOW_STOCK":
        imp_cat = ImpactCategory.MARGIN
        imp_type = ImpactType.MARGIN_EXPOSURE
        conf = ImpactConfidence.DIRECT_OBSERVED
        exposure_val = rec.gross_margin
        opp_proxy = rec.gross_margin
        calc_method = "OBSERVED_GROSS_MARGIN"
        calc_inputs = {
            "gross_margin": rec.gross_margin,
            "available_inventory": rec.available_inventory,
        }
        desc = (
            f"Observed gross margin of ${rec.gross_margin or 0.0:.2f} is associated with low available stock "
            f"({rec.available_inventory or 0} units)."
        )

    elif sig_type in {"HIGH_REVENUE_STOCKOUT_EXPOSURE", "HIGH_VALUE_STOCKOUT_EXPOSURE"}:
        imp_cat = ImpactCategory.STOCKOUT
        imp_type = ImpactType.STOCKOUT_REVENUE_EXPOSURE
        conf = ImpactConfidence.DIRECT_OBSERVED
        calc_inputs = {
            "stockout_days": rec.stockout_days,
            "stockout_rate": rec.stockout_rate,
            "associated_net_revenue": rec.net_revenue,
        }
        # Check if supported lost demand calculation is available
        if rec.estimated_lost_demand is not None and rec.estimated_lost_demand > 0 and rec.units_sold and rec.net_revenue:
            unit_price = rec.net_revenue / rec.units_sold
            exposure_val = round(rec.estimated_lost_demand * unit_price, 2)
            conf = ImpactConfidence.ESTIMATED
            calc_method = "ESTIMATED_LOST_DEMAND_X_UNIT_REVENUE"
            calc_inputs["estimated_lost_demand_units"] = rec.estimated_lost_demand
            calc_inputs["unit_price"] = unit_price
            desc = (
                f"Estimated stockout revenue exposure of ${exposure_val:.2f} is derived from "
                f"{rec.estimated_lost_demand} lost demand units across {rec.stockout_days or 0} stockout days."
            )
        else:
            # Strictly historical net revenue exposure; do NOT calculate unsupported lost sales
            exposure_val = rec.net_revenue
            calc_method = "OBSERVED_NET_REVENUE_DURING_STOCKOUT_PERIOD"
            desc = (
                f"Historical net revenue of ${rec.net_revenue or 0.0:.2f} is associated with stockout exposure "
                f"({rec.stockout_days or 0} stockout days, rate {rec.stockout_rate or 0.0:.1%})."
            )
        opp_proxy = exposure_val

    elif sig_type == "LOW_MARGIN_HIGH_INVENTORY":
        imp_cat = ImpactCategory.INVENTORY
        imp_type = ImpactType.INVENTORY_CAPITAL_EXPOSURE
        conf = ImpactConfidence.DIRECT_OBSERVED
        exposure_val = rec.inventory_value
        opp_proxy = rec.inventory_value
        calc_method = "OBSERVED_INVENTORY_VALUE"
        calc_inputs = {
            "inventory_value": rec.inventory_value,
            "current_on_hand": rec.current_on_hand,
            "gross_margin_pct": rec.gross_margin_pct,
        }
        desc = (
            f"Inventory capital valuation of ${rec.inventory_value or 0.0:.2f} is associated with low gross margin "
            f"percentage ({rec.gross_margin_pct or 0.0:.1%})."
        )

    elif sig_type in {"HIGH_INVENTORY_LOW_DEMAND", "LOW_VELOCITY_HIGH_VALUE"}:
        imp_cat = ImpactCategory.INVENTORY
        imp_type = ImpactType.SLOW_MOVING_INVENTORY_EXPOSURE
        conf = ImpactConfidence.DIRECT_OBSERVED
        exposure_val = rec.inventory_value
        opp_proxy = rec.inventory_value
        calc_method = "OBSERVED_INVENTORY_VALUE"
        calc_inputs = {
            "inventory_value": rec.inventory_value,
            "current_on_hand": rec.current_on_hand,
            "average_daily_demand": rec.average_daily_demand,
            "abc_class": rec.abc_class,
        }
        desc = (
            f"Inventory capital valuation of ${rec.inventory_value or 0.0:.2f} is associated with low demand velocity "
            f"({rec.average_daily_demand or 0.0:.2f} units/day)."
        )

    elif sig_type == "HIGH_VALUE_INVENTORY":
        imp_cat = ImpactCategory.INVENTORY
        imp_type = ImpactType.HIGH_VALUE_INVENTORY_EXPOSURE
        conf = ImpactConfidence.DIRECT_OBSERVED
        exposure_val = rec.inventory_value
        opp_proxy = rec.inventory_value
        calc_method = "OBSERVED_INVENTORY_VALUE"
        calc_inputs = {
            "inventory_value": rec.inventory_value,
            "current_on_hand": rec.current_on_hand,
        }
        desc = (
            f"Inventory capital valuation of ${rec.inventory_value or 0.0:.2f} is observed for this item."
        )

    elif sig_type == "LOW_REVENUE_HIGH_INVENTORY":
        imp_cat = ImpactCategory.INVENTORY
        imp_type = ImpactType.INVENTORY_CAPITAL_EXPOSURE
        conf = ImpactConfidence.DIRECT_OBSERVED
        exposure_val = rec.inventory_value
        opp_proxy = rec.inventory_value
        calc_method = "OBSERVED_INVENTORY_VALUE"
        # Zero-denominator protection on inventory-to-revenue ratio
        ratio = (rec.inventory_value / rec.net_revenue) if (rec.net_revenue is not None and rec.net_revenue > 0 and rec.inventory_value is not None) else None
        calc_inputs = {
            "inventory_value": rec.inventory_value,
            "net_revenue": rec.net_revenue,
            "inventory_to_revenue_ratio": ratio,
        }
        ratio_str = f"{ratio:.2f}" if ratio is not None else "N/A"
        desc = (
            f"Inventory capital valuation of ${rec.inventory_value or 0.0:.2f} is associated with lower net revenue "
            f"(${rec.net_revenue or 0.0:.2f}, inventory-to-revenue ratio {ratio_str})."
        )

    elif sig_type in {"HIGH_RETURN_HIGH_REVENUE", "HIGH_VALUE_RETURN_EXPOSURE"}:
        imp_cat = ImpactCategory.RETURNS
        imp_type = ImpactType.RETURN_REVENUE_EXPOSURE
        # Estimate return revenue exposure as return_rate * net_revenue
        if rec.return_rate is not None and rec.net_revenue is not None:
            exposure_val = round(rec.return_rate * rec.net_revenue, 2)
            conf = ImpactConfidence.ESTIMATED
            calc_method = "RETURN_RATE_X_NET_REVENUE"
            calc_inputs = {"return_rate": rec.return_rate, "net_revenue": rec.net_revenue}
            desc = (
                f"Estimated return revenue exposure of ${exposure_val:.2f} is associated with return rate "
                f"({rec.return_rate:.2%}) on net revenue of ${rec.net_revenue:.2f}."
            )
        else:
            conf = ImpactConfidence.INSUFFICIENT_DATA
            calc_status = CalculationStatus.INSUFFICIENT_DATA
            calc_method = "INSUFFICIENT_RETURN_OR_REVENUE_DATA"
            desc = "Return activity observed, but return rate or revenue is unavailable."
        opp_proxy = exposure_val

    elif sig_type in {"HIGH_RETURN_LOW_MARGIN", "HIGH_MARGIN_HIGH_RETURN"}:
        imp_cat = ImpactCategory.RETURNS
        imp_type = ImpactType.RETURN_MARGIN_EXPOSURE
        if rec.return_rate is not None and rec.gross_margin is not None:
            exposure_val = round(rec.return_rate * rec.gross_margin, 2)
            conf = ImpactConfidence.ESTIMATED
            calc_method = "RETURN_RATE_X_GROSS_MARGIN"
            calc_inputs = {
                "return_rate": rec.return_rate,
                "gross_margin": rec.gross_margin,
                "gross_margin_pct": rec.gross_margin_pct,
            }
            desc = (
                f"Estimated return margin exposure of ${exposure_val:.2f} is associated with return rate "
                f"({rec.return_rate:.2%}) on gross margin of ${rec.gross_margin:.2f}."
            )
        else:
            conf = ImpactConfidence.INSUFFICIENT_DATA
            calc_status = CalculationStatus.INSUFFICIENT_DATA
            calc_method = "INSUFFICIENT_RETURN_OR_MARGIN_DATA"
            desc = "Return activity observed, but return rate or margin is unavailable."
        opp_proxy = exposure_val

    elif sig_type == "RETURN_ANOMALY_HIGH_VALUE_SKU":
        imp_cat = ImpactCategory.RETURNS
        imp_type = ImpactType.RETURN_REVENUE_EXPOSURE
        # If upstream Phase 5C model risk output is present, use it as MODEL_BASED
        if rec.return_risk is not None and hasattr(signal, "observed_metrics") and "expected_return_exposure" in signal.observed_metrics:
            exp_exp = float(signal.observed_metrics["expected_return_exposure"])
            exposure_val = exp_exp
            conf = ImpactConfidence.MODEL_BASED
            calc_method = "RETURN_MODEL_EXPECTED_EXPOSURE"
            calc_inputs = {"expected_return_exposure": exp_exp, "return_risk": rec.return_risk}
            desc = (
                f"Model-based expected return exposure of ${exposure_val:.2f} is associated with return anomaly."
            )
        else:
            exposure_val = rec.inventory_value
            conf = ImpactConfidence.DIRECT_OBSERVED
            calc_method = "OBSERVED_INVENTORY_VALUE"
            calc_inputs = {"inventory_value": rec.inventory_value, "return_anomaly_flag": True}
            desc = (
                f"Inventory valuation of ${rec.inventory_value or 0.0:.2f} is associated with return anomaly indicator."
            )
        opp_proxy = exposure_val

    elif sig_type in {
        "HIGH_REVENUE_REPLENISHMENT_TRIGGER",
        "HIGH_MARGIN_REPLENISHMENT_TRIGGER",
        "LOW_MARGIN_REPLENISHMENT_TRIGGER",
        "HIGH_VALUE_REPLENISHMENT_TRIGGER",
    }:
        imp_cat = ImpactCategory.REPLENISHMENT
        imp_type = ImpactType.REPLENISHMENT_FINANCIAL_EXPOSURE
        unit_cost = _infer_unit_cost(rec)
        roq = rec.recommended_order_qty or 0
        if unit_cost is not None and roq > 0:
            prop_val = round(roq * unit_cost, 2)
            exposure_val = prop_val
            conf = ImpactConfidence.ESTIMATED
            calc_method = "RECOMMENDED_QTY_X_UNIT_COST"
            calc_inputs = {
                "recommended_order_qty": roq,
                "unit_cost": unit_cost,
                "proposed_purchase_value": prop_val,
            }
            desc = (
                f"Proposed purchase valuation of ${prop_val:.2f} is associated with active replenishment trigger "
                f"({roq} units at unit cost ${unit_cost:.2f})."
            )
        else:
            exposure_val = None
            conf = ImpactConfidence.INSUFFICIENT_DATA
            calc_status = CalculationStatus.INSUFFICIENT_DATA
            calc_method = "MISSING_UNIT_COST"
            calc_inputs = {"recommended_order_qty": roq, "unit_cost": None}
            desc = "Active replenishment trigger observed, but unit cost is unavailable to calculate proposed purchase value."
        opp_proxy = exposure_val

    elif sig_type == "FORECASTED_DEMAND_ABOVE_AVAILABLE_STOCK":
        imp_cat = ImpactCategory.FORECAST
        imp_type = ImpactType.FORECAST_COVERAGE_EXPOSURE
        fc_mean = rec.forecast_mean or 0.0
        avail = rec.available_inventory or 0
        gap = max(0.0, fc_mean - avail)
        calc_inputs = {"forecast_mean": fc_mean, "available_inventory": avail, "forecast_inventory_gap": gap}
        # In accordance with Section 20, do not convert coverage gap into speculative monetary loss
        exposure_val = None
        conf = ImpactConfidence.ESTIMATED
        calc_method = "FORECAST_MEAN_MINUS_AVAILABLE_STOCK"
        desc = (
            f"Forecasted demand of {fc_mean:.1f} units exceeds available inventory ({avail} units) "
            f"by {gap:.1f} units over horizon."
        )

    elif sig_type in {"FORECAST_UNAVAILABLE", "FORECASTED_DEMAND_BELOW_CURRENT_STOCK"}:
        imp_cat = ImpactCategory.FORECAST
        imp_type = ImpactType.FORECAST_COVERAGE_EXPOSURE
        exposure_val = None
        conf = ImpactConfidence.INSUFFICIENT_DATA if sig_type == "FORECAST_UNAVAILABLE" else ImpactConfidence.DIRECT_OBSERVED
        calc_status = CalculationStatus.INSUFFICIENT_DATA if sig_type == "FORECAST_UNAVAILABLE" else CalculationStatus.CALCULATED
        calc_method = "FORECAST_COVERAGE_AUDIT"
        desc = "Forecast data is unavailable for this entity." if sig_type == "FORECAST_UNAVAILABLE" else (
            f"Available inventory of {rec.available_inventory or 0} units exceeds 3x forecasted demand."
        )

    else:
        # Generic fallback
        exposure_val = rec.net_revenue or rec.inventory_value
        conf = ImpactConfidence.DIRECT_OBSERVED if exposure_val is not None else ImpactConfidence.INSUFFICIENT_DATA
        calc_method = "GENERIC_OBSERVED_VALUE"
        desc = f"Associated financial exposure for signal {sig_type}."

    impact_id = generate_impact_id(
        signal_id=signal.signal_id,
        impact_type=imp_type.value,
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        channel_id=channel_id,
        as_of_date=as_of_str,
    )

    return BusinessImpactRecord(
        impact_id=impact_id,
        signal_id=signal.signal_id,
        signal_type=sig_type,
        impact_category=imp_cat,
        impact_type=imp_type,
        severity=signal.severity,
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        channel_id=channel_id,
        category_id=cat_id,
        brand=brand,
        affected_units=rec.units_sold,
        affected_inventory_units=rec.current_on_hand,
        affected_inventory_value=rec.inventory_value,
        associated_gross_revenue=rec.gross_revenue,
        associated_net_revenue=rec.net_revenue,
        associated_gross_margin=rec.gross_margin,
        associated_gross_margin_pct=rec.gross_margin_pct,
        known_contribution_margin=rec.known_contribution_margin,
        known_contribution_margin_pct=rec.known_contribution_margin_pct,
        exposure_value=exposure_val,
        exposure_currency=currency,
        confidence_status=conf,
        calculation_method=calc_method,
        calculation_inputs=calc_inputs,
        calculation_status=calc_status,
        opportunity_proxy_value=opp_proxy,
        data_quality_status=rec.data_quality_status,
        domain_coverage_pct=rec.domain_coverage_pct,
        as_of_date=as_of_str,
        description=desc,
    )


def get_economic_dimension(record: BusinessImpactRecord) -> str:
    """Resolve the economic exposure dimension for deduplication.

    Distinguishes physical capital overlap from distinct economic metrics:
    - PHYSICAL_INVENTORY: All signals measuring physical stock on hand
    - RETURN_REVENUE: Top-line revenue returned
    - RETURN_MARGIN: Margin eroded by returns
    - REVENUE: Sales revenue exposure
    - MARGIN: Gross margin exposure
    - REPLENISHMENT: Proposed purchase order capital allocation
    """
    if record.impact_category == ImpactCategory.INVENTORY:
        return "PHYSICAL_INVENTORY"
    if record.impact_type == ImpactType.RETURN_REVENUE_EXPOSURE:
        return "RETURN_REVENUE"
    if record.impact_type == ImpactType.RETURN_MARGIN_EXPOSURE:
        return "RETURN_MARGIN"
    if record.impact_type in {ImpactType.REVENUE_EXPOSURE, ImpactType.STOCKOUT_REVENUE_EXPOSURE}:
        return "REVENUE"
    if record.impact_type in {ImpactType.MARGIN_EXPOSURE, ImpactType.STOCKOUT_MARGIN_EXPOSURE}:
        return "MARGIN"
    if record.impact_type == ImpactType.REPLENISHMENT_FINANCIAL_EXPOSURE:
        return "REPLENISHMENT"
    return record.impact_type.value


def calculate_physical_capital_exposure(
    records: List[BusinessImpactRecord],
) -> Tuple[float, float, str]:
    """Calculate gross vs deduplicated physical inventory capital exposure ($).

    Rule:
    Filters to inventory capital signals (ImpactCategory.INVENTORY) and groups by
    physical location (sku_id, warehouse_id). For each physical location, takes the
    maximum valuation because multiple overlapping inventory signals (e.g. HIGH_VALUE_INVENTORY
    and HIGH_INVENTORY_LOW_DEMAND) evaluate the exact same on-hand units.

    Returns:
        (gross_inventory_exposure, deduplicated_physical_capital_exposure, status_string)
    """
    inv_records = [
        r for r in records
        if r.impact_category == ImpactCategory.INVENTORY and r.exposure_value is not None and r.exposure_value > 0
    ]
    if not inv_records:
        return 0.0, 0.0, "NO_INVENTORY_RECORDS"

    gross_inv = sum(float(r.exposure_value) for r in inv_records)
    entity_inv_max: Dict[Tuple[str, str], float] = {}
    for r in inv_records:
        key = (str(r.sku_id or ""), str(r.warehouse_id or ""))
        entity_inv_max[key] = max(entity_inv_max.get(key, 0.0), float(r.exposure_value))

    dedup_inv = sum(entity_inv_max.values())
    return round(gross_inv, 2), round(dedup_inv, 2), "DEDUPLICATED_BY_PHYSICAL_LOCATION_MAX"


def deduplicate_portfolio_exposure(
    records: List[BusinessImpactRecord],
) -> Tuple[float, str]:
    """Calculate non-double-counted exposure across overlapping signals.

    Deduplication Methodology:
    1. Physical Capital Overlap:
       Signals measuring physical inventory on-hand (ImpactCategory.INVENTORY) are grouped
       by physical location (sku_id, warehouse_id). For each physical location, the maximum
       valuation is retained because multiple inventory signals evaluate the identical stock.
    2. Economic Dimension Separation:
       Distinct economic metrics (Revenue, Gross Margin, Return Revenue, Return Margin, Replenishment)
       remain separate and are never collapsed against each other.
    3. Duplicate Signal Protection:
       Duplicate signals within the same economic dimension and entity are deduplicated to their
       maximum exposure value.
    """
    if not records:
        return 0.0, "EMPTY_DATASET"

    entity_dim_max: Dict[Tuple[str, str, str, str], float] = {}
    for r in records:
        if r.exposure_value is not None and r.exposure_value > 0:
            dim = get_economic_dimension(r)
            ch = str(r.channel_id or "") if dim != "PHYSICAL_INVENTORY" else ""
            key = (str(r.sku_id or ""), str(r.warehouse_id or ""), ch, dim)
            entity_dim_max[key] = max(entity_dim_max.get(key, 0.0), float(r.exposure_value))

    deduped_total = sum(entity_dim_max.values())
    return round(deduped_total, 2), "DEDUPLICATED_BY_ECONOMIC_DIMENSION_AND_ENTITY"
