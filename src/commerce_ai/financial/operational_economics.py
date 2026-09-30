"""Operational Economics Engine & Cost Completeness Architecture (Phase 6D).

Provides deterministic calculations for:
- Operational cost resolution with explicit provenance (SOURCE_DATA, CATALOG_ESTIMATE, CONFIGURED_ASSUMPTION, UNAVAILABLE)
- Multi-grain cost allocation (UNIT, TRANSACTION, ORDER, RETURN_EVENT) without multi-line double-counting
- Integration of actual return events with reverse logistics assumptions
- Strict separation of observed zero cost ($0.00) from unobserved/missing cost (None)
- Configurable Cost Completeness auditing (COMPLETE, PARTIALLY_COMPLETE, INSUFFICIENT_DATA)
- Two-tier contribution margin: Known Contribution Margin vs Final Contribution Margin
- Order-level economics aggregating true order costs
- Dimensional aggregations (SKU, Category, Brand, Channel, Warehouse, Cross-slices, Time series)
- Executive network portfolio operational economics summary
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.financial.schemas import (
    ContributionMarginStatus,
    CostAssumptionsConfig,
    CostComponent,
    CostComponentDetail,
    CostComponentStatus,
    CostCompletenessReport,
    CostCompletenessStatus,
    CostGrain,
    CostSourceType,
    OperationalCostDetail,
    OperationalEconomicsConfig,
    OperationalEconomicsRecord,
    OperationalEconomicsSegment,
    OperationalEconomicsStatus,
    OperationalEconomicsSummary,
    OrderOperationalEconomicsRecord,
    TimeGrain,
)
from commerce_ai.financial.revenue_margin import filter_sales_by_as_of_date


# =============================================================================
# Deterministic Identifiers
# =============================================================================


def generate_operational_record_id(
    transaction_id: str,
    sku_id: str,
    order_id: str,
    date_val: str,
    currency: str = "USD",
) -> str:
    """Generate a deterministic 16-character SHA-256 identifier for an operational economics record.

    Format: FIN-OE-{hexdigest[:16]}
    """
    raw = f"OE|{str(transaction_id).strip()}|{str(sku_id).strip()}|{str(order_id).strip()}|{str(date_val).strip()}|{str(currency).strip().upper()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"FIN-OE-{digest[:16]}"


def generate_operational_order_id(
    order_id: str,
    date_val: str,
    currency: str = "USD",
) -> str:
    """Generate a deterministic 16-character SHA-256 identifier for an order operational economics record.

    Format: FIN-ORD-{hexdigest[:16]}
    """
    raw = f"ORD|{str(order_id).strip()}|{str(date_val).strip()}|{str(currency).strip().upper()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"FIN-ORD-{digest[:16]}"


def generate_operational_segment_id(
    dimension: str,
    segment_key: str,
    as_of_date: Optional[str] = None,
    currency: str = "USD",
) -> str:
    """Generate a deterministic 16-character SHA-256 identifier for an operational economics segment.

    Format: FIN-OESEG-{hexdigest[:16]}
    """
    raw = f"OESEG|{str(dimension).strip().upper()}|{str(segment_key).strip()}|{str(as_of_date or '').strip()}|{str(currency).strip().upper()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"FIN-OESEG-{digest[:16]}"


# =============================================================================
# Single Record Cost Resolution
# =============================================================================


def resolve_transaction_operational_costs(
    row: Union[pd.Series, Dict[str, Any]],
    unit_cost: Optional[float] = None,
    order_context: Optional[Dict[str, Any]] = None,
    return_context: Optional[Dict[str, Any]] = None,
    config: Optional[OperationalEconomicsConfig] = None,
) -> Dict[str, OperationalCostDetail]:
    """Resolve operational cost details for an individual transaction line with full provenance.

    Args:
        row: Sales transaction line data.
        unit_cost: Optional standard unit cost from catalog.
        order_context: Optional multi-line order context (line_count, order_net_revenue, order_quantity).
        return_context: Optional return event context (has_return_data, returned_units, return_count).
        config: Operational economics configuration.

    Returns:
        Dictionary mapping CostComponent name to OperationalCostDetail.
    """
    cfg = config or OperationalEconomicsConfig()
    asm = cfg.assumptions
    ctx = order_context or {}
    ret_ctx = return_context or {}

    def get_val(key: str, alt_key: Optional[str] = None) -> Any:
        if isinstance(row, dict):
            val = row.get(key)
            if val is None and alt_key:
                val = row.get(alt_key)
            return val
        if key in row:
            return row[key]
        if alt_key and alt_key in row:
            return row[alt_key]
        return None

    def safe_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
            return None if np.isnan(f) or np.isinf(f) else f
        except (ValueError, TypeError):
            return None

    def safe_int(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            f = float(v)
            return None if np.isnan(f) or np.isinf(f) else int(f)
        except (ValueError, TypeError):
            return None

    qty = safe_int(get_val("quantity"))
    unit_price = safe_float(get_val("unit_price"))
    discount = safe_float(get_val("discount")) or 0.0
    channel_id = str(get_val("channel_id") or "").strip()
    warehouse_id = str(get_val("warehouse_id") or "").strip()
    currency = str(get_val("currency") or cfg.default_currency).strip().upper()

    gross_rev = (qty * unit_price) if (qty is not None and unit_price is not None) else None
    net_rev = (gross_rev - discount) if gross_rev is not None else None

    # Multi-line allocation factor
    order_line_count = int(ctx.get("line_count", 1))
    order_net_rev = safe_float(ctx.get("order_net_revenue"))
    order_qty = safe_int(ctx.get("order_quantity"))

    allocation_ratio = 1.0
    if order_line_count > 1:
        if cfg.order_cost_allocation_method == "NET_REVENUE" and order_net_rev is not None and order_net_rev > 0 and net_rev is not None:
            allocation_ratio = max(0.0, net_rev / order_net_rev)
        elif cfg.order_cost_allocation_method in ("NET_REVENUE", "QUANTITY") and order_qty is not None and order_qty > 0 and qty is not None:
            allocation_ratio = max(0.0, qty / order_qty)
        else:
            allocation_ratio = 1.0 / order_line_count

    details: Dict[str, OperationalCostDetail] = {}

    def satisfies_completeness(src: CostSourceType) -> bool:
        if cfg.strict_source_only:
            return src == CostSourceType.SOURCE_DATA
        if src == CostSourceType.SOURCE_DATA:
            return True
        if src == CostSourceType.CATALOG_ESTIMATE and cfg.allow_estimated_costs_for_completeness:
            return True
        if src == CostSourceType.CONFIGURED_ASSUMPTION and cfg.allow_assumed_costs_for_completeness:
            return True
        return False

    # 1. PRODUCT_COST
    raw_prod_cost = safe_float(get_val("product_cost"))
    raw_unit_cost = safe_float(unit_cost if unit_cost is not None else get_val("unit_cost", "unit_product_cost"))

    if raw_prod_cost is not None:
        p_amt = raw_prod_cost
        p_unit = p_amt / qty if qty and qty > 0 else None
        p_src = CostSourceType.SOURCE_DATA
        p_stat = CostComponentStatus.AVAILABLE_SOURCE
        p_grain = CostGrain.TRANSACTION
        p_ref = "sales.product_cost"
        p_desc = "Transactional observed product cost"
    elif raw_unit_cost is not None and raw_unit_cost >= 0 and qty is not None:
        p_amt = raw_unit_cost * qty
        p_unit = raw_unit_cost
        p_src = CostSourceType.CATALOG_ESTIMATE
        p_stat = CostComponentStatus.AVAILABLE_ESTIMATED
        p_grain = CostGrain.UNIT
        p_ref = "products.csv:unit_cost"
        p_desc = "Catalog standard unit procurement cost"
    else:
        p_amt = None
        p_unit = None
        p_src = CostSourceType.UNAVAILABLE
        p_stat = CostComponentStatus.UNAVAILABLE
        p_grain = CostGrain.UNKNOWN
        p_ref = None
        p_desc = "Product cost unavailable"

    details[CostComponent.PRODUCT_COST.value] = OperationalCostDetail(
        component=CostComponent.PRODUCT_COST,
        amount=p_amt,
        unit_amount=p_unit,
        currency=currency,
        source_type=p_src,
        availability_status=p_stat,
        cost_grain=p_grain,
        is_estimated=(p_src == CostSourceType.CATALOG_ESTIMATE),
        is_assumed=(p_src == CostSourceType.CONFIGURED_ASSUMPTION),
        included_in_known_contribution=(p_amt is not None),
        included_in_final_contribution=(p_amt is not None and satisfies_completeness(p_src)),
        source_reference=p_ref,
        description=p_desc,
    )

    # 2. SHIPPING_COST
    raw_ship = safe_float(get_val("shipping_cost"))
    if raw_ship is not None:
        s_amt = raw_ship
        s_unit = s_amt / qty if qty and qty > 0 else None
        s_src = CostSourceType.SOURCE_DATA
        s_stat = CostComponentStatus.AVAILABLE_SOURCE
        s_grain = CostGrain.TRANSACTION
        s_ref = "sales.shipping_cost"
        s_desc = "Observed transaction shipping cost"
    else:
        # Check configured assumptions
        has_assumed = False
        assumed_ship = 0.0

        # Unit shipping
        if asm.shipping_cost_per_unit is not None and qty is not None:
            assumed_ship += asm.shipping_cost_per_unit * qty
            has_assumed = True

        # Percentage of net revenue
        if asm.shipping_pct_of_net_revenue is not None and net_rev is not None and net_rev > 0:
            assumed_ship += asm.shipping_pct_of_net_revenue * net_rev
            has_assumed = True

        # Flat channel shipping fee (order-level allocated)
        if channel_id in asm.shipping_cost_by_channel:
            assumed_ship += asm.shipping_cost_by_channel[channel_id] * allocation_ratio
            has_assumed = True
        elif asm.shipping_cost_per_order is not None:
            assumed_ship += asm.shipping_cost_per_order * allocation_ratio
            has_assumed = True

        if has_assumed:
            s_amt = float(assumed_ship)
            s_unit = s_amt / qty if qty and qty > 0 else None
            s_src = CostSourceType.CONFIGURED_ASSUMPTION
            s_stat = CostComponentStatus.AVAILABLE_ASSUMED
            s_grain = CostGrain.ORDER if order_line_count > 1 else CostGrain.UNIT
            s_ref = "config.assumptions.shipping"
            s_desc = f"Configured shipping assumption (allocation ratio {allocation_ratio:.4f})"
        else:
            s_amt = None
            s_unit = None
            s_src = CostSourceType.UNAVAILABLE
            s_stat = CostComponentStatus.UNAVAILABLE
            s_grain = CostGrain.UNKNOWN
            s_ref = None
            s_desc = "Shipping cost unavailable"

    details[CostComponent.SHIPPING_COST.value] = OperationalCostDetail(
        component=CostComponent.SHIPPING_COST,
        amount=s_amt,
        unit_amount=s_unit,
        currency=currency,
        source_type=s_src,
        availability_status=s_stat,
        cost_grain=s_grain,
        is_estimated=(s_src == CostSourceType.CATALOG_ESTIMATE),
        is_assumed=(s_src == CostSourceType.CONFIGURED_ASSUMPTION),
        included_in_known_contribution=(s_amt is not None),
        included_in_final_contribution=(s_amt is not None and satisfies_completeness(s_src)),
        source_reference=s_ref,
        description=s_desc,
    )

    # 3. PAYMENT_PROCESSING_COST
    raw_pay = safe_float(get_val("payment_processing_cost", "payment_fee"))
    if raw_pay is not None:
        pay_amt = raw_pay
        pay_unit = pay_amt / qty if qty and qty > 0 else None
        pay_src = CostSourceType.SOURCE_DATA
        pay_stat = CostComponentStatus.AVAILABLE_SOURCE
        pay_grain = CostGrain.TRANSACTION
        pay_ref = "sales.payment_processing_cost"
        pay_desc = "Observed merchant payment processing fee"
    else:
        # Check channel specific or default assumptions
        rate = asm.payment_rate_by_channel.get(channel_id, asm.payment_processing_rate)
        fixed = asm.payment_fixed_by_channel.get(channel_id, asm.payment_processing_fixed_per_order)

        if rate is not None or fixed is not None:
            fee = 0.0
            if rate is not None and net_rev is not None and net_rev > 0:
                fee += rate * net_rev
            if fixed is not None:
                fee += fixed * allocation_ratio
            pay_amt = float(fee)
            pay_unit = pay_amt / qty if qty and qty > 0 else None
            pay_src = CostSourceType.CONFIGURED_ASSUMPTION
            pay_stat = CostComponentStatus.AVAILABLE_ASSUMED
            pay_grain = CostGrain.TRANSACTION if fixed is None else CostGrain.ORDER
            pay_ref = f"config.assumptions.payment[{channel_id}]"
            pay_desc = f"Configured payment fee assumption (rate={rate}, fixed={fixed})"
        else:
            pay_amt = None
            pay_unit = None
            pay_src = CostSourceType.UNAVAILABLE
            pay_stat = CostComponentStatus.UNAVAILABLE
            pay_grain = CostGrain.UNKNOWN
            pay_ref = None
            pay_desc = "Payment processing fee unavailable"

    details[CostComponent.PAYMENT_PROCESSING_COST.value] = OperationalCostDetail(
        component=CostComponent.PAYMENT_PROCESSING_COST,
        amount=pay_amt,
        unit_amount=pay_unit,
        currency=currency,
        source_type=pay_src,
        availability_status=pay_stat,
        cost_grain=pay_grain,
        is_estimated=(pay_src == CostSourceType.CATALOG_ESTIMATE),
        is_assumed=(pay_src == CostSourceType.CONFIGURED_ASSUMPTION),
        included_in_known_contribution=(pay_amt is not None),
        included_in_final_contribution=(pay_amt is not None and satisfies_completeness(pay_src)),
        source_reference=pay_ref,
        description=pay_desc,
    )

    # 4. PACKAGING_COST
    raw_pkg = safe_float(get_val("packaging_cost"))
    if raw_pkg is not None:
        pkg_amt = raw_pkg
        pkg_unit = pkg_amt / qty if qty and qty > 0 else None
        pkg_src = CostSourceType.SOURCE_DATA
        pkg_stat = CostComponentStatus.AVAILABLE_SOURCE
        pkg_grain = CostGrain.TRANSACTION
        pkg_ref = "sales.packaging_cost"
        pkg_desc = "Observed packaging materials cost"
    else:
        has_pkg = False
        assumed_pkg = 0.0
        if asm.packaging_cost_per_unit is not None and qty is not None:
            assumed_pkg += asm.packaging_cost_per_unit * qty
            has_pkg = True
        if asm.packaging_cost_per_order is not None:
            assumed_pkg += asm.packaging_cost_per_order * allocation_ratio
            has_pkg = True

        if has_pkg:
            pkg_amt = float(assumed_pkg)
            pkg_unit = pkg_amt / qty if qty and qty > 0 else None
            pkg_src = CostSourceType.CONFIGURED_ASSUMPTION
            pkg_stat = CostComponentStatus.AVAILABLE_ASSUMED
            pkg_grain = CostGrain.UNIT if asm.packaging_cost_per_order is None else CostGrain.ORDER
            pkg_ref = "config.assumptions.packaging"
            pkg_desc = "Configured packaging materials assumption"
        else:
            pkg_amt = None
            pkg_unit = None
            pkg_src = CostSourceType.UNAVAILABLE
            pkg_stat = CostComponentStatus.UNAVAILABLE
            pkg_grain = CostGrain.UNKNOWN
            pkg_ref = None
            pkg_desc = "Packaging cost unavailable"

    details[CostComponent.PACKAGING_COST.value] = OperationalCostDetail(
        component=CostComponent.PACKAGING_COST,
        amount=pkg_amt,
        unit_amount=pkg_unit,
        currency=currency,
        source_type=pkg_src,
        availability_status=pkg_stat,
        cost_grain=pkg_grain,
        is_estimated=(pkg_src == CostSourceType.CATALOG_ESTIMATE),
        is_assumed=(pkg_src == CostSourceType.CONFIGURED_ASSUMPTION),
        included_in_known_contribution=(pkg_amt is not None),
        included_in_final_contribution=(pkg_amt is not None and satisfies_completeness(pkg_src)),
        source_reference=pkg_ref,
        description=pkg_desc,
    )

    # 5. WAREHOUSE_HANDLING_COST
    raw_wh = safe_float(get_val("warehouse_handling_cost", "handling_cost"))
    if raw_wh is not None:
        wh_amt = raw_wh
        wh_unit = wh_amt / qty if qty and qty > 0 else None
        wh_src = CostSourceType.SOURCE_DATA
        wh_stat = CostComponentStatus.AVAILABLE_SOURCE
        wh_grain = CostGrain.TRANSACTION
        wh_ref = "sales.warehouse_handling_cost"
        wh_desc = "Observed warehouse pick/pack handling cost"
    else:
        has_wh = False
        assumed_wh = 0.0

        # Warehouse specific rate or unit default
        rate = asm.warehouse_handling_by_facility.get(warehouse_id, asm.warehouse_handling_cost_per_unit)
        if rate is not None and qty is not None:
            assumed_wh += rate * qty
            has_wh = True

        if asm.warehouse_handling_cost_per_order is not None:
            assumed_wh += asm.warehouse_handling_cost_per_order * allocation_ratio
            has_wh = True

        if has_wh:
            wh_amt = float(assumed_wh)
            wh_unit = wh_amt / qty if qty and qty > 0 else None
            wh_src = CostSourceType.CONFIGURED_ASSUMPTION
            wh_stat = CostComponentStatus.AVAILABLE_ASSUMED
            wh_grain = CostGrain.UNIT if asm.warehouse_handling_cost_per_order is None else CostGrain.ORDER
            wh_ref = f"config.assumptions.handling[{warehouse_id}]"
            wh_desc = "Configured warehouse handling labor assumption"
        else:
            wh_amt = None
            wh_unit = None
            wh_src = CostSourceType.UNAVAILABLE
            wh_stat = CostComponentStatus.UNAVAILABLE
            wh_grain = CostGrain.UNKNOWN
            wh_ref = None
            wh_desc = "Warehouse handling cost unavailable"

    details[CostComponent.WAREHOUSE_HANDLING_COST.value] = OperationalCostDetail(
        component=CostComponent.WAREHOUSE_HANDLING_COST,
        amount=wh_amt,
        unit_amount=wh_unit,
        currency=currency,
        source_type=wh_src,
        availability_status=wh_stat,
        cost_grain=wh_grain,
        is_estimated=(wh_src == CostSourceType.CATALOG_ESTIMATE),
        is_assumed=(wh_src == CostSourceType.CONFIGURED_ASSUMPTION),
        included_in_known_contribution=(wh_amt is not None),
        included_in_final_contribution=(wh_amt is not None and satisfies_completeness(wh_src)),
        source_reference=wh_ref,
        description=wh_desc,
    )

    # 6. RETURN_PROCESSING_COST
    raw_ret = safe_float(get_val("return_processing_cost"))
    has_returns_dataset = bool(ret_ctx.get("has_returns_dataset", False))
    returned_units = safe_int(ret_ctx.get("returned_units", 0)) or 0
    return_events = safe_int(ret_ctx.get("return_count", 0)) or 0
    has_fee_assumption = (asm.return_cost_per_returned_unit is not None) or (asm.return_cost_per_return_event is not None)

    if raw_ret is not None:
        ret_amt = raw_ret
        ret_unit = ret_amt / qty if qty and qty > 0 else None
        ret_src = CostSourceType.SOURCE_DATA
        ret_stat = CostComponentStatus.AVAILABLE_SOURCE
        ret_grain = CostGrain.TRANSACTION
        ret_ref = "sales.return_processing_cost"
        ret_desc = "Observed return processing cost"
    elif has_returns_dataset and has_fee_assumption:
        if returned_units > 0 or return_events > 0:
            # A return event occurred for this line
            ret_fee = 0.0
            if asm.return_cost_per_returned_unit is not None:
                ret_fee += asm.return_cost_per_returned_unit * returned_units
            if asm.return_cost_per_return_event is not None and return_events > 0:
                ret_fee += asm.return_cost_per_return_event * return_events

            ret_amt = float(ret_fee)
            ret_unit = ret_amt / qty if qty and qty > 0 else None
            ret_src = CostSourceType.CONFIGURED_ASSUMPTION
            ret_stat = CostComponentStatus.AVAILABLE_ASSUMED
            ret_grain = CostGrain.RETURN_EVENT
            ret_ref = "returns_df:assumed_fee"
            ret_desc = f"Return event observed ({returned_units} returned units) with configured return fee"
        else:
            # Verified no returns in returns dataset under configured return fee policy
            ret_amt = 0.0
            ret_unit = 0.0
            ret_src = CostSourceType.CONFIGURED_ASSUMPTION
            ret_stat = CostComponentStatus.AVAILABLE_ASSUMED
            ret_grain = CostGrain.TRANSACTION
            ret_ref = "returns_df:no_return"
            ret_desc = "Verified zero returns in returns repository under configured return fee policy"
    else:
        ret_amt = None
        ret_unit = None
        ret_src = CostSourceType.UNAVAILABLE
        ret_stat = CostComponentStatus.UNAVAILABLE
        ret_grain = CostGrain.UNKNOWN
        ret_ref = None
        ret_desc = "Return processing cost unavailable (no source monetary cost and no configured return fee rate)"

    details[CostComponent.RETURN_PROCESSING_COST.value] = OperationalCostDetail(
        component=CostComponent.RETURN_PROCESSING_COST,
        amount=ret_amt,
        unit_amount=ret_unit,
        currency=currency,
        source_type=ret_src,
        availability_status=ret_stat,
        cost_grain=ret_grain,
        is_estimated=(ret_src == CostSourceType.CATALOG_ESTIMATE),
        is_assumed=(ret_src == CostSourceType.CONFIGURED_ASSUMPTION),
        included_in_known_contribution=(ret_amt is not None),
        included_in_final_contribution=(ret_amt is not None and satisfies_completeness(ret_src)),
        source_reference=ret_ref,
        description=ret_desc,
    )

    # 7. OTHER_VARIABLE_COST
    raw_oth = safe_float(get_val("other_variable_cost"))
    if raw_oth is not None:
        oth_amt = raw_oth
        oth_unit = oth_amt / qty if qty and qty > 0 else None
        oth_src = CostSourceType.SOURCE_DATA
        oth_stat = CostComponentStatus.AVAILABLE_SOURCE
        oth_grain = CostGrain.TRANSACTION
        oth_ref = "sales.other_variable_cost"
        oth_desc = "Observed other variable cost"
    else:
        has_oth = False
        assumed_oth = 0.0
        if asm.other_variable_cost_per_unit is not None and qty is not None:
            assumed_oth += asm.other_variable_cost_per_unit * qty
            has_oth = True
        if asm.other_variable_cost_pct is not None and net_rev is not None and net_rev > 0:
            assumed_oth += asm.other_variable_cost_pct * net_rev
            has_oth = True

        if has_oth:
            oth_amt = float(assumed_oth)
            oth_unit = oth_amt / qty if qty and qty > 0 else None
            oth_src = CostSourceType.CONFIGURED_ASSUMPTION
            oth_stat = CostComponentStatus.AVAILABLE_ASSUMED
            oth_grain = CostGrain.UNIT
            oth_ref = "config.assumptions.other"
            oth_desc = "Configured other variable cost assumption"
        else:
            oth_amt = None
            oth_unit = None
            oth_src = CostSourceType.UNAVAILABLE
            oth_stat = CostComponentStatus.UNAVAILABLE
            oth_grain = CostGrain.UNKNOWN
            oth_ref = None
            oth_desc = "Other variable cost unavailable"

    details[CostComponent.OTHER_VARIABLE_COST.value] = OperationalCostDetail(
        component=CostComponent.OTHER_VARIABLE_COST,
        amount=oth_amt,
        unit_amount=oth_unit,
        currency=currency,
        source_type=oth_src,
        availability_status=oth_stat,
        cost_grain=oth_grain,
        is_estimated=(oth_src == CostSourceType.CATALOG_ESTIMATE),
        is_assumed=(oth_src == CostSourceType.CONFIGURED_ASSUMPTION),
        included_in_known_contribution=(oth_amt is not None),
        included_in_final_contribution=(oth_amt is not None and satisfies_completeness(oth_src)),
        source_reference=oth_ref,
        description=oth_desc,
    )

    return details


def calculate_operational_economics_record(
    row: Union[pd.Series, Dict[str, Any]],
    unit_cost: Optional[float] = None,
    order_context: Optional[Dict[str, Any]] = None,
    return_context: Optional[Dict[str, Any]] = None,
    config: Optional[OperationalEconomicsConfig] = None,
) -> OperationalEconomicsRecord:
    """Calculate deterministic operational economics record for an individual sales transaction line."""
    cfg = config or OperationalEconomicsConfig()

    def get_val(key: str, alt_key: Optional[str] = None) -> Any:
        if isinstance(row, dict):
            val = row.get(key)
            if val is None and alt_key:
                val = row.get(alt_key)
            return val
        if key in row:
            return row[key]
        if alt_key and alt_key in row:
            return row[alt_key]
        return None

    def safe_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
            return None if np.isnan(f) or np.isinf(f) else f
        except (ValueError, TypeError):
            return None

    def safe_int(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            f = float(v)
            return None if np.isnan(f) or np.isinf(f) else int(f)
        except (ValueError, TypeError):
            return None

    tx_id = str(get_val("transaction_id", "sale_id") or "").strip()
    order_id = str(get_val("order_id") or "").strip()
    date_val = str(get_val("date", "sale_date") or "").strip()
    sku_id = str(get_val("sku_id") or "").strip()
    warehouse_id = str(get_val("warehouse_id") or "").strip()
    channel_id = str(get_val("channel_id") or "").strip()
    currency = str(get_val("currency") or cfg.default_currency).strip().upper()

    qty = safe_int(get_val("quantity"))
    unit_price = safe_float(get_val("unit_price"))
    discount = safe_float(get_val("discount"))
    source_revenue = safe_float(get_val("revenue", "source_revenue"))

    record_id = generate_operational_record_id(tx_id, sku_id, order_id, date_val, currency)

    # 1. Validation & INVALID detection
    invalid_reasons: List[str] = []
    if not sku_id:
        invalid_reasons.append("Missing SKU identifier")
    if qty is not None and qty < 0:
        invalid_reasons.append(f"Negative quantity: {qty}")
    if unit_price is not None and unit_price < 0:
        invalid_reasons.append(f"Negative unit price: {unit_price}")
    if discount is not None and discount < 0:
        invalid_reasons.append(f"Negative discount: {discount}")

    if invalid_reasons:
        return OperationalEconomicsRecord(
            record_id=record_id,
            transaction_id=tx_id,
            order_id=order_id,
            order_date=date_val,
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            channel_id=channel_id,
            currency=currency,
            quantity=qty,
            unit_price=unit_price,
            gross_revenue=None,
            discount=discount,
            net_revenue=None,
            unit_product_cost=None,
            product_cost=None,
            gross_margin=None,
            gross_margin_pct=None,
            shipping_cost=None,
            payment_processing_cost=None,
            packaging_cost=None,
            warehouse_handling_cost=None,
            return_processing_cost=None,
            other_variable_cost=None,
            total_known_variable_cost=0.0,
            known_contribution_margin=None,
            known_contribution_margin_pct=None,
            final_contribution_margin=None,
            final_contribution_margin_pct=None,
            final_contribution_margin_status=ContributionMarginStatus.NOT_CALCULABLE,
            net_revenue_per_unit=None,
            gross_margin_per_unit=None,
            variable_cost_per_unit=None,
            contribution_margin_per_unit=None,
            cost_completeness_pct=0.0,
            cost_completeness_status=CostCompletenessStatus.INSUFFICIENT_DATA,
            economics_status=OperationalEconomicsStatus.INVALID_INPUT,
            status_rationale=f"Invalid transaction line data: {'; '.join(invalid_reasons)}",
            cost_details={},
        )

    # 2. Revenue calculations
    gross_rev = (qty * unit_price) if (qty is not None and unit_price is not None) else None
    effective_discount = discount if discount is not None else 0.0
    net_rev = (gross_rev - effective_discount) if gross_rev is not None else source_revenue

    # 3. Cost resolution
    cost_details = resolve_transaction_operational_costs(
        row=row,
        unit_cost=unit_cost,
        order_context=order_context,
        return_context=return_context,
        config=cfg,
    )

    p_detail = cost_details.get(CostComponent.PRODUCT_COST.value)
    s_detail = cost_details.get(CostComponent.SHIPPING_COST.value)
    pay_detail = cost_details.get(CostComponent.PAYMENT_PROCESSING_COST.value)
    pkg_detail = cost_details.get(CostComponent.PACKAGING_COST.value)
    wh_detail = cost_details.get(CostComponent.WAREHOUSE_HANDLING_COST.value)
    ret_detail = cost_details.get(CostComponent.RETURN_PROCESSING_COST.value)
    oth_detail = cost_details.get(CostComponent.OTHER_VARIABLE_COST.value)

    prod_cost = p_detail.amount if p_detail else None
    unit_prod_cost = p_detail.unit_amount if p_detail else None

    ship_cost = s_detail.amount if s_detail else None
    pay_cost = pay_detail.amount if pay_detail else None
    pkg_cost = pkg_detail.amount if pkg_detail else None
    wh_cost = wh_detail.amount if wh_detail else None
    ret_cost = ret_detail.amount if ret_detail else None
    oth_cost = oth_detail.amount if oth_detail else None

    var_costs = [ship_cost, pay_cost, pkg_cost, wh_cost, ret_cost, oth_cost]
    total_known_var = sum(c for c in var_costs if c is not None)

    # Gross Margin
    gross_margin = (net_rev - prod_cost) if (net_rev is not None and prod_cost is not None) else None
    gross_margin_pct = (gross_margin / net_rev) if (gross_margin is not None and net_rev is not None and net_rev > 0) else None

    # Known Contribution Margin
    known_cm = (gross_margin - total_known_var) if gross_margin is not None else None
    known_cm_pct = (known_cm / net_rev) if (known_cm is not None and net_rev is not None and net_rev > 0) else None

    # 4. Completeness check
    comp_report = calculate_cost_completeness_report(cost_details, cfg)
    meets_comp = comp_report.completeness_pct >= cfg.min_completeness_threshold

    if gross_margin is not None:
        if meets_comp:
            final_cm = known_cm
            final_cm_pct = known_cm_pct
            cm_status = ContributionMarginStatus.CALCULABLE
            econ_status = OperationalEconomicsStatus.COMPLETE
            rationale = "Complete operational economics: all required cost components satisfied"
        else:
            final_cm = None
            final_cm_pct = None
            cm_status = ContributionMarginStatus.INSUFFICIENT_COST_DATA
            econ_status = OperationalEconomicsStatus.PARTIALLY_CALCULABLE
            rationale = "Gross margin calculable; required operational costs incomplete"
    else:
        final_cm = None
        final_cm_pct = None
        cm_status = ContributionMarginStatus.INSUFFICIENT_COST_DATA
        econ_status = OperationalEconomicsStatus.INSUFFICIENT_COST_DATA
        rationale = "Product procurement cost is missing or unobserved"

    # 5. Per-unit metrics
    net_rev_per_unit = (net_rev / qty) if (qty is not None and qty > 0 and net_rev is not None) else None
    gm_per_unit = (gross_margin / qty) if (qty is not None and qty > 0 and gross_margin is not None) else None
    var_per_unit = (total_known_var / qty) if (qty is not None and qty > 0) else None
    cm_per_unit = (known_cm / qty) if (qty is not None and qty > 0 and known_cm is not None) else None

    return OperationalEconomicsRecord(
        record_id=record_id,
        transaction_id=tx_id,
        order_id=order_id,
        order_date=date_val,
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        channel_id=channel_id,
        currency=currency,
        quantity=qty,
        unit_price=unit_price,
        gross_revenue=gross_rev,
        discount=discount,
        net_revenue=net_rev,
        unit_product_cost=unit_prod_cost,
        product_cost=prod_cost,
        gross_margin=gross_margin,
        gross_margin_pct=gross_margin_pct,
        shipping_cost=ship_cost,
        payment_processing_cost=pay_cost,
        packaging_cost=pkg_cost,
        warehouse_handling_cost=wh_cost,
        return_processing_cost=ret_cost,
        other_variable_cost=oth_cost,
        total_known_variable_cost=total_known_var,
        known_contribution_margin=known_cm,
        known_contribution_margin_pct=known_cm_pct,
        final_contribution_margin=final_cm,
        final_contribution_margin_pct=final_cm_pct,
        final_contribution_margin_status=cm_status,
        net_revenue_per_unit=net_rev_per_unit,
        gross_margin_per_unit=gm_per_unit,
        variable_cost_per_unit=var_per_unit,
        contribution_margin_per_unit=cm_per_unit,
        cost_completeness_pct=comp_report.completeness_pct,
        cost_completeness_status=comp_report.completeness_status,
        economics_status=econ_status,
        status_rationale=rationale,
        cost_details=cost_details,
    )


# =============================================================================
# Completeness Auditing
# =============================================================================


def calculate_cost_completeness_report(
    details: Dict[str, OperationalCostDetail],
    config: Optional[OperationalEconomicsConfig] = None,
) -> CostCompletenessReport:
    """Evaluate deterministic cost completeness report for a cost details dictionary."""
    cfg = config or OperationalEconomicsConfig()
    req_keys = [c.value if isinstance(c, CostComponent) else str(c) for c in cfg.required_cost_components]

    available: List[str] = []
    unavailable: List[str] = []
    source: List[str] = []
    estimated: List[str] = []
    assumed: List[str] = []
    notes: List[str] = []

    def satisfies_completeness(src: CostSourceType) -> bool:
        if cfg.strict_source_only:
            return src == CostSourceType.SOURCE_DATA
        if src == CostSourceType.SOURCE_DATA:
            return True
        if src == CostSourceType.CATALOG_ESTIMATE and cfg.allow_estimated_costs_for_completeness:
            return True
        if src == CostSourceType.CONFIGURED_ASSUMPTION and cfg.allow_assumed_costs_for_completeness:
            return True
        return False

    for comp_name, d in details.items():
        if d.amount is not None:
            if d.source_type == CostSourceType.SOURCE_DATA:
                source.append(comp_name)
            elif d.source_type == CostSourceType.CATALOG_ESTIMATE:
                estimated.append(comp_name)
            elif d.source_type == CostSourceType.CONFIGURED_ASSUMPTION:
                assumed.append(comp_name)

            if satisfies_completeness(d.source_type):
                available.append(comp_name)
            else:
                unavailable.append(comp_name)
                notes.append(
                    f"{comp_name} is present via {d.source_type.value} but disallowed by completeness policy"
                )
        else:
            unavailable.append(comp_name)

    # Check required components
    avail_req_count = sum(1 for k in req_keys if k in available)
    tot_req = len(req_keys)
    comp_pct = (avail_req_count / tot_req * 100.0) if tot_req > 0 else 100.0
    missing_req = tot_req - avail_req_count

    if comp_pct >= 100.0:
        status = CostCompletenessStatus.COMPLETE
    elif comp_pct > 0.0:
        status = CostCompletenessStatus.PARTIALLY_COMPLETE
    else:
        status = CostCompletenessStatus.INSUFFICIENT_DATA

    return CostCompletenessReport(
        completeness_status=status,
        completeness_pct=round(comp_pct, 2),
        total_required_components=tot_req,
        available_required_components=avail_req_count,
        missing_required_components=missing_req,
        required_components=req_keys,
        available_components=available,
        unavailable_components=unavailable,
        source_components=source,
        estimated_components=estimated,
        assumed_components=assumed,
        audit_notes=notes,
    )


# =============================================================================
# Vectorized DataFrame Computation
# =============================================================================


def compute_operational_economics_dataframe(
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
    config: Optional[OperationalEconomicsConfig] = None,
) -> pd.DataFrame:
    """Vectorized calculation of operational economics and cost completeness.

    Optimized for high-volume transactions (773k+ records in ~1 second).
    Multi-line order costs are allocated proportionally without duplication.
    """
    cfg = config or OperationalEconomicsConfig()
    asm = cfg.assumptions

    if sales_df.empty:
        return pd.DataFrame()

    df = sales_df.copy()

    # Point-in-time filtering
    if cfg.as_of_date is not None:
        df = filter_sales_by_as_of_date(df, cfg.as_of_date)
        if df.empty:
            return pd.DataFrame()

    # Normalize column names & required fields
    if "date" not in df.columns and "sale_date" in df.columns:
        df["date"] = df["sale_date"]
    if "date" not in df.columns:
        df["date"] = "1970-01-01"
    if "sale_id" not in df.columns and "transaction_id" in df.columns:
        df["sale_id"] = df["transaction_id"]
    if "transaction_id" not in df.columns and "sale_id" in df.columns:
        df["transaction_id"] = df["sale_id"]
    if "transaction_id" not in df.columns:
        df["transaction_id"] = df.index.astype(str)
    if "sale_id" not in df.columns:
        df["sale_id"] = df["transaction_id"]
    if "order_id" not in df.columns:
        df["order_id"] = df["transaction_id"]
    if "channel_id" not in df.columns:
        df["channel_id"] = "DEFAULT_CHANNEL"
    if "warehouse_id" not in df.columns:
        df["warehouse_id"] = "DEFAULT_WAREHOUSE"
    if "currency" not in df.columns:
        df["currency"] = cfg.default_currency

    qty = pd.to_numeric(df["quantity"], errors="coerce")
    price = pd.to_numeric(df["unit_price"], errors="coerce")
    disc = pd.to_numeric(df.get("discount", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)

    # Input validation flags
    is_invalid = (
        df["sku_id"].isna()
        | (df["sku_id"].astype(str).str.strip() == "")
        | (qty.notna() & (qty < 0))
        | (price.notna() & (price < 0))
        | (disc.notna() & (disc < 0))
    )

    # Core revenues
    gross_rev = qty * price
    gross_rev.loc[is_invalid] = np.nan

    if "revenue" in df.columns:
        net_rev = pd.to_numeric(df["revenue"], errors="coerce")
    else:
        net_rev = gross_rev - disc
    net_rev.loc[is_invalid] = np.nan

    # Order-level context for multi-line allocations
    has_orders = "order_id" in df.columns
    if has_orders:
        order_grouped = df.groupby("order_id", sort=False)
        order_line_count = order_grouped["order_id"].transform("count")
        order_total_net_rev = df.groupby("order_id", sort=False)["revenue"].transform("sum") if "revenue" in df.columns else df.assign(_net=net_rev).groupby("order_id", sort=False)["_net"].transform("sum")
        order_total_qty = order_grouped["quantity"].transform("sum")
    else:
        order_line_count = pd.Series(1, index=df.index)
        order_total_net_rev = net_rev
        order_total_qty = qty

    # Allocation weights for order-level costs
    is_multiline = order_line_count > 1
    alloc_weight = pd.Series(1.0, index=df.index, dtype=float)

    if is_multiline.any():
        if cfg.order_cost_allocation_method == "NET_REVENUE":
            # Net revenue proportion where order net rev > 0
            has_pos_order_net = is_multiline & (order_total_net_rev > 0) & net_rev.notna() & (net_rev >= 0)
            alloc_weight.loc[has_pos_order_net] = net_rev.loc[has_pos_order_net] / order_total_net_rev.loc[has_pos_order_net]

            # Fallback to quantity where order net rev <= 0 but order qty > 0
            fallback_qty = is_multiline & (~has_pos_order_net) & (order_total_qty > 0) & qty.notna() & (qty > 0)
            alloc_weight.loc[fallback_qty] = qty.loc[fallback_qty] / order_total_qty.loc[fallback_qty]

            # Fallback to equal split
            fallback_equal = is_multiline & (~has_pos_order_net) & (~fallback_qty)
            alloc_weight.loc[fallback_equal] = 1.0 / order_line_count.loc[fallback_equal]
        elif cfg.order_cost_allocation_method == "QUANTITY":
            has_pos_qty = is_multiline & (order_total_qty > 0) & qty.notna() & (qty > 0)
            alloc_weight.loc[has_pos_qty] = qty.loc[has_pos_qty] / order_total_qty.loc[has_pos_qty]
            fallback_equal = is_multiline & (~has_pos_qty)
            alloc_weight.loc[fallback_equal] = 1.0 / order_line_count.loc[fallback_equal]
        else:  # EQUAL
            alloc_weight.loc[is_multiline] = 1.0 / order_line_count.loc[is_multiline]

    # Ensure weights are bounded [0, 1]
    alloc_weight = alloc_weight.clip(lower=0.0, upper=1.0)

    # ---------------------------------------------------------
    # 1. PRODUCT_COST
    # ---------------------------------------------------------
    prod_cost = pd.Series(np.nan, index=df.index, dtype=float)
    unit_prod_cost = pd.Series(np.nan, index=df.index, dtype=float)
    prod_src_type = pd.Series(CostSourceType.UNAVAILABLE.value, index=df.index, dtype=object)

    if "product_cost" in df.columns:
        src_pc = pd.to_numeric(df["product_cost"], errors="coerce")
        has_src_pc = src_pc.notna()
        prod_cost.loc[has_src_pc] = src_pc.loc[has_src_pc]
        pos_qty = has_src_pc & (qty > 0)
        unit_prod_cost.loc[pos_qty] = src_pc.loc[pos_qty] / qty.loc[pos_qty]
        prod_src_type.loc[has_src_pc] = CostSourceType.SOURCE_DATA.value

    # Catalog estimate fallback
    missing_pc = prod_cost.isna()
    if missing_pc.any() and products_df is not None and not products_df.empty:
        prod_map = products_df.set_index("sku_id")["unit_cost"].to_dict()
        cat_cost = df["sku_id"].map(prod_map)
        cat_cost = pd.to_numeric(cat_cost, errors="coerce")
        has_cat = missing_pc & cat_cost.notna() & (cat_cost >= 0)
        unit_prod_cost.loc[has_cat] = cat_cost.loc[has_cat]
        prod_cost.loc[has_cat] = cat_cost.loc[has_cat] * qty.loc[has_cat]
        prod_src_type.loc[has_cat] = CostSourceType.CATALOG_ESTIMATE.value

    # Direct unit_cost column fallback
    missing_pc = prod_cost.isna()
    if missing_pc.any() and "unit_cost" in df.columns:
        direct_uc = pd.to_numeric(df["unit_cost"], errors="coerce")
        has_duc = missing_pc & direct_uc.notna() & (direct_uc >= 0)
        unit_prod_cost.loc[has_duc] = direct_uc.loc[has_duc]
        prod_cost.loc[has_duc] = direct_uc.loc[has_duc] * qty.loc[has_duc]
        prod_src_type.loc[has_duc] = CostSourceType.CATALOG_ESTIMATE.value

    # ---------------------------------------------------------
    # 2. SHIPPING_COST
    # ---------------------------------------------------------
    ship_cost = pd.Series(np.nan, index=df.index, dtype=float)
    ship_src_type = pd.Series(CostSourceType.UNAVAILABLE.value, index=df.index, dtype=object)

    if "shipping_cost" in df.columns:
        src_sc = pd.to_numeric(df["shipping_cost"], errors="coerce")
        has_src_sc = src_sc.notna()
        ship_cost.loc[has_src_sc] = src_sc.loc[has_src_sc]
        ship_src_type.loc[has_src_sc] = CostSourceType.SOURCE_DATA.value

    missing_sc = ship_cost.isna()
    if missing_sc.any():
        assumed_s = pd.Series(0.0, index=df.index, dtype=float)
        has_s_asm = False

        if asm.shipping_cost_per_unit is not None:
            assumed_s += asm.shipping_cost_per_unit * qty.fillna(0.0)
            has_s_asm = True

        if asm.shipping_pct_of_net_revenue is not None:
            assumed_s += asm.shipping_pct_of_net_revenue * net_rev.clip(lower=0.0).fillna(0.0)
            has_s_asm = True

        if asm.shipping_cost_by_channel:
            chan_rates = df["channel_id"].map(asm.shipping_cost_by_channel).fillna(0.0)
            assumed_s += chan_rates * alloc_weight
            has_s_asm = True
        elif asm.shipping_cost_per_order is not None:
            assumed_s += asm.shipping_cost_per_order * alloc_weight
            has_s_asm = True

        if has_s_asm:
            ship_cost.loc[missing_sc] = assumed_s.loc[missing_sc]
            ship_src_type.loc[missing_sc] = CostSourceType.CONFIGURED_ASSUMPTION.value

    # ---------------------------------------------------------
    # 3. PAYMENT_PROCESSING_COST
    # ---------------------------------------------------------
    pay_cost = pd.Series(np.nan, index=df.index, dtype=float)
    pay_src_type = pd.Series(CostSourceType.UNAVAILABLE.value, index=df.index, dtype=object)

    pay_col = "payment_processing_cost" if "payment_processing_cost" in df.columns else ("payment_fee" if "payment_fee" in df.columns else None)
    if pay_col is not None:
        src_pay = pd.to_numeric(df[pay_col], errors="coerce")
        has_src_pay = src_pay.notna()
        pay_cost.loc[has_src_pay] = src_pay.loc[has_src_pay]
        pay_src_type.loc[has_src_pay] = CostSourceType.SOURCE_DATA.value

    missing_pay = pay_cost.isna()
    if missing_pay.any():
        rates = df["channel_id"].map(asm.payment_rate_by_channel)
        if asm.payment_processing_rate is not None:
            rates = rates.fillna(asm.payment_processing_rate)

        fixed = df["channel_id"].map(asm.payment_fixed_by_channel)
        if asm.payment_processing_fixed_per_order is not None:
            fixed = fixed.fillna(asm.payment_processing_fixed_per_order)

        if rates.notna().any() or fixed.notna().any():
            fee = pd.Series(0.0, index=df.index, dtype=float)
            has_r = rates.notna() & (net_rev > 0)
            if has_r.any():
                fee.loc[has_r] += rates.loc[has_r] * net_rev.loc[has_r]
            has_f = fixed.notna()
            if has_f.any():
                fee.loc[has_f] += fixed.loc[has_f] * alloc_weight.loc[has_f]

            pay_cost.loc[missing_pay] = fee.loc[missing_pay]
            pay_src_type.loc[missing_pay] = CostSourceType.CONFIGURED_ASSUMPTION.value

    # ---------------------------------------------------------
    # 4. PACKAGING_COST
    # ---------------------------------------------------------
    pkg_cost = pd.Series(np.nan, index=df.index, dtype=float)
    pkg_src_type = pd.Series(CostSourceType.UNAVAILABLE.value, index=df.index, dtype=object)

    if "packaging_cost" in df.columns:
        src_pkg = pd.to_numeric(df["packaging_cost"], errors="coerce")
        has_src_pkg = src_pkg.notna()
        pkg_cost.loc[has_src_pkg] = src_pkg.loc[has_src_pkg]
        pkg_src_type.loc[has_src_pkg] = CostSourceType.SOURCE_DATA.value

    missing_pkg = pkg_cost.isna()
    if missing_pkg.any():
        assumed_pkg = pd.Series(0.0, index=df.index, dtype=float)
        has_pkg_asm = False

        if asm.packaging_cost_per_unit is not None:
            assumed_pkg += asm.packaging_cost_per_unit * qty.fillna(0.0)
            has_pkg_asm = True
        if asm.packaging_cost_per_order is not None:
            assumed_pkg += asm.packaging_cost_per_order * alloc_weight
            has_pkg_asm = True

        if has_pkg_asm:
            pkg_cost.loc[missing_pkg] = assumed_pkg.loc[missing_pkg]
            pkg_src_type.loc[missing_pkg] = CostSourceType.CONFIGURED_ASSUMPTION.value

    # ---------------------------------------------------------
    # 5. WAREHOUSE_HANDLING_COST
    # ---------------------------------------------------------
    wh_cost = pd.Series(np.nan, index=df.index, dtype=float)
    wh_src_type = pd.Series(CostSourceType.UNAVAILABLE.value, index=df.index, dtype=object)

    wh_col = "warehouse_handling_cost" if "warehouse_handling_cost" in df.columns else ("handling_cost" if "handling_cost" in df.columns else None)
    if wh_col is not None:
        src_wh = pd.to_numeric(df[wh_col], errors="coerce")
        has_src_wh = src_wh.notna()
        wh_cost.loc[has_src_wh] = src_wh.loc[has_src_wh]
        wh_src_type.loc[has_src_wh] = CostSourceType.SOURCE_DATA.value

    missing_wh = wh_cost.isna()
    if missing_wh.any():
        assumed_wh = pd.Series(0.0, index=df.index, dtype=float)
        has_wh_asm = False

        wh_rates = df["warehouse_id"].map(asm.warehouse_handling_by_facility)
        if asm.warehouse_handling_cost_per_unit is not None:
            wh_rates = wh_rates.fillna(asm.warehouse_handling_cost_per_unit)

        if wh_rates.notna().any():
            assumed_wh += wh_rates.fillna(0.0) * qty.fillna(0.0)
            has_wh_asm = True

        if asm.warehouse_handling_cost_per_order is not None:
            assumed_wh += asm.warehouse_handling_cost_per_order * alloc_weight
            has_wh_asm = True

        if has_wh_asm:
            wh_cost.loc[missing_wh] = assumed_wh.loc[missing_wh]
            wh_src_type.loc[missing_wh] = CostSourceType.CONFIGURED_ASSUMPTION.value

    # ---------------------------------------------------------
    # 6. RETURN_PROCESSING_COST
    # ---------------------------------------------------------
    ret_cost = pd.Series(np.nan, index=df.index, dtype=float)
    ret_src_type = pd.Series(CostSourceType.UNAVAILABLE.value, index=df.index, dtype=object)

    if "return_processing_cost" in df.columns:
        src_ret = pd.to_numeric(df["return_processing_cost"], errors="coerce")
        has_src_ret = src_ret.notna()
        ret_cost.loc[has_src_ret] = src_ret.loc[has_src_ret]
        ret_src_type.loc[has_src_ret] = CostSourceType.SOURCE_DATA.value

    missing_ret = ret_cost.isna()
    has_ret_fee_asm = (asm.return_cost_per_returned_unit is not None) or (asm.return_cost_per_return_event is not None)

    if missing_ret.any() and returns_df is not None and has_ret_fee_asm:
        rdf = returns_df.copy()
        if cfg.as_of_date is not None and "return_date" in rdf.columns:
            rdf = rdf[rdf["return_date"].astype(str) <= cfg.as_of_date]

        # Aggregate returns by order_id and sku_id
        if not rdf.empty and "order_id" in rdf.columns and "sku_id" in rdf.columns:
            ret_agg = rdf.groupby(["order_id", "sku_id"], as_index=False).agg(
                ret_qty=("quantity", "sum"),
                ret_events=("order_id", "count"),
            )
            # Create a composite join key
            df["_ret_key"] = df["order_id"].astype(str) + "|" + df["sku_id"].astype(str)
            ret_agg["_ret_key"] = ret_agg["order_id"].astype(str) + "|" + ret_agg["sku_id"].astype(str)
            ret_map = ret_agg.set_index("_ret_key")

            mapped_ret_qty = df["_ret_key"].map(ret_map["ret_qty"]).fillna(0).astype(int)
            mapped_ret_ev = df["_ret_key"].map(ret_map["ret_events"]).fillna(0).astype(int)

            has_return_event = (mapped_ret_qty > 0) | (mapped_ret_ev > 0)

            # Lines WITH return event
            lines_with_ret = missing_ret & has_return_event
            if lines_with_ret.any():
                fee = pd.Series(0.0, index=df.index, dtype=float)
                if asm.return_cost_per_returned_unit is not None:
                    fee += asm.return_cost_per_returned_unit * mapped_ret_qty
                if asm.return_cost_per_return_event is not None:
                    fee += asm.return_cost_per_return_event * mapped_ret_ev

                ret_cost.loc[lines_with_ret] = fee.loc[lines_with_ret]
                ret_src_type.loc[lines_with_ret] = CostSourceType.CONFIGURED_ASSUMPTION.value

            # Lines WITHOUT return event in verified returns dataset under configured fee policy -> 0.0
            lines_no_ret = missing_ret & (~has_return_event)
            ret_cost.loc[lines_no_ret] = 0.0
            ret_src_type.loc[lines_no_ret] = CostSourceType.CONFIGURED_ASSUMPTION.value

            df.drop(columns=["_ret_key"], inplace=True)
        else:
            # Empty returns repository -> all lines have 0 return cost under configured fee policy
            ret_cost.loc[missing_ret] = 0.0
            ret_src_type.loc[missing_ret] = CostSourceType.CONFIGURED_ASSUMPTION.value

    # ---------------------------------------------------------
    # 7. OTHER_VARIABLE_COST
    # ---------------------------------------------------------
    oth_cost = pd.Series(np.nan, index=df.index, dtype=float)
    oth_src_type = pd.Series(CostSourceType.UNAVAILABLE.value, index=df.index, dtype=object)

    if "other_variable_cost" in df.columns:
        src_oth = pd.to_numeric(df["other_variable_cost"], errors="coerce")
        has_src_oth = src_oth.notna()
        oth_cost.loc[has_src_oth] = src_oth.loc[has_src_oth]
        oth_src_type.loc[has_src_oth] = CostSourceType.SOURCE_DATA.value

    missing_oth = oth_cost.isna()
    if missing_oth.any():
        assumed_oth = pd.Series(0.0, index=df.index, dtype=float)
        has_oth_asm = False
        if asm.other_variable_cost_per_unit is not None:
            assumed_oth += asm.other_variable_cost_per_unit * qty.fillna(0.0)
            has_oth_asm = True
        if asm.other_variable_cost_pct is not None:
            assumed_oth += asm.other_variable_cost_pct * net_rev.clip(lower=0.0).fillna(0.0)
            has_oth_asm = True

        if has_oth_asm:
            oth_cost.loc[missing_oth] = assumed_oth.loc[missing_oth]
            oth_src_type.loc[missing_oth] = CostSourceType.CONFIGURED_ASSUMPTION.value

    # ---------------------------------------------------------
    # Totals, Margins, and Completeness Checks
    # ---------------------------------------------------------
    var_cols = [ship_cost, pay_cost, pkg_cost, wh_cost, ret_cost, oth_cost]
    total_known_var = pd.Series(0.0, index=df.index, dtype=float)
    for c in var_cols:
        total_known_var += c.fillna(0.0)

    # Gross Margin
    gross_margin = pd.Series(np.nan, index=df.index, dtype=float)
    gross_margin_pct = pd.Series(np.nan, index=df.index, dtype=float)
    has_prod = prod_cost.notna() & net_rev.notna()
    gross_margin.loc[has_prod] = net_rev.loc[has_prod] - prod_cost.loc[has_prod]
    pos_rev = has_prod & (net_rev > 0)
    gross_margin_pct.loc[pos_rev] = gross_margin.loc[pos_rev] / net_rev.loc[pos_rev]

    # Known Contribution Margin
    known_cm = pd.Series(np.nan, index=df.index, dtype=float)
    known_cm_pct = pd.Series(np.nan, index=df.index, dtype=float)
    known_cm.loc[has_prod] = gross_margin.loc[has_prod] - total_known_var.loc[has_prod]
    known_cm_pct.loc[pos_rev] = known_cm.loc[pos_rev] / net_rev.loc[pos_rev]

    # Completeness verification for required components
    req_keys = [c.value if isinstance(c, CostComponent) else str(c) for c in cfg.required_cost_components]

    def is_comp_satisfied(cost_s: pd.Series, src_s: pd.Series) -> pd.Series:
        is_avail = cost_s.notna()
        if cfg.strict_source_only:
            return is_avail & (src_s == CostSourceType.SOURCE_DATA.value)
        cond = (src_s == CostSourceType.SOURCE_DATA.value)
        if cfg.allow_estimated_costs_for_completeness:
            cond = cond | (src_s == CostSourceType.CATALOG_ESTIMATE.value)
        if cfg.allow_assumed_costs_for_completeness:
            cond = cond | (src_s == CostSourceType.CONFIGURED_ASSUMPTION.value)
        return is_avail & cond

    comp_satisfied = {
        CostComponent.PRODUCT_COST.value: is_comp_satisfied(prod_cost, prod_src_type),
        CostComponent.SHIPPING_COST.value: is_comp_satisfied(ship_cost, ship_src_type),
        CostComponent.PAYMENT_PROCESSING_COST.value: is_comp_satisfied(pay_cost, pay_src_type),
        CostComponent.PACKAGING_COST.value: is_comp_satisfied(pkg_cost, pkg_src_type),
        CostComponent.WAREHOUSE_HANDLING_COST.value: is_comp_satisfied(wh_cost, wh_src_type),
        CostComponent.RETURN_PROCESSING_COST.value: is_comp_satisfied(ret_cost, ret_src_type),
        CostComponent.OTHER_VARIABLE_COST.value: is_comp_satisfied(oth_cost, oth_src_type),
    }

    avail_req_count = pd.Series(0, index=df.index, dtype=int)
    for rk in req_keys:
        if rk in comp_satisfied:
            avail_req_count += comp_satisfied[rk].astype(int)

    tot_req = len(req_keys)
    completeness_pct = (avail_req_count / tot_req * 100.0) if tot_req > 0 else pd.Series(100.0, index=df.index)

    meets_completeness = completeness_pct >= cfg.min_completeness_threshold

    # Final Contribution Margin
    final_cm = pd.Series(np.nan, index=df.index, dtype=float)
    final_cm_pct = pd.Series(np.nan, index=df.index, dtype=float)
    cm_status = pd.Series(ContributionMarginStatus.INSUFFICIENT_COST_DATA.value, index=df.index, dtype=object)
    econ_status = pd.Series(OperationalEconomicsStatus.INSUFFICIENT_COST_DATA.value, index=df.index, dtype=object)
    rationale = pd.Series("Product cost missing or unobserved", index=df.index, dtype=object)

    if is_invalid.any():
        econ_status.loc[is_invalid] = OperationalEconomicsStatus.INVALID_INPUT.value
        cm_status.loc[is_invalid] = ContributionMarginStatus.NOT_CALCULABLE.value
        rationale.loc[is_invalid] = "Invalid input values detected (negative quantity, price, or discount; or missing SKU)"

    valid_mask = ~is_invalid

    # Partially calculable rows
    part_mask = valid_mask & has_prod & (~meets_completeness)
    if part_mask.any():
        econ_status.loc[part_mask] = OperationalEconomicsStatus.PARTIALLY_CALCULABLE.value
        cm_status.loc[part_mask] = ContributionMarginStatus.INSUFFICIENT_COST_DATA.value
        rationale.loc[part_mask] = "Gross margin calculable; required operational costs incomplete"

    # Complete rows
    full_mask = valid_mask & has_prod & meets_completeness
    if full_mask.any():
        econ_status.loc[full_mask] = OperationalEconomicsStatus.COMPLETE.value
        cm_status.loc[full_mask] = ContributionMarginStatus.CALCULABLE.value
        final_cm.loc[full_mask] = known_cm.loc[full_mask]
        pos_rev_full = full_mask & (net_rev > 0)
        final_cm_pct.loc[pos_rev_full] = known_cm.loc[pos_rev_full] / net_rev.loc[pos_rev_full]
        rationale.loc[full_mask] = "Complete operational economics: all required cost components satisfied"

    # Completeness Status
    comp_status = pd.Series(CostCompletenessStatus.INSUFFICIENT_DATA.value, index=df.index, dtype=object)
    comp_status.loc[completeness_pct >= 100.0] = CostCompletenessStatus.COMPLETE.value
    comp_status.loc[(completeness_pct > 0.0) & (completeness_pct < 100.0)] = CostCompletenessStatus.PARTIALLY_COMPLETE.value

    # Per-unit metrics
    net_rev_per_unit = pd.Series(np.nan, index=df.index, dtype=float)
    gm_per_unit = pd.Series(np.nan, index=df.index, dtype=float)
    var_per_unit = pd.Series(np.nan, index=df.index, dtype=float)
    cm_per_unit = pd.Series(np.nan, index=df.index, dtype=float)

    has_pos_q = valid_mask & (qty > 0)
    net_rev_per_unit.loc[has_pos_q & net_rev.notna()] = net_rev.loc[has_pos_q & net_rev.notna()] / qty.loc[has_pos_q & net_rev.notna()]
    gm_per_unit.loc[has_pos_q & gross_margin.notna()] = gross_margin.loc[has_pos_q & gross_margin.notna()] / qty.loc[has_pos_q & gross_margin.notna()]
    var_per_unit.loc[has_pos_q] = total_known_var.loc[has_pos_q] / qty.loc[has_pos_q]
    cm_per_unit.loc[has_pos_q & known_cm.notna()] = known_cm.loc[has_pos_q & known_cm.notna()] / qty.loc[has_pos_q & known_cm.notna()]

    # Deterministic record IDs
    rec_ids = [
        generate_operational_record_id(t, s, o, d, c)
        for t, s, o, d, c in zip(
            df["transaction_id"].astype(str),
            df["sku_id"].astype(str),
            df["order_id"].astype(str),
            df["date"].astype(str),
            df["currency"].astype(str),
        )
    ]

    out_df = pd.DataFrame(
        {
            "record_id": rec_ids,
            "transaction_id": df["transaction_id"].astype(str),
            "sale_id": df["sale_id"].astype(str),
            "order_id": df["order_id"].astype(str),
            "order_date": df["date"].astype(str),
            "sku_id": df["sku_id"].astype(str),
            "warehouse_id": df["warehouse_id"].astype(str),
            "channel_id": df["channel_id"].astype(str),
            "currency": df["currency"].astype(str),
            "quantity": qty,
            "unit_price": price,
            "gross_revenue": gross_rev,
            "discount": disc,
            "net_revenue": net_rev,
            "unit_product_cost": unit_prod_cost,
            "product_cost": prod_cost,
            "gross_margin": gross_margin,
            "gross_margin_pct": gross_margin_pct,
            "shipping_cost": ship_cost,
            "payment_processing_cost": pay_cost,
            "packaging_cost": pkg_cost,
            "warehouse_handling_cost": wh_cost,
            "return_processing_cost": ret_cost,
            "other_variable_cost": oth_cost,
            "total_known_variable_cost": total_known_var,
            "known_contribution_margin": known_cm,
            "known_contribution_margin_pct": known_cm_pct,
            "final_contribution_margin": final_cm,
            "final_contribution_margin_pct": final_cm_pct,
            "final_contribution_margin_status": cm_status,
            "net_revenue_per_unit": net_rev_per_unit,
            "gross_margin_per_unit": gm_per_unit,
            "variable_cost_per_unit": var_per_unit,
            "contribution_margin_per_unit": cm_per_unit,
            "cost_completeness_pct": completeness_pct,
            "cost_completeness_status": comp_status,
            "economics_status": econ_status,
            "status_rationale": rationale,
            "product_cost_source": prod_src_type,
            "shipping_cost_source": ship_src_type,
            "payment_cost_source": pay_src_type,
            "packaging_cost_source": pkg_src_type,
            "warehouse_cost_source": wh_src_type,
            "return_cost_source": ret_src_type,
            "other_cost_source": oth_src_type,
        },
        index=df.index,
    )

    return out_df


# =============================================================================
# Order-Level Operational Economics
# =============================================================================


def compute_order_operational_economics_dataframe(
    line_economics_df: pd.DataFrame,
    config: Optional[OperationalEconomicsConfig] = None,
) -> pd.DataFrame:
    """Aggregate transaction lines to true order-level operational economics.

    Ensures order-level costs (e.g. flat order shipping) are counted exactly once per order,
    and line-level costs are summed accurately.
    """
    if line_economics_df.empty:
        return pd.DataFrame()

    cfg = config or OperationalEconomicsConfig()
    df = line_economics_df

    # Group by order_id
    grouped = df.groupby("order_id", as_index=False).agg(
        order_date=("order_date", "first"),
        channel_id=("channel_id", "first"),
        warehouse_id=("warehouse_id", "first"),
        currency=("currency", "first"),
        line_count=("record_id", "count"),
        total_quantity=("quantity", "sum"),
        gross_revenue=("gross_revenue", "sum"),
        discount=("discount", "sum"),
        net_revenue=("net_revenue", "sum"),
        product_cost=("product_cost", "sum"),
        shipping_cost=("shipping_cost", "sum"),
        payment_processing_cost=("payment_processing_cost", "sum"),
        packaging_cost=("packaging_cost", "sum"),
        warehouse_handling_cost=("warehouse_handling_cost", "sum"),
        return_processing_cost=("return_processing_cost", "sum"),
        other_variable_cost=("other_variable_cost", "sum"),
        total_known_variable_cost=("total_known_variable_cost", "sum"),
        avg_completeness_pct=("cost_completeness_pct", "mean"),
    )

    # Order margins
    has_prod = grouped["product_cost"].notna() & grouped["net_revenue"].notna()
    grouped["gross_margin"] = np.nan
    grouped.loc[has_prod, "gross_margin"] = grouped.loc[has_prod, "net_revenue"] - grouped.loc[has_prod, "product_cost"]

    grouped["gross_margin_pct"] = np.nan
    pos_rev = has_prod & (grouped["net_revenue"] > 0)
    grouped.loc[pos_rev, "gross_margin_pct"] = grouped.loc[pos_rev, "gross_margin"] / grouped.loc[pos_rev, "net_revenue"]

    grouped["known_contribution_margin"] = np.nan
    grouped.loc[has_prod, "known_contribution_margin"] = (
        grouped.loc[has_prod, "gross_margin"] - grouped.loc[has_prod, "total_known_variable_cost"]
    )

    grouped["known_contribution_margin_pct"] = np.nan
    grouped.loc[pos_rev, "known_contribution_margin_pct"] = (
        grouped.loc[pos_rev, "known_contribution_margin"] / grouped.loc[pos_rev, "net_revenue"]
    )

    # Order completeness check
    order_meets_comp = grouped["avg_completeness_pct"] >= cfg.min_completeness_threshold

    grouped["final_contribution_margin"] = np.nan
    grouped["final_contribution_margin_pct"] = np.nan
    grouped["final_contribution_margin_status"] = ContributionMarginStatus.INSUFFICIENT_COST_DATA.value
    grouped["economics_status"] = OperationalEconomicsStatus.PARTIALLY_CALCULABLE.value

    full_orders = has_prod & order_meets_comp
    if full_orders.any():
        grouped.loc[full_orders, "final_contribution_margin"] = grouped.loc[full_orders, "known_contribution_margin"]
        pos_rev_full = full_orders & (grouped["net_revenue"] > 0)
        grouped.loc[pos_rev_full, "final_contribution_margin_pct"] = (
            grouped.loc[pos_rev_full, "known_contribution_margin"] / grouped.loc[pos_rev_full, "net_revenue"]
        )
        grouped.loc[full_orders, "final_contribution_margin_status"] = ContributionMarginStatus.CALCULABLE.value
        grouped.loc[full_orders, "economics_status"] = OperationalEconomicsStatus.COMPLETE.value

    # Cost completeness status
    grouped["cost_completeness_status"] = CostCompletenessStatus.INSUFFICIENT_DATA.value
    grouped.loc[grouped["avg_completeness_pct"] >= 100.0, "cost_completeness_status"] = CostCompletenessStatus.COMPLETE.value
    grouped.loc[
        (grouped["avg_completeness_pct"] > 0.0) & (grouped["avg_completeness_pct"] < 100.0),
        "cost_completeness_status",
    ] = CostCompletenessStatus.PARTIALLY_COMPLETE.value

    grouped.rename(columns={"avg_completeness_pct": "cost_completeness_pct"}, inplace=True)

    return grouped


# =============================================================================
# Dimensional and Temporal Aggregations
# =============================================================================


def aggregate_operational_dimension(
    df: pd.DataFrame,
    dimension: str,
    config: Optional[OperationalEconomicsConfig] = None,
) -> List[OperationalEconomicsSegment]:
    """Aggregate operational economics by analytical dimension.

    Supports: SKU, CATEGORY, BRAND, CHANNEL, WAREHOUSE, CHANNEL_WAREHOUSE, etc.
    """
    if df.empty:
        return []

    cfg = config or OperationalEconomicsConfig()
    dim_col = dimension.lower()

    if dim_col not in df.columns:
        if dimension.upper() == "CHANNEL_WAREHOUSE" and "channel_id" in df.columns and "warehouse_id" in df.columns:
            group_cols = ["channel_id", "warehouse_id"]
        else:
            return []
    else:
        group_cols = [dim_col]

    segments: List[OperationalEconomicsSegment] = []

    for key, group in df.groupby(group_cols, as_index=False):
        seg_key = "_".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
        rec_cnt = len(group)
        ord_cnt = group["order_id"].nunique() if "order_id" in group.columns else rec_cnt
        units = int(group["quantity"].sum()) if "quantity" in group.columns else 0

        gross_rev = float(group["gross_revenue"].sum()) if group["gross_revenue"].notna().any() else None
        disc = float(group["discount"].sum()) if group["discount"].notna().any() else None
        net_rev = float(group["net_revenue"].sum()) if group["net_revenue"].notna().any() else None
        prod_cost = float(group["product_cost"].sum()) if group["product_cost"].notna().any() else None

        gm = (net_rev - prod_cost) if (net_rev is not None and prod_cost is not None) else None
        gm_pct = (gm / net_rev) if (gm is not None and net_rev is not None and net_rev > 0) else None

        ship = float(group["shipping_cost"].sum()) if group["shipping_cost"].notna().any() else None
        pay = float(group["payment_processing_cost"].sum()) if group["payment_processing_cost"].notna().any() else None
        pkg = float(group["packaging_cost"].sum()) if group["packaging_cost"].notna().any() else None
        wh = float(group["warehouse_handling_cost"].sum()) if group["warehouse_handling_cost"].notna().any() else None
        ret = float(group["return_processing_cost"].sum()) if group["return_processing_cost"].notna().any() else None
        oth = float(group["other_variable_cost"].sum()) if group["other_variable_cost"].notna().any() else None
        total_var = float(group["total_known_variable_cost"].sum())

        known_cm = (gm - total_var) if gm is not None else None
        known_cm_pct = (known_cm / net_rev) if (known_cm is not None and net_rev is not None and net_rev > 0) else None

        avg_comp = float(group["cost_completeness_pct"].mean()) if "cost_completeness_pct" in group.columns else 0.0
        complete_cnt = int((group["cost_completeness_pct"] >= cfg.min_completeness_threshold).sum())
        part_cnt = int(((group["cost_completeness_pct"] > 0) & (group["cost_completeness_pct"] < cfg.min_completeness_threshold)).sum())
        insuf_cnt = int((group["cost_completeness_pct"] == 0).sum())

        # Final contribution margin for segment is populated only if all records in segment are complete
        if complete_cnt == rec_cnt and known_cm is not None:
            final_cm = known_cm
            final_cm_pct = known_cm_pct
            cm_status = ContributionMarginStatus.CALCULABLE
        else:
            final_cm = None
            final_cm_pct = None
            cm_status = ContributionMarginStatus.INSUFFICIENT_COST_DATA

        segments.append(
            OperationalEconomicsSegment(
                dimension=dimension.upper(),
                segment_key=seg_key,
                record_count=rec_cnt,
                order_count=ord_cnt,
                total_units=units,
                gross_revenue=gross_rev,
                discount=disc,
                net_revenue=net_rev,
                product_cost=prod_cost,
                gross_margin=gm,
                gross_margin_pct=gm_pct,
                shipping_cost=ship,
                payment_processing_cost=pay,
                packaging_cost=pkg,
                warehouse_handling_cost=wh,
                return_processing_cost=ret,
                other_variable_cost=oth,
                total_known_variable_cost=total_var,
                known_contribution_margin=known_cm,
                known_contribution_margin_pct=known_cm_pct,
                final_contribution_margin=final_cm,
                final_contribution_margin_pct=final_cm_pct,
                final_contribution_margin_status=cm_status,
                avg_cost_completeness_pct=avg_comp,
                complete_records_count=complete_cnt,
                partially_complete_records_count=part_cnt,
                insufficient_records_count=insuf_cnt,
            )
        )

    return segments


def aggregate_operational_time_series(
    df: pd.DataFrame,
    time_grain: Union[TimeGrain, str] = TimeGrain.MONTHLY,
    config: Optional[OperationalEconomicsConfig] = None,
) -> List[OperationalEconomicsSegment]:
    """Aggregate operational economics by temporal grain (DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY)."""
    if df.empty or "order_date" not in df.columns:
        return []

    cfg = config or OperationalEconomicsConfig()
    tdf = df.copy()
    dt = pd.to_datetime(tdf["order_date"], errors="coerce")

    tg_str = time_grain.value.upper() if isinstance(time_grain, TimeGrain) else str(time_grain).upper()

    if tg_str in ("DAILY", "DAY"):
        tdf["time_bucket"] = dt.dt.strftime("%Y-%m-%d")
    elif tg_str in ("WEEKLY", "WEEK"):
        tdf["time_bucket"] = dt.dt.to_period("W").astype(str)
    elif tg_str in ("MONTHLY", "MONTH"):
        tdf["time_bucket"] = dt.dt.strftime("%Y-%m")
    elif tg_str in ("QUARTERLY", "QUARTER"):
        tdf["time_bucket"] = dt.dt.to_period("Q").astype(str)
    elif tg_str in ("YEARLY", "YEAR"):
        tdf["time_bucket"] = dt.dt.strftime("%Y")
    else:
        tdf["time_bucket"] = dt.dt.strftime("%Y-%m")

    # Reuse dimension aggregation logic
    return aggregate_operational_dimension(tdf, "time_bucket", cfg)


# =============================================================================
# Executive Portfolio Summary
# =============================================================================


def summarize_operational_portfolio(
    df: pd.DataFrame,
    config: Optional[OperationalEconomicsConfig] = None,
) -> OperationalEconomicsSummary:
    """Produce executive network portfolio summary and cost completeness audit report."""
    cfg = config or OperationalEconomicsConfig()
    now_iso = datetime.now(timezone.utc).isoformat()

    if df.empty:
        comp_rep = CostCompletenessReport(
            completeness_status=CostCompletenessStatus.INSUFFICIENT_DATA,
            completeness_pct=0.0,
            total_required_components=len(cfg.required_cost_components),
            available_required_components=0,
            missing_required_components=len(cfg.required_cost_components),
            required_components=[c.value for c in cfg.required_cost_components],
            available_components=[],
            unavailable_components=[c.value for c in cfg.required_cost_components],
            source_components=[],
            estimated_components=[],
            assumed_components=[],
            audit_notes=["Empty transaction dataset"],
        )
        return OperationalEconomicsSummary(
            total_records=0,
            total_orders=0,
            total_units=0,
            gross_revenue=0.0,
            discount=0.0,
            net_revenue=0.0,
            product_cost=0.0,
            gross_margin=0.0,
            gross_margin_pct=None,
            shipping_cost=0.0,
            payment_processing_cost=0.0,
            packaging_cost=0.0,
            warehouse_handling_cost=0.0,
            return_processing_cost=0.0,
            other_variable_cost=0.0,
            total_known_variable_cost=0.0,
            known_contribution_margin=0.0,
            known_contribution_margin_pct=None,
            final_contribution_margin=None,
            final_contribution_margin_pct=None,
            final_contribution_margin_status=ContributionMarginStatus.INSUFFICIENT_COST_DATA,
            cost_completeness_report=comp_rep,
            cost_details={},
            currency=cfg.default_currency,
            as_of_date=cfg.as_of_date,
            generated_at=now_iso,
        )

    rec_cnt = len(df)
    ord_cnt = df["order_id"].nunique() if "order_id" in df.columns else rec_cnt
    tot_units = int(df["quantity"].sum()) if "quantity" in df.columns else 0

    gross_rev = float(df["gross_revenue"].sum()) if df["gross_revenue"].notna().any() else 0.0
    disc = float(df["discount"].sum()) if df["discount"].notna().any() else 0.0
    net_rev = float(df["net_revenue"].sum()) if df["net_revenue"].notna().any() else 0.0
    prod_cost = float(df["product_cost"].sum()) if df["product_cost"].notna().any() else 0.0

    gm = net_rev - prod_cost
    gm_pct = (gm / net_rev) if net_rev > 0 else None

    ship = float(df["shipping_cost"].sum()) if df["shipping_cost"].notna().any() else 0.0
    pay = float(df["payment_processing_cost"].sum()) if df["payment_processing_cost"].notna().any() else 0.0
    pkg = float(df["packaging_cost"].sum()) if df["packaging_cost"].notna().any() else 0.0
    wh = float(df["warehouse_handling_cost"].sum()) if df["warehouse_handling_cost"].notna().any() else 0.0
    ret = float(df["return_processing_cost"].sum()) if df["return_processing_cost"].notna().any() else 0.0
    oth = float(df["other_variable_cost"].sum()) if df["other_variable_cost"].notna().any() else 0.0
    tot_var = float(df["total_known_variable_cost"].sum())

    known_cm = gm - tot_var
    known_cm_pct = (known_cm / net_rev) if net_rev > 0 else None

    # Completeness analysis
    req_keys = [c.value if isinstance(c, CostComponent) else str(c) for c in cfg.required_cost_components]
    avail_comps: List[str] = []
    unavail_comps: List[str] = []
    src_comps: List[str] = []
    est_comps: List[str] = []
    asm_comps: List[str] = []
    notes: List[str] = []

    comp_cols = [
        (CostComponent.PRODUCT_COST.value, "product_cost", "product_cost_source"),
        (CostComponent.SHIPPING_COST.value, "shipping_cost", "shipping_cost_source"),
        (CostComponent.PAYMENT_PROCESSING_COST.value, "payment_processing_cost", "payment_cost_source"),
        (CostComponent.PACKAGING_COST.value, "packaging_cost", "packaging_cost_source"),
        (CostComponent.WAREHOUSE_HANDLING_COST.value, "warehouse_handling_cost", "warehouse_cost_source"),
        (CostComponent.RETURN_PROCESSING_COST.value, "return_processing_cost", "return_cost_source"),
        (CostComponent.OTHER_VARIABLE_COST.value, "other_variable_cost", "other_cost_source"),
    ]

    details: Dict[str, OperationalCostDetail] = {}

    for c_name, c_col, s_col in comp_cols:
        is_col_present = c_col in df.columns and df[c_col].notna().any()
        src_val = df[s_col].iloc[0] if s_col in df.columns and len(df) > 0 else CostSourceType.UNAVAILABLE.value

        if is_col_present:
            tot_amt = float(df[c_col].sum())
            unit_amt = tot_amt / tot_units if tot_units > 0 else None

            if src_val == CostSourceType.SOURCE_DATA.value:
                src_comps.append(c_name)
            elif src_val == CostSourceType.CATALOG_ESTIMATE.value:
                est_comps.append(c_name)
            elif src_val == CostSourceType.CONFIGURED_ASSUMPTION.value:
                asm_comps.append(c_name)

            # Completeness evaluation
            satisfies = False
            if cfg.strict_source_only:
                satisfies = (src_val == CostSourceType.SOURCE_DATA.value)
            else:
                if src_val == CostSourceType.SOURCE_DATA.value:
                    satisfies = True
                elif src_val == CostSourceType.CATALOG_ESTIMATE.value and cfg.allow_estimated_costs_for_completeness:
                    satisfies = True
                elif src_val == CostSourceType.CONFIGURED_ASSUMPTION.value and cfg.allow_assumed_costs_for_completeness:
                    satisfies = True

            if satisfies:
                avail_comps.append(c_name)
            else:
                unavail_comps.append(c_name)
                notes.append(f"{c_name} present via {src_val} but excluded from completeness by policy")

            details[c_name] = OperationalCostDetail(
                component=CostComponent(c_name),
                amount=tot_amt,
                unit_amount=unit_amt,
                currency=cfg.default_currency,
                source_type=CostSourceType(src_val),
                availability_status=CostComponentStatus.AVAILABLE_SOURCE if src_val == CostSourceType.SOURCE_DATA.value else CostComponentStatus.AVAILABLE_ASSUMED,
                cost_grain=CostGrain.TRANSACTION,
                is_estimated=(src_val == CostSourceType.CATALOG_ESTIMATE.value),
                is_assumed=(src_val == CostSourceType.CONFIGURED_ASSUMPTION.value),
                included_in_known_contribution=True,
                included_in_final_contribution=satisfies,
                source_reference=c_col,
                description=f"Portfolio aggregated {c_name}",
            )
        else:
            unavail_comps.append(c_name)
            details[c_name] = OperationalCostDetail(
                component=CostComponent(c_name),
                amount=None,
                unit_amount=None,
                currency=cfg.default_currency,
                source_type=CostSourceType.UNAVAILABLE,
                availability_status=CostComponentStatus.UNAVAILABLE,
                cost_grain=CostGrain.UNKNOWN,
                is_estimated=False,
                is_assumed=False,
                included_in_known_contribution=False,
                included_in_final_contribution=False,
                source_reference=None,
                description=f"{c_name} unobserved and unconfigured",
            )

    avail_req = sum(1 for k in req_keys if k in avail_comps)
    tot_req = len(req_keys)
    comp_pct = (avail_req / tot_req * 100.0) if tot_req > 0 else 100.0
    missing_req = tot_req - avail_req

    if comp_pct >= 100.0:
        c_status = CostCompletenessStatus.COMPLETE
    elif comp_pct > 0.0:
        c_status = CostCompletenessStatus.PARTIALLY_COMPLETE
    else:
        c_status = CostCompletenessStatus.INSUFFICIENT_DATA

    comp_report = CostCompletenessReport(
        completeness_status=c_status,
        completeness_pct=round(comp_pct, 2),
        total_required_components=tot_req,
        available_required_components=avail_req,
        missing_required_components=missing_req,
        required_components=req_keys,
        available_components=avail_comps,
        unavailable_components=unavail_comps,
        source_components=src_comps,
        estimated_components=est_comps,
        assumed_components=asm_comps,
        audit_notes=notes,
    )

    # Final contribution margin for portfolio
    all_lines_complete = (df["cost_completeness_pct"] >= cfg.min_completeness_threshold).all() if "cost_completeness_pct" in df.columns else False
    if all_lines_complete and comp_pct >= cfg.min_completeness_threshold:
        final_cm = known_cm
        final_cm_pct = known_cm_pct
        cm_status = ContributionMarginStatus.CALCULABLE
    else:
        final_cm = None
        final_cm_pct = None
        cm_status = ContributionMarginStatus.INSUFFICIENT_COST_DATA

    return OperationalEconomicsSummary(
        total_records=rec_cnt,
        total_orders=ord_cnt,
        total_units=tot_units,
        gross_revenue=gross_rev,
        discount=disc,
        net_revenue=net_rev,
        product_cost=prod_cost,
        gross_margin=gm,
        gross_margin_pct=gm_pct,
        shipping_cost=ship,
        payment_processing_cost=pay,
        packaging_cost=pkg,
        warehouse_handling_cost=wh,
        return_processing_cost=ret,
        other_variable_cost=oth,
        total_known_variable_cost=tot_var,
        known_contribution_margin=known_cm,
        known_contribution_margin_pct=known_cm_pct,
        final_contribution_margin=final_cm,
        final_contribution_margin_pct=final_cm_pct,
        final_contribution_margin_status=cm_status,
        cost_completeness_report=comp_report,
        cost_details=details,
        currency=cfg.default_currency,
        as_of_date=cfg.as_of_date,
        generated_at=now_iso,
    )
