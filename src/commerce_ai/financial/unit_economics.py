"""Unit Economics Engine & Margin Erosion Analysis (Phase 6B).

Provides deterministic calculations for:
- Line-level unit economics records with variable cost breakdown
- Vectorized DataFrame computation for high-volume transactions
- Multi-dimensional aggregations (Overall, SKU, Category, Brand, Channel, Warehouse, Cross-slices, Time series)
- Descriptive margin erosion reporting (negative margin, low margin, high discount, contribution mismatches)
- Executive portfolio unit economics summary
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.financial.schemas import (
    ContributionMarginStatus,
    CostComponent,
    CostComponentDetail,
    CostComponentStatus,
    CostModelConfig,
    CostSourceType,
    FinancialDataQualityReport,
    MarginErosionReport,
    MarginErosionSegment,
    TimeGrain,
    UnitEconomicsDimensionMetric,
    UnitEconomicsPortfolioSummary,
    UnitEconomicsRecord,
    UnitEconomicsStatus,
)
from commerce_ai.financial.cost_model import CostModelResolver
from commerce_ai.financial.revenue_margin import filter_sales_by_as_of_date


# =====================================================================
# Deterministic Identifiers
# =====================================================================


def generate_unit_economics_record_id(
    sale_id: str,
    sku_id: str,
    order_id: str,
    date_val: str,
    currency: str = "USD",
) -> str:
    """Generate a deterministic 16-character SHA-256 identifier for a unit economics line record.

    Format: FIN-UE-{hexdigest[:16]}
    """
    raw = f"UE|{str(sale_id).strip()}|{str(sku_id).strip()}|{str(order_id).strip()}|{str(date_val).strip()}|{str(currency).strip().upper()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"FIN-UE-{digest[:16]}"


def generate_unit_economics_segment_id(
    dimension: str,
    segment_key: str,
    as_of_date: Optional[str] = None,
    currency: str = "USD",
) -> str:
    """Generate a deterministic 16-character SHA-256 identifier for an economics segment.

    Format: FIN-UESEG-{hexdigest[:16]}
    """
    raw = f"UESEG|{str(dimension).strip().upper()}|{str(segment_key).strip()}|{str(as_of_date or '').strip()}|{str(currency).strip().upper()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"FIN-UESEG-{digest[:16]}"


# =====================================================================
# Record-Level Unit Economics Calculation
# =====================================================================


def calculate_unit_economics_record(
    row: Union[pd.Series, Dict[str, Any]],
    unit_cost: Optional[float] = None,
    config: Optional[CostModelConfig] = None,
    cost_resolver: Optional[CostModelResolver] = None,
) -> UnitEconomicsRecord:
    """Calculate deterministic unit economics and cost breakdown for a single transaction line."""
    cfg = config or CostModelConfig()
    resolver = cost_resolver or CostModelResolver(cfg)

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

    sale_id = str(get_val("sale_id") or "").strip()
    order_id = str(get_val("order_id") or "").strip()
    date_val = str(get_val("date", "sale_date") or "").strip()
    sku_id = str(get_val("sku_id") or "").strip()
    warehouse_id = str(get_val("warehouse_id") or "").strip()
    channel_id = str(get_val("channel_id") or "").strip()
    currency = str(get_val("currency") or cfg.default_currency).strip().upper()

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
    discount = safe_float(get_val("discount"))
    source_revenue = safe_float(get_val("revenue", "source_revenue"))
    parsed_cost = safe_float(unit_cost if unit_cost is not None else get_val("unit_cost"))

    record_id = generate_unit_economics_record_id(sale_id, sku_id, order_id, date_val, currency)

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
    if parsed_cost is not None and parsed_cost < 0:
        invalid_reasons.append(f"Negative unit cost: {parsed_cost}")

    if invalid_reasons:
        return UnitEconomicsRecord(
            record_id=record_id,
            sale_id=sale_id,
            order_id=order_id,
            date=date_val,
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            channel_id=channel_id,
            currency=currency,
            quantity=qty,
            unit_price=unit_price,
            gross_revenue=None,
            discount=discount,
            net_revenue=None,
            source_revenue=source_revenue,
            unit_product_cost=parsed_cost,
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
            contribution_margin=None,
            contribution_margin_pct=None,
            contribution_margin_status=ContributionMarginStatus.NOT_CALCULABLE,
            cost_completeness_pct=0.0,
            required_cost_components=[c.value for c in cfg.required_cost_components],
            available_cost_components=[],
            unavailable_cost_components=[c.value for c in cfg.required_cost_components],
            estimated_cost_components=[],
            assumed_cost_components=[],
            economics_status=UnitEconomicsStatus.INVALID_DATA,
            status_rationale="; ".join(invalid_reasons),
            as_of_date=str(cfg.as_of_date) if cfg.as_of_date else None,
            cost_details={},
        )

    # 2. INSUFFICIENT detection
    if qty is None or unit_price is None:
        return UnitEconomicsRecord(
            record_id=record_id,
            sale_id=sale_id,
            order_id=order_id,
            date=date_val,
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            channel_id=channel_id,
            currency=currency,
            quantity=qty,
            unit_price=unit_price,
            gross_revenue=None,
            discount=discount,
            net_revenue=None,
            source_revenue=source_revenue,
            unit_product_cost=parsed_cost,
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
            contribution_margin=None,
            contribution_margin_pct=None,
            contribution_margin_status=ContributionMarginStatus.INSUFFICIENT_COST_DATA,
            cost_completeness_pct=0.0,
            required_cost_components=[c.value for c in cfg.required_cost_components],
            available_cost_components=[],
            unavailable_cost_components=[c.value for c in cfg.required_cost_components],
            estimated_cost_components=[],
            assumed_cost_components=[],
            economics_status=UnitEconomicsStatus.INSUFFICIENT_COST_DATA,
            status_rationale="Missing critical revenue inputs (quantity or unit_price)",
            as_of_date=str(cfg.as_of_date) if cfg.as_of_date else None,
            cost_details={},
        )

    # 3. Revenue calculations
    gross_revenue = float(qty * unit_price)
    discount_val = discount if discount is not None else 0.0
    net_revenue = float(gross_revenue - discount_val)

    # 4. Product cost & gross margin
    if parsed_cost is not None:
        product_cost = float(qty * parsed_cost)
        gross_margin = float(net_revenue - product_cost)
        gross_margin_pct = float(gross_margin / net_revenue) if net_revenue > 0 else None
    else:
        product_cost = None
        gross_margin = None
        gross_margin_pct = None

    # 5. Variable cost resolution
    # Extract any row-level variable cost fields if present
    txn_costs: Dict[str, Any] = {}
    for k in [
        "shipping_cost", "payment_processing_cost", "payment_fee",
        "packaging_cost", "warehouse_handling_cost", "handling_cost",
        "return_processing_cost", "other_variable_cost"
    ]:
        v = get_val(k)
        if v is not None:
            txn_costs[k] = safe_float(v)

    cost_details = resolver.resolve_line_costs(
        quantity=qty,
        unit_price=unit_price,
        net_revenue=net_revenue,
        sku_id=sku_id,
        channel_id=channel_id,
        warehouse_id=warehouse_id,
        unit_cost=parsed_cost,
        transaction_costs=txn_costs,
    )

    ship_cost = cost_details[CostComponent.SHIPPING_COST.value].amount
    pay_cost = cost_details[CostComponent.PAYMENT_PROCESSING_COST.value].amount
    pkg_cost = cost_details[CostComponent.PACKAGING_COST.value].amount
    wh_cost = cost_details[CostComponent.WAREHOUSE_HANDLING_COST.value].amount
    ret_cost = cost_details[CostComponent.RETURN_PROCESSING_COST.value].amount
    oth_cost = cost_details[CostComponent.OTHER_VARIABLE_COST.value].amount

    # Sum of available variable costs
    known_var_costs = [
        c for c in [ship_cost, pay_cost, pkg_cost, wh_cost, ret_cost, oth_cost] if c is not None
    ]
    total_known_var = float(sum(known_var_costs)) if known_var_costs else 0.0

    # 6. Contribution economics
    if product_cost is not None:
        known_cm = float(net_revenue - product_cost - total_known_var)
        known_cm_pct = float(known_cm / net_revenue) if net_revenue > 0 else None
    else:
        known_cm = None
        known_cm_pct = None

    # Completeness calculation
    completeness_pct, avail_comps, missing_comps, est_comps, assumed_comps = (
        resolver.calculate_cost_completeness(cost_details)
    )

    req_keys = [c.value if isinstance(c, CostComponent) else str(c) for c in cfg.required_cost_components]
    all_required_present = all(k in avail_comps for k in req_keys)

    if all_required_present and product_cost is not None:
        final_cm = known_cm
        final_cm_pct = known_cm_pct
        cm_status = ContributionMarginStatus.CALCULABLE
        econ_status = UnitEconomicsStatus.FULLY_CALCULABLE
        rationale = "Complete unit economics: all required cost components available"
    elif product_cost is not None:
        final_cm = None
        final_cm_pct = None
        cm_status = ContributionMarginStatus.INSUFFICIENT_COST_DATA
        econ_status = UnitEconomicsStatus.PARTIALLY_CALCULABLE
        missing_req_names = [k for k in req_keys if k not in avail_comps]
        rationale = f"Partially calculable: product cost observed; missing required variable costs: {', '.join(missing_req_names)}"
    else:
        final_cm = None
        final_cm_pct = None
        cm_status = ContributionMarginStatus.INSUFFICIENT_COST_DATA
        econ_status = UnitEconomicsStatus.INSUFFICIENT_COST_DATA
        rationale = "Product procurement cost is missing in catalog"

    return UnitEconomicsRecord(
        record_id=record_id,
        sale_id=sale_id,
        order_id=order_id,
        date=date_val,
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        channel_id=channel_id,
        currency=currency,
        quantity=qty,
        unit_price=unit_price,
        gross_revenue=gross_revenue,
        discount=discount,
        net_revenue=net_revenue,
        source_revenue=source_revenue,
        unit_product_cost=parsed_cost,
        product_cost=product_cost,
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
        contribution_margin=final_cm,
        contribution_margin_pct=final_cm_pct,
        contribution_margin_status=cm_status,
        cost_completeness_pct=completeness_pct,
        required_cost_components=req_keys,
        available_cost_components=avail_comps,
        unavailable_cost_components=missing_comps,
        estimated_cost_components=est_comps,
        assumed_cost_components=assumed_comps,
        economics_status=econ_status,
        status_rationale=rationale,
        as_of_date=str(cfg.as_of_date) if cfg.as_of_date else None,
        cost_details=cost_details,
    )


# =====================================================================
# Vectorized DataFrame Computation Engine
# =====================================================================


def compute_unit_economics_dataframe(
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    config: Optional[CostModelConfig] = None,
) -> pd.DataFrame:
    """Compute unit economics, cost breakdowns, and contribution margins vectorially across a DataFrame.

    Optimized for processing large transaction volumes (e.g. 750k+ rows) in sub-seconds.
    """
    cfg = config or CostModelConfig()
    resolver = CostModelResolver(cfg)

    if sales_df.empty:
        cols = [
            "record_id", "sale_id", "order_id", "date", "sku_id", "warehouse_id", "channel_id",
            "currency", "quantity", "unit_price", "gross_revenue", "discount", "net_revenue",
            "source_revenue", "unit_product_cost", "product_cost", "gross_margin", "gross_margin_pct",
            "shipping_cost", "payment_processing_cost", "packaging_cost", "warehouse_handling_cost",
            "return_processing_cost", "other_variable_cost", "total_known_variable_cost",
            "known_contribution_margin", "known_contribution_margin_pct", "contribution_margin",
            "contribution_margin_pct", "contribution_margin_status", "cost_completeness_pct",
            "economics_status", "status_rationale", "as_of_date",
        ]
        return pd.DataFrame(columns=cols)

    df = sales_df.copy()

    # Standardize column names
    if "sale_date" in df.columns and "date" not in df.columns:
        df["date"] = df["sale_date"]
    if "date" not in df.columns:
        df["date"] = ""
    df["date"] = df["date"].astype(str)

    if "sale_id" not in df.columns:
        df["sale_id"] = [f"SALE_{i}" for i in range(len(df))]
    df["sale_id"] = df["sale_id"].astype(str)

    if "order_id" not in df.columns:
        df["order_id"] = ""
    df["order_id"] = df["order_id"].astype(str)

    if "sku_id" not in df.columns:
        df["sku_id"] = ""
    df["sku_id"] = df["sku_id"].astype(str)

    if "warehouse_id" not in df.columns:
        df["warehouse_id"] = ""
    df["warehouse_id"] = df["warehouse_id"].astype(str)

    if "channel_id" not in df.columns:
        df["channel_id"] = ""
    df["channel_id"] = df["channel_id"].astype(str)

    if "currency" not in df.columns:
        df["currency"] = cfg.default_currency
    df["currency"] = df["currency"].fillna(cfg.default_currency).astype(str).str.strip().str.upper()

    # Numeric conversion
    qty = pd.to_numeric(df.get("quantity"), errors="coerce")
    unit_price = pd.to_numeric(df.get("unit_price"), errors="coerce")
    discount = pd.to_numeric(df.get("discount"), errors="coerce")
    source_rev = pd.to_numeric(df.get("revenue", df.get("source_revenue")), errors="coerce")

    # Product enrichment for unit_cost, category_id, brand
    unit_cost = None
    if products_df is not None and not products_df.empty and "sku_id" in products_df.columns:
        if "category_id" in products_df.columns and "category_id" not in df.columns:
            cat_map = products_df.set_index("sku_id")["category_id"].to_dict()
            df["category_id"] = df["sku_id"].map(cat_map)
        if "brand" in products_df.columns and "brand" not in df.columns:
            brand_map = products_df.set_index("sku_id")["brand"].to_dict()
            df["brand"] = df["sku_id"].map(brand_map)
        if "unit_cost" in products_df.columns:
            cost_map = products_df.set_index("sku_id")["unit_cost"].to_dict()
            unit_cost = df["sku_id"].map(cost_map)

    if unit_cost is None:
        if "unit_cost" in df.columns:
            unit_cost = pd.to_numeric(df["unit_cost"], errors="coerce")
        else:
            unit_cost = pd.Series(np.nan, index=df.index, dtype=float)
    else:
        unit_cost = pd.to_numeric(unit_cost, errors="coerce")

    # Invalid masks
    missing_sku = df["sku_id"].str.strip() == ""
    invalid_qty = (qty < 0)
    invalid_price = (unit_price < 0)
    invalid_disc = (discount < 0)
    invalid_cost = (unit_cost < 0)
    is_invalid = missing_sku | invalid_qty | invalid_price | invalid_disc | invalid_cost

    # Insufficient mask
    is_insufficient = ~is_invalid & (qty.isna() | unit_price.isna())
    is_valid = ~is_invalid & ~is_insufficient

    # Revenue calculation
    gross_rev = pd.Series(np.nan, index=df.index, dtype=float)
    disc_val = discount.fillna(0.0)
    net_rev = pd.Series(np.nan, index=df.index, dtype=float)

    valid_idx = df.index[is_valid]
    if len(valid_idx) > 0:
        gross_rev.loc[valid_idx] = qty.loc[valid_idx] * unit_price.loc[valid_idx]
        net_rev.loc[valid_idx] = gross_rev.loc[valid_idx] - disc_val.loc[valid_idx]

    # Product cost & Gross Margin
    prod_cost = pd.Series(np.nan, index=df.index, dtype=float)
    gross_margin = pd.Series(np.nan, index=df.index, dtype=float)
    gross_margin_pct = pd.Series(np.nan, index=df.index, dtype=float)

    has_cost = is_valid & unit_cost.notna()
    if has_cost.any():
        cost_idx = df.index[has_cost]
        cogs = qty.loc[cost_idx] * unit_cost.loc[cost_idx]
        gm = net_rev.loc[cost_idx] - cogs

        prod_cost.loc[cost_idx] = cogs
        gross_margin.loc[cost_idx] = gm

        pos_net = cost_idx[net_rev.loc[cost_idx] > 0]
        if len(pos_net) > 0:
            gross_margin_pct.loc[pos_net] = gm.loc[pos_net] / net_rev.loc[pos_net]

    # Variable costs resolution (vectorized where possible)
    # 1. Shipping
    ship_cost = pd.Series(np.nan, index=df.index, dtype=float)
    if "shipping_cost" in df.columns:
        ship_cost = pd.to_numeric(df["shipping_cost"], errors="coerce")
    elif cfg.channel_shipping_costs:
        ship_rate = df["channel_id"].map(cfg.channel_shipping_costs)
        ship_cost = ship_rate * qty
    elif cfg.default_shipping_cost_per_unit is not None:
        ship_cost = cfg.default_shipping_cost_per_unit * qty
    elif cfg.default_shipping_cost_per_order is not None:
        ship_cost = pd.Series(cfg.default_shipping_cost_per_order, index=df.index, dtype=float)

    # 2. Payment processing
    pay_cost = pd.Series(np.nan, index=df.index, dtype=float)
    if "payment_processing_cost" in df.columns or "payment_fee" in df.columns:
        pay_col = "payment_processing_cost" if "payment_processing_cost" in df.columns else "payment_fee"
        pay_cost = pd.to_numeric(df[pay_col], errors="coerce")
    else:
        # Check channel payment rates or default
        rates = df["channel_id"].map(cfg.channel_payment_processing_rates)
        if cfg.default_payment_processing_rate is not None:
            rates = rates.fillna(cfg.default_payment_processing_rate)
        fixed_fees = df["channel_id"].map(cfg.channel_payment_processing_fixed_fees)
        if cfg.default_payment_processing_fixed_fee is not None:
            fixed_fees = fixed_fees.fillna(cfg.default_payment_processing_fixed_fee)

        if rates.notna().any() or fixed_fees.notna().any():
            fee = pd.Series(0.0, index=df.index, dtype=float)
            has_rate = rates.notna() & is_valid & (net_rev > 0)
            if has_rate.any():
                fee.loc[has_rate] += rates.loc[has_rate] * net_rev.loc[has_rate]
            has_fixed = fixed_fees.notna()
            if has_fixed.any():
                fee.loc[has_fixed] += fixed_fees.loc[has_fixed]
            pay_cost = fee

    # 3. Packaging
    pkg_cost = pd.Series(np.nan, index=df.index, dtype=float)
    if "packaging_cost" in df.columns:
        pkg_cost = pd.to_numeric(df["packaging_cost"], errors="coerce")
    elif cfg.default_packaging_cost_per_unit is not None:
        pkg_cost = cfg.default_packaging_cost_per_unit * qty

    # 4. Warehouse handling
    wh_cost = pd.Series(np.nan, index=df.index, dtype=float)
    wh_col = "warehouse_handling_cost" if "warehouse_handling_cost" in df.columns else ("handling_cost" if "handling_cost" in df.columns else None)
    if wh_col is not None:
        wh_cost = pd.to_numeric(df[wh_col], errors="coerce")
    elif cfg.warehouse_handling_cost_per_unit:
        wh_rates = df["warehouse_id"].map(cfg.warehouse_handling_cost_per_unit)
        if cfg.default_warehouse_handling_cost_per_unit is not None:
            wh_rates = wh_rates.fillna(cfg.default_warehouse_handling_cost_per_unit)
        wh_cost = wh_rates * qty
    elif cfg.default_warehouse_handling_cost_per_unit is not None:
        wh_cost = cfg.default_warehouse_handling_cost_per_unit * qty

    # 5. Return processing
    ret_cost = pd.Series(np.nan, index=df.index, dtype=float)
    if "return_processing_cost" in df.columns:
        ret_cost = pd.to_numeric(df["return_processing_cost"], errors="coerce")
    elif cfg.default_return_processing_cost_per_unit is not None:
        ret_cost = cfg.default_return_processing_cost_per_unit * qty

    # 6. Other variable
    oth_cost = pd.Series(np.nan, index=df.index, dtype=float)
    if "other_variable_cost" in df.columns:
        oth_cost = pd.to_numeric(df["other_variable_cost"], errors="coerce")
    elif cfg.default_other_variable_cost_per_unit is not None:
        oth_cost = cfg.default_other_variable_cost_per_unit * qty

    # Total known variable costs: sum of non-null components
    var_cols = [ship_cost, pay_cost, pkg_cost, wh_cost, ret_cost, oth_cost]
    total_known_var = pd.Series(0.0, index=df.index, dtype=float)
    for c in var_cols:
        total_known_var += c.fillna(0.0)

    # Contribution margin
    known_cm = pd.Series(np.nan, index=df.index, dtype=float)
    known_cm_pct = pd.Series(np.nan, index=df.index, dtype=float)
    final_cm = pd.Series(np.nan, index=df.index, dtype=float)
    final_cm_pct = pd.Series(np.nan, index=df.index, dtype=float)

    if has_cost.any():
        kcm = net_rev.loc[cost_idx] - prod_cost.loc[cost_idx] - total_known_var.loc[cost_idx]
        known_cm.loc[cost_idx] = kcm

        pos_net = cost_idx[net_rev.loc[cost_idx] > 0]
        if len(pos_net) > 0:
            known_cm_pct.loc[pos_net] = kcm.loc[pos_net] / net_rev.loc[pos_net]

    # Required cost components completeness check
    req_keys = [c.value if isinstance(c, CostComponent) else str(c) for c in cfg.required_cost_components]
    comp_map = {
        CostComponent.PRODUCT_COST.value: unit_cost.notna(),
        CostComponent.SHIPPING_COST.value: ship_cost.notna(),
        CostComponent.PAYMENT_PROCESSING_COST.value: pay_cost.notna(),
        CostComponent.PACKAGING_COST.value: pkg_cost.notna(),
        CostComponent.WAREHOUSE_HANDLING_COST.value: wh_cost.notna(),
        CostComponent.RETURN_PROCESSING_COST.value: ret_cost.notna(),
        CostComponent.OTHER_VARIABLE_COST.value: oth_cost.notna(),
    }

    # Count of available required components per row
    available_req_count = pd.Series(0, index=df.index, dtype=int)
    for req_k in req_keys:
        if req_k in comp_map:
            available_req_count += comp_map[req_k].astype(int)

    completeness_pct = (available_req_count / len(req_keys)) * 100.0 if req_keys else pd.Series(100.0, index=df.index)

    all_req_present = available_req_count == len(req_keys)

    cm_status = pd.Series(ContributionMarginStatus.INSUFFICIENT_COST_DATA.value, index=df.index, dtype=object)
    econ_status = pd.Series(UnitEconomicsStatus.INSUFFICIENT_COST_DATA.value, index=df.index, dtype=object)
    rationale = pd.Series("Product procurement cost is missing in catalog", index=df.index, dtype=object)

    if is_invalid.any():
        econ_status.loc[is_invalid] = UnitEconomicsStatus.INVALID_DATA.value
        cm_status.loc[is_invalid] = ContributionMarginStatus.NOT_CALCULABLE.value
        rationale.loc[is_invalid] = "Invalid input values detected (negative or malformed fields)"

    if has_cost.any():
        idx_partial = cost_idx[~all_req_present.loc[cost_idx]]
        if len(idx_partial) > 0:
            econ_status.loc[idx_partial] = UnitEconomicsStatus.PARTIALLY_CALCULABLE.value
            cm_status.loc[idx_partial] = ContributionMarginStatus.INSUFFICIENT_COST_DATA.value
            rationale.loc[idx_partial] = "Product cost observed; missing required variable costs"

        idx_full = cost_idx[all_req_present.loc[cost_idx]]
        if len(idx_full) > 0:
            econ_status.loc[idx_full] = UnitEconomicsStatus.FULLY_CALCULABLE.value
            cm_status.loc[idx_full] = ContributionMarginStatus.CALCULABLE.value
            final_cm.loc[idx_full] = known_cm.loc[idx_full]
            final_cm_pct.loc[idx_full] = known_cm_pct.loc[idx_full]
            rationale.loc[idx_full] = "Complete unit economics: all required cost components available"

    # Deterministic record IDs
    raw_str = (
        "UE|"
        + df["sale_id"].str.strip()
        + "|"
        + df["sku_id"].str.strip()
        + "|"
        + df["order_id"].str.strip()
        + "|"
        + df["date"].str.strip()
        + "|"
        + df["currency"].str.strip().str.upper()
    )
    rec_ids = [
        f"FIN-UE-{hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]}"
        for s in raw_str
    ]

    out_df = pd.DataFrame(
        {
            "record_id": rec_ids,
            "sale_id": df["sale_id"],
            "order_id": df["order_id"],
            "date": df["date"],
            "sku_id": df["sku_id"],
            "warehouse_id": df["warehouse_id"],
            "channel_id": df["channel_id"],
            "currency": df["currency"],
            "quantity": qty,
            "unit_price": unit_price,
            "gross_revenue": gross_rev,
            "discount": discount,
            "net_revenue": net_rev,
            "source_revenue": source_rev,
            "unit_product_cost": unit_cost,
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
            "contribution_margin": final_cm,
            "contribution_margin_pct": final_cm_pct,
            "contribution_margin_status": cm_status,
            "cost_completeness_pct": completeness_pct,
            "economics_status": econ_status,
            "status_rationale": rationale,
            "as_of_date": str(cfg.as_of_date) if cfg.as_of_date else None,
        },
        index=df.index,
    )

    if "category_id" in df.columns:
        out_df["category_id"] = df["category_id"]
    if "brand" in df.columns:
        out_df["brand"] = df["brand"]

    return out_df


def compute_unit_economics_records(
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    config: Optional[CostModelConfig] = None,
) -> List[UnitEconomicsRecord]:
    """Compute unit economics and return as typed UnitEconomicsRecord objects."""
    df = compute_unit_economics_dataframe(sales_df, products_df, config)
    cfg = config or CostModelConfig()
    resolver = CostModelResolver(cfg)
    records: List[UnitEconomicsRecord] = []

    for row in df.itertuples(index=False):
        # Build individual cost details for the record
        qty_val = int(row.quantity) if pd.notna(row.quantity) else None
        price_val = float(row.unit_price) if pd.notna(row.unit_price) else None
        net_rev_val = float(row.net_revenue) if pd.notna(row.net_revenue) else None
        cost_val = float(row.unit_product_cost) if pd.notna(row.unit_product_cost) else None

        txn_costs = {
            "shipping_cost": float(row.shipping_cost) if pd.notna(row.shipping_cost) else None,
            "payment_processing_cost": float(row.payment_processing_cost) if pd.notna(row.payment_processing_cost) else None,
            "packaging_cost": float(row.packaging_cost) if pd.notna(row.packaging_cost) else None,
            "warehouse_handling_cost": float(row.warehouse_handling_cost) if pd.notna(row.warehouse_handling_cost) else None,
            "return_processing_cost": float(row.return_processing_cost) if pd.notna(row.return_processing_cost) else None,
            "other_variable_cost": float(row.other_variable_cost) if pd.notna(row.other_variable_cost) else None,
        }

        cost_details = resolver.resolve_line_costs(
            quantity=qty_val,
            unit_price=price_val,
            net_revenue=net_rev_val,
            sku_id=row.sku_id,
            channel_id=row.channel_id,
            warehouse_id=row.warehouse_id,
            unit_cost=cost_val,
            transaction_costs=txn_costs,
        )

        _, avail_c, missing_c, est_c, assumed_c = resolver.calculate_cost_completeness(cost_details)

        records.append(
            UnitEconomicsRecord(
                record_id=row.record_id,
                sale_id=row.sale_id,
                order_id=row.order_id,
                date=row.date,
                sku_id=row.sku_id,
                warehouse_id=row.warehouse_id,
                channel_id=row.channel_id,
                currency=row.currency,
                quantity=qty_val,
                unit_price=price_val,
                gross_revenue=float(row.gross_revenue) if pd.notna(row.gross_revenue) else None,
                discount=float(row.discount) if pd.notna(row.discount) else None,
                net_revenue=net_rev_val,
                source_revenue=float(row.source_revenue) if pd.notna(row.source_revenue) else None,
                unit_product_cost=cost_val,
                product_cost=float(row.product_cost) if pd.notna(row.product_cost) else None,
                gross_margin=float(row.gross_margin) if pd.notna(row.gross_margin) else None,
                gross_margin_pct=float(row.gross_margin_pct) if pd.notna(row.gross_margin_pct) else None,
                shipping_cost=float(row.shipping_cost) if pd.notna(row.shipping_cost) else None,
                payment_processing_cost=float(row.payment_processing_cost) if pd.notna(row.payment_processing_cost) else None,
                packaging_cost=float(row.packaging_cost) if pd.notna(row.packaging_cost) else None,
                warehouse_handling_cost=float(row.warehouse_handling_cost) if pd.notna(row.warehouse_handling_cost) else None,
                return_processing_cost=float(row.return_processing_cost) if pd.notna(row.return_processing_cost) else None,
                other_variable_cost=float(row.other_variable_cost) if pd.notna(row.other_variable_cost) else None,
                total_known_variable_cost=float(row.total_known_variable_cost) if pd.notna(row.total_known_variable_cost) else 0.0,
                known_contribution_margin=float(row.known_contribution_margin) if pd.notna(row.known_contribution_margin) else None,
                known_contribution_margin_pct=float(row.known_contribution_margin_pct) if pd.notna(row.known_contribution_margin_pct) else None,
                contribution_margin=float(row.contribution_margin) if pd.notna(row.contribution_margin) else None,
                contribution_margin_pct=float(row.contribution_margin_pct) if pd.notna(row.contribution_margin_pct) else None,
                contribution_margin_status=ContributionMarginStatus(row.contribution_margin_status),
                cost_completeness_pct=float(row.cost_completeness_pct) if pd.notna(row.cost_completeness_pct) else 0.0,
                required_cost_components=[c.value for c in cfg.required_cost_components],
                available_cost_components=avail_c,
                unavailable_cost_components=missing_c,
                estimated_cost_components=est_c,
                assumed_cost_components=assumed_c,
                economics_status=UnitEconomicsStatus(row.economics_status),
                status_rationale=row.status_rationale,
                as_of_date=row.as_of_date,
                cost_details=cost_details,
            )
        )
    return records


# =====================================================================
# Aggregation Engine
# =====================================================================


def aggregate_unit_economics_dimension(
    df: pd.DataFrame,
    dimension: str,
    total_portfolio_revenue: Optional[float] = None,
    total_portfolio_margin: Optional[float] = None,
    config: Optional[CostModelConfig] = None,
) -> List[UnitEconomicsDimensionMetric]:
    """Aggregate unit economics metrics across any supported dimensional slice.

    Supported dimensions:
    - OVERALL
    - SKU
    - CATEGORY
    - BRAND
    - CHANNEL
    - WAREHOUSE
    - SKU_CHANNEL (or SKU_X_CHANNEL)
    - SKU_WAREHOUSE (or SKU_X_WAREHOUSE)
    - CHANNEL_WAREHOUSE (or CHANNEL_X_WAREHOUSE)
    - DATE
    - MONTH (YYYY-MM)
    """
    cfg = config or CostModelConfig()
    dim_upper = dimension.strip().upper().replace(" ", "_")

    if df.empty:
        return []

    # Map dimension to group columns
    if dim_upper == "OVERALL":
        group_cols = None
    elif dim_upper == "SKU":
        group_cols = ["sku_id"]
    elif dim_upper == "CATEGORY":
        group_cols = ["category_id"] if "category_id" in df.columns else ["sku_id"]
    elif dim_upper == "BRAND":
        group_cols = ["brand"] if "brand" in df.columns else ["sku_id"]
    elif dim_upper == "CHANNEL":
        group_cols = ["channel_id"]
    elif dim_upper == "WAREHOUSE":
        group_cols = ["warehouse_id"]
    elif dim_upper in ("SKU_CHANNEL", "SKU_X_CHANNEL"):
        group_cols = ["sku_id", "channel_id"]
    elif dim_upper in ("SKU_WAREHOUSE", "SKU_X_WAREHOUSE"):
        group_cols = ["sku_id", "warehouse_id"]
    elif dim_upper in ("CHANNEL_WAREHOUSE", "CHANNEL_X_WAREHOUSE"):
        group_cols = ["channel_id", "warehouse_id"]
    elif dim_upper == "DATE":
        group_cols = ["date"]
    elif dim_upper == "MONTH":
        df = df.copy()
        df["month"] = df["date"].astype(str).str.slice(0, 7)
        group_cols = ["month"]
    else:
        raise ValueError(f"Unsupported unit economics dimension: {dimension}")

    def compute_slice_metrics(sub_df: pd.DataFrame, seg_key: str) -> UnitEconomicsDimensionMetric:
        rec_count = len(sub_df)
        order_count = int(sub_df["order_id"].nunique()) if "order_id" in sub_df.columns else 0

        # Physical units
        qty_series = sub_df["quantity"].dropna()
        total_units = int(qty_series.sum()) if not qty_series.empty else 0

        # Core revenue & product cost
        gross_series = sub_df["gross_revenue"].dropna()
        total_gross = float(gross_series.sum()) if not gross_series.empty else None

        disc_series = sub_df["discount"].dropna()
        total_disc = float(disc_series.sum()) if not disc_series.empty else None

        net_series = sub_df["net_revenue"].dropna()
        total_net = float(net_series.sum()) if not net_series.empty else None

        prod_series = sub_df["product_cost"].dropna()
        total_prod_cost = float(prod_series.sum()) if not prod_series.empty else None

        margin_series = sub_df["gross_margin"].dropna()
        total_margin = float(margin_series.sum()) if not margin_series.empty else None

        # Gross margin %
        if total_net is not None and total_net > 0 and total_margin is not None:
            gm_pct = float(total_margin / total_net)
        else:
            gm_pct = None

        # Variable costs
        def sum_or_none(col_name: str) -> Optional[float]:
            s = sub_df[col_name].dropna()
            return float(s.sum()) if not s.empty else None

        tot_ship = sum_or_none("shipping_cost")
        tot_pay = sum_or_none("payment_processing_cost")
        tot_pkg = sum_or_none("packaging_cost")
        tot_wh = sum_or_none("warehouse_handling_cost")
        tot_ret = sum_or_none("return_processing_cost")
        tot_oth = sum_or_none("other_variable_cost")

        tot_known_var = float(sub_df["total_known_variable_cost"].dropna().sum()) if not sub_df["total_known_variable_cost"].dropna().empty else 0.0

        # Known contribution margin
        kcm_series = sub_df["known_contribution_margin"].dropna()
        total_kcm = float(kcm_series.sum()) if not kcm_series.empty else None

        if total_net is not None and total_net > 0 and total_kcm is not None:
            kcm_pct = float(total_kcm / total_net)
        else:
            kcm_pct = None

        # Final contribution margin (only when fully calculable across slice)
        fully_calc_mask = sub_df["economics_status"] == UnitEconomicsStatus.FULLY_CALCULABLE.value
        all_fully_calc = bool(fully_calc_mask.all()) and rec_count > 0

        if all_fully_calc:
            cm_series = sub_df["contribution_margin"].dropna()
            total_cm = float(cm_series.sum()) if not cm_series.empty else None
            cm_pct = float(total_cm / total_net) if (total_net is not None and total_net > 0 and total_cm is not None) else None
            cm_status = ContributionMarginStatus.CALCULABLE
            econ_status = UnitEconomicsStatus.FULLY_CALCULABLE
        else:
            total_cm = None
            cm_pct = None
            cm_status = ContributionMarginStatus.INSUFFICIENT_COST_DATA
            econ_status = UnitEconomicsStatus.PARTIALLY_CALCULABLE

        # Completeness & averages
        comp_series = sub_df["cost_completeness_pct"].dropna()
        avg_comp = float(comp_series.mean()) if not comp_series.empty else 0.0

        aov = float(total_net / order_count) if (total_net is not None and order_count > 0) else None
        aur = float(total_net / total_units) if (total_net is not None and total_units > 0) else None
        auc = float(total_prod_cost / total_units) if (total_prod_cost is not None and total_units > 0) else None

        rev_contrib = float(total_net / total_portfolio_revenue) if (total_net is not None and total_portfolio_revenue and total_portfolio_revenue > 0) else None
        margin_contrib = float(total_margin / total_portfolio_margin) if (total_margin is not None and total_portfolio_margin and abs(total_portfolio_margin) > 0) else None

        currency = str(sub_df["currency"].iloc[0]) if "currency" in sub_df.columns and not sub_df["currency"].empty else cfg.default_currency

        return UnitEconomicsDimensionMetric(
            dimension=dim_upper,
            segment_key=seg_key,
            record_count=rec_count,
            order_count=order_count,
            total_units=total_units,
            total_gross_revenue=total_gross,
            total_discount=total_disc,
            total_net_revenue=total_net,
            total_product_cost=total_prod_cost,
            total_gross_margin=total_margin,
            gross_margin_pct=gm_pct,
            total_shipping_cost=tot_ship,
            total_payment_processing_cost=tot_pay,
            total_packaging_cost=tot_pkg,
            total_warehouse_handling_cost=tot_wh,
            total_return_processing_cost=tot_ret,
            total_other_variable_cost=tot_oth,
            total_known_variable_cost=tot_known_var,
            total_known_contribution_margin=total_kcm,
            known_contribution_margin_pct=kcm_pct,
            total_contribution_margin=total_cm,
            contribution_margin_pct=cm_pct,
            contribution_margin_status=cm_status,
            cost_completeness_pct=avg_comp,
            average_order_value=aov,
            average_unit_revenue=aur,
            average_unit_cost=auc,
            revenue_contribution=rev_contrib,
            margin_contribution=margin_contrib,
            economics_status=econ_status,
            currency=currency,
        )

    results: List[UnitEconomicsDimensionMetric] = []

    if group_cols is None:
        results.append(compute_slice_metrics(df, "OVERALL"))
    else:
        grouped = df.groupby(group_cols, dropna=False, sort=True)
        for name, sub in grouped:
            if isinstance(name, tuple):
                key = ":".join(str(part or "") for part in name)
            else:
                key = str(name or "")
            results.append(compute_slice_metrics(sub, key))

    return results


def aggregate_unit_economics_time_series(
    df: pd.DataFrame,
    grain: Union[str, TimeGrain] = TimeGrain.MONTHLY,
    total_portfolio_revenue: Optional[float] = None,
    total_portfolio_margin: Optional[float] = None,
    config: Optional[CostModelConfig] = None,
) -> List[UnitEconomicsDimensionMetric]:
    """Aggregate unit economics across temporal granularity (DAILY, WEEKLY, MONTHLY)."""
    g_str = grain.value if isinstance(grain, TimeGrain) else str(grain).strip().upper()
    df_copy = df.copy()

    if df_copy.empty:
        return []

    date_series = pd.to_datetime(df_copy["date"], errors="coerce")

    if g_str in ("DAILY", "DAY", "D"):
        df_copy["time_grain_key"] = date_series.dt.strftime("%Y-%m-%d")
        dim_label = "DATE"
    elif g_str in ("WEEKLY", "WEEK", "W"):
        df_copy["time_grain_key"] = date_series.dt.strftime("%Y-W%U")
        dim_label = "WEEK"
    elif g_str in ("MONTHLY", "MONTH", "M"):
        df_copy["time_grain_key"] = date_series.dt.strftime("%Y-%m")
        dim_label = "MONTH"
    else:
        raise ValueError(f"Unsupported temporal grain: {grain}")

    grouped = df_copy.groupby("time_grain_key", dropna=False, sort=True)
    results: List[UnitEconomicsDimensionMetric] = []

    for name, sub in grouped:
        key = str(name or "")
        metric = aggregate_unit_economics_dimension(
            sub,
            dimension="OVERALL",
            total_portfolio_revenue=total_portfolio_revenue,
            total_portfolio_margin=total_portfolio_margin,
            config=config,
        )[0]
        metric.dimension = dim_label
        metric.segment_key = key
        results.append(metric)

    return results


# =====================================================================
# Margin Erosion Analysis (Descriptive)
# =====================================================================


def analyze_margin_erosion(
    sku_metrics: Sequence[UnitEconomicsDimensionMetric],
    channel_metrics: Sequence[UnitEconomicsDimensionMetric],
    warehouse_metrics: Sequence[UnitEconomicsDimensionMetric],
    portfolio_margin_pct: Optional[float] = None,
    low_margin_threshold: float = 0.20,
    mismatch_threshold: float = 0.02,
) -> MarginErosionReport:
    """Analyze margin erosion patterns across SKUs, channels, and warehouses.

    Descriptive, factual analysis only. Does not generate ML predictions or causal claims.
    """
    negative_skus: List[MarginErosionSegment] = []
    low_margin_skus: List[MarginErosionSegment] = []
    high_discount_skus: List[MarginErosionSegment] = []
    mismatches: List[MarginErosionSegment] = []

    for m in sku_metrics:
        rev = m.total_net_revenue or 0.0
        gm = m.total_gross_margin or 0.0
        rev_share = m.revenue_contribution or 0.0
        margin_share = m.margin_contribution or 0.0
        gm_pct = m.gross_margin_pct

        # 1. Negative margin
        if gm < 0:
            negative_skus.append(
                MarginErosionSegment(
                    segment_dimension="SKU",
                    segment_key=m.segment_key,
                    revenue=rev,
                    revenue_share=rev_share,
                    gross_margin=gm,
                    margin_share=margin_share,
                    gross_margin_pct=gm_pct,
                    erosion_type="NEGATIVE_MARGIN",
                    description=f"Negative gross margin (${gm:,.2f}) on ${rev:,.2f} net revenue",
                )
            )

        # 2. Low gross margin
        elif gm_pct is not None and gm_pct < low_margin_threshold:
            low_margin_skus.append(
                MarginErosionSegment(
                    segment_dimension="SKU",
                    segment_key=m.segment_key,
                    revenue=rev,
                    revenue_share=rev_share,
                    gross_margin=gm,
                    margin_share=margin_share,
                    gross_margin_pct=gm_pct,
                    erosion_type="LOW_MARGIN",
                    description=f"Gross margin percentage ({gm_pct*100:.1f}%) below threshold ({low_margin_threshold*100:.1f}%)",
                )
            )

        # 3. High promotional discount dilution
        tot_gross = m.total_gross_revenue or 0.0
        tot_disc = m.total_discount or 0.0
        if tot_gross > 0 and (tot_disc / tot_gross) > 0.25:
            disc_rate = tot_disc / tot_gross
            high_discount_skus.append(
                MarginErosionSegment(
                    segment_dimension="SKU",
                    segment_key=m.segment_key,
                    revenue=rev,
                    revenue_share=rev_share,
                    gross_margin=gm,
                    margin_share=margin_share,
                    gross_margin_pct=gm_pct,
                    erosion_type="HIGH_DISCOUNT",
                    description=f"Promotional discount rate ({disc_rate*100:.1f}%) exceeds 25% of gross revenue",
                )
            )

        # 4. Revenue vs Margin Contribution Mismatch (for positive, non-low-margin SKUs)
        # E.g. SKU generates 5% of revenue but only 1% of gross margin
        if gm >= 0 and (gm_pct is None or gm_pct >= low_margin_threshold):
            if rev_share - margin_share > mismatch_threshold:
                mismatches.append(
                MarginErosionSegment(
                    segment_dimension="SKU",
                    segment_key=m.segment_key,
                    revenue=rev,
                    revenue_share=rev_share,
                    gross_margin=gm,
                    margin_share=margin_share,
                    gross_margin_pct=gm_pct,
                    erosion_type="CONTRIBUTION_MISMATCH",
                    description=f"Revenue share ({rev_share*100:.2f}%) significantly outpaces margin share ({margin_share*100:.2f}%) by {(rev_share - margin_share)*100:.2f}%",
                )
            )

    # Low margin channels (below portfolio average)
    low_channels: List[MarginErosionSegment] = []
    avg_gm = portfolio_margin_pct or 0.50
    for ch in channel_metrics:
        rev = ch.total_net_revenue or 0.0
        gm = ch.total_gross_margin or 0.0
        gm_pct = ch.gross_margin_pct
        if gm_pct is not None and gm_pct < avg_gm:
            low_channels.append(
                MarginErosionSegment(
                    segment_dimension="CHANNEL",
                    segment_key=ch.segment_key,
                    revenue=rev,
                    revenue_share=ch.revenue_contribution or 0.0,
                    gross_margin=gm,
                    margin_share=ch.margin_contribution or 0.0,
                    gross_margin_pct=gm_pct,
                    erosion_type="LOW_MARGIN",
                    description=f"Channel gross margin ({gm_pct*100:.1f}%) is below portfolio average ({avg_gm*100:.1f}%)",
                )
            )

    # Low margin warehouses (below portfolio average)
    low_warehouses: List[MarginErosionSegment] = []
    for wh in warehouse_metrics:
        rev = wh.total_net_revenue or 0.0
        gm = wh.total_gross_margin or 0.0
        gm_pct = wh.gross_margin_pct
        if gm_pct is not None and gm_pct < avg_gm:
            low_warehouses.append(
                MarginErosionSegment(
                    segment_dimension="WAREHOUSE",
                    segment_key=wh.segment_key,
                    revenue=rev,
                    revenue_share=wh.revenue_contribution or 0.0,
                    gross_margin=gm,
                    margin_share=wh.margin_contribution or 0.0,
                    gross_margin_pct=gm_pct,
                    erosion_type="LOW_MARGIN",
                    description=f"Warehouse gross margin ({gm_pct*100:.1f}%) is below portfolio average ({avg_gm*100:.1f}%)",
                )
            )

    # Sort descriptive lists
    negative_skus.sort(key=lambda x: x.gross_margin)
    low_margin_skus.sort(key=lambda x: x.gross_margin_pct or 0.0)
    high_discount_skus.sort(key=lambda x: x.revenue, reverse=True)
    mismatches.sort(key=lambda x: (x.revenue_share - x.margin_share), reverse=True)

    summary_notes = [
        f"Identified {len(negative_skus)} SKUs with negative gross margin",
        f"Identified {len(low_margin_skus)} SKUs with gross margin < {low_margin_threshold*100:.0f}%",
        f"Identified {len(high_discount_skus)} SKUs with promotional discounts exceeding 25%",
        f"Identified {len(mismatches)} SKUs with significant margin dilution (revenue share outpaces margin share)",
        f"Identified {len(low_channels)} channels and {len(low_warehouses)} warehouses performing below network average gross margin",
    ]

    return MarginErosionReport(
        negative_margin_skus=negative_skus,
        low_margin_skus=low_margin_skus,
        high_discount_skus=high_discount_skus,
        low_margin_channels=low_channels,
        low_margin_warehouses=low_warehouses,
        margin_contribution_mismatches=mismatches,
        summary_notes=summary_notes,
    )


# =====================================================================
# Executive Portfolio Summary
# =====================================================================


def summarize_unit_economics_portfolio(
    df: pd.DataFrame,
    quality_report: Optional[FinancialDataQualityReport] = None,
    config: Optional[CostModelConfig] = None,
) -> UnitEconomicsPortfolioSummary:
    """Produce executive portfolio-level unit economics totals and completeness metrics."""
    cfg = config or CostModelConfig()
    total_lines = len(df)

    if total_lines == 0:
        return UnitEconomicsPortfolioSummary(
            total_order_lines=0,
            total_orders=0,
            total_units=0,
            total_known_variable_cost=0.0,
            cost_completeness_pct=0.0,
            economics_status=UnitEconomicsStatus.INSUFFICIENT_COST_DATA,
            currency=cfg.default_currency,
            as_of_date=str(cfg.as_of_date) if cfg.as_of_date else None,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    total_orders = int(df["order_id"].nunique()) if "order_id" in df.columns else 0

    qty_s = df["quantity"].dropna()
    total_units = int(qty_s.sum()) if not qty_s.empty else 0

    gross_s = df["gross_revenue"].dropna()
    total_gross = float(gross_s.sum()) if not gross_s.empty else None

    disc_s = df["discount"].dropna()
    total_disc = float(disc_s.sum()) if not disc_s.empty else None

    net_s = df["net_revenue"].dropna()
    total_net = float(net_s.sum()) if not net_s.empty else None

    prod_s = df["product_cost"].dropna()
    total_prod = float(prod_s.sum()) if not prod_s.empty else None

    margin_s = df["gross_margin"].dropna()
    total_margin = float(margin_s.sum()) if not margin_s.empty else None

    gm_pct = float(total_margin / total_net) if (total_net is not None and total_net > 0 and total_margin is not None) else None

    def sum_or_none(col: str) -> Optional[float]:
        s = df[col].dropna()
        return float(s.sum()) if not s.empty else None

    tot_ship = sum_or_none("shipping_cost")
    tot_pay = sum_or_none("payment_processing_cost")
    tot_pkg = sum_or_none("packaging_cost")
    tot_wh = sum_or_none("warehouse_handling_cost")
    tot_ret = sum_or_none("return_processing_cost")
    tot_oth = sum_or_none("other_variable_cost")

    tot_known_var = float(df["total_known_variable_cost"].dropna().sum()) if not df["total_known_variable_cost"].dropna().empty else 0.0

    kcm_s = df["known_contribution_margin"].dropna()
    total_kcm = float(kcm_s.sum()) if not kcm_s.empty else None
    kcm_pct = float(total_kcm / total_net) if (total_net is not None and total_net > 0 and total_kcm is not None) else None

    # Final contribution margin
    fully_calc_mask = df["economics_status"] == UnitEconomicsStatus.FULLY_CALCULABLE.value
    all_fully_calc = bool(fully_calc_mask.all()) and total_lines > 0

    if all_fully_calc:
        cm_s = df["contribution_margin"].dropna()
        total_cm = float(cm_s.sum()) if not cm_s.empty else None
        cm_pct = float(total_cm / total_net) if (total_net is not None and total_net > 0 and total_cm is not None) else None
        cm_status = ContributionMarginStatus.CALCULABLE
        econ_status = UnitEconomicsStatus.FULLY_CALCULABLE
    else:
        total_cm = None
        cm_pct = None
        cm_status = ContributionMarginStatus.INSUFFICIENT_COST_DATA
        econ_status = UnitEconomicsStatus.PARTIALLY_CALCULABLE

    comp_s = df["cost_completeness_pct"].dropna()
    avg_comp = float(comp_s.mean()) if not comp_s.empty else 0.0

    aov = float(total_net / total_orders) if (total_net is not None and total_orders > 0) else None
    aur = float(total_net / total_units) if (total_net is not None and total_units > 0) else None
    auc = float(total_prod / total_units) if (total_prod is not None and total_units > 0) else None

    rec_with_cost = int(df["unit_product_cost"].notna().sum())
    rec_without_cost = total_lines - rec_with_cost

    currency = str(df["currency"].iloc[0]) if "currency" in df.columns and not df["currency"].empty else cfg.default_currency

    return UnitEconomicsPortfolioSummary(
        total_order_lines=total_lines,
        total_orders=total_orders,
        total_units=total_units,
        total_gross_revenue=total_gross,
        total_discount=total_disc,
        total_net_revenue=total_net,
        total_product_cost=total_prod,
        total_gross_margin=total_margin,
        gross_margin_pct=gm_pct,
        total_shipping_cost=tot_ship,
        total_payment_processing_cost=tot_pay,
        total_packaging_cost=tot_pkg,
        total_warehouse_handling_cost=tot_wh,
        total_return_processing_cost=tot_ret,
        total_other_variable_cost=tot_oth,
        total_known_variable_cost=tot_known_var,
        total_known_contribution_margin=total_kcm,
        known_contribution_margin_pct=kcm_pct,
        total_contribution_margin=total_cm,
        contribution_margin_pct=cm_pct,
        contribution_margin_status=cm_status,
        cost_completeness_pct=avg_comp,
        economics_status=econ_status,
        average_order_value=aov,
        average_unit_revenue=aur,
        average_unit_cost=auc,
        records_with_product_cost=rec_with_cost,
        records_without_product_cost=rec_without_cost,
        currency=currency,
        as_of_date=str(cfg.as_of_date) if cfg.as_of_date else None,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
