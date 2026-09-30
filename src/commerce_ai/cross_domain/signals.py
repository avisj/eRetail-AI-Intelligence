"""Deterministic Cross-Domain Signal Engine (Phase 6E).

Evaluates cross-domain business rules against CrossDomainBusinessRecord objects.
All generated signals are:
- Strictly DESCRIPTIVE and NON-CAUSAL (no causal assertions, no recommendations)
- Deterministic with reproducible hashes for signal IDs
- Categorized into standard SignalCategory values
- Assigned rule-based SignalSeverity levels (INFO, LOW, MEDIUM, HIGH, CRITICAL)
- Tested against forbidden causal terms ("caused", "resulted in", "because of", "drove", "responsible for", "best", "worst")
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from commerce_ai.cross_domain.schemas import (
    CrossDomainBusinessRecord,
    CrossDomainIntelligenceConfig,
    CrossDomainSignal,
    SignalCategory,
    SignalSeverity,
)

FORBIDDEN_CAUSAL_WORDS = {
    "caused",
    "resulted in",
    "because of",
    "drove",
    "responsible for",
    "best",
    "worst",
    "recommend",
    "recommended",
    "should",
}


def generate_signal_id(
    signal_type: str,
    dimension: str,
    sku_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    as_of_date: Optional[str] = None,
) -> str:
    """Generate a deterministic, reproducible signal identifier."""
    key = f"{signal_type}:{dimension}:{sku_id or ''}:{warehouse_id or ''}:{channel_id or ''}:{as_of_date or ''}"
    hash_hex = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"SIG-{hash_hex}"


def _verify_non_causal_text(text: str) -> None:
    """Audit signal description to guarantee no forbidden causal or ranking words appear."""
    lower_text = text.lower()
    for word in FORBIDDEN_CAUSAL_WORDS:
        if word in lower_text:
            raise ValueError(f"Forbidden causal/recommendation term '{word}' found in signal description: '{text}'")


def evaluate_signals_for_record(
    record: CrossDomainBusinessRecord,
    config: CrossDomainIntelligenceConfig,
    high_revenue_cutoff: float,
    high_margin_cutoff: float,
    high_inv_val_cutoff: float,
    high_demand_cutoff: float,
) -> List[CrossDomainSignal]:
    """Evaluate deterministic cross-domain rules for a single business record."""
    signals: List[CrossDomainSignal] = []

    dim = "SKU_WAREHOUSE"
    if record.sku_id and record.warehouse_id:
        dim = "SKU_WAREHOUSE"
    elif record.sku_id and not record.warehouse_id and not record.channel_id:
        dim = "SKU"
    elif record.warehouse_id and not record.sku_id:
        dim = "WAREHOUSE"
    elif record.channel_id and not record.sku_id:
        dim = "CHANNEL"

    # Extract metrics safely
    net_rev = record.net_revenue
    gross_mar = record.gross_margin
    gross_mar_pct = record.gross_margin_pct
    avail_inv = record.available_inventory
    on_hand = record.current_on_hand
    inv_val = record.inventory_value
    avg_demand = record.average_daily_demand
    so_days = record.stockout_days
    so_rate = record.stockout_rate
    so_flag = record.stockout_flag
    ret_rate = record.return_rate
    ret_anomaly = record.return_anomaly_flag
    re为其 = record.replenishment_trigger
    reorder_trigger = record.replenishment_trigger
    fc_mean = record.forecast_mean
    fc_avail = record.forecast_available
    units_sold = record.units_sold or 0
    order_cnt = record.order_count or 0

    has_sample = units_sold >= config.min_sample_size_units and order_cnt >= config.min_sample_size_orders

    def add_signal(
        signal_type: str,
        category: SignalCategory,
        severity: SignalSeverity,
        metrics: Dict[str, Any],
        reasons: List[str],
        description: str,
    ) -> None:
        _verify_non_causal_text(description)
        sig_id = generate_signal_id(
            signal_type=signal_type,
            dimension=dim,
            sku_id=record.sku_id,
            warehouse_id=record.warehouse_id,
            channel_id=record.channel_id,
            as_of_date=record.as_of_date,
        )
        signals.append(
            CrossDomainSignal(
                signal_id=sig_id,
                signal_type=signal_type,
                category=category,
                severity=severity,
                dimension=dim,
                sku_id=record.sku_id,
                warehouse_id=record.warehouse_id,
                channel_id=record.channel_id,
                observed_metrics=metrics,
                reason_codes=reasons,
                description=description,
                data_quality=record.data_quality_status,
                as_of_date=record.as_of_date,
            )
        )

    # 1. HIGH_REVENUE_LOW_STOCK
    if net_rev is not None and net_rev >= high_revenue_cutoff and avail_inv is not None and avail_inv <= config.low_stock_threshold:
        sev = SignalSeverity.CRITICAL if avail_inv <= 0 else SignalSeverity.HIGH
        desc = (
            f"Observed net revenue of {net_rev:.2f} coincides with available inventory "
            f"of {avail_inv} units at or below threshold {config.low_stock_threshold}."
        )
        add_signal(
            "HIGH_REVENUE_LOW_STOCK",
            SignalCategory.CROSS_DOMAIN,
            sev,
            {"net_revenue": net_rev, "available_inventory": avail_inv},
            ["NET_REVENUE_GE_HIGH_CUTOFF", "AVAILABLE_INVENTORY_LE_LOW_THRESHOLD"],
            desc,
        )

    # 2. HIGH_MARGIN_LOW_STOCK
    if gross_mar is not None and gross_mar >= high_margin_cutoff and avail_inv is not None and avail_inv <= config.low_stock_threshold:
        desc = (
            f"Observed gross margin of {gross_mar:.2f} coincides with available inventory "
            f"of {avail_inv} units at or below threshold {config.low_stock_threshold}."
        )
        add_signal(
            "HIGH_MARGIN_LOW_STOCK",
            SignalCategory.CROSS_DOMAIN,
            SignalSeverity.HIGH,
            {"gross_margin": gross_mar, "available_inventory": avail_inv},
            ["GROSS_MARGIN_GE_HIGH_CUTOFF", "AVAILABLE_INVENTORY_LE_LOW_THRESHOLD"],
            desc,
        )

    # 3. HIGH_REVENUE_STOCKOUT_EXPOSURE
    is_so = (so_flag is True) or (so_days is not None and so_days > 0) or (so_rate is not None and so_rate >= config.stockout_rate_threshold)
    if net_rev is not None and net_rev >= high_revenue_cutoff and is_so:
        desc = (
            f"Observed net revenue of {net_rev:.2f} coincides with stockout exposure "
            f"({so_days or 0} stockout days, rate {so_rate or 0.0:.2%})."
        )
        add_signal(
            "HIGH_REVENUE_STOCKOUT_EXPOSURE",
            SignalCategory.CROSS_DOMAIN,
            SignalSeverity.CRITICAL,
            {"net_revenue": net_rev, "stockout_days": so_days, "stockout_rate": so_rate},
            ["NET_REVENUE_GE_HIGH_CUTOFF", "STOCKOUT_CONDITION_PRESENT"],
            desc,
        )

    # 4. LOW_MARGIN_HIGH_INVENTORY
    if gross_mar_pct is not None and gross_mar_pct <= config.low_margin_pct_threshold:
        elevated_inv = (inv_val is not None and inv_val >= high_inv_val_cutoff) or (on_hand is not None and on_hand >= config.high_inventory_units_threshold)
        if elevated_inv:
            desc = (
                f"Observed gross margin percentage of {gross_mar_pct:.1%} co-occurs with "
                f"elevated inventory ({on_hand} on-hand units, valuation {inv_val or 0.0:.2f})."
            )
            add_signal(
                "LOW_MARGIN_HIGH_INVENTORY",
                SignalCategory.CROSS_DOMAIN,
                SignalSeverity.MEDIUM,
                {"gross_margin_pct": gross_mar_pct, "inventory_value": inv_val, "on_hand": on_hand},
                ["GROSS_MARGIN_PCT_LE_THRESHOLD", "INVENTORY_ELEVATED"],
                desc,
            )

    # 5. HIGH_INVENTORY_LOW_DEMAND
    is_low_demand = (avg_demand is not None and avg_demand <= config.low_velocity_threshold) or (record.abc_class == "C")
    if on_hand is not None and on_hand >= config.high_inventory_units_threshold and is_low_demand:
        desc = (
            f"Observed physical inventory of {on_hand} units co-occurs with low daily demand "
            f"({avg_demand or 0.0:.2f} units/day, ABC class {record.abc_class or 'N/A'})."
        )
        add_signal(
            "HIGH_INVENTORY_LOW_DEMAND",
            SignalCategory.CROSS_DOMAIN,
            SignalSeverity.MEDIUM,
            {"on_hand": on_hand, "average_daily_demand": avg_demand, "abc_class": record.abc_class},
            ["ON_HAND_GE_THRESHOLD", "LOW_DEMAND_VELOCITY"],
            desc,
        )

    # 6. HIGH_RETURN_HIGH_REVENUE
    if has_sample and ret_rate is not None and ret_rate >= config.high_return_rate_threshold and net_rev is not None and net_rev >= high_revenue_cutoff:
        desc = (
            f"Observed return rate of {ret_rate:.2%} is associated with elevated net revenue "
            f"of {net_rev:.2f}."
        )
        add_signal(
            "HIGH_RETURN_HIGH_REVENUE",
            SignalCategory.CROSS_DOMAIN,
            SignalSeverity.HIGH,
            {"return_rate": ret_rate, "net_revenue": net_rev},
            ["RETURN_RATE_GE_THRESHOLD", "NET_REVENUE_GE_HIGH_CUTOFF", "SUFFICIENT_SAMPLE"],
            desc,
        )

    # 7. HIGH_RETURN_LOW_MARGIN
    if has_sample and ret_rate is not None and ret_rate >= config.high_return_rate_threshold and gross_mar_pct is not None and gross_mar_pct <= config.low_margin_pct_threshold:
        desc = (
            f"Observed return rate of {ret_rate:.2%} co-occurs with lower gross margin percentage "
            f"of {gross_mar_pct:.1%}."
        )
        add_signal(
            "HIGH_RETURN_LOW_MARGIN",
            SignalCategory.CROSS_DOMAIN,
            SignalSeverity.HIGH,
            {"return_rate": ret_rate, "gross_margin_pct": gross_mar_pct},
            ["RETURN_RATE_GE_THRESHOLD", "GROSS_MARGIN_PCT_LE_THRESHOLD", "SUFFICIENT_SAMPLE"],
            desc,
        )

    # 8. HIGH_MARGIN_HIGH_RETURN
    if has_sample and ret_rate is not None and ret_rate >= config.high_return_rate_threshold and gross_mar_pct is not None and gross_mar_pct >= 0.50:
        desc = (
            f"Observed return rate of {ret_rate:.2%} is observed alongside high gross margin percentage "
            f"of {gross_mar_pct:.1%}."
        )
        add_signal(
            "HIGH_MARGIN_HIGH_RETURN",
            SignalCategory.CROSS_DOMAIN,
            SignalSeverity.MEDIUM,
            {"return_rate": ret_rate, "gross_margin_pct": gross_mar_pct},
            ["RETURN_RATE_GE_THRESHOLD", "GROSS_MARGIN_PCT_HIGH", "SUFFICIENT_SAMPLE"],
            desc,
        )

    # 9. LOW_REVENUE_HIGH_INVENTORY
    if net_rev is not None and net_rev < high_revenue_cutoff and inv_val is not None and inv_val >= high_inv_val_cutoff:
        desc = (
            f"Observed inventory valuation of {inv_val:.2f} coincides with lower net revenue "
            f"of {net_rev:.2f}."
        )
        add_signal(
            "LOW_REVENUE_HIGH_INVENTORY",
            SignalCategory.CROSS_DOMAIN,
            SignalSeverity.MEDIUM,
            {"inventory_value": inv_val, "net_revenue": net_rev},
            ["INVENTORY_VALUE_GE_HIGH_CUTOFF", "NET_REVENUE_BELOW_HIGH_CUTOFF"],
            desc,
        )

    # 10. HIGH_VALUE_INVENTORY
    if inv_val is not None and inv_val >= high_inv_val_cutoff:
        desc = (
            f"Inventory valuation of {inv_val:.2f} ranks in the upper percentile of the portfolio."
        )
        add_signal(
            "HIGH_VALUE_INVENTORY",
            SignalCategory.INVENTORY,
            SignalSeverity.INFO,
            {"inventory_value": inv_val},
            ["INVENTORY_VALUE_GE_HIGH_CUTOFF"],
            desc,
        )

    # 11. LOW_VELOCITY_HIGH_VALUE
    if inv_val is not None and inv_val >= high_inv_val_cutoff and is_low_demand:
        desc = (
            f"Observed elevated inventory valuation of {inv_val:.2f} coincides with low demand velocity "
            f"({avg_demand or 0.0:.2f} units/day)."
        )
        add_signal(
            "LOW_VELOCITY_HIGH_VALUE",
            SignalCategory.CROSS_DOMAIN,
            SignalSeverity.HIGH,
            {"inventory_value": inv_val, "average_daily_demand": avg_demand, "abc_class": record.abc_class},
            ["INVENTORY_VALUE_GE_HIGH_CUTOFF", "LOW_DEMAND_VELOCITY"],
            desc,
        )

    # 12. HIGH_VALUE_STOCKOUT_EXPOSURE
    if inv_val is not None and inv_val >= high_inv_val_cutoff and is_so:
        desc = (
            f"Observed stockout exposure ({so_days or 0} days) coincides with high inventory valuation "
            f"of {inv_val:.2f}."
        )
        add_signal(
            "HIGH_VALUE_STOCKOUT_EXPOSURE",
            SignalCategory.CROSS_DOMAIN,
            SignalSeverity.HIGH,
            {"inventory_value": inv_val, "stockout_days": so_days, "stockout_rate": so_rate},
            ["INVENTORY_VALUE_GE_HIGH_CUTOFF", "STOCKOUT_CONDITION_PRESENT"],
            desc,
        )

    # 13. HIGH_VALUE_RETURN_EXPOSURE
    if has_sample and inv_val is not None and inv_val >= high_inv_val_cutoff and ret_rate is not None and ret_rate >= config.high_return_rate_threshold:
        desc = (
            f"Observed return rate of {ret_rate:.2%} co-occurs with high inventory valuation "
            f"of {inv_val:.2f}."
        )
        add_signal(
            "HIGH_VALUE_RETURN_EXPOSURE",
            SignalCategory.CROSS_DOMAIN,
            SignalSeverity.MEDIUM,
            {"inventory_value": inv_val, "return_rate": ret_rate},
            ["INVENTORY_VALUE_GE_HIGH_CUTOFF", "RETURN_RATE_GE_THRESHOLD", "SUFFICIENT_SAMPLE"],
            desc,
        )

    # 14. FORECAST SIGNALS
    if fc_avail and fc_mean is not None and avail_inv is not None:
        if fc_mean > avail_inv:
            desc = (
                f"Forecasted demand of {fc_mean:.1f} units exceeds current available inventory "
                f"of {avail_inv} units."
            )
            add_signal(
                "FORECASTED_DEMAND_ABOVE_AVAILABLE_STOCK",
                SignalCategory.FORECAST,
                SignalSeverity.HIGH,
                {"forecast_mean": fc_mean, "available_inventory": avail_inv},
                ["FORECAST_EXCEEDS_AVAILABLE_STOCK"],
                desc,
            )
        elif avail_inv > (fc_mean * 3.0) and fc_mean > 0:
            desc = (
                f"Current available inventory of {avail_inv} units exceeds 3x forecasted demand "
                f"of {fc_mean:.1f} units."
            )
            add_signal(
                "FORECASTED_DEMAND_BELOW_CURRENT_STOCK",
                SignalCategory.FORECAST,
                SignalSeverity.INFO,
                {"forecast_mean": fc_mean, "available_inventory": avail_inv},
                ["AVAILABLE_STOCK_EXCEEDS_3X_FORECAST"],
                desc,
            )
    elif not fc_avail:
        desc = "Demand forecast is unavailable for this entity."
        add_signal(
            "FORECAST_UNAVAILABLE",
            SignalCategory.FORECAST,
            SignalSeverity.INFO,
            {},
            ["FORECAST_UNAVAILABLE_IN_DATASET"],
            desc,
        )

    # 15. RETURN ANOMALY + HIGH VALUE
    if ret_anomaly is True and inv_val is not None and inv_val >= high_inv_val_cutoff:
        desc = (
            f"Return anomaly indicator coincides with high inventory valuation of {inv_val:.2f}."
        )
        add_signal(
            "RETURN_ANOMALY_HIGH_VALUE_SKU",
            SignalCategory.RETURNS,
            SignalSeverity.CRITICAL,
            {"return_anomaly_flag": ret_anomaly, "inventory_value": inv_val},
            ["RETURN_ANOMALY_TRUE", "INVENTORY_VALUE_GE_HIGH_CUTOFF"],
            desc,
        )

    # 16. REPLENISHMENT + FINANCIAL SIGNALS
    if reorder_trigger is True:
        if net_rev is not None and net_rev >= high_revenue_cutoff:
            desc = (
                f"Active replenishment trigger coincides with high net revenue of {net_rev:.2f}."
            )
            add_signal(
                "HIGH_REVENUE_REPLENISHMENT_TRIGGER",
                SignalCategory.REPLENISHMENT,
                SignalSeverity.HIGH,
                {"net_revenue": net_rev, "replenishment_trigger": True},
                ["REPLENISHMENT_TRIGGER_TRUE", "NET_REVENUE_GE_HIGH_CUTOFF"],
                desc,
            )
        if gross_mar is not None and gross_mar >= high_margin_cutoff:
            desc = (
                f"Active replenishment trigger coincides with high gross margin of {gross_mar:.2f}."
            )
            add_signal(
                "HIGH_MARGIN_REPLENISHMENT_TRIGGER",
                SignalCategory.REPLENISHMENT,
                SignalSeverity.HIGH,
                {"gross_margin": gross_mar, "replenishment_trigger": True},
                ["REPLENISHMENT_TRIGGER_TRUE", "GROSS_MARGIN_GE_HIGH_CUTOFF"],
                desc,
            )
        if gross_mar_pct is not None and gross_mar_pct <= config.low_margin_pct_threshold:
            desc = (
                f"Active replenishment trigger coincides with lower gross margin percentage "
                f"of {gross_mar_pct:.1%}."
            )
            add_signal(
                "LOW_MARGIN_REPLENISHMENT_TRIGGER",
                SignalCategory.REPLENISHMENT,
                SignalSeverity.LOW,
                {"gross_margin_pct": gross_mar_pct, "replenishment_trigger": True},
                ["REPLENISHMENT_TRIGGER_TRUE", "GROSS_MARGIN_PCT_LE_THRESHOLD"],
                desc,
            )
        if inv_val is not None and inv_val >= high_inv_val_cutoff:
            desc = (
                f"Active replenishment trigger coincides with high inventory valuation of {inv_val:.2f}."
            )
            add_signal(
                "HIGH_VALUE_REPLENISHMENT_TRIGGER",
                SignalCategory.REPLENISHMENT,
                SignalSeverity.MEDIUM,
                {"inventory_value": inv_val, "replenishment_trigger": True},
                ["REPLENISHMENT_TRIGGER_TRUE", "INVENTORY_VALUE_GE_HIGH_CUTOFF"],
                desc,
            )

    return signals


def generate_cross_domain_signals(
    records: List[CrossDomainBusinessRecord],
    config: Optional[CrossDomainIntelligenceConfig] = None,
) -> List[CrossDomainSignal]:
    """Generate all cross-domain descriptive signals across a portfolio of records."""
    cfg = config or CrossDomainIntelligenceConfig()
    if not records:
        return []

    # Calculate portfolio cutoffs based on distribution of valid observations
    valid_revs = [r.net_revenue for r in records if r.net_revenue is not None and r.net_revenue > 0]
    high_revenue_cutoff = float(np.percentile(valid_revs, cfg.high_revenue_percentile * 100)) if valid_revs else 1000.0

    valid_margins = [r.gross_margin for r in records if r.gross_margin is not None and r.gross_margin > 0]
    high_margin_cutoff = float(np.percentile(valid_margins, cfg.high_margin_percentile * 100)) if valid_margins else 500.0

    valid_inv_vals = [r.inventory_value for r in records if r.inventory_value is not None and r.inventory_value > 0]
    high_inv_val_cutoff = float(np.percentile(valid_inv_vals, cfg.high_inventory_percentile * 100)) if valid_inv_vals else 5000.0

    valid_demands = [r.average_daily_demand for r in records if r.average_daily_demand is not None and r.average_daily_demand > 0]
    high_demand_cutoff = float(np.percentile(valid_demands, 80.0)) if valid_demands else 5.0

    all_signals: List[CrossDomainSignal] = []
    for rec in records:
        rec_signals = evaluate_signals_for_record(
            record=rec,
            config=cfg,
            high_revenue_cutoff=high_revenue_cutoff,
            high_margin_cutoff=high_margin_cutoff,
            high_inv_val_cutoff=high_inv_val_cutoff,
            high_demand_cutoff=high_demand_cutoff,
        )
        all_signals.extend(rec_signals)

    return all_signals
