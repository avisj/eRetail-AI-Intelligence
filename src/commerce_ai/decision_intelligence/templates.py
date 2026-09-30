"""Deterministic Explainability and Language Templates for Decision Intelligence (Phase 6H).

Provides auditable, rule-grounded textual summaries without using LLMs, generative AI,
or causal assertions.
All templates adhere to:
- Conditional, neutral phrasing ("may", "could", "observed", "reflects", "depends on")
- Absolute prohibition against causal claims ("caused", "proves", "will guarantee")
- Transparent citation of underlying evidence metrics and dimensions
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendation,
    RecommendationType,
)


def format_decision_title(rec: BusinessRecommendation) -> str:
    """Generate a concise, standardized decision package title."""
    entity = f"{rec.sku_id or 'GLOBAL'}"
    if rec.warehouse_id:
        entity += f" at {rec.warehouse_id}"
    elif rec.channel_id:
        entity += f" on {rec.channel_id}"

    type_titles = {
        RecommendationType.REPLENISHMENT_REVIEW: f"Replenishment Order Decision: {entity}",
        RecommendationType.PURCHASE_ORDER_REVIEW: f"Purchase Order Approval Decision: {entity}",
        RecommendationType.WAREHOUSE_REBALANCING_REVIEW: f"Inter-Warehouse Transfer Decision: {entity}",
        RecommendationType.INVENTORY_REVIEW: f"Inventory Position Review Decision: {entity}",
        RecommendationType.SLOW_MOVING_INVENTORY_REVIEW: f"Slow-Moving Inventory Disposition Decision: {entity}",
        RecommendationType.HIGH_VALUE_INVENTORY_REVIEW: f"High-Value Working Capital Decision: {entity}",
        RecommendationType.STOCKOUT_REVIEW: f"Stockout Mitigation & Recovery Decision: {entity}",
        RecommendationType.RETURN_REVIEW: f"Product Return Review Decision: {entity}",
        RecommendationType.RETURN_ROOT_CAUSE_REVIEW: f"Return Pattern Investigation Decision: {entity}",
        RecommendationType.MARGIN_REVIEW: f"Operating Margin Review Decision: {entity}",
        RecommendationType.FORECAST_REVIEW: f"Demand Forecast Coverage Decision: {entity}",
        RecommendationType.DATA_QUALITY_REVIEW: f"Data Remediation Action Decision: {entity}",
    }
    return type_titles.get(rec.recommendation_type, f"Operational Review Decision: {entity}")


def format_decision_summary(rec: BusinessRecommendation) -> str:
    """Generate a structured, non-causal decision summary grounding reviewer choices in evidence."""
    rec_type = rec.recommendation_type
    op = rec.operational_context
    fin = rec.financial_context

    if rec_type == RecommendationType.REPLENISHMENT_REVIEW:
        qty = op.recommended_order_qty if op and op.recommended_order_qty is not None else rec.recommended_quantity
        rop = op.reorder_point if op and op.reorder_point is not None else "ROP threshold"
        pos = op.inventory_position if op and op.inventory_position is not None else "current position"
        return (
            f"Replenishment review is presented for decision because inventory position ({pos}) is below the "
            f"calculated reorder point ({rop}). A solved replenishment order of {qty} units is proposed. "
            f"Reviewers should evaluate supplier lead time, purchase capital requirements, and dock capacity "
            f"before choosing between approving the replenishment review, deferring, or maintaining the current position."
        )

    if rec_type == RecommendationType.PURCHASE_ORDER_REVIEW:
        val_str = f"${fin.proposed_purchase_value:,.2f}" if fin and fin.proposed_purchase_value else "stated value"
        qty_str = f"{op.recommended_order_qty} units" if op and op.recommended_order_qty else "consolidated volume"
        return (
            f"Purchase order proposal review is presented for decision covering {qty_str} with estimated "
            f"valuation of {val_str}. Reviewers should examine supplier ordering terms, minimum spend criteria, "
            f"and budget authorization before approving the PO proposal review, deferring, or maintaining no change."
        )

    if rec_type == RecommendationType.WAREHOUSE_REBALANCING_REVIEW:
        src = op.source_warehouse if op and op.source_warehouse else "surplus facility"
        dest = op.destination_warehouse if op and op.destination_warehouse else "deficit facility"
        qty = op.transfer_quantity if op and op.transfer_quantity else rec.recommended_quantity
        return (
            f"Warehouse transfer review is presented for decision proposing movement of {qty} units from {src} to {dest}. "
            f"Reviewers should verify carrier transit availability, route freight cost, and destination receiving capacity "
            f"before choosing between proceeding with the transfer review, deferring, or keeping inventory unchanged."
        )

    if rec_type == RecommendationType.SLOW_MOVING_INVENTORY_REVIEW:
        val_str = f"${fin.inventory_value:,.2f}" if fin and fin.inventory_value else "elevated capital"
        return (
            f"Slow-moving inventory review is presented for decision because the entity exhibits elevated carrying "
            f"volume ({val_str}) coexisting with low recent demand velocity. Reviewers should examine catalog status, "
            f"carrying cost accumulation, and potential transfer options before selecting an inventory audit, transfer review, "
            f"deferral, or no-change disposition."
        )

    if rec_type == RecommendationType.HIGH_VALUE_INVENTORY_REVIEW:
        val_str = f"${fin.inventory_value:,.2f}" if fin and fin.inventory_value else "significant valuation"
        return (
            f"High-value inventory review is presented for decision due to concentrated working capital ({val_str}) "
            f"held in stock. Reviewers should inspect holding policies, insurance constraints, and demand run-rate "
            f"before deciding on an operational audit, deferral, or maintaining the current position."
        )

    if rec_type == RecommendationType.STOCKOUT_REVIEW:
        days = op.stockout_days if op and op.stockout_days is not None else "depleted"
        return (
            f"Stockout review is presented for decision following observed inventory depletion ({days} stockout days). "
            f"Reviewers should investigate whether historical sales velocity indicates unmet customer demand or substitution, "
            f"and review available replenishment or transfer candidates before selecting an action option."
        )

    if rec_type in (RecommendationType.RETURN_REVIEW, RecommendationType.RETURN_ROOT_CAUSE_REVIEW):
        ret_rate = f"{op.return_rate:.1%}" if op and op.return_rate is not None else "elevated rate"
        return (
            f"Return review is presented for decision following an observed return rate of {ret_rate}. "
            f"Reviewers should inspect customer feedback, quality assurance records, and warehouse handling protocols "
            f"before choosing between an operational return review, demand specification audit, data review, or deferral. "
            f"Pricing or refund policy changes are strictly excluded from automated scope."
        )

    if rec_type == RecommendationType.MARGIN_REVIEW:
        mar_str = f"${fin.associated_margin:,.2f}" if fin and fin.associated_margin else "observed margin"
        return (
            f"Operating margin review is presented for decision based on observed margin compression ({mar_str}) "
            f"coexisting with current operational holding conditions. Reviewers should evaluate cost component completeness, "
            f"supplier contract pricing, and logistics fees before selecting an action option. Automated price alterations "
            f"are strictly prohibited."
        )

    if rec_type == RecommendationType.FORECAST_REVIEW:
        return (
            f"Forecast review is presented for decision to evaluate forward demand projections against available "
            f"warehouse stock coverage. Reviewers should assess forecast uncertainty, historical variance, and prediction intervals "
            f"before adjusting replenishment or allocation schedules."
        )

    if rec_type == RecommendationType.DATA_QUALITY_REVIEW:
        return (
            f"Operational data quality remediation is presented for decision due to missing or incomplete operational telemetry. "
            f"Reviewers should audit originating source systems of record and update master data attributes before relying on "
            f"downstream algorithmic calculations."
        )

    return (
        f"Operational review is presented for decision based on {rec.recommendation_type.value}. Reviewers should audit "
        f"current evidence and context before selecting an action option."
    )
