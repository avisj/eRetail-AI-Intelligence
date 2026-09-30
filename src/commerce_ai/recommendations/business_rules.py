"""Deterministic Recommendation Rules and Rule Definitions (Phase 6G).

Defines explicit, testable, and auditable recommendation rules.
Each rule establishes:
- rule_id and rule_version ("1.0")
- input signals and required evidence criteria
- deterministic condition checks
- target recommendation type and action text
- explicit limitations and audit guardrails
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendationConfig,
    RecommendationPriority,
    RecommendationType,
)


class RecommendationRule(BaseModel):
    """Explicit, testable specification for a deterministic business recommendation rule."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(..., description="Unique deterministic rule identifier")
    rule_version: str = Field(default="1.0", description="Rule semantic version")
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(..., description="Detailed description of rule logic and intent")
    recommendation_type: RecommendationType = Field(..., description="Emitted recommendation type")
    input_signals: List[str] = Field(default_factory=list, description="Supported trigger signal names")
    required_evidence: List[str] = Field(default_factory=list, description="Mandatory evidence metric keys")
    action_text: str = Field(..., description="Recommended human action text")
    action_category: str = Field(..., description="Action classification (REVIEW, INVESTIGATE, REPLENISH, REBALANCE)")
    limitations: str = Field(..., description="Explicit analytical limitations and scope bounds")

    def determine_priority(
        self,
        severity: str,
        exposure_value: Optional[float],
        config: BusinessRecommendationConfig,
        urgency: Optional[str] = None,
    ) -> RecommendationPriority:
        """Deterministically determine recommendation priority tier without ML or rankings."""
        sev = (severity or "MEDIUM").upper()
        urg = (urgency or "").upper()
        exp = exposure_value if exposure_value is not None else 0.0

        # Critical triggers
        if sev == "CRITICAL" or urg == "CRITICAL":
            return RecommendationPriority.CRITICAL
        if exp >= config.critical_impact_threshold and sev in ("HIGH", "CRITICAL"):
            return RecommendationPriority.CRITICAL

        # High triggers
        if sev == "HIGH" or urg == "HIGH":
            return RecommendationPriority.HIGH
        if exp >= config.high_impact_threshold:
            return RecommendationPriority.HIGH

        # Medium triggers
        if sev == "MEDIUM" or urg == "MEDIUM":
            return RecommendationPriority.MEDIUM
        if exp >= config.medium_impact_threshold:
            return RecommendationPriority.MEDIUM

        # Low & Info triggers
        if sev == "INFO":
            return RecommendationPriority.INFO
        return RecommendationPriority.LOW


# ---------------------------------------------------------------------------
# Canonical Catalog of Deterministic Recommendation Rules (Phase 6G)
# ---------------------------------------------------------------------------

RULE_REPLENISHMENT_REVIEW_001 = RecommendationRule(
    rule_id="RULE_REPLENISHMENT_REVIEW_001",
    rule_version="1.0",
    name="Replenishment Order Review",
    description=(
        "Triggered when an entity has an active replenishment requirement and positive recommended order "
        "quantity with sufficient inventory and demand evidence."
    ),
    recommendation_type=RecommendationType.REPLENISHMENT_REVIEW,
    input_signals=[
        "HIGH_REVENUE_REPLENISHMENT_TRIGGER",
        "HIGH_MARGIN_REPLENISHMENT_TRIGGER",
        "REPLENISHMENT_TRIGGER",
    ],
    required_evidence=["net_inventory_position", "reorder_point", "recommended_order_qty"],
    action_text="Review the existing replenishment recommendation.",
    action_category="REPLENISH",
    limitations=(
        "Does not place or draft purchase orders; requires review of supplier terms, pack size multiples, "
        "and warehouse dock availability."
    ),
)

RULE_PURCHASE_ORDER_REVIEW_001 = RecommendationRule(
    rule_id="RULE_PURCHASE_ORDER_REVIEW_001",
    rule_version="1.0",
    name="Draft Purchase Order Proposal Review",
    description=(
        "Triggered when Phase 4B-2 has consolidated replenishment lines into a valid draft Purchase Order Proposal."
    ),
    recommendation_type=RecommendationType.PURCHASE_ORDER_REVIEW,
    input_signals=["PURCHASE_ORDER_PROPOSAL_GENERATED"],
    required_evidence=["supplier_id", "warehouse_id", "total_quantity", "proposal_id"],
    action_text="Review the draft purchase order proposal.",
    action_category="REVIEW",
    limitations=(
        "Draft proposal requiring supplier confirmation, purchasing budget authorization, and procurement approval; "
        "does not submit or transmit orders to external ERP/WMS systems."
    ),
)

RULE_WAREHOUSE_REBALANCING_REVIEW_001 = RecommendationRule(
    rule_id="RULE_WAREHOUSE_REBALANCING_REVIEW_001",
    rule_version="1.0",
    name="Warehouse Stock Transfer Review",
    description=(
        "Triggered when Phase 4C provides a valid transfer recommendation between a surplus and deficit warehouse."
    ),
    recommendation_type=RecommendationType.WAREHOUSE_REBALANCING_REVIEW,
    input_signals=["WAREHOUSE_TRANSFER_RECOMMENDED"],
    required_evidence=["source_warehouse", "destination_warehouse", "transfer_quantity"],
    action_text="Review the existing warehouse transfer recommendation.",
    action_category="REBALANCE",
    limitations=(
        "Does not initiate physical freight transfer; assumes logistics route availability, carrier transit times, "
        "and transfer cost viability."
    ),
)

RULE_SLOW_MOVING_INVENTORY_001 = RecommendationRule(
    rule_id="RULE_SLOW_MOVING_INVENTORY_001",
    rule_version="1.0",
    name="Slow-Moving Inventory Exposure Review",
    description=(
        "Triggered when inventory holding volume or capital is elevated while daily demand velocity is low or stagnant."
    ),
    recommendation_type=RecommendationType.SLOW_MOVING_INVENTORY_REVIEW,
    input_signals=[
        "HIGH_INVENTORY_LOW_DEMAND",
        "SLOW_MOVING_INVENTORY_EXPOSURE",
    ],
    required_evidence=["inventory_position", "inventory_value", "demand"],
    action_text="Review inventory disposition or rebalancing options.",
    action_category="REVIEW",
    limitations=(
        "Observes low velocity relative to holding volume; does not account for planned future promotional campaigns "
        "or seasonal re-acceleration without forward forecast integration."
    ),
)

RULE_HIGH_VALUE_INVENTORY_001 = RecommendationRule(
    rule_id="RULE_HIGH_VALUE_INVENTORY_001",
    rule_version="1.0",
    name="High-Value Inventory Capital Review",
    description=(
        "Triggered when capital tied up in inventory exceeds threshold, requiring working capital visibility."
    ),
    recommendation_type=RecommendationType.HIGH_VALUE_INVENTORY_REVIEW,
    input_signals=[
        "HIGH_VALUE_INVENTORY",
        "HIGH_VALUE_INVENTORY_EXPOSURE",
    ],
    required_evidence=["inventory_value", "inventory_position"],
    action_text="Review the high-value inventory position.",
    action_category="REVIEW",
    limitations=(
        "Capital concentration observation; does not imply inventory obsolescence, defect, or immediate holding risk."
    ),
)

RULE_STOCKOUT_REVIEW_001 = RecommendationRule(
    rule_id="RULE_STOCKOUT_REVIEW_001",
    rule_version="1.0",
    name="Stockout Exposure & Recovery Review",
    description=(
        "Triggered when an entity experiences stockout or critical depletion with associated lost sales exposure."
    ),
    recommendation_type=RecommendationType.STOCKOUT_REVIEW,
    input_signals=[
        "HIGH_REVENUE_STOCKOUT_EXPOSURE",
        "HIGH_REVENUE_LOW_STOCK",
        "HIGH_MARGIN_LOW_STOCK",
        "STOCKOUT_DURATION_ANOMALY",
    ],
    required_evidence=["inventory_position", "stockout_days"],
    action_text="Review stock availability and replenishment/transfer options.",
    action_category="INVESTIGATE",
    limitations=(
        "Reflects lost sales exposure under historical run rate; actual unconstrained demand may vary if customer "
        "substitution occurred."
    ),
)

RULE_RETURN_REVIEW_001 = RecommendationRule(
    rule_id="RULE_RETURN_REVIEW_001",
    rule_version="1.0",
    name="Product Return Exposure Review",
    description=(
        "Triggered when high return rates coincide with significant revenue volume or degraded margin."
    ),
    recommendation_type=RecommendationType.RETURN_REVIEW,
    input_signals=[
        "HIGH_RETURN_HIGH_REVENUE",
        "HIGH_RETURN_LOW_MARGIN",
        "RETURN_ANOMALY_HIGH_VALUE_SKU",
    ],
    required_evidence=["return_rate", "associated_revenue"],
    action_text="Review return drivers and operational handling.",
    action_category="INVESTIGATE",
    limitations=(
        "Correlational return volume observation; root causes require physical inspection, customer feedback audit, "
        "or size/specification verification before taking product action."
    ),
)

RULE_RETURN_ROOT_CAUSE_001 = RecommendationRule(
    rule_id="RULE_RETURN_ROOT_CAUSE_001",
    rule_version="1.0",
    name="Return Reason Shift Investigation",
    description=(
        "Triggered when Phase 5B identifies a statistically significant shift in a specific return reason category."
    ),
    recommendation_type=RecommendationType.RETURN_ROOT_CAUSE_REVIEW,
    input_signals=[
        "RETURN_REASON_SHIFT_ANOMALY",
        "RETURN_REASON_SPIKE",
    ],
    required_evidence=["return_reason", "current_rate"],
    action_text="Review the identified return reason pattern.",
    action_category="INVESTIGATE",
    limitations=(
        "Statistical anomaly in customer-reported return reasons; 'root cause' denotes the observed dimension "
        "requiring investigation and does not prove manufacturer defect or merchant liability."
    ),
)

RULE_MARGIN_REVIEW_001 = RecommendationRule(
    rule_id="RULE_MARGIN_REVIEW_001",
    rule_version="1.0",
    name="Operating Margin & Cost Compression Review",
    description=(
        "Triggered when low margin coexists with high inventory holding, elevated return cost, or fee compression."
    ),
    recommendation_type=RecommendationType.MARGIN_REVIEW,
    input_signals=[
        "LOW_MARGIN_HIGH_INVENTORY",
        "HIGH_RETURN_LOW_MARGIN",
        "MARGIN_DILUTION_RISK",
    ],
    required_evidence=["associated_revenue", "associated_margin"],
    action_text="Review margin drivers and associated operating conditions.",
    action_category="REVIEW",
    limitations=(
        "Financial analysis based on currently known cost components; does not recommend price or discount changes "
        "without explicit pricing policy constraints."
    ),
)

RULE_FORECAST_REVIEW_001 = RecommendationRule(
    rule_id="RULE_FORECAST_REVIEW_001",
    rule_version="1.0",
    name="Forecast Demand vs Stock Coverage Review",
    description=(
        "Triggered when forecasted forward demand exceeds available stock or forecast coverage reveals gaps."
    ),
    recommendation_type=RecommendationType.FORECAST_REVIEW,
    input_signals=[
        "FORECAST_COVERAGE_EXPOSURE",
        "FORECASTED_DEMAND_ABOVE_AVAILABLE_STOCK",
    ],
    required_evidence=["forecast", "inventory_position"],
    action_text="Review forecast coverage against current inventory.",
    action_category="REVIEW",
    limitations=(
        "Subject to model forecasting variance, prediction intervals, and demand intermittency; does not convert "
        "forecast uncertainty into certainty."
    ),
)

RULE_DATA_QUALITY_REVIEW_001 = RecommendationRule(
    rule_id="RULE_DATA_QUALITY_REVIEW_001",
    rule_version="1.0",
    name="Operational Data Quality Remediation Review",
    description=(
        "Triggered when required master data or operational telemetry is missing, preventing downstream optimization."
    ),
    recommendation_type=RecommendationType.DATA_QUALITY_REVIEW,
    input_signals=[
        "MISSING_UNIT_COST",
        "MISSING_SUPPLIER",
        "MISSING_FORECAST",
        "MISSING_INVENTORY",
        "MISSING_RETURN_DATA",
        "DATA_QUALITY_ALERT",
    ],
    required_evidence=["missing_field", "entity_id"],
    action_text="Review and remediate missing upstream operational and master data.",
    action_category="INVESTIGATE",
    limitations=(
        "Identifies data gaps that degrade or suppress automated calculation; data entry must be corrected in the "
        "originating system of record."
    ),
)

RULE_INVENTORY_REVIEW_001 = RecommendationRule(
    rule_id="RULE_INVENTORY_REVIEW_001",
    rule_version="1.0",
    name="General Inventory Position Review",
    description=(
        "Triggered for unclassified inventory imbalances or positions requiring operational audit."
    ),
    recommendation_type=RecommendationType.INVENTORY_REVIEW,
    input_signals=[
        "INVENTORY_IMBALANCE",
        "STOCK_COVERAGE_ALERT",
    ],
    required_evidence=["inventory_position"],
    action_text="Review current inventory position and warehouse distribution.",
    action_category="REVIEW",
    limitations=(
        "General inventory check based on current snapshot without forward rebalancing optimization."
    ),
)

ALL_RULES: List[RecommendationRule] = [
    RULE_REPLENISHMENT_REVIEW_001,
    RULE_PURCHASE_ORDER_REVIEW_001,
    RULE_WAREHOUSE_REBALANCING_REVIEW_001,
    RULE_SLOW_MOVING_INVENTORY_001,
    RULE_HIGH_VALUE_INVENTORY_001,
    RULE_STOCKOUT_REVIEW_001,
    RULE_RETURN_REVIEW_001,
    RULE_RETURN_ROOT_CAUSE_001,
    RULE_MARGIN_REVIEW_001,
    RULE_FORECAST_REVIEW_001,
    RULE_DATA_QUALITY_REVIEW_001,
    RULE_INVENTORY_REVIEW_001,
]

RULES_BY_ID: Dict[str, RecommendationRule] = {r.rule_id: r for r in ALL_RULES}
