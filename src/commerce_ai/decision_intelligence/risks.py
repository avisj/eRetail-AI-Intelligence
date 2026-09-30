"""Deterministic Risk Flags and Required Information Evaluator for Decision Intelligence (Phase 6H).

Evaluates:
- Qualitative, informational operational risk flags (NO probability scores or subjective ratings)
- Required information completeness, marking unavailable attributes transparently (NO fabricated data)
- Conflict propagation from Phase 6G
- Multi-currency inconsistency detection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from commerce_ai.decision_intelligence.schemas import (
    DecisionIntelligenceConfig,
    DecisionRiskFlag,
    InformationAvailability,
    RequiredDecisionInformation,
    RiskFlagType,
    RiskSeverity,
    generate_info_id,
    generate_risk_id,
)
from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendation,
    ConfidenceProvenance,
    RecommendationConflict,
    RecommendationType,
)


class DecisionRiskEvaluator:
    """Evaluates qualitative risk flags and required information completeness."""

    def evaluate_risks(
        self,
        decision_id: str,
        rec: BusinessRecommendation,
        conflicts: List[RecommendationConflict],
        config: DecisionIntelligenceConfig,
    ) -> List[DecisionRiskFlag]:
        """Construct deterministic risk flags for a decision package."""
        flags: List[DecisionRiskFlag] = []
        op = rec.operational_context
        fin = rec.financial_context

        # 1. Operational Conflicts Risk
        rec_conflicts = [
            c for c in conflicts if rec.recommendation_id in c.conflicting_recommendation_ids
        ]
        if rec_conflicts:
            c_types = ", ".join(sorted(set(c.conflict_type.value for c in rec_conflicts)))
            flags.append(
                DecisionRiskFlag(
                    risk_id=generate_risk_id(decision_id, RiskFlagType.CONFLICT_RISK),
                    decision_id=decision_id,
                    risk_type=RiskFlagType.CONFLICT_RISK,
                    severity=config.conflict_risk_severity,
                    description=f"Contradictory operational indicators detected ({c_types}). Human review mandatory.",
                    mitigation_hint="Review competing signals and conflicting recommendations before taking action.",
                )
            )

        # 2. Insufficient Evidence Risk
        if rec.confidence == ConfidenceProvenance.INSUFFICIENT_DATA:
            flags.append(
                DecisionRiskFlag(
                    risk_id=generate_risk_id(decision_id, RiskFlagType.INSUFFICIENT_EVIDENCE),
                    decision_id=decision_id,
                    risk_type=RiskFlagType.INSUFFICIENT_EVIDENCE,
                    severity=RiskSeverity.HIGH,
                    description="Supporting evidence is incomplete or derived from missing data fallbacks.",
                    mitigation_hint="Confirm empirical values in system of record before committing to order or transfer.",
                )
            )

        # 3. Forecast Uncertainty Risk
        has_forecast = op and op.forecast is not None
        if not has_forecast or rec.confidence == ConfidenceProvenance.MODEL_BASED:
            sev = RiskSeverity.MEDIUM if not has_forecast else RiskSeverity.LOW
            desc = "Demand forecast is unavailable; decisions rely on historical run-rates." if not has_forecast else "Forward projections subject to model forecast variance."
            flags.append(
                DecisionRiskFlag(
                    risk_id=generate_risk_id(decision_id, RiskFlagType.FORECAST_UNCERTAINTY),
                    decision_id=decision_id,
                    risk_type=RiskFlagType.FORECAST_UNCERTAINTY,
                    severity=sev,
                    description=desc,
                    mitigation_hint="Cross-reference with recent promotional calendar or sales trends.",
                )
            )

        # 4. Inventory Risk (Excess capital or prolonged stockout)
        if fin and fin.inventory_value is not None and fin.inventory_value >= config.critical_risk_threshold:
            flags.append(
                DecisionRiskFlag(
                    risk_id=generate_risk_id(decision_id, RiskFlagType.INVENTORY_RISK),
                    decision_id=decision_id,
                    risk_type=RiskFlagType.INVENTORY_RISK,
                    severity=RiskSeverity.CRITICAL,
                    description=f"Critical capital concentration in inventory (${fin.inventory_value:,.2f}).",
                    mitigation_hint="Inspect holding duration and evaluate rebalancing or rationalization.",
                )
            )
        elif fin and fin.inventory_value is not None and fin.inventory_value >= config.high_risk_threshold:
            flags.append(
                DecisionRiskFlag(
                    risk_id=generate_risk_id(decision_id, RiskFlagType.INVENTORY_RISK),
                    decision_id=decision_id,
                    risk_type=RiskFlagType.INVENTORY_RISK,
                    severity=RiskSeverity.HIGH,
                    description=f"Elevated capital concentration in inventory (${fin.inventory_value:,.2f}).",
                    mitigation_hint="Review inventory coverage against forward velocity.",
                )
            )

        if op and op.stockout_days is not None and op.stockout_days > 0:
            flags.append(
                DecisionRiskFlag(
                    risk_id=generate_risk_id(decision_id, f"{RiskFlagType.INVENTORY_RISK.value}_STOCKOUT"),
                    decision_id=decision_id,
                    risk_type=RiskFlagType.INVENTORY_RISK,
                    severity=RiskSeverity.HIGH if op.stockout_days >= 7 else RiskSeverity.MEDIUM,
                    description=f"Active stockout duration of {op.stockout_days} days.",
                    mitigation_hint="Review unfulfilled backorders and customer substitution.",
                )
            )

        # 5. Return Risk
        if op and op.return_rate is not None and op.return_rate >= config.high_return_rate_threshold:
            flags.append(
                DecisionRiskFlag(
                    risk_id=generate_risk_id(decision_id, RiskFlagType.RETURN_RISK),
                    decision_id=decision_id,
                    risk_type=RiskFlagType.RETURN_RISK,
                    severity=RiskSeverity.HIGH if op.return_rate >= 0.20 else RiskSeverity.MEDIUM,
                    description=f"Elevated customer return rate ({op.return_rate:.1%}).",
                    mitigation_hint="Audit customer feedback and quality inspection records.",
                )
            )

        # 6. Margin Risk
        if fin and fin.associated_margin is not None and fin.associated_revenue:
            margin_pct = fin.associated_margin / fin.associated_revenue if fin.associated_revenue > 0 else 0.0
            if margin_pct <= config.low_margin_threshold:
                flags.append(
                    DecisionRiskFlag(
                        risk_id=generate_risk_id(decision_id, RiskFlagType.MARGIN_RISK),
                        decision_id=decision_id,
                        risk_type=RiskFlagType.MARGIN_RISK,
                        severity=RiskSeverity.HIGH if margin_pct <= 0.10 else RiskSeverity.MEDIUM,
                        description=f"Compressed operating margin ({margin_pct:.1%}).",
                        mitigation_hint="Verify wholesale acquisition cost and operational handling fee attribution.",
                    )
                )

        # 7. Currency Inconsistency Risk (Section 26)
        currencies_found = set()
        if fin and fin.currency:
            currencies_found.add(fin.currency)
        for e in rec.evidence:
            if e.currency:
                currencies_found.add(e.currency)
        if len(currencies_found) > 1:
            flags.append(
                DecisionRiskFlag(
                    risk_id=generate_risk_id(decision_id, f"{RiskFlagType.DATA_QUALITY_RISK.value}_CURRENCY"),
                    decision_id=decision_id,
                    risk_type=RiskFlagType.DATA_QUALITY_RISK,
                    severity=RiskSeverity.CRITICAL,
                    description=f"Multiple currencies detected in context ({', '.join(sorted(currencies_found))}).",
                    mitigation_hint="Isolate and reconcile transaction currencies before taking financial action.",
                )
            )

        # 8. Data Quality Risk
        if rec.data_quality_status != "COMPLETE":
            flags.append(
                DecisionRiskFlag(
                    risk_id=generate_risk_id(decision_id, RiskFlagType.DATA_QUALITY_RISK),
                    decision_id=decision_id,
                    risk_type=RiskFlagType.DATA_QUALITY_RISK,
                    severity=RiskSeverity.MEDIUM,
                    description=f"Underlying telemetry status flagged as {rec.data_quality_status}.",
                    mitigation_hint="Validate source fields in ERP or master catalog.",
                )
            )

        return flags

    def evaluate_required_information(
        self,
        decision_id: str,
        rec: BusinessRecommendation,
    ) -> List[RequiredDecisionInformation]:
        """Identify required information and determine its availability."""
        rec_type = rec.recommendation_type
        op = rec.operational_context
        fin = rec.financial_context
        info_items: List[RequiredDecisionInformation] = []

        if rec_type == RecommendationType.REPLENISHMENT_REVIEW:
            # 1. Current inventory
            inv_avail = InformationAvailability.AVAILABLE if (op and op.inventory_position is not None) else InformationAvailability.UNAVAILABLE
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "current_inventory"),
                    decision_id=decision_id,
                    field_name="current_inventory",
                    description="Physical on-hand and reserved inventory count at target warehouse.",
                    reason_required="Mandatory to confirm stock deficit before committing purchase funds.",
                    availability=inv_avail,
                    source="inventory.csv",
                    blocking=True,
                )
            )
            # 2. Supplier lead time
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "supplier_lead_time"),
                    decision_id=decision_id,
                    field_name="supplier_lead_time",
                    description="Expected calendar days between purchase order placement and receipt.",
                    reason_required="Necessary to assess stock coverage during replenishment replenishment cycle.",
                    availability=InformationAvailability.AVAILABLE,
                    source="suppliers.csv",
                    blocking=False,
                )
            )
            # 3. Unit cost
            cost_avail = InformationAvailability.AVAILABLE if (fin and fin.proposed_purchase_value is not None) else InformationAvailability.UNAVAILABLE
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "unit_cost"),
                    decision_id=decision_id,
                    field_name="unit_cost",
                    description="Contracted product acquisition cost per unit.",
                    reason_required="Required to calculate purchase order capital commitment.",
                    availability=cost_avail,
                    source="products.csv",
                    blocking=False,
                )
            )
            # 4. Forecast demand
            fc_avail = InformationAvailability.AVAILABLE if (op and op.forecast is not None) else InformationAvailability.UNAVAILABLE
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "forecast_demand"),
                    decision_id=decision_id,
                    field_name="forecast_demand",
                    description="Statistical forward demand projection across replenishment horizon.",
                    reason_required="Assesses forward demand coverage and protects against ordering into a lull.",
                    availability=fc_avail,
                    source="forecasting_service",
                    blocking=False,
                )
            )

        elif rec_type == RecommendationType.PURCHASE_ORDER_REVIEW:
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "supplier_order_minimum"),
                    decision_id=decision_id,
                    field_name="supplier_order_minimum",
                    description="Supplier minimum order quantity (MOQ) or minimum spend tier.",
                    reason_required="Ensures consolidated bundle satisfies supplier commercial terms.",
                    availability=InformationAvailability.AVAILABLE,
                    source="suppliers.csv",
                    blocking=False,
                )
            )
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "procurement_budget_authorization"),
                    decision_id=decision_id,
                    field_name="procurement_budget_authorization",
                    description="Approved purchasing budget allocation for the facility/category.",
                    reason_required="Mandatory enterprise authorization before PO dispatch.",
                    availability=InformationAvailability.UNAVAILABLE,
                    source="erp_budget",
                    blocking=True,
                )
            )

        elif rec_type == RecommendationType.WAREHOUSE_REBALANCING_REVIEW:
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "route_transit_capacity"),
                    decision_id=decision_id,
                    field_name="route_transit_capacity",
                    description="Carrier freight lane capacity and scheduled transit time.",
                    reason_required="Confirms physical route viability before booking freight.",
                    availability=InformationAvailability.AVAILABLE,
                    source="routes.csv",
                    blocking=False,
                )
            )
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "destination_bay_availability"),
                    decision_id=decision_id,
                    field_name="destination_bay_availability",
                    description="Receiving dock schedule at destination facility.",
                    reason_required="Prevents freight detention and receiving congestion.",
                    availability=InformationAvailability.UNAVAILABLE,
                    source="wms_dock_schedule",
                    blocking=False,
                )
            )

        elif rec_type in (RecommendationType.RETURN_REVIEW, RecommendationType.RETURN_ROOT_CAUSE_REVIEW):
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "return_processing_cost"),
                    decision_id=decision_id,
                    field_name="return_processing_cost",
                    description="Direct monetary labor and handling cost per return.",
                    reason_required="Determines true economic impact of return volume.",
                    availability=InformationAvailability.UNAVAILABLE,
                    source="returns.csv",
                    blocking=False,
                )
            )
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "physical_defect_inspection"),
                    decision_id=decision_id,
                    field_name="physical_defect_inspection",
                    description="Quality assurance report on returned physical merchandise.",
                    reason_required="Distinguishes manufacturing defect from customer buyer's remorse.",
                    availability=InformationAvailability.UNAVAILABLE,
                    source="warehouse_qc",
                    blocking=False,
                )
            )

        elif rec_type == RecommendationType.MARGIN_REVIEW:
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "contract_unit_cost"),
                    decision_id=decision_id,
                    field_name="contract_unit_cost",
                    description="Active supplier contract pricing terms.",
                    reason_required="Verifies whether margin compression is driven by wholesale cost inflation.",
                    availability=InformationAvailability.AVAILABLE,
                    source="products.csv",
                    blocking=False,
                )
            )
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "freight_cost_per_unit"),
                    decision_id=decision_id,
                    field_name="freight_cost_per_unit",
                    description="Inbound freight attribution per unit.",
                    reason_required="Completes unit economic margin breakdown.",
                    availability=InformationAvailability.UNAVAILABLE,
                    source="inbound_freight_log",
                    blocking=False,
                )
            )

        elif rec_type == RecommendationType.DATA_QUALITY_REVIEW:
            missing_field = "master_data_attribute"
            for e in rec.evidence:
                if "missing" in e.metric.lower():
                    missing_field = str(e.value)
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "source_system_record"),
                    decision_id=decision_id,
                    field_name=f"source_{missing_field}",
                    description=f"Originating master data record for {missing_field}.",
                    reason_required="Required to update catalog system of record.",
                    availability=InformationAvailability.UNAVAILABLE,
                    source="pim_master_catalog",
                    blocking=True,
                )
            )

        else:
            info_items.append(
                RequiredDecisionInformation(
                    information_id=generate_info_id(decision_id, "current_inventory"),
                    decision_id=decision_id,
                    field_name="current_inventory",
                    description="Current on-hand inventory position.",
                    reason_required="Validates baseline physical stock.",
                    availability=InformationAvailability.AVAILABLE if op and op.inventory_position is not None else InformationAvailability.UNAVAILABLE,
                    source="inventory.csv",
                    blocking=False,
                )
            )

        return info_items
