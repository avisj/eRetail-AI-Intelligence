"""Business Recommendation Engine and Orchestrator (Phase 6G).

Orchestrates cross-domain signals (Phase 6E), quantified business impacts (Phase 6F),
and deterministic domain recommendation engines (Phases 4B, 4C, 5B) into auditable,
verifiable, and human-approved business recommendations.

Guarantees:
- Strict Human-in-the-Loop governance: approval_required=True, execution_allowed=False.
- Deterministic SHA-256 recommendation ID generation preventing duplicate creation.
- Explicit evidence models and machine-readable traceability back to signals and source datasets.
- Operational contradiction detection without autonomous resolution (human review mandatory).
- Neutral, factual, non-causal language without portfolio rankings or subjective scoring.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.business_impact.schemas import (
    BusinessImpactRecord,
    BusinessImpactResult,
    CalculationStatus,
    ImpactCategory,
    ImpactConfidence,
    ImpactType,
)
from commerce_ai.cross_domain.schemas import (
    CrossDomainBusinessRecord,
    CrossDomainIntelligenceResult,
    CrossDomainSignal,
    SignalCategory,
    SignalSeverity,
)
from commerce_ai.recommendations.business_rules import (
    ALL_RULES,
    RULES_BY_ID,
    RULE_DATA_QUALITY_REVIEW_001,
    RULE_FORECAST_REVIEW_001,
    RULE_HIGH_VALUE_INVENTORY_001,
    RULE_INVENTORY_REVIEW_001,
    RULE_MARGIN_REVIEW_001,
    RULE_PURCHASE_ORDER_REVIEW_001,
    RULE_REPLENISHMENT_REVIEW_001,
    RULE_RETURN_REVIEW_001,
    RULE_RETURN_ROOT_CAUSE_001,
    RULE_SLOW_MOVING_INVENTORY_001,
    RULE_STOCKOUT_REVIEW_001,
    RULE_WAREHOUSE_REBALANCING_REVIEW_001,
    RecommendationRule,
)
from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendation,
    BusinessRecommendationConfig,
    BusinessRecommendationResult,
    ConfidenceProvenance,
    ConflictType,
    RecommendationConflict,
    RecommendationEvidence,
    RecommendationEvidenceType,
    RecommendationFinancialContext,
    RecommendationOperationalContext,
    RecommendationPriority,
    RecommendationStatus,
    RecommendationTraceability,
    RecommendationType,
)
from commerce_ai.recommendations.schemas import (
    PurchaseOrderProposal,
    ReplenishmentRecommendation,
    WarehouseTransferRecommendation,
)


def _parse_date(val: Union[str, date, None]) -> Optional[date]:
    """Parse date or ISO string to date object safely."""
    if val is None:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        try:
            return date.fromisoformat(val[:10])
        except Exception:
            return None
    return None


def generate_recommendation_id(
    entity_key: str,
    recommendation_type: Union[RecommendationType, str],
    as_of_date: date,
    rule_id: str,
) -> str:
    """Generate a deterministic, tamper-evident recommendation ID using SHA-256."""
    rec_type_str = recommendation_type.value if isinstance(recommendation_type, RecommendationType) else str(recommendation_type)
    content = f"{entity_key}|{rec_type_str}|{as_of_date.isoformat()}|{rule_id}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"REC-{digest}"


def generate_conflict_id(
    conflict_type: Union[ConflictType, str],
    entity_key: str,
    as_of_date: date,
) -> str:
    """Generate a deterministic conflict ID using SHA-256."""
    c_type_str = conflict_type.value if isinstance(conflict_type, ConflictType) else str(conflict_type)
    content = f"{c_type_str}|{entity_key}|{as_of_date.isoformat()}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"CNF-{digest}"


def calculate_valid_until(
    created_as_of: date,
    rec_type: RecommendationType,
    config: BusinessRecommendationConfig,
) -> date:
    """Calculate deterministic expiration date for a recommendation based on its type."""
    if rec_type == RecommendationType.REPLENISHMENT_REVIEW:
        days = config.validity_days_replenishment
    elif rec_type == RecommendationType.PURCHASE_ORDER_REVIEW:
        days = config.validity_days_po
    elif rec_type == RecommendationType.WAREHOUSE_REBALANCING_REVIEW:
        days = config.validity_days_rebalancing
    elif rec_type == RecommendationType.STOCKOUT_REVIEW:
        days = config.validity_days_stockout
    elif rec_type in (RecommendationType.RETURN_REVIEW, RecommendationType.RETURN_ROOT_CAUSE_REVIEW):
        days = config.validity_days_returns
    elif rec_type == RecommendationType.FORECAST_REVIEW:
        days = config.validity_days_forecast
    elif rec_type == RecommendationType.DATA_QUALITY_REVIEW:
        days = config.validity_days_data_quality
    else:
        days = config.validity_days_inventory
    return created_as_of + timedelta(days=days)


class RecommendationEvidenceBuilder:
    """Utility class to construct verifiable, structured recommendation evidence."""

    @staticmethod
    def build_signal_evidence(signal: CrossDomainSignal) -> List[RecommendationEvidence]:
        """Convert a cross-domain signal into structured evidence records."""
        sig_type = getattr(signal, "signal_type", getattr(signal, "signal_name", "UNKNOWN"))
        items: List[RecommendationEvidence] = [
            RecommendationEvidence(
                evidence_type=RecommendationEvidenceType.SIGNAL,
                source_id=signal.signal_id,
                metric="signal_type",
                value=sig_type,
                description=f"Observed condition: {signal.description}",
            )
        ]
        metrics = getattr(signal, "observed_metrics", None) or getattr(signal, "underlying_metrics", {}) or {}
        for k, v in metrics.items():
            if v is not None:
                items.append(
                    RecommendationEvidence(
                        evidence_type=RecommendationEvidenceType.DOMAIN_METRIC,
                        source_id=signal.signal_id,
                        metric=k,
                        value=v,
                        description=f"Underlying signal telemetry metric {k}",
                    )
                )
        return items

    @staticmethod
    def build_impact_evidence(impact: BusinessImpactRecord) -> List[RecommendationEvidence]:
        """Convert a business impact record into structured evidence records."""
        imp_type = impact.impact_type.value if hasattr(impact.impact_type, "value") else str(impact.impact_type)
        curr = getattr(impact, "exposure_currency", getattr(impact, "currency", "USD"))
        desc = getattr(impact, "description", getattr(impact, "reason", ""))
        items: List[RecommendationEvidence] = [
            RecommendationEvidence(
                evidence_type=RecommendationEvidenceType.IMPACT,
                source_id=impact.impact_id,
                metric="exposure_value",
                value=impact.exposure_value,
                currency=curr,
                description=f"Quantified financial exposure ({imp_type}): {desc}",
            )
        ]
        units = getattr(impact, "affected_units", getattr(impact, "exposure_units", None))
        if units is not None:
            items.append(
                RecommendationEvidence(
                    evidence_type=RecommendationEvidenceType.IMPACT,
                    source_id=impact.impact_id,
                    metric="exposure_units",
                    value=units,
                    unit="units",
                    description=f"Physical exposure volume: {units} units",
                )
            )
        return items


class RecommendationConflictDetector:
    """Identifies contradictory or competing recommendations and signals on the same entity."""

    @staticmethod
    def detect_conflicts(
        recommendations: List[BusinessRecommendation],
        signals: List[CrossDomainSignal],
        as_of_date: date,
    ) -> List[RecommendationConflict]:
        """Detect operational conflicts between coexisting recommendations and signals.

        Does NOT resolve conflicts autonomously. Emits conflict records requiring human review.
        """
        conflicts: List[RecommendationConflict] = []

        def _get_entity_key(item) -> str:
            sku = getattr(item, "sku_id", None)
            wh = getattr(item, "warehouse_id", None)
            if sku and wh:
                return f"{sku}:{wh}"
            if sku:
                return sku
            if wh:
                return f"ALL:{wh}"
            return "GLOBAL"

        recs_by_entity: Dict[str, List[BusinessRecommendation]] = {}
        for r in recommendations:
            recs_by_entity.setdefault(_get_entity_key(r), []).append(r)

        signals_by_entity: Dict[str, List[CrossDomainSignal]] = {}
        for s in signals:
            signals_by_entity.setdefault(_get_entity_key(s), []).append(s)

        all_keys = sorted(set(recs_by_entity.keys()).union(signals_by_entity.keys()))

        seen_conflict_keys: Set[str] = set()

        for entity_key in all_keys:
            entity_recs = recs_by_entity.get(entity_key, [])
            entity_sigs = signals_by_entity.get(entity_key, [])

            rec_types = {r.recommendation_type for r in entity_recs}
            sig_names = {getattr(s, "signal_type", getattr(s, "signal_name", "")) for s in entity_sigs}

            # 1. INVENTORY_DEMAND_CONFLICT / REPLENISHMENT_SURPLUS_CONFLICT
            # Coexistence of surplus/slow-moving inventory and replenishment reorder
            has_slow_or_surplus = (
                RecommendationType.SLOW_MOVING_INVENTORY_REVIEW in rec_types
                or "HIGH_INVENTORY_LOW_DEMAND" in sig_names
                or "SLOW_MOVING_INVENTORY_EXPOSURE" in sig_names
            )
            has_replenish = (
                RecommendationType.REPLENISHMENT_REVIEW in rec_types
                or "HIGH_REVENUE_REPLENISHMENT_TRIGGER" in sig_names
                or "HIGH_MARGIN_REPLENISHMENT_TRIGGER" in sig_names
                or "REPLENISHMENT_TRIGGER" in sig_names
            )

            if has_slow_or_surplus and has_replenish:
                c_key = f"{ConflictType.INVENTORY_DEMAND_CONFLICT.value}:{entity_key}"
                if c_key not in seen_conflict_keys:
                    seen_conflict_keys.add(c_key)
                    conflicting_rec_ids = [
                        r.recommendation_id
                        for r in entity_recs
                        if r.recommendation_type in (
                            RecommendationType.SLOW_MOVING_INVENTORY_REVIEW,
                            RecommendationType.REPLENISHMENT_REVIEW,
                        )
                    ]
                    conflicting_sig_ids = [
                        s.signal_id
                        for s in entity_sigs
                        if getattr(s, "signal_type", getattr(s, "signal_name", "")) in (
                            "HIGH_INVENTORY_LOW_DEMAND",
                            "SLOW_MOVING_INVENTORY_EXPOSURE",
                            "HIGH_REVENUE_REPLENISHMENT_TRIGGER",
                            "HIGH_MARGIN_REPLENISHMENT_TRIGGER",
                            "REPLENISHMENT_TRIGGER",
                        )
                    ]
                    conflicts.append(
                        RecommendationConflict(
                            conflict_id=generate_conflict_id(
                                ConflictType.INVENTORY_DEMAND_CONFLICT, entity_key, as_of_date
                            ),
                            conflict_type=ConflictType.INVENTORY_DEMAND_CONFLICT,
                            conflicting_recommendation_ids=conflicting_rec_ids,
                            conflicting_signal_ids=conflicting_sig_ids,
                            entity_key=entity_key,
                            explanation=(
                                "High inventory and replenishment signals coexist for the same SKU/warehouse. "
                                "Additional review is required before selecting an action."
                            ),
                            requires_human_review=True,
                            created_as_of=as_of_date,
                        )
                    )

            # 2. RETURN_QUALITY_CONFLICT
            # Coexistence of severe return anomaly and replenishment order
            has_return_anomaly = (
                RecommendationType.RETURN_ROOT_CAUSE_REVIEW in rec_types
                or RecommendationType.RETURN_REVIEW in rec_types
                or "RETURN_REASON_SHIFT_ANOMALY" in sig_names
                or "RETURN_ANOMALY_HIGH_VALUE_SKU" in sig_names
            )
            if has_return_anomaly and has_replenish:
                c_key = f"{ConflictType.RETURN_QUALITY_CONFLICT.value}:{entity_key}"
                if c_key not in seen_conflict_keys:
                    seen_conflict_keys.add(c_key)
                    conflicting_rec_ids = [
                        r.recommendation_id
                        for r in entity_recs
                        if r.recommendation_type in (
                            RecommendationType.RETURN_REVIEW,
                            RecommendationType.RETURN_ROOT_CAUSE_REVIEW,
                            RecommendationType.REPLENISHMENT_REVIEW,
                        )
                    ]
                    conflicting_sig_ids = [
                        s.signal_id
                        for s in entity_sigs
                        if getattr(s, "signal_type", getattr(s, "signal_name", "")) in (
                            "RETURN_REASON_SHIFT_ANOMALY",
                            "RETURN_ANOMALY_HIGH_VALUE_SKU",
                            "HIGH_RETURN_HIGH_REVENUE",
                            "HIGH_REVENUE_REPLENISHMENT_TRIGGER",
                            "REPLENISHMENT_TRIGGER",
                        )
                    ]
                    conflicts.append(
                        RecommendationConflict(
                            conflict_id=generate_conflict_id(
                                ConflictType.RETURN_QUALITY_CONFLICT, entity_key, as_of_date
                            ),
                            conflict_type=ConflictType.RETURN_QUALITY_CONFLICT,
                            conflicting_recommendation_ids=conflicting_rec_ids,
                            conflicting_signal_ids=conflicting_sig_ids,
                            entity_key=entity_key,
                            explanation=(
                                "Replenishment order recommended for SKU undergoing active return quality or "
                                "reason shift investigation. Human verification of product quality is required."
                            ),
                            requires_human_review=True,
                            created_as_of=as_of_date,
                        )
                    )

        return conflicts


class BusinessRecommendationEngine:
    """Deterministic Business Recommendation Engine (Phase 6G)."""

    def __init__(self, config: Optional[BusinessRecommendationConfig] = None) -> None:
        self.config = config or BusinessRecommendationConfig()
        self.evidence_builder = RecommendationEvidenceBuilder()
        self.conflict_detector = RecommendationConflictDetector()

    def generate_recommendations(
        self,
        signals: Optional[List[CrossDomainSignal]] = None,
        impact_result: Optional[BusinessImpactResult] = None,
        impacts: Optional[List[BusinessImpactRecord]] = None,
        replenishment_recommendations: Optional[List[ReplenishmentRecommendation]] = None,
        purchase_order_proposals: Optional[List[PurchaseOrderProposal]] = None,
        warehouse_transfers: Optional[List[WarehouseTransferRecommendation]] = None,
        cross_domain_records: Optional[List[CrossDomainBusinessRecord]] = None,
        products_df: Optional[pd.DataFrame] = None,
        inventory_df: Optional[pd.DataFrame] = None,
        sales_df: Optional[pd.DataFrame] = None,
        suppliers_df: Optional[pd.DataFrame] = None,
        returns_df: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[date, str]] = None,
    ) -> BusinessRecommendationResult:
        """Execute end-to-end deterministic recommendation synthesis and conflict detection."""
        # 1. Resolve effective as_of_date
        effective_date = _parse_date(as_of_date)
        if effective_date is None:
            dates = []
            for s in signals or []:
                d = _parse_date(s.as_of_date)
                if d:
                    dates.append(d)
            for imp in impacts or (impact_result.impact_records if impact_result else []):
                d = _parse_date(getattr(imp, "as_of_date", getattr(imp, "created_as_of", None)))
                if d:
                    dates.append(d)
            effective_date = max(dates) if dates else date.today()

        signals_list = list(signals) if signals else []
        impacts_list = list(impacts) if impacts else (impact_result.impact_records if impact_result else [])
        replenish_list = list(replenishment_recommendations) if replenishment_recommendations else []
        po_proposals_list = list(purchase_order_proposals) if purchase_order_proposals else []
        transfers_list = list(warehouse_transfers) if warehouse_transfers else []

        # Point-in-time safety filtering: exclude signals / impacts dated after effective_date
        signals_list = [
            s for s in signals_list
            if s.as_of_date is None or _parse_date(s.as_of_date) is None or _parse_date(s.as_of_date) <= effective_date
        ]
        impacts_list = [
            i for i in impacts_list
            if getattr(i, "as_of_date", None) is None
            or _parse_date(getattr(i, "as_of_date", None)) is None
            or _parse_date(getattr(i, "as_of_date", None)) <= effective_date
        ]

        # 2. Build index mappings for fast lookup
        # Index signals by entity: (sku_id, warehouse_id) and sku_id
        signals_by_sku_wh: Dict[Tuple[str, str], List[CrossDomainSignal]] = {}
        signals_by_sku: Dict[str, List[CrossDomainSignal]] = {}
        for s in signals_list:
            if s.sku_id and s.warehouse_id:
                signals_by_sku_wh.setdefault((s.sku_id, s.warehouse_id), []).append(s)
            if s.sku_id:
                signals_by_sku.setdefault(s.sku_id, []).append(s)

        # Index impacts by entity
        impacts_by_sku_wh: Dict[Tuple[str, str], List[BusinessImpactRecord]] = {}
        impacts_by_sku: Dict[str, List[BusinessImpactRecord]] = {}
        for imp in impacts_list:
            if imp.sku_id and imp.warehouse_id:
                impacts_by_sku_wh.setdefault((imp.sku_id, imp.warehouse_id), []).append(imp)
            if imp.sku_id:
                impacts_by_sku.setdefault(imp.sku_id, []).append(imp)

        # Index replenishment recommendations by (sku_id, warehouse_id)
        replenish_by_sku_wh: Dict[Tuple[str, str], ReplenishmentRecommendation] = {}
        for r in replenish_list:
            replenish_by_sku_wh[(r.sku_id, r.warehouse_id)] = r

        # Index rebalancing transfers by (sku_id, dest_wh)
        transfers_by_dest: Dict[Tuple[str, str], List[WarehouseTransferRecommendation]] = {}
        for t in transfers_list:
            dest_wh = getattr(t, "destination_warehouse_id", getattr(t, "destination_warehouse", None))
            if dest_wh:
                transfers_by_dest.setdefault((t.sku_id, dest_wh), []).append(t)

        recommendations: List[BusinessRecommendation] = []
        emitted_rec_ids: Set[str] = set()

        # -----------------------------------------------------------------------
        # ENGINE ORCHESTRATION 1: Replenishment Reviews (RULE_REPLENISHMENT_REVIEW_001)
        # -----------------------------------------------------------------------
        for rep in replenish_list:
            if rep.recommendation_required and rep.recommended_order_qty > 0:
                entity_key = f"{rep.sku_id}:{rep.warehouse_id}"
                rule = RULE_REPLENISHMENT_REVIEW_001
                rec_id = generate_recommendation_id(
                    entity_key, rule.recommendation_type, effective_date, rule.rule_id
                )
                if rec_id in emitted_rec_ids:
                    continue
                emitted_rec_ids.add(rec_id)

                # Supporting signals & impacts
                sup_sigs = signals_by_sku_wh.get((rep.sku_id, rep.warehouse_id), [])
                sup_imps = impacts_by_sku_wh.get((rep.sku_id, rep.warehouse_id), [])

                evidence_items: List[RecommendationEvidence] = [
                    RecommendationEvidence(
                        evidence_type=RecommendationEvidenceType.ENGINE_OUTPUT,
                        source_id=rep.effective_recommendation_id,
                        metric="recommended_order_qty",
                        value=rep.recommended_order_qty,
                        unit="units",
                        description=f"Replenishment solver recommended order quantity: {rep.recommended_order_qty} units",
                    ),
                    RecommendationEvidence(
                        evidence_type=RecommendationEvidenceType.DOMAIN_METRIC,
                        source_id=rep.effective_recommendation_id,
                        metric="net_inventory_position",
                        value=rep.net_inventory_position,
                        unit="units",
                        description=f"Net inventory position: {rep.net_inventory_position} units",
                    ),
                    RecommendationEvidence(
                        evidence_type=RecommendationEvidenceType.DOMAIN_METRIC,
                        source_id=rep.effective_recommendation_id,
                        metric="reorder_point",
                        value=rep.reorder_point,
                        unit="units",
                        description=f"Dynamic reorder point: {rep.reorder_point} units",
                    ),
                ]

                for s in sup_sigs:
                    evidence_items.extend(self.evidence_builder.build_signal_evidence(s))
                for imp in sup_imps:
                    evidence_items.extend(self.evidence_builder.build_impact_evidence(imp))

                # Exposure & financial context
                rep_val = rep.estimated_order_cost
                fin_ctx = RecommendationFinancialContext(
                    currency=rep.currency or self.config.default_currency,
                    proposed_purchase_value=rep_val,
                    exposure_value=rep_val,
                )
                op_ctx = RecommendationOperationalContext(
                    inventory_position=rep.net_inventory_position,
                    reorder_point=rep.reorder_point,
                    recommended_order_qty=rep.recommended_order_qty,
                    demand=rep.forecast_daily_demand,
                    forecast=rep.forecast_daily_demand * 14 if rep.forecast_daily_demand else None,
                )

                priority = rule.determine_priority(
                    rep.urgency,
                    rep_val,
                    self.config,
                    urgency=rep.urgency,
                )

                recommendations.append(
                    BusinessRecommendation(
                        recommendation_id=rec_id,
                        recommendation_type=rule.recommendation_type,
                        status=RecommendationStatus.DRAFT,
                        priority=priority,
                        severity=rep.urgency,
                        sku_id=rep.sku_id,
                        warehouse_id=rep.warehouse_id,
                        currency=rep.currency or self.config.default_currency,
                        title=f"Replenishment Review: {rep.sku_id} at {rep.warehouse_id}",
                        summary=(
                            f"Net inventory position ({rep.net_inventory_position} units) has breached the reorder "
                            f"point ({rep.reorder_point:.1f} units). Recommended order quantity is {rep.recommended_order_qty} units."
                        ),
                        reason=rep.rationale,
                        action=rule.action_text,
                        action_category=rule.action_category,
                        evidence=evidence_items,
                        supporting_signal_ids=[s.signal_id for s in sup_sigs],
                        supporting_impact_ids=[imp.impact_id for imp in sup_imps],
                        financial_context=fin_ctx,
                        operational_context=op_ctx,
                        recommended_quantity=rep.recommended_order_qty,
                        recommended_value=rep_val,
                        source_engine="replenishment_solver",
                        source_engine_version="1.0",
                        rule_id=rule.rule_id,
                        rule_version=rule.rule_version,
                        confidence=ConfidenceProvenance.DETERMINISTIC_DERIVED,
                        confidence_reason="Derived directly from deterministic replenishment solver logic (Phase 4B-1).",
                        approval_required=True,
                        execution_allowed=False,
                        created_as_of=effective_date,
                        valid_until=calculate_valid_until(effective_date, rule.recommendation_type, self.config),
                        data_quality_status="COMPLETE",
                        calculation_status="CALCULATED",
                        traceability=RecommendationTraceability(
                            recommendation_id=rec_id,
                            supporting_signal_ids=[s.signal_id for s in sup_sigs],
                            supporting_impact_ids=[imp.impact_id for imp in sup_imps],
                            domain_metrics=["net_inventory_position", "reorder_point", "recommended_order_qty"],
                            source_datasets=["inventory.csv", "sales.csv"],
                        ),
                        requires_human_review=True,
                    )
                )

        # -----------------------------------------------------------------------
        # ENGINE ORCHESTRATION 2: Purchase Order Proposals (RULE_PURCHASE_ORDER_REVIEW_001)
        # -----------------------------------------------------------------------
        for po in po_proposals_list:
            entity_key = f"{po.supplier_id}:{po.warehouse_id}"
            rule = RULE_PURCHASE_ORDER_REVIEW_001
            rec_id = generate_recommendation_id(
                entity_key, rule.recommendation_type, effective_date, rule.rule_id
            )
            if rec_id in emitted_rec_ids:
                continue
            emitted_rec_ids.add(rec_id)

            evidence_items = [
                RecommendationEvidence(
                    evidence_type=RecommendationEvidenceType.ENGINE_OUTPUT,
                    source_id=po.proposal_id,
                    metric="total_quantity",
                    value=po.total_quantity,
                    unit="units",
                    description=f"Consolidated proposed order quantity across {len(po.lines)} SKUs",
                ),
                RecommendationEvidence(
                    evidence_type=RecommendationEvidenceType.ENGINE_OUTPUT,
                    source_id=po.proposal_id,
                    metric="sku_count",
                    value=len(po.lines),
                    description="Distinct SKUs in proposal bundle",
                ),
            ]
            if po.total_value is not None:
                evidence_items.append(
                    RecommendationEvidence(
                        evidence_type=RecommendationEvidenceType.ENGINE_OUTPUT,
                        source_id=po.proposal_id,
                        metric="total_value",
                        value=po.total_value,
                        currency=po.currency or self.config.default_currency,
                        description="Consolidated proposal purchase valuation",
                    )
                )

            fin_ctx = RecommendationFinancialContext(
                currency=po.currency or self.config.default_currency,
                proposed_purchase_value=po.total_value,
                exposure_value=po.total_value,
            )
            op_ctx = RecommendationOperationalContext(
                recommended_order_qty=po.total_quantity,
                destination_warehouse=po.warehouse_id,
            )
            priority = rule.determine_priority(
                po.urgency,
                po.total_value,
                self.config,
                urgency=po.urgency,
            )

            recommendations.append(
                BusinessRecommendation(
                    recommendation_id=rec_id,
                    recommendation_type=rule.recommendation_type,
                    status=RecommendationStatus.DRAFT,
                    priority=priority,
                    severity=po.urgency,
                    warehouse_id=po.warehouse_id,
                    currency=po.currency or self.config.default_currency,
                    title=f"Purchase Order Review: Supplier {po.supplier_id} to Warehouse {po.warehouse_id}",
                    summary=(
                        f"Draft purchase order proposal {po.proposal_id} consolidated for supplier {po.supplier_id} "
                        f"covering {len(po.lines)} line items with total quantity of {po.total_quantity} units."
                    ),
                    reason=po.rationale,
                    action=rule.action_text,
                    action_category=rule.action_category,
                    evidence=evidence_items,
                    supporting_signal_ids=[],
                    supporting_impact_ids=[],
                    financial_context=fin_ctx,
                    operational_context=op_ctx,
                    recommended_quantity=po.total_quantity,
                    recommended_value=po.total_value,
                    source_engine="purchase_order_engine",
                    source_engine_version="1.0",
                    rule_id=rule.rule_id,
                    rule_version=rule.rule_version,
                    confidence=ConfidenceProvenance.DETERMINISTIC_DERIVED,
                    confidence_reason="Aggregated deterministically by Phase 4B-2 PO Proposal Service.",
                    approval_required=True,
                    execution_allowed=False,
                    created_as_of=effective_date,
                    valid_until=calculate_valid_until(effective_date, rule.recommendation_type, self.config),
                    data_quality_status=po.cost_data_status,
                    calculation_status="CALCULATED",
                    traceability=RecommendationTraceability(
                        recommendation_id=rec_id,
                        supporting_signal_ids=[],
                        supporting_impact_ids=[],
                        domain_metrics=["total_quantity", "total_value", "lines_count"],
                        source_datasets=["suppliers.csv", "products.csv", "purchases.csv"],
                    ),
                    requires_human_review=True,
                )
            )

        # -----------------------------------------------------------------------
        # ENGINE ORCHESTRATION 3: Warehouse Rebalancing (RULE_WAREHOUSE_REBALANCING_REVIEW_001)
        # -----------------------------------------------------------------------
        for tr in transfers_list:
            src_wh = getattr(tr, "source_warehouse_id", getattr(tr, "source_warehouse", ""))
            dest_wh = getattr(tr, "destination_warehouse_id", getattr(tr, "destination_warehouse", ""))
            tr_id = getattr(tr, "recommendation_id", getattr(tr, "transfer_id", f"TR_{tr.sku_id}_{src_wh}_{dest_wh}"))
            transfer_qty = int(tr.transfer_quantity)

            entity_key = f"{tr.sku_id}:{src_wh}:{dest_wh}"
            rule = RULE_WAREHOUSE_REBALANCING_REVIEW_001
            rec_id = generate_recommendation_id(
                entity_key, rule.recommendation_type, effective_date, rule.rule_id
            )
            if rec_id in emitted_rec_ids:
                continue
            emitted_rec_ids.add(rec_id)

            surplus = getattr(tr, "source_transferable_surplus", 0.0)
            req_qty = getattr(tr, "destination_required_qty", 0.0)

            evidence_items = [
                RecommendationEvidence(
                    evidence_type=RecommendationEvidenceType.ENGINE_OUTPUT,
                    source_id=tr_id,
                    metric="transfer_quantity",
                    value=transfer_qty,
                    unit="units",
                    description=f"Recommended stock transfer quantity: {transfer_qty} units",
                ),
                RecommendationEvidence(
                    evidence_type=RecommendationEvidenceType.DOMAIN_METRIC,
                    source_id=tr_id,
                    metric="source_transferable_surplus",
                    value=surplus,
                    unit="units",
                    description=f"Transferable surplus at source facility {src_wh}: {surplus:.1f} units",
                ),
                RecommendationEvidence(
                    evidence_type=RecommendationEvidenceType.DOMAIN_METRIC,
                    source_id=tr_id,
                    metric="destination_required_qty",
                    value=req_qty,
                    unit="units",
                    description=f"Unmet demand requirement at destination {dest_wh}: {req_qty:.1f} units",
                ),
            ]

            fin_ctx = RecommendationFinancialContext(
                currency=self.config.default_currency,
                exposure_value=tr.estimated_transfer_cost,
            )
            op_ctx = RecommendationOperationalContext(
                transfer_quantity=transfer_qty,
                source_warehouse=src_wh,
                destination_warehouse=dest_wh,
            )
            priority = rule.determine_priority(
                tr.priority,
                tr.estimated_transfer_cost,
                self.config,
                urgency=tr.priority,
            )

            recommendations.append(
                BusinessRecommendation(
                    recommendation_id=rec_id,
                    recommendation_type=rule.recommendation_type,
                    status=RecommendationStatus.DRAFT,
                    priority=priority,
                    severity=tr.priority,
                    sku_id=tr.sku_id,
                    warehouse_id=dest_wh,
                    currency=self.config.default_currency,
                    title=f"Warehouse Transfer Review: {tr.sku_id} ({src_wh} -> {dest_wh})",
                    summary=(
                        f"Transfer of {transfer_qty} units recommended from surplus warehouse {src_wh} "
                        f"to deficit warehouse {dest_wh} to satisfy local reorder threshold."
                    ),
                    reason=tr.rationale,
                    action=rule.action_text,
                    action_category=rule.action_category,
                    evidence=evidence_items,
                    supporting_signal_ids=[],
                    supporting_impact_ids=[],
                    financial_context=fin_ctx,
                    operational_context=op_ctx,
                    recommended_quantity=transfer_qty,
                    recommended_value=tr.estimated_transfer_cost,
                    source_engine="warehouse_rebalancer",
                    source_engine_version="1.0",
                    rule_id=rule.rule_id,
                    rule_version=rule.rule_version,
                    confidence=ConfidenceProvenance.DETERMINISTIC_DERIVED,
                    confidence_reason="Derived deterministically by Phase 4C Warehouse Rebalancing solver.",
                    approval_required=True,
                    execution_allowed=False,
                    created_as_of=effective_date,
                    valid_until=calculate_valid_until(effective_date, rule.recommendation_type, self.config),
                    data_quality_status="COMPLETE",
                    calculation_status="CALCULATED",
                    traceability=RecommendationTraceability(
                        recommendation_id=rec_id,
                        supporting_signal_ids=[],
                        supporting_impact_ids=[],
                        domain_metrics=["transfer_quantity", "source_surplus", "destination_required_qty"],
                        source_datasets=["inventory.csv", "warehouses.csv"],
                    ),
                    requires_human_review=True,
                )
            )

        # -----------------------------------------------------------------------
        # CROSS-DOMAIN & IMPACT SIGNALS ORCHESTRATION
        # -----------------------------------------------------------------------
        for sig in signals_list:
            sig_name = getattr(sig, "signal_type", getattr(sig, "signal_name", ""))
            sku_id = sig.sku_id
            wh_id = sig.warehouse_id
            entity_key = f"{sku_id or 'GLOBAL'}:{wh_id or 'ALL'}"

            # Identify matching rule
            rule: Optional[RecommendationRule] = None
            if sig_name in ("HIGH_INVENTORY_LOW_DEMAND", "SLOW_MOVING_INVENTORY_EXPOSURE"):
                rule = RULE_SLOW_MOVING_INVENTORY_001
            elif sig_name in ("HIGH_VALUE_INVENTORY", "HIGH_VALUE_INVENTORY_EXPOSURE"):
                rule = RULE_HIGH_VALUE_INVENTORY_001
            elif sig_name in ("HIGH_REVENUE_STOCKOUT_EXPOSURE", "HIGH_REVENUE_LOW_STOCK", "HIGH_MARGIN_LOW_STOCK", "STOCKOUT_DURATION_ANOMALY"):
                rule = RULE_STOCKOUT_REVIEW_001
            elif sig_name in ("HIGH_RETURN_HIGH_REVENUE", "HIGH_RETURN_LOW_MARGIN", "RETURN_ANOMALY_HIGH_VALUE_SKU"):
                rule = RULE_RETURN_REVIEW_001
            elif sig_name in ("RETURN_REASON_SHIFT_ANOMALY", "RETURN_REASON_SPIKE"):
                rule = RULE_RETURN_ROOT_CAUSE_001
            elif sig_name in ("LOW_MARGIN_HIGH_INVENTORY", "MARGIN_DILUTION_RISK"):
                rule = RULE_MARGIN_REVIEW_001
            elif sig_name in ("FORECAST_COVERAGE_EXPOSURE", "FORECASTED_DEMAND_ABOVE_AVAILABLE_STOCK"):
                rule = RULE_FORECAST_REVIEW_001
            elif sig_name.startswith("MISSING_") or sig_name == "DATA_QUALITY_ALERT":
                rule = RULE_DATA_QUALITY_REVIEW_001
            elif "INVENTORY" in sig_name:
                rule = RULE_INVENTORY_REVIEW_001

            if rule is None:
                continue

            rec_id = generate_recommendation_id(
                entity_key, rule.recommendation_type, effective_date, rule.rule_id
            )
            if rec_id in emitted_rec_ids:
                continue
            emitted_rec_ids.add(rec_id)

            # Match associated impacts
            matched_impacts: List[BusinessImpactRecord] = []
            if sku_id and wh_id:
                matched_impacts = impacts_by_sku_wh.get((sku_id, wh_id), [])
            elif sku_id:
                matched_impacts = impacts_by_sku.get(sku_id, [])

            for imp in impacts_list:
                if imp.signal_id == sig.signal_id and imp not in matched_impacts:
                    matched_impacts.append(imp)

            evidence_items = self.evidence_builder.build_signal_evidence(sig)
            for imp in matched_impacts:
                evidence_items.extend(self.evidence_builder.build_impact_evidence(imp))

            exposure_val: Optional[float] = None
            if matched_impacts:
                exposure_val = sum(
                    imp.exposure_value for imp in matched_impacts if imp.exposure_value is not None
                )

            metrics = getattr(sig, "observed_metrics", None) or getattr(sig, "underlying_metrics", {}) or {}
            inv_val = metrics.get("inventory_value") or metrics.get("on_hand_value")
            rev_val = metrics.get("gross_revenue") or metrics.get("net_revenue") or metrics.get("revenue")
            margin_val = metrics.get("gross_margin") or metrics.get("margin")

            fin_ctx = RecommendationFinancialContext(
                currency=self.config.default_currency,
                associated_revenue=float(rev_val) if rev_val is not None else None,
                associated_margin=float(margin_val) if margin_val is not None else None,
                inventory_value=float(inv_val) if inv_val is not None else None,
                exposure_value=exposure_val,
            )

            linked_replenish = replenish_by_sku_wh.get((sku_id, wh_id)) if (sku_id and wh_id) else None
            linked_transfers = transfers_by_dest.get((sku_id, wh_id), []) if (sku_id and wh_id) else []

            inv_pos = metrics.get("inventory_position") or metrics.get("on_hand") or metrics.get("current_stock")
            rop = metrics.get("reorder_point")
            stockout_days = metrics.get("stockout_days") or metrics.get("days_out_of_stock")
            ret_rate = metrics.get("return_rate")
            demand_rate = metrics.get("daily_demand") or metrics.get("demand_mean") or metrics.get("average_daily_demand")
            forecast_val = metrics.get("forecast") or metrics.get("forecast_demand")

            tr_qty = None
            src_wh_linked = None
            if linked_transfers:
                tr_qty = int(linked_transfers[0].transfer_quantity)
                src_wh_linked = getattr(linked_transfers[0], "source_warehouse_id", getattr(linked_transfers[0], "source_warehouse", None))

            op_ctx = RecommendationOperationalContext(
                inventory_position=int(inv_pos) if inv_pos is not None else None,
                reorder_point=float(rop) if rop is not None else None,
                recommended_order_qty=linked_replenish.recommended_order_qty if linked_replenish else None,
                stockout_days=int(stockout_days) if stockout_days is not None else None,
                return_rate=float(ret_rate) if ret_rate is not None else None,
                demand=float(demand_rate) if demand_rate is not None else None,
                forecast=float(forecast_val) if forecast_val is not None else None,
                transfer_quantity=tr_qty,
                source_warehouse=src_wh_linked,
                destination_warehouse=wh_id,
            )

            action_text = rule.action_text
            summary_text = sig.description
            if rule.recommendation_type == RecommendationType.STOCKOUT_REVIEW:
                if linked_replenish:
                    action_text = f"Review existing replenishment order ({linked_replenish.recommended_order_qty} units) to address stockout."
                elif linked_transfers:
                    action_text = f"Review proposed warehouse transfer ({tr_qty} units from {src_wh_linked}) to resolve stockout."
                else:
                    action_text = "Review stock availability and replenishment/transfer options."

            priority = rule.determine_priority(
                sig.severity.value if hasattr(sig.severity, "value") else str(sig.severity),
                exposure_val,
                self.config,
            )

            confidence = ConfidenceProvenance.DETERMINISTIC_DERIVED
            if sig_name.startswith("MISSING_") or "INSUFFICIENT" in sig_name:
                confidence = ConfidenceProvenance.INSUFFICIENT_DATA
            elif "RETURN_PREDICTION" in sig_name or "FORECAST" in sig_name:
                confidence = ConfidenceProvenance.MODEL_BASED
            elif "OBSERVED" in sig_name:
                confidence = ConfidenceProvenance.DIRECT_OBSERVED

            recommendations.append(
                BusinessRecommendation(
                    recommendation_id=rec_id,
                    recommendation_type=rule.recommendation_type,
                    status=RecommendationStatus.DRAFT,
                    priority=priority,
                    severity=sig.severity.value if hasattr(sig.severity, "value") else str(sig.severity),
                    sku_id=sku_id,
                    warehouse_id=wh_id,
                    channel_id=sig.channel_id,
                    currency=self.config.default_currency,
                    title=f"{rule.name}: {sig_name} ({entity_key})",
                    summary=summary_text,
                    reason=f"Triggered by cross-domain signal {sig_name} with severity {sig.severity.value if hasattr(sig.severity, 'value') else sig.severity}.",
                    action=action_text,
                    action_category=rule.action_category,
                    evidence=evidence_items,
                    supporting_signal_ids=[sig.signal_id],
                    supporting_impact_ids=[imp.impact_id for imp in matched_impacts],
                    financial_context=fin_ctx,
                    operational_context=op_ctx,
                    recommended_quantity=linked_replenish.recommended_order_qty if linked_replenish else None,
                    recommended_value=exposure_val,
                    source_engine="cross_domain_intelligence",
                    source_engine_version="1.0",
                    rule_id=rule.rule_id,
                    rule_version=rule.rule_version,
                    confidence=confidence,
                    confidence_reason=f"Synthesized from {sig_name} signal and quantified business exposure.",
                    approval_required=True,
                    execution_allowed=False,
                    created_as_of=effective_date,
                    valid_until=calculate_valid_until(effective_date, rule.recommendation_type, self.config),
                    data_quality_status="COMPLETE" if confidence != ConfidenceProvenance.INSUFFICIENT_DATA else "INSUFFICIENT",
                    calculation_status="CALCULATED",
                    traceability=RecommendationTraceability(
                        recommendation_id=rec_id,
                        supporting_signal_ids=[sig.signal_id],
                        supporting_impact_ids=[imp.impact_id for imp in matched_impacts],
                        domain_metrics=list(metrics.keys()),
                        source_datasets=["sales.csv", "inventory.csv", "returns.csv"],
                    ),
                    requires_human_review=True,
                )
            )

        # -----------------------------------------------------------------------
        # CONFLICT DETECTION
        # -----------------------------------------------------------------------
        conflicts: List[RecommendationConflict] = []
        if self.config.conflict_detection_enabled:
            conflicts = self.conflict_detector.detect_conflicts(
                recommendations, signals_list, effective_date
            )

        # -----------------------------------------------------------------------
        # AGGREGATE SUMMARY CALCULATIONS (NO INDIVIDUAL RANKINGS)
        # -----------------------------------------------------------------------
        rec_by_type: Dict[str, int] = {}
        rec_by_priority: Dict[str, int] = {}
        rec_by_status: Dict[str, int] = {}
        rec_by_conf: Dict[str, int] = {}
        linked_sig_count = 0
        linked_imp_count = 0
        insufficient_data_cnt = 0

        for r in recommendations:
            rec_by_type[r.recommendation_type.value] = rec_by_type.get(r.recommendation_type.value, 0) + 1
            rec_by_priority[r.priority.value] = rec_by_priority.get(r.priority.value, 0) + 1
            rec_by_status[r.status.value] = rec_by_status.get(r.status.value, 0) + 1
            rec_by_conf[r.confidence.value] = rec_by_conf.get(r.confidence.value, 0) + 1
            linked_sig_count += len(r.supporting_signal_ids)
            linked_imp_count += len(r.supporting_impact_ids)
            if r.confidence == ConfidenceProvenance.INSUFFICIENT_DATA or r.data_quality_status == "INSUFFICIENT":
                insufficient_data_cnt += 1

        return BusinessRecommendationResult(
            recommendations=recommendations,
            conflicts=conflicts,
            total_recommendations=len(recommendations),
            recommendations_by_type=rec_by_type,
            recommendations_by_priority=rec_by_priority,
            recommendations_by_status=rec_by_status,
            recommendations_by_confidence=rec_by_conf,
            conflict_count=len(conflicts),
            human_review_count=len(recommendations),
            insufficient_data_count=insufficient_data_cnt,
            linked_signal_count=linked_sig_count,
            linked_impact_count=linked_imp_count,
            as_of_date=effective_date,
            currency=self.config.default_currency,
        )
