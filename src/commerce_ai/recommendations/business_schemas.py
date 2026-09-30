"""Data Contracts and Schema Definitions for Business Recommendations (Phase 6G).

Defines standardized Pydantic models for:
- Controlled recommendation taxonomy (12 deterministic types)
- Recommendation lifecycle status, priority, and confidence provenance
- Evidence model and granular traceability linking back to signals, impacts, and datasets
- Financial and operational recommendation contexts
- Recommendation conflict detection and human review guardrails
- Recommendation portfolio configuration and aggregate results
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RecommendationType(str, Enum):
    """Controlled deterministic business recommendation taxonomy (12 types)."""

    REPLENISHMENT_REVIEW = "REPLENISHMENT_REVIEW"
    PURCHASE_ORDER_REVIEW = "PURCHASE_ORDER_REVIEW"
    WAREHOUSE_REBALANCING_REVIEW = "WAREHOUSE_REBALANCING_REVIEW"
    INVENTORY_REVIEW = "INVENTORY_REVIEW"
    SLOW_MOVING_INVENTORY_REVIEW = "SLOW_MOVING_INVENTORY_REVIEW"
    HIGH_VALUE_INVENTORY_REVIEW = "HIGH_VALUE_INVENTORY_REVIEW"
    STOCKOUT_REVIEW = "STOCKOUT_REVIEW"
    RETURN_REVIEW = "RETURN_REVIEW"
    RETURN_ROOT_CAUSE_REVIEW = "RETURN_ROOT_CAUSE_REVIEW"
    MARGIN_REVIEW = "MARGIN_REVIEW"
    FORECAST_REVIEW = "FORECAST_REVIEW"
    DATA_QUALITY_REVIEW = "DATA_QUALITY_REVIEW"


class RecommendationStatus(str, Enum):
    """Lifecycle workflow status of a business recommendation."""

    DRAFT = "DRAFT"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class RecommendationPriority(str, Enum):
    """Deterministic, rule-based recommendation priority tiers."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class RecommendationEvidenceType(str, Enum):
    """Classification of supporting evidence attached to a recommendation."""

    SIGNAL = "SIGNAL"
    IMPACT = "IMPACT"
    DOMAIN_METRIC = "DOMAIN_METRIC"
    ENGINE_OUTPUT = "ENGINE_OUTPUT"
    DATA_QUALITY = "DATA_QUALITY"


class ConfidenceProvenance(str, Enum):
    """Methodological provenance and confidence basis for a recommendation."""

    DIRECT_OBSERVED = "DIRECT_OBSERVED"
    DETERMINISTIC_DERIVED = "DETERMINISTIC_DERIVED"
    MODEL_BASED = "MODEL_BASED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ConflictType(str, Enum):
    """Classification of operational contradictions between coexisting recommendations or signals."""

    INVENTORY_DEMAND_CONFLICT = "INVENTORY_DEMAND_CONFLICT"
    REPLENISHMENT_SURPLUS_CONFLICT = "REPLENISHMENT_SURPLUS_CONFLICT"
    TRANSFER_REPLENISHMENT_CONFLICT = "TRANSFER_REPLENISHMENT_CONFLICT"
    RETURN_QUALITY_CONFLICT = "RETURN_QUALITY_CONFLICT"
    GENERAL_CONFLICT = "GENERAL_CONFLICT"


class RecommendationEvidence(BaseModel):
    """Structured, verifiable evidence piece supporting a recommendation."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    evidence_type: RecommendationEvidenceType = Field(..., description="Classification of evidence")
    source_id: str = Field(..., description="Identifier of the origin entity (signal_id, impact_id, record_id, etc.)")
    metric: str = Field(..., description="Name of the underlying metric or indicator")
    value: Any = Field(..., description="Observed or calculated value")
    unit: Optional[str] = Field(default=None, description="Optional unit of measurement (units, days, pct, etc.)")
    currency: Optional[str] = Field(default=None, description="Monetary currency code if value is financial")
    description: str = Field(..., description="Deterministic explanation of why this evidence matters")


class RecommendationFinancialContext(BaseModel):
    """Associated financial metrics providing economic visibility for the recommendation."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    currency: str = Field(default="USD", description="ISO currency code")
    associated_revenue: Optional[float] = Field(default=None, ge=0.0, description="Gross or net revenue associated with entity")
    associated_margin: Optional[float] = Field(default=None, description="Gross margin associated with entity")
    inventory_value: Optional[float] = Field(default=None, ge=0.0, description="Inventory carrying valuation")
    known_variable_cost: Optional[float] = Field(default=None, ge=0.0, description="Known variable costs")
    known_contribution_margin: Optional[float] = Field(default=None, description="Known contribution margin")
    exposure_value: Optional[float] = Field(default=None, ge=0.0, description="Quantified financial risk or opportunity exposure")
    proposed_purchase_value: Optional[float] = Field(default=None, ge=0.0, description="Proposed purchase or order valuation")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to rounded dictionary representation."""
        return {
            "currency": self.currency,
            "associated_revenue": round(self.associated_revenue, 2) if self.associated_revenue is not None else None,
            "associated_margin": round(self.associated_margin, 2) if self.associated_margin is not None else None,
            "inventory_value": round(self.inventory_value, 2) if self.inventory_value is not None else None,
            "known_variable_cost": round(self.known_variable_cost, 2) if self.known_variable_cost is not None else None,
            "known_contribution_margin": round(self.known_contribution_margin, 2) if self.known_contribution_margin is not None else None,
            "exposure_value": round(self.exposure_value, 2) if self.exposure_value is not None else None,
            "proposed_purchase_value": round(self.proposed_purchase_value, 2) if self.proposed_purchase_value is not None else None,
        }


class RecommendationOperationalContext(BaseModel):
    """Operational parameters describing physical or logistical conditions."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    inventory_position: Optional[int] = Field(default=None, description="Available or net inventory units")
    reorder_point: Optional[float] = Field(default=None, ge=0.0, description="Dynamic reorder point")
    recommended_order_qty: Optional[int] = Field(default=None, ge=0, description="Recommended order units")
    stockout_days: Optional[int] = Field(default=None, ge=0, description="Observed or projected stockout duration")
    return_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Return rate fraction")
    demand: Optional[float] = Field(default=None, ge=0.0, description="Observed historical average daily demand")
    forecast: Optional[float] = Field(default=None, ge=0.0, description="Forward forecast demand units")
    transfer_quantity: Optional[int] = Field(default=None, ge=0, description="Proposed transfer units between facilities")
    source_warehouse: Optional[str] = Field(default=None, description="Origin warehouse for transfers")
    destination_warehouse: Optional[str] = Field(default=None, description="Destination warehouse for transfers")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to rounded dictionary representation."""
        return {
            "inventory_position": self.inventory_position,
            "reorder_point": round(self.reorder_point, 2) if self.reorder_point is not None else None,
            "recommended_order_qty": self.recommended_order_qty,
            "stockout_days": self.stockout_days,
            "return_rate": round(self.return_rate, 4) if self.return_rate is not None else None,
            "demand": round(self.demand, 4) if self.demand is not None else None,
            "forecast": round(self.forecast, 4) if self.forecast is not None else None,
            "transfer_quantity": self.transfer_quantity,
            "source_warehouse": self.source_warehouse,
            "destination_warehouse": self.destination_warehouse,
        }


class RecommendationTraceability(BaseModel):
    """Machine-readable audit trail mapping a recommendation to signals, impacts, and datasets."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    recommendation_id: str = Field(..., description="Target recommendation ID")
    supporting_signal_ids: List[str] = Field(default_factory=list, description="IDs of triggering cross-domain signals")
    supporting_impact_ids: List[str] = Field(default_factory=list, description="IDs of supporting business impact records")
    domain_metrics: List[str] = Field(default_factory=list, description="Domain metric names evaluated")
    source_datasets: List[str] = Field(default_factory=list, description="Underlying source tables (e.g. sales.csv, inventory.csv)")


class RecommendationConflict(BaseModel):
    """Audit record capturing conflicting coexisting recommendations or signals on the same entity."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    conflict_id: str = Field(..., description="Deterministic conflict identifier (CNF-xxx)")
    conflict_type: ConflictType = Field(..., description="Nature of the operational contradiction")
    conflicting_recommendation_ids: List[str] = Field(default_factory=list, description="IDs of recommendations in conflict")
    conflicting_signal_ids: List[str] = Field(default_factory=list, description="IDs of signals in conflict")
    entity_key: str = Field(..., description="Entity identifier (e.g. SKU, SKU:WH) where conflict occurred")
    explanation: str = Field(..., description="Deterministic explanation of why review is required")
    requires_human_review: bool = Field(default=True, frozen=True, description="Human review is strictly mandatory")
    created_as_of: date = Field(..., description="As-of date when conflict was detected")

    def to_dict(self) -> Dict[str, Any]:
        """Convert conflict record to standardized dictionary representation."""
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "conflicting_recommendation_ids": self.conflicting_recommendation_ids,
            "conflicting_signal_ids": self.conflicting_signal_ids,
            "entity_key": self.entity_key,
            "explanation": self.explanation,
            "requires_human_review": self.requires_human_review,
            "created_as_of": self.created_as_of.isoformat(),
        }


class BusinessRecommendation(BaseModel):
    """Auditable, deterministic business recommendation requiring human approval."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    recommendation_id: str = Field(..., description="Deterministic SHA-256 identifier (REC-xxx)")
    recommendation_type: RecommendationType = Field(..., description="Controlled recommendation classification")
    status: RecommendationStatus = Field(default=RecommendationStatus.DRAFT, description="Current workflow status")
    priority: RecommendationPriority = Field(..., description="Deterministic priority tier")
    severity: str = Field(default="MEDIUM", description="Underlying signal/impact severity (CRITICAL, HIGH, MEDIUM, LOW, INFO)")
    sku_id: Optional[str] = Field(default=None, description="Target SKU identifier")
    warehouse_id: Optional[str] = Field(default=None, description="Target warehouse identifier")
    channel_id: Optional[str] = Field(default=None, description="Target sales channel identifier")
    category_id: Optional[str] = Field(default=None, description="Target product category identifier")
    brand: Optional[str] = Field(default=None, description="Target product brand identifier")
    currency: str = Field(default="USD", description="Currency code")
    title: str = Field(..., description="Concise human-readable title")
    summary: str = Field(..., description="Structured summary of the recommendation")
    reason: str = Field(..., description="Deterministic rationale grounded in evidence")
    action: str = Field(..., description="Recommended human action to consider")
    action_category: str = Field(..., description="Category of action (REVIEW, INVESTIGATE, REPLENISH, REBALANCE)")
    evidence: List[RecommendationEvidence] = Field(default_factory=list, description="Verifiable supporting evidence items")
    supporting_signal_ids: List[str] = Field(default_factory=list, description="Referenced cross-domain signal IDs")
    supporting_impact_ids: List[str] = Field(default_factory=list, description="Referenced business impact IDs")
    financial_context: Optional[RecommendationFinancialContext] = Field(default=None, description="Financial figures")
    operational_context: Optional[RecommendationOperationalContext] = Field(default=None, description="Operational figures")
    recommended_quantity: Optional[int] = Field(default=None, ge=0, description="Recommended physical units if applicable")
    recommended_value: Optional[float] = Field(default=None, ge=0.0, description="Recommended monetary valuation if applicable")
    source_engine: str = Field(..., description="Upstream engine originating the recommendation (e.g. replenishment_solver)")
    source_engine_version: str = Field(default="1.0", description="Version of the originating engine")
    rule_id: str = Field(..., description="Deterministic business rule ID applied")
    rule_version: str = Field(default="1.0", description="Version of the business rule")
    confidence: ConfidenceProvenance = Field(..., description="Methodological provenance tier")
    confidence_reason: str = Field(..., description="Explanation for confidence classification")
    approval_required: bool = Field(default=True, frozen=True, description="Strictly True; human approval mandatory")
    execution_allowed: bool = Field(default=False, frozen=True, description="Strictly False; autonomous execution prohibited")
    created_as_of: date = Field(..., description="Point-in-time creation date")
    valid_until: date = Field(..., description="Expiration date after which recommendation is invalid")
    data_quality_status: str = Field(default="COMPLETE", description="COMPLETE, PARTIAL, or INSUFFICIENT")
    calculation_status: str = Field(default="CALCULATED", description="CALCULATED or ESTIMATED")
    traceability: Optional[RecommendationTraceability] = Field(default=None, description="Machine-readable traceability graph")
    requires_human_review: bool = Field(default=True, description="Whether review is required (always True for draft)")

    @field_validator("recommendation_id")
    @classmethod
    def validate_rec_id(cls, v: str) -> str:
        """Validate recommendation ID prefix."""
        v_stripped = v.strip()
        if not v_stripped.startswith("REC-"):
            raise ValueError(f"recommendation_id must start with 'REC-', got '{v}'")
        return v_stripped

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to standardized dictionary representation."""
        d = self.model_dump()
        d["recommendation_type"] = self.recommendation_type.value
        d["status"] = self.status.value
        d["priority"] = self.priority.value
        d["confidence"] = self.confidence.value
        d["created_as_of"] = self.created_as_of.isoformat()
        d["valid_until"] = self.valid_until.isoformat()
        if self.financial_context:
            d["financial_context"] = self.financial_context.to_dict()
        if self.operational_context:
            d["operational_context"] = self.operational_context.to_dict()
        d["evidence"] = [e.model_dump() for e in self.evidence]
        if self.traceability:
            d["traceability"] = self.traceability.model_dump()
        if self.recommended_value is not None:
            d["recommended_value"] = round(self.recommended_value, 2)
        return d


class BusinessRecommendationConfig(BaseModel):
    """Configuration parameters and thresholds for the business recommendation engine."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    critical_impact_threshold: float = Field(default=10000.0, ge=0.0, description="Minimum financial exposure for CRITICAL priority")
    high_impact_threshold: float = Field(default=2500.0, ge=0.0, description="Minimum financial exposure for HIGH priority")
    medium_impact_threshold: float = Field(default=500.0, ge=0.0, description="Minimum financial exposure for MEDIUM priority")
    min_sample_size: int = Field(default=10, ge=1, description="Minimum sample size required for statistical anomaly confidence")
    validity_days_replenishment: int = Field(default=7, ge=1, description="Validity horizon for replenishment reviews in days")
    validity_days_po: int = Field(default=7, ge=1, description="Validity horizon for purchase order reviews in days")
    validity_days_rebalancing: int = Field(default=7, ge=1, description="Validity horizon for warehouse transfer reviews in days")
    validity_days_inventory: int = Field(default=7, ge=1, description="Validity horizon for general inventory reviews in days")
    validity_days_stockout: int = Field(default=7, ge=1, description="Validity horizon for stockout reviews in days")
    validity_days_returns: int = Field(default=30, ge=1, description="Validity horizon for return reviews in days")
    validity_days_forecast: int = Field(default=14, ge=1, description="Validity horizon for forecast reviews in days")
    validity_days_data_quality: int = Field(default=3, ge=1, description="Validity horizon for data quality reviews in days")
    conflict_detection_enabled: bool = Field(default=True, description="Whether to check for and record recommendation conflicts")
    default_currency: str = Field(default="USD", description="Default currency code")


class BusinessRecommendationResult(BaseModel):
    """Portfolio-wide business recommendation bundle."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    recommendations: List[BusinessRecommendation] = Field(default_factory=list, description="Emitted business recommendations")
    conflicts: List[RecommendationConflict] = Field(default_factory=list, description="Detected recommendation conflicts")
    total_recommendations: int = 0
    recommendations_by_type: Dict[str, int] = Field(default_factory=dict)
    recommendations_by_priority: Dict[str, int] = Field(default_factory=dict)
    recommendations_by_status: Dict[str, int] = Field(default_factory=dict)
    recommendations_by_confidence: Dict[str, int] = Field(default_factory=dict)
    conflict_count: int = 0
    human_review_count: int = 0
    insufficient_data_count: int = 0
    linked_signal_count: int = 0
    linked_impact_count: int = 0
    as_of_date: date
    currency: str = "USD"
    execution_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dataframe(self) -> pd.DataFrame:
        """Export high-level recommendations as a flat pandas DataFrame."""
        if not self.recommendations:
            return pd.DataFrame()
        rows = []
        for r in self.recommendations:
            row = {
                "recommendation_id": r.recommendation_id,
                "recommendation_type": r.recommendation_type.value,
                "status": r.status.value,
                "priority": r.priority.value,
                "severity": r.severity,
                "sku_id": r.sku_id,
                "warehouse_id": r.warehouse_id,
                "channel_id": r.channel_id,
                "category_id": r.category_id,
                "brand": r.brand,
                "currency": r.currency,
                "title": r.title,
                "action": r.action,
                "action_category": r.action_category,
                "recommended_quantity": r.recommended_quantity,
                "recommended_value": r.recommended_value,
                "rule_id": r.rule_id,
                "rule_version": r.rule_version,
                "confidence": r.confidence.value,
                "approval_required": r.approval_required,
                "execution_allowed": r.execution_allowed,
                "created_as_of": r.created_as_of.isoformat(),
                "valid_until": r.valid_until.isoformat(),
                "requires_human_review": r.requires_human_review,
                "evidence_count": len(r.evidence),
                "supporting_signals_count": len(r.supporting_signal_ids),
                "supporting_impacts_count": len(r.supporting_impact_ids),
            }
            if r.financial_context:
                fc = r.financial_context.to_dict()
                row["exposure_value"] = fc.get("exposure_value")
                row["inventory_value"] = fc.get("inventory_value")
                row["associated_revenue"] = fc.get("associated_revenue")
                row["associated_margin"] = fc.get("associated_margin")
            if r.operational_context:
                oc = r.operational_context.to_dict()
                row["inventory_position"] = oc.get("inventory_position")
                row["reorder_point"] = oc.get("reorder_point")
                row["demand"] = oc.get("demand")
                row["return_rate"] = oc.get("return_rate")
            rows.append(row)
        return pd.DataFrame(rows)

    def conflicts_to_dataframe(self) -> pd.DataFrame:
        """Export conflict records as a flat pandas DataFrame."""
        if not self.conflicts:
            return pd.DataFrame()
        return pd.DataFrame([c.to_dict() for c in self.conflicts])
