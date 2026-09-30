"""Data Contracts and Schema Definitions for Recommendations and PO Proposals (Phase 4B).

Defines standardized Pydantic models and data classes for:
- Replenishment configuration and constraints
- Replenishment recommendation records
- Urgency classifications and constraint audit flags
- High-level portfolio replenishment results
- Purchase Order Line and Proposal draft contracts (Phase 4B-2)
- Purchase Order Proposal Result bundles with source traceability
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class UrgencyLevel(str, Enum):
    """Deterministic replenishment urgency tiers."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ReplenishmentConstraint(str, Enum):
    """Audit identifiers for constraints modifying replenishment quantities."""

    MOQ_APPLIED = "MOQ_APPLIED"
    PACK_SIZE_ROUNDED = "PACK_SIZE_ROUNDED"
    MAX_ORDER_QTY_CAPPED = "MAX_ORDER_QTY_CAPPED"
    MAX_INVENTORY_DAYS_CAPPED = "MAX_INVENTORY_DAYS_CAPPED"
    MAX_INVENTORY_VALUE_CAPPED = "MAX_INVENTORY_VALUE_CAPPED"


class ProposalStatus(str, Enum):
    """Lifecycle status of a Purchase Order Proposal.

    In Phase 4B-2, all generated proposals are strictly DRAFT.
    Subsequent lifecycle states (APPROVED, SUBMITTED, SENT, CANCELLED)
    belong to future human-in-the-loop and OMS integration workflows.
    """

    DRAFT = "DRAFT"
    # Reserved future workflow states (not emitted by Phase 4B-2 engine):
    # APPROVED = "APPROVED"
    # REJECTED = "REJECTED"
    # SUBMITTED = "SUBMITTED"
    # SENT = "SENT"


class CostDataStatus(str, Enum):
    """Completeness status of financial cost/valuation data for a proposal or portfolio."""

    COMPLETE_COST_DATA = "COMPLETE_COST_DATA"
    PARTIAL_COST_DATA = "PARTIAL_COST_DATA"
    NO_COST_DATA = "NO_COST_DATA"


class ReplenishmentConfig(BaseModel):
    """Configuration settings, thresholds, and safety guards for replenishment sizing."""

    model_config = ConfigDict(extra="forbid")

    min_demand_threshold: float = Field(
        default=0.01,
        ge=0.0,
        description="Minimum daily demand rate below which an item is considered dormant",
    )
    default_pack_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="Global fallback lot/pack size if not specified on SKU or supplier master",
    )
    maximum_order_quantity: Optional[int] = Field(
        default=None,
        gt=0,
        description="Hard maximum limit on units ordered in a single replenishment cycle",
    )
    maximum_inventory_days: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Cap on total post-replenishment net inventory days of forward demand",
    )
    maximum_inventory_value: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Cap on total post-replenishment inventory valuation (units * unit_cost)",
    )
    sku_pack_sizes: Dict[str, int] = Field(
        default_factory=dict,
        description="SKU-specific pack/lot size multiples overriding catalog defaults",
    )
    sku_moq_overrides: Dict[str, int] = Field(
        default_factory=dict,
        description="SKU-specific Minimum Order Quantity overrides",
    )
    use_eoq: bool = Field(
        default=False,
        description="Whether to use Economic Order Quantity as the order size when validly available",
    )
    allow_zero_stock_dormant_reorder: bool = Field(
        default=False,
        description="If True, permits dormant zero-stock items to generate reorders (strictly False by default)",
    )


class ReplenishmentRecommendation(BaseModel):
    """Standardized deterministic replenishment recommendation for a SKU at a warehouse."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    sku_id: str = Field(..., description="Stock Keeping Unit ID")
    warehouse_id: str = Field(..., description="Fulfillment Warehouse facility ID")
    supplier_id: Optional[str] = Field(default=None, description="Preferred supplier/vendor ID")
    recommendation_required: bool = Field(..., description="True if replenishment order must be placed")
    urgency: str = Field(..., description="Replenishment urgency: CRITICAL, HIGH, MEDIUM, LOW")
    risk_category: str = Field(..., description="Phase 4A inventory risk state")
    on_hand: int = Field(..., ge=0, description="Available pickable inventory on hand")
    reserved: int = Field(default=0, ge=0, description="Allocated or reserved units")
    on_order: int = Field(default=0, ge=0, description="Inbound replenishment units in transit or open POs")
    net_inventory_position: int = Field(..., description="Effective net position: on_hand + on_order - reserved")
    forecast_daily_demand: float = Field(..., ge=0.0, description="Expected daily demand rate")
    lead_time_days: Optional[float] = Field(default=None, ge=0.0, description="Replenishment lead time in days")
    reorder_point: float = Field(..., ge=0.0, description="Dynamic Reorder Point (ROP)")
    target_stock_level: Optional[float] = Field(default=None, description="Configured target stock level S")
    shortfall_qty: int = Field(..., ge=0, description="Diagnostic shortfall: max(0, ROP - NetPosition)")
    required_qty: int = Field(..., ge=0, description="Unconstrained deficit to target stock: max(0, S - NetPosition)")
    minimum_order_quantity: Optional[int] = Field(default=None, ge=1, description="Supplier Minimum Order Quantity (MOQ)")
    pack_size: Optional[int] = Field(default=None, ge=1, description="Pack or lot size order multiple")
    recommended_order_qty: int = Field(..., ge=0, description="Final recommended order quantity")
    unit_cost: Optional[float] = Field(default=None, ge=0.0, description="Landed unit cost from product master")
    estimated_order_cost: Optional[float] = Field(default=None, ge=0.0, description="Total order valuation (qty * unit_cost)")
    currency: Optional[str] = Field(default=None, description="Explicit currency code (e.g. USD, EUR, INR)")
    projected_runout_date: Optional[str] = Field(default=None, description="Forecasted stock depletion date")
    rationale: str = Field(..., description="Deterministic, explainable rationale for the recommendation")
    constraints_applied: List[str] = Field(default_factory=list, description="List of constraints modifying the quantity")
    status: str = Field(default="COMPLETED", description="Outcome status: COMPLETED, INSUFFICIENT_POLICY_INPUTS, DORMANT_SUPPRESSED")
    recommendation_id: Optional[str] = Field(default=None, description="Optional unique identifier for audit traceability")

    @property
    def effective_recommendation_id(self) -> str:
        """Stable deterministic identifier for recommendation."""
        return self.recommendation_id or f"REC_{self.sku_id}_{self.warehouse_id}"

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v: str) -> str:
        valid = {u.value for u in UrgencyLevel}
        v_upper = v.strip().upper()
        if v_upper not in valid:
            raise ValueError(f"Invalid urgency '{v}'. Must be one of {valid}")
        return v_upper

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to standardized dictionary representation."""
        d = self.model_dump()
        d["recommendation_id"] = self.effective_recommendation_id
        d["forecast_daily_demand"] = round(self.forecast_daily_demand, 4)
        d["lead_time_days"] = round(self.lead_time_days, 2) if self.lead_time_days is not None else None
        d["reorder_point"] = round(self.reorder_point, 2)
        d["target_stock_level"] = round(self.target_stock_level, 2) if self.target_stock_level is not None else None
        d["estimated_order_cost"] = round(self.estimated_order_cost, 2) if self.estimated_order_cost is not None else None
        d["unit_cost"] = round(self.unit_cost, 2) if self.unit_cost is not None else None
        return d


class ReplenishmentResult(BaseModel):
    """Portfolio-wide replenishment solver result bundle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    recommendations: List[ReplenishmentRecommendation] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)

    @property
    def actionable_recommendations(self) -> List[ReplenishmentRecommendation]:
        """Subset of recommendations where an order is required and quantity > 0."""
        return [r for r in self.recommendations if r.recommendation_required and r.recommended_order_qty > 0]

    @property
    def non_recommendations(self) -> List[ReplenishmentRecommendation]:
        """Subset of evaluated items where no replenishment order is required."""
        return [r for r in self.recommendations if not r.recommendation_required]

    @property
    def excluded_dormant(self) -> List[ReplenishmentRecommendation]:
        """Subset of items excluded or suppressed due to dormant / dead stock demand."""
        return [r for r in self.recommendations if r.status == "DORMANT_SUPPRESSED"]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all recommendations into a flat pandas DataFrame."""
        if not self.recommendations:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.recommendations])


class PurchaseOrderLineProposal(BaseModel):
    """Line item within a draft Purchase Order Proposal (Phase 4B-2)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    sku_id: str = Field(..., description="Stock Keeping Unit ID")
    warehouse_id: str = Field(..., description="Receiving warehouse ID")
    supplier_id: str = Field(..., description="Fulfilling vendor supplier ID")
    product_name: Optional[str] = Field(default=None, description="Descriptive product display name")
    quantity: int = Field(..., gt=0, description="Recommended replenishment order quantity")
    unit_cost: Optional[float] = Field(default=None, ge=0.0, description="Purchased unit cost if available")
    estimated_line_value: Optional[float] = Field(default=None, ge=0.0, description="Line valuation: quantity * unit_cost")
    currency: Optional[str] = Field(default=None, description="Explicit currency code (e.g. USD, EUR, INR)")
    lead_time_days: Optional[float] = Field(default=None, ge=0.0, description="Replenishment lead time in days")
    expected_delivery_date: Optional[str] = Field(default=None, description="Forecasted dock receipt date (ISO YYYY-MM-DD)")
    urgency: str = Field(..., description="Replenishment urgency tier")
    risk_category: str = Field(..., description="Phase 4A inventory risk state")
    source_recommendation_id: str = Field(..., description="Identifier of originating recommendation for audit traceability")
    rationale: str = Field(..., description="Constituent replenishment rationale")

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v: str) -> str:
        valid = {u.value for u in UrgencyLevel}
        v_upper = v.strip().upper()
        if v_upper not in valid:
            raise ValueError(f"Invalid urgency '{v}'. Must be one of {valid}")
        return v_upper

    def to_dict(self) -> Dict[str, Any]:
        """Convert line proposal to standardized dictionary representation."""
        d = self.model_dump()
        d["estimated_line_value"] = round(self.estimated_line_value, 2) if self.estimated_line_value is not None else None
        d["unit_cost"] = round(self.unit_cost, 2) if self.unit_cost is not None else None
        d["lead_time_days"] = round(self.lead_time_days, 2) if self.lead_time_days is not None else None
        return d


class PurchaseOrderProposal(BaseModel):
    """Consolidated draft Purchase Order Proposal grouping lines by (supplier_id, warehouse_id, currency)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    proposal_id: str = Field(..., description="Deterministic unique proposal identifier")
    supplier_id: str = Field(..., description="Vendor supplier ID")
    warehouse_id: str = Field(..., description="Destination warehouse facility ID")
    proposal_date: Optional[str] = Field(default=None, description="Draft creation date (ISO YYYY-MM-DD)")
    currency: Optional[str] = Field(default=None, description="Explicit currency code; None if unavailable")
    lines: List[PurchaseOrderLineProposal] = Field(default_factory=list, description="Constituent line items")
    total_quantity: int = Field(..., ge=0, description="Total aggregated units across all lines")
    total_value: Optional[float] = Field(default=None, ge=0.0, description="Total monetary valuation if cost data available")
    cost_data_status: str = Field(..., description="COMPLETE_COST_DATA, PARTIAL_COST_DATA, or NO_COST_DATA")
    expected_delivery_date: Optional[str] = Field(default=None, description="Overall expected dock delivery date")
    urgency: str = Field(..., description="Highest urgency among constituent lines (CRITICAL > HIGH > MEDIUM > LOW)")
    status: str = Field(default="DRAFT", description="Workflow state; strictly DRAFT in Phase 4B-2")
    approval_required: bool = Field(default=True, description="Strictly True; human approval mandatory")
    source_recommendation_ids: List[str] = Field(default_factory=list, description="Traceability audit list of source recommendation IDs")
    rationale: str = Field(..., description="Deterministic summary explaining the proposal grouping")
    constraints_applied: List[str] = Field(default_factory=list, description="Distinct constraints applied across lines")

    def to_dict(self) -> Dict[str, Any]:
        """Convert proposal to standardized dictionary representation."""
        d = self.model_dump()
        d["total_value"] = round(self.total_value, 2) if self.total_value is not None else None
        d["lines"] = [line.to_dict() for line in self.lines]
        return d


class PurchaseOrderProposalResult(BaseModel):
    """Portfolio-wide purchase order proposals result bundle (Phase 4B-2)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    proposals: List[PurchaseOrderProposal] = Field(default_factory=list)
    actionable_line_count: int = 0
    excluded_missing_supplier: List[ReplenishmentRecommendation] = Field(default_factory=list)
    excluded_zero_quantity: List[ReplenishmentRecommendation] = Field(default_factory=list)
    non_actionable_recommendations: List[ReplenishmentRecommendation] = Field(default_factory=list)
    total_proposed_quantity: int = 0
    total_proposed_value: Optional[float] = None
    proposal_count: int = 0
    cost_data_status: str = "NO_COST_DATA"
    summary: Dict[str, Any] = Field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Export high-level proposals as a flat pandas DataFrame."""
        if not self.proposals:
            return pd.DataFrame()
        rows = []
        for p in self.proposals:
            row = p.to_dict()
            row["lines_count"] = len(p.lines)
            row.pop("lines", None)
            rows.append(row)
        return pd.DataFrame(rows)

    def lines_to_dataframe(self) -> pd.DataFrame:
        """Export all line items across all proposals as a flat pandas DataFrame."""
        rows = []
        for p in self.proposals:
            for line in p.lines:
                line_dict = line.to_dict()
                line_dict["proposal_id"] = p.proposal_id
                rows.append(line_dict)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)


class TransferPriority(str, Enum):
    """Deterministic transfer priority tiers."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TransferConstraint(str, Enum):
    """Audit identifiers for constraints modifying transfer quantities."""

    TRANSFER_PACK_SIZE_ROUNDED = "TRANSFER_PACK_SIZE_ROUNDED"
    MIN_TRANSFER_QTY_APPLIED = "MIN_TRANSFER_QTY_APPLIED"
    MAX_TRANSFER_QTY_CAPPED = "MAX_TRANSFER_QTY_CAPPED"


class WarehouseTransferRecommendation(BaseModel):
    """Standardized deterministic warehouse transfer recommendation (Phase 4C)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    recommendation_id: str = Field(..., description="Deterministic unique recommendation identifier")
    sku_id: str = Field(..., description="Stock Keeping Unit ID")
    source_warehouse_id: str = Field(..., description="Originating warehouse providing stock")
    destination_warehouse_id: str = Field(..., description="Receiving warehouse with inventory need")
    transfer_quantity: float = Field(..., gt=0.0, description="Recommended units to transfer")
    source_net_inventory_position: float = Field(..., description="Source warehouse net position before transfer")
    destination_net_inventory_position: float = Field(..., description="Destination warehouse net position before transfer")
    source_reorder_point: float = Field(..., ge=0.0, description="Source warehouse reorder point (protection level)")
    destination_reorder_point: float = Field(..., ge=0.0, description="Destination warehouse reorder point")
    source_transferable_surplus: float = Field(..., ge=0.0, description="Available surplus at source above protection level")
    destination_required_qty: float = Field(..., ge=0.0, description="Unconstrained deficit to destination target stock")
    destination_risk_category: str = Field(..., description="Phase 4A inventory risk state of destination")
    priority: str = Field(..., description="Transfer priority tier: CRITICAL, HIGH, MEDIUM, LOW")
    forecast_daily_demand: float = Field(..., ge=0.0, description="Destination forecast daily demand")
    estimated_transfer_cost: Optional[float] = Field(default=None, ge=0.0, description="Total estimated transfer cost")
    distance_km: Optional[float] = Field(default=None, ge=0.0, description="Geographic transfer distance in kilometers")
    transfer_lead_time_days: Optional[float] = Field(default=None, ge=0.0, description="Transit/transfer lead time in days")
    rationale: str = Field(..., description="Deterministic, explainable rationale for transfer")
    constraints_applied: List[str] = Field(default_factory=list, description="Transfer constraints applied")
    approval_required: bool = Field(default=True, description="Strictly True; human review mandatory")
    status: str = Field(default="DRAFT", description="Workflow state; strictly DRAFT")

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        valid = {p.value for p in TransferPriority}
        v_upper = v.strip().upper()
        if v_upper not in valid:
            raise ValueError(f"Invalid priority '{v}'. Must be one of {valid}")
        return v_upper

    @field_validator("approval_required")
    @classmethod
    def validate_approval(cls, v: bool) -> bool:
        if not v:
            raise ValueError("approval_required must be strictly True for warehouse transfer recommendations.")
        return True

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v.strip().upper() != ProposalStatus.DRAFT.value:
            raise ValueError(f"Initial status must be strictly '{ProposalStatus.DRAFT.value}'.")
        return ProposalStatus.DRAFT.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert transfer recommendation to standardized dictionary representation."""
        d = self.model_dump()
        d["transfer_quantity"] = round(self.transfer_quantity, 2)
        d["source_reorder_point"] = round(self.source_reorder_point, 2)
        d["destination_reorder_point"] = round(self.destination_reorder_point, 2)
        d["source_transferable_surplus"] = round(self.source_transferable_surplus, 2)
        d["destination_required_qty"] = round(self.destination_required_qty, 2)
        d["forecast_daily_demand"] = round(self.forecast_daily_demand, 4)
        if self.estimated_transfer_cost is not None:
            d["estimated_transfer_cost"] = round(self.estimated_transfer_cost, 2)
        if self.distance_km is not None:
            d["distance_km"] = round(self.distance_km, 2)
        if self.transfer_lead_time_days is not None:
            d["transfer_lead_time_days"] = round(self.transfer_lead_time_days, 2)
        return d


class WarehouseRebalancingConfig(BaseModel):
    """Configuration settings, routing parameters, and transfer constraints."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    min_demand_threshold: float = Field(
        default=0.01,
        ge=0.0,
        description="Daily demand threshold below which destination demand is considered dormant",
    )
    transfer_pack_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="Global fallback transfer lot/pack size",
    )
    sku_transfer_pack_sizes: Dict[str, int] = Field(
        default_factory=dict,
        description="SKU-specific transfer lot/pack size overrides",
    )
    minimum_transfer_quantity: Optional[int] = Field(
        default=None,
        gt=0,
        description="Global minimum transfer shipment size",
    )
    sku_min_transfer_quantities: Dict[str, int] = Field(
        default_factory=dict,
        description="SKU-specific minimum transfer quantity overrides",
    )
    maximum_transfer_quantity: Optional[int] = Field(
        default=None,
        gt=0,
        description="Global maximum transfer shipment size cap",
    )
    sku_max_transfer_quantities: Dict[str, int] = Field(
        default_factory=dict,
        description="SKU-specific maximum transfer quantity caps",
    )
    default_transfer_cost_per_unit: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Fallback transfer shipping cost per unit",
    )
    transfer_costs_per_unit: Dict[Any, float] = Field(
        default_factory=dict,
        description="Route-specific transfer cost per unit keyed by (src_wh, dest_wh) or 'src:dest'",
    )
    warehouse_distances_km: Dict[Any, float] = Field(
        default_factory=dict,
        description="Route distances in km keyed by (src_wh, dest_wh) or 'src:dest'",
    )
    warehouse_transfer_lead_times: Dict[Any, float] = Field(
        default_factory=dict,
        description="Route transfer transit days keyed by (src_wh, dest_wh) or 'src:dest'",
    )


class WarehouseRebalancingResult(BaseModel):
    """Portfolio-wide warehouse rebalancing result bundle (Phase 4C)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    recommendations: List[WarehouseTransferRecommendation] = Field(default_factory=list)
    excluded_dormant: List[Dict[str, Any]] = Field(default_factory=list)
    insufficient_policy_inputs: List[Dict[str, Any]] = Field(default_factory=list)
    unmet_destination_demand: List[Dict[str, Any]] = Field(default_factory=list)
    source_inventory_remaining: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    total_transfer_quantity: float = 0.0
    recommendation_count: int = 0
    estimated_total_transfer_cost: Optional[float] = None
    cost_data_status: str = CostDataStatus.NO_COST_DATA.value
    summary: Dict[str, Any] = Field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Export recommendations as a flat pandas DataFrame."""
        if not self.recommendations:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.recommendations])

