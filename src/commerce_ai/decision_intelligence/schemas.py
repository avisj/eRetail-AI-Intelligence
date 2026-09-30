"""Data Contracts and Schema Definitions for Decision Intelligence (Phase 6H).

Defines standardized Pydantic models for:
- DecisionPackage: Full decision context package presented to business reviewers
- DecisionOption: Discrete, auditable action options without automated winner selection
- DecisionTradeOff: Explicit trade-off evaluations using non-causal, conditional language
- RequiredDecisionInformation: Information completeness and missing data tracking
- DecisionRiskFlag: Qualitative, informational operational risk categorization
- DecisionIntelligenceConfig: Configuration parameters for decision packaging
- DecisionIntelligenceResult: Portfolio summary and aggregate distributions
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator

from commerce_ai.business_impact.schemas import (
    BusinessImpactRecord,
    CalculationStatus,
    ImpactConfidence,
)
from commerce_ai.cross_domain.schemas import CrossDomainSignal
from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendation,
    ConfidenceProvenance,
    ConflictType,
    RecommendationConflict,
    RecommendationEvidence,
    RecommendationEvidenceType,
    RecommendationFinancialContext,
    RecommendationOperationalContext,
    RecommendationPriority,
    RecommendationStatus,
    RecommendationType,
)


class DecisionStatus(str, Enum):
    """Lifecycle workflow status of a decision package."""

    PENDING_REVIEW = "PENDING_REVIEW"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"
    EXPIRED = "EXPIRED"


class DecisionOptionType(str, Enum):
    """Controlled taxonomy of permissible decision option classifications."""

    REVIEW_ONLY = "REVIEW_ONLY"
    REPLENISHMENT_REVIEW = "REPLENISHMENT_REVIEW"
    PURCHASE_ORDER_REVIEW = "PURCHASE_ORDER_REVIEW"
    TRANSFER_REVIEW = "TRANSFER_REVIEW"
    INVENTORY_REVIEW = "INVENTORY_REVIEW"
    DEMAND_REVIEW = "DEMAND_REVIEW"
    RETURN_REVIEW = "RETURN_REVIEW"
    MARGIN_REVIEW = "MARGIN_REVIEW"
    DATA_REVIEW = "DATA_REVIEW"
    DEFER_DECISION = "DEFER_DECISION"
    NO_CHANGE = "NO_CHANGE"


class OptionReversibility(str, Enum):
    """Qualitative assessment of how easily an action can be undone if executed."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    REVERSIBLE = "REVERSIBLE"
    IRREVERSIBLE = "IRREVERSIBLE"


class DecisionTradeOffDimension(str, Enum):
    """Categorical trade-off evaluation dimensions."""

    INVENTORY_CAPITAL = "INVENTORY_CAPITAL"
    SERVICE_LEVEL = "SERVICE_LEVEL"
    HOLDING_COST = "HOLDING_COST"
    OPERATIONAL_WORKLOAD = "OPERATIONAL_WORKLOAD"
    MARGIN_PROTECTION = "MARGIN_PROTECTION"
    DATA_INTEGRITY = "DATA_INTEGRITY"
    LOGISTICS_CAPACITY = "LOGISTICS_CAPACITY"
    SUPPLIER_RELATIONSHIP = "SUPPLIER_RELATIONSHIP"
    RETURN_RATE = "RETURN_RATE"


class InformationAvailability(str, Enum):
    """Information availability status for decision-making."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    PARTIAL = "PARTIAL"


class RiskFlagType(str, Enum):
    """Controlled qualitative risk flag categories."""

    DATA_QUALITY_RISK = "DATA_QUALITY_RISK"
    FORECAST_UNCERTAINTY = "FORECAST_UNCERTAINTY"
    INVENTORY_RISK = "INVENTORY_RISK"
    RETURN_RISK = "RETURN_RISK"
    MARGIN_RISK = "MARGIN_RISK"
    SUPPLIER_RISK = "SUPPLIER_RISK"
    LOGISTICS_RISK = "LOGISTICS_RISK"
    CONFLICT_RISK = "CONFLICT_RISK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class RiskSeverity(str, Enum):
    """Informational severity tiers for risk flags."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DecisionOption(BaseModel):
    """Structured, auditable decision option presented to human reviewer."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    option_id: str = Field(..., description="Deterministic identifier for the option (OPT-xxx)")
    option_type: DecisionOptionType = Field(..., description="Controlled option classification")
    title: str = Field(..., description="Actionable concise title of option")
    description: str = Field(..., description="Factual explanation of what this option involves")
    supporting_evidence: List[RecommendationEvidence] = Field(default_factory=list, description="Evidence justifying this option")
    expected_effect: str = Field(..., description="Conditional description of expected effect ('may', 'could', 'depends on')")
    financial_context: Optional[RecommendationFinancialContext] = Field(default=None, description="Financial implications if applicable")
    operational_context: Optional[RecommendationOperationalContext] = Field(default=None, description="Operational parameters if applicable")
    risks: List[str] = Field(default_factory=list, description="Associated operational considerations")
    dependencies: List[str] = Field(default_factory=list, description="Required prerequisite steps or approvals")
    reversibility: OptionReversibility = Field(default=OptionReversibility.MEDIUM, description="Reversibility classification")
    requires_external_action: bool = Field(default=False, description="Whether action requires work in external ERP/WMS")
    recommended_by_engine: bool = Field(default=False, description="Neutral flag indicating if supported by engine; NO winner declared")

    @field_validator("option_id")
    @classmethod
    def validate_option_id(cls, v: str) -> str:
        """Validate option ID prefix."""
        v_stripped = v.strip()
        if not v_stripped.startswith("OPT-"):
            raise ValueError(f"option_id must start with 'OPT-', got '{v}'")
        return v_stripped


class DecisionTradeOff(BaseModel):
    """Auditable trade-off analysis between competing operational objectives."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    tradeoff_id: str = Field(..., description="Deterministic trade-off identifier (TRD-xxx)")
    decision_id: str = Field(..., description="Target decision package identifier")
    option_id: str = Field(..., description="Associated option identifier")
    dimension: Union[DecisionTradeOffDimension, str] = Field(..., description="Evaluation dimension")
    positive_effect: str = Field(..., description="Potential positive outcome using non-causal, conditional language ('may')")
    negative_effect: str = Field(..., description="Potential negative outcome using non-causal, conditional language ('may')")
    uncertainty: str = Field(..., description="Underlying factors determining actual outcome ('depends on')")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence source IDs")

    @field_validator("tradeoff_id")
    @classmethod
    def validate_tradeoff_id(cls, v: str) -> str:
        """Validate tradeoff ID prefix."""
        v_stripped = v.strip()
        if not v_stripped.startswith("TRD-"):
            raise ValueError(f"tradeoff_id must start with 'TRD-', got '{v}'")
        return v_stripped


class RequiredDecisionInformation(BaseModel):
    """Specification of required information and its current availability."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    information_id: str = Field(..., description="Deterministic identifier (INF-xxx)")
    decision_id: str = Field(..., description="Target decision package identifier")
    field_name: str = Field(..., description="Name of the required data attribute or metric")
    description: str = Field(..., description="Explanation of what this information represents")
    reason_required: str = Field(..., description="Why this information is necessary before deciding")
    availability: InformationAvailability = Field(..., description="Current availability in datasets")
    source: str = Field(..., description="Originating source system or table (e.g. suppliers.csv, ERP)")
    blocking: bool = Field(default=False, description="Whether absence blocks an actionable approval")

    @field_validator("information_id")
    @classmethod
    def validate_info_id(cls, v: str) -> str:
        """Validate info ID prefix."""
        v_stripped = v.strip()
        if not v_stripped.startswith("INF-"):
            raise ValueError(f"information_id must start with 'INF-', got '{v}'")
        return v_stripped


class DecisionRiskFlag(BaseModel):
    """Informational operational risk flag attached to a decision package."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    risk_id: str = Field(..., description="Deterministic risk flag identifier (RSK-xxx)")
    decision_id: str = Field(..., description="Target decision package identifier")
    risk_type: RiskFlagType = Field(..., description="Classification of the risk")
    severity: RiskSeverity = Field(..., description="Informational severity tier")
    description: str = Field(..., description="Factual description of the risk condition")
    mitigation_hint: Optional[str] = Field(default=None, description="Informational consideration for review")

    @field_validator("risk_id")
    @classmethod
    def validate_risk_id(cls, v: str) -> str:
        """Validate risk ID prefix."""
        v_stripped = v.strip()
        if not v_stripped.startswith("RSK-"):
            raise ValueError(f"risk_id must start with 'RSK-', got '{v}'")
        return v_stripped


class DecisionTraceability(BaseModel):
    """Machine-readable audit trail mapping a decision package back through recommendation to source data."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    decision_id: str = Field(..., description="Target decision package ID")
    recommendation_id: str = Field(..., description="Referenced recommendation ID")
    supporting_signal_ids: List[str] = Field(default_factory=list, description="Underlying signal IDs")
    supporting_impact_ids: List[str] = Field(default_factory=list, description="Underlying business impact IDs")
    domain_metrics: List[str] = Field(default_factory=list, description="Evaluated domain metrics")
    source_datasets: List[str] = Field(default_factory=list, description="Source data tables")


class DecisionEntityContext(BaseModel):
    """Physical and catalog entity identification for a decision."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    sku_id: Optional[str] = Field(default=None, description="Target SKU")
    warehouse_id: Optional[str] = Field(default=None, description="Target warehouse facility")
    channel_id: Optional[str] = Field(default=None, description="Target sales channel")
    category_id: Optional[str] = Field(default=None, description="Product category")
    brand: Optional[str] = Field(default=None, description="Product brand")
    entity_key: str = Field(..., description="Primary entity composite key (e.g. SKU:WH)")


class DecisionBusinessContext(BaseModel):
    """Operational governance and provenance context for a decision."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    priority: RecommendationPriority = Field(..., description="Priority tier from recommendation")
    severity: str = Field(default="MEDIUM", description="Underlying signal/impact severity")
    source_engine: str = Field(..., description="Originating domain engine")
    rule_id: str = Field(..., description="Deterministic business rule ID applied")
    rule_version: str = Field(default="1.0", description="Rule semantic version")
    confidence: ConfidenceProvenance = Field(..., description="Methodological provenance tier")
    confidence_reason: str = Field(..., description="Explanation of confidence tier")
    currency: str = Field(default="USD", description="Currency code")


class DecisionPackage(BaseModel):
    """Complete, self-contained decision support package for human review and action planning.

    Guarantees:
    - approval_required=True (strictly True, human review mandatory)
    - execution_allowed=False (strictly False, autonomous execution prohibited)
    - selected_option=None (strictly None, engine does not declare a winner)
    - Deterministic SHA-256 identifier starting with 'DEC-'
    - Complete traceability back to recommendation, signals, impacts, and datasets
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    decision_id: str = Field(..., description="Deterministic SHA-256 identifier (DEC-xxx)")
    recommendation_id: str = Field(..., description="Source recommendation identifier")
    recommendation_type: RecommendationType = Field(..., description="Classification of the recommendation")
    decision_status: DecisionStatus = Field(default=DecisionStatus.PENDING_REVIEW, description="Workflow lifecycle status")
    decision_title: str = Field(..., description="Human-readable decision title")
    decision_summary: str = Field(..., description="Deterministic summary of decision context and trade-offs")
    entity_context: DecisionEntityContext = Field(..., description="Target SKU/facility/channel identifiers")
    business_context: DecisionBusinessContext = Field(..., description="Governance, rule, and confidence metadata")
    financial_context: Optional[RecommendationFinancialContext] = Field(default=None, description="Associated financial metrics")
    operational_context: Optional[RecommendationOperationalContext] = Field(default=None, description="Physical and logistical context")
    evidence: List[RecommendationEvidence] = Field(default_factory=list, description="Verifiable supporting evidence items")
    decision_options: List[DecisionOption] = Field(default_factory=list, description="Permissible action options for human reviewer")
    trade_offs: List[DecisionTradeOff] = Field(default_factory=list, description="Objective trade-offs across operational dimensions")
    required_information: List[RequiredDecisionInformation] = Field(default_factory=list, description="Required information and availability")
    risk_flags: List[DecisionRiskFlag] = Field(default_factory=list, description="Informational operational risk flags")
    conflicts: List[RecommendationConflict] = Field(default_factory=list, description="Operational contradictions detected in recommendation")
    approval_required: bool = Field(default=True, frozen=True, description="Strictly True; human review and approval mandatory")
    execution_allowed: bool = Field(default=False, frozen=True, description="Strictly False; autonomous execution prohibited")
    selected_option: Optional[str] = Field(default=None, description="Selected option ID; engine NEVER selects a winner (must remain None)")
    requires_human_review: bool = Field(default=True, description="Whether review is required before taking external action")
    created_as_of: date = Field(..., description="Point-in-time creation date")
    valid_until: date = Field(..., description="Expiration date after which decision is invalid")
    rule_version: str = Field(default="1.0", description="Version of the decision packaging rules")
    traceability: Optional[DecisionTraceability] = Field(default=None, description="Full machine-readable audit trail")

    @field_validator("decision_id")
    @classmethod
    def validate_decision_id(cls, v: str) -> str:
        """Validate decision ID prefix."""
        v_stripped = v.strip()
        if not v_stripped.startswith("DEC-"):
            raise ValueError(f"decision_id must start with 'DEC-', got '{v}'")
        return v_stripped

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision package to standardized dictionary representation."""
        d = self.model_dump()
        d["decision_status"] = self.decision_status.value
        d["recommendation_type"] = self.recommendation_type.value
        d["created_as_of"] = self.created_as_of.isoformat()
        d["valid_until"] = self.valid_until.isoformat()
        return d


class DecisionIntelligenceConfig(BaseModel):
    """Configuration governing decision packaging, risk thresholds, and validation rules."""

    model_config = ConfigDict(extra="forbid")

    critical_risk_threshold: float = Field(default=10000.0, description="Financial threshold for critical risk flag")
    high_risk_threshold: float = Field(default=2500.0, description="Financial threshold for high risk flag")
    medium_risk_threshold: float = Field(default=500.0, description="Financial threshold for medium risk flag")
    default_currency: str = Field(default="USD", description="Default currency code")
    auto_expire_decisions: bool = Field(default=True, description="Automatically mark status EXPIRED if past valid_until")
    conflict_risk_severity: RiskSeverity = Field(default=RiskSeverity.HIGH, description="Default severity for operational conflicts")
    high_return_rate_threshold: float = Field(default=0.10, description="Threshold above which return risk is flagged")
    low_margin_threshold: float = Field(default=0.20, description="Threshold below which margin risk is flagged")


class DecisionIntelligenceResult(BaseModel):
    """Aggregate output of the Decision Intelligence Service."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    decision_packages: List[DecisionPackage] = Field(default_factory=list, description="Emitted decision support packages")
    total_decisions: int = Field(default=0, description="Total count of decision packages")
    decisions_by_type: Dict[str, int] = Field(default_factory=dict, description="Count of decisions by recommendation type")
    decisions_by_status: Dict[str, int] = Field(default_factory=dict, description="Count of decisions by workflow status")
    decisions_by_risk_severity: Dict[str, int] = Field(default_factory=dict, description="Count of decisions by highest risk severity")
    decisions_by_option_count: Dict[int, int] = Field(default_factory=dict, description="Distribution of available options per package")
    human_review_count: int = Field(default=0, description="Decisions requiring human review (100%)")
    conflict_count: int = Field(default=0, description="Decisions with coexisting operational conflicts")
    insufficient_information_count: int = Field(default=0, description="Decisions with unavailable required information")
    decisions_by_confidence: Dict[str, int] = Field(default_factory=dict, description="Count of decisions by confidence provenance")
    as_of_date: date = Field(..., description="Point-in-time date of execution")


def generate_decision_id(
    recommendation_id: str,
    decision_type: Union[RecommendationType, str],
    as_of_date: date,
    rule_version: str = "1.0",
) -> str:
    """Generate a deterministic, tamper-evident decision ID using SHA-256."""
    d_type = decision_type.value if isinstance(decision_type, RecommendationType) else str(decision_type)
    content = f"{recommendation_id}|{d_type}|{as_of_date.isoformat()}|{rule_version}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"DEC-{digest}"


def generate_option_id(decision_id: str, option_type: Union[DecisionOptionType, str]) -> str:
    """Generate a deterministic option ID using SHA-256."""
    o_type = option_type.value if isinstance(option_type, DecisionOptionType) else str(option_type)
    content = f"{decision_id}|{o_type}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"OPT-{digest}"


def generate_tradeoff_id(decision_id: str, option_id: str, dimension: Union[DecisionTradeOffDimension, str]) -> str:
    """Generate a deterministic trade-off ID using SHA-256."""
    dim = dimension.value if isinstance(dimension, DecisionTradeOffDimension) else str(dimension)
    content = f"{decision_id}|{option_id}|{dim}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"TRD-{digest}"


def generate_info_id(decision_id: str, field_name: str) -> str:
    """Generate a deterministic required information ID using SHA-256."""
    content = f"{decision_id}|{field_name}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"INF-{digest}"


def generate_risk_id(decision_id: str, risk_type: Union[RiskFlagType, str]) -> str:
    """Generate a deterministic risk flag ID using SHA-256."""
    r_type = risk_type.value if isinstance(risk_type, RiskFlagType) else str(risk_type)
    content = f"{decision_id}|{r_type}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"RSK-{digest}"
