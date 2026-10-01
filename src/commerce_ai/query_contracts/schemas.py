"""Core Pydantic Schemas for Business Query Contracts (Phase 7B).

Provides structured, serializable contracts defining:
- TimeRangeContract & ComparisonContract
- BusinessQueryFilter (with injection prevention)
- EvidenceRequirement & ConfidenceRequirement
- ContractGovernance (read-only assertions)
- BusinessQueryContract (primary query specification)
- ValidationIssue & ValidationResult
- ToolCallStep & QueryPlan (deterministic execution blueprint)
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    BusinessGrain,
    ComparisonType,
    ContractPriority,
    MetricIdentifier,
    OutputGrain,
    PlanStatus,
    ProvenanceLevel,
    QueryIntent,
    RequestedOutput,
    TimeGranularity,
    TimeRangePreset,
)

# Injection and malicious syntax detection patterns
_DISALLOWED_PATTERNS = [
    re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|UNION|GRANT|REVOKE)\b", re.IGNORECASE),
    re.compile(r"(--|/\*|\*/|;|\||`|\${)", re.IGNORECASE),
    re.compile(r"\b(import|eval|exec|compile|__import__|os\.|sys\.|subprocess\.)\b", re.IGNORECASE),
    re.compile(r"(\.\./|\.\.\\)", re.IGNORECASE),
]


def _check_disallowed_syntax(val: str, field_name: str) -> str:
    """Validate that string value does not contain code, SQL, or path injection patterns."""
    for pattern in _DISALLOWED_PATTERNS:
        if pattern.search(val):
            raise ValueError(
                f"Disallowed expression detected in '{field_name}': arbitrary SQL, code, or path traversal is rejected."
            )
    return val


def _normalize_iso_date(val: Optional[Union[str, date, datetime]]) -> Optional[str]:
    """Convert input date/datetime/string into standard ISO YYYY-MM-DD string."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date().isoformat()
    if isinstance(val, date):
        return val.isoformat()
    s = str(val).strip()
    if len(s) >= 10:
        return s[:10]
    return s


class TimeRangeContract(BaseModel):
    """Specification of temporal window and point-in-time constraints."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    preset: TimeRangePreset = Field(
        default=TimeRangePreset.CUSTOM,
        description="Standard relative temporal preset or CUSTOM for explicit dates.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Explicit start date (YYYY-MM-DD), inclusive.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Explicit end date (YYYY-MM-DD), inclusive. Cannot exceed as_of_date.",
    )
    as_of_date: Optional[str] = Field(
        default=None,
        description="Point-in-time reference cutoff. Historical data is bounded strictly by date <= as_of_date.",
    )

    @field_validator("start_date", "end_date", "as_of_date", mode="before")
    @classmethod
    def _coerce_date(cls, v: Any) -> Optional[str]:
        return _normalize_iso_date(v)

    @model_validator(mode="after")
    def _validate_chronology_and_presets(self) -> TimeRangeContract:
        # Enforce end_date <= as_of_date if both provided
        if self.end_date and self.as_of_date:
            if self.end_date > self.as_of_date:
                raise ValueError(
                    f"end_date ({self.end_date}) cannot exceed point-in-time as_of_date ({self.as_of_date})."
                )
        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                raise ValueError(
                    f"start_date ({self.start_date}) cannot be after end_date ({self.end_date})."
                )
        return self


class ComparisonContract(BaseModel):
    """Specification of comparative analysis against historical or custom baseline periods."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_type: ComparisonType = Field(
        default=ComparisonType.NONE,
        description="Type of comparison (e.g. PREVIOUS_PERIOD, PREVIOUS_YEAR_PERIOD, CUSTOM_COMPARISON).",
    )
    baseline_start_date: Optional[str] = Field(
        default=None,
        description="Explicit baseline period start date for CUSTOM_COMPARISON.",
    )
    baseline_end_date: Optional[str] = Field(
        default=None,
        description="Explicit baseline period end date for CUSTOM_COMPARISON.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of comparison intent.",
    )

    @field_validator("baseline_start_date", "baseline_end_date", mode="before")
    @classmethod
    def _coerce_date(cls, v: Any) -> Optional[str]:
        return _normalize_iso_date(v)


class BusinessQueryFilter(BaseModel):
    """Controlled multi-dimensional filters with strict injection protection."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    sku_id: Optional[str] = Field(default=None, description="Single SKU filter")
    sku_ids: Optional[List[str]] = Field(default=None, description="Collection of SKU filters")
    warehouse_id: Optional[str] = Field(default=None, description="Single warehouse filter")
    warehouse_ids: Optional[List[str]] = Field(default=None, description="Collection of warehouse filters")
    channel_id: Optional[str] = Field(default=None, description="Single channel filter")
    channel_ids: Optional[List[str]] = Field(default=None, description="Collection of channel filters")
    category_id: Optional[str] = Field(default=None, description="Single category filter")
    category_ids: Optional[List[str]] = Field(default=None, description="Collection of category filters")
    brand: Optional[str] = Field(default=None, description="Single brand filter")
    brands: Optional[List[str]] = Field(default=None, description="Collection of brand filters")
    supplier_id: Optional[str] = Field(default=None, description="Single supplier filter")
    supplier_ids: Optional[List[str]] = Field(default=None, description="Collection of supplier filters")
    velocity_tier: Optional[str] = Field(default=None, description="Velocity tier filter: FAST, MEDIUM, SLOW, NON_MOVING")
    abc_class: Optional[str] = Field(default=None, description="ABC revenue classification filter: A, B, C")
    xyz_class: Optional[str] = Field(default=None, description="XYZ variability classification filter: X, Y, Z")
    currency: Optional[str] = Field(default=None, description="ISO-4217 currency filter for strict currency isolation")
    limit: Optional[int] = Field(default=None, description="Pagination row limit")
    offset: Optional[int] = Field(default=None, description="Pagination row offset")

    @field_validator(
        "sku_id", "warehouse_id", "channel_id", "category_id", "brand",
        "supplier_id", "velocity_tier", "abc_class", "xyz_class", "currency",
        mode="before"
    )
    @classmethod
    def _validate_str_fields(cls, v: Any, info: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        return _check_disallowed_syntax(s, info.field_name)

    @field_validator(
        "sku_ids", "warehouse_ids", "channel_ids", "category_ids", "brands", "supplier_ids",
        mode="before"
    )
    @classmethod
    def _validate_list_fields(cls, v: Any, info: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        cleaned = []
        for item in v:
            s = str(item).strip()
            if s:
                _check_disallowed_syntax(s, info.field_name)
                cleaned.append(s)
        return sorted(list(set(cleaned))) if cleaned else None


class ConfidenceRequirement(BaseModel):
    """Specification of acceptable data provenance and confidence standards."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    required_provenance: Optional[ProvenanceLevel] = Field(
        default=None,
        description="Minimum acceptable data pedigree tier.",
    )
    allow_estimated: bool = Field(
        default=True,
        description="Whether heuristic or standard estimated figures are permitted.",
    )
    allow_model_based: bool = Field(
        default=True,
        description="Whether ML/statistical forecast or risk outputs are permitted.",
    )


class EvidenceRequirement(BaseModel):
    """Formal definition of evidence records and metadata required from Phase 7A."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    required_metrics: List[str] = Field(
        default_factory=list,
        description="List of specific ground truth metrics that must be accompanied by source evidence.",
    )
    required_sources: List[str] = Field(
        default_factory=list,
        description="List of expected source engines or module namespaces.",
    )
    require_source_ids: bool = Field(
        default=False,
        description="Whether upstream record identifiers (SIG-*, RULE-*, REC-*, DEC-*) must be present.",
    )
    as_of_date_required: bool = Field(
        default=True,
        description="Whether explicit point-in-time timestamp is required on evidence references.",
    )


class ContractGovernance(BaseModel):
    """Immutable safety assertions guaranteeing non-mutating query behavior."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    read_only: bool = Field(
        default=True,
        description="100% Read-Only assertion. Query contracts never mutate platform state.",
    )
    execution_allowed: bool = Field(
        default=False,
        description="Execution prohibition. Query contracts never execute orders, transfers, or price changes.",
    )
    approval_required: bool = Field(
        default=False,
        description="True for recommendation or decision review queries reflecting underlying object governance.",
    )
    action_execution: bool = Field(
        default=False,
        description="Explicit flag confirming this contract is informational, not an actionable execution.",
    )


class BusinessQueryContract(BaseModel):
    """Primary Business Query Contract (Phase 7B).

    Defines the structured semantic representation of a business question before tool selection:
    - What intent is being satisfied
    - What domain(s) are relevant
    - What metric(s) and dimension(s) are requested
    - What temporal, dimensional, and currency filters are applied
    - What output format and grain are required
    - What tools are needed from Phase 7A
    - Explicit governance and evidence requirements
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str = Field(
        description="Deterministic SHA-256 fingerprint representing the normalized contract.",
    )
    intent: QueryIntent = Field(
        description="Primary controlled business query intent.",
    )
    domain: BusinessDomain = Field(
        description="Primary functional platform domain matching Phase 7A.",
    )
    domains: List[BusinessDomain] = Field(
        default_factory=list,
        description="Multi-domain list for MULTI_DOMAIN_ANALYSIS contracts.",
    )
    metrics: List[MetricIdentifier] = Field(
        default_factory=list,
        description="Controlled business metrics requested by the question.",
    )
    dimensions: List[BusinessDimension] = Field(
        default_factory=list,
        description="Controlled analytical dimensions for grouping or filtering.",
    )
    filters: BusinessQueryFilter = Field(
        default_factory=BusinessQueryFilter,
        description="Validated multi-dimensional filters.",
    )
    time_range: TimeRangeContract = Field(
        default_factory=TimeRangeContract,
        description="Temporal boundaries and point-in-time constraints.",
    )
    comparison: ComparisonContract = Field(
        default_factory=ComparisonContract,
        description="Comparative analysis specifications.",
    )
    as_of_date: Optional[str] = Field(
        default=None,
        description="Effective point-in-time date for query execution.",
    )
    currency: Optional[str] = Field(
        default=None,
        description="Target ISO currency code for strict currency isolation.",
    )
    requested_grain: BusinessGrain = Field(
        default=BusinessGrain.PORTFOLIO,
        description="Requested business entity grouping grain.",
    )
    time_granularity: TimeGranularity = Field(
        default=TimeGranularity.NONE,
        description="Controlled temporal aggregation frequency (e.g. DAY, WEEK, MONTH, QUARTER, or NONE).",
    )
    requested_output: RequestedOutput = Field(
        default=RequestedOutput.KPI,
        description="Target envelope format for response.",
    )
    required_tools: List[str] = Field(
        default_factory=list,
        description="Deterministic list of Phase 7A tools required to fulfill this contract.",
    )
    priority: ContractPriority = Field(
        default=ContractPriority.NORMAL,
        description="Execution priority classification.",
    )
    explanation_context: Optional[str] = Field(
        default=None,
        description="Contextual diagnostic flag for WHY-style investigation questions.",
    )
    evidence_requirements: EvidenceRequirement = Field(
        default_factory=EvidenceRequirement,
        description="Requirements for underlying evidence lineage.",
    )
    provenance_requirements: ConfidenceRequirement = Field(
        default_factory=ConfidenceRequirement,
        description="Acceptable data pedigree standards.",
    )
    governance: ContractGovernance = Field(
        default_factory=ContractGovernance,
        description="Safety and read-only assertions.",
    )

    @field_validator("as_of_date", mode="before")
    @classmethod
    def _coerce_as_of_date(cls, v: Any) -> Optional[str]:
        return _normalize_iso_date(v)


class ValidationIssue(BaseModel):
    """Specification of a contract validation failure or warning."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(description="Deterministic error code identifier")
    message: str = Field(description="Human-readable diagnostic description")
    field: Optional[str] = Field(default=None, description="Contract field path associated with the issue")
    severity: str = Field(default="ERROR", description="ERROR or WARNING")


class ValidationResult(BaseModel):
    """Outcome of validating a BusinessQueryContract against business and governance rules."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool = Field(description="True if contract conforms to all rules with 0 errors")
    errors: List[ValidationIssue] = Field(default_factory=list, description="List of validation errors")
    warnings: List[ValidationIssue] = Field(default_factory=list, description="List of advisory warnings")


class ToolCallStep(BaseModel):
    """Single planned Phase 7A tool invocation within a QueryPlan."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str = Field(description="Canonical Phase 7A tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="QueryContext-compatible keyword arguments")
    purpose: str = Field(description="Business explanation of why this tool call is required")
    required: bool = Field(default=True, description="Whether failure of this tool fails the contract plan")
    sequence: int = Field(default=1, description="Execution sequence order (1-indexed)")


class QueryPlan(BaseModel):
    """Deterministic, unexecuted execution plan synthesizing a BusinessQueryContract into Phase 7A tool calls."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str = Field(description="Deterministic SHA-256 fingerprint of the execution plan")
    contract_id: str = Field(description="Associated BusinessQueryContract query_id")
    steps: List[ToolCallStep] = Field(default_factory=list, description="Ordered list of tool call steps")
    required_tools: List[str] = Field(default_factory=list, description="Set of unique required tool names")
    dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="Step dependency graph")
    execution_order: List[str] = Field(default_factory=list, description="Ordered tool names for execution")
    governance: ContractGovernance = Field(default_factory=ContractGovernance, description="Inherited read-only assertions")
    status: PlanStatus = Field(default=PlanStatus.PLANNED, description="Plan lifecycle status")
