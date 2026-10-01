"""Business Query Contracts Package (Phase 7B).

Defines the contract layer between conversational AI / Dashboards and the deterministic Phase 7A Query Layer:
- Controlled taxonomies: QueryIntent, BusinessDomain, MetricIdentifier, BusinessDimension, OutputGrain, RequestedOutput
- Core schemas: BusinessQueryContract, BusinessQueryFilter, TimeRangeContract, ComparisonContract, QueryPlan, ToolCallStep
- Deterministic validation and governance assertions (read-only, zero action execution)
- Deterministic tool mapping to Phase 7A canonical query tools
- Canonical business question templates
- QueryContractService unified facade
"""

from __future__ import annotations

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
from commerce_ai.query_contracts.schemas import (
    BusinessQueryContract,
    BusinessQueryFilter,
    ComparisonContract,
    ConfidenceRequirement,
    ContractGovernance,
    EvidenceRequirement,
    QueryPlan,
    TimeRangeContract,
    ToolCallStep,
    ValidationIssue,
    ValidationResult,
)
from commerce_ai.query_contracts.intent import (
    INTENT_PRIMARY_DOMAIN_MAP,
    get_default_domain_for_intent,
    is_governance_review_intent,
    normalize_intent,
)
from commerce_ai.query_contracts.metrics import (
    get_default_metrics_for_intent,
    is_metric_compatible_with_intent,
    normalize_metric,
)
from commerce_ai.query_contracts.dimensions import (
    DOMAIN_SUPPORTED_DIMENSIONS,
    is_dimension_supported,
    validate_dimensions,
)
from commerce_ai.query_contracts.tool_mapping import (
    PHASE_7A_APPROVED_ALIASES,
    PHASE_7A_CANONICAL_TOOLS,
    PHASE_7A_VALID_TOOLS,
    map_contract_to_tools,
)
from commerce_ai.query_contracts.validation import validate_contract
from commerce_ai.query_contracts.planner import create_query_plan
from commerce_ai.query_contracts.templates import (
    CANONICAL_TEMPLATES,
    QuestionTemplate,
    find_template_by_id,
    get_canonical_templates,
    match_template,
)
from commerce_ai.query_contracts.service import QueryContractService

__all__ = [
    # Enums
    "BusinessDimension",
    "BusinessDomain",
    "BusinessGrain",
    "ComparisonType",
    "ContractPriority",
    "MetricIdentifier",
    "OutputGrain",
    "PlanStatus",
    "ProvenanceLevel",
    "QueryIntent",
    "RequestedOutput",
    "TimeGranularity",
    "TimeRangePreset",
    # Schemas
    "BusinessQueryContract",
    "BusinessQueryFilter",
    "ComparisonContract",
    "ConfidenceRequirement",
    "ContractGovernance",
    "EvidenceRequirement",
    "QueryPlan",
    "TimeRangeContract",
    "ToolCallStep",
    "ValidationIssue",
    "ValidationResult",
    # Logic & Mapping
    "INTENT_PRIMARY_DOMAIN_MAP",
    "get_default_domain_for_intent",
    "is_governance_review_intent",
    "normalize_intent",
    "get_default_metrics_for_intent",
    "is_metric_compatible_with_intent",
    "normalize_metric",
    "DOMAIN_SUPPORTED_DIMENSIONS",
    "is_dimension_supported",
    "validate_dimensions",
    "PHASE_7A_CANONICAL_TOOLS",
    "PHASE_7A_APPROVED_ALIASES",
    "PHASE_7A_VALID_TOOLS",
    "map_contract_to_tools",
    "validate_contract",
    "create_query_plan",
    # Templates & Service
    "CANONICAL_TEMPLATES",
    "QuestionTemplate",
    "find_template_by_id",
    "get_canonical_templates",
    "match_template",
    "QueryContractService",
]
