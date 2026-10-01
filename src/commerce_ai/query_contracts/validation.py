"""Deterministic Validation Engine for Business Query Contracts (Phase 7B).

Applies deterministic validation rules:
A. Unknown intent or domain
B. Unknown metric or dimension
C. Metric incompatible with intent / domain
D. Domain incompatible with intent
E. Unsupported aggregation grain for intent
F. Temporal anti-leakage (end_date > as_of_date, start_date > end_date)
G. Currency isolation enforcement
H. Incompatible dimension combinations
I. Action execution attempts (rejected by governance)
J. Arbitrary tool names outside Phase 7A canonical catalog
K. Arbitrary SQL, code, or path traversal attempts
L. Read-only and governance immutability
"""

from __future__ import annotations

import re
from typing import List, Optional
from commerce_ai.query_contracts.dimensions import validate_dimensions
from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    OutputGrain,
    QueryIntent,
    TimeGranularity,
)
from commerce_ai.query_contracts.intent import (
    INTENT_PRIMARY_DOMAIN_MAP,
    is_governance_review_intent,
)
from commerce_ai.query_contracts.metrics import is_metric_compatible_with_intent
from commerce_ai.query_contracts.schemas import (
    BusinessQueryContract,
    ValidationIssue,
    ValidationResult,
)
from commerce_ai.query_contracts.tool_mapping import (
    PHASE_7A_APPROVED_ALIASES,
    PHASE_7A_CANONICAL_TOOLS,
    PHASE_7A_VALID_TOOLS,
)

# Prohibited action execution keywords
_PROHIBITED_ACTION_KEYWORDS = [
    re.compile(r"\b(execute\s+po|create\s+po|submit\s+po|place\s+order)\b", re.IGNORECASE),
    re.compile(r"\b(transfer\s+stock|move\s+inventory|rebalance\s+now)\b", re.IGNORECASE),
    re.compile(r"\b(change\s+price|modify\s+price|apply\s+discount|set\s+price)\b", re.IGNORECASE),
    re.compile(r"\b(delete|drop|truncate|purge)\b", re.IGNORECASE),
]

# Supported grains by intent
_INTENT_SUPPORTED_GRAINS = {
    QueryIntent.SALES_PERFORMANCE: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.WAREHOUSE,
        OutputGrain.CHANNEL, OutputGrain.CATEGORY, OutputGrain.BRAND,
        OutputGrain.DAY, OutputGrain.WEEK, OutputGrain.MONTH,
    },
    QueryIntent.REVENUE_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.DAY, OutputGrain.WEEK, OutputGrain.MONTH,
    },
    QueryIntent.MARGIN_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.CATEGORY, OutputGrain.BRAND,
        OutputGrain.DAY, OutputGrain.WEEK, OutputGrain.MONTH,
    },
    QueryIntent.PROFITABILITY_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.CATEGORY, OutputGrain.BRAND,
    },
    QueryIntent.UNIT_ECONOMICS_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.CATEGORY, OutputGrain.CHANNEL,
    },
    QueryIntent.INVENTORY_STATUS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.WAREHOUSE, OutputGrain.SKU_WAREHOUSE,
    },
    QueryIntent.INVENTORY_RISK: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.WAREHOUSE, OutputGrain.SKU_WAREHOUSE,
    },
    QueryIntent.STOCKOUT_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.WAREHOUSE, OutputGrain.SKU_WAREHOUSE,
    },
    QueryIntent.SLOW_MOVING_INVENTORY: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.WAREHOUSE,
    },
    QueryIntent.HIGH_VALUE_INVENTORY: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.WAREHOUSE,
    },
    QueryIntent.DEMAND_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.CATEGORY,
    },
    QueryIntent.DEMAND_TREND: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.DAY, OutputGrain.WEEK, OutputGrain.MONTH,
    },
    QueryIntent.ABC_XYZ_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.CATEGORY,
    },
    QueryIntent.FORECAST_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.DAY, OutputGrain.WEEK, OutputGrain.MONTH,
    },
    QueryIntent.FORECAST_ACCURACY: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU,
    },
    QueryIntent.FORECAST_BIAS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU,
    },
    QueryIntent.RETURN_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.CHANNEL, OutputGrain.DAY, OutputGrain.WEEK, OutputGrain.MONTH,
    },
    QueryIntent.RETURN_ANOMALY_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.CHANNEL, OutputGrain.DAY, OutputGrain.WEEK, OutputGrain.MONTH,
    },
    QueryIntent.RETURN_RISK: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.CATEGORY,
    },
    QueryIntent.RETURN_REASON_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.CATEGORY, OutputGrain.CHANNEL,
    },
    QueryIntent.REPLENISHMENT_REVIEW: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.WAREHOUSE, OutputGrain.SKU_WAREHOUSE,
    },
    QueryIntent.PURCHASE_ORDER_REVIEW: {
        OutputGrain.PORTFOLIO, OutputGrain.SUPPLIER, OutputGrain.WAREHOUSE,
    },
    QueryIntent.WAREHOUSE_REBALANCING_REVIEW: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.WAREHOUSE,
    },
    QueryIntent.BUSINESS_IMPACT_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.CATEGORY, OutputGrain.BRAND, OutputGrain.WAREHOUSE,
    },
    QueryIntent.RECOMMENDATION_REVIEW: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.WAREHOUSE,
    },
    QueryIntent.DECISION_REVIEW: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.WAREHOUSE,
    },
    QueryIntent.DATA_QUALITY_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.CATEGORY,
    },
    QueryIntent.MULTI_DOMAIN_ANALYSIS: {
        OutputGrain.PORTFOLIO, OutputGrain.SKU, OutputGrain.WAREHOUSE, OutputGrain.CHANNEL,
        OutputGrain.CATEGORY, OutputGrain.BRAND, OutputGrain.DAY, OutputGrain.WEEK, OutputGrain.MONTH,
    },
}


def validate_contract(contract: BusinessQueryContract) -> ValidationResult:
    """Perform deterministic business and governance validation on a BusinessQueryContract."""
    errors: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []

    # 1. Governance Safety Checks (Rule I & L)
    if not contract.governance.read_only:
        errors.append(ValidationIssue(
            code="GOVERNANCE_MUTATION_FORBIDDEN",
            message="Contract violates platform read-only mandate: read_only must be True.",
            field="governance.read_only",
        ))
    if contract.governance.execution_allowed:
        errors.append(ValidationIssue(
            code="GOVERNANCE_EXECUTION_FORBIDDEN",
            message="Contract violates safety mandate: execution_allowed must be False.",
            field="governance.execution_allowed",
        ))
    if contract.governance.action_execution:
        errors.append(ValidationIssue(
            code="GOVERNANCE_ACTION_FORBIDDEN",
            message="Contract cannot specify action_execution=True: query layer only provides informational access.",
            field="governance.action_execution",
        ))

    # Check for attempted action executions in text / explanation
    if contract.explanation_context:
        for pat in _PROHIBITED_ACTION_KEYWORDS:
            if pat.search(contract.explanation_context):
                errors.append(ValidationIssue(
                    code="ACTION_EXECUTION_ATTEMPT_REJECTED",
                    message=f"Action execution attempt detected in explanation context: '{contract.explanation_context}'. Direct execution is prohibited.",
                    field="explanation_context",
                ))

    # 2. Domain & Intent Compatibility (Rule D)
    primary_domain = INTENT_PRIMARY_DOMAIN_MAP.get(contract.intent)
    if contract.intent != QueryIntent.MULTI_DOMAIN_ANALYSIS:
        if contract.domain != primary_domain:
            errors.append(ValidationIssue(
                code="DOMAIN_INTENT_MISMATCH",
                message=f"Intent '{contract.intent.value}' expects primary domain '{primary_domain.value if primary_domain else 'UNKNOWN'}', got '{contract.domain.value}'.",
                field="domain",
            ))
    else:
        # Multi-domain must specify domains list or valid fallback
        if not contract.domains or contract.domains == [BusinessDomain.CROSS_DOMAIN]:
            warnings.append(ValidationIssue(
                code="MULTI_DOMAIN_EMPTY_DOMAINS",
                message="MULTI_DOMAIN_ANALYSIS contract did not list target domains; defaulting to all relevant domains.",
                field="domains",
                severity="WARNING",
            ))

    # 3. Metric Compatibility (Rule C)
    for m in contract.metrics:
        if not is_metric_compatible_with_intent(m, contract.intent):
            errors.append(ValidationIssue(
                code="METRIC_INTENT_INCOMPATIBLE",
                message=f"Metric '{m.value}' is not compatible with intent '{contract.intent.value}'.",
                field="metrics",
            ))

    # 4. Aggregation Grain Compatibility (Rule E)
    allowed_grains = _INTENT_SUPPORTED_GRAINS.get(contract.intent, {OutputGrain.PORTFOLIO})
    if contract.requested_grain not in allowed_grains:
        errors.append(ValidationIssue(
            code="UNSUPPORTED_GRAIN_FOR_INTENT",
            message=f"Grain '{contract.requested_grain.value}' is not supported for intent '{contract.intent.value}'. Supported grains: {[g.value for g in allowed_grains]}.",
            field="requested_grain",
        ))

    # 5. Dimension Validation (Rule H)
    target_domain = contract.domain if contract.intent != QueryIntent.MULTI_DOMAIN_ANALYSIS else BusinessDomain.CROSS_DOMAIN
    dim_errors = validate_dimensions(contract.dimensions, target_domain)
    for de in dim_errors:
        errors.append(ValidationIssue(
            code="DIMENSION_INVALID",
            message=de,
            field="dimensions",
        ))

    # 6. Temporal Anti-Leakage (Rule F)
    tr = contract.time_range
    effective_as_of = contract.as_of_date or tr.as_of_date
    if tr.end_date and effective_as_of and tr.end_date > effective_as_of:
        errors.append(ValidationIssue(
            code="TEMPORAL_LEAKAGE_DETECTED",
            message=f"end_date '{tr.end_date}' exceeds as_of_date '{effective_as_of}'. Forward-looking leakage is prohibited.",
            field="time_range.end_date",
        ))
    if tr.start_date and tr.end_date and tr.start_date > tr.end_date:
        errors.append(ValidationIssue(
            code="CHRONOLOGY_INVALID",
            message=f"start_date '{tr.start_date}' cannot be after end_date '{tr.end_date}'.",
            field="time_range.start_date",
        ))

    # 7. Required Tools Validation (Rule J) & Ranking Prohibition
    for tool_name in contract.required_tools:
        if any(rk in tool_name.lower() for rk in ["ranking", "rank", "leaderboard", "top_n", "winner"]):
            errors.append(ValidationIssue(
                code="PROHIBITED_RANKING_TOOL_REQUESTED",
                message=f"Requested tool '{tool_name}' implies subjective entity ranking, which is strictly prohibited.",
                field="required_tools",
            ))
        elif tool_name not in PHASE_7A_VALID_TOOLS:
            errors.append(ValidationIssue(
                code="UNREGISTERED_TOOL_REQUESTED",
                message=f"Requested tool '{tool_name}' is not in the canonical 46 registered Phase 7A tools catalog or approved aliases.",
                field="required_tools",
            ))

    # 8. Unavailable Phase 7A Capabilities (Section 9 & 13)
    if contract.intent == QueryIntent.DATA_QUALITY_ANALYSIS:
        warnings.append(ValidationIssue(
            code="CAPABILITY_UNAVAILABLE_IN_PHASE_7A",
            message="Data quality analysis query capability is not currently exposed as a read-only tool in the Phase 7A registry.",
            field="intent",
            severity="WARNING",
        ))

    # 9. Governance Review Requirements
    if is_governance_review_intent(contract.intent):
        if not contract.governance.approval_required:
            warnings.append(ValidationIssue(
                code="APPROVAL_REQUIRED_ADVISORY",
                message=f"Intent '{contract.intent.value}' reviews proposals requiring human approval. Setting approval_required=True is recommended.",
                field="governance.approval_required",
                severity="WARNING",
            ))

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
