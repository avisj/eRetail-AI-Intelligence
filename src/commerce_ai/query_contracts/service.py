"""Query Contract Service Facade (Phase 7B).

Provides a centralized service for building, normalizing, validating, and planning
BusinessQueryContracts for Dashboards, eRetail Copilot, and future agentic workflows:
- build_contract: Construct and normalize a validated BusinessQueryContract
- validate_contract: Deterministic validation against governance and domain constraints
- create_plan: Generate an unexecuted QueryPlan with ordered ToolCallStep instances
- build_from_template: Instantiate a contract from a canonical business question template
- match_template: Match a natural-language query against canonical templates
- is_request_supported: Assess whether an intent/action is supported or prohibited
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    BusinessGrain,
    ComparisonType,
    ContractPriority,
    MetricIdentifier,
    OutputGrain,
    ProvenanceLevel,
    QueryIntent,
    RequestedOutput,
    TimeGranularity,
    TimeRangePreset,
)
from commerce_ai.query_contracts.intent import (
    get_default_domain_for_intent,
    is_governance_review_intent,
    normalize_intent,
)
from commerce_ai.query_contracts.metrics import (
    get_default_metrics_for_intent,
    normalize_metric,
)
from commerce_ai.query_contracts.planner import create_query_plan
from commerce_ai.query_contracts.schemas import (
    BusinessQueryContract,
    BusinessQueryFilter,
    ComparisonContract,
    ConfidenceRequirement,
    ContractGovernance,
    EvidenceRequirement,
    QueryPlan,
    TimeRangeContract,
    ValidationResult,
)
from commerce_ai.query_contracts.templates import (
    CANONICAL_TEMPLATES,
    QuestionTemplate,
    find_template_by_id,
    match_template,
)
from commerce_ai.query_contracts.tool_mapping import map_contract_to_tools
from commerce_ai.query_contracts.validation import validate_contract


def _compute_contract_query_id(
    intent: QueryIntent,
    domain: BusinessDomain,
    domains: List[BusinessDomain],
    metrics: List[MetricIdentifier],
    dimensions: List[BusinessDimension],
    filters: BusinessQueryFilter,
    time_range: TimeRangeContract,
    comparison: ComparisonContract,
    grain: BusinessGrain,
    output: RequestedOutput,
    tools: List[str],
    explanation: Optional[str],
    time_granularity: TimeGranularity = TimeGranularity.NONE,
    as_of_date: Optional[str] = None,
    currency: Optional[str] = None,
) -> str:
    """Compute deterministic SHA-256 query ID from contract parameters."""
    canonical_dict = {
        "intent": intent.value,
        "domain": domain.value,
        "domains": sorted([d.value for d in domains]),
        "metrics": sorted([m.value for m in metrics]),
        "dimensions": sorted([d.value for d in dimensions]),
        "filters": filters.model_dump(),
        "time_range": time_range.model_dump(),
        "comparison": comparison.model_dump(),
        "grain": grain.value,
        "time_granularity": time_granularity.value,
        "output": output.value,
        "tools": sorted(tools),
        "explanation": explanation or "",
        "as_of_date": as_of_date or "",
        "currency": currency or "",
    }
    encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
    return f"QRY-{hashlib.sha256(encoded).hexdigest()[:16]}"


class QueryContractService:
    """Unified service facade for Business Query Contracts and deterministic planning."""

    def __init__(self) -> None:
        pass

    def build_contract(
        self,
        intent: Union[str, QueryIntent],
        domain: Optional[Union[str, BusinessDomain]] = None,
        domains: Optional[Sequence[Union[str, BusinessDomain]]] = None,
        metrics: Optional[Sequence[Union[str, MetricIdentifier]]] = None,
        dimensions: Optional[Sequence[Union[str, BusinessDimension]]] = None,
        filters: Optional[Union[Dict[str, Any], BusinessQueryFilter]] = None,
        time_range: Optional[Union[Dict[str, Any], TimeRangeContract]] = None,
        comparison: Optional[Union[Dict[str, Any], ComparisonContract]] = None,
        as_of_date: Optional[str] = None,
        currency: Optional[str] = None,
        requested_grain: Union[str, BusinessGrain] = BusinessGrain.PORTFOLIO,
        time_granularity: Union[str, TimeGranularity] = TimeGranularity.NONE,
        requested_output: Union[str, RequestedOutput] = RequestedOutput.KPI,
        priority: Union[str, ContractPriority] = ContractPriority.NORMAL,
        explanation_context: Optional[str] = None,
        required_provenance: Optional[Union[str, ProvenanceLevel]] = None,
        required_tools: Optional[List[str]] = None,
        auto_validate: bool = True,
    ) -> Tuple[BusinessQueryContract, ValidationResult]:
        """Construct, normalize, and validate a primary BusinessQueryContract."""
        # 1. Normalize Intent
        norm_intent = normalize_intent(intent) if isinstance(intent, str) else intent
        if not norm_intent or not isinstance(norm_intent, QueryIntent):
            raise ValueError(f"Unknown or invalid query intent: '{intent}'")

        # 2. Determine Domain
        if domain is None:
            norm_domain = get_default_domain_for_intent(norm_intent)
        elif isinstance(domain, str):
            norm_domain = BusinessDomain(domain.upper())
        else:
            norm_domain = domain

        # Multi-domains list
        norm_domains: List[BusinessDomain] = []
        if domains:
            for d in domains:
                norm_domains.append(BusinessDomain(d.upper()) if isinstance(d, str) else d)
        elif norm_intent == QueryIntent.MULTI_DOMAIN_ANALYSIS:
            norm_domains = [norm_domain]

        # 3. Normalize Metrics
        norm_metrics: List[MetricIdentifier] = []
        if metrics is not None:
            for m in metrics:
                nm = normalize_metric(m) if isinstance(m, str) else m
                if nm and isinstance(nm, MetricIdentifier):
                    norm_metrics.append(nm)
                else:
                    raise ValueError(f"Unknown or invalid metric identifier: '{m}'")
        else:
            norm_metrics = get_default_metrics_for_intent(norm_intent)

        # 4. Normalize Dimensions
        norm_dims: List[BusinessDimension] = []
        if dimensions is not None:
            for d in dimensions:
                norm_dims.append(BusinessDimension(d.upper()) if isinstance(d, str) else d)

        # 5. Build Sub-Contracts
        eff_currency = currency
        if filters is None:
            b_filters = BusinessQueryFilter(currency=eff_currency)
        elif isinstance(filters, dict):
            f_dict = dict(filters)
            if eff_currency and "currency" not in f_dict:
                f_dict["currency"] = eff_currency
            b_filters = BusinessQueryFilter(**f_dict)
        else:
            b_filters = filters
            if not eff_currency and b_filters.currency:
                eff_currency = b_filters.currency

        if time_range is None:
            b_time = TimeRangeContract(as_of_date=as_of_date)
        elif isinstance(time_range, dict):
            if as_of_date and "as_of_date" not in time_range:
                time_range["as_of_date"] = as_of_date
            b_time = TimeRangeContract(**time_range)
        else:
            b_time = time_range

        eff_as_of = as_of_date or b_time.as_of_date

        if comparison is None:
            b_comp = ComparisonContract()
        elif isinstance(comparison, dict):
            b_comp = ComparisonContract(**comparison)
        else:
            b_comp = comparison

        # Output grain, time granularity & format
        norm_grain = BusinessGrain(requested_grain.upper()) if isinstance(requested_grain, str) else requested_grain
        norm_time_gran = TimeGranularity(time_granularity.upper()) if isinstance(time_granularity, str) else time_granularity
        norm_output = RequestedOutput(requested_output.upper()) if isinstance(requested_output, str) else requested_output
        norm_priority = ContractPriority(priority.upper()) if isinstance(priority, str) else priority

        # Governance setup
        is_gov = is_governance_review_intent(norm_intent)
        gov = ContractGovernance(
            read_only=True,
            execution_allowed=False,
            approval_required=is_gov,
            action_execution=False,
        )

        # Provenance requirement
        norm_prov = ProvenanceLevel(required_provenance.upper()) if isinstance(required_provenance, str) else required_provenance
        prov_req = ConfidenceRequirement(required_provenance=norm_prov)

        # Build preliminary contract to map tools
        prelim = BusinessQueryContract(
            query_id="TEMP",
            intent=norm_intent,
            domain=norm_domain,
            domains=norm_domains,
            metrics=norm_metrics,
            dimensions=norm_dims,
            filters=b_filters,
            time_range=b_time,
            comparison=b_comp,
            as_of_date=eff_as_of,
            currency=eff_currency,
            requested_grain=norm_grain,
            time_granularity=norm_time_gran,
            requested_output=norm_output,
            required_tools=[],
            priority=norm_priority,
            explanation_context=explanation_context,
            provenance_requirements=prov_req,
            governance=gov,
        )

        # Map tools deterministically if not provided
        resolved_tools = required_tools if required_tools is not None else map_contract_to_tools(prelim)

        # Compute deterministic Query ID
        query_id = _compute_contract_query_id(
            intent=norm_intent,
            domain=norm_domain,
            domains=norm_domains,
            metrics=norm_metrics,
            dimensions=norm_dims,
            filters=b_filters,
            time_range=b_time,
            comparison=b_comp,
            grain=norm_grain,
            time_granularity=norm_time_gran,
            output=norm_output,
            tools=resolved_tools,
            explanation=explanation_context,
            as_of_date=eff_as_of,
            currency=eff_currency,
        )

        final_contract = BusinessQueryContract(
            query_id=query_id,
            intent=norm_intent,
            domain=norm_domain,
            domains=norm_domains,
            metrics=norm_metrics,
            dimensions=norm_dims,
            filters=b_filters,
            time_range=b_time,
            comparison=b_comp,
            as_of_date=as_of_date or b_time.as_of_date,
            currency=currency or b_filters.currency,
            requested_grain=norm_grain,
            time_granularity=norm_time_gran,
            requested_output=norm_output,
            required_tools=resolved_tools,
            priority=norm_priority,
            explanation_context=explanation_context,
            provenance_requirements=prov_req,
            governance=gov,
        )

        validation_result = validate_contract(final_contract)
        if auto_validate and not validation_result.is_valid:
            # Still return contract along with failed validation result for inspection
            pass

        return final_contract, validation_result

    def validate(self, contract: BusinessQueryContract) -> ValidationResult:
        """Validate an existing BusinessQueryContract."""
        return validate_contract(contract)

    def plan(self, contract: BusinessQueryContract) -> QueryPlan:
        """Synthesize an unexecuted QueryPlan for the contract."""
        val = validate_contract(contract)
        if not val.is_valid:
            error_msgs = "; ".join(f"[{e.code}] {e.message}" for e in val.errors)
            raise ValueError(f"Cannot generate QueryPlan for invalid contract: {error_msgs}")
        return create_query_plan(contract)

    def list_templates(self) -> List[QuestionTemplate]:
        """List all canonical business question templates."""
        return list(CANONICAL_TEMPLATES)

    def match_template(self, question: str) -> Optional[QuestionTemplate]:
        """Match question string to a canonical template."""
        return match_template(question)

    def build_from_template(
        self,
        template_id: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[BusinessQueryContract, ValidationResult]:
        """Build a validated contract instantiated from a question template."""
        tpl = find_template_by_id(template_id)
        if not tpl:
            raise KeyError(f"Template with ID '{template_id}' was not found.")

        kwargs: Dict[str, Any] = {
            "intent": tpl.intent,
            "domain": tpl.domain,
            "domains": tpl.domains,
            "metrics": tpl.metrics,
            "dimensions": tpl.dimensions,
            "requested_grain": tpl.requested_grain,
            "requested_output": tpl.requested_output,
            "explanation_context": tpl.explanation_context,
            "required_tools": tpl.expected_tools,
        }
        if overrides:
            kwargs.update(overrides)

        return self.build_contract(**kwargs)
