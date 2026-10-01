"""Deterministic QueryPlan Generation for Business Query Contracts (Phase 7B).

Synthesizes a validated BusinessQueryContract into an ordered execution blueprint (QueryPlan)
composed of discrete ToolCallStep instances matching Phase 7A query layer contracts.

Does NOT execute tools or make API calls; produces an unexecuted plan for Phase 7C orchestration.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional
from commerce_ai.query_contracts.enums import (
    OutputGrain,
    PlanStatus,
    TimeGranularity,
)
from commerce_ai.query_contracts.schemas import (
    BusinessQueryContract,
    QueryPlan,
    ToolCallStep,
)
from commerce_ai.query_contracts.tool_mapping import (
    PHASE_7A_CANONICAL_TOOLS,
    map_contract_to_tools,
)


def _build_tool_arguments(contract: BusinessQueryContract, tool_name: str) -> Dict[str, Any]:
    """Convert contract filters and temporal constraints into QueryContext-compatible keyword arguments."""
    f = contract.filters
    tr = contract.time_range

    args: Dict[str, Any] = {}

    as_of = contract.as_of_date or tr.as_of_date
    if as_of:
        args["as_of_date"] = as_of
    if tr.start_date:
        args["start_date"] = tr.start_date
    if tr.end_date:
        args["end_date"] = tr.end_date

    curr = contract.currency or f.currency
    if curr:
        args["currency"] = curr

    if f.sku_id:
        args["sku_id"] = f.sku_id
    if f.sku_ids:
        args["sku_ids"] = f.sku_ids
    if f.warehouse_id:
        args["warehouse_id"] = f.warehouse_id
    if f.warehouse_ids:
        args["warehouse_ids"] = f.warehouse_ids
    if f.channel_id:
        args["channel_id"] = f.channel_id
    if f.channel_ids:
        args["channel_ids"] = f.channel_ids
    if f.category_id:
        args["category_id"] = f.category_id
    if f.category_ids:
        args["category_ids"] = f.category_ids
    if f.brand:
        args["brand"] = f.brand
    if f.brands:
        args["brands"] = f.brands
    if f.supplier_id:
        args["supplier_id"] = f.supplier_id
    if f.supplier_ids:
        args["supplier_ids"] = f.supplier_ids
    if f.velocity_tier:
        args["velocity_tier"] = f.velocity_tier
    if f.abc_class:
        args["abc_class"] = f.abc_class
    if f.xyz_class:
        args["xyz_class"] = f.xyz_class

    if contract.time_granularity != TimeGranularity.NONE:
        args["time_grain"] = contract.time_granularity.value.lower()
    elif contract.requested_grain in {OutputGrain.DAY, OutputGrain.WEEK, OutputGrain.MONTH}:
        args["time_grain"] = contract.requested_grain.value.lower()

    if f.limit is not None:
        args["limit"] = f.limit
    if f.offset is not None:
        args["offset"] = f.offset

    return args


def _describe_tool_purpose(tool_name: str, contract: BusinessQueryContract) -> str:
    """Return a descriptive explanation of the tool's role in fulfilling the contract."""
    purposes = {
        "get_sales_summary": "Retrieve high-level commercial sales and order volume KPIs",
        "get_sales_trend": "Extract temporal revenue and unit time series trend",
        "get_sales_by_sku": "Retrieve individual SKU performance records",
        "get_sales_by_channel": "Segment commercial revenue by sales channel",
        "get_sales_by_warehouse": "Segment sales fulfillment volume by distribution center",
        "get_sales_by_category": "Segment sales performance across product categories",
        "get_sales_by_brand": "Segment sales performance across merchandise brands",
        "get_revenue_summary": "Reconcile commercial gross revenue, discounts, and net revenue",
        "get_margin_summary": "Extract portfolio COGS, gross margins, and profitability ratios",
        "get_unit_economics": "Analyze unit net revenue and direct variable cost breakdown",
        "get_margin_drivers": "Extract margin waterfall and profitability driver distributions",
        "get_operational_economics": "Evaluate fulfillment operational expenses and cost completeness",
        "get_profitability_attribution": "Analyze price, cost, volume, and mix margin variances",
        "get_inventory_summary": "Retrieve network-wide stock on hand, units, and tied-up capital",
        "get_inventory_position": "Extract detailed SKU-level stock positions and coverage",
        "get_stockout_risk": "Identify stockout risks and projected coverage runouts",
        "get_slow_moving_inventory": "Quantify capital tied up in aged and slow-moving stock tiers",
        "get_high_value_inventory": "Evaluate capital exposure in high-unit-cost merchandise",
        "get_inventory_risk": "Categorize portfolio items into critical stockout, understock, healthy, or overstock risk tiers",
        "get_inventory_risk_breakdown": "Summarize aggregate inventory risk classifications",
        "get_demand_summary": "Retrieve network daily velocity and aggregate demand volume",
        "get_demand_trend": "Extract historical demand volume trend across chosen grain",
        "get_demand_profile": "Analyze demand intermittency profiles and variability",
        "get_abc_xyz_distribution": "Examine portfolio distribution across ABC-XYZ matrices",
        "get_forecast": "Retrieve forward-looking demand predictions",
        "get_forecast_accuracy": "Evaluate historical forecast precision (MAE, RMSE, WAPE)",
        "get_forecast_bias": "Assess forecast tracking signal and directional bias",
        "get_forecast_summary": "Summarize portfolio-level forecast performance",
        "get_return_summary": "Retrieve return volumes, order counts, and value exposure",
        "get_return_rate": "Examine return rates against platform benchmarks",
        "get_return_trend": "Analyze temporal return volume and rate progression",
        "get_return_anomalies": "Identify statistically anomalous return spikes",
        "get_return_risk": "Score products and categories with elevated return risk",
        "get_return_reason_breakdown": "Break down return volume by customer-reported reason codes",
        "get_replenishment_reviews": "Review recommended replenishment proposals and order quantities",
        "get_purchase_order_reviews": "Review draft purchase order proposals and supplier constraints",
        "get_warehouse_rebalancing_reviews": "Review proposed inter-facility inventory transfers",
        "get_business_impact_summary": "Extract gross, deduplicated, and capital exposure metrics",
        "get_business_impact_by_category": "Break down financial exposure by impact category",
        "get_business_impact_by_type": "Break down financial exposure by economic impact type",
        "get_physical_capital_exposure": "Extract deduplicated physical inventory capital at risk",
        "get_recommendation_summary": "Summarize pending business recommendations and priorities",
        "get_recommendations": "Retrieve detailed list of candidate recommendations for review",
        "get_recommendation_by_id": "Inspect specific recommendation lineage and audit trail",
        "get_decision_summary": "Summarize packaged decision proposals awaiting human review",
        "get_decision_packages": "Retrieve decision packages with trade-offs and option evaluations",
        "get_decision_package_by_id": "Inspect specific decision package options and risk flags",
    }
    return purposes.get(tool_name, f"Execute canonical tool {tool_name} for intent {contract.intent.value}")


def create_query_plan(contract: BusinessQueryContract) -> QueryPlan:
    """Deterministically synthesize a validated BusinessQueryContract into an unexecuted QueryPlan."""
    # Resolve tools
    tool_names = contract.required_tools if contract.required_tools else map_contract_to_tools(contract)

    # Handle unavailable capabilities gracefully
    if not tool_names:
        plan_body = {
            "contract_id": contract.query_id,
            "tools": [],
            "steps": [],
            "status": PlanStatus.UNAVAILABLE.value,
        }
        plan_json = json.dumps(plan_body, sort_keys=True)
        plan_id = f"PLAN-{hashlib.sha256(plan_json.encode('utf-8')).hexdigest()[:16]}"
        return QueryPlan(
            plan_id=plan_id,
            contract_id=contract.query_id,
            steps=[],
            required_tools=[],
            dependencies={},
            execution_order=[],
            governance=contract.governance,
            status=PlanStatus.UNAVAILABLE,
        )

    # Determine execution sequencing:
    # Sequence 1: Summary / Top-level KPI queries
    # Sequence 2: Trend / Breakdown queries
    # Sequence 3: Deep dive / Table / Diagnostic queries
    steps: List[ToolCallStep] = []
    seq = 1

    # First pass: summaries
    summary_tools = [t for t in tool_names if "summary" in t]
    other_tools = [t for t in tool_names if t not in summary_tools]

    execution_order: List[str] = []

    for t in summary_tools:
        args = _build_tool_arguments(contract, t)
        purpose = _describe_tool_purpose(t, contract)
        steps.append(ToolCallStep(
            tool_name=t,
            arguments=args,
            purpose=purpose,
            required=True,
            sequence=seq,
        ))
        execution_order.append(t)
        seq += 1

    for t in other_tools:
        args = _build_tool_arguments(contract, t)
        purpose = _describe_tool_purpose(t, contract)
        steps.append(ToolCallStep(
            tool_name=t,
            arguments=args,
            purpose=purpose,
            required=False if summary_tools else True,
            sequence=seq,
        ))
        execution_order.append(t)
        seq += 1

    # Dependencies: secondary steps depend on first step if a summary exists
    dependencies: Dict[str, List[str]] = {}
    if summary_tools:
        primary = summary_tools[0]
        for t in other_tools:
            dependencies[t] = [primary]

    # Deterministic plan ID via SHA-256
    plan_body = {
        "contract_id": contract.query_id,
        "tools": execution_order,
        "steps": [s.model_dump() for s in steps],
    }
    plan_json = json.dumps(plan_body, sort_keys=True)
    plan_id = f"PLAN-{hashlib.sha256(plan_json.encode('utf-8')).hexdigest()[:16]}"

    return QueryPlan(
        plan_id=plan_id,
        contract_id=contract.query_id,
        steps=steps,
        required_tools=tool_names,
        dependencies=dependencies,
        execution_order=execution_order,
        governance=contract.governance,
        status=PlanStatus.PLANNED,
    )
