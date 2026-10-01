"""Deterministic Mapping from Query Intent to Phase 7A Query Tools (Phase 7B).

Provides deterministic resolution from:
- QueryIntent
- Requested dimensions / grains
- Requested output types
- WHY-style investigation contexts
- Multi-domain requests
into allowed canonical Phase 7A tool names.
"""

from __future__ import annotations

from typing import List, Set
from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    OutputGrain,
    QueryIntent,
    RequestedOutput,
    TimeGranularity,
)
from commerce_ai.query_contracts.schemas import BusinessQueryContract


# Authoritative set of all 46 canonical registered Phase 7A query tools
PHASE_7A_CANONICAL_TOOLS: Set[str] = {
    # Sales (7)
    "get_sales_summary",
    "get_sales_trend",
    "get_sales_by_sku",
    "get_sales_by_channel",
    "get_sales_by_warehouse",
    "get_sales_by_category",
    "get_sales_by_brand",
    # Financial (6)
    "get_revenue_summary",
    "get_margin_summary",
    "get_unit_economics",
    "get_margin_drivers",
    "get_operational_economics",
    "get_profitability_attribution",
    # Inventory (6)
    "get_inventory_summary",
    "get_inventory_position",
    "get_stockout_risk",
    "get_slow_moving_inventory",
    "get_high_value_inventory",
    "get_inventory_risk",
    # Demand (4)
    "get_demand_summary",
    "get_demand_trend",
    "get_demand_profile",
    "get_abc_xyz_distribution",
    # Forecasting (4)
    "get_forecast",
    "get_forecast_accuracy",
    "get_forecast_bias",
    "get_forecast_summary",
    # Returns (6)
    "get_return_summary",
    "get_return_rate",
    "get_return_trend",
    "get_return_anomalies",
    "get_return_risk",
    "get_return_reason_breakdown",
    # Operations (3)
    "get_replenishment_reviews",
    "get_purchase_order_reviews",
    "get_warehouse_rebalancing_reviews",
    # Business Impact (4) - Strictly aggregate non-ranking tools
    "get_business_impact_summary",
    "get_business_impact_by_category",
    "get_business_impact_by_type",
    "get_physical_capital_exposure",
    # Recommendations (3)
    "get_recommendation_summary",
    "get_recommendations",
    "get_recommendation_by_id",
    # Decisions (3)
    "get_decision_summary",
    "get_decision_packages",
    "get_decision_package_by_id",
}

# Approved aliases in Phase 7A registry (mapping to canonical implementations)
PHASE_7A_APPROVED_ALIASES: Set[str] = {
    "get_inventory_risk_breakdown",  # alias for get_inventory_risk
    "get_recommendation_list",       # alias for get_recommendations
    "get_decision_list",             # alias for get_decision_packages
}

PHASE_7A_VALID_TOOLS: Set[str] = PHASE_7A_CANONICAL_TOOLS | PHASE_7A_APPROVED_ALIASES


def map_contract_to_tools(contract: BusinessQueryContract) -> List[str]:
    """Deterministically resolve a BusinessQueryContract to required Phase 7A tools."""
    tools: List[str] = []
    intent = contract.intent
    output = contract.requested_output
    dims = contract.dimensions
    grain = contract.requested_grain

    # 1. WHY-style investigations (Section 15)
    if contract.explanation_context:
        if intent in {QueryIntent.MARGIN_ANALYSIS, QueryIntent.PROFITABILITY_ANALYSIS}:
            return sorted(list(set([
                "get_margin_summary",
                "get_margin_drivers",
                "get_sales_by_channel",
                "get_sales_by_warehouse",
            ])))
        elif intent in {QueryIntent.RETURN_ANALYSIS, QueryIntent.RETURN_ANOMALY_ANALYSIS}:
            return sorted(list(set([
                "get_return_summary",
                "get_return_trend",
                "get_return_anomalies",
                "get_return_reason_breakdown",
            ])))
        elif intent in {QueryIntent.INVENTORY_RISK, QueryIntent.STOCKOUT_ANALYSIS}:
            return sorted(list(set([
                "get_inventory_summary",
                "get_stockout_risk",
                "get_slow_moving_inventory",
                "get_inventory_risk",
            ])))

    # 2. Multi-domain queries (Section 14)
    if intent == QueryIntent.MULTI_DOMAIN_ANALYSIS:
        target_domains = contract.domains if contract.domains else [contract.domain]
        selected: Set[str] = set()
        for dom in target_domains:
            if dom in {BusinessDomain.SALES, BusinessDomain.FINANCIAL}:
                selected.add("get_revenue_summary")
                selected.add("get_margin_summary")
            elif dom == BusinessDomain.INVENTORY:
                selected.add("get_inventory_summary")
            elif dom == BusinessDomain.DEMAND:
                selected.add("get_demand_summary")
            elif dom == BusinessDomain.FORECASTING:
                selected.add("get_forecast_summary")
            elif dom == BusinessDomain.RETURNS:
                selected.add("get_return_summary")
            elif dom == BusinessDomain.OPERATIONS:
                selected.add("get_replenishment_reviews")
            elif dom == BusinessDomain.BUSINESS_IMPACT:
                selected.add("get_business_impact_summary")
            elif dom == BusinessDomain.RECOMMENDATIONS:
                selected.add("get_recommendation_summary")
            elif dom == BusinessDomain.DECISIONS:
                selected.add("get_decision_summary")
            elif dom == BusinessDomain.CROSS_DOMAIN:
                # Default cross-domain triad: revenue, margin, inventory
                selected.add("get_revenue_summary")
                selected.add("get_margin_summary")
                selected.add("get_inventory_summary")
        return sorted(list(selected))

    # 3. Domain/Intent specific mappings
    if intent == QueryIntent.SALES_PERFORMANCE:
        if (
            output == RequestedOutput.TIME_SERIES
            or contract.time_granularity != TimeGranularity.NONE
            or grain in {OutputGrain.DAY, OutputGrain.WEEK, OutputGrain.MONTH}
        ):
            tools.append("get_sales_trend")
        elif BusinessDimension.SKU in dims or grain == OutputGrain.SKU:
            tools.append("get_sales_by_sku")
        elif BusinessDimension.CHANNEL in dims or grain == OutputGrain.CHANNEL:
            tools.append("get_sales_by_channel")
        elif BusinessDimension.WAREHOUSE in dims or grain == OutputGrain.WAREHOUSE:
            tools.append("get_sales_by_warehouse")
        elif BusinessDimension.CATEGORY in dims or grain == OutputGrain.CATEGORY:
            tools.append("get_sales_by_category")
        elif BusinessDimension.BRAND in dims or grain == OutputGrain.BRAND:
            tools.append("get_sales_by_brand")
        else:
            tools.append("get_sales_summary")

    elif intent == QueryIntent.REVENUE_ANALYSIS:
        tools.append("get_revenue_summary")
        if output == RequestedOutput.TIME_SERIES:
            tools.append("get_sales_trend")

    elif intent == QueryIntent.MARGIN_ANALYSIS:
        tools.append("get_margin_summary")
        if output in {RequestedOutput.BREAKDOWN, RequestedOutput.DETAIL}:
            tools.append("get_margin_drivers")

    elif intent == QueryIntent.PROFITABILITY_ANALYSIS:
        tools.append("get_profitability_attribution")
        tools.append("get_margin_summary")

    elif intent == QueryIntent.UNIT_ECONOMICS_ANALYSIS:
        tools.append("get_unit_economics")
        if output in {RequestedOutput.DETAIL, RequestedOutput.TABLE}:
            tools.append("get_operational_economics")

    elif intent == QueryIntent.INVENTORY_STATUS:
        if output in {RequestedOutput.TABLE, RequestedOutput.DETAIL} or BusinessDimension.SKU in dims:
            tools.append("get_inventory_position")
        else:
            tools.append("get_inventory_summary")

    elif intent == QueryIntent.INVENTORY_RISK:
        tools.append("get_inventory_risk")
        if output in {RequestedOutput.TABLE, RequestedOutput.DETAIL}:
            tools.append("get_stockout_risk")
            tools.append("get_slow_moving_inventory")

    elif intent == QueryIntent.STOCKOUT_ANALYSIS:
        tools.append("get_stockout_risk")

    elif intent == QueryIntent.SLOW_MOVING_INVENTORY:
        tools.append("get_slow_moving_inventory")

    elif intent == QueryIntent.HIGH_VALUE_INVENTORY:
        tools.append("get_high_value_inventory")

    elif intent == QueryIntent.DEMAND_ANALYSIS:
        tools.append("get_demand_summary")
        if output in {RequestedOutput.TABLE, RequestedOutput.DETAIL}:
            tools.append("get_demand_profile")

    elif intent == QueryIntent.DEMAND_TREND:
        tools.append("get_demand_trend")

    elif intent == QueryIntent.ABC_XYZ_ANALYSIS:
        tools.append("get_abc_xyz_distribution")

    elif intent == QueryIntent.FORECAST_ANALYSIS:
        tools.append("get_forecast")
        tools.append("get_forecast_summary")

    elif intent == QueryIntent.FORECAST_ACCURACY:
        tools.append("get_forecast_accuracy")

    elif intent == QueryIntent.FORECAST_BIAS:
        tools.append("get_forecast_bias")

    elif intent == QueryIntent.RETURN_ANALYSIS:
        tools.append("get_return_summary")
        tools.append("get_return_rate")
        if output == RequestedOutput.TIME_SERIES:
            tools.append("get_return_trend")

    elif intent == QueryIntent.RETURN_ANOMALY_ANALYSIS:
        tools.append("get_return_anomalies")

    elif intent == QueryIntent.RETURN_RISK:
        tools.append("get_return_risk")

    elif intent == QueryIntent.RETURN_REASON_ANALYSIS:
        tools.append("get_return_reason_breakdown")

    elif intent == QueryIntent.REPLENISHMENT_REVIEW:
        tools.append("get_replenishment_reviews")

    elif intent == QueryIntent.PURCHASE_ORDER_REVIEW:
        tools.append("get_purchase_order_reviews")

    elif intent == QueryIntent.WAREHOUSE_REBALANCING_REVIEW:
        tools.append("get_warehouse_rebalancing_reviews")

    elif intent == QueryIntent.BUSINESS_IMPACT_ANALYSIS:
        # Strictly aggregate business impact metrics - never ranking
        tools.append("get_business_impact_summary")
        if output == RequestedOutput.BREAKDOWN:
            tools.append("get_business_impact_by_category")
            tools.append("get_business_impact_by_type")
        elif output == RequestedOutput.DETAIL:
            tools.append("get_physical_capital_exposure")

    elif intent == QueryIntent.RECOMMENDATION_REVIEW:
        tools.append("get_recommendation_summary")
        tools.append("get_recommendations")

    elif intent == QueryIntent.DECISION_REVIEW:
        tools.append("get_decision_summary")
        tools.append("get_decision_packages")

    elif intent == QueryIntent.DATA_QUALITY_ANALYSIS:
        # Phase 7A has no dedicated read-only tool for data quality query layer.
        # Capability is marked unavailable in contract planning.
        return []

    # Fallback to safe summary tool only if not explicitly an unavailable capability
    if not tools and intent != QueryIntent.DATA_QUALITY_ANALYSIS:
        tools.append("get_sales_summary")

    # Return deduplicated, sorted list
    return sorted(list(set(tools)))
