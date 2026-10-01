"""Query Intent Definitions, Taxonomy Grouping, and Normalization (Phase 7B).

Provides deterministic intent categorization and helpers:
- Mapping from QueryIntent to canonical primary BusinessDomain
- Intent classification (commercial, inventory, demand, forecasting, returns, operations, impact, governance)
- Intent string normalization
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set
from commerce_ai.query_contracts.enums import BusinessDomain, QueryIntent

# Primary domain mapping for each canonical intent
INTENT_PRIMARY_DOMAIN_MAP: Dict[QueryIntent, BusinessDomain] = {
    # Sales & Financial
    QueryIntent.SALES_PERFORMANCE: BusinessDomain.SALES,
    QueryIntent.REVENUE_ANALYSIS: BusinessDomain.FINANCIAL,
    QueryIntent.MARGIN_ANALYSIS: BusinessDomain.FINANCIAL,
    QueryIntent.PROFITABILITY_ANALYSIS: BusinessDomain.FINANCIAL,
    QueryIntent.UNIT_ECONOMICS_ANALYSIS: BusinessDomain.FINANCIAL,
    # Inventory
    QueryIntent.INVENTORY_STATUS: BusinessDomain.INVENTORY,
    QueryIntent.INVENTORY_RISK: BusinessDomain.INVENTORY,
    QueryIntent.STOCKOUT_ANALYSIS: BusinessDomain.INVENTORY,
    QueryIntent.SLOW_MOVING_INVENTORY: BusinessDomain.INVENTORY,
    QueryIntent.HIGH_VALUE_INVENTORY: BusinessDomain.INVENTORY,
    # Demand
    QueryIntent.DEMAND_ANALYSIS: BusinessDomain.DEMAND,
    QueryIntent.DEMAND_TREND: BusinessDomain.DEMAND,
    QueryIntent.ABC_XYZ_ANALYSIS: BusinessDomain.DEMAND,
    # Forecasting
    QueryIntent.FORECAST_ANALYSIS: BusinessDomain.FORECASTING,
    QueryIntent.FORECAST_ACCURACY: BusinessDomain.FORECASTING,
    QueryIntent.FORECAST_BIAS: BusinessDomain.FORECASTING,
    # Returns
    QueryIntent.RETURN_ANALYSIS: BusinessDomain.RETURNS,
    QueryIntent.RETURN_ANOMALY_ANALYSIS: BusinessDomain.RETURNS,
    QueryIntent.RETURN_RISK: BusinessDomain.RETURNS,
    QueryIntent.RETURN_REASON_ANALYSIS: BusinessDomain.RETURNS,
    # Operations
    QueryIntent.REPLENISHMENT_REVIEW: BusinessDomain.OPERATIONS,
    QueryIntent.PURCHASE_ORDER_REVIEW: BusinessDomain.OPERATIONS,
    QueryIntent.WAREHOUSE_REBALANCING_REVIEW: BusinessDomain.OPERATIONS,
    # Business Impact
    QueryIntent.BUSINESS_IMPACT_ANALYSIS: BusinessDomain.BUSINESS_IMPACT,
    # Recommendations & Decisions
    QueryIntent.RECOMMENDATION_REVIEW: BusinessDomain.RECOMMENDATIONS,
    QueryIntent.DECISION_REVIEW: BusinessDomain.DECISIONS,
    # Data Quality & Multi-domain
    QueryIntent.DATA_QUALITY_ANALYSIS: BusinessDomain.DATA_QUALITY,
    QueryIntent.MULTI_DOMAIN_ANALYSIS: BusinessDomain.CROSS_DOMAIN,
}

# Intents requiring approval_required flag reflecting underlying governance
GOVERNANCE_REVIEW_INTENTS: Set[QueryIntent] = {
    QueryIntent.REPLENISHMENT_REVIEW,
    QueryIntent.PURCHASE_ORDER_REVIEW,
    QueryIntent.WAREHOUSE_REBALANCING_REVIEW,
    QueryIntent.RECOMMENDATION_REVIEW,
    QueryIntent.DECISION_REVIEW,
}

# Common phrase to intent lookup table for deterministic normalization
INTENT_NORMALIZATION_MAP: Dict[str, QueryIntent] = {
    "sales": QueryIntent.SALES_PERFORMANCE,
    "sales_performance": QueryIntent.SALES_PERFORMANCE,
    "revenue": QueryIntent.REVENUE_ANALYSIS,
    "revenue_analysis": QueryIntent.REVENUE_ANALYSIS,
    "margin": QueryIntent.MARGIN_ANALYSIS,
    "margins": QueryIntent.MARGIN_ANALYSIS,
    "margin_analysis": QueryIntent.MARGIN_ANALYSIS,
    "profitability": QueryIntent.PROFITABILITY_ANALYSIS,
    "profitability_analysis": QueryIntent.PROFITABILITY_ANALYSIS,
    "unit_economics": QueryIntent.UNIT_ECONOMICS_ANALYSIS,
    "inventory": QueryIntent.INVENTORY_STATUS,
    "inventory_status": QueryIntent.INVENTORY_STATUS,
    "inventory_position": QueryIntent.INVENTORY_STATUS,
    "inventory_risk": QueryIntent.INVENTORY_RISK,
    "stockout": QueryIntent.STOCKOUT_ANALYSIS,
    "stockouts": QueryIntent.STOCKOUT_ANALYSIS,
    "stockout_risk": QueryIntent.STOCKOUT_ANALYSIS,
    "stockout_analysis": QueryIntent.STOCKOUT_ANALYSIS,
    "slow_moving": QueryIntent.SLOW_MOVING_INVENTORY,
    "slow_moving_inventory": QueryIntent.SLOW_MOVING_INVENTORY,
    "aged_inventory": QueryIntent.SLOW_MOVING_INVENTORY,
    "high_value": QueryIntent.HIGH_VALUE_INVENTORY,
    "high_value_inventory": QueryIntent.HIGH_VALUE_INVENTORY,
    "demand": QueryIntent.DEMAND_ANALYSIS,
    "demand_analysis": QueryIntent.DEMAND_ANALYSIS,
    "demand_trend": QueryIntent.DEMAND_TREND,
    "abc_xyz": QueryIntent.ABC_XYZ_ANALYSIS,
    "abc_xyz_analysis": QueryIntent.ABC_XYZ_ANALYSIS,
    "forecast": QueryIntent.FORECAST_ANALYSIS,
    "forecasting": QueryIntent.FORECAST_ANALYSIS,
    "forecast_analysis": QueryIntent.FORECAST_ANALYSIS,
    "forecast_accuracy": QueryIntent.FORECAST_ACCURACY,
    "forecast_bias": QueryIntent.FORECAST_BIAS,
    "returns": QueryIntent.RETURN_ANALYSIS,
    "return_rate": QueryIntent.RETURN_ANALYSIS,
    "return_analysis": QueryIntent.RETURN_ANALYSIS,
    "return_anomalies": QueryIntent.RETURN_ANOMALY_ANALYSIS,
    "return_anomaly_analysis": QueryIntent.RETURN_ANOMALY_ANALYSIS,
    "return_risk": QueryIntent.RETURN_RISK,
    "return_reasons": QueryIntent.RETURN_REASON_ANALYSIS,
    "return_reason_analysis": QueryIntent.RETURN_REASON_ANALYSIS,
    "replenishment": QueryIntent.REPLENISHMENT_REVIEW,
    "replenishment_review": QueryIntent.REPLENISHMENT_REVIEW,
    "purchase_order": QueryIntent.PURCHASE_ORDER_REVIEW,
    "purchase_order_review": QueryIntent.PURCHASE_ORDER_REVIEW,
    "po_review": QueryIntent.PURCHASE_ORDER_REVIEW,
    "warehouse_rebalancing": QueryIntent.WAREHOUSE_REBALANCING_REVIEW,
    "warehouse_rebalancing_review": QueryIntent.WAREHOUSE_REBALANCING_REVIEW,
    "rebalancing": QueryIntent.WAREHOUSE_REBALANCING_REVIEW,
    "business_impact": QueryIntent.BUSINESS_IMPACT_ANALYSIS,
    "business_impact_analysis": QueryIntent.BUSINESS_IMPACT_ANALYSIS,
    "impact_analysis": QueryIntent.BUSINESS_IMPACT_ANALYSIS,
    "recommendations": QueryIntent.RECOMMENDATION_REVIEW,
    "recommendation_review": QueryIntent.RECOMMENDATION_REVIEW,
    "decisions": QueryIntent.DECISION_REVIEW,
    "decision_review": QueryIntent.DECISION_REVIEW,
    "data_quality": QueryIntent.DATA_QUALITY_ANALYSIS,
    "data_quality_analysis": QueryIntent.DATA_QUALITY_ANALYSIS,
    "multi_domain": QueryIntent.MULTI_DOMAIN_ANALYSIS,
    "multi_domain_analysis": QueryIntent.MULTI_DOMAIN_ANALYSIS,
}


def get_default_domain_for_intent(intent: QueryIntent) -> BusinessDomain:
    """Return the canonical business domain associated with an intent."""
    return INTENT_PRIMARY_DOMAIN_MAP.get(intent, BusinessDomain.SALES)


def is_governance_review_intent(intent: QueryIntent) -> bool:
    """Check if the intent is an operational or governance proposal review requiring approval."""
    return intent in GOVERNANCE_REVIEW_INTENTS


def normalize_intent(val: Union[str, QueryIntent]) -> Optional[QueryIntent]:
    """Deterministically normalize string or enum into a canonical QueryIntent."""
    if isinstance(val, QueryIntent):
        return val
    cleaned = str(val).strip().lower().replace(" ", "_").replace("-", "_")
    return INTENT_NORMALIZATION_MAP.get(cleaned)
