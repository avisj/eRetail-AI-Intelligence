"""Metric Definitions, Domain Mappings, and Intent Compatibility (Phase 7B).

Provides controlled metric taxonomy logic:
- Maps MetricIdentifier to primary BusinessDomain
- Defines compatible metrics per QueryIntent
- Normalizes metric strings into canonical MetricIdentifier enums
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set
from commerce_ai.query_contracts.enums import BusinessDomain, MetricIdentifier, QueryIntent

# Primary domain mapping for each metric
METRIC_DOMAIN_MAP: Dict[MetricIdentifier, BusinessDomain] = {
    # Revenue & Commercial (Sales / Financial)
    MetricIdentifier.GROSS_REVENUE: BusinessDomain.SALES,
    MetricIdentifier.NET_REVENUE: BusinessDomain.SALES,
    MetricIdentifier.AOV: BusinessDomain.SALES,
    MetricIdentifier.UNITS: BusinessDomain.SALES,
    MetricIdentifier.ORDERS: BusinessDomain.SALES,
    # Margins & Economics (Financial)
    MetricIdentifier.GROSS_MARGIN: BusinessDomain.FINANCIAL,
    MetricIdentifier.GROSS_MARGIN_PERCENT: BusinessDomain.FINANCIAL,
    MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN: BusinessDomain.FINANCIAL,
    MetricIdentifier.FINAL_CONTRIBUTION_MARGIN: BusinessDomain.FINANCIAL,
    MetricIdentifier.COST_COMPLETENESS: BusinessDomain.FINANCIAL,
    # Inventory
    MetricIdentifier.INVENTORY_UNITS: BusinessDomain.INVENTORY,
    MetricIdentifier.INVENTORY_VALUE: BusinessDomain.INVENTORY,
    MetricIdentifier.STOCKOUT_EXPOSURE: BusinessDomain.INVENTORY,
    MetricIdentifier.SLOW_MOVING_EXPOSURE: BusinessDomain.INVENTORY,
    MetricIdentifier.HIGH_VALUE_EXPOSURE: BusinessDomain.INVENTORY,
    MetricIdentifier.DAYS_OF_SUPPLY: BusinessDomain.INVENTORY,
    # Demand
    MetricIdentifier.DAILY_DEMAND: BusinessDomain.DEMAND,
    MetricIdentifier.DEMAND_TREND: BusinessDomain.DEMAND,
    MetricIdentifier.VELOCITY: BusinessDomain.DEMAND,
    MetricIdentifier.INTERMITTENCY: BusinessDomain.DEMAND,
    MetricIdentifier.ABC_CLASS: BusinessDomain.DEMAND,
    MetricIdentifier.XYZ_CLASS: BusinessDomain.DEMAND,
    # Forecasting
    MetricIdentifier.FORECAST_VALUE: BusinessDomain.FORECASTING,
    MetricIdentifier.FORECAST_ACCURACY: BusinessDomain.FORECASTING,
    MetricIdentifier.FORECAST_BIAS: BusinessDomain.FORECASTING,
    MetricIdentifier.FORECAST_ERROR: BusinessDomain.FORECASTING,
    # Returns
    MetricIdentifier.RETURN_RATE: BusinessDomain.RETURNS,
    MetricIdentifier.RETURN_COUNT: BusinessDomain.RETURNS,
    MetricIdentifier.RETURNED_UNITS: BusinessDomain.RETURNS,
    MetricIdentifier.RETURN_REVENUE_EXPOSURE: BusinessDomain.RETURNS,
    MetricIdentifier.RETURN_RISK: BusinessDomain.RETURNS,
    MetricIdentifier.RETURN_ANOMALY_COUNT: BusinessDomain.RETURNS,
    # Business Impact
    MetricIdentifier.GROSS_SIGNAL_EXPOSURE: BusinessDomain.BUSINESS_IMPACT,
    MetricIdentifier.DEDUPLICATED_EXPOSURE: BusinessDomain.BUSINESS_IMPACT,
    MetricIdentifier.DEDUPLICATED_PHYSICAL_CAPITAL_EXPOSURE: BusinessDomain.BUSINESS_IMPACT,
    # Recommendations
    MetricIdentifier.RECOMMENDATION_COUNT: BusinessDomain.RECOMMENDATIONS,
    # Decisions
    MetricIdentifier.DECISION_PACKAGE_COUNT: BusinessDomain.DECISIONS,
    MetricIdentifier.PENDING_REVIEW_COUNT: BusinessDomain.DECISIONS,
    MetricIdentifier.CONFLICT_COUNT: BusinessDomain.DECISIONS,
    MetricIdentifier.INSUFFICIENT_INFORMATION_COUNT: BusinessDomain.DECISIONS,
}

# Compatible metrics per intent
INTENT_COMPATIBLE_METRICS: Dict[QueryIntent, Set[MetricIdentifier]] = {
    QueryIntent.SALES_PERFORMANCE: {
        MetricIdentifier.NET_REVENUE,
        MetricIdentifier.GROSS_REVENUE,
        MetricIdentifier.ORDERS,
        MetricIdentifier.UNITS,
        MetricIdentifier.AOV,
    },
    QueryIntent.REVENUE_ANALYSIS: {
        MetricIdentifier.GROSS_REVENUE,
        MetricIdentifier.NET_REVENUE,
        MetricIdentifier.AOV,
    },
    QueryIntent.MARGIN_ANALYSIS: {
        MetricIdentifier.GROSS_MARGIN,
        MetricIdentifier.GROSS_MARGIN_PERCENT,
        MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN,
        MetricIdentifier.FINAL_CONTRIBUTION_MARGIN,
    },
    QueryIntent.PROFITABILITY_ANALYSIS: {
        MetricIdentifier.GROSS_MARGIN,
        MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN,
        MetricIdentifier.FINAL_CONTRIBUTION_MARGIN,
        MetricIdentifier.COST_COMPLETENESS,
    },
    QueryIntent.UNIT_ECONOMICS_ANALYSIS: {
        MetricIdentifier.NET_REVENUE,
        MetricIdentifier.GROSS_MARGIN,
        MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN,
        MetricIdentifier.FINAL_CONTRIBUTION_MARGIN,
        MetricIdentifier.COST_COMPLETENESS,
    },
    QueryIntent.INVENTORY_STATUS: {
        MetricIdentifier.INVENTORY_UNITS,
        MetricIdentifier.INVENTORY_VALUE,
        MetricIdentifier.DAYS_OF_SUPPLY,
    },
    QueryIntent.INVENTORY_RISK: {
        MetricIdentifier.STOCKOUT_EXPOSURE,
        MetricIdentifier.SLOW_MOVING_EXPOSURE,
        MetricIdentifier.HIGH_VALUE_EXPOSURE,
        MetricIdentifier.DAYS_OF_SUPPLY,
        MetricIdentifier.INVENTORY_VALUE,
    },
    QueryIntent.STOCKOUT_ANALYSIS: {
        MetricIdentifier.STOCKOUT_EXPOSURE,
        MetricIdentifier.DAYS_OF_SUPPLY,
    },
    QueryIntent.SLOW_MOVING_INVENTORY: {
        MetricIdentifier.SLOW_MOVING_EXPOSURE,
        MetricIdentifier.INVENTORY_VALUE,
    },
    QueryIntent.HIGH_VALUE_INVENTORY: {
        MetricIdentifier.HIGH_VALUE_EXPOSURE,
        MetricIdentifier.INVENTORY_VALUE,
    },
    QueryIntent.DEMAND_ANALYSIS: {
        MetricIdentifier.DAILY_DEMAND,
        MetricIdentifier.VELOCITY,
        MetricIdentifier.INTERMITTENCY,
    },
    QueryIntent.DEMAND_TREND: {
        MetricIdentifier.DAILY_DEMAND,
        MetricIdentifier.DEMAND_TREND,
    },
    QueryIntent.ABC_XYZ_ANALYSIS: {
        MetricIdentifier.ABC_CLASS,
        MetricIdentifier.XYZ_CLASS,
        MetricIdentifier.VELOCITY,
    },
    QueryIntent.FORECAST_ANALYSIS: {
        MetricIdentifier.FORECAST_VALUE,
    },
    QueryIntent.FORECAST_ACCURACY: {
        MetricIdentifier.FORECAST_ACCURACY,
        MetricIdentifier.FORECAST_ERROR,
    },
    QueryIntent.FORECAST_BIAS: {
        MetricIdentifier.FORECAST_BIAS,
        MetricIdentifier.FORECAST_ERROR,
    },
    QueryIntent.RETURN_ANALYSIS: {
        MetricIdentifier.RETURN_RATE,
        MetricIdentifier.RETURN_COUNT,
        MetricIdentifier.RETURNED_UNITS,
        MetricIdentifier.RETURN_REVENUE_EXPOSURE,
    },
    QueryIntent.RETURN_ANOMALY_ANALYSIS: {
        MetricIdentifier.RETURN_ANOMALY_COUNT,
        MetricIdentifier.RETURN_REVENUE_EXPOSURE,
    },
    QueryIntent.RETURN_RISK: {
        MetricIdentifier.RETURN_RISK,
        MetricIdentifier.RETURN_RATE,
    },
    QueryIntent.RETURN_REASON_ANALYSIS: {
        MetricIdentifier.RETURN_COUNT,
        MetricIdentifier.RETURNED_UNITS,
    },
    QueryIntent.REPLENISHMENT_REVIEW: {
        MetricIdentifier.INVENTORY_UNITS,
        MetricIdentifier.DAYS_OF_SUPPLY,
    },
    QueryIntent.PURCHASE_ORDER_REVIEW: {
        MetricIdentifier.INVENTORY_VALUE,
        MetricIdentifier.ORDERS,
    },
    QueryIntent.WAREHOUSE_REBALANCING_REVIEW: {
        MetricIdentifier.INVENTORY_UNITS,
        MetricIdentifier.INVENTORY_VALUE,
    },
    QueryIntent.BUSINESS_IMPACT_ANALYSIS: {
        MetricIdentifier.GROSS_SIGNAL_EXPOSURE,
        MetricIdentifier.DEDUPLICATED_EXPOSURE,
        MetricIdentifier.DEDUPLICATED_PHYSICAL_CAPITAL_EXPOSURE,
    },
    QueryIntent.RECOMMENDATION_REVIEW: {
        MetricIdentifier.RECOMMENDATION_COUNT,
    },
    QueryIntent.DECISION_REVIEW: {
        MetricIdentifier.DECISION_PACKAGE_COUNT,
        MetricIdentifier.PENDING_REVIEW_COUNT,
        MetricIdentifier.CONFLICT_COUNT,
        MetricIdentifier.INSUFFICIENT_INFORMATION_COUNT,
    },
    QueryIntent.DATA_QUALITY_ANALYSIS: {
        MetricIdentifier.COST_COMPLETENESS,
    },
}

# Normalization string to metric lookup table
METRIC_NORMALIZATION_MAP: Dict[str, MetricIdentifier] = {
    "revenue": MetricIdentifier.NET_REVENUE,
    "net_revenue": MetricIdentifier.NET_REVENUE,
    "gross_revenue": MetricIdentifier.GROSS_REVENUE,
    "aov": MetricIdentifier.AOV,
    "average_order_value": MetricIdentifier.AOV,
    "units": MetricIdentifier.UNITS,
    "units_sold": MetricIdentifier.UNITS,
    "orders": MetricIdentifier.ORDERS,
    "order_count": MetricIdentifier.ORDERS,
    "margin": MetricIdentifier.GROSS_MARGIN,
    "gross_margin": MetricIdentifier.GROSS_MARGIN,
    "margin_percentage": MetricIdentifier.GROSS_MARGIN_PERCENT,
    "gross_margin_percent": MetricIdentifier.GROSS_MARGIN_PERCENT,
    "gross_margin_percentage": MetricIdentifier.GROSS_MARGIN_PERCENT,
    "contribution_margin": MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN,
    "known_contribution_margin": MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN,
    "final_contribution_margin": MetricIdentifier.FINAL_CONTRIBUTION_MARGIN,
    "cost_completeness": MetricIdentifier.COST_COMPLETENESS,
    "inventory": MetricIdentifier.INVENTORY_UNITS,
    "inventory_units": MetricIdentifier.INVENTORY_UNITS,
    "inventory_value": MetricIdentifier.INVENTORY_VALUE,
    "stockout_exposure": MetricIdentifier.STOCKOUT_EXPOSURE,
    "slow_moving_exposure": MetricIdentifier.SLOW_MOVING_EXPOSURE,
    "high_value_exposure": MetricIdentifier.HIGH_VALUE_EXPOSURE,
    "days_of_supply": MetricIdentifier.DAYS_OF_SUPPLY,
    "demand": MetricIdentifier.DAILY_DEMAND,
    "daily_demand": MetricIdentifier.DAILY_DEMAND,
    "demand_trend": MetricIdentifier.DEMAND_TREND,
    "velocity": MetricIdentifier.VELOCITY,
    "forecast": MetricIdentifier.FORECAST_VALUE,
    "forecast_value": MetricIdentifier.FORECAST_VALUE,
    "forecast_accuracy": MetricIdentifier.FORECAST_ACCURACY,
    "forecast_bias": MetricIdentifier.FORECAST_BIAS,
    "return_rate": MetricIdentifier.RETURN_RATE,
    "returns": MetricIdentifier.RETURN_COUNT,
    "return_count": MetricIdentifier.RETURN_COUNT,
    "returned_units": MetricIdentifier.RETURNED_UNITS,
    "return_exposure": MetricIdentifier.RETURN_REVENUE_EXPOSURE,
    "return_risk": MetricIdentifier.RETURN_RISK,
    "anomalies": MetricIdentifier.RETURN_ANOMALY_COUNT,
    "return_anomalies": MetricIdentifier.RETURN_ANOMALY_COUNT,
    "business_impact": MetricIdentifier.GROSS_SIGNAL_EXPOSURE,
    "gross_impact": MetricIdentifier.GROSS_SIGNAL_EXPOSURE,
    "deduplicated_exposure": MetricIdentifier.DEDUPLICATED_EXPOSURE,
    "capital_exposure": MetricIdentifier.DEDUPLICATED_PHYSICAL_CAPITAL_EXPOSURE,
    "recommendation_count": MetricIdentifier.RECOMMENDATION_COUNT,
    "decision_count": MetricIdentifier.DECISION_PACKAGE_COUNT,
}


def get_default_metrics_for_intent(intent: QueryIntent) -> List[MetricIdentifier]:
    """Return default metrics for an intent if none were explicitly specified."""
    metrics = INTENT_COMPATIBLE_METRICS.get(intent)
    return sorted(list(metrics), key=lambda x: x.value) if metrics else []


def is_metric_compatible_with_intent(metric: MetricIdentifier, intent: QueryIntent) -> bool:
    """Check if a specific metric is compatible with a given intent."""
    if intent == QueryIntent.MULTI_DOMAIN_ANALYSIS:
        return True
    compatible = INTENT_COMPATIBLE_METRICS.get(intent, set())
    return metric in compatible


def normalize_metric(val: Union[str, MetricIdentifier]) -> Optional[MetricIdentifier]:
    """Normalize a string or enum into a canonical MetricIdentifier."""
    if isinstance(val, MetricIdentifier):
        return val
    cleaned = str(val).strip().lower().replace(" ", "_").replace("-", "_")
    return METRIC_NORMALIZATION_MAP.get(cleaned)
