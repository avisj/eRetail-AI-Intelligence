"""Controlled Taxonomies and Enumerations for Business Query Contracts (Phase 7B).

Defines immutable, controlled vocabulary for:
- QueryIntent: What business intent is expressed
- BusinessDomain: Functional platform domain matching Phase 7A
- MetricIdentifier: Controlled business metrics mapping to Phase 7A calculations
- BusinessDimension: Valid slice-and-dice dimensions
- TimeRangePreset: Controlled temporal ranges
- ComparisonType: Baseline period comparisons
- OutputGrain: Granularity of query responses
- RequestedOutput: Format of requested business intelligence
- ProvenanceLevel: Acceptable data pedigree
"""

from __future__ import annotations

from enum import Enum


class QueryIntent(str, Enum):
    """Controlled taxonomy of business query intents.

    Arbitrary free-form intents outside this taxonomy are rejected by contract validation.
    """
    SALES_PERFORMANCE = "SALES_PERFORMANCE"
    REVENUE_ANALYSIS = "REVENUE_ANALYSIS"
    MARGIN_ANALYSIS = "MARGIN_ANALYSIS"
    PROFITABILITY_ANALYSIS = "PROFITABILITY_ANALYSIS"
    UNIT_ECONOMICS_ANALYSIS = "UNIT_ECONOMICS_ANALYSIS"
    INVENTORY_STATUS = "INVENTORY_STATUS"
    INVENTORY_RISK = "INVENTORY_RISK"
    STOCKOUT_ANALYSIS = "STOCKOUT_ANALYSIS"
    SLOW_MOVING_INVENTORY = "SLOW_MOVING_INVENTORY"
    HIGH_VALUE_INVENTORY = "HIGH_VALUE_INVENTORY"
    DEMAND_ANALYSIS = "DEMAND_ANALYSIS"
    DEMAND_TREND = "DEMAND_TREND"
    ABC_XYZ_ANALYSIS = "ABC_XYZ_ANALYSIS"
    FORECAST_ANALYSIS = "FORECAST_ANALYSIS"
    FORECAST_ACCURACY = "FORECAST_ACCURACY"
    FORECAST_BIAS = "FORECAST_BIAS"
    RETURN_ANALYSIS = "RETURN_ANALYSIS"
    RETURN_ANOMALY_ANALYSIS = "RETURN_ANOMALY_ANALYSIS"
    RETURN_RISK = "RETURN_RISK"
    RETURN_REASON_ANALYSIS = "RETURN_REASON_ANALYSIS"
    REPLENISHMENT_REVIEW = "REPLENISHMENT_REVIEW"
    PURCHASE_ORDER_REVIEW = "PURCHASE_ORDER_REVIEW"
    WAREHOUSE_REBALANCING_REVIEW = "WAREHOUSE_REBALANCING_REVIEW"
    BUSINESS_IMPACT_ANALYSIS = "BUSINESS_IMPACT_ANALYSIS"
    RECOMMENDATION_REVIEW = "RECOMMENDATION_REVIEW"
    DECISION_REVIEW = "DECISION_REVIEW"
    DATA_QUALITY_ANALYSIS = "DATA_QUALITY_ANALYSIS"
    MULTI_DOMAIN_ANALYSIS = "MULTI_DOMAIN_ANALYSIS"


class BusinessDomain(str, Enum):
    """Controlled functional domains matching Phase 7A platform topology."""
    SALES = "SALES"
    FINANCIAL = "FINANCIAL"
    INVENTORY = "INVENTORY"
    DEMAND = "DEMAND"
    FORECASTING = "FORECASTING"
    RETURNS = "RETURNS"
    OPERATIONS = "OPERATIONS"
    BUSINESS_IMPACT = "BUSINESS_IMPACT"
    RECOMMENDATIONS = "RECOMMENDATIONS"
    DECISIONS = "DECISIONS"
    DATA_QUALITY = "DATA_QUALITY"
    CROSS_DOMAIN = "CROSS_DOMAIN"


class MetricIdentifier(str, Enum):
    """Controlled catalog of business metrics mapping to Phase 7A and upstream services."""
    # Revenue & Commercial
    GROSS_REVENUE = "GROSS_REVENUE"
    NET_REVENUE = "NET_REVENUE"
    AOV = "AOV"
    UNITS = "UNITS"
    ORDERS = "ORDERS"

    # Margins & Economics
    GROSS_MARGIN = "GROSS_MARGIN"
    GROSS_MARGIN_PERCENT = "GROSS_MARGIN_PERCENT"
    KNOWN_CONTRIBUTION_MARGIN = "KNOWN_CONTRIBUTION_MARGIN"
    FINAL_CONTRIBUTION_MARGIN = "FINAL_CONTRIBUTION_MARGIN"
    COST_COMPLETENESS = "COST_COMPLETENESS"

    # Inventory
    INVENTORY_UNITS = "INVENTORY_UNITS"
    INVENTORY_VALUE = "INVENTORY_VALUE"
    STOCKOUT_EXPOSURE = "STOCKOUT_EXPOSURE"
    SLOW_MOVING_EXPOSURE = "SLOW_MOVING_EXPOSURE"
    HIGH_VALUE_EXPOSURE = "HIGH_VALUE_EXPOSURE"
    DAYS_OF_SUPPLY = "DAYS_OF_SUPPLY"

    # Demand & Classification
    DAILY_DEMAND = "DAILY_DEMAND"
    DEMAND_TREND = "DEMAND_TREND"
    VELOCITY = "VELOCITY"
    INTERMITTENCY = "INTERMITTENCY"
    ABC_CLASS = "ABC_CLASS"
    XYZ_CLASS = "XYZ_CLASS"

    # Forecasting
    FORECAST_VALUE = "FORECAST_VALUE"
    FORECAST_ACCURACY = "FORECAST_ACCURACY"
    FORECAST_BIAS = "FORECAST_BIAS"
    FORECAST_ERROR = "FORECAST_ERROR"

    # Returns
    RETURN_RATE = "RETURN_RATE"
    RETURN_COUNT = "RETURN_COUNT"
    RETURNED_UNITS = "RETURNED_UNITS"
    RETURN_REVENUE_EXPOSURE = "RETURN_REVENUE_EXPOSURE"
    RETURN_RISK = "RETURN_RISK"
    RETURN_ANOMALY_COUNT = "RETURN_ANOMALY_COUNT"

    # Impact & Exposure
    GROSS_SIGNAL_EXPOSURE = "GROSS_SIGNAL_EXPOSURE"
    DEDUPLICATED_EXPOSURE = "DEDUPLICATED_EXPOSURE"
    DEDUPLICATED_PHYSICAL_CAPITAL_EXPOSURE = "DEDUPLICATED_PHYSICAL_CAPITAL_EXPOSURE"

    # Recommendations
    RECOMMENDATION_COUNT = "RECOMMENDATION_COUNT"

    # Decisions
    DECISION_PACKAGE_COUNT = "DECISION_PACKAGE_COUNT"
    PENDING_REVIEW_COUNT = "PENDING_REVIEW_COUNT"
    CONFLICT_COUNT = "CONFLICT_COUNT"
    INSUFFICIENT_INFORMATION_COUNT = "INSUFFICIENT_INFORMATION_COUNT"


class BusinessDimension(str, Enum):
    """Controlled analytical dimensions for slicing and dicing business metrics."""
    SKU = "SKU"
    WAREHOUSE = "WAREHOUSE"
    CHANNEL = "CHANNEL"
    CATEGORY = "CATEGORY"
    BRAND = "BRAND"
    SUPPLIER = "SUPPLIER"
    VELOCITY_TIER = "VELOCITY_TIER"
    ABC_CLASS = "ABC_CLASS"
    XYZ_CLASS = "XYZ_CLASS"
    DATE = "DATE"
    WEEK = "WEEK"
    MONTH = "MONTH"
    QUARTER = "QUARTER"


class TimeRangePreset(str, Enum):
    """Standardized relative temporal range presets."""
    TODAY = "TODAY"
    LAST_7_DAYS = "LAST_7_DAYS"
    LAST_30_DAYS = "LAST_30_DAYS"
    LAST_90_DAYS = "LAST_90_DAYS"
    THIS_MONTH = "THIS_MONTH"
    PREVIOUS_MONTH = "PREVIOUS_MONTH"
    THIS_QUARTER = "THIS_QUARTER"
    PREVIOUS_QUARTER = "PREVIOUS_QUARTER"
    CUSTOM = "CUSTOM"


class ComparisonType(str, Enum):
    """Controlled comparative analysis specifications."""
    NONE = "NONE"
    PREVIOUS_PERIOD = "PREVIOUS_PERIOD"
    PREVIOUS_YEAR_PERIOD = "PREVIOUS_YEAR_PERIOD"
    CUSTOM_COMPARISON = "CUSTOM_COMPARISON"


class BusinessGrain(str, Enum):
    """Controlled analytical business grouping grains."""
    PORTFOLIO = "PORTFOLIO"
    SKU = "SKU"
    SKU_WAREHOUSE = "SKU_WAREHOUSE"
    SKU_CHANNEL = "SKU_CHANNEL"
    WAREHOUSE = "WAREHOUSE"
    CHANNEL = "CHANNEL"
    CATEGORY = "CATEGORY"
    BRAND = "BRAND"
    SUPPLIER = "SUPPLIER"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"


# Backward-compatible alias for OutputGrain
OutputGrain = BusinessGrain


class TimeGranularity(str, Enum):
    """Controlled temporal aggregation frequency, independent of business grouping grain."""
    NONE = "NONE"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    QUARTER = "QUARTER"


class RequestedOutput(str, Enum):
    """Controlled output response format for downstream consumers."""
    KPI = "KPI"
    TIME_SERIES = "TIME_SERIES"
    BREAKDOWN = "BREAKDOWN"
    TABLE = "TABLE"
    DETAIL = "DETAIL"
    INSIGHT_EVIDENCE = "INSIGHT_EVIDENCE"


class ProvenanceLevel(str, Enum):
    """Acceptable data pedigree tiers."""
    DIRECT_OBSERVED = "DIRECT_OBSERVED"
    DETERMINISTIC_DERIVED = "DETERMINISTIC_DERIVED"
    MODEL_BASED = "MODEL_BASED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ContractPriority(str, Enum):
    """Execution priority classification for query contracts."""
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    URGENT = "URGENT"


class PlanStatus(str, Enum):
    """Status classification of synthesized QueryPlan."""
    PLANNED = "PLANNED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    UNAVAILABLE = "UNAVAILABLE"

