"""Canonical Business Question Templates (Phase 7B).

Provides deterministic templates for representative business questions across all platform domains.
Templates serve as canonical test references, documentation benchmarks, and contract blueprints.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    BusinessGrain,
    MetricIdentifier,
    OutputGrain,
    QueryIntent,
    RequestedOutput,
    TimeGranularity,
)


class QuestionTemplate(BaseModel):
    """Specification of a canonical business question and its structured contract parameters."""
    model_config = ConfigDict(frozen=True)

    template_id: str = Field(description="Unique deterministic template identifier")
    question_text: str = Field(description="Natural-language representative question")
    category: str = Field(description="Functional category name")
    intent: QueryIntent = Field(description="Target canonical QueryIntent")
    domain: BusinessDomain = Field(description="Target canonical BusinessDomain")
    domains: List[BusinessDomain] = Field(default_factory=list, description="Multi-domain list if applicable")
    metrics: List[MetricIdentifier] = Field(default_factory=list, description="Standard metrics for this question")
    dimensions: List[BusinessDimension] = Field(default_factory=list, description="Standard dimensions for this question")
    requested_grain: BusinessGrain = Field(default=BusinessGrain.PORTFOLIO, description="Default response grain")
    time_granularity: TimeGranularity = Field(default=TimeGranularity.NONE, description="Default temporal frequency")
    requested_output: RequestedOutput = Field(default=RequestedOutput.KPI, description="Default response format")
    explanation_context: Optional[str] = Field(default=None, description="Diagnostic context for why-style questions")
    expected_tools: List[str] = Field(default_factory=list, description="Deterministic Phase 7A tools expected")


# Canonical repository of 20 standardized business question templates
CANONICAL_TEMPLATES: List[QuestionTemplate] = [
    # Sales
    QuestionTemplate(
        template_id="TPL_SALES_PERFORMANCE",
        question_text="How are sales performing?",
        category="Sales",
        intent=QueryIntent.SALES_PERFORMANCE,
        domain=BusinessDomain.SALES,
        metrics=[MetricIdentifier.NET_REVENUE, MetricIdentifier.ORDERS, MetricIdentifier.UNITS, MetricIdentifier.AOV],
        requested_output=RequestedOutput.KPI,
        expected_tools=["get_sales_summary"],
    ),
    QuestionTemplate(
        template_id="TPL_SALES_TREND",
        question_text="Show revenue trend.",
        category="Sales",
        intent=QueryIntent.SALES_PERFORMANCE,
        domain=BusinessDomain.SALES,
        metrics=[MetricIdentifier.NET_REVENUE],
        dimensions=[BusinessDimension.DATE],
        requested_grain=BusinessGrain.PORTFOLIO,
        time_granularity=TimeGranularity.DAY,
        requested_output=RequestedOutput.TIME_SERIES,
        expected_tools=["get_sales_trend"],
    ),
    QuestionTemplate(
        template_id="TPL_SALES_BY_WAREHOUSE",
        question_text="Show sales by warehouse.",
        category="Sales",
        intent=QueryIntent.SALES_PERFORMANCE,
        domain=BusinessDomain.SALES,
        metrics=[MetricIdentifier.NET_REVENUE],
        dimensions=[BusinessDimension.WAREHOUSE],
        requested_grain=OutputGrain.WAREHOUSE,
        requested_output=RequestedOutput.BREAKDOWN,
        expected_tools=["get_sales_by_warehouse"],
    ),
    # Financial
    QuestionTemplate(
        template_id="TPL_MARGIN_STATUS",
        question_text="What is our margin?",
        category="Financial",
        intent=QueryIntent.MARGIN_ANALYSIS,
        domain=BusinessDomain.FINANCIAL,
        metrics=[MetricIdentifier.GROSS_MARGIN, MetricIdentifier.GROSS_MARGIN_PERCENT],
        requested_output=RequestedOutput.KPI,
        expected_tools=["get_margin_summary"],
    ),
    QuestionTemplate(
        template_id="TPL_MARGIN_DRIVERS",
        question_text="What is affecting margin?",
        category="Financial",
        intent=QueryIntent.MARGIN_ANALYSIS,
        domain=BusinessDomain.FINANCIAL,
        metrics=[MetricIdentifier.GROSS_MARGIN, MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN],
        requested_output=RequestedOutput.BREAKDOWN,
        expected_tools=["get_margin_drivers", "get_margin_summary"],
    ),
    QuestionTemplate(
        template_id="TPL_UNIT_ECONOMICS",
        question_text="Show unit economics.",
        category="Financial",
        intent=QueryIntent.UNIT_ECONOMICS_ANALYSIS,
        domain=BusinessDomain.FINANCIAL,
        metrics=[MetricIdentifier.NET_REVENUE, MetricIdentifier.GROSS_MARGIN, MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN],
        requested_output=RequestedOutput.DETAIL,
        expected_tools=["get_operational_economics", "get_unit_economics"],
    ),
    # Inventory
    QuestionTemplate(
        template_id="TPL_INVENTORY_POSITION",
        question_text="What is our inventory position?",
        category="Inventory",
        intent=QueryIntent.INVENTORY_STATUS,
        domain=BusinessDomain.INVENTORY,
        metrics=[MetricIdentifier.INVENTORY_UNITS, MetricIdentifier.INVENTORY_VALUE],
        dimensions=[BusinessDimension.SKU],
        requested_grain=OutputGrain.SKU,
        requested_output=RequestedOutput.TABLE,
        expected_tools=["get_inventory_position"],
    ),
    QuestionTemplate(
        template_id="TPL_STOCKOUT_RISK",
        question_text="Where is stockout risk?",
        category="Inventory",
        intent=QueryIntent.STOCKOUT_ANALYSIS,
        domain=BusinessDomain.INVENTORY,
        metrics=[MetricIdentifier.STOCKOUT_EXPOSURE, MetricIdentifier.DAYS_OF_SUPPLY],
        dimensions=[BusinessDimension.SKU, BusinessDimension.WAREHOUSE],
        requested_grain=OutputGrain.SKU_WAREHOUSE,
        requested_output=RequestedOutput.TABLE,
        expected_tools=["get_stockout_risk"],
    ),
    QuestionTemplate(
        template_id="TPL_SLOW_MOVING",
        question_text="Show slow-moving inventory.",
        category="Inventory",
        intent=QueryIntent.SLOW_MOVING_INVENTORY,
        domain=BusinessDomain.INVENTORY,
        metrics=[MetricIdentifier.SLOW_MOVING_EXPOSURE, MetricIdentifier.INVENTORY_VALUE],
        dimensions=[BusinessDimension.SKU],
        requested_grain=OutputGrain.SKU,
        requested_output=RequestedOutput.TABLE,
        expected_tools=["get_slow_moving_inventory"],
    ),
    # Demand
    QuestionTemplate(
        template_id="TPL_DEMAND_TREND",
        question_text="How is demand trending?",
        category="Demand",
        intent=QueryIntent.DEMAND_TREND,
        domain=BusinessDomain.DEMAND,
        metrics=[MetricIdentifier.DAILY_DEMAND, MetricIdentifier.DEMAND_TREND],
        dimensions=[BusinessDimension.DATE],
        requested_grain=BusinessGrain.PORTFOLIO,
        time_granularity=TimeGranularity.DAY,
        requested_output=RequestedOutput.TIME_SERIES,
        expected_tools=["get_demand_trend"],
    ),
    QuestionTemplate(
        template_id="TPL_ABC_XYZ",
        question_text="Show ABC XYZ distribution.",
        category="Demand",
        intent=QueryIntent.ABC_XYZ_ANALYSIS,
        domain=BusinessDomain.DEMAND,
        metrics=[MetricIdentifier.ABC_CLASS, MetricIdentifier.XYZ_CLASS],
        requested_output=RequestedOutput.BREAKDOWN,
        expected_tools=["get_abc_xyz_distribution"],
    ),
    # Forecast
    QuestionTemplate(
        template_id="TPL_FORECAST",
        question_text="Show the forecast.",
        category="Forecasting",
        intent=QueryIntent.FORECAST_ANALYSIS,
        domain=BusinessDomain.FORECASTING,
        metrics=[MetricIdentifier.FORECAST_VALUE],
        requested_output=RequestedOutput.TIME_SERIES,
        expected_tools=["get_forecast", "get_forecast_summary"],
    ),
    QuestionTemplate(
        template_id="TPL_FORECAST_ACCURACY",
        question_text="How accurate is the forecast?",
        category="Forecasting",
        intent=QueryIntent.FORECAST_ACCURACY,
        domain=BusinessDomain.FORECASTING,
        metrics=[MetricIdentifier.FORECAST_ACCURACY],
        requested_output=RequestedOutput.KPI,
        expected_tools=["get_forecast_accuracy"],
    ),
    QuestionTemplate(
        template_id="TPL_FORECAST_BIAS",
        question_text="Is the forecast biased?",
        category="Forecasting",
        intent=QueryIntent.FORECAST_BIAS,
        domain=BusinessDomain.FORECASTING,
        metrics=[MetricIdentifier.FORECAST_BIAS],
        requested_output=RequestedOutput.KPI,
        expected_tools=["get_forecast_bias"],
    ),
    # Returns
    QuestionTemplate(
        template_id="TPL_RETURN_RATE",
        question_text="What is our return rate?",
        category="Returns",
        intent=QueryIntent.RETURN_ANALYSIS,
        domain=BusinessDomain.RETURNS,
        metrics=[MetricIdentifier.RETURN_RATE, MetricIdentifier.RETURN_COUNT],
        requested_output=RequestedOutput.KPI,
        expected_tools=["get_return_rate", "get_return_summary"],
    ),
    QuestionTemplate(
        template_id="TPL_RETURN_ANOMALIES",
        question_text="Where are return anomalies?",
        category="Returns",
        intent=QueryIntent.RETURN_ANOMALY_ANALYSIS,
        domain=BusinessDomain.RETURNS,
        metrics=[MetricIdentifier.RETURN_ANOMALY_COUNT],
        requested_output=RequestedOutput.TABLE,
        expected_tools=["get_return_anomalies"],
    ),
    QuestionTemplate(
        template_id="TPL_RETURN_REASONS",
        question_text="What are the main return reasons?",
        category="Returns",
        intent=QueryIntent.RETURN_REASON_ANALYSIS,
        domain=BusinessDomain.RETURNS,
        metrics=[MetricIdentifier.RETURN_COUNT],
        requested_output=RequestedOutput.BREAKDOWN,
        expected_tools=["get_return_reason_breakdown"],
    ),
    # Business Impact
    QuestionTemplate(
        template_id="TPL_BUSINESS_IMPACT",
        question_text="What is the business impact?",
        category="Impact",
        intent=QueryIntent.BUSINESS_IMPACT_ANALYSIS,
        domain=BusinessDomain.BUSINESS_IMPACT,
        metrics=[MetricIdentifier.GROSS_SIGNAL_EXPOSURE, MetricIdentifier.DEDUPLICATED_EXPOSURE],
        requested_output=RequestedOutput.KPI,
        expected_tools=["get_business_impact_summary"],
    ),
    # Governance & Reviews
    QuestionTemplate(
        template_id="TPL_RECOMMENDATIONS_PENDING",
        question_text="What recommendations are pending review?",
        category="Governance",
        intent=QueryIntent.RECOMMENDATION_REVIEW,
        domain=BusinessDomain.RECOMMENDATIONS,
        metrics=[MetricIdentifier.RECOMMENDATION_COUNT],
        requested_output=RequestedOutput.TABLE,
        expected_tools=["get_recommendation_summary", "get_recommendations"],
    ),
    QuestionTemplate(
        template_id="TPL_DECISIONS_AWAITING",
        question_text="What decisions are awaiting review?",
        category="Governance",
        intent=QueryIntent.DECISION_REVIEW,
        domain=BusinessDomain.DECISIONS,
        metrics=[MetricIdentifier.DECISION_PACKAGE_COUNT, MetricIdentifier.PENDING_REVIEW_COUNT],
        requested_output=RequestedOutput.TABLE,
        expected_tools=["get_decision_packages", "get_decision_summary"],
    ),
    # WHY-style investigations
    QuestionTemplate(
        template_id="TPL_WHY_MARGIN_DOWN",
        question_text="Why is margin down?",
        category="Investigation",
        intent=QueryIntent.MARGIN_ANALYSIS,
        domain=BusinessDomain.FINANCIAL,
        metrics=[MetricIdentifier.GROSS_MARGIN],
        explanation_context="Why is margin down investigation",
        expected_tools=[
            "get_margin_drivers",
            "get_margin_summary",
            "get_sales_by_channel",
            "get_sales_by_warehouse",
        ],
    ),
    QuestionTemplate(
        template_id="TPL_WHY_RETURNS_INCREASING",
        question_text="Why are returns increasing?",
        category="Investigation",
        intent=QueryIntent.RETURN_ANALYSIS,
        domain=BusinessDomain.RETURNS,
        metrics=[MetricIdentifier.RETURN_RATE],
        explanation_context="Why are returns increasing investigation",
        expected_tools=[
            "get_return_anomalies",
            "get_return_reason_breakdown",
            "get_return_summary",
            "get_return_trend",
        ],
    ),
    QuestionTemplate(
        template_id="TPL_WHY_INVENTORY_RISK_HIGH",
        question_text="Why is inventory risk high?",
        category="Investigation",
        intent=QueryIntent.INVENTORY_RISK,
        domain=BusinessDomain.INVENTORY,
        metrics=[MetricIdentifier.STOCKOUT_EXPOSURE, MetricIdentifier.SLOW_MOVING_EXPOSURE],
        explanation_context="Why is inventory risk high investigation",
        expected_tools=[
            "get_inventory_risk",
            "get_inventory_summary",
            "get_slow_moving_inventory",
            "get_stockout_risk",
        ],
    ),
    # Multi-Domain
    QuestionTemplate(
        template_id="TPL_MULTI_DOMAIN_PERFORMANCE",
        question_text="How are revenue, margin and inventory performing?",
        category="Cross-Domain",
        intent=QueryIntent.MULTI_DOMAIN_ANALYSIS,
        domain=BusinessDomain.CROSS_DOMAIN,
        domains=[BusinessDomain.FINANCIAL, BusinessDomain.INVENTORY],
        metrics=[MetricIdentifier.NET_REVENUE, MetricIdentifier.GROSS_MARGIN, MetricIdentifier.INVENTORY_VALUE],
        expected_tools=[
            "get_inventory_summary",
            "get_margin_summary",
            "get_revenue_summary",
        ],
    ),
]


def get_canonical_templates() -> List[QuestionTemplate]:
    """Retrieve all canonical business question templates."""
    return list(CANONICAL_TEMPLATES)


def find_template_by_id(template_id: str) -> Optional[QuestionTemplate]:
    """Find template by its unique identifier."""
    for tpl in CANONICAL_TEMPLATES:
        if tpl.template_id == template_id:
            return tpl
    return None


def match_template(query_str: str) -> Optional[QuestionTemplate]:
    """Match a natural language question against canonical templates."""
    cleaned = query_str.strip().lower()
    for tpl in CANONICAL_TEMPLATES:
        if tpl.question_text.lower() == cleaned:
            return tpl
    return None
