"""Deterministic Natural Language Interpretation Engine (Phase 7C).

Maps normalized user business questions into structured CopilotInterpretation
specifications using:
- Phase 7B Canonical Question Templates
- Domain & Intent Keyword/Phrase Rule Matchers
- Why-Style Diagnostic Investigation Parsers
- Multi-Domain Cross-Functional Analyzers
- Entity & Dimension Extractors (SKU, Warehouse, Channel, Dates, Currency)
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from commerce_ai.copilot.enums import (
    InterpretationConfidence,
    InterpretationSource,
)
from commerce_ai.copilot.normalization import normalize_question
from commerce_ai.copilot.schemas import CopilotInterpretation, CopilotRequest
from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    BusinessGrain,
    MetricIdentifier,
    OutputGrain,
    QueryIntent,
    RequestedOutput,
    TimeGranularity,
    TimeRangePreset,
)
from commerce_ai.query_contracts.templates import CANONICAL_TEMPLATES, match_template


# Why-Style Investigation Mappings
_WHY_PATTERNS = [
    (
        re.compile(r"\b(why\s+(is\s+|did\s+)?margin\s+(down|declining|declined|decline|falling|dropped|drop|decreasing|decrease|decreased|low)|what\s+is\s+affecting\s+margin)\b", re.IGNORECASE),
        QueryIntent.MARGIN_ANALYSIS,
        BusinessDomain.FINANCIAL,
        "Why is margin down investigation",
        [MetricIdentifier.GROSS_MARGIN, MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN],
        RequestedOutput.BREAKDOWN,
        "TPL_WHY_MARGIN_DOWN",
    ),
    (
        re.compile(r"\b(why\s+(is\s+|did\s+)?inventory\s+risk\s+(high|increase|increasing|rose|rise)|why\s+high\s+inventory\s+risk|inventory\s+risk\s+drivers)\b", re.IGNORECASE),
        QueryIntent.INVENTORY_RISK,
        BusinessDomain.INVENTORY,
        "Why is inventory risk high investigation",
        [MetricIdentifier.STOCKOUT_EXPOSURE, MetricIdentifier.SLOW_MOVING_EXPOSURE],
        RequestedOutput.BREAKDOWN,
        "TPL_WHY_INVENTORY_RISK_HIGH",
    ),
    (
        re.compile(r"\b(why\s+(are\s+|did\s+)?returns\s+(increasing|increase|high|up|rising|rose)|what\s+is\s+driving\s+(the\s+increase\s+in\s+)?returns)\b", re.IGNORECASE),
        QueryIntent.RETURN_ANALYSIS,
        BusinessDomain.RETURNS,
        "Why are returns increasing investigation",
        [MetricIdentifier.RETURN_RATE, MetricIdentifier.RETURN_COUNT],
        RequestedOutput.BREAKDOWN,
        "TPL_WHY_RETURNS_INCREASING",
    ),
]


# Canonical Intent Keyword Rules
_INTENT_KEYWORD_RULES: List[Tuple[re.Pattern, QueryIntent, BusinessDomain, List[MetricIdentifier], RequestedOutput]] = [
    # Sales & Revenue
    (re.compile(r"\b(sales\s+trend|revenue\s+trend|sales\s+over\s+time|revenue\s+over\s+time)\b", re.IGNORECASE),
     QueryIntent.SALES_PERFORMANCE, BusinessDomain.SALES, [MetricIdentifier.NET_REVENUE], RequestedOutput.TIME_SERIES),
    (re.compile(r"\b(sales\s+performance|how\s+are\s+sales|overall\s+sales|total\s+sales|sales\s+summary|show\s+sales|sales\s+for|sales\s+by|sales|revenue.*(channel|warehouse|sku|category|brand))\b", re.IGNORECASE),
     QueryIntent.SALES_PERFORMANCE, BusinessDomain.SALES, [MetricIdentifier.NET_REVENUE, MetricIdentifier.ORDERS, MetricIdentifier.UNITS], RequestedOutput.KPI),
    (re.compile(r"\b(revenue\s+analysis|net\s+revenue|gross\s+revenue|revenue\s+summary|what\s+happened\s+to\s+revenue|show\s+revenue|revenue\s+for|revenue\s+from|revenue\s+by|revenue)\b", re.IGNORECASE),
     QueryIntent.REVENUE_ANALYSIS, BusinessDomain.FINANCIAL, [MetricIdentifier.NET_REVENUE, MetricIdentifier.GROSS_REVENUE], RequestedOutput.KPI),

    # Financial & Margins
    (re.compile(r"\b(margin\s+status|gross\s+margin|what\s+is\s+our\s+margin|margin\s+analysis|margin\s+percentage|show\s+margin|margin\s+for|margin\s+by|margin)\b", re.IGNORECASE),
     QueryIntent.MARGIN_ANALYSIS, BusinessDomain.FINANCIAL, [MetricIdentifier.GROSS_MARGIN, MetricIdentifier.GROSS_MARGIN_PERCENT], RequestedOutput.KPI),
    (re.compile(r"\b(margin\s+drivers|cost\s+drivers|what\s+is\s+driving\s+margin)\b", re.IGNORECASE),
     QueryIntent.MARGIN_ANALYSIS, BusinessDomain.FINANCIAL, [MetricIdentifier.GROSS_MARGIN, MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN], RequestedOutput.BREAKDOWN),
    (re.compile(r"\b(profitability|profitability\s+attribution|net\s+profit|operating\s+margin)\b", re.IGNORECASE),
     QueryIntent.PROFITABILITY_ANALYSIS, BusinessDomain.FINANCIAL, [MetricIdentifier.FINAL_CONTRIBUTION_MARGIN, MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN], RequestedOutput.DETAIL),
    (re.compile(r"\b(unit\s+economics|operational\s+economics|cost\s+per\s+unit|variable\s+cost)\b", re.IGNORECASE),
     QueryIntent.UNIT_ECONOMICS_ANALYSIS, BusinessDomain.FINANCIAL, [MetricIdentifier.NET_REVENUE, MetricIdentifier.KNOWN_CONTRIBUTION_MARGIN], RequestedOutput.DETAIL),

    # Inventory
    (re.compile(r"\b(stockout\s+risk|out\s+of\s+stock|stockout\s+analysis|where\s+is\s+stockout)\b", re.IGNORECASE),
     QueryIntent.STOCKOUT_ANALYSIS, BusinessDomain.INVENTORY, [MetricIdentifier.STOCKOUT_EXPOSURE, MetricIdentifier.DAYS_OF_SUPPLY], RequestedOutput.TABLE),
    (re.compile(r"\b(slow\s+moving|excess\s+inventory|aging\s+stock|slow\s+moving\s+inventory)\b", re.IGNORECASE),
     QueryIntent.SLOW_MOVING_INVENTORY, BusinessDomain.INVENTORY, [MetricIdentifier.SLOW_MOVING_EXPOSURE, MetricIdentifier.INVENTORY_VALUE], RequestedOutput.TABLE),
    (re.compile(r"\b(high\s+value|expensive\s+inventory|capital\s+concentration|high\s+value\s+inventory)\b", re.IGNORECASE),
     QueryIntent.HIGH_VALUE_INVENTORY, BusinessDomain.INVENTORY, [MetricIdentifier.HIGH_VALUE_EXPOSURE, MetricIdentifier.INVENTORY_VALUE], RequestedOutput.TABLE),
    (re.compile(r"\b(inventory\s+risk|risk\s+breakdown|inventory\s+exposure|risk\s+tiers)\b", re.IGNORECASE),
     QueryIntent.INVENTORY_RISK, BusinessDomain.INVENTORY, [MetricIdentifier.STOCKOUT_EXPOSURE, MetricIdentifier.SLOW_MOVING_EXPOSURE], RequestedOutput.BREAKDOWN),
    (re.compile(r"\b(inventory\s+position|stock\s+on\s+hand|stock\s+levels|what\s+is\s+our\s+inventory\s+position|inventory\s+status|show\s+inventory|inventory\s+for|inventory\s+in|inventory\s+summary|inventory)\b", re.IGNORECASE),
     QueryIntent.INVENTORY_STATUS, BusinessDomain.INVENTORY, [MetricIdentifier.INVENTORY_UNITS, MetricIdentifier.INVENTORY_VALUE], RequestedOutput.TABLE),

    # Demand
    (re.compile(r"\b(demand\s+trend|demand\s+over\s+time|how\s+is\s+demand\s+trending)\b", re.IGNORECASE),
     QueryIntent.DEMAND_TREND, BusinessDomain.DEMAND, [MetricIdentifier.DAILY_DEMAND, MetricIdentifier.DEMAND_TREND], RequestedOutput.TIME_SERIES),
    (re.compile(r"\b(abc\s+xyz|abc\s+analysis|xyz\s+analysis|abc\s*xyz\s+distribution)\b", re.IGNORECASE),
     QueryIntent.ABC_XYZ_ANALYSIS, BusinessDomain.DEMAND, [MetricIdentifier.ABC_CLASS, MetricIdentifier.XYZ_CLASS], RequestedOutput.BREAKDOWN),
    (re.compile(r"\b(demand\s+analysis|demand\s+profile|demand\s+velocity|intermittency)\b", re.IGNORECASE),
     QueryIntent.DEMAND_ANALYSIS, BusinessDomain.DEMAND, [MetricIdentifier.DAILY_DEMAND, MetricIdentifier.VELOCITY], RequestedOutput.TABLE),

    # Forecasting
    (re.compile(r"\b(forecast\s+accuracy|forecast\s+precision|forecast\s+error|how\s+accurate\s+is\s+the\s+forecast)\b", re.IGNORECASE),
     QueryIntent.FORECAST_ACCURACY, BusinessDomain.FORECASTING, [MetricIdentifier.FORECAST_ACCURACY], RequestedOutput.KPI),
    (re.compile(r"\b(forecast\s+bias|tracking\s+signal|is\s+the\s+forecast\s+biased)\b", re.IGNORECASE),
     QueryIntent.FORECAST_BIAS, BusinessDomain.FORECASTING, [MetricIdentifier.FORECAST_BIAS], RequestedOutput.KPI),
    (re.compile(r"\b(forecast|future\s+demand|show\s+the\s+forecast|forecast\s+summary)\b", re.IGNORECASE),
     QueryIntent.FORECAST_ANALYSIS, BusinessDomain.FORECASTING, [MetricIdentifier.FORECAST_VALUE], RequestedOutput.TIME_SERIES),

    # Returns
    (re.compile(r"\b(return\s+anomalies|anomalous\s+returns|return\s+spikes)\b", re.IGNORECASE),
     QueryIntent.RETURN_ANOMALY_ANALYSIS, BusinessDomain.RETURNS, [MetricIdentifier.RETURN_ANOMALY_COUNT], RequestedOutput.TABLE),
    (re.compile(r"\b(return\s+reasons|why\s+customers\s+return|reason\s+breakdown)\b", re.IGNORECASE),
     QueryIntent.RETURN_REASON_ANALYSIS, BusinessDomain.RETURNS, [MetricIdentifier.RETURN_COUNT], RequestedOutput.BREAKDOWN),
    (re.compile(r"\b(return\s+risk|high\s+return\s+skus|risky\s+products)\b", re.IGNORECASE),
     QueryIntent.RETURN_RISK, BusinessDomain.RETURNS, [MetricIdentifier.RETURN_RISK], RequestedOutput.TABLE),
    (re.compile(r"\b(return\s+rate|what\s+is\s+our\s+return\s+rate|return\s+analysis|returns\s+summary)\b", re.IGNORECASE),
     QueryIntent.RETURN_ANALYSIS, BusinessDomain.RETURNS, [MetricIdentifier.RETURN_RATE, MetricIdentifier.RETURN_COUNT], RequestedOutput.KPI),

    # Governance & Operational Reviews
    (re.compile(r"\b(replenishment(\s+\w+)*\s+review|review(\s+\w+)*\s+replenishment|what\s+should\s+(we|i)\s+(review\s+regarding\s+)?(reorder|replenish(ment)?)|reorder\s+review|replenishment\s+proposals)\b", re.IGNORECASE),
     QueryIntent.REPLENISHMENT_REVIEW, BusinessDomain.OPERATIONS, [], RequestedOutput.TABLE),
    (re.compile(r"\b(purchase\s+order\s+review|draft\s+po|planned\s+po|review\s+purchase\s+orders)\b", re.IGNORECASE),
     QueryIntent.PURCHASE_ORDER_REVIEW, BusinessDomain.OPERATIONS, [], RequestedOutput.TABLE),
    (re.compile(r"\b(warehouse\s+rebalancing\s+review|rebalancing\s+review|stock\s+transfers\s+review|transfer\s+proposals)\b", re.IGNORECASE),
     QueryIntent.WAREHOUSE_REBALANCING_REVIEW, BusinessDomain.OPERATIONS, [], RequestedOutput.TABLE),
    (re.compile(r"\b(business\s+impact|exposure\s+quantification|capital\s+exposure|deduplicated\s+exposure)\b", re.IGNORECASE),
     QueryIntent.BUSINESS_IMPACT_ANALYSIS, BusinessDomain.BUSINESS_IMPACT, [MetricIdentifier.GROSS_SIGNAL_EXPOSURE, MetricIdentifier.DEDUPLICATED_EXPOSURE], RequestedOutput.BREAKDOWN),
    (re.compile(r"\b(recommendation\s+review|pending\s+recommendations|review\s+recommendations|why\s+is\s+this\s+recommendation\s+present)\b", re.IGNORECASE),
     QueryIntent.RECOMMENDATION_REVIEW, BusinessDomain.RECOMMENDATIONS, [MetricIdentifier.RECOMMENDATION_COUNT], RequestedOutput.TABLE),
    (re.compile(r"\b(decision\s+review|pending\s+decisions|review\s+decisions|decision\s+packages)\b", re.IGNORECASE),
     QueryIntent.DECISION_REVIEW, BusinessDomain.DECISIONS, [MetricIdentifier.DECISION_PACKAGE_COUNT, MetricIdentifier.PENDING_REVIEW_COUNT], RequestedOutput.TABLE),
    (re.compile(r"\b(data\s+quality|ingestion\s+completeness|source\s+completeness|audit\s+completeness)\b", re.IGNORECASE),
     QueryIntent.DATA_QUALITY_ANALYSIS, BusinessDomain.DATA_QUALITY, [MetricIdentifier.COST_COMPLETENESS], RequestedOutput.TABLE),
]


def _extract_entities_and_filters(text: str, caller_filters: Dict[str, Any]) -> Tuple[Dict[str, Any], List[BusinessDimension], BusinessGrain, TimeGranularity]:
    """Extract entities, dimensions, grain, and temporal granularity from text and explicit filters."""
    filters: Dict[str, Any] = dict(caller_filters)
    dims: List[BusinessDimension] = []
    grain: BusinessGrain = BusinessGrain.PORTFOLIO
    time_gran: TimeGranularity = TimeGranularity.NONE

    # Extract SKU
    sku_match = re.search(r"\b(sku[_-]\w+|sku\d+)\b", text, re.IGNORECASE)
    if sku_match and "sku_id" not in filters:
        filters["sku_id"] = sku_match.group(1).upper()
        dims.append(BusinessDimension.SKU)
        grain = BusinessGrain.SKU

    # Extract Warehouse
    wh_match = re.search(r"\b(wh[_-]\w+|wh\d+|warehouse\s+([A-Za-z0-9_-]+))\b", text, re.IGNORECASE)
    if wh_match and "warehouse_id" not in filters:
        val = wh_match.group(2) if wh_match.group(2) else wh_match.group(1)
        filters["warehouse_id"] = val.upper()
        dims.append(BusinessDimension.WAREHOUSE)
        if grain == BusinessGrain.SKU:
            grain = BusinessGrain.SKU_WAREHOUSE
        else:
            grain = BusinessGrain.WAREHOUSE

    # Extract Channel
    ch_match = re.search(r"\b(channel\s+([A-Za-z0-9_-]+)|(online|wholesale|retail|b2b)\s+channel)\b", text, re.IGNORECASE)
    if ch_match and "channel_id" not in filters:
        val = ch_match.group(2) if ch_match.group(2) else ch_match.group(3)
        filters["channel_id"] = val.upper()
        dims.append(BusinessDimension.CHANNEL)
        grain = BusinessGrain.CHANNEL

    # Extract Category
    cat_match = re.search(r"\bcategory\s+(\w+)\b", text, re.IGNORECASE)
    if cat_match and "category_id" not in filters:
        filters["category_id"] = cat_match.group(1).upper()
        dims.append(BusinessDimension.CATEGORY)
        grain = BusinessGrain.CATEGORY

    # Extract Brand
    brand_match = re.search(r"\bbrand\s+(\w+)\b", text, re.IGNORECASE)
    if brand_match and "brand" not in filters:
        filters["brand"] = brand_match.group(1).upper()
        dims.append(BusinessDimension.BRAND)
        grain = BusinessGrain.BRAND

    # Extract Grain phrases
    if re.search(r"\bby\s+sku\b", text, re.IGNORECASE):
        grain = BusinessGrain.SKU
        dims.append(BusinessDimension.SKU)
    elif re.search(r"\bby\s+warehouse\b", text, re.IGNORECASE):
        grain = BusinessGrain.WAREHOUSE
        dims.append(BusinessDimension.WAREHOUSE)
    elif re.search(r"\bby\s+channel\b", text, re.IGNORECASE):
        grain = BusinessGrain.CHANNEL
        dims.append(BusinessDimension.CHANNEL)
    elif re.search(r"\bby\s+category\b", text, re.IGNORECASE):
        grain = BusinessGrain.CATEGORY
        dims.append(BusinessDimension.CATEGORY)
    elif re.search(r"\bby\s+brand\b", text, re.IGNORECASE):
        grain = BusinessGrain.BRAND
        dims.append(BusinessDimension.BRAND)

    # Extract Time Granularity
    if re.search(r"\b(daily|by\s+day|day-by-day)\b", text, re.IGNORECASE):
        time_gran = TimeGranularity.DAY
    elif re.search(r"\b(weekly|by\s+week)\b", text, re.IGNORECASE):
        time_gran = TimeGranularity.WEEK
    elif re.search(r"\b(monthly|by\s+month)\b", text, re.IGNORECASE):
        time_gran = TimeGranularity.MONTH
    elif re.search(r"\b(quarterly|by\s+quarter)\b", text, re.IGNORECASE):
        time_gran = TimeGranularity.QUARTER

    # Extract Currency
    curr_match = re.search(r"\b(USD|EUR|GBP|CAD|AUD|JPY)\b", text, re.IGNORECASE)
    if curr_match and "currency" not in filters:
        filters["currency"] = curr_match.group(1).upper()

    return filters, sorted(list(set(dims))), grain, time_gran


def _extract_time_range(text: str, req: CopilotRequest) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract date presets or bounds from text or request."""
    as_of = req.as_of_date
    tr: Dict[str, Any] = {}

    if as_of:
        tr["as_of_date"] = as_of

    # Detect relative presets
    if re.search(r"\b(this\s+month)\b", text, re.IGNORECASE):
        tr["preset"] = TimeRangePreset.THIS_MONTH
    elif re.search(r"\b(last\s+month|previous\s+month)\b", text, re.IGNORECASE):
        tr["preset"] = TimeRangePreset.PREVIOUS_MONTH
    elif re.search(r"\b(last\s+7\s+days|past\s+week)\b", text, re.IGNORECASE):
        tr["preset"] = TimeRangePreset.LAST_7_DAYS
    elif re.search(r"\b(last\s+30\s+days|past\s+month)\b", text, re.IGNORECASE):
        tr["preset"] = TimeRangePreset.LAST_30_DAYS
    elif re.search(r"\b(last\s+90\s+days|past\s+quarter)\b", text, re.IGNORECASE):
        tr["preset"] = TimeRangePreset.LAST_90_DAYS
    elif re.search(r"\b(this\s+quarter)\b", text, re.IGNORECASE):
        tr["preset"] = TimeRangePreset.THIS_QUARTER
    elif re.search(r"\b(previous\s+quarter|last\s+quarter)\b", text, re.IGNORECASE):
        tr["preset"] = TimeRangePreset.PREVIOUS_QUARTER

    # Detect explicit ISO dates (YYYY-MM-DD)
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if len(dates) == 1:
        if not as_of:
            as_of = dates[0]
            tr["as_of_date"] = as_of
    elif len(dates) >= 2:
        tr["start_date"] = dates[0]
        tr["end_date"] = dates[1]
        if not as_of:
            as_of = dates[1]
            tr["as_of_date"] = as_of

    return (tr if tr else None), as_of


def _match_canonical_template(clean_text: str):
    """Match clean question text against canonical templates ignoring punctuation."""
    norm = re.sub(r"[?!.]+$", "", clean_text).strip().lower()
    for tpl in CANONICAL_TEMPLATES:
        t_clean = re.sub(r"[?!.]+$", "", tpl.question_text).strip().lower()
        if t_clean == norm:
            return tpl
    return None


def interpret_question(request: CopilotRequest) -> Optional[CopilotInterpretation]:
    """Deterministically interpret normalized question into a structured CopilotInterpretation.

    Returns None if the question cannot be safely and unambiguously mapped to an authorized
    Phase 7B canonical intent. Silent fallback to sales performance is strictly prohibited.
    """
    norm_text = normalize_question(request.question)
    iid = "INT-" + hashlib.sha256((norm_text + (request.as_of_date or "")).encode("utf-8")).hexdigest()[:16]

    # 1. Exact canonical template match
    tpl = _match_canonical_template(norm_text)
    if tpl:
        filters, dims, grain, time_gran = _extract_entities_and_filters(norm_text, request.filters)
        tr, as_of = _extract_time_range(norm_text, request)
        if request.currency and "currency" not in filters:
            filters["currency"] = request.currency

        # Template grain & granularity take precedence if query doesn't override
        effective_grain = grain if grain != BusinessGrain.PORTFOLIO else tpl.requested_grain
        effective_time_gran = time_gran if time_gran != TimeGranularity.NONE else tpl.time_granularity
        effective_output = request.requested_output or tpl.requested_output

        return CopilotInterpretation(
            interpretation_id=iid,
            normalized_question=norm_text,
            intent=tpl.intent,
            domain=tpl.domain,
            domains=tpl.domains,
            metrics=tpl.metrics,
            dimensions=dims if dims else tpl.dimensions,
            filters=filters,
            time_range=tr,
            time_granularity=effective_time_gran,
            requested_grain=effective_grain,
            requested_output=effective_output,
            explanation_context=tpl.explanation_context,
            is_investigation=tpl.explanation_context is not None,
            is_multi_domain=len(tpl.domains) > 0 or tpl.domain == BusinessDomain.CROSS_DOMAIN,
            confidence=InterpretationConfidence.HIGH,
            source=InterpretationSource.CANONICAL_TEMPLATE,
            matched_template_id=tpl.template_id,
        )

    # 2. Why-Style Diagnostic Investigation Parsers
    for pattern, intent, domain, ctx, metrics, output, tpl_id in _WHY_PATTERNS:
        if pattern.search(norm_text):
            filters, dims, grain, time_gran = _extract_entities_and_filters(norm_text, request.filters)
            tr, as_of = _extract_time_range(norm_text, request)
            if request.currency and "currency" not in filters:
                filters["currency"] = request.currency

            return CopilotInterpretation(
                interpretation_id=iid,
                normalized_question=norm_text,
                intent=intent,
                domain=domain,
                metrics=metrics,
                dimensions=dims,
                filters=filters,
                time_range=tr,
                time_granularity=time_gran,
                requested_grain=grain,
                requested_output=request.requested_output or output,
                explanation_context=ctx,
                is_investigation=True,
                is_multi_domain=False,
                confidence=InterpretationConfidence.HIGH,
                source=InterpretationSource.DETERMINISTIC_RULE,
                matched_template_id=tpl_id,
            )

    # 3. Multi-Domain Questions (requires at least 2 distinct domain areas)
    detected_domains = []
    if re.search(r"\b(sales)\b", norm_text, re.IGNORECASE):
        detected_domains.append(BusinessDomain.SALES)
    if re.search(r"\b(revenue|margin|profitability)\b", norm_text, re.IGNORECASE):
        detected_domains.append(BusinessDomain.FINANCIAL)
    if re.search(r"\b(inventory|stock)\b", norm_text, re.IGNORECASE):
        detected_domains.append(BusinessDomain.INVENTORY)
    if re.search(r"\b(returns?)\b", norm_text, re.IGNORECASE):
        detected_domains.append(BusinessDomain.RETURNS)

    is_multi = (
        len(detected_domains) >= 3
        or (len(detected_domains) >= 2 and re.search(r"\b(connected|relationship|across|together|and\b.*\b(margin|inventory|sales|revenue|returns))\b", norm_text, re.IGNORECASE))
    )
    if is_multi:
        filters, dims, grain, time_gran = _extract_entities_and_filters(norm_text, request.filters)
        tr, as_of = _extract_time_range(norm_text, request)
        if request.currency and "currency" not in filters:
            filters["currency"] = request.currency

        return CopilotInterpretation(
            interpretation_id=iid,
            normalized_question=norm_text,
            intent=QueryIntent.MULTI_DOMAIN_ANALYSIS,
            domain=BusinessDomain.CROSS_DOMAIN,
            domains=detected_domains,
            metrics=[MetricIdentifier.NET_REVENUE, MetricIdentifier.GROSS_MARGIN, MetricIdentifier.INVENTORY_VALUE],
            dimensions=dims,
            filters=filters,
            time_range=tr,
            time_granularity=time_gran,
            requested_grain=grain,
            requested_output=request.requested_output or RequestedOutput.KPI,
            is_investigation=False,
            is_multi_domain=True,
            confidence=InterpretationConfidence.HIGH,
            source=InterpretationSource.DETERMINISTIC_RULE,
            matched_template_id="TPL_MULTI_DOMAIN_PERFORMANCE",
        )

    # 4. Keyword / Rule Matchers
    for pattern, intent, domain, metrics, output in _INTENT_KEYWORD_RULES:
        if pattern.search(norm_text):
            filters, dims, grain, time_gran = _extract_entities_and_filters(norm_text, request.filters)
            tr, as_of = _extract_time_range(norm_text, request)
            if request.currency and "currency" not in filters:
                filters["currency"] = request.currency

            return CopilotInterpretation(
                interpretation_id=iid,
                normalized_question=norm_text,
                intent=intent,
                domain=domain,
                metrics=metrics,
                dimensions=dims,
                filters=filters,
                time_range=tr,
                time_granularity=time_gran,
                requested_grain=grain,
                requested_output=request.requested_output or output,
                is_investigation=False,
                is_multi_domain=False,
                confidence=InterpretationConfidence.HIGH,
                source=InterpretationSource.DETERMINISTIC_RULE,
            )

    # 5. Unrecognized / Unsupported Query (No silent fallback to sales)
    return None
