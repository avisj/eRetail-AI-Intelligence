"""Business Impact & Opportunity Quantification Query Tools (Phase 7A).

Exposes Phase 6F quantified commercial exposures:
- get_business_impact_summary: Gross signal exposure, deduplicated exposure, physical capital exposure
- get_business_impact_by_category: Aggregate distribution across ImpactCategory
- get_business_impact_by_type: Aggregate distribution across ImpactType
- get_physical_capital_exposure: Deduplicated physical capital exposure KPI and breakdown

PRESERVES STRICT PHASE 6F AUDIT RULES:
- Preserves exact terminology: gross_signal_exposure, deduplicated_exposure, deduplicated_physical_capital_exposure
- NO entity rankings, NO 'top 10', NO leaderboards
- Only aggregate distributions or explicitly filtered results
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

from commerce_ai.business_impact.schemas import (
    BusinessImpactPortfolioSummary,
    BusinessImpactResult,
    ImpactCategory,
    ImpactType,
)
from commerce_ai.business_impact.service import BusinessImpactService
from commerce_ai.query_layer.base import (
    build_empty_response,
    build_metadata,
    build_unavailable_response,
)
from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.schemas import (
    BreakdownItem,
    BreakdownResult,
    CalculationStatus,
    EvidenceReference,
    MetricResult,
    QueryDomain,
    QueryResponse,
    TableResult,
)


def get_business_impact_summary(
    context: QueryContext,
    impact_result: Optional[BusinessImpactResult] = None,
) -> QueryResponse:
    """Query portfolio business impact exposures with deduplicated financial integrity."""
    t0 = time.perf_counter()
    tool_name = "get_business_impact_summary"

    summary = getattr(impact_result, "portfolio_summary", getattr(impact_result, "summary", None)) if impact_result else None
    if summary is None:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.IMPACT,
            context=context,
            start_time=t0,
            reason="Business impact quantification results are unavailable. Run BusinessImpactService first.",
            source_engine="commerce_ai.business_impact",
        )

    curr = summary.currency or context.effective_currency
    total_recs = getattr(summary, "impact_record_count", 0) or getattr(summary, "total_impact_records", 0)
    gross_exp = getattr(summary, "gross_signal_exposure", 0.0) or getattr(summary, "gross_exposure", 0.0)
    dedup_exp = getattr(summary, "deduplicated_exposure", None)
    if dedup_exp is None:
        dedup_exp = getattr(summary, "deduplicated_portfolio_exposure", None)

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.IMPACT,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.business_impact.BusinessImpactService",
    )

    evidence = [
        EvidenceReference(
            source_engine="commerce_ai.business_impact",
            source_type="impact_portfolio_summary",
            source_id=f"impact_summary_{total_recs}",
            metric="deduplicated_exposure",
            value=dedup_exp,
            currency=curr,
            as_of_date=context.iso_as_of_date,
            notes=f"Deduplicated across {total_recs:,d} quantified signal impact records",
        )
    ]

    metrics = [
        MetricResult(
            metric_name="total_impact_records",
            display_name="Quantified Impact Records",
            value=total_recs,
            unit="records",
            currency=None,
            source="BusinessImpactService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="gross_signal_exposure",
            display_name="Gross Signal Exposure",
            value=gross_exp,
            unit=curr or "USD",
            currency=curr,
            source="BusinessImpactService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="deduplicated_exposure",
            display_name="Deduplicated Portfolio Exposure",
            value=dedup_exp,
            unit=curr or "USD",
            currency=curr,
            source="BusinessImpactService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
    ]

    if getattr(summary, "deduplicated_physical_capital_exposure", None) is not None:
        metrics.append(
            MetricResult(
                metric_name="deduplicated_physical_capital_exposure",
                display_name="Physical Capital Exposure (Deduplicated)",
                value=summary.deduplicated_physical_capital_exposure,
                unit=curr or "USD",
                currency=curr,
                source="BusinessImpactService",
                as_of_date=context.iso_as_of_date,
                evidence=evidence,
            )
        )

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        metrics=metrics,
    )


def get_business_impact_by_category(
    context: QueryContext,
    impact_result: Optional[BusinessImpactResult] = None,
) -> QueryResponse:
    """Query aggregate exposure breakdown across ImpactCategory (no rankings)."""
    t0 = time.perf_counter()
    tool_name = "get_business_impact_by_category"

    summary = getattr(impact_result, "portfolio_summary", getattr(impact_result, "summary", None)) if impact_result else None
    if summary is None:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.IMPACT,
            context=context,
            start_time=t0,
            reason="Business impact quantification results are unavailable.",
            source_engine="commerce_ai.business_impact",
        )

    curr = summary.currency or context.effective_currency
    total_recs = getattr(summary, "impact_record_count", getattr(summary, "total_impact_records", 0))
    cat_counts = getattr(summary, "impact_category_counts", getattr(summary, "records_by_category", {}))
    cat_exposures = getattr(summary, "exposure_totals_by_category", getattr(summary, "exposure_by_category", {}))
    gross_exp = getattr(summary, "gross_signal_exposure", getattr(summary, "gross_exposure", 0.0))

    items = []
    for cat_name, count in sorted(cat_counts.items()):
        exp = cat_exposures.get(cat_name, 0.0)
        pct = (count / total_recs) * 100.0 if total_recs > 0 else 0.0
        items.append(
            BreakdownItem(
                dimension_name="impact_category",
                dimension_value=cat_name,
                metric_value=round(exp, 2),
                metric_name="exposure_amount",
                percentage_of_total=round(pct, 2),
                currency=curr,
                additional_metrics={"record_count": count},
            )
        )

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.IMPACT,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.business_impact",
    )

    breakdown = BreakdownResult(
        metric_name="exposure_by_category",
        display_name="Business Impact Exposure by Category",
        dimension_name="impact_category",
        items=items,
        total_value=gross_exp,
        unit=curr or "USD",
        currency=curr,
        metadata=meta,
    )

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        breakdown=breakdown,
    )


def get_business_impact_by_type(
    context: QueryContext,
    impact_result: Optional[BusinessImpactResult] = None,
) -> QueryResponse:
    """Query aggregate exposure breakdown across ImpactType."""
    t0 = time.perf_counter()
    tool_name = "get_business_impact_by_type"

    summary = getattr(impact_result, "portfolio_summary", getattr(impact_result, "summary", None)) if impact_result else None
    if summary is None:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.IMPACT,
            context=context,
            start_time=t0,
            reason="Business impact quantification results are unavailable.",
            source_engine="commerce_ai.business_impact",
        )

    curr = summary.currency or context.effective_currency
    total_recs = getattr(summary, "impact_record_count", getattr(summary, "total_impact_records", 0))
    type_counts = getattr(summary, "impact_type_counts", getattr(summary, "records_by_type", {}))
    type_exposures = getattr(summary, "exposure_totals_by_type", getattr(summary, "exposure_by_type", {}))
    gross_exp = getattr(summary, "gross_signal_exposure", getattr(summary, "gross_exposure", 0.0))

    items = []
    for type_name, count in sorted(type_counts.items()):
        exp = type_exposures.get(type_name, 0.0)
        pct = (count / total_recs) * 100.0 if total_recs > 0 else 0.0
        items.append(
            BreakdownItem(
                dimension_name="impact_type",
                dimension_value=type_name,
                metric_value=round(exp, 2),
                metric_name="exposure_amount",
                percentage_of_total=round(pct, 2),
                currency=curr,
                additional_metrics={"record_count": count},
            )
        )

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.IMPACT,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.business_impact",
    )

    breakdown = BreakdownResult(
        metric_name="exposure_by_type",
        display_name="Business Impact Exposure by Type",
        dimension_name="impact_type",
        items=items,
        total_value=gross_exp,
        unit=curr or "USD",
        currency=curr,
        metadata=meta,
    )

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        breakdown=breakdown,
    )


def get_physical_capital_exposure(
    context: QueryContext,
    impact_result: Optional[BusinessImpactResult] = None,
) -> QueryResponse:
    """Query deduplicated physical capital exposure KPI."""
    t0 = time.perf_counter()
    tool_name = "get_physical_capital_exposure"

    res = get_business_impact_summary(context, impact_result=impact_result)
    res.tool_name = tool_name
    return res
