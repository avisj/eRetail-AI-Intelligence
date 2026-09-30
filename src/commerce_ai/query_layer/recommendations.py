"""Business Recommendation Query Tools (Phase 7A).

Exposes Phase 6G Business Recommendations under strict human-in-the-loop governance:
- get_recommendation_summary: Total recommendations, count by type, count by priority, human review status
- get_recommendations: Filterable tabular recommendation list
- get_recommendation_by_id: Detailed recommendation inspection with evidence references

STRICT GOVERNANCE GUARANTEES:
- approval_required = True (100%)
- execution_allowed = False (100%)
- requires_human_review = True (100%)
- status = DRAFT
- Does NOT trigger automated purchase orders, transfers, or pricing changes
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendation,
    BusinessRecommendationResult,
    RecommendationPriority,
    RecommendationStatus,
    RecommendationType,
)
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
    InsightResult,
    MetricResult,
    QueryDomain,
    QueryResponse,
    TableResult,
)


def get_recommendation_summary(
    context: QueryContext,
    recommendation_result: Optional[BusinessRecommendationResult] = None,
) -> QueryResponse:
    """Query aggregate recommendation counts, priority breakdown, and governance assertions."""
    t0 = time.perf_counter()
    tool_name = "get_recommendation_summary"

    if recommendation_result is None:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.RECOMMENDATIONS,
            context=context,
            start_time=t0,
            reason="Business recommendations are unavailable. Run BusinessRecommendationEngine first.",
            source_engine="commerce_ai.recommendations.business_recommendations",
        )

    res = recommendation_result
    recs = res.recommendations

    # Filter by context if specified
    if context.sku_ids:
        recs = [r for r in recs if r.sku_id in context.sku_ids]
    if context.warehouse_ids:
        recs = [r for r in recs if r.warehouse_id in context.warehouse_ids]
    if context.channel_ids:
        recs = [r for r in recs if r.channel_id in context.channel_ids]

    total_count = len(recs)
    human_review_count = sum(1 for r in recs if r.requires_human_review)
    critical_count = sum(1 for r in recs if r.priority == RecommendationPriority.CRITICAL)
    high_count = sum(1 for r in recs if r.priority == RecommendationPriority.HIGH)
    medium_count = sum(1 for r in recs if r.priority == RecommendationPriority.MEDIUM)

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.RECOMMENDATIONS,
        context=context,
        start_time=t0,
        currency=context.effective_currency,
        source_engine="commerce_ai.recommendations.business_recommendations",
    )

    evidence = [
        EvidenceReference(
            source_engine="commerce_ai.recommendations.business_recommendations",
            source_type="recommendation_portfolio",
            source_id=f"rec_portfolio_{total_count}",
            metric="total_recommendations",
            value=total_count,
            as_of_date=context.iso_as_of_date,
            notes=f"100% require human review. Detected conflicts: {len(res.conflicts)}",
        )
    ]

    metrics = [
        MetricResult(
            metric_name="total_recommendations",
            display_name="Total Business Recommendations",
            value=total_count,
            unit="recommendations",
            source="BusinessRecommendationEngine",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="human_review_count",
            display_name="Requiring Human Review",
            value=human_review_count,
            unit="recommendations",
            source="BusinessRecommendationEngine",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="critical_priority_count",
            display_name="Critical Priority Recommendations",
            value=critical_count,
            unit="recommendations",
            source="BusinessRecommendationEngine",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="high_priority_count",
            display_name="High Priority Recommendations",
            value=high_count,
            unit="recommendations",
            source="BusinessRecommendationEngine",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="conflict_count",
            display_name="Operational Conflicts Detected",
            value=len(res.conflicts),
            unit="conflicts",
            source="BusinessRecommendationEngine",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
    ]

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        metrics=metrics,
    )


def get_recommendations(
    context: QueryContext,
    recommendation_result: Optional[BusinessRecommendationResult] = None,
    recommendation_type: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
) -> QueryResponse:
    """Query filterable list of recommendations (strictly non-executable DRAFTs)."""
    t0 = time.perf_counter()
    tool_name = "get_recommendations"

    if recommendation_result is None:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.RECOMMENDATIONS,
            context=context,
            start_time=t0,
            reason="Business recommendations are unavailable.",
            source_engine="commerce_ai.recommendations.business_recommendations",
        )

    recs = recommendation_result.recommendations

    # Apply context and parameter filters
    if context.sku_ids:
        recs = [r for r in recs if r.sku_id in context.sku_ids]
    if context.warehouse_ids:
        recs = [r for r in recs if r.warehouse_id in context.warehouse_ids]
    if context.channel_ids:
        recs = [r for r in recs if r.channel_id in context.channel_ids]
    if context.effective_currency:
        recs = [r for r in recs if (r.currency or "").upper() == context.effective_currency]

    if recommendation_type:
        rt_norm = recommendation_type.strip().upper()
        recs = [r for r in recs if (r.recommendation_type.value if hasattr(r.recommendation_type, "value") else str(r.recommendation_type)).upper() == rt_norm]

    if priority:
        pri_norm = priority.strip().upper()
        recs = [r for r in recs if (r.priority.value if hasattr(r.priority, "value") else str(r.priority)).upper() == pri_norm]

    if status:
        st_norm = status.strip().upper()
        recs = [r for r in recs if (r.status.value if hasattr(r.status, "value") else str(r.status)).upper() == st_norm]

    if not recs:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.RECOMMENDATIONS,
            context=context,
            start_time=t0,
            currency=context.effective_currency,
            source_engine="commerce_ai.recommendations.business_recommendations",
        )

    rows = []
    for r in recs:
        exposure = 0.0
        if r.financial_context and getattr(r.financial_context, "exposure_value", None) is not None:
            exposure = float(r.financial_context.exposure_value)
        elif getattr(r, "recommended_value", None) is not None:
            exposure = float(r.recommended_value)

        conf_str = r.confidence.value if hasattr(r.confidence, "value") else str(r.confidence)
        rec_type_str = r.recommendation_type.value if hasattr(r.recommendation_type, "value") else str(r.recommendation_type)
        pri_str = r.priority.value if hasattr(r.priority, "value") else str(r.priority)
        stat_str = r.status.value if hasattr(r.status, "value") else str(r.status)

        rows.append({
            "recommendation_id": r.recommendation_id,
            "recommendation_type": rec_type_str,
            "priority": pri_str,
            "sku_id": r.sku_id,
            "warehouse_id": r.warehouse_id,
            "channel_id": r.channel_id,
            "title": r.title,
            "status": stat_str,
            "approval_required": r.approval_required,
            "execution_allowed": r.execution_allowed,
            "requires_human_review": r.requires_human_review,
            "confidence_provenance": conf_str,
            "financial_exposure": exposure,
            "currency": r.currency,
            "rule_id": r.rule_id,
            "rule_version": r.rule_version,
        })

    # Deterministic sorting by recommendation_id
    rows.sort(key=lambda x: x["recommendation_id"])

    # Pagination
    if context.offset is not None:
        rows = rows[context.offset:]
    if context.limit is not None:
        rows = rows[:context.limit]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.RECOMMENDATIONS,
        context=context,
        start_time=t0,
        currency=context.effective_currency,
        source_engine="commerce_ai.recommendations.business_recommendations",
    )

    table = TableResult(
        columns=[
            "recommendation_id", "recommendation_type", "priority", "sku_id",
            "warehouse_id", "channel_id", "title", "status", "approval_required",
            "execution_allowed", "requires_human_review", "confidence_provenance",
            "financial_exposure", "currency", "rule_id", "rule_version"
        ],
        column_types={
            "recommendation_id": "string",
            "recommendation_type": "string",
            "priority": "string",
            "sku_id": "string",
            "warehouse_id": "string",
            "channel_id": "string",
            "title": "string",
            "status": "string",
            "approval_required": "boolean",
            "execution_allowed": "boolean",
            "requires_human_review": "boolean",
            "confidence_provenance": "string",
            "financial_exposure": "float",
            "currency": "string",
            "rule_id": "string",
            "rule_version": "string",
        },
        rows=rows,
        total_rows=len(rows),
        metadata=meta,
    )

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        table=table,
    )


def get_recommendation_by_id(
    context: QueryContext,
    recommendation_id: str,
    recommendation_result: Optional[BusinessRecommendationResult] = None,
) -> QueryResponse:
    """Retrieve full audit detail and evidence lineage for a specific recommendation."""
    t0 = time.perf_counter()
    tool_name = "get_recommendation_by_id"

    if recommendation_result is None:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.RECOMMENDATIONS,
            context=context,
            start_time=t0,
            reason="Business recommendations are unavailable.",
            source_engine="commerce_ai.recommendations.business_recommendations",
        )

    match = next((r for r in recommendation_result.recommendations if r.recommendation_id == recommendation_id), None)
    if match is None:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.RECOMMENDATIONS,
            context=context,
            start_time=t0,
            message=f"Recommendation with ID '{recommendation_id}' was not found.",
        )

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.RECOMMENDATIONS,
        context=context,
        start_time=t0,
        currency=match.currency,
        source_engine="commerce_ai.recommendations.business_recommendations",
    )

    exposure = 0.0
    if match.financial_context and hasattr(match.financial_context, "exposure_value") and match.financial_context.exposure_value is not None:
        exposure = float(match.financial_context.exposure_value)
    elif match.recommended_value is not None:
        exposure = float(match.recommended_value)

    evidences: List[EvidenceReference] = []
    for sig_id in getattr(match, "supporting_signal_ids", []):
        evidences.append(
            EvidenceReference(
                source_engine="commerce_ai.recommendations",
                source_type="signal_evidence",
                source_id=sig_id,
                metric="cross_domain_signal",
                as_of_date=context.iso_as_of_date,
            )
        )
    for imp_id in getattr(match, "supporting_impact_ids", []):
        evidences.append(
            EvidenceReference(
                source_engine="commerce_ai.business_impact",
                source_type="business_impact_evidence",
                source_id=imp_id,
                metric="financial_exposure",
                value=exposure,
                currency=match.currency,
                as_of_date=context.iso_as_of_date,
            )
        )
    for ev in getattr(match, "evidence", []):
        evidences.append(
            EvidenceReference(
                source_engine=match.source_engine,
                source_type=getattr(ev, "evidence_type", "evidence"),
                source_id=getattr(ev, "evidence_id", match.recommendation_id),
                metric=getattr(ev, "metric_name", "recommendation_evidence"),
                value=getattr(ev, "metric_value", None),
                notes=getattr(ev, "description", None),
                as_of_date=context.iso_as_of_date,
            )
        )

    insight = InsightResult(
        insight_id=match.recommendation_id,
        title=match.title,
        summary=f"{match.reason} | Proposed Action: {match.action}",
        category=match.recommendation_type.value if hasattr(match.recommendation_type, "value") else str(match.recommendation_type),
        severity=match.priority.value if hasattr(match.priority, "value") else str(match.priority),
        impact_value=exposure,
        currency=match.currency,
        evidence=evidences,
        requires_human_review=match.requires_human_review,
    )

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        insights=[insight],
    )
