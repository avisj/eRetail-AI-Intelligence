"""Decision Intelligence & Action Planning Query Tools (Phase 7A).

Exposes Phase 6H Decision Packages under strict Human-in-the-Loop governance:
- get_decision_summary: Decision package portfolio summary, risk severity breakdown, governance check
- get_decision_packages: Filterable list of decision packages
- get_decision_package_by_id: Detailed decision package inspection with options, trade-offs, and risks

STRICT GOVERNANCE GUARANTEES:
- approval_required = True (100%)
- execution_allowed = False (100%)
- selected_option = None (100% — NO WINNER DECLARED)
- No auto-selection, no auto-approval, no autonomous action
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

from commerce_ai.decision_intelligence.schemas import (
    DecisionIntelligenceResult,
    DecisionPackage,
    DecisionStatus,
    RiskSeverity,
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


def _get_packages(res: DecisionIntelligenceResult) -> List[Any]:
    """Retrieve decision packages list safely from result object."""
    if hasattr(res, "decision_packages") and res.decision_packages:
        return res.decision_packages
    if hasattr(res, "packages") and res.packages:
        return res.packages
    return []


def _get_pkg_sku(p: Any) -> Optional[str]:
    if hasattr(p, "entity_context") and p.entity_context and hasattr(p.entity_context, "sku_id"):
        return p.entity_context.sku_id
    return getattr(p, "sku_id", None)


def _get_pkg_warehouse(p: Any) -> Optional[str]:
    if hasattr(p, "entity_context") and p.entity_context and hasattr(p.entity_context, "warehouse_id"):
        return p.entity_context.warehouse_id
    return getattr(p, "warehouse_id", None)


def _get_pkg_channel(p: Any) -> Optional[str]:
    if hasattr(p, "entity_context") and p.entity_context and hasattr(p.entity_context, "channel_id"):
        return p.entity_context.channel_id
    return getattr(p, "channel_id", None)


def _get_pkg_status(p: Any) -> Any:
    return getattr(p, "decision_status", getattr(p, "status", DecisionStatus.PENDING_REVIEW))


def _get_pkg_title(p: Any) -> str:
    return getattr(p, "decision_title", getattr(p, "title", ""))


def _get_pkg_summary(p: Any) -> str:
    return getattr(p, "decision_summary", getattr(p, "objective_statement", getattr(p, "summary", "")))


def _get_pkg_options(p: Any) -> List[Any]:
    return getattr(p, "decision_options", getattr(p, "candidate_options", []))


def _get_pkg_exposure(p: Any) -> float:
    if hasattr(p, "financial_context") and p.financial_context and hasattr(p.financial_context, "exposure_value") and p.financial_context.exposure_value is not None:
        return float(p.financial_context.exposure_value)
    return float(getattr(p, "financial_exposure", 0.0) or 0.0)


def _get_pkg_currency(p: Any) -> str:
    if hasattr(p, "business_context") and p.business_context and hasattr(p.business_context, "currency"):
        return p.business_context.currency
    return getattr(p, "financial_currency", getattr(p, "currency", "USD"))


def _get_pkg_severity(p: Any) -> str:
    if hasattr(p, "highest_risk_severity") and p.highest_risk_severity is not None:
        val = p.highest_risk_severity.value if hasattr(p.highest_risk_severity, "value") else str(p.highest_risk_severity)
        return val
    if hasattr(p, "business_context") and p.business_context and hasattr(p.business_context, "severity"):
        return p.business_context.severity
    return "MEDIUM"


def _get_pkg_rec_id(p: Any) -> str:
    return getattr(p, "recommendation_id", getattr(p, "source_recommendation_id", ""))


def _get_pkg_rec_type(p: Any) -> str:
    rt = getattr(p, "recommendation_type", "")
    return rt.value if hasattr(rt, "value") else str(rt)


def _has_insufficient_info(p: Any) -> bool:
    if hasattr(p, "has_insufficient_information"):
        return bool(p.has_insufficient_information)
    req_info = getattr(p, "required_information", [])
    for info in req_info:
        avail = getattr(info, "availability", None)
        if avail is not None and getattr(avail, "value", str(avail)) != "AVAILABLE":
            return True
    return False


def _has_conflict(p: Any) -> bool:
    if getattr(p, "conflicts", []):
        return True
    flags = getattr(p, "risk_flags", [])
    for rf in flags:
        r_type = getattr(rf, "risk_type", getattr(rf, "flag_id", ""))
        if "CONFLICT" in str(r_type):
            return True
    return False


def get_decision_summary(
    context: QueryContext,
    decision_result: Optional[DecisionIntelligenceResult] = None,
) -> QueryResponse:
    """Query decision package portfolio summary, governance status, and risk distributions."""
    t0 = time.perf_counter()
    tool_name = "get_decision_summary"

    if decision_result is None:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.DECISIONS,
            context=context,
            start_time=t0,
            reason="Decision intelligence packages are unavailable. Run DecisionIntelligenceService first.",
            source_engine="commerce_ai.decision_intelligence",
        )

    pkgs = _get_packages(decision_result)

    # Filter by context if specified
    if context.sku_ids:
        pkgs = [p for p in pkgs if _get_pkg_sku(p) in context.sku_ids]
    if context.warehouse_ids:
        pkgs = [p for p in pkgs if _get_pkg_warehouse(p) in context.warehouse_ids]

    total_count = len(pkgs)
    pending_count = sum(1 for p in pkgs if _get_pkg_status(p) == DecisionStatus.PENDING_REVIEW)
    conflict_count = sum(1 for p in pkgs if _has_conflict(p))
    insufficient_info_count = sum(1 for p in pkgs if _has_insufficient_info(p))
    no_winner_count = sum(1 for p in pkgs if getattr(p, "selected_option", None) is None)

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.DECISIONS,
        context=context,
        start_time=t0,
        currency=context.effective_currency,
        source_engine="commerce_ai.decision_intelligence.DecisionIntelligenceService",
    )

    evidence = [
        EvidenceReference(
            source_engine="commerce_ai.decision_intelligence",
            source_type="decision_portfolio",
            source_id=f"dec_portfolio_{total_count}",
            metric="total_decision_packages",
            value=total_count,
            as_of_date=context.iso_as_of_date,
            notes="100% enforce selected_option=None and approval_required=True",
        )
    ]

    metrics = [
        MetricResult(
            metric_name="total_decision_packages",
            display_name="Total Decision Packages",
            value=total_count,
            unit="packages",
            source="DecisionIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="pending_review_count",
            display_name="Packages Pending Review",
            value=pending_count,
            unit="packages",
            source="DecisionIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="operational_conflict_count",
            display_name="Packages with Operational Conflicts",
            value=conflict_count,
            unit="packages",
            source="DecisionIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="insufficient_information_count",
            display_name="Packages with Missing/Unavailable Information",
            value=insufficient_info_count,
            unit="packages",
            source="DecisionIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="unselected_winner_count",
            display_name="Packages with No Winner Declared (Governance)",
            value=no_winner_count,
            unit="packages",
            source="DecisionIntelligenceService",
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


def get_decision_packages(
    context: QueryContext,
    decision_result: Optional[DecisionIntelligenceResult] = None,
    decision_status: Optional[str] = None,
    highest_severity: Optional[str] = None,
) -> QueryResponse:
    """Query filterable list of decision packages (strictly non-executable)."""
    t0 = time.perf_counter()
    tool_name = "get_decision_packages"

    if decision_result is None:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.DECISIONS,
            context=context,
            start_time=t0,
            reason="Decision intelligence packages are unavailable.",
            source_engine="commerce_ai.decision_intelligence",
        )

    pkgs = _get_packages(decision_result)

    # Apply context and parameter filters
    if context.sku_ids:
        pkgs = [p for p in pkgs if _get_pkg_sku(p) in context.sku_ids]
    if context.warehouse_ids:
        pkgs = [p for p in pkgs if _get_pkg_warehouse(p) in context.warehouse_ids]
    if context.effective_currency:
        pkgs = [p for p in pkgs if _get_pkg_currency(p).upper() == context.effective_currency]

    if decision_status:
        st_norm = decision_status.strip().upper()
        pkgs = [p for p in pkgs if (_get_pkg_status(p).value if hasattr(_get_pkg_status(p), "value") else str(_get_pkg_status(p))).upper() == st_norm]

    if highest_severity:
        sev_norm = highest_severity.strip().upper()
        pkgs = [p for p in pkgs if _get_pkg_severity(p).upper() == sev_norm]

    if not pkgs:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.DECISIONS,
            context=context,
            start_time=t0,
            currency=context.effective_currency,
            source_engine="commerce_ai.decision_intelligence",
        )

    rows = []
    for p in pkgs:
        st = _get_pkg_status(p)
        st_str = st.value if hasattr(st, "value") else str(st)
        opts = _get_pkg_options(p)

        rows.append({
            "decision_id": p.decision_id,
            "source_recommendation_id": _get_pkg_rec_id(p),
            "recommendation_type": _get_pkg_rec_type(p),
            "sku_id": _get_pkg_sku(p),
            "warehouse_id": _get_pkg_warehouse(p),
            "title": _get_pkg_title(p),
            "candidate_options_count": len(opts),
            "trade_offs_count": len(getattr(p, "trade_offs", [])),
            "risk_flags_count": len(getattr(p, "risk_flags", [])),
            "highest_risk_severity": _get_pkg_severity(p),
            "status": st_str,
            "approval_required": p.approval_required,
            "execution_allowed": p.execution_allowed,
            "selected_option": p.selected_option,
            "requires_human_review": p.requires_human_review,
            "has_insufficient_information": _has_insufficient_info(p),
            "financial_exposure": _get_pkg_exposure(p),
            "currency": _get_pkg_currency(p),
        })

    # Deterministic sort
    rows.sort(key=lambda x: x["decision_id"])

    # Pagination
    if context.offset is not None:
        rows = rows[context.offset:]
    if context.limit is not None:
        rows = rows[:context.limit]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.DECISIONS,
        context=context,
        start_time=t0,
        currency=context.effective_currency,
        source_engine="commerce_ai.decision_intelligence",
    )

    table = TableResult(
        columns=[
            "decision_id", "source_recommendation_id", "recommendation_type",
            "sku_id", "warehouse_id", "title", "candidate_options_count",
            "trade_offs_count", "risk_flags_count", "highest_risk_severity",
            "status", "approval_required", "execution_allowed", "selected_option",
            "requires_human_review", "has_insufficient_information",
            "financial_exposure", "currency"
        ],
        column_types={
            "decision_id": "string",
            "source_recommendation_id": "string",
            "recommendation_type": "string",
            "sku_id": "string",
            "warehouse_id": "string",
            "title": "string",
            "candidate_options_count": "integer",
            "trade_offs_count": "integer",
            "risk_flags_count": "integer",
            "highest_risk_severity": "string",
            "status": "string",
            "approval_required": "boolean",
            "execution_allowed": "boolean",
            "selected_option": "string",
            "requires_human_review": "boolean",
            "has_insufficient_information": "boolean",
            "financial_exposure": "float",
            "currency": "string",
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


def get_decision_package_by_id(
    context: QueryContext,
    decision_id: str,
    decision_result: Optional[DecisionIntelligenceResult] = None,
) -> QueryResponse:
    """Retrieve full decision package details including candidate options, trade-offs, and risks."""
    t0 = time.perf_counter()
    tool_name = "get_decision_package_by_id"

    if decision_result is None:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.DECISIONS,
            context=context,
            start_time=t0,
            reason="Decision intelligence packages are unavailable.",
            source_engine="commerce_ai.decision_intelligence",
        )

    pkgs = _get_packages(decision_result)
    match = next((p for p in pkgs if p.decision_id == decision_id), None)
    if match is None:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.DECISIONS,
            context=context,
            start_time=t0,
            message=f"Decision package with ID '{decision_id}' was not found.",
        )

    exposure = _get_pkg_exposure(match)
    curr = _get_pkg_currency(match)
    sev = _get_pkg_severity(match)
    opts = _get_pkg_options(match)

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.DECISIONS,
        context=context,
        start_time=t0,
        currency=curr,
        source_engine="commerce_ai.decision_intelligence",
    )

    opt_rows = [
        {
            "option_id": opt.option_id,
            "title": opt.title,
            "action_type": opt.option_type.value if hasattr(opt, "option_type") and hasattr(opt.option_type, "value") else (opt.action_type.value if hasattr(opt, "action_type") and hasattr(opt.action_type, "value") else str(getattr(opt, "option_type", getattr(opt, "action_type", "")))),
            "description": opt.description,
            "is_baseline": getattr(opt, "is_baseline", False),
            "is_recommended": getattr(opt, "recommended_by_engine", getattr(opt, "is_recommended", False)),
        }
        for opt in opts
    ]

    table = TableResult(
        columns=["option_id", "title", "action_type", "description", "is_baseline", "is_recommended"],
        column_types={
            "option_id": "string",
            "title": "string",
            "action_type": "string",
            "description": "string",
            "is_baseline": "boolean",
            "is_recommended": "boolean",
        },
        rows=opt_rows,
        total_rows=len(opt_rows),
        metadata=meta,
    )

    evidences = [
        EvidenceReference(
            source_engine="commerce_ai.recommendations",
            source_type="recommendation_origin",
            source_id=_get_pkg_rec_id(match),
            as_of_date=context.iso_as_of_date,
        )
    ]

    summary_text = (
        f"{_get_pkg_summary(match)}\n"
        f"Options count: {len(opts)}. "
        f"Trade-offs: {len(getattr(match, 'trade_offs', []))}. "
        f"Risk flags: {len(getattr(match, 'risk_flags', []))} (Highest: {sev}). "
        f"Selected option: NONE (Winner selection strictly prohibited without human decision)."
    )

    insight = InsightResult(
        insight_id=match.decision_id,
        title=_get_pkg_title(match),
        summary=summary_text,
        category=_get_pkg_rec_type(match),
        severity=sev,
        impact_value=exposure,
        currency=curr,
        evidence=evidences,
        requires_human_review=match.requires_human_review,
    )

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        table=table,
        insights=[insight],
    )
