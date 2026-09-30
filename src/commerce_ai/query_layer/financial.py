"""Financial Intelligence & Profitability Query Tools (Phase 7A).

Exposes existing Phase 6 financial intelligence without recalculating formulas independently:
- get_revenue_summary: Gross revenue, discounts, net revenue, source variance (Phase 6A)
- get_margin_summary: Gross margin, gross margin %, COGS, profitability status (Phase 6A)
- get_unit_economics: Variable costs (shipping, handling, payment, returns) & contribution margin (Phase 6B)
- get_margin_drivers: Commercial margin waterfall stages and erosion drivers (Phase 6C)
- get_operational_economics: Operational cost detail and cost completeness report (Phase 6D)
- get_profitability_attribution: Dimensional profitability and contribution gap profiles (Phase 6C)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from commerce_ai.financial.service import (
    FinancialIntelligenceService,
    OperationalEconomicsService,
    UnitEconomicsService,
)
from commerce_ai.financial.attribution import ProfitabilityAttributionService
from commerce_ai.financial.margin_drivers import build_margin_waterfall
from commerce_ai.financial.schemas import (
    FinancialIntelligenceConfig,
    OperationalEconomicsConfig,
    ProfitabilityAttributionConfig,
)
from commerce_ai.query_layer.base import (
    build_currency_inconsistency_response,
    build_empty_response,
    build_metadata,
    build_unavailable_response,
)
from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.filters import apply_context_filters
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


def get_revenue_summary(
    context: QueryContext,
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Query reconciled revenue figures using Phase 6A Financial Intelligence."""
    t0 = time.perf_counter()
    tool_name = "get_revenue_summary"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
        products_df=products_df,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            currency=curr,
        )

    # Invoke Phase 6A service
    cfg = FinancialIntelligenceConfig(as_of_date=context.iso_as_of_date)
    service = FinancialIntelligenceService(config=cfg)
    fin_res = service.analyze(
        sales=filtered,
        products=products_df,
        as_of_date=context.iso_as_of_date,
    )
    summary = fin_res.portfolio_summary
    currency = summary.currency or curr or "USD"

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.FINANCIAL,
        context=context,
        start_time=t0,
        currency=currency,
        source_engine="commerce_ai.financial.FinancialIntelligenceService",
    )

    rec_status = summary.revenue_reconciliation_status.value if hasattr(summary.revenue_reconciliation_status, "value") else str(summary.revenue_reconciliation_status)

    evidence = [
        EvidenceReference(
            source_engine="commerce_ai.financial",
            source_type="portfolio_summary",
            source_id=f"fin_summary_{len(filtered)}",
            metric="net_revenue",
            value=summary.total_net_revenue,
            currency=currency,
            as_of_date=context.iso_as_of_date,
            notes=f"Reconciliation status: {rec_status}. Financial completeness: {summary.financial_completeness_rate:.1f}%",
        )
    ]

    gross_rev = summary.total_gross_revenue if summary.total_gross_revenue is not None else 0.0
    disc = summary.total_discount if summary.total_discount is not None else 0.0
    net_rev = summary.total_net_revenue if summary.total_net_revenue is not None else 0.0

    metrics = [
        MetricResult(
            metric_name="gross_revenue",
            display_name="Gross Revenue",
            value=gross_rev,
            unit=currency,
            currency=currency,
            source="FinancialIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="total_discount",
            display_name="Total Discounts",
            value=disc,
            unit=currency,
            currency=currency,
            source="FinancialIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="net_revenue",
            display_name="Net Revenue",
            value=net_rev,
            unit=currency,
            currency=currency,
            source="FinancialIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="reconciliation_status",
            display_name="Reconciliation Status",
            value=rec_status,
            unit="status",
            currency=None,
            source="FinancialIntelligenceService",
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


def get_margin_summary(
    context: QueryContext,
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Query gross margin, COGS, and margin percentage using Phase 6A."""
    t0 = time.perf_counter()
    tool_name = "get_margin_summary"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
        products_df=products_df,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            currency=curr,
        )

    cfg = FinancialIntelligenceConfig(as_of_date=context.iso_as_of_date)
    service = FinancialIntelligenceService(config=cfg)
    fin_res = service.analyze(
        sales=filtered,
        products=products_df,
        as_of_date=context.iso_as_of_date,
    )
    summary = fin_res.portfolio_summary
    currency = summary.currency or curr or "USD"

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.FINANCIAL,
        context=context,
        start_time=t0,
        currency=currency,
        source_engine="commerce_ai.financial.FinancialIntelligenceService",
    )

    product_cost = summary.total_estimated_cogs if summary.total_estimated_cogs is not None else 0.0
    gross_margin = summary.total_gross_margin if summary.total_gross_margin is not None else 0.0
    gross_margin_pct = summary.gross_margin_percentage if summary.gross_margin_percentage is not None else 0.0
    neg_lines = summary.count_of_negative_margin_order_lines

    evidence = [
        EvidenceReference(
            source_engine="commerce_ai.financial",
            source_type="portfolio_summary",
            source_id=f"fin_summary_{len(filtered)}",
            metric="gross_margin",
            value=gross_margin,
            currency=currency,
            as_of_date=context.iso_as_of_date,
            notes=f"Completeness rate: {summary.financial_completeness_rate:.1f}%",
        )
    ]

    metrics = [
        MetricResult(
            metric_name="product_cost",
            display_name="Product Cost (COGS)",
            value=product_cost,
            unit=currency,
            currency=currency,
            source="FinancialIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="gross_margin",
            display_name="Gross Margin",
            value=gross_margin,
            unit=currency,
            currency=currency,
            source="FinancialIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="gross_margin_pct",
            display_name="Gross Margin %",
            value=gross_margin_pct,
            unit="%",
            currency=None,
            source="FinancialIntelligenceService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="negative_margin_transactions",
            display_name="Negative Margin Transactions",
            value=neg_lines,
            unit="records",
            currency=None,
            source="FinancialIntelligenceService",
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


def get_unit_economics(
    context: QueryContext,
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Query variable cost breakdowns and true contribution margins using Phase 6B."""
    t0 = time.perf_counter()
    tool_name = "get_unit_economics"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
        products_df=products_df,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            currency=curr,
        )

    ue_service = UnitEconomicsService()
    ue_res = ue_service.calculate_unit_economics(
        sales=filtered,
        products=products_df,
        as_of_date=context.iso_as_of_date,
    )
    summary = ue_res.portfolio_summary
    currency = summary.currency or curr or "USD"

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.FINANCIAL,
        context=context,
        start_time=t0,
        currency=currency,
        source_engine="commerce_ai.financial.UnitEconomicsService",
    )

    cm_status = summary.contribution_margin_status.value if hasattr(summary.contribution_margin_status, "value") else str(summary.contribution_margin_status)

    evidence = [
        EvidenceReference(
            source_engine="commerce_ai.financial.unit_economics",
            source_type="unit_economics_portfolio",
            source_id=f"ue_summary_{len(filtered)}",
            metric="known_contribution_margin",
            value=summary.total_known_contribution_margin,
            currency=currency,
            as_of_date=context.iso_as_of_date,
            notes=f"Contribution margin status: {cm_status}. Cost completeness: {summary.cost_completeness_pct:.1f}%",
        )
    ]

    metrics = [
        MetricResult(
            metric_name="shipping_cost",
            display_name="Fulfillment Shipping Cost",
            value=summary.total_shipping_cost or 0.0,
            unit=currency,
            currency=currency,
            source="UnitEconomicsService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="payment_processing_fee",
            display_name="Payment Processing Fees",
            value=summary.total_payment_processing_cost or 0.0,
            unit=currency,
            currency=currency,
            source="UnitEconomicsService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="packaging_cost",
            display_name="Packaging Cost",
            value=summary.total_packaging_cost or 0.0,
            unit=currency,
            currency=currency,
            source="UnitEconomicsService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="handling_cost",
            display_name="Handling & Pick/Pack Cost",
            value=summary.total_warehouse_handling_cost or 0.0,
            unit=currency,
            currency=currency,
            source="UnitEconomicsService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="return_fees",
            display_name="Return & Reverse Logistics Fees",
            value=summary.total_return_processing_cost or 0.0,
            unit=currency,
            currency=currency,
            source="UnitEconomicsService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="total_variable_costs",
            display_name="Total Variable Costs",
            value=summary.total_known_variable_cost or 0.0,
            unit=currency,
            currency=currency,
            source="UnitEconomicsService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="known_contribution_margin",
            display_name="Known Contribution Margin",
            value=summary.total_known_contribution_margin or 0.0,
            unit=currency,
            currency=currency,
            source="UnitEconomicsService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="known_contribution_margin_pct",
            display_name="Known Contribution Margin %",
            value=summary.known_contribution_margin_pct or 0.0,
            unit="%",
            currency=None,
            source="UnitEconomicsService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="full_contribution_margin_status",
            display_name="Full Contribution Margin Status",
            value=cm_status,
            unit="status",
            currency=None,
            source="UnitEconomicsService",
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


def get_margin_drivers(
    context: QueryContext,
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Query portfolio margin waterfall stages and margin driver factors using Phase 6C."""
    t0 = time.perf_counter()
    tool_name = "get_margin_drivers"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
        products_df=products_df,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            currency=curr,
        )

    ue_service = UnitEconomicsService()
    ue_res = ue_service.calculate_unit_economics(
        sales=filtered,
        products=products_df,
        as_of_date=context.iso_as_of_date,
    )
    ps = ue_res.portfolio_summary
    currency = ps.currency or curr or "USD"

    waterfall = build_margin_waterfall(
        gross_revenue=ps.total_gross_revenue or 0.0,
        discount=ps.total_discount or 0.0,
        net_revenue=ps.total_net_revenue or 0.0,
        product_cost=ps.total_product_cost or 0.0,
        gross_margin=ps.total_gross_margin or 0.0,
        known_variable_costs=ps.total_known_variable_cost or 0.0,
        known_contribution_margin=ps.total_known_contribution_margin,
        final_contribution_margin=ps.total_contribution_margin,
        currency=currency,
    )

    items = [
        BreakdownItem(
            dimension_name="waterfall_stage",
            dimension_value=stage.stage_name,
            metric_value=stage.amount or 0.0,
            metric_name="stage_amount",
            percentage_of_total=round((stage.percentage_of_gross_revenue or 0.0) * 100.0, 2),
            currency=currency,
            additional_metrics={
                "stage_type": stage.stage_type,
                "stage_order": stage.stage_order,
            },
        )
        for stage in waterfall.stages
    ]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.FINANCIAL,
        context=context,
        start_time=t0,
        currency=currency,
        source_engine="commerce_ai.financial.margin_drivers",
    )

    breakdown = BreakdownResult(
        metric_name="waterfall_amount",
        display_name="Commercial Margin Waterfall",
        dimension_name="waterfall_stage",
        items=items,
        total_value=ps.total_known_contribution_margin or 0.0,
        unit=currency,
        currency=currency,
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


def get_operational_economics(
    context: QueryContext,
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Query operational cost allocations and cost completeness audits using Phase 6D."""
    t0 = time.perf_counter()
    tool_name = "get_operational_economics"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
        products_df=products_df,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            currency=curr,
        )

    op_service = OperationalEconomicsService()
    summary = op_service.calculate_portfolio_economics(
        sales=filtered,
        products=products_df,
        as_of_date=context.iso_as_of_date,
    )
    report = summary.cost_completeness_report
    currency = summary.currency or curr or "USD"

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.FINANCIAL,
        context=context,
        start_time=t0,
        currency=currency,
        source_engine="commerce_ai.financial.OperationalEconomicsService",
    )

    compl_pct = getattr(report, "completeness_pct", getattr(report, "overall_completeness_pct", 100.0))
    compl_status = getattr(report, "completeness_status", getattr(report, "status", "COMPLETE"))
    avail_comps = getattr(report, "available_components", getattr(report, "component_reports", []))

    evidence = [
        EvidenceReference(
            source_engine="commerce_ai.financial.operational_economics",
            source_type="cost_completeness_report",
            source_id=f"op_audit_{len(filtered)}",
            metric="completeness_percentage",
            value=compl_pct,
            as_of_date=context.iso_as_of_date,
            notes=f"Completeness status: {compl_status}. Measured components: {len(avail_comps)}",
        )
    ]

    cost_per_order = round(summary.total_known_variable_cost / summary.total_orders, 2) if summary.total_orders > 0 else 0.0

    metrics = [
        MetricResult(
            metric_name="total_operational_cost",
            display_name="Total Operational Cost",
            value=summary.total_known_variable_cost,
            unit=currency,
            currency=currency,
            source="OperationalEconomicsService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="operational_cost_per_order",
            display_name="Operational Cost per Order",
            value=cost_per_order,
            unit=currency,
            currency=currency,
            source="OperationalEconomicsService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="cost_completeness_score",
            display_name="Cost Completeness Score %",
            value=compl_pct,
            unit="%",
            currency=None,
            source="OperationalEconomicsService",
            as_of_date=context.iso_as_of_date,
            evidence=evidence,
        ),
        MetricResult(
            metric_name="cost_completeness_status",
            display_name="Cost Completeness Status",
            value=compl_status.value if hasattr(compl_status, "value") else str(compl_status),
            unit="status",
            currency=None,
            source="OperationalEconomicsService",
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


def get_profitability_attribution(
    context: QueryContext,
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Query multi-dimensional profitability and margin contribution gaps using Phase 6C."""
    t0 = time.perf_counter()
    tool_name = "get_profitability_attribution"

    filtered, curr, is_isolated, err_msg = apply_context_filters(
        sales_df,
        context=context,
        date_col="date",
        currency_col="currency",
        products_df=products_df,
    )
    if not is_isolated:
        return build_currency_inconsistency_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            error_message=err_msg or "Currency isolation failure",
        )

    if filtered.empty:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.FINANCIAL,
            context=context,
            start_time=t0,
            currency=curr,
        )

    attr_service = ProfitabilityAttributionService()
    attr_res = attr_service.compute_profitability_attribution(
        sales_df=filtered,
        products_df=products_df,
        as_of_date=context.iso_as_of_date,
    )

    rows = []
    for p in attr_res.sku_profiles:
        driver = "NEUTRAL"
        if p.driver_classifications:
            d0 = p.driver_classifications[0]
            driver = d0.value if hasattr(d0, "value") else str(d0)

        rows.append({
            "sku_id": p.sku_id,
            "product_name": p.product_name or p.sku_id,
            "net_revenue": p.net_revenue or 0.0,
            "gross_margin": p.gross_margin or 0.0,
            "gross_margin_pct": p.gross_margin_pct or 0.0,
            "known_contribution_margin": p.known_contribution_margin or 0.0,
            "contribution_margin_pct": p.gross_margin_pct or 0.0,
            "contribution_gap": p.contribution_gap or 0.0,
            "primary_erosion_factor": driver,
            "concentration_tier": "CORE",
            "currency": p.currency or curr or "USD",
        })

    # Sort deterministically by sku_id
    rows.sort(key=lambda x: x["sku_id"])

    # Pagination if requested
    if context.offset is not None:
        rows = rows[context.offset:]
    if context.limit is not None:
        rows = rows[:context.limit]

    currency = curr or "USD"
    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.FINANCIAL,
        context=context,
        start_time=t0,
        currency=currency,
        source_engine="commerce_ai.financial.ProfitabilityAttributionService",
    )

    table = TableResult(
        columns=[
            "sku_id", "product_name", "net_revenue", "gross_margin", "gross_margin_pct",
            "known_contribution_margin", "contribution_margin_pct", "contribution_gap",
            "primary_erosion_factor", "concentration_tier", "currency"
        ],
        column_types={
            "sku_id": "string",
            "product_name": "string",
            "net_revenue": "float",
            "gross_margin": "float",
            "gross_margin_pct": "float",
            "known_contribution_margin": "float",
            "contribution_margin_pct": "float",
            "contribution_gap": "float",
            "primary_erosion_factor": "string",
            "concentration_tier": "string",
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
