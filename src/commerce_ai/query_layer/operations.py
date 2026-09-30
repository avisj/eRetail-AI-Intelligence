"""Operations & Replenishment Review Query Tools (Phase 7A).

Provides READ-ONLY review access to Phase 4 operational recommendations:
- get_replenishment_reviews: Read-only view of replenishment candidates
- get_purchase_order_reviews: Read-only view of draft PO proposals
- get_warehouse_rebalancing_reviews: Read-only view of inter-warehouse stock transfer candidates

STRICT GOVERNANCE GUARANTEES:
- Pure informational review access only
- NO automated purchase order creation or submission
- NO autonomous inventory transfers or stock movements
- NO pricing or replenishment execution
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from commerce_ai.recommendations.replenishment import ReplenishmentSolver
from commerce_ai.recommendations.purchase_orders import PurchaseOrderProposalService
from commerce_ai.recommendations.warehouse_rebalancing import WarehouseRebalancingService
from commerce_ai.recommendations.schemas import (
    PurchaseOrderProposalResult,
    ReplenishmentResult,
    WarehouseRebalancingResult,
)
from commerce_ai.query_layer.base import (
    build_empty_response,
    build_metadata,
    build_unavailable_response,
)
from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.schemas import (
    CalculationStatus,
    EvidenceReference,
    MetricResult,
    QueryDomain,
    QueryResponse,
    TableResult,
)


def get_replenishment_reviews(
    context: QueryContext,
    replenishment_result: Optional[ReplenishmentResult] = None,
    inventory_result: Optional[Any] = None,
) -> QueryResponse:
    """Query read-only replenishment recommendations (Phase 4B-1).
    
    Does NOT execute replenishment orders or mutate inventory state.
    """
    t0 = time.perf_counter()
    tool_name = "get_replenishment_reviews"

    res = replenishment_result
    if res is None and inventory_result is not None:
        solver = ReplenishmentSolver()
        res = solver.solve(inventory_result=inventory_result)

    if res is None or not res.recommendations:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.OPERATIONS,
            context=context,
            start_time=t0,
            reason="Replenishment recommendations are unavailable. Ingest inventory and run replenishment solver first.",
            source_engine="commerce_ai.recommendations.replenishment",
        )

    recs = res.recommendations
    if context.sku_ids:
        recs = [r for r in recs if r.sku_id in context.sku_ids]
    if context.warehouse_ids:
        recs = [r for r in recs if r.warehouse_id in context.warehouse_ids]

    if not recs:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.OPERATIONS,
            context=context,
            start_time=t0,
            source_engine="commerce_ai.recommendations.replenishment",
        )

    rows = [
        {
            "recommendation_id": r.recommendation_id,
            "sku_id": r.sku_id,
            "warehouse_id": r.warehouse_id,
            "recommended_order_quantity": r.recommended_order_quantity,
            "urgency": r.urgency,
            "reorder_point": round(r.reorder_point, 1),
            "safety_stock": round(r.safety_stock, 1),
            "net_position": r.net_position,
            "status": r.status,
            "approval_required": r.approval_required,
            "rationale": r.rationale,
        }
        for r in recs
    ]

    # Pagination
    if context.offset is not None:
        rows = rows[context.offset:]
    if context.limit is not None:
        rows = rows[:context.limit]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.OPERATIONS,
        context=context,
        start_time=t0,
        source_engine="commerce_ai.recommendations.replenishment",
    )

    table = TableResult(
        columns=[
            "recommendation_id", "sku_id", "warehouse_id", "recommended_order_quantity",
            "urgency", "reorder_point", "safety_stock", "net_position", "status",
            "approval_required", "rationale"
        ],
        column_types={
            "recommendation_id": "string",
            "sku_id": "string",
            "warehouse_id": "string",
            "recommended_order_quantity": "integer",
            "urgency": "string",
            "reorder_point": "float",
            "safety_stock": "float",
            "net_position": "integer",
            "status": "string",
            "approval_required": "boolean",
            "rationale": "string",
        },
        rows=rows,
        total_rows=len(rows),
        metadata=meta,
    )

    kpi = MetricResult(
        metric_name="total_replenishment_recommendations",
        display_name="Replenishment Review Candidates",
        value=len(recs),
        unit="recommendations",
        source="ReplenishmentSolver",
        status="INFORMATIONAL_ONLY",
    )

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        metrics=[kpi],
        table=table,
    )


def get_purchase_order_reviews(
    context: QueryContext,
    po_result: Optional[PurchaseOrderProposalResult] = None,
    replenishment_result: Optional[ReplenishmentResult] = None,
    suppliers_df: Optional[pd.DataFrame] = None,
    products_df: Optional[pd.DataFrame] = None,
) -> QueryResponse:
    """Query read-only draft Purchase Order Proposals (Phase 4B-2).
    
    Does NOT submit purchase orders or trigger procurement transactions.
    """
    t0 = time.perf_counter()
    tool_name = "get_purchase_order_reviews"

    res = po_result
    if res is None and replenishment_result is not None:
        po_service = PurchaseOrderProposalService()
        res = po_service.generate(
            replenishment_input=replenishment_result,
            suppliers=suppliers_df,
            products=products_df,
        )

    if res is None or not res.proposals:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.OPERATIONS,
            context=context,
            start_time=t0,
            reason="Purchase order proposals are unavailable. Replenishment recommendations and proposals must be computed first.",
            source_engine="commerce_ai.recommendations.purchase_orders",
        )

    props = res.proposals
    if context.warehouse_ids:
        props = [p for p in props if p.warehouse_id in context.warehouse_ids]
    if context.supplier_ids:
        props = [p for p in props if p.supplier_id in context.supplier_ids]
    if context.effective_currency:
        props = [p for p in props if (p.currency or "").upper() == context.effective_currency]

    if not props:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.OPERATIONS,
            context=context,
            start_time=t0,
            currency=context.effective_currency,
            source_engine="commerce_ai.recommendations.purchase_orders",
        )

    rows = [
        {
            "proposal_id": p.proposal_id,
            "supplier_id": p.supplier_id,
            "warehouse_id": p.warehouse_id,
            "total_quantity": p.total_quantity,
            "total_value": p.total_value,
            "currency": p.currency,
            "urgency": p.urgency,
            "status": p.status,
            "approval_required": p.approval_required,
            "cost_data_status": p.cost_data_status,
            "expected_delivery_date": p.expected_delivery_date,
            "rationale": p.rationale,
        }
        for p in props
    ]

    # Pagination
    if context.offset is not None:
        rows = rows[context.offset:]
    if context.limit is not None:
        rows = rows[:context.limit]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.OPERATIONS,
        context=context,
        start_time=t0,
        currency=context.effective_currency,
        source_engine="commerce_ai.recommendations.purchase_orders",
    )

    table = TableResult(
        columns=[
            "proposal_id", "supplier_id", "warehouse_id", "total_quantity",
            "total_value", "currency", "urgency", "status", "approval_required",
            "cost_data_status", "expected_delivery_date", "rationale"
        ],
        column_types={
            "proposal_id": "string",
            "supplier_id": "string",
            "warehouse_id": "string",
            "total_quantity": "integer",
            "total_value": "float",
            "currency": "string",
            "urgency": "string",
            "status": "string",
            "approval_required": "boolean",
            "cost_data_status": "string",
            "expected_delivery_date": "string",
            "rationale": "string",
        },
        rows=rows,
        total_rows=len(rows),
        metadata=meta,
    )

    kpi = MetricResult(
        metric_name="total_po_proposals",
        display_name="Draft Purchase Order Proposals",
        value=len(props),
        unit="proposals",
        source="PurchaseOrderProposalService",
        status="INFORMATIONAL_ONLY",
    )

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        metrics=[kpi],
        table=table,
    )


def get_warehouse_rebalancing_reviews(
    context: QueryContext,
    rebalancing_result: Optional[WarehouseRebalancingResult] = None,
    inventory_result: Optional[Any] = None,
) -> QueryResponse:
    """Query read-only inter-warehouse rebalancing transfer candidates (Phase 4C).
    
    Does NOT execute stock movements or initiate warehouse transfer workflows.
    """
    t0 = time.perf_counter()
    tool_name = "get_warehouse_rebalancing_reviews"

    res = rebalancing_result
    if res is None and inventory_result is not None:
        service = WarehouseRebalancingService()
        res = service.rebalance(inventory_analysis=inventory_result)

    if res is None or not res.transfers:
        return build_unavailable_response(
            tool_name=tool_name,
            domain=QueryDomain.OPERATIONS,
            context=context,
            start_time=t0,
            reason="Warehouse rebalancing recommendations are unavailable. Ingest inventory network and run rebalancing solver first.",
            source_engine="commerce_ai.recommendations.warehouse_rebalancing",
        )

    trans = res.transfers
    if context.sku_ids:
        trans = [t for t in trans if t.sku_id in context.sku_ids]
    if context.warehouse_ids:
        trans = [t for t in trans if t.source_warehouse_id in context.warehouse_ids or t.destination_warehouse_id in context.warehouse_ids]

    if not trans:
        return build_empty_response(
            tool_name=tool_name,
            domain=QueryDomain.OPERATIONS,
            context=context,
            start_time=t0,
            source_engine="commerce_ai.recommendations.warehouse_rebalancing",
        )

    rows = [
        {
            "transfer_id": t.transfer_id,
            "sku_id": t.sku_id,
            "source_warehouse_id": t.source_warehouse_id,
            "destination_warehouse_id": t.destination_warehouse_id,
            "quantity": t.quantity,
            "priority": t.priority,
            "status": t.status,
            "approval_required": t.approval_required,
            "cost_data_status": t.cost_data_status,
            "rationale": t.rationale,
        }
        for t in trans
    ]

    # Pagination
    if context.offset is not None:
        rows = rows[context.offset:]
    if context.limit is not None:
        rows = rows[:context.limit]

    meta = build_metadata(
        tool_name=tool_name,
        domain=QueryDomain.OPERATIONS,
        context=context,
        start_time=t0,
        source_engine="commerce_ai.recommendations.warehouse_rebalancing",
    )

    table = TableResult(
        columns=[
            "transfer_id", "sku_id", "source_warehouse_id", "destination_warehouse_id",
            "quantity", "priority", "status", "approval_required", "cost_data_status", "rationale"
        ],
        column_types={
            "transfer_id": "string",
            "sku_id": "string",
            "source_warehouse_id": "string",
            "destination_warehouse_id": "string",
            "quantity": "integer",
            "priority": "string",
            "status": "string",
            "approval_required": "boolean",
            "cost_data_status": "string",
            "rationale": "string",
        },
        rows=rows,
        total_rows=len(rows),
        metadata=meta,
    )

    kpi = MetricResult(
        metric_name="total_rebalancing_transfers",
        display_name="Warehouse Transfer Candidates",
        value=len(trans),
        unit="transfers",
        source="WarehouseRebalancingService",
        status="INFORMATIONAL_ONLY",
    )

    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.SUCCESS,
        metadata=meta,
        metrics=[kpi],
        table=table,
    )
