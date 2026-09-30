"""Base abstractions and helpers for Query Layer tools (Phase 7A).

Provides:
- Deterministic query identifier generation
- Standardized QueryMetadata construction
- Structured responses for Currency Inconsistency, Empty Result, and Unavailable metrics
- BaseQueryTool interface ensuring all tools adhere to read-only contracts
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd

from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.schemas import (
    CalculationStatus,
    EvidenceReference,
    MetricResult,
    QueryDomain,
    QueryMetadata,
    QueryResponse,
)


def generate_deterministic_query_id(tool_name: str, context: QueryContext) -> str:
    """Generate deterministic query ID based on tool name and normalized filters."""
    filter_dict = context.to_filter_dict()
    filter_json = json.dumps(filter_dict, sort_keys=True)
    raw = f"{tool_name}::{filter_json}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"qry_{tool_name}_{digest}"


def build_metadata(
    tool_name: str,
    domain: Union[str, QueryDomain],
    context: QueryContext,
    start_time: float,
    currency: Optional[str] = None,
    calculation_status: CalculationStatus = CalculationStatus.SUCCESS,
    confidence_provenance: str = "DETERMINISTIC_DERIVED",
    source_engine: str = "commerce_ai.query_layer",
) -> QueryMetadata:
    """Construct standardized QueryMetadata with duration and active filters."""
    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
    dom_str = domain.value if isinstance(domain, QueryDomain) else str(domain)
    
    data_period = None
    if context.iso_start_date or context.effective_end_date:
        data_period = {
            "start_date": context.iso_start_date,
            "end_date": context.effective_end_date,
        }

    return QueryMetadata(
        query_id=generate_deterministic_query_id(tool_name, context),
        tool_name=tool_name,
        domain=dom_str,
        generated_as_of=context.iso_as_of_date or time.strftime("%Y-%m-%d"),
        currency=currency or context.effective_currency,
        filters_applied=context.to_filter_dict(),
        data_period=data_period,
        calculation_status=calculation_status,
        confidence_provenance=confidence_provenance,
        source_engine=source_engine,
        execution_time_ms=elapsed_ms,
    )


def build_currency_inconsistency_response(
    tool_name: str,
    domain: Union[str, QueryDomain],
    context: QueryContext,
    start_time: float,
    error_message: str,
    source_engine: str = "commerce_ai.query_layer",
) -> QueryResponse:
    """Standardized response when multi-currency dataset is detected without currency filter."""
    meta = build_metadata(
        tool_name=tool_name,
        domain=domain,
        context=context,
        start_time=start_time,
        currency=None,
        calculation_status=CalculationStatus.CURRENCY_INCONSISTENCY,
        source_engine=source_engine,
    )
    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.CURRENCY_INCONSISTENCY,
        metadata=meta,
        error_message=error_message,
    )


def build_empty_response(
    tool_name: str,
    domain: Union[str, QueryDomain],
    context: QueryContext,
    start_time: float,
    currency: Optional[str] = None,
    source_engine: str = "commerce_ai.query_layer",
    message: str = "No records matched the specified query filters and point-in-time constraints.",
) -> QueryResponse:
    """Standardized response when query filters match zero records."""
    meta = build_metadata(
        tool_name=tool_name,
        domain=domain,
        context=context,
        start_time=start_time,
        currency=currency,
        calculation_status=CalculationStatus.EMPTY_RESULT,
        source_engine=source_engine,
    )
    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.EMPTY_RESULT,
        metadata=meta,
        error_message=message,
    )


def build_unavailable_response(
    tool_name: str,
    domain: Union[str, QueryDomain],
    context: QueryContext,
    start_time: float,
    reason: str,
    source_engine: str = "commerce_ai.query_layer",
) -> QueryResponse:
    """Standardized response when requested metric or model output is unavailable without fabrication."""
    meta = build_metadata(
        tool_name=tool_name,
        domain=domain,
        context=context,
        start_time=start_time,
        currency=context.effective_currency,
        calculation_status=CalculationStatus.UNAVAILABLE,
        source_engine=source_engine,
    )
    return QueryResponse(
        query_id=meta.query_id,
        tool_name=tool_name,
        domain=meta.domain,
        status=CalculationStatus.UNAVAILABLE,
        metadata=meta,
        error_message=reason,
    )
