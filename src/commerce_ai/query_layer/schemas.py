"""Response contracts and data schemas for Business Intelligence Tool & Dashboard Layer (Phase 7A).

Provides standardized, typed response contracts serving both Executive Dashboards and eRetail Copilot:
- QueryMetadata: Provenance, execution timing, data period, calculation status
- EvidenceReference: Deterministic lineage and auditability back to source engines
- MetricResult: Standardized KPI card contract (display name, value, changes, unit, currency)
- TimeSeriesResult: Chart/trend response contract
- BreakdownResult: Dimensional breakdown contract (e.g. by warehouse, channel, category)
- TableResult: Tabular dataset response contract
- InsightResult: Analytical findings and observations for dashboards and Copilot reasoning
- QueryResponse: Unified query container envelope
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class CalculationStatus(str, Enum):
    """Execution and calculation statuses for queries."""
    SUCCESS = "SUCCESS"
    PARTIAL_DATA = "PARTIAL_DATA"
    UNAVAILABLE = "UNAVAILABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    CURRENCY_INCONSISTENCY = "CURRENCY_INCONSISTENCY"
    INVALID_FILTER = "INVALID_FILTER"
    EMPTY_RESULT = "EMPTY_RESULT"
    ERROR = "ERROR"


class QueryDomain(str, Enum):
    """Functional domains within the eRetail business intelligence system."""
    SALES = "sales"
    INVENTORY = "inventory"
    DEMAND = "demand"
    FORECASTING = "forecasting"
    RETURNS = "returns"
    FINANCIAL = "financial"
    OPERATIONS = "operations"
    IMPACT = "impact"
    RECOMMENDATIONS = "recommendations"
    DECISIONS = "decisions"


class EvidenceReference(BaseModel):
    """Machine-readable traceability back to source engine, signal, record, or model."""
    model_config = ConfigDict(frozen=True)

    source_engine: str = Field(description="Originating engine or module name")
    source_type: str = Field(description="Record category: signal, impact_record, order, snapshot, etc.")
    source_id: str = Field(description="Unique deterministic identifier of the upstream record")
    metric: Optional[str] = Field(default=None, description="Specific metric name ground truth relates to")
    value: Optional[Union[float, int, str]] = Field(default=None, description="Recorded value")
    currency: Optional[str] = Field(default=None, description="Currency ISO code if monetary")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time timestamp of evidence")
    notes: Optional[str] = Field(default=None, description="Contextual note or provenance explanation")


class QueryMetadata(BaseModel):
    """Standardized metadata accompanying every query response."""
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(description="Unique deterministic or generated query invocation identifier")
    tool_name: str = Field(description="Name of the business intelligence tool invoked")
    domain: str = Field(description="Business domain of the query")
    generated_as_of: str = Field(description="ISO timestamp/date when query was generated")
    currency: Optional[str] = Field(default=None, description="Currency of results, if monetary")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Normalized filters applied")
    data_period: Optional[Dict[str, Optional[str]]] = Field(
        default=None,
        description="Temporal boundary of data (start_date, end_date)",
    )
    calculation_status: CalculationStatus = Field(
        default=CalculationStatus.SUCCESS,
        description="Calculation status code",
    )
    confidence_provenance: str = Field(
        default="DETERMINISTIC_DERIVED",
        description="Confidence provenance tier (e.g. DETERMINISTIC_DERIVED, DIRECT_OBSERVED, MODEL_BASED)",
    )
    source_engine: str = Field(
        default="commerce_ai.query_layer",
        description="Primary upstream engine providing results",
    )
    execution_time_ms: float = Field(default=0.0, description="Server-side execution time in milliseconds")


class MetricResult(BaseModel):
    """Standardized Dashboard KPI card contract.
    
    Powers KPI widgets on dashboards and factual answers in eRetail Copilot.
    """
    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(description="Canonical metric identifier, e.g. gross_revenue, gross_margin")
    display_name: str = Field(description="Human-readable business label, e.g. 'Gross Revenue'")
    value: Optional[Union[float, int, str]] = Field(default=None, description="Calculated metric value")
    unit: str = Field(default="USD", description="Unit of measurement: USD, units, ratio, %, days, etc.")
    currency: Optional[str] = Field(default=None, description="Currency ISO code if monetary")
    period: Optional[str] = Field(default=None, description="Period description (e.g. '2026-01-01 to 2026-01-31')")
    previous_period_value: Optional[Union[float, int]] = Field(default=None, description="Prior comparison value")
    absolute_change: Optional[Union[float, int]] = Field(default=None, description="Change vs prior period")
    percentage_change: Optional[float] = Field(default=None, description="Percentage change vs prior period")
    status: str = Field(default="AVAILABLE", description="Metric availability status: AVAILABLE, UNAVAILABLE, etc.")
    source: str = Field(default="commerce_ai.query_layer", description="Source module or calculation")
    as_of_date: Optional[str] = Field(default=None, description="Point-in-time date for the metric")
    evidence: List[EvidenceReference] = Field(default_factory=list, description="Linked evidence references")


class TimeSeriesPoint(BaseModel):
    """Individual time-series data point."""
    model_config = ConfigDict(frozen=True)

    date: str = Field(description="ISO date string (YYYY-MM-DD or period)")
    value: Optional[Union[float, int]] = Field(default=None, description="Metric value at this timestamp")
    metric_name: str = Field(description="Metric identifier")
    currency: Optional[str] = Field(default=None, description="Currency if monetary")
    dimensions: Dict[str, str] = Field(default_factory=dict, description="Segment dimensions if multi-series")


class TimeSeriesResult(BaseModel):
    """Time-series chart contract for trends and forecasts."""
    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(description="Metric identifier")
    display_name: str = Field(description="Human-readable series name")
    time_grain: str = Field(default="daily", description="Time granularity: daily, weekly, monthly")
    points: List[TimeSeriesPoint] = Field(default_factory=list, description="Ordered time series observations")
    unit: str = Field(default="", description="Unit of measurement")
    currency: Optional[str] = Field(default=None, description="Currency ISO code")
    metadata: QueryMetadata = Field(description="Execution metadata")
    evidence: List[EvidenceReference] = Field(default_factory=list, description="Linked evidence references")


class BreakdownItem(BaseModel):
    """Individual slice/bucket in a dimensional breakdown."""
    model_config = ConfigDict(frozen=True)

    dimension_name: str = Field(description="Dimension name (e.g. warehouse_id, category_id)")
    dimension_value: str = Field(description="Dimension key (e.g. WH-001, Electronics)")
    metric_value: Optional[Union[float, int]] = Field(default=None, description="Metric value for this bucket")
    metric_name: str = Field(description="Metric identifier")
    percentage_of_total: Optional[float] = Field(default=None, description="Share of total portfolio (0-100)")
    currency: Optional[str] = Field(default=None, description="Currency if monetary")
    additional_metrics: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary metrics for this bucket")


class BreakdownResult(BaseModel):
    """Dimensional breakdown contract for charts, pies, and bar breakdowns."""
    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(description="Metric identifier")
    display_name: str = Field(description="Human-readable breakdown title")
    dimension_name: str = Field(description="Dimension sliced on (e.g. channel, warehouse, category)")
    items: List[BreakdownItem] = Field(default_factory=list, description="List of breakdown items")
    total_value: Optional[Union[float, int]] = Field(default=None, description="Sum of metrics across breakdown")
    unit: str = Field(default="", description="Unit of measurement")
    currency: Optional[str] = Field(default=None, description="Currency ISO code")
    metadata: QueryMetadata = Field(description="Execution metadata")
    evidence: List[EvidenceReference] = Field(default_factory=list, description="Linked evidence references")


class TableResult(BaseModel):
    """Tabular dataset contract for analytical grids and detailed drill-downs."""
    model_config = ConfigDict(extra="forbid")

    columns: List[str] = Field(default_factory=list, description="Ordered column names")
    column_types: Dict[str, str] = Field(default_factory=dict, description="Data type mapping per column")
    rows: List[Dict[str, Any]] = Field(default_factory=list, description="Tabular row dictionaries")
    total_rows: int = Field(default=0, description="Total matching row count")
    metadata: QueryMetadata = Field(description="Execution metadata")
    evidence: List[EvidenceReference] = Field(default_factory=list, description="Linked evidence references")


class InsightResult(BaseModel):
    """Structured analytical insight or diagnostic finding."""
    model_config = ConfigDict(extra="forbid")

    insight_id: str = Field(description="Deterministic insight identifier")
    title: str = Field(description="Concise insight title")
    summary: str = Field(description="Fact-based summary statement")
    category: str = Field(description="Business category: margin, inventory, return, etc.")
    severity: str = Field(default="INFO", description="Severity: INFO, LOW, MEDIUM, HIGH, CRITICAL")
    impact_value: Optional[float] = Field(default=None, description="Estimated monetary impact if quantified")
    currency: Optional[str] = Field(default=None, description="Currency if monetary")
    evidence: List[EvidenceReference] = Field(default_factory=list, description="Traceability references")
    requires_human_review: bool = Field(default=True, description="Strict governance guard")


class QueryResponse(BaseModel):
    """Unified query envelope returned by all query layer tools."""
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(description="Unique query invocation identifier")
    tool_name: str = Field(description="Name of the query tool invoked")
    domain: str = Field(description="Functional domain")
    status: CalculationStatus = Field(default=CalculationStatus.SUCCESS, description="Calculation status")
    metadata: QueryMetadata = Field(description="Standard execution metadata")
    metrics: List[MetricResult] = Field(default_factory=list, description="KPI metrics if present")
    time_series: Optional[TimeSeriesResult] = Field(default=None, description="Time series result if applicable")
    breakdown: Optional[BreakdownResult] = Field(default=None, description="Breakdown result if applicable")
    table: Optional[TableResult] = Field(default=None, description="Table result if applicable")
    insights: List[InsightResult] = Field(default_factory=list, description="Generated insights if applicable")
    error_message: Optional[str] = Field(default=None, description="Error or reason message if not successful")
