"""Unit tests for Phase 7D Evidence Extraction and Lineage Preservation."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.enums import StepStatus
from commerce_ai.copilot.schemas import CopilotStepResult
from commerce_ai.explanations.enums import (
    EvidenceType,
    ProvenanceType,
)
from commerce_ai.explanations.evidence import extract_evidence_from_step_result
from commerce_ai.query_layer.schemas import (
    BreakdownItem,
    BreakdownResult,
    CalculationStatus,
    InsightResult,
    MetricResult,
    QueryMetadata,
    QueryResponse,
    TableResult,
    TimeSeriesPoint,
    TimeSeriesResult,
)


@pytest.fixture
def base_metadata() -> QueryMetadata:
    return QueryMetadata(
        query_id="QRY-001",
        tool_name="get_sales_summary",
        domain="sales",
        generated_as_of="2026-06-30",
        currency="USD",
        source_engine="commerce_ai.sales",
    )


class TestExplanationEvidence:
    def test_extract_evidence_from_metric_result(self, base_metadata):
        """Extract evidence from MetricResult items."""
        metric = MetricResult(
            metric_name="gross_revenue",
            display_name="Gross Revenue",
            value=150000.0,
            unit="USD",
            currency="USD",
            previous_period_value=140000.0,
            percentage_change=7.14,
        )
        resp = QueryResponse(
            query_id="QRY-1",
            tool_name="get_sales_summary",
            domain="sales",
            metadata=base_metadata,
            metrics=[metric],
        )
        step = CopilotStepResult(
            step_id="STEP-1",
            tool_name="get_sales_summary",
            status=StepStatus.COMPLETED,
            response=resp,
        )

        evidence = extract_evidence_from_step_result(step)
        assert len(evidence) == 1
        ev = evidence[0]
        assert ev.source_step_id == "STEP-1"
        assert ev.source_tool == "get_sales_summary"
        assert ev.metric == "gross_revenue"
        assert ev.value == 150000.0
        assert ev.comparison_value == 140000.0
        assert ev.currency == "USD"
        assert ev.provenance == ProvenanceType.PHASE_7B_DERIVED  # derived due to percentage change

    def test_extract_evidence_from_time_series(self, base_metadata):
        """Extract evidence from TimeSeriesResult points."""
        points = [
            TimeSeriesPoint(date="2026-06-01", value=1000.0, metric_name="daily_sales"),
            TimeSeriesPoint(date="2026-06-02", value=1200.0, metric_name="daily_sales"),
        ]
        ts = TimeSeriesResult(
            metric_name="daily_sales",
            display_name="Daily Sales",
            points=points,
            metadata=base_metadata,
        )
        resp = QueryResponse(
            query_id="QRY-2",
            tool_name="get_sales_trend",
            domain="sales",
            metadata=base_metadata,
            time_series=ts,
        )
        step = CopilotStepResult(
            step_id="STEP-2",
            tool_name="get_sales_trend",
            status=StepStatus.COMPLETED,
            response=resp,
        )

        evidence = extract_evidence_from_step_result(step)
        assert len(evidence) == 2
        assert evidence[0].time_range == "2026-06-01"
        assert evidence[0].value == 1000.0
        assert evidence[1].time_range == "2026-06-02"
        assert evidence[1].value == 1200.0

    def test_extract_evidence_from_breakdown(self, base_metadata):
        """Extract evidence from BreakdownResult items."""
        items = [
            BreakdownItem(dimension_name="channel_id", dimension_value="ONLINE", metric_value=75000.0, metric_name="revenue", percentage_of_total=60.0),
            BreakdownItem(dimension_name="channel_id", dimension_value="RETAIL", metric_value=50000.0, metric_name="revenue", percentage_of_total=40.0),
        ]
        bk = BreakdownResult(
            metric_name="revenue",
            display_name="Revenue by Channel",
            dimension_name="channel_id",
            items=items,
            metadata=base_metadata,
        )
        resp = QueryResponse(
            query_id="QRY-3",
            tool_name="get_sales_by_channel",
            domain="sales",
            metadata=base_metadata,
            breakdown=bk,
        )
        step = CopilotStepResult(
            step_id="STEP-3",
            tool_name="get_sales_by_channel",
            status=StepStatus.COMPLETED,
            response=resp,
        )

        evidence = extract_evidence_from_step_result(step)
        assert len(evidence) == 2
        assert evidence[0].dimension == "channel_id"
        assert evidence[0].entity == "ONLINE"
        assert evidence[0].value == 75000.0
        assert evidence[0].comparison_value == 60.0
        assert evidence[1].entity == "RETAIL"

    def test_extract_evidence_from_table(self, base_metadata):
        """Extract evidence from TableResult summary and rows."""
        rows = [
            {"sku_id": "SKU_01", "stockout_risk": "CRITICAL", "days_of_supply": 2.5},
            {"sku_id": "SKU_02", "stockout_risk": "HIGH", "days_of_supply": 5.0},
        ]
        tbl = TableResult(
            columns=["sku_id", "stockout_risk", "days_of_supply"],
            rows=rows,
            total_rows=2,
            metadata=base_metadata,
        )
        resp = QueryResponse(
            query_id="QRY-4",
            tool_name="get_stockout_risk",
            domain="inventory",
            metadata=base_metadata,
            table=tbl,
        )
        step = CopilotStepResult(
            step_id="STEP-4",
            tool_name="get_stockout_risk",
            status=StepStatus.COMPLETED,
            response=resp,
        )

        evidence = extract_evidence_from_step_result(step)
        assert len(evidence) == 3  # 1 summary + 2 rows
        assert evidence[0].metric == "total_rows"
        assert evidence[0].value == 2
        assert evidence[1].entity == "SKU_01"
        assert evidence[2].entity == "SKU_02"

    def test_extract_evidence_from_insights(self, base_metadata):
        """Extract evidence from InsightResult items."""
        insight = InsightResult(
            insight_id="INS-01",
            title="Margin Shift",
            summary="Channel shift contributed to 1.2% margin variance.",
            category="margin",
            severity="MEDIUM",
            impact_value=12500.0,
            currency="USD",
        )
        resp = QueryResponse(
            query_id="QRY-5",
            tool_name="get_margin_summary",
            domain="financial",
            metadata=base_metadata,
            insights=[insight],
        )
        step = CopilotStepResult(
            step_id="STEP-5",
            tool_name="get_margin_summary",
            status=StepStatus.COMPLETED,
            response=resp,
        )

        evidence = extract_evidence_from_step_result(step)
        assert len(evidence) == 1
        assert evidence[0].value == 12500.0
        assert "Margin Shift" in (evidence[0].notes or "")

    def test_extract_evidence_from_failed_step(self):
        """Extract evidence from step with None response envelope."""
        step = CopilotStepResult(
            step_id="STEP-ERR",
            tool_name="get_sales_summary",
            status=StepStatus.FAILED,
            response=None,
        )
        evidence = extract_evidence_from_step_result(step)
        assert len(evidence) == 1
        assert evidence[0].provenance == ProvenanceType.INSUFFICIENT_DATA
        assert evidence[0].value is None

    def test_forecast_provenance_inference(self, base_metadata):
        """Forecast tools must automatically infer MODEL_BASED provenance and MODEL_PROJECTED type."""
        meta = QueryMetadata(
            query_id="QRY-F",
            tool_name="get_forecast",
            domain="forecasting",
            generated_as_of="2026-06-30",
            source_engine="commerce_ai.forecasting",
        )
        metric = MetricResult(
            metric_name="forecast_mean",
            display_name="Forecast Mean",
            value=5200.0,
        )
        resp = QueryResponse(
            query_id="QRY-F",
            tool_name="get_forecast",
            domain="forecasting",
            metadata=meta,
            metrics=[metric],
        )
        step = CopilotStepResult(
            step_id="STEP-F",
            tool_name="get_forecast",
            status=StepStatus.COMPLETED,
            response=resp,
        )
        evidence = extract_evidence_from_step_result(step)
        assert evidence[0].provenance == ProvenanceType.MODEL_BASED
        assert evidence[0].evidence_type == EvidenceType.MODEL_PROJECTED
