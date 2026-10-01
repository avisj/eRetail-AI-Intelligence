"""Unit tests for Trend Explanations (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import (
    CopilotExecutionResult,
    CopilotResponse,
    CopilotStepResult,
)
from commerce_ai.explanations.enums import ExplanationConfidence, ExplanationStatus, ExplanationType
from commerce_ai.explanations.templates import build_trend_explanation
from commerce_ai.query_layer.schemas import (
    QueryMetadata,
    QueryResponse,
    TimeSeriesPoint,
    TimeSeriesResult,
)


def _make_trend_response(points: list[TimeSeriesPoint]) -> CopilotResponse:
    meta = QueryMetadata(
        query_id="Q-TR",
        tool_name="get_demand_trend",
        domain="demand",
        generated_as_of="2026-06-30",
        currency="USD",
        source_engine="commerce_ai.demand",
    )
    ts = TimeSeriesResult(
        metric_name="daily_demand",
        display_name="Daily Demand",
        time_grain="daily",
        points=points,
        metadata=meta,
    )
    q_resp = QueryResponse(
        query_id="Q-TR",
        tool_name="get_demand_trend",
        domain="demand",
        metadata=meta,
        time_series=ts,
    )
    step = CopilotStepResult(
        step_id="STEP-TR",
        tool_name="get_demand_trend",
        status=StepStatus.COMPLETED,
        response=q_resp,
    )
    exec_res = CopilotExecutionResult(
        result_id="RES-TR",
        request_id="REQ-TR",
        status=PlanExecutionStatus.COMPLETE,
        step_results=[step],
    )
    return CopilotResponse(
        response_id="RESP-TR",
        request_id="REQ-TR",
        state=CopilotState.COMPLETED,
        normalized_question="How is demand trending?",
        execution_result=exec_res,
    )


class TestTrendExplanations:
    def test_increasing_trend(self):
        pts = [
            TimeSeriesPoint(date="2026-06-01", value=100.0, metric_name="daily_demand"),
            TimeSeriesPoint(date="2026-06-02", value=150.0, metric_name="daily_demand"),
        ]
        resp = _make_trend_response(pts)
        exp = build_trend_explanation(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert exp.explanation_type == ExplanationType.TREND_EXPLANATION
        assert "increased" in exp.headline.lower()
        assert len(exp.key_findings) == 1
        assert "increased" in exp.key_findings[0].statement
        assert exp.trends[0]["direction"] == "increased"

    def test_decreasing_trend(self):
        pts = [
            TimeSeriesPoint(date="2026-06-01", value=200.0, metric_name="daily_demand"),
            TimeSeriesPoint(date="2026-06-02", value=120.0, metric_name="daily_demand"),
        ]
        resp = _make_trend_response(pts)
        exp = build_trend_explanation(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert "decreased" in exp.headline.lower()
        assert exp.trends[0]["direction"] == "decreased"

    def test_flat_trend(self):
        pts = [
            TimeSeriesPoint(date="2026-06-01", value=150.0, metric_name="daily_demand"),
            TimeSeriesPoint(date="2026-06-02", value=150.0, metric_name="daily_demand"),
        ]
        resp = _make_trend_response(pts)
        exp = build_trend_explanation(resp)
        assert exp.status == ExplanationStatus.SUCCESS
        assert "remained flat" in exp.headline.lower()
        assert exp.trends[0]["direction"] == "remained flat"

    def test_single_point_insufficient_trend(self):
        pts = [
            TimeSeriesPoint(date="2026-06-01", value=100.0, metric_name="daily_demand"),
        ]
        resp = _make_trend_response(pts)
        exp = build_trend_explanation(resp)
        assert exp.status == ExplanationStatus.INSUFFICIENT_DATA
        assert "insufficient" in exp.summary.lower()

    def test_causal_limitation_attached(self):
        pts = [
            TimeSeriesPoint(date="2026-06-01", value=100.0, metric_name="daily_demand"),
            TimeSeriesPoint(date="2026-06-02", value=150.0, metric_name="daily_demand"),
        ]
        resp = _make_trend_response(pts)
        exp = build_trend_explanation(resp)
        assert any("causal" in lim.lower() for lim in exp.limitations)
