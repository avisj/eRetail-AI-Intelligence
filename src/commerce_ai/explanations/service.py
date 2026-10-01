"""High-Level Business Explanation Service (Phase 7D).

Provides the unified entry point for generating grounded, auditable, human-readable
business explanations from Phase 7C CopilotResponse envelopes.
"""

from __future__ import annotations

from typing import Optional

from commerce_ai.copilot.enums import CopilotState
from commerce_ai.copilot.schemas import CopilotResponse
from commerce_ai.query_layer.schemas import CalculationStatus
from commerce_ai.explanations.auditor import ClaimAuditor, default_auditor
from commerce_ai.explanations.enums import (
    ExplanationStatus,
    ExplanationType,
)
from commerce_ai.explanations.renderer import render_explanation_to_text
from commerce_ai.explanations.schemas import AuditResult, BusinessExplanation
from commerce_ai.explanations.templates import (
    build_breakdown_explanation,
    build_comparison_explanation,
    build_forecast_explanation,
    build_insufficient_data_explanation,
    build_kpi_explanation,
    build_multi_domain_explanation,
    build_operational_review_explanation,
    build_trend_explanation,
    build_unavailable_explanation,
    build_why_inventory_risk_explanation,
    build_why_margin_explanation,
    build_why_returns_explanation,
)
from commerce_ai.query_contracts.enums import QueryIntent, RequestedOutput


class ExplanationService:
    """Unified service for converting CopilotResponse artifacts into auditable BusinessExplanations."""

    def __init__(self, auditor: Optional[ClaimAuditor] = None) -> None:
        self.auditor = auditor or default_auditor

    def explain(self, response: CopilotResponse) -> BusinessExplanation:
        """Transform a CopilotResponse into a grounded, validated BusinessExplanation."""
        # 1. Handle unavailable capabilities
        is_step_unavailable = (
            response.execution_result is not None
            and any(
                s.response and s.response.status == CalculationStatus.UNAVAILABLE
                for s in response.execution_result.step_results
            )
        )
        if response.state == CopilotState.UNAVAILABLE or is_step_unavailable:
            exp = build_unavailable_explanation(response)
            self._audit_and_validate(exp)
            return exp

        # 2. Handle clarification required or security/execution failures
        if response.state == CopilotState.CLARIFICATION_REQUIRED:
            reason = response.clarification.reason if response.clarification else "Question is ambiguous."
            exp = build_insufficient_data_explanation(response, reason=reason)
            self._audit_and_validate(exp)
            return exp

        if response.state == CopilotState.FAILED:
            reason = "Execution halted due to security violation or execution policy failure."
            exp = build_insufficient_data_explanation(response, reason=reason)
            self._audit_and_validate(exp)
            return exp

        # 3. Handle empty execution results
        if not response.execution_result or not response.execution_result.step_results:
            exp = build_insufficient_data_explanation(response, reason="No step results were produced by execution.")
            self._audit_and_validate(exp)
            return exp

        # 4. Route by template or intent
        interp = response.interpretation
        template_id = interp.matched_template_id if interp else None
        intent = interp.intent if interp else None

        # Diagnostic Why Investigations
        if template_id == "TPL_WHY_MARGIN_DOWN" or (interp and interp.explanation_context == "Why is margin down investigation"):
            exp = build_why_margin_explanation(response)
        elif template_id == "TPL_WHY_INVENTORY_RISK_HIGH" or (interp and interp.explanation_context == "Why is inventory risk high investigation"):
            exp = build_why_inventory_risk_explanation(response)
        elif template_id == "TPL_WHY_RETURNS_INCREASING" or (interp and interp.explanation_context == "Why are returns increasing investigation"):
            exp = build_why_returns_explanation(response)
        # Multi-Domain
        elif template_id == "TPL_MULTI_DOMAIN_PERFORMANCE" or intent == QueryIntent.MULTI_DOMAIN_ANALYSIS:
            exp = build_multi_domain_explanation(response)
        # Forecasting
        elif intent in (QueryIntent.FORECAST_ANALYSIS, QueryIntent.FORECAST_ACCURACY, QueryIntent.FORECAST_BIAS):
            exp = build_forecast_explanation(response)
        # Operational reviews & governance
        elif intent in (
            QueryIntent.REPLENISHMENT_REVIEW,
            QueryIntent.PURCHASE_ORDER_REVIEW,
            QueryIntent.WAREHOUSE_REBALANCING_REVIEW,
            QueryIntent.RECOMMENDATION_REVIEW,
            QueryIntent.DECISION_REVIEW,
        ):
            exp = build_operational_review_explanation(response)
        # Time-series trends
        elif interp and (interp.requested_output == RequestedOutput.TIME_SERIES or any(s.response and s.response.time_series for s in response.execution_result.step_results)):
            exp = build_trend_explanation(response)
        # Dimensional breakdowns
        elif interp and (interp.requested_output == RequestedOutput.BREAKDOWN or any(s.response and s.response.breakdown for s in response.execution_result.step_results)):
            exp = build_breakdown_explanation(response)
        # Check if comparison data is present
        elif any(
            any(m.previous_period_value is not None for m in s.response.metrics)
            for s in response.execution_result.step_results if s.response and s.response.metrics
        ):
            exp = build_comparison_explanation(response)
        # Default to KPI explanation
        else:
            exp = build_kpi_explanation(response)

        # 5. Audit explanation claims
        self._audit_and_validate(exp)
        return exp

    def explain_text(self, response: CopilotResponse) -> str:
        """Generate structured explanation and render it into human-readable business text."""
        explanation = self.explain(response)
        return render_explanation_to_text(explanation)

    def _audit_and_validate(self, explanation: BusinessExplanation) -> AuditResult:
        """Run claim audit and ensure governance compliance."""
        audit_res = self.auditor.audit(explanation)
        if not audit_res.is_valid:
            # If critical violations occur, raise ValueError to protect integrity
            msg = f"Explanation audit failed: {'; '.join(audit_res.violations)}"
            raise ValueError(msg)
        return audit_res
