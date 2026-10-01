"""Deterministic Template Builders for Business Explanations (Phase 7D).

Provides deterministic, grounded explanation builders across all 15 ExplanationTypes:
- KPI, Trend, Breakdown, Comparison
- Diagnostic Why (Margin Down, Inventory Risk High, Returns Increasing)
- Multi-Domain Cross-Functional Synthesis
- Financial, Inventory, Returns, Forecast Domain Inquiries
- Operational Reviews (Replenishment, POs, Rebalancing, Recommendations, Decisions)
- Data Quality, Insufficient Data, and Unavailable Capabilities
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from commerce_ai.copilot.enums import CopilotState, StepStatus
from commerce_ai.copilot.schemas import CopilotResponse, CopilotStepResult
from commerce_ai.explanations.confidence import evaluate_evidence_confidence
from commerce_ai.explanations.enums import (
    ClaimCategory,
    EvidenceType,
    ExplanationConfidence,
    ExplanationStatus,
    ExplanationType,
    ProvenanceType,
)
from commerce_ai.explanations.evidence import extract_evidence_from_step_result
from commerce_ai.explanations.schemas import (
    BusinessExplanation,
    ExplanationEvidence,
    ExplanationGovernance,
    Finding,
)

# Standardized limitations
FINANCIAL_LIMITATIONS = [
    "Product cost is based on the catalog standard procurement cost model and is not accounting-level FIFO/LIFO COGS.",
    "Operating margin and contribution margin reflect configured operational cost assumptions rather than finalized ledger entries.",
]

FORECAST_LIMITATIONS = [
    "Forecast values represent forward-looking statistical projections and historical pattern extrapolations, not guaranteed outcomes.",
    "Model performance may vary during periods of unexpected macroeconomic volatility or demand disruption.",
]

RETURN_RISK_LIMITATIONS = [
    "Return risk tiers represent machine learning predictive signals and should be validated alongside customer feedback.",
]

CAUSAL_LIMITATIONS = [
    "Observed correlations and simultaneous variations indicate co-occurrence; causal mechanisms have not been empirically modeled.",
]

OPERATIONAL_GOVERNANCE_NOTES = [
    "Operational proposals require explicit human review and authorization; autonomous execution is prohibited.",
]


def _make_explanation_id(request_id: str, explanation_type: str) -> str:
    raw = f"{request_id}:{explanation_type}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"EXP-{digest}"


def _make_finding_id(explanation_id: str, index: int) -> str:
    return f"{explanation_id}-FND-{index:02d}"


def _format_metric_val(val: Any, unit: Optional[str] = None, currency: Optional[str] = None) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        if unit == "%":
            return f"{val:.1f}%"
        if unit == "USD" or currency == "USD":
            return f"${val:,.2f}"
        if isinstance(val, float):
            return f"{val:,.2f}"
        return f"{val:,d}"
    return str(val)


# ============================================================
# 1. KPI EXPLANATION BUILDER
# ============================================================

def build_kpi_explanation(response: CopilotResponse) -> BusinessExplanation:
    """Build deterministic explanation for single or multi-KPI inquiries."""
    exp_id = _make_explanation_id(response.request_id, ExplanationType.KPI_EXPLANATION.value)
    step_results = response.execution_result.step_results if response.execution_result else []

    all_evidence: List[ExplanationEvidence] = []
    for s in step_results:
        all_evidence.extend(extract_evidence_from_step_result(s))

    findings: List[Finding] = []
    metrics_dict: Dict[str, Any] = {}
    headline = "Key performance indicators retrieved."
    summary_parts = []

    f_idx = 1
    for ev in all_evidence:
        if ev.metric and ev.value is not None:
            metrics_dict[ev.metric] = ev.value
            val_str = _format_metric_val(ev.value, ev.unit, ev.currency)
            m_name = ev.metric.replace("_", " ").title()
            if ev.provenance == ProvenanceType.MODEL_BASED:
                stmt = f"Model-based {m_name} was estimated at {val_str}."
            else:
                stmt = f"{m_name} was {val_str}."
            if ev.comparison_value is not None:
                comp_str = _format_metric_val(ev.comparison_value, ev.unit, ev.currency)
                stmt += f" Previous period was {comp_str}."

            findings.append(
                Finding(
                    finding_id=_make_finding_id(exp_id, f_idx),
                    statement=stmt,
                    evidence=[ev],
                    provenance=ev.provenance,
                    evidence_type=ev.evidence_type,
                    confidence=ExplanationConfidence.HIGH,
                    claim_category=ClaimCategory.OBSERVED_FACT,
                    supporting_metric=ev.metric,
                    source_step=ev.source_step_id,
                    source_tool=ev.source_tool,
                )
            )
            f_idx += 1
            summary_parts.append(f"{m_name} is {val_str}")

    if summary_parts:
        headline = f"Reported {summary_parts[0]} for the selected period."
        summary = "Analytical evaluation: " + "; ".join(summary_parts) + "."
    else:
        headline = "KPI values unavailable."
        summary = "No metric values were returned for the requested scope."

    confidence = evaluate_evidence_confidence(all_evidence, response.execution_result)

    limitations = []
    if any("margin" in k or "cost" in k or "revenue" in k for k in metrics_dict.keys()):
        limitations.extend(FINANCIAL_LIMITATIONS)
    if any("return" in k for k in metrics_dict.keys()) or any("return" in e.source_tool for e in all_evidence):
        limitations.extend(RETURN_RISK_LIMITATIONS)

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.SUCCESS if findings else ExplanationStatus.INSUFFICIENT_DATA,
        explanation_type=ExplanationType.KPI_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=findings,
        supporting_evidence=all_evidence,
        metrics=metrics_dict,
        limitations=limitations,
        provenance=ProvenanceType.PHASE_7A_OBSERVED,
        confidence=confidence,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(),
    )


# ============================================================
# 2. TREND EXPLANATION BUILDER
# ============================================================

def build_trend_explanation(response: CopilotResponse) -> BusinessExplanation:
    """Build deterministic explanation for time-series trend inquiries."""
    exp_id = _make_explanation_id(response.request_id, ExplanationType.TREND_EXPLANATION.value)
    step_results = response.execution_result.step_results if response.execution_result else []

    all_evidence: List[ExplanationEvidence] = []
    for s in step_results:
        all_evidence.extend(extract_evidence_from_step_result(s))

    ts_evidence = [e for e in all_evidence if e.time_range and e.value is not None]
    findings: List[Finding] = []
    trends_list: List[Dict[str, Any]] = []

    if len(ts_evidence) >= 2:
        start_pt = ts_evidence[0]
        end_pt = ts_evidence[-1]
        start_val = float(start_pt.value) if isinstance(start_pt.value, (int, float)) else 0.0
        end_val = float(end_pt.value) if isinstance(end_pt.value, (int, float)) else 0.0
        delta = end_val - start_val
        direction = "increased" if delta > 0 else ("decreased" if delta < 0 else "remained flat")

        m_name = (start_pt.metric or "Metric").replace("_", " ").title()
        val_start_str = _format_metric_val(start_val, start_pt.unit, start_pt.currency)
        val_end_str = _format_metric_val(end_val, end_pt.unit, end_pt.currency)

        stmt = f"{m_name} {direction} from {val_start_str} on {start_pt.time_range} to {val_end_str} on {end_pt.time_range}."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, 1),
                statement=stmt,
                evidence=[start_pt, end_pt],
                provenance=ProvenanceType.PHASE_7A_OBSERVED,
                evidence_type=EvidenceType.DIRECT_OBSERVED,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.TREND_STATEMENT,
                supporting_metric=start_pt.metric,
                source_step=start_pt.source_step_id,
                source_tool=start_pt.source_tool,
            )
        )
        headline = f"{m_name} {direction} over the observed time series."
        summary = f"Evaluation across {len(ts_evidence)} observations: {stmt}"
        trends_list.append({
            "metric": start_pt.metric,
            "start_date": start_pt.time_range,
            "end_date": end_pt.time_range,
            "start_value": start_val,
            "end_value": end_val,
            "delta": delta,
            "direction": direction,
        })
    else:
        headline = "Time-series trend data evaluated."
        summary = "Insufficient time-series points to calculate directional change."

    confidence = evaluate_evidence_confidence(all_evidence, response.execution_result)

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.SUCCESS if findings else ExplanationStatus.INSUFFICIENT_DATA,
        explanation_type=ExplanationType.TREND_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=findings,
        supporting_evidence=all_evidence,
        trends=trends_list,
        limitations=CAUSAL_LIMITATIONS,
        provenance=ProvenanceType.PHASE_7A_OBSERVED,
        confidence=confidence,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(),
    )


# ============================================================
# 3. BREAKDOWN EXPLANATION BUILDER
# ============================================================

def build_breakdown_explanation(response: CopilotResponse) -> BusinessExplanation:
    """Build deterministic explanation for dimensional breakdowns."""
    exp_id = _make_explanation_id(response.request_id, ExplanationType.BREAKDOWN_EXPLANATION.value)
    step_results = response.execution_result.step_results if response.execution_result else []

    all_evidence: List[ExplanationEvidence] = []
    for s in step_results:
        all_evidence.extend(extract_evidence_from_step_result(s))

    bk_evidence = [e for e in all_evidence if e.dimension and e.entity]
    findings: List[Finding] = []

    dim_name = bk_evidence[0].dimension.replace("_", " ").title() if bk_evidence else "Dimension"
    f_idx = 1
    for ev in bk_evidence:
        val_str = _format_metric_val(ev.value, ev.unit, ev.currency)
        share_str = f" ({ev.comparison_value:.1f}% share)" if ev.comparison_value is not None else ""
        stmt = f"{dim_name} segment '{ev.entity}' accounted for {val_str}{share_str}."

        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=[ev],
                provenance=ev.provenance,
                evidence_type=ev.evidence_type,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.BREAKDOWN_STATEMENT,
                supporting_metric=ev.metric,
                source_step=ev.source_step_id,
                source_tool=ev.source_tool,
            )
        )
        f_idx += 1

    headline = f"Breakdown by {dim_name.lower()} evaluated."
    summary = f"Observed distribution across {len(bk_evidence)} {dim_name.lower()} segments without ranking or subjective ordering."

    confidence = evaluate_evidence_confidence(all_evidence, response.execution_result)

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.SUCCESS if findings else ExplanationStatus.INSUFFICIENT_DATA,
        explanation_type=ExplanationType.BREAKDOWN_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=findings,
        supporting_evidence=all_evidence,
        limitations=CAUSAL_LIMITATIONS,
        provenance=ProvenanceType.PHASE_7A_OBSERVED,
        confidence=confidence,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(),
    )


# ============================================================
# 4. COMPARISON EXPLANATION BUILDER
# ============================================================

def build_comparison_explanation(response: CopilotResponse) -> BusinessExplanation:
    """Build deterministic explanation comparing current vs prior periods or baselines."""
    exp_id = _make_explanation_id(response.request_id, ExplanationType.COMPARISON_EXPLANATION.value)
    step_results = response.execution_result.step_results if response.execution_result else []

    all_evidence: List[ExplanationEvidence] = []
    for s in step_results:
        all_evidence.extend(extract_evidence_from_step_result(s))

    comp_evidence = [e for e in all_evidence if e.value is not None and e.comparison_value is not None]
    findings: List[Finding] = []
    comparisons_list: List[Dict[str, Any]] = []

    f_idx = 1
    for ev in comp_evidence:
        cur_val = float(ev.value) if isinstance(ev.value, (int, float)) else 0.0
        prev_val = float(ev.comparison_value) if isinstance(ev.comparison_value, (int, float)) else 0.0
        delta = cur_val - prev_val
        pct_delta = ((cur_val - prev_val) / prev_val * 100.0) if prev_val != 0 else None

        cur_str = _format_metric_val(cur_val, ev.unit, ev.currency)
        prev_str = _format_metric_val(prev_val, ev.unit, ev.currency)
        m_name = (ev.metric or "Metric").replace("_", " ").title()

        change_phrase = f"a change of {delta:+.2f}"
        if pct_delta is not None:
            change_phrase += f" ({pct_delta:+.1f}%)"

        stmt = f"{m_name} was {cur_str} compared to {prev_str} in the prior period, {change_phrase}."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=[ev],
                provenance=ProvenanceType.PHASE_7B_DERIVED,
                evidence_type=EvidenceType.DETERMINISTIC_DERIVED,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.COMPARISON_STATEMENT,
                supporting_metric=ev.metric,
                source_step=ev.source_step_id,
                source_tool=ev.source_tool,
            )
        )
        comparisons_list.append({
            "metric": ev.metric,
            "current_value": cur_val,
            "comparison_value": prev_val,
            "absolute_change": delta,
            "percentage_change": pct_delta,
        })
        f_idx += 1

    if findings:
        headline = "Period-over-period comparison evaluated."
        summary = f"Comparison across {len(findings)} reported metric indicators against prior baseline."
    else:
        headline = "Comparison data evaluated."
        summary = "No prior baseline or comparative metric values were available in the evidence."

    confidence = evaluate_evidence_confidence(all_evidence, response.execution_result)

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.SUCCESS if findings else ExplanationStatus.INSUFFICIENT_DATA,
        explanation_type=ExplanationType.COMPARISON_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=findings,
        supporting_evidence=all_evidence,
        comparisons=comparisons_list,
        limitations=CAUSAL_LIMITATIONS,
        provenance=ProvenanceType.PHASE_7B_DERIVED,
        confidence=confidence,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(),
    )


# ============================================================
# 5. WHY MARGIN DOWN EXPLANATION BUILDER
# ============================================================

def build_why_margin_explanation(response: CopilotResponse) -> BusinessExplanation:
    """Build multi-step diagnostic explanation for 'Why did margin decline?' inquiry."""
    exp_id = _make_explanation_id(response.request_id, "WHY_MARGIN_DOWN")
    step_results = response.execution_result.step_results if response.execution_result else []

    all_evidence: List[ExplanationEvidence] = []
    for s in step_results:
        all_evidence.extend(extract_evidence_from_step_result(s))

    findings: List[Finding] = []
    f_idx = 1

    # Step 1: get_margin_summary
    margin_ev = [e for e in all_evidence if e.source_tool == "get_margin_summary" and e.metric in ("gross_margin", "gross_margin_percent", "margin_rate")]
    if margin_ev:
        ev = margin_ev[0]
        val_str = _format_metric_val(ev.value, ev.unit, ev.currency)
        stmt = f"Gross margin was reported at {val_str} for the selected analytical period."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=[ev],
                provenance=ev.provenance,
                evidence_type=ev.evidence_type,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.OBSERVED_FACT,
                supporting_metric=ev.metric,
                source_step=ev.source_step_id,
                source_tool=ev.source_tool,
            )
        )
        f_idx += 1

    # Step 2: get_margin_drivers
    driver_ev = [e for e in all_evidence if e.source_tool == "get_margin_drivers"]
    if driver_ev:
        stmt = f"Variance analysis identified {len(driver_ev)} cost and contribution margin driver records."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=driver_ev[:3],
                provenance=ProvenanceType.PHASE_7B_DERIVED,
                evidence_type=EvidenceType.DETERMINISTIC_DERIVED,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.DERIVED_METRIC,
                supporting_metric="margin_drivers",
                source_step=driver_ev[0].source_step_id,
                source_tool=driver_ev[0].source_tool,
            )
        )
        f_idx += 1

    # Step 3: get_sales_by_channel
    channel_ev = [e for e in all_evidence if e.source_tool == "get_sales_by_channel" and e.entity]
    if channel_ev:
        stmt = f"Commercial sales revenue was distributed across {len(channel_ev)} channels, coinciding with channel mix variation."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=channel_ev[:3],
                provenance=ProvenanceType.PHASE_7A_OBSERVED,
                evidence_type=EvidenceType.DIRECT_OBSERVED,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.BREAKDOWN_STATEMENT,
                supporting_metric="channel_sales",
                source_step=channel_ev[0].source_step_id,
                source_tool=channel_ev[0].source_tool,
            )
        )
        f_idx += 1

    # Step 4: get_sales_by_warehouse
    wh_ev = [e for e in all_evidence if e.source_tool == "get_sales_by_warehouse" and e.entity]
    if wh_ev:
        stmt = f"Fulfillment revenue was distributed across {len(wh_ev)} distribution centers."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=wh_ev[:3],
                provenance=ProvenanceType.PHASE_7A_OBSERVED,
                evidence_type=EvidenceType.DIRECT_OBSERVED,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.BREAKDOWN_STATEMENT,
                supporting_metric="warehouse_sales",
                source_step=wh_ev[0].source_step_id,
                source_tool=wh_ev[0].source_tool,
            )
        )
        f_idx += 1

    headline = "Diagnostic analysis of gross margin factors."
    summary = (
        "Gross margin performance was evaluated across margin drivers, channel distribution, and fulfillment locations. "
        "Observed shifts in revenue mix and procurement cost drivers coincided with margin movement without establishing direct singular causality."
    )

    confidence = evaluate_evidence_confidence(all_evidence, response.execution_result)
    limitations = list(FINANCIAL_LIMITATIONS) + list(CAUSAL_LIMITATIONS)

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.SUCCESS if findings else ExplanationStatus.INSUFFICIENT_DATA,
        explanation_type=ExplanationType.WHY_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=findings,
        supporting_evidence=all_evidence,
        limitations=limitations,
        provenance=ProvenanceType.PHASE_7C_ORCHESTRATED,
        confidence=confidence,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(),
    )


# ============================================================
# 6. WHY INVENTORY RISK HIGH EXPLANATION BUILDER
# ============================================================

def build_why_inventory_risk_explanation(response: CopilotResponse) -> BusinessExplanation:
    """Build multi-step diagnostic explanation for 'Why is inventory risk high?' inquiry."""
    exp_id = _make_explanation_id(response.request_id, "WHY_INVENTORY_RISK_HIGH")
    step_results = response.execution_result.step_results if response.execution_result else []

    all_evidence: List[ExplanationEvidence] = []
    for s in step_results:
        all_evidence.extend(extract_evidence_from_step_result(s))

    findings: List[Finding] = []
    f_idx = 1

    # Step 1: get_inventory_summary
    inv_sum = [e for e in all_evidence if e.source_tool == "get_inventory_summary" and e.metric in ("total_units", "available_units", "inventory_units")]
    if inv_sum:
        ev = inv_sum[0]
        val_str = _format_metric_val(ev.value, ev.unit, ev.currency)
        stmt = f"Total inventory volume was {val_str} across the active portfolio."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=[ev],
                provenance=ev.provenance,
                evidence_type=ev.evidence_type,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.OBSERVED_FACT,
                supporting_metric=ev.metric,
                source_step=ev.source_step_id,
                source_tool=ev.source_tool,
            )
        )
        f_idx += 1

    # Step 2: get_inventory_risk
    risk_ev = [e for e in all_evidence if e.source_tool == "get_inventory_risk"]
    if risk_ev:
        stmt = "Portfolio inventory risk breakdown categorizes risk exposure across operational exposure segments."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=risk_ev[:3],
                provenance=ProvenanceType.PHASE_7A_OBSERVED,
                evidence_type=EvidenceType.DIRECT_OBSERVED,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.RISK_STATEMENT,
                supporting_metric="inventory_risk",
                source_step=risk_ev[0].source_step_id,
                source_tool=risk_ev[0].source_tool,
            )
        )
        f_idx += 1

    # Step 3: get_slow_moving_inventory
    slow_ev = [e for e in all_evidence if e.source_tool == "get_slow_moving_inventory"]
    if slow_ev:
        tot_row_ev = [e for e in slow_ev if e.metric == "total_rows"]
        if tot_row_ev and tot_row_ev[0].value is not None:
            stmt = f"Slow-moving inventory diagnostics identified {int(tot_row_ev[0].value):,d} catalog records with elevated holding exposure."
            evidence_used = tot_row_ev
        else:
            stmt = "Slow-moving inventory diagnostics identified catalog records with elevated holding exposure."
            evidence_used = slow_ev[:3]
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=evidence_used,
                provenance=ProvenanceType.PHASE_7A_OBSERVED,
                evidence_type=EvidenceType.DIRECT_OBSERVED,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.RISK_STATEMENT,
                supporting_metric="slow_moving_inventory",
                source_step=slow_ev[0].source_step_id,
                source_tool=slow_ev[0].source_tool,
            )
        )
        f_idx += 1

    # Step 4: get_stockout_risk
    stockout_ev = [e for e in all_evidence if e.source_tool == "get_stockout_risk"]
    if stockout_ev:
        tot_row_ev = [e for e in stockout_ev if e.metric == "total_rows"]
        if tot_row_ev and tot_row_ev[0].value is not None:
            stmt = f"Stockout risk evaluation identified {int(tot_row_ev[0].value):,d} items with accelerated runout vulnerability."
            evidence_used = tot_row_ev
        else:
            stmt = "Stockout risk evaluation identified items with accelerated runout vulnerability."
            evidence_used = stockout_ev[:3]
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=evidence_used,
                provenance=ProvenanceType.PHASE_7A_OBSERVED,
                evidence_type=EvidenceType.DIRECT_OBSERVED,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.RISK_STATEMENT,
                supporting_metric="stockout_risk",
                source_step=stockout_ev[0].source_step_id,
                source_tool=stockout_ev[0].source_tool,
            )
        )
        f_idx += 1

    headline = "Diagnostic analysis of inventory risk concentration."
    summary = (
        "Inventory evaluation indicates concurrent exposure to both slow-moving excess stock and stockout vulnerabilities. "
        "Risk concentration is distributed across distinct catalog segments without establishing a causal relationship between excess stock and stockout conditions."
    )

    confidence = evaluate_evidence_confidence(all_evidence, response.execution_result)
    limitations = list(CAUSAL_LIMITATIONS)

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.SUCCESS if findings else ExplanationStatus.INSUFFICIENT_DATA,
        explanation_type=ExplanationType.WHY_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=findings,
        supporting_evidence=all_evidence,
        risks=[
            "Slow-moving stock increases working capital tie-up and inventory holding depreciation.",
            "Items under stockout vulnerability risk unfulfilled consumer demand.",
        ],
        limitations=limitations,
        provenance=ProvenanceType.PHASE_7C_ORCHESTRATED,
        confidence=confidence,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(),
    )


# ============================================================
# 7. WHY RETURNS INCREASING EXPLANATION BUILDER
# ============================================================

def build_why_returns_explanation(response: CopilotResponse) -> BusinessExplanation:
    """Build multi-step diagnostic explanation for 'Why are returns increasing?' inquiry."""
    exp_id = _make_explanation_id(response.request_id, "WHY_RETURNS_INCREASING")
    step_results = response.execution_result.step_results if response.execution_result else []

    all_evidence: List[ExplanationEvidence] = []
    for s in step_results:
        all_evidence.extend(extract_evidence_from_step_result(s))

    findings: List[Finding] = []
    f_idx = 1

    # Step 1: get_return_summary
    ret_sum = [e for e in all_evidence if e.source_tool == "get_return_summary" and e.metric in ("return_rate", "return_count", "returns_total")]
    if ret_sum:
        ev = ret_sum[0]
        val_str = _format_metric_val(ev.value, ev.unit, ev.currency)
        stmt = f"Overall return rate was reported at {val_str}."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=[ev],
                provenance=ev.provenance,
                evidence_type=ev.evidence_type,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.OBSERVED_FACT,
                supporting_metric=ev.metric,
                source_step=ev.source_step_id,
                source_tool=ev.source_tool,
            )
        )
        f_idx += 1

    # Step 2: get_return_trend
    trend_ev = [e for e in all_evidence if e.source_tool == "get_return_trend" and e.time_range and e.value is not None]
    if len(trend_ev) >= 2:
        start_pt = trend_ev[0]
        end_pt = trend_ev[-1]
        val1 = float(start_pt.value) if isinstance(start_pt.value, (int, float)) else 0.0
        val2 = float(end_pt.value) if isinstance(end_pt.value, (int, float)) else 0.0
        dir_word = "increased" if val2 > val1 else "decreased"
        stmt = f"Return trend {dir_word} from {val1:.2f} ({start_pt.time_range}) to {val2:.2f} ({end_pt.time_range})."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=[start_pt, end_pt],
                provenance=ProvenanceType.PHASE_7A_OBSERVED,
                evidence_type=EvidenceType.DIRECT_OBSERVED,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.TREND_STATEMENT,
                supporting_metric="return_trend",
                source_step=start_pt.source_step_id,
                source_tool=start_pt.source_tool,
            )
        )
        f_idx += 1

    # Step 3: get_return_anomalies
    anom_ev = [e for e in all_evidence if e.source_tool == "get_return_anomalies"]
    if anom_ev:
        tot_row_ev = [e for e in anom_ev if e.metric == "total_rows"]
        if tot_row_ev and tot_row_ev[0].value is not None:
            stmt = f"Anomaly detection recorded {int(tot_row_ev[0].value):,d} statistical return spike signals across categories."
            evidence_used = tot_row_ev
        else:
            stmt = "Anomaly detection recorded statistical return spike signals across categories."
            evidence_used = anom_ev[:3]
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=evidence_used,
                provenance=ProvenanceType.PHASE_7A_OBSERVED,
                evidence_type=EvidenceType.DIRECT_OBSERVED,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.RISK_STATEMENT,
                supporting_metric="return_anomalies",
                source_step=anom_ev[0].source_step_id,
                source_tool=anom_ev[0].source_tool,
            )
        )
        f_idx += 1

    # Step 4: get_return_reason_breakdown
    reason_ev = [e for e in all_evidence if e.source_tool == "get_return_reason_breakdown" and e.entity]
    if reason_ev:
        stmt = "Customer return reason breakdown identifies primary observed defect and dissatisfaction codes."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=reason_ev[:3],
                provenance=ProvenanceType.PHASE_7A_OBSERVED,
                evidence_type=EvidenceType.DIRECT_OBSERVED,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.BREAKDOWN_STATEMENT,
                supporting_metric="return_reasons",
                source_step=reason_ev[0].source_step_id,
                source_tool=reason_ev[0].source_tool,
            )
        )
        f_idx += 1

    headline = "Diagnostic analysis of customer return patterns."
    summary = (
        "Return activity was analyzed across aggregate volume, temporal trend, anomaly spikes, and stated return reasons. "
        "Observed reason distributions coincide with category return spikes without proving direct singular causation."
    )

    confidence = evaluate_evidence_confidence(all_evidence, response.execution_result)
    limitations = list(RETURN_RISK_LIMITATIONS) + list(CAUSAL_LIMITATIONS)

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.SUCCESS if findings else ExplanationStatus.INSUFFICIENT_DATA,
        explanation_type=ExplanationType.WHY_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=findings,
        supporting_evidence=all_evidence,
        limitations=limitations,
        provenance=ProvenanceType.PHASE_7C_ORCHESTRATED,
        confidence=confidence,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(),
    )


# ============================================================
# 8. MULTI-DOMAIN EXPLANATION BUILDER
# ============================================================

def build_multi_domain_explanation(response: CopilotResponse) -> BusinessExplanation:
    """Build multi-domain cross-functional explanation (Sales, Financial, Inventory, Returns)."""
    exp_id = _make_explanation_id(response.request_id, ExplanationType.MULTI_DOMAIN_EXPLANATION.value)
    step_results = response.execution_result.step_results if response.execution_result else []

    all_evidence: List[ExplanationEvidence] = []
    for s in step_results:
        all_evidence.extend(extract_evidence_from_step_result(s))

    findings: List[Finding] = []
    domain_sections: Dict[str, List[str]] = {"Sales": [], "Financial": [], "Inventory": [], "Returns": []}

    f_idx = 1
    # Sales domain (get_revenue_summary / get_sales_summary)
    sales_ev = [e for e in all_evidence if "sales" in e.source_tool or e.source_tool == "get_revenue_summary"]
    if sales_ev:
        val_str = _format_metric_val(sales_ev[0].value, sales_ev[0].unit, sales_ev[0].currency)
        stmt = f"Commercial revenue domain: {sales_ev[0].metric or 'Revenue'} recorded at {val_str}."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=[sales_ev[0]],
                provenance=sales_ev[0].provenance,
                evidence_type=sales_ev[0].evidence_type,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.OBSERVED_FACT,
                supporting_metric=sales_ev[0].metric,
                source_step=sales_ev[0].source_step_id,
                source_tool=sales_ev[0].source_tool,
            )
        )
        domain_sections["Sales"].append(stmt)
        f_idx += 1

    # Financial domain (get_margin_summary)
    margin_ev = [e for e in all_evidence if "margin" in e.source_tool]
    if margin_ev:
        val_str = _format_metric_val(margin_ev[0].value, margin_ev[0].unit, margin_ev[0].currency)
        stmt = f"Financial margin domain: {margin_ev[0].metric or 'Margin'} recorded at {val_str}."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=[margin_ev[0]],
                provenance=margin_ev[0].provenance,
                evidence_type=margin_ev[0].evidence_type,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.OBSERVED_FACT,
                supporting_metric=margin_ev[0].metric,
                source_step=margin_ev[0].source_step_id,
                source_tool=margin_ev[0].source_tool,
            )
        )
        domain_sections["Financial"].append(stmt)
        f_idx += 1

    # Inventory domain (get_inventory_summary)
    inv_ev = [e for e in all_evidence if "inventory" in e.source_tool]
    if inv_ev:
        val_str = _format_metric_val(inv_ev[0].value, inv_ev[0].unit, inv_ev[0].currency)
        stmt = f"Inventory position domain: {inv_ev[0].metric or 'Inventory'} recorded at {val_str}."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=[inv_ev[0]],
                provenance=inv_ev[0].provenance,
                evidence_type=inv_ev[0].evidence_type,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.OBSERVED_FACT,
                supporting_metric=inv_ev[0].metric,
                source_step=inv_ev[0].source_step_id,
                source_tool=inv_ev[0].source_tool,
            )
        )
        domain_sections["Inventory"].append(stmt)
        f_idx += 1

    # Returns domain (get_return_summary)
    ret_ev = [e for e in all_evidence if "return" in e.source_tool]
    if ret_ev:
        val_str = _format_metric_val(ret_ev[0].value, ret_ev[0].unit, ret_ev[0].currency)
        stmt = f"Customer returns domain: {ret_ev[0].metric or 'Returns'} recorded at {val_str}."
        findings.append(
            Finding(
                finding_id=_make_finding_id(exp_id, f_idx),
                statement=stmt,
                evidence=[ret_ev[0]],
                provenance=ret_ev[0].provenance,
                evidence_type=ret_ev[0].evidence_type,
                confidence=ExplanationConfidence.HIGH,
                claim_category=ClaimCategory.OBSERVED_FACT,
                supporting_metric=ret_ev[0].metric,
                source_step=ret_ev[0].source_step_id,
                source_tool=ret_ev[0].source_tool,
            )
        )
        domain_sections["Returns"].append(stmt)
        f_idx += 1

    headline = "Multi-domain commercial performance briefing."
    summary = (
        "Integrated cross-functional summary across Sales, Financial, Inventory, and Returns domains. "
        "Observations reflect verified signals across independent upstream systems without inferring cross-domain causal links."
    )

    confidence = evaluate_evidence_confidence(all_evidence, response.execution_result)
    limitations = list(FINANCIAL_LIMITATIONS) + list(CAUSAL_LIMITATIONS)

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.SUCCESS if findings else ExplanationStatus.INSUFFICIENT_DATA,
        explanation_type=ExplanationType.MULTI_DOMAIN_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=findings,
        supporting_evidence=all_evidence,
        limitations=limitations,
        provenance=ProvenanceType.PHASE_7C_ORCHESTRATED,
        confidence=confidence,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(),
    )


# ============================================================
# 9. FORECAST EXPLANATION BUILDER
# ============================================================

def build_forecast_explanation(response: CopilotResponse) -> BusinessExplanation:
    """Build deterministic explanation for forward-looking demand forecast inquiries."""
    exp_id = _make_explanation_id(response.request_id, ExplanationType.FORECAST_EXPLANATION.value)
    step_results = response.execution_result.step_results if response.execution_result else []

    all_evidence: List[ExplanationEvidence] = []
    for s in step_results:
        all_evidence.extend(extract_evidence_from_step_result(s))

    findings: List[Finding] = []
    f_idx = 1
    for ev in all_evidence:
        if ev.value is not None:
            val_str = _format_metric_val(ev.value, ev.unit, ev.currency)
            m_name = (ev.metric or "Demand").replace("_", " ").title()
            time_phrase = f" for {ev.time_range}" if ev.time_range else ""
            stmt = f"The forecast model projects {m_name} at {val_str}{time_phrase}."

            findings.append(
                Finding(
                    finding_id=_make_finding_id(exp_id, f_idx),
                    statement=stmt,
                    evidence=[ev],
                    provenance=ProvenanceType.MODEL_BASED,
                    evidence_type=EvidenceType.MODEL_PROJECTED,
                    confidence=ExplanationConfidence.MEDIUM,
                    claim_category=ClaimCategory.FORECAST_STATEMENT,
                    supporting_metric=ev.metric,
                    source_step=ev.source_step_id,
                    source_tool=ev.source_tool,
                )
            )
            f_idx += 1

    headline = "Demand forecast projection summary."
    summary = "Forward-looking demand projections generated by statistical and machine learning models. Projections represent expected baseline demand."

    confidence = evaluate_evidence_confidence(all_evidence, response.execution_result)

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.SUCCESS if findings else ExplanationStatus.INSUFFICIENT_DATA,
        explanation_type=ExplanationType.FORECAST_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=findings,
        supporting_evidence=all_evidence,
        limitations=FORECAST_LIMITATIONS,
        provenance=ProvenanceType.MODEL_BASED,
        confidence=confidence,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(),
    )


# ============================================================
# 10. OPERATIONAL REVIEW EXPLANATION BUILDER
# ============================================================

def build_operational_review_explanation(response: CopilotResponse) -> BusinessExplanation:
    """Build deterministic explanation for governance review inquiries (Replenishment, PO, Rebalancing, Recs, Decisions)."""
    exp_id = _make_explanation_id(response.request_id, ExplanationType.OPERATIONAL_REVIEW_EXPLANATION.value)
    step_results = response.execution_result.step_results if response.execution_result else []

    all_evidence: List[ExplanationEvidence] = []
    for s in step_results:
        all_evidence.extend(extract_evidence_from_step_result(s))

    findings: List[Finding] = []
    table_ev = [e for e in all_evidence if e.metric == "total_rows"]
    row_count = int(table_ev[0].value) if table_ev and table_ev[0].value is not None else 0

    stmt = f"Operational intelligence engine generated {row_count} candidate review items pending human authorization."
    findings.append(
        Finding(
            finding_id=_make_finding_id(exp_id, 1),
            statement=stmt,
            evidence=table_ev if table_ev else all_evidence[:1],
            provenance=ProvenanceType.PHASE_7A_OBSERVED,
            evidence_type=EvidenceType.DIRECT_OBSERVED,
            confidence=ExplanationConfidence.HIGH,
            claim_category=ClaimCategory.GOVERNANCE_NOTE,
            supporting_metric="review_items_count",
            source_step=all_evidence[0].source_step_id if all_evidence else None,
            source_tool=all_evidence[0].source_tool if all_evidence else None,
        )
    )

    headline = "Operational review items pending authorization."
    summary = (
        f"A total of {row_count} operational proposals or decision packages are currently in draft status. "
        "In accordance with platform governance, all proposals require explicit human review and approval prior to execution."
    )

    confidence = evaluate_evidence_confidence(all_evidence, response.execution_result)

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.SUCCESS,
        explanation_type=ExplanationType.OPERATIONAL_REVIEW_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=findings,
        supporting_evidence=all_evidence,
        limitations=OPERATIONAL_GOVERNANCE_NOTES,
        provenance=ProvenanceType.PHASE_7A_OBSERVED,
        confidence=confidence,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(approval_required=True, execution_allowed=False),
    )


# ============================================================
# 11. INSUFFICIENT DATA & UNAVAILABLE BUILDERS
# ============================================================

def build_insufficient_data_explanation(response: CopilotResponse, reason: Optional[str] = None) -> BusinessExplanation:
    """Build honest, non-fabricating explanation when evidence is insufficient."""
    exp_id = _make_explanation_id(response.request_id, ExplanationType.INSUFFICIENT_DATA_EXPLANATION.value)
    headline = "Insufficient data to evaluate inquiry."
    msg = reason or "Required underlying business metrics or evidence records were unavailable in the query response."
    summary = f"Analytical evaluation cannot be completed: {msg}"

    finding = Finding(
        finding_id=_make_finding_id(exp_id, 1),
        statement=f"Data insufficiency detected: {msg}",
        evidence=[],
        provenance=ProvenanceType.INSUFFICIENT_DATA,
        evidence_type=EvidenceType.DIRECT_OBSERVED,
        confidence=ExplanationConfidence.INSUFFICIENT,
        claim_category=ClaimCategory.INSUFFICIENCY_NOTE,
    )

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.INSUFFICIENT_DATA,
        explanation_type=ExplanationType.INSUFFICIENT_DATA_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=[finding],
        supporting_evidence=[],
        limitations=["Analytical calculation was halted to prevent fabricating ungrounded business metrics."],
        provenance=ProvenanceType.INSUFFICIENT_DATA,
        confidence=ExplanationConfidence.INSUFFICIENT,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(),
    )


def build_unavailable_explanation(response: CopilotResponse) -> BusinessExplanation:
    """Build explanation when requested capability is not supported or unavailable."""
    exp_id = _make_explanation_id(response.request_id, ExplanationType.UNAVAILABLE_EXPLANATION.value)
    headline = "Requested capability is currently unavailable."
    summary = (
        "The requested analytical capability is recognized in the business taxonomy but does not currently "
        "have an active Phase 7A tool implementation."
    )

    finding = Finding(
        finding_id=_make_finding_id(exp_id, 1),
        statement="Capability currently lacks an executable query tool implementation.",
        evidence=[],
        provenance=ProvenanceType.INSUFFICIENT_DATA,
        evidence_type=EvidenceType.DIRECT_OBSERVED,
        confidence=ExplanationConfidence.INSUFFICIENT,
        claim_category=ClaimCategory.INSUFFICIENCY_NOTE,
    )

    return BusinessExplanation(
        explanation_id=exp_id,
        status=ExplanationStatus.UNAVAILABLE,
        explanation_type=ExplanationType.UNAVAILABLE_EXPLANATION,
        headline=headline,
        summary=summary,
        key_findings=[finding],
        supporting_evidence=[],
        limitations=["Tool execution was not dispatched for this unsupported capability."],
        provenance=ProvenanceType.INSUFFICIENT_DATA,
        confidence=ExplanationConfidence.INSUFFICIENT,
        generated_from={"request_id": response.request_id, "question": response.normalized_question},
        governance=ExplanationGovernance(),
    )
