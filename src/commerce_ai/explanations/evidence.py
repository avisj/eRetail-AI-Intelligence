"""Evidence Extraction and Lineage Preservation for Business Explanations (Phase 7D).

Extracts strongly typed ExplanationEvidence items from CopilotStepResult and
QueryResponse envelopes while preserving upstream IDs, source engines, and pedigree.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

from commerce_ai.copilot.schemas import CopilotStepResult
from commerce_ai.explanations.enums import (
    EvidenceType,
    ProvenanceType,
)
from commerce_ai.explanations.schemas import ExplanationEvidence
from commerce_ai.query_layer.schemas import QueryResponse


def _deterministic_evidence_id(step_id: str, tool_name: str, item_key: str) -> str:
    """Generate deterministic evidence identifier."""
    raw = f"{step_id}:{tool_name}:{item_key}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"EVI-{step_id}-{digest}"


def _infer_provenance_and_evidence_type(
    tool_name: str,
    source_engine: str,
    is_forecast: bool = False,
    is_prediction: bool = False,
    is_derived: bool = False,
) -> tuple[ProvenanceType, EvidenceType]:
    """Infer provenance and empirical nature from tool and engine metadata."""
    if "forecast" in tool_name or is_forecast:
        return ProvenanceType.MODEL_BASED, EvidenceType.MODEL_PROJECTED
    if "prediction" in tool_name or "risk" in tool_name and "return" in tool_name or is_prediction:
        return ProvenanceType.MODEL_BASED, EvidenceType.STATISTICAL_DERIVED
    if is_derived or "drivers" in tool_name or "attribution" in tool_name or "economics" in tool_name:
        return ProvenanceType.PHASE_7B_DERIVED, EvidenceType.DETERMINISTIC_DERIVED
    return ProvenanceType.PHASE_7A_OBSERVED, EvidenceType.DIRECT_OBSERVED


def extract_evidence_from_step_result(step_result: CopilotStepResult) -> List[ExplanationEvidence]:
    """Extract granular ExplanationEvidence records from a single CopilotStepResult."""
    evidence_items: List[ExplanationEvidence] = []
    step_id = step_result.step_id
    tool_name = step_result.tool_name

    response: Optional[QueryResponse] = step_result.response
    if response is None:
        # Create an insufficiency evidence record for unexecuted or failed steps
        evidence_items.append(
            ExplanationEvidence(
                evidence_id=_deterministic_evidence_id(step_id, tool_name, "insufficient"),
                source_step_id=step_id,
                source_tool=tool_name,
                source_engine="commerce_ai.query_layer",
                metric=None,
                value=None,
                provenance=ProvenanceType.INSUFFICIENT_DATA,
                evidence_type=EvidenceType.DIRECT_OBSERVED,
                notes=f"Step {step_id} ({tool_name}) produced no response envelope.",
            )
        )
        return evidence_items

    source_engine = response.metadata.source_engine if response.metadata else "commerce_ai.query_layer"
    as_of_date = response.metadata.generated_as_of if response.metadata else None
    currency = response.metadata.currency if response.metadata else None

    # 1. Extract Metrics
    if response.metrics:
        for idx, m in enumerate(response.metrics):
            is_derived = m.percentage_change is not None or m.absolute_change is not None
            is_forecast = "forecast" in tool_name or "forecast" in m.metric_name
            prov, ev_type = _infer_provenance_and_evidence_type(
                tool_name, source_engine, is_forecast=is_forecast, is_derived=is_derived
            )
            evidence_items.append(
                ExplanationEvidence(
                    evidence_id=_deterministic_evidence_id(step_id, tool_name, f"metric_{m.metric_name}_{idx}"),
                    source_step_id=step_id,
                    source_tool=tool_name,
                    source_engine=source_engine,
                    metric=m.metric_name,
                    value=m.value,
                    comparison_value=m.previous_period_value,
                    unit=m.unit or None,
                    currency=m.currency or currency,
                    time_range=m.period or as_of_date,
                    provenance=prov,
                    evidence_type=ev_type,
                    notes=f"Display name: {m.display_name}. Status: {m.status}.",
                )
            )

    # 2. Extract Time Series
    if response.time_series:
        ts = response.time_series
        is_forecast = "forecast" in tool_name
        prov, ev_type = _infer_provenance_and_evidence_type(tool_name, source_engine, is_forecast=is_forecast)
        for idx, pt in enumerate(ts.points):
            evidence_items.append(
                ExplanationEvidence(
                    evidence_id=_deterministic_evidence_id(step_id, tool_name, f"ts_{pt.date}_{idx}"),
                    source_step_id=step_id,
                    source_tool=tool_name,
                    source_engine=source_engine,
                    metric=pt.metric_name,
                    value=pt.value,
                    unit=ts.unit or None,
                    currency=pt.currency or ts.currency or currency,
                    time_range=pt.date,
                    provenance=prov,
                    evidence_type=ev_type,
                    notes=f"Time-series grain: {ts.time_grain}.",
                )
            )

    # 3. Extract Breakdowns
    if response.breakdown:
        bk = response.breakdown
        prov, ev_type = _infer_provenance_and_evidence_type(tool_name, source_engine)
        for idx, item in enumerate(bk.items):
            evidence_items.append(
                ExplanationEvidence(
                    evidence_id=_deterministic_evidence_id(step_id, tool_name, f"bk_{item.dimension_value}_{idx}"),
                    source_step_id=step_id,
                    source_tool=tool_name,
                    source_engine=source_engine,
                    metric=item.metric_name,
                    value=item.metric_value,
                    comparison_value=item.percentage_of_total,
                    unit=bk.unit or item.currency or currency,
                    currency=item.currency or bk.currency or currency,
                    dimension=item.dimension_name,
                    entity=item.dimension_value,
                    time_range=as_of_date,
                    provenance=prov,
                    evidence_type=ev_type,
                    notes=f"Share of total: {item.percentage_of_total}%.",
                )
            )

    # 4. Extract Tables
    if response.table:
        tbl = response.table
        prov, ev_type = _infer_provenance_and_evidence_type(tool_name, source_engine)
        # Record table summary evidence
        evidence_items.append(
            ExplanationEvidence(
                evidence_id=_deterministic_evidence_id(step_id, tool_name, "table_summary"),
                source_step_id=step_id,
                source_tool=tool_name,
                source_engine=source_engine,
                metric="total_rows",
                value=tbl.total_rows,
                unit="rows",
                currency=None,
                time_range=as_of_date,
                provenance=prov,
                evidence_type=ev_type,
                notes=f"Table columns: {', '.join(tbl.columns)}.",
            )
        )
        # Record individual rows as evidence items
        for r_idx, row in enumerate(tbl.rows):
            entity_val = str(row.get("segment") or row.get("sku_id") or row.get("warehouse_id") or row.get("recommendation_id") or row.get("decision_id") or f"row_{r_idx}")
            num_cols = [k for k, v in row.items() if isinstance(v, (int, float))]
            if num_cols:
                for col in num_cols:
                    val = row[col]
                    evidence_items.append(
                        ExplanationEvidence(
                            evidence_id=_deterministic_evidence_id(step_id, tool_name, f"row_{entity_val}_{col}_{r_idx}"),
                            source_step_id=step_id,
                            source_tool=tool_name,
                            source_engine=source_engine,
                            metric=col,
                            value=val,
                            unit=None,
                            currency=currency if any(w in col for w in ["profit", "revenue", "cost"]) else None,
                            dimension="segment" if "segment" in row else "row",
                            entity=entity_val,
                            time_range=as_of_date,
                            provenance=prov,
                            evidence_type=ev_type,
                            notes=str(row),
                        )
                    )
            else:
                evidence_items.append(
                    ExplanationEvidence(
                        evidence_id=_deterministic_evidence_id(step_id, tool_name, f"row_{entity_val}_{r_idx}"),
                        source_step_id=step_id,
                        source_tool=tool_name,
                        source_engine=source_engine,
                        metric="row_data",
                        value=None,
                        dimension="row",
                        entity=entity_val,
                        time_range=as_of_date,
                        provenance=prov,
                        evidence_type=ev_type,
                        notes=str(row),
                    )
                )

    # 5. Extract Insights
    if response.insights:
        for idx, ins in enumerate(response.insights):
            evidence_items.append(
                ExplanationEvidence(
                    evidence_id=_deterministic_evidence_id(step_id, tool_name, f"insight_{ins.insight_id}_{idx}"),
                    source_step_id=step_id,
                    source_tool=tool_name,
                    source_engine=source_engine,
                    metric=ins.category,
                    value=ins.impact_value,
                    currency=ins.currency or currency,
                    time_range=as_of_date,
                    provenance=ProvenanceType.PHASE_7A_OBSERVED,
                    evidence_type=EvidenceType.DIRECT_OBSERVED,
                    notes=f"{ins.title}: {ins.summary} (Severity: {ins.severity})",
                )
            )

    return evidence_items
