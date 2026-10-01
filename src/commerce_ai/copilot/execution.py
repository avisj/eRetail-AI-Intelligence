"""Step Execution Coordinator against Phase 7A Query Tools (Phase 7C).

Executes reasoning steps generated from Phase 7B plans strictly through the
Phase 7A QueryLayerService. Enforces dependency sequencing, failure isolation,
read-only governance, and evidence lineage.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Set

from commerce_ai.copilot.dependencies import DependencyGraph
from commerce_ai.copilot.enums import (
    CopilotProvenanceSource,
    CopilotState,
    PlanExecutionStatus,
    StepStatus,
)
from commerce_ai.copilot.governance import (
    enforce_decision_package_neutrality,
    validate_plan_governance,
)
from commerce_ai.copilot.results import assemble_execution_result
from commerce_ai.copilot.schemas import (
    CopilotEvidence,
    CopilotExecutionPlan,
    CopilotExecutionResult,
    CopilotFailure,
    CopilotStep,
    CopilotStepResult,
)
from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.schemas import QueryResponse
from commerce_ai.query_layer.service import QueryLayerService


def _build_query_context(args: Dict[str, Any]) -> QueryContext:
    """Build strongly validated QueryContext from step arguments."""
    ctx_kwargs: Dict[str, Any] = {}
    for field in [
        "as_of_date", "start_date", "end_date", "currency",
        "sku_id", "warehouse_id", "channel_id", "category_id",
        "brand", "supplier_id", "velocity_tier", "abc_class", "xyz_class",
        "time_grain", "limit", "offset",
    ]:
        if field in args and args[field] is not None:
            ctx_kwargs[field] = args[field]

    return QueryContext(**ctx_kwargs)


def _extract_evidence(step_id: str, tool_name: str, response: QueryResponse) -> CopilotEvidence:
    """Extract structured evidence package from Phase 7A tool response envelope."""
    # Determine metric names present
    metrics: List[str] = [m.metric_name for m in response.metrics]
    if response.time_series and response.time_series.metric_name not in metrics:
        metrics.append(response.time_series.metric_name)
    if response.breakdown and response.breakdown.metric_name not in metrics:
        metrics.append(response.breakdown.metric_name)

    entity_count = 0
    if response.table:
        entity_count = response.table.total_rows or len(response.table.rows)
    elif response.breakdown:
        entity_count = len(response.breakdown.items)
    elif response.time_series:
        entity_count = len(response.time_series.points)
    elif response.metrics:
        entity_count = len(response.metrics)
    else:
        entity_count = 1

    summary: Dict[str, Any] = {}
    for m in response.metrics:
        summary[m.metric_name] = m.value
    if response.breakdown:
        summary["breakdown_dimension"] = response.breakdown.dimension_name
        summary["breakdown_total"] = response.breakdown.total_value
        summary["breakdown_items_count"] = len(response.breakdown.items)
    if response.time_series:
        summary["time_series_metric"] = response.time_series.metric_name
        summary["time_series_points_count"] = len(response.time_series.points)
    if response.table:
        summary["table_total_rows"] = response.table.total_rows
        summary["table_columns"] = response.table.columns
    if response.insights:
        summary["insights_count"] = len(response.insights)

    # Enforce neutrality on any decision packages
    enforce_decision_package_neutrality(summary)

    source_engine = response.metadata.source_engine if response.metadata else "commerce_ai.query_layer"
    as_of_date = response.metadata.generated_as_of if response.metadata else None
    currency = response.metadata.currency if response.metadata else None

    eid = "EVI-" + hashlib.sha256((step_id + tool_name + response.query_id).encode("utf-8")).hexdigest()[:16]

    return CopilotEvidence(
        evidence_id=eid,
        step_id=step_id,
        tool_name=tool_name,
        source_engine=source_engine,
        metrics=metrics[:10],
        entity_count=entity_count,
        as_of_date=as_of_date,
        currency=currency,
        data_summary=summary,
        provenance=CopilotProvenanceSource.PHASE_7A_TOOL,
    )


def execute_plan(
    plan: CopilotExecutionPlan,
    query_layer: Optional[QueryLayerService] = None,
) -> CopilotExecutionResult:
    """Execute all reasoning steps in a CopilotExecutionPlan against the Phase 7A query layer.

    Respects dependency order, isolates individual step failures, extracts evidence,
    and returns a structured CopilotExecutionResult.
    """
    t0 = time.perf_counter()

    # 1. Validate plan governance before executing
    is_gov_valid, gov_violations = validate_plan_governance(plan)
    if not is_gov_valid:
        t_ms = (time.perf_counter() - t0) * 1000.0
        failures = [
            CopilotFailure(
                step_id="GOV-0",
                tool_name="governance_check",
                error_code="GOVERNANCE_VIOLATION",
                error_message=v,
                fatal=True,
            )
            for v in gov_violations
        ]
        return CopilotExecutionResult(
            result_id="RES-governance-violation",
            request_id=plan.request_id,
            status=PlanExecutionStatus.FAILED,
            step_results=[],
            evidence_chain=[],
            failures=failures,
            total_execution_time_ms=t_ms,
        )

    # 2. Check if plan is for an unavailable capability (e.g. DATA_QUALITY_ANALYSIS)
    if plan.state == CopilotState.UNAVAILABLE or len(plan.steps) == 0:
        t_ms = (time.perf_counter() - t0) * 1000.0
        return CopilotExecutionResult(
            result_id="RES-unavailable",
            request_id=plan.request_id,
            status=PlanExecutionStatus.UNAVAILABLE,
            step_results=[],
            evidence_chain=[],
            failures=[],
            total_execution_time_ms=t_ms,
        )

    ql = query_layer or QueryLayerService()
    dep_graph = DependencyGraph(plan.steps)

    completed_step_ids: Set[str] = set()
    failed_step_ids: Set[str] = set()
    step_results: List[CopilotStepResult] = []

    # Map to access steps
    steps_by_id = {s.step_id: s for s in plan.steps}
    executed_ids: Set[str] = set()

    # Loop until all steps are resolved
    while len(executed_ids) < len(plan.steps):
        ready_steps = dep_graph.get_ready_steps(completed_step_ids)
        # Filter ready steps that haven't executed yet
        ready_to_run = [s for s in ready_steps if s.step_id not in executed_ids]

        if not ready_to_run:
            # Check for blocked steps due to failures
            blocked_ids = dep_graph.mark_blocked_or_skipped(failed_step_ids)
            for b_id in blocked_ids:
                if b_id not in executed_ids:
                    executed_ids.add(b_id)
                    b_step = steps_by_id[b_id]
                    fail = CopilotFailure(
                        step_id=b_id,
                        tool_name=b_step.tool_name,
                        error_code="STEP_DEPENDENCY_BLOCKED",
                        error_message="Step was blocked because a required prerequisite step failed.",
                        fatal=False,
                    )
                    step_results.append(
                        CopilotStepResult(
                            step_id=b_id,
                            tool_name=b_step.tool_name,
                            status=StepStatus.BLOCKED,
                            execution_time_ms=0.0,
                            failure=fail,
                        )
                    )
            # Break if no forward progress can be made
            remaining = set(steps_by_id.keys()) - executed_ids
            for r_id in remaining:
                executed_ids.add(r_id)
                r_step = steps_by_id[r_id]
                step_results.append(
                    CopilotStepResult(
                        step_id=r_id,
                        tool_name=r_step.tool_name,
                        status=StepStatus.SKIPPED,
                        execution_time_ms=0.0,
                    )
                )
            break

        # Execute ready steps
        for step in ready_to_run:
            executed_ids.add(step.step_id)
            step_t0 = time.perf_counter()
            try:
                ctx = _build_query_context(step.arguments)
                # QueryLayerService execute_tool invocation
                response = ql.execute_tool(step.tool_name, context=ctx)
                step_t_ms = (time.perf_counter() - step_t0) * 1000.0

                evidence = _extract_evidence(step.step_id, step.tool_name, response)
                step_results.append(
                    CopilotStepResult(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        status=StepStatus.COMPLETED,
                        response=response,
                        evidence=evidence,
                        execution_time_ms=step_t_ms,
                    )
                )
                completed_step_ids.add(step.step_id)

            except Exception as exc:
                step_t_ms = (time.perf_counter() - step_t0) * 1000.0
                fail = CopilotFailure(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    error_code="TOOL_EXECUTION_FAILURE",
                    error_message=str(exc),
                    fatal=step.required,
                )
                step_results.append(
                    CopilotStepResult(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        status=StepStatus.FAILED,
                        execution_time_ms=step_t_ms,
                        failure=fail,
                    )
                )
                failed_step_ids.add(step.step_id)

    total_time_ms = (time.perf_counter() - t0) * 1000.0
    return assemble_execution_result(plan.request_id, step_results, total_time_ms)
