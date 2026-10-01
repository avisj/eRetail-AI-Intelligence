"""Result Assembly & Evidence Lineage for Copilot (Phase 7C).

Assembles individual step execution results into composite CopilotExecutionResult
objects, preserving complete evidence chains, provenance, execution timings,
and partial failure diagnostics.
"""

from __future__ import annotations

import hashlib
from typing import List

from commerce_ai.copilot.enums import PlanExecutionStatus, StepStatus
from commerce_ai.copilot.schemas import (
    CopilotEvidence,
    CopilotExecutionResult,
    CopilotFailure,
    CopilotStepResult,
)


def assemble_execution_result(
    request_id: str,
    step_results: List[CopilotStepResult],
    total_time_ms: float,
) -> CopilotExecutionResult:
    """Aggregate individual step results into an overall CopilotExecutionResult."""
    rid = "RES-" + hashlib.sha256((request_id + str(len(step_results))).encode("utf-8")).hexdigest()[:16]

    evidence_chain: List[CopilotEvidence] = []
    failures: List[CopilotFailure] = []

    completed_count = 0
    failed_count = 0
    unavailable_count = 0
    skipped_count = 0

    for sr in step_results:
        if sr.evidence:
            evidence_chain.append(sr.evidence)
        if sr.failure:
            failures.append(sr.failure)

        if sr.status == StepStatus.COMPLETED:
            completed_count += 1
        elif sr.status == StepStatus.FAILED:
            failed_count += 1
        elif sr.status == StepStatus.UNAVAILABLE:
            unavailable_count += 1
        elif sr.status in {StepStatus.SKIPPED, StepStatus.BLOCKED}:
            skipped_count += 1

    total_steps = len(step_results)

    if total_steps == 0:
        overall_status = PlanExecutionStatus.UNAVAILABLE
    elif unavailable_count == total_steps:
        overall_status = PlanExecutionStatus.UNAVAILABLE
    elif completed_count == total_steps:
        overall_status = PlanExecutionStatus.COMPLETE
    elif completed_count > 0:
        overall_status = PlanExecutionStatus.PARTIAL
    else:
        overall_status = PlanExecutionStatus.FAILED

    return CopilotExecutionResult(
        result_id=rid,
        request_id=request_id,
        status=overall_status,
        step_results=step_results,
        evidence_chain=evidence_chain,
        failures=failures,
        total_execution_time_ms=total_time_ms,
    )
