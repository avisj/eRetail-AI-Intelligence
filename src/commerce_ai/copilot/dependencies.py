"""Dependency Management for Copilot Reasoning Steps (Phase 7C).

Coordinates step dependency graphs:
- Identifies prerequisite steps and execution sequencing
- Determines when dependent steps become READY vs BLOCKED
- Tracks parallelizable independent steps
- Propagates failure states so dependent steps are gracefully SKIPPED or BLOCKED
"""

from __future__ import annotations

from typing import Dict, List, Set

from commerce_ai.copilot.enums import StepStatus
from commerce_ai.copilot.schemas import CopilotStep


class DependencyGraph:
    """Directed dependency graph for reasoning and execution steps."""

    def __init__(self, steps: List[CopilotStep]) -> None:
        self.steps: Dict[str, CopilotStep] = {s.step_id: s for s in steps}
        # Mapping from step_id to list of prerequisite step_ids
        self.prerequisites: Dict[str, List[str]] = {s.step_id: list(s.dependencies) for s in steps}
        # Mapping from tool_name to step_id for cross-referencing plan tool dependencies
        self.tool_to_step_id: Dict[str, str] = {s.tool_name: s.step_id for s in steps}

    def get_ready_steps(self, completed_steps: Set[str]) -> List[CopilotStep]:
        """Return steps whose prerequisites are fully completed and are not already completed or failed."""
        ready: List[CopilotStep] = []
        for step_id, step in self.steps.items():
            if step_id in completed_steps:
                continue
            if step.status not in {StepStatus.PENDING, StepStatus.READY}:
                continue
            prereqs = self.prerequisites.get(step_id, [])
            # Prereqs can be specified either by step_id or by prerequisite tool_name
            prereqs_met = True
            for p in prereqs:
                resolved_id = self.tool_to_step_id.get(p, p)
                if resolved_id not in completed_steps:
                    prereqs_met = False
                    break
            if prereqs_met:
                ready.append(step)
        return ready

    def mark_blocked_or_skipped(self, failed_steps: Set[str]) -> List[str]:
        """Identify and return IDs of steps that are blocked because a prerequisite step failed."""
        blocked_ids: List[str] = []
        for step_id, step in self.steps.items():
            if step.status in {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.UNAVAILABLE}:
                continue
            prereqs = self.prerequisites.get(step_id, [])
            for p in prereqs:
                resolved_id = self.tool_to_step_id.get(p, p)
                if resolved_id in failed_steps:
                    blocked_ids.append(step_id)
                    break
        return blocked_ids
