"""Unit tests for Step Dependency Graph & Execution Sequencing (Phase 7C).

Tests:
- Identification of ready steps
- Blocking of dependent steps until prerequisites complete
- Failure propagation and marking of skipped/blocked steps
- Fan-out / multi-branch dependency graphs
- Independent parallelizable steps
"""

from __future__ import annotations

import pytest

from commerce_ai.copilot.dependencies import DependencyGraph
from commerce_ai.copilot.enums import StepStatus
from commerce_ai.copilot.schemas import CopilotStep


def _make_step(step_id: str, tool_name: str, deps: list[str]) -> CopilotStep:
    return CopilotStep(
        step_id=step_id,
        sequence=int(step_id.split("-")[1]),
        contract_id="QRY-test",
        plan_id="PLAN-test",
        tool_name=tool_name,
        arguments={},
        purpose=f"Execute {tool_name}",
        required=True,
        dependencies=deps,
        status=StepStatus.PENDING,
    )


class TestDependencyGraph:
    def test_independent_steps_all_ready(self):
        s1 = _make_step("STEP-1", "get_sales_summary", [])
        s2 = _make_step("STEP-2", "get_inventory_summary", [])
        graph = DependencyGraph([s1, s2])

        ready = graph.get_ready_steps(set())
        assert len(ready) == 2
        ready_ids = {s.step_id for s in ready}
        assert "STEP-1" in ready_ids
        assert "STEP-2" in ready_ids

    def test_sequential_dependency_chain(self):
        s1 = _make_step("STEP-1", "get_margin_summary", [])
        s2 = _make_step("STEP-2", "get_margin_drivers", ["get_margin_summary"])
        graph = DependencyGraph([s1, s2])

        # Initially only STEP-1 is ready
        ready1 = graph.get_ready_steps(set())
        assert len(ready1) == 1
        assert ready1[0].step_id == "STEP-1"

        # After STEP-1 completes, STEP-2 becomes ready
        ready2 = graph.get_ready_steps({"STEP-1"})
        assert len(ready2) == 1
        assert ready2[0].step_id == "STEP-2"

    def test_fan_out_dependency(self):
        # STEP-1 summary -> STEP-2, STEP-3, STEP-4 all depend on STEP-1
        s1 = _make_step("STEP-1", "get_margin_summary", [])
        s2 = _make_step("STEP-2", "get_margin_drivers", ["get_margin_summary"])
        s3 = _make_step("STEP-3", "get_sales_by_channel", ["get_margin_summary"])
        s4 = _make_step("STEP-4", "get_sales_by_warehouse", ["get_margin_summary"])
        graph = DependencyGraph([s1, s2, s3, s4])

        ready_init = graph.get_ready_steps(set())
        assert len(ready_init) == 1
        assert ready_init[0].step_id == "STEP-1"

        ready_after_s1 = graph.get_ready_steps({"STEP-1"})
        assert len(ready_after_s1) == 3
        ids = {s.step_id for s in ready_after_s1}
        assert ids == {"STEP-2", "STEP-3", "STEP-4"}

    def test_failure_propagation_marks_blocked(self):
        s1 = _make_step("STEP-1", "get_margin_summary", [])
        s2 = _make_step("STEP-2", "get_margin_drivers", ["get_margin_summary"])
        s3 = _make_step("STEP-3", "get_sales_by_channel", ["get_margin_summary"])
        graph = DependencyGraph([s1, s2, s3])

        # If STEP-1 fails, STEP-2 and STEP-3 are blocked
        blocked = graph.mark_blocked_or_skipped({"STEP-1"})
        assert set(blocked) == {"STEP-2", "STEP-3"}

    def test_step_id_direct_reference_in_prereq(self):
        s1 = _make_step("STEP-1", "tool_a", [])
        s2 = _make_step("STEP-2", "tool_b", ["STEP-1"])
        graph = DependencyGraph([s1, s2])

        ready = graph.get_ready_steps({"STEP-1"})
        assert len(ready) == 1
        assert ready[0].step_id == "STEP-2"
