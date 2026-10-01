"""Regression & Semantic Reconciliation Tests for Phase 7C (Phase 7A/7B Alignment).

Verifies:
1. Every executable Phase 7C tool name exists in the Phase 7A canonical registry.
2. No invalid or invented tool names (e.g. get_margin_status, get_inventory_balance,
   get_known_contribution_margin, get_return_reasons) are ever generated.
3. Phase 7B contract and planner layers remain authoritative for tool mapping.
4. Unknown business questions do NOT silently fall back to SALES_PERFORMANCE.
5. "Why did margin decline?" uses the exact approved canonical Phase 7A tools.
6. "Why is inventory risk high?" uses the exact approved canonical Phase 7A tools.
7. "Why are returns increasing?" uses the exact approved canonical Phase 7A tools.
8. Multi-domain cross-functional orchestration uses canonical Phase 7A tools.
9. Aliases resolve to canonical targets or are registered Phase 7A aliases.
10. Phase 7C cannot bypass Phase 7B validation.
"""

from __future__ import annotations

import pytest

from commerce_ai.copilot.decomposition import decompose_and_plan
from commerce_ai.copilot.enums import CopilotState, StepStatus
from commerce_ai.copilot.interpreter import interpret_question
from commerce_ai.copilot.schemas import CopilotRequest
from commerce_ai.copilot.service import CopilotService
from commerce_ai.query_contracts.enums import BusinessDomain, BusinessGrain, QueryIntent
from commerce_ai.query_contracts.service import QueryContractService
from commerce_ai.query_contracts.templates import CANONICAL_TEMPLATES
from commerce_ai.query_contracts.tool_mapping import (
    PHASE_7A_APPROVED_ALIASES,
    PHASE_7A_CANONICAL_TOOLS,
    PHASE_7A_VALID_TOOLS,
)
from commerce_ai.query_layer.registry import default_registry, register_standard_tools


class TestCopilotRegistryReconciliation:
    @pytest.fixture(autouse=True)
    def setup_registry(self):
        register_standard_tools(default_registry)

    @pytest.fixture
    def service(self) -> CopilotService:
        return CopilotService()

    @pytest.fixture
    def contract_service(self) -> QueryContractService:
        return QueryContractService()

    def test_all_canonical_templates_produce_valid_registered_tools(
        self,
        contract_service: QueryContractService,
    ):
        """1. Every executable Phase 7C tool name exists in the Phase 7A registry."""
        registry_tools = set(default_registry.get_all_tool_names())

        for tpl in CANONICAL_TEMPLATES:
            req = CopilotRequest(question=tpl.question_text, as_of_date="2026-06-30")
            interp = interpret_question(req)
            assert interp is not None, f"Template {tpl.template_id} failed to interpret"

            plan, warnings, errors = decompose_and_plan(interp, req, contract_service=contract_service)
            assert plan is not None, f"Template {tpl.template_id} failed planning: {errors}"

            for step in plan.steps:
                assert step.tool_name in registry_tools, (
                    f"Template {tpl.template_id} produced unregistered tool: '{step.tool_name}'"
                )
                assert step.tool_name in PHASE_7A_VALID_TOOLS, (
                    f"Template {tpl.template_id} produced invalid tool: '{step.tool_name}'"
                )

    def test_no_invented_or_invalid_tool_names_exist(self, service: CopilotService):
        """2. Prohibit invented tool names across all canonical questions."""
        banned_tools = {
            "get_margin_status",
            "get_known_contribution_margin",
            "get_inventory_balance",
            "get_return_reasons",
            "get_opportunity_ranking",
            "top_n",
            "winners",
        }

        test_questions = [
            "Why did margin decline?",
            "Why is inventory risk high?",
            "Why are returns increasing?",
            "Show sales, margin, inventory and returns together",
            "What is our gross margin?",
            "What is our inventory position?",
            "What are the main return reasons?",
            "Show unit economics.",
        ]

        for q in test_questions:
            plan = service.plan(q, as_of_date="2026-06-30", currency="USD")
            assert plan is not None
            for step in plan.steps:
                assert step.tool_name not in banned_tools, (
                    f"Banned/invented tool '{step.tool_name}' found in plan for question '{q}'"
                )

    def test_phase_7b_is_authoritative_for_tool_mapping(
        self,
        contract_service: QueryContractService,
    ):
        """3. Phase 7B contract/planner layer remains strictly authoritative for tool mapping."""
        req = CopilotRequest(question="Why did margin decline?", as_of_date="2026-06-30")
        interp = interpret_question(req)
        assert interp is not None

        # Build plan through Copilot decomposition
        plan, _, _ = decompose_and_plan(interp, req, contract_service=contract_service)
        assert plan is not None

        # Build plan directly through Phase 7B contract service
        contract, val = contract_service.build_contract(
            intent=interp.intent,
            domain=interp.domain,
            as_of_date="2026-06-30",
            explanation_context=interp.explanation_context,
        )
        assert val.is_valid is True
        direct_7b_plan = contract_service.plan(contract)

        # Assert identical tools and sequences
        copilot_tools = [s.tool_name for s in plan.steps]
        direct_7b_tools = [s.tool_name for s in direct_7b_plan.steps]
        assert copilot_tools == direct_7b_tools

    def test_unknown_question_never_silently_falls_back_to_sales(self, service: CopilotService):
        """4. Unknown business questions do NOT silently fall back to SALES_PERFORMANCE."""
        unknown_questions = [
            "What happened in the business?",
            "How is marketing performing?",
            "What is our customer churn rate?",
            "Tell me about store foot traffic",
            "Show weather impact on conversions",
            "completely unrecognized random query 12345",
        ]

        for q in unknown_questions:
            # 1. Interpreter must return None
            interp = interpret_question(CopilotRequest(question=q))
            assert interp is None, f"Unknown question '{q}' was unexpectedly interpreted as {interp}"

            # 2. Orchestration must return CLARIFICATION_REQUIRED, not COMPLETED sales
            resp = service.process(q)
            assert resp.state == CopilotState.CLARIFICATION_REQUIRED, (
                f"Unknown question '{q}' returned state {resp.state.value} instead of CLARIFICATION_REQUIRED"
            )
            assert resp.interpretation is None
            assert resp.clarification is not None
            assert len(resp.clarification.possible_interpretations) >= 3

    def test_why_margin_uses_exact_canonical_tools(self, service: CopilotService):
        """5. 'Why did margin decline?' uses the exact approved canonical Phase 7A tools."""
        expected_tools = {
            "get_margin_summary",
            "get_margin_drivers",
            "get_sales_by_channel",
            "get_sales_by_warehouse",
        }

        plan = service.plan("Why did margin decline?", as_of_date="2026-06-30", currency="USD")
        assert plan is not None
        actual_tools = {s.tool_name for s in plan.steps}
        assert actual_tools == expected_tools
        assert all(t in PHASE_7A_CANONICAL_TOOLS for t in actual_tools)

    def test_why_inventory_risk_uses_exact_canonical_tools(self, service: CopilotService):
        """6. 'Why is inventory risk high?' uses the exact approved canonical Phase 7A tools."""
        expected_tools = {
            "get_inventory_risk",
            "get_inventory_summary",
            "get_slow_moving_inventory",
            "get_stockout_risk",
        }

        plan = service.plan("Why is inventory risk high?", as_of_date="2026-06-30")
        assert plan is not None
        actual_tools = {s.tool_name for s in plan.steps}
        assert actual_tools == expected_tools
        assert all(t in PHASE_7A_CANONICAL_TOOLS for t in actual_tools)

    def test_why_returns_uses_exact_canonical_tools(self, service: CopilotService):
        """7. 'Why are returns increasing?' uses the exact approved canonical Phase 7A tools."""
        expected_tools = {
            "get_return_anomalies",
            "get_return_reason_breakdown",
            "get_return_summary",
            "get_return_trend",
        }

        plan = service.plan("Why are returns increasing?", as_of_date="2026-06-30")
        assert plan is not None
        actual_tools = {s.tool_name for s in plan.steps}
        assert actual_tools == expected_tools
        assert all(t in PHASE_7A_CANONICAL_TOOLS for t in actual_tools)

    def test_multi_domain_orchestration_uses_canonical_tools(self, service: CopilotService):
        """8. Multi-domain cross-functional orchestration uses canonical Phase 7A tools."""
        plan = service.plan(
            "Show sales, margin, inventory and returns together",
            as_of_date="2026-06-30",
            currency="USD",
        )
        assert plan is not None
        actual_tools = {s.tool_name for s in plan.steps}
        expected_tools = {
            "get_revenue_summary",
            "get_margin_summary",
            "get_inventory_summary",
            "get_return_summary",
        }
        assert actual_tools == expected_tools
        assert all(t in PHASE_7A_CANONICAL_TOOLS for t in actual_tools)

    def test_aliases_resolve_to_canonical_tools_or_approved_aliases(self, service: CopilotService):
        """9. All planned tools belong strictly to Phase 7A canonical or approved alias catalog."""
        test_questions = [
            "What recommendations are pending review?",
            "What decisions are awaiting review?",
            "Where is stockout risk?",
            "Show ABC XYZ distribution.",
        ]

        for q in test_questions:
            plan = service.plan(q, as_of_date="2026-06-30")
            assert plan is not None
            for s in plan.steps:
                assert (
                    s.tool_name in PHASE_7A_CANONICAL_TOOLS
                    or s.tool_name in PHASE_7A_APPROVED_ALIASES
                ), f"Tool '{s.tool_name}' is neither canonical nor an approved alias"

    def test_phase_7c_cannot_bypass_phase_7b_validation(
        self,
        service: CopilotService,
        contract_service: QueryContractService,
    ):
        """10. Phase 7C cannot execute when Phase 7B contract validation fails."""
        # Create an artificial interpretation with an invalid grain for the intent
        from commerce_ai.copilot.schemas import CopilotInterpretation
        from commerce_ai.query_contracts.enums import RequestedOutput

        invalid_interp = CopilotInterpretation(
            interpretation_id="INT-invalid",
            normalized_question="Invalid grain test",
            intent=QueryIntent.REVENUE_ANALYSIS,
            domain=BusinessDomain.FINANCIAL,
            metrics=[],
            dimensions=[],
            filters={},
            requested_grain=BusinessGrain.CHANNEL,  # Invalid grain for REVENUE_ANALYSIS
            requested_output=RequestedOutput.KPI,
        )

        req = CopilotRequest(question="Invalid grain test")
        plan, warnings, errors = decompose_and_plan(invalid_interp, req, contract_service=contract_service)
        assert plan is None
        assert len(errors) > 0
        assert any("UNSUPPORTED_GRAIN_FOR_INTENT" in e for e in errors)

    def test_execution_pedigree_contains_valid_source_engine_identifiers(
        self,
        service: CopilotService,
    ):
        """11. Evidence lineage records valid logical source-engine identifiers."""
        resp = service.process("Why did margin decline?", as_of_date="2026-06-30", currency="USD")
        assert resp.state == CopilotState.COMPLETED
        assert resp.execution_result is not None
        assert len(resp.execution_result.evidence_chain) == 4

        for evi in resp.execution_result.evidence_chain:
            assert evi.source_engine.startswith("commerce_ai.")
            assert evi.tool_name in default_registry.get_all_tool_names()

    def test_programmatic_source_registry_truth_audit(self, service: CopilotService):
        """12. Programmatically verify that every planned tool exists in default_registry without hardcoding."""
        import inspect
        from commerce_ai.query_contracts.templates import CANONICAL_TEMPLATES

        all_registry_tools = set(default_registry.get_all_tool_names())
        assert len(all_registry_tools) == 49, f"Expected 49 registered tools, found {len(all_registry_tools)}"

        # 1. Test all canonical templates
        for tpl in CANONICAL_TEMPLATES:
            plan = service.plan(tpl.question_text, as_of_date="2026-06-30", currency="USD")
            assert plan is not None, f"Failed to plan for template {tpl.template_id}"
            for step in plan.steps:
                assert step.tool_name in all_registry_tools, (
                    f"Template {tpl.template_id} generated unregistered tool '{step.tool_name}'"
                )
                assert default_registry.get_tool(step.tool_name) is not None

        # 2. Test diagnostic why-questions
        why_questions = [
            "Why did margin decline?",
            "Why is inventory risk high?",
            "Why are returns increasing?",
            "Show sales, margin, inventory and returns together",
        ]
        for q in why_questions:
            plan = service.plan(q, as_of_date="2026-06-30", currency="USD")
            assert plan is not None, f"Failed to plan for '{q}'"
            for step in plan.steps:
                assert step.tool_name in all_registry_tools, (
                    f"Inquiry '{q}' generated unregistered tool '{step.tool_name}'"
                )

    def test_programmatic_ast_audit_for_unregistered_tools(self):
        """13. Programmatically parse Copilot source code AST to prohibit any unregistered get_* strings."""
        import ast
        from pathlib import Path

        registry_tools = set(default_registry.get_all_tool_names())
        copilot_dir = Path("src/commerce_ai/copilot")

        for py_file in copilot_dir.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    val = node.value
                    if val.startswith("get_") and val not in ("get_ready_steps",):
                        assert val in registry_tools, (
                            f"Disallowed/unregistered tool name '{val}' found in {py_file}"
                        )

