"""Unit tests for Explanation Rendering (Phase 7D)."""

from __future__ import annotations

import pytest

from commerce_ai.copilot.service import CopilotService
from commerce_ai.explanations.renderer import render_explanation_to_text
from commerce_ai.explanations.service import ExplanationService
from commerce_ai.query_layer.service import QueryLayerService


class TestExplanationRendering:
    def test_render_contains_all_major_sections(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process(
            "Why did margin decline?",
            as_of_date="2026-06-30",
            currency="USD",
        )
        exp = explanation_service.explain(copilot_resp)
        rendered_text = render_explanation_to_text(exp)

        assert "HEADLINE:" in rendered_text
        assert "SUMMARY:" in rendered_text
        assert "KEY FINDINGS:" in rendered_text
        assert "LIMITATIONS & GOVERNANCE:" in rendered_text
        assert "PROVENANCE & AUDIT:" in rendered_text
        assert "```" not in rendered_text  # Strict prohibition of triple backticks

    def test_explain_text_convenience_method(self):
        ql = QueryLayerService.from_sample_data()
        copilot_service = CopilotService(query_layer=ql)
        explanation_service = ExplanationService()

        copilot_resp = copilot_service.process(
            "How are sales performing?",
            as_of_date="2026-06-30",
            currency="USD",
        )
        text_output = explanation_service.explain_text(copilot_resp)

        assert isinstance(text_output, str)
        assert len(text_output) > 50
        assert "HEADLINE:" in text_output
        assert "```" not in text_output
