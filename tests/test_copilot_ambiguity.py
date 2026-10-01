"""Unit tests for Ambiguity Detection & Clarification Policies (Phase 7C).

Tests:
- Detection of underspecified inventory queries
- Detection of underspecified returns queries
- Detection of broad unanchored status requests
- Non-triggering of clarification when safe defaults exist (canonical templates)
- Structure and determinism of CopilotClarification objects
"""

from __future__ import annotations

import pytest

from commerce_ai.copilot.ambiguity import check_query_ambiguity
from commerce_ai.copilot.enums import ClarificationSeverity


class TestAmbiguityDetection:
    def test_underspecified_inventory_query(self):
        clar = check_query_ambiguity("show inventory")
        assert clar is not None
        assert "underspecified" in clar.reason
        assert "intent" in clar.missing_fields
        assert clar.severity == ClarificationSeverity.BLOCKING
        assert len(clar.possible_interpretations) >= 3

    def test_underspecified_stock_query(self):
        clar = check_query_ambiguity("check stock")
        assert clar is not None
        assert clar.clarification_id.startswith("CLR-")

    def test_underspecified_returns_query(self):
        clar = check_query_ambiguity("what about returns")
        assert clar is not None
        assert "returns request is underspecified" in clar.reason
        assert "intent" in clar.missing_fields
        assert any("return rate" in opt for opt in clar.possible_interpretations)

    def test_underspecified_returns_check(self):
        clar = check_query_ambiguity("check returns")
        assert clar is not None

    def test_generic_status_report_query(self):
        clar = check_query_ambiguity("status report")
        assert clar is not None
        assert "does not specify a business domain" in clar.reason
        assert "domain" in clar.missing_fields

    def test_generic_health_check_query(self):
        clar = check_query_ambiguity("health check")
        assert clar is not None

    def test_generic_show_me_data(self):
        clar = check_query_ambiguity("show me data")
        assert clar is not None
        assert "completely unspecified" in clar.reason

    def test_generic_analyze_query(self):
        clar = check_query_ambiguity("analyze")
        assert clar is not None

    def test_unambiguous_sales_performance_no_clarification(self):
        clar = check_query_ambiguity("How are sales performing")
        assert clar is None

    def test_unambiguous_margin_status_no_clarification(self):
        clar = check_query_ambiguity("What is our margin")
        assert clar is None

    def test_unambiguous_why_margin_down_no_clarification(self):
        clar = check_query_ambiguity("Why did margin decline")
        assert clar is None

    def test_clarification_id_determinism(self):
        c1 = check_query_ambiguity("show inventory")
        c2 = check_query_ambiguity("show inventory")
        assert c1 is not None and c2 is not None
        assert c1.clarification_id == c2.clarification_id
