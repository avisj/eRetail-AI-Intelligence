"""Unit tests for Phase 7E Context Schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from commerce_ai.context.enums import (
    CompatibilityStatus,
    ContextCategory,
    ContextConfidence,
    ContextProvenance,
    ContextScope,
)
from commerce_ai.context.schemas import (
    AmbiguityStateValue,
    ContextGovernance,
    ContextItem,
    ContextResolutionResult,
    ConversationContextSnapshot,
    CorrectionEvent,
    EntityContextValue,
    EvidenceReferenceValue,
    FilterContextValue,
    ResultReferenceValue,
    SessionContextValue,
    TimeRangeContextValue,
)


def test_context_governance_defaults() -> None:
    gov = ContextGovernance()
    assert gov.read_only is True
    assert gov.action_execution is False
    assert gov.execution_allowed is False
    assert gov.no_ranking_enforced is True
    assert gov.no_decision_selection_enforced is True
    assert gov.allow_injection is False


def test_context_governance_immutability() -> None:
    gov = ContextGovernance()
    with pytest.raises(ValidationError):
        gov.read_only = False  # frozen model


def test_context_item_valid() -> None:
    item = ContextItem(
        item_id="ITEM-01",
        category=ContextCategory.ENTITY_CONTEXT,
        scope=ContextScope.CONVERSATION,
        key="sku_id",
        value="SKU_100",
        provenance=ContextProvenance.EXPLICIT_USER,
    )
    assert item.item_id == "ITEM-01"
    assert item.key == "sku_id"
    assert item.value == "SKU_100"
    assert item.confidence == ContextConfidence.HIGH
    assert item.compatibility_status == CompatibilityStatus.REUSABLE


def test_context_item_forbid_extra() -> None:
    with pytest.raises(ValidationError):
        ContextItem(
            item_id="ITEM-02",
            category=ContextCategory.ENTITY_CONTEXT,
            key="sku_id",
            value="SKU_100",
            unauthorized_field="foo",  # type: ignore
        )


def test_session_context_value() -> None:
    sess = SessionContextValue(
        session_id="SESS-001",
        user_id="user_admin",
        tenant_id="tenant_retail",
        default_currency="USD",
    )
    assert sess.session_id == "SESS-001"
    assert sess.user_id == "user_admin"
    assert sess.default_currency == "USD"


def test_entity_context_to_dict() -> None:
    ent = EntityContextValue(
        sku_id="SKU_01",
        warehouse_id="WH_01",
        channel_id="ONLINE",
    )
    d = ent.to_dict()
    assert d == {"sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "ONLINE"}
    assert "supplier_id" not in d


def test_time_range_context_value() -> None:
    tr = TimeRangeContextValue(
        preset="LAST_30_DAYS",
        start_date="2026-06-01",
        end_date="2026-06-30",
        as_of_date="2026-06-30",
        grain="DAY",
    )
    assert tr.preset == "LAST_30_DAYS"
    assert tr.start_date == "2026-06-01"
    assert tr.end_date == "2026-06-30"


def test_filter_context_value() -> None:
    fc = FilterContextValue(
        dimension="warehouse",
        currency="USD",
        filters={"min_revenue": 1000},
    )
    assert fc.dimension == "warehouse"
    assert fc.currency == "USD"
    assert fc.filters["min_revenue"] == 1000


def test_evidence_reference_value() -> None:
    ev = EvidenceReferenceValue(
        evidence_id="EV-1234",
        tool_name="get_sales_summary",
        query_id="QRY-5678",
        turn_index=1,
        intent="SALES_PERFORMANCE",
        calculation_status="SUCCESS",
    )
    assert ev.evidence_id == "EV-1234"
    assert ev.tool_name == "get_sales_summary"
    assert ev.calculation_status == "SUCCESS"


def test_result_reference_value() -> None:
    rr = ResultReferenceValue(
        query_id="QRY-001",
        turn_index=2,
        intent="INVENTORY_SUMMARY",
        record_count=15,
        summary_text="Inventory summary shows 15 active SKUs.",
    )
    assert rr.query_id == "QRY-001"
    assert rr.record_count == 15
    assert rr.summary_text is not None


def test_correction_event() -> None:
    corr = CorrectionEvent(
        correction_id="CORR-01",
        turn_index=3,
        target_category=ContextCategory.ENTITY_CONTEXT,
        target_key="warehouse_id",
        previous_value="WH_01",
        new_value="WH_02",
        reason="User corrected warehouse",
    )
    assert corr.previous_value == "WH_01"
    assert corr.new_value == "WH_02"
    assert corr.reason == "User corrected warehouse"


def test_ambiguity_state_value() -> None:
    amb = AmbiguityStateValue(
        ambiguity_id="AMB-01",
        turn_index=1,
        original_question="Show inventory",
        missing_fields=["warehouse_id"],
        candidate_intents=["INVENTORY_POSITION", "INVENTORY_SUMMARY"],
        clarification_prompt="Please specify which warehouse.",
    )
    assert amb.status == "PENDING"
    assert "warehouse_id" in amb.missing_fields


def test_conversation_context_snapshot_defaults() -> None:
    snap = ConversationContextSnapshot(
        session_id="SESS-1",
        conversation_id="CONV-1",
    )
    assert snap.version == 1
    assert snap.current_turn_index == 0
    assert snap.governance.read_only is True
    assert len(snap.items) == 0


def test_context_resolution_result() -> None:
    res = ContextResolutionResult(
        original_question="What about its inventory?",
        resolved_question="What about its inventory for SKU SKU_01?",
        inherited_entities={"sku_id": "SKU_01"},
    )
    assert res.original_question == "What about its inventory?"
    assert res.inherited_entities["sku_id"] == "SKU_01"
    assert res.confidence == ContextConfidence.HIGH
