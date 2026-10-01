"""Unit tests for Context State management and transitions (Phase 7E)."""

from __future__ import annotations

from commerce_ai.context.enums import ContextCategory, ContextProvenance, ContextScope
from commerce_ai.context.schemas import (
    AmbiguityStateValue,
    ContextItem,
    ConversationContextSnapshot,
    CorrectionEvent,
    EntityContextValue,
    EvidenceReferenceValue,
    FilterContextValue,
    ResultReferenceValue,
    SessionContextValue,
)


def test_snapshot_initial_state() -> None:
    snapshot = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
    )
    assert snapshot.version == 1
    assert snapshot.current_turn_index == 0
    assert snapshot.session_context is None
    assert len(snapshot.recent_queries) == 0
    assert len(snapshot.evidence_refs) == 0
    assert len(snapshot.result_refs) == 0
    assert snapshot.active_ambiguity is None


def test_snapshot_add_context_items() -> None:
    snapshot = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
    )
    item = ContextItem(
        item_id="ITEM-1",
        category=ContextCategory.ENTITY_CONTEXT,
        scope=ContextScope.CONVERSATION,
        key="warehouse_id",
        value="WH_01",
        provenance=ContextProvenance.EXPLICIT_USER,
    )
    snapshot.items["warehouse_id"] = item
    assert "warehouse_id" in snapshot.items
    assert snapshot.items["warehouse_id"].value == "WH_01"


def test_snapshot_entity_context_mutations() -> None:
    snapshot = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
        entities=EntityContextValue(sku_id="SKU_01", warehouse_id="WH_01"),
    )
    assert snapshot.entities.sku_id == "SKU_01"
    snapshot.entities.warehouse_id = "WH_02"
    assert snapshot.entities.warehouse_id == "WH_02"


def test_snapshot_record_evidence_and_results() -> None:
    snapshot = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
    )
    ev = EvidenceReferenceValue(
        evidence_id="EV-1",
        tool_name="get_inventory_summary",
        query_id="Q-1",
        turn_index=1,
    )
    res = ResultReferenceValue(
        query_id="Q-1",
        turn_index=1,
        intent="INVENTORY_SUMMARY",
        record_count=10,
    )
    snapshot.evidence_refs.append(ev)
    snapshot.result_refs.append(res)

    assert len(snapshot.evidence_refs) == 1
    assert len(snapshot.result_refs) == 1
    assert snapshot.result_refs[0].record_count == 10


def test_snapshot_corrections_log() -> None:
    snapshot = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
    )
    corr = CorrectionEvent(
        correction_id="CORR-1",
        turn_index=1,
        target_category=ContextCategory.ENTITY_CONTEXT,
        target_key="sku_id",
        previous_value="SKU_01",
        new_value="SKU_02",
        reason="User corrected SKU",
    )
    snapshot.corrections.append(corr)
    assert len(snapshot.corrections) == 1
    assert snapshot.corrections[0].target_key == "sku_id"


def test_snapshot_active_ambiguity_lifecycle() -> None:
    snapshot = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
    )
    assert snapshot.active_ambiguity is None

    snapshot.active_ambiguity = AmbiguityStateValue(
        ambiguity_id="AMB-1",
        turn_index=1,
        original_question="What is the inventory?",
        missing_fields=["warehouse_id"],
        clarification_prompt="Please specify a warehouse.",
    )
    assert snapshot.active_ambiguity.status == "PENDING"
    assert snapshot.active_ambiguity.missing_fields == ["warehouse_id"]

    # Resolve ambiguity
    snapshot.active_ambiguity = None
    assert snapshot.active_ambiguity is None


def test_snapshot_recent_queries_log() -> None:
    snapshot = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
    )
    snapshot.recent_queries.append({
        "request_id": "REQ-1",
        "turn_index": 1,
        "question": "Show sales",
    })
    snapshot.recent_queries.append({
        "request_id": "REQ-2",
        "turn_index": 2,
        "question": "What about its inventory?",
    })
    assert len(snapshot.recent_queries) == 2
    assert snapshot.recent_queries[0]["request_id"] == "REQ-1"


def test_snapshot_deep_copy_isolation() -> None:
    snapshot = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
        entities=EntityContextValue(sku_id="SKU_01"),
    )
    copy_snap = snapshot.model_copy(deep=True)
    copy_snap.entities.sku_id = "SKU_99"

    assert snapshot.entities.sku_id == "SKU_01"
    assert copy_snap.entities.sku_id == "SKU_99"


def test_snapshot_filter_context_storage() -> None:
    snapshot = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
        filters=FilterContextValue(currency="EUR", dimension="channel"),
    )
    assert snapshot.filters.currency == "EUR"
    assert snapshot.filters.dimension == "channel"


def test_snapshot_model_dump_and_reconstruct() -> None:
    snapshot = ConversationContextSnapshot(
        session_id="S-DUMP",
        conversation_id="C-DUMP",
        entities=EntityContextValue(sku_id="SKU_SERIALIZE", warehouse_id="WH_SERIALIZE"),
    )
    dumped = snapshot.model_dump()
    reconstructed = ConversationContextSnapshot.model_validate(dumped)

    assert reconstructed.session_id == "S-DUMP"
    assert reconstructed.entities.sku_id == "SKU_SERIALIZE"
    assert reconstructed.entities.warehouse_id == "WH_SERIALIZE"


def test_snapshot_session_context_update() -> None:
    sess = SessionContextValue(
        session_id="S-1",
        user_id="analyst_01",
        tenant_id="enterprise_retail",
        default_currency="GBP",
    )
    snapshot = ConversationContextSnapshot(
        session_id="S-1",
        conversation_id="C-1",
        session_context=sess,
    )
    assert snapshot.session_context.user_id == "analyst_01"
    assert snapshot.session_context.default_currency == "GBP"

