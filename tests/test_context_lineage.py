"""Unit tests for Context Lineage and Provenance Tracking (Phase 7E)."""

from __future__ import annotations

from commerce_ai.context.enums import ContextProvenance
from commerce_ai.context.lineage import ContextLineageEvent, ContextLineageTracker


def test_lineage_tracker_empty() -> None:
    tracker = ContextLineageTracker()
    assert len(tracker.get_events()) == 0
    assert tracker.explain_item_lineage("unknown_key") == "No context lineage recorded for attribute 'unknown_key'."


def test_lineage_tracker_record_event() -> None:
    tracker = ContextLineageTracker()
    ev = tracker.record(
        turn_index=1,
        item_key="sku_id",
        operation="CREATED",
        provenance=ContextProvenance.EXPLICIT_USER,
        new_value="SKU_100",
        query_id="QRY-001",
    )
    assert ev.event_id.startswith("LIN-")
    assert ev.turn_index == 1
    assert ev.item_key == "sku_id"
    assert ev.operation == "CREATED"
    assert ev.new_value == "SKU_100"
    assert len(tracker.get_events()) == 1


def test_lineage_tracker_item_history() -> None:
    tracker = ContextLineageTracker()
    tracker.record(turn_index=1, item_key="sku_id", operation="CREATED", provenance=ContextProvenance.EXPLICIT_USER, new_value="SKU_1")
    tracker.record(turn_index=1, item_key="warehouse_id", operation="CREATED", provenance=ContextProvenance.EXPLICIT_USER, new_value="WH_1")
    tracker.record(turn_index=2, item_key="sku_id", operation="INHERITED", provenance=ContextProvenance.DERIVED_FROM_VALIDATED_QUERY, new_value="SKU_1")

    sku_history = tracker.get_item_history("sku_id")
    assert len(sku_history) == 2
    assert sku_history[0].operation == "CREATED"
    assert sku_history[1].operation == "INHERITED"

    wh_history = tracker.get_item_history("warehouse_id")
    assert len(wh_history) == 1
    assert wh_history[0].new_value == "WH_1"


def test_lineage_tracker_turn_events() -> None:
    tracker = ContextLineageTracker()
    tracker.record(turn_index=1, item_key="sku_id", operation="CREATED", provenance=ContextProvenance.EXPLICIT_USER, new_value="SKU_1")
    tracker.record(turn_index=2, item_key="sku_id", operation="INHERITED", provenance=ContextProvenance.DERIVED_FROM_VALIDATED_QUERY, new_value="SKU_1")
    tracker.record(turn_index=2, item_key="warehouse_id", operation="OVERRIDDEN", provenance=ContextProvenance.EXPLICIT_USER, new_value="WH_2")

    t1_events = tracker.get_turn_events(1)
    assert len(t1_events) == 1

    t2_events = tracker.get_turn_events(2)
    assert len(t2_events) == 2


def test_lineage_explain_narrative() -> None:
    tracker = ContextLineageTracker()
    tracker.record(
        turn_index=1,
        item_key="warehouse_id",
        operation="CREATED",
        provenance=ContextProvenance.EXPLICIT_USER,
        new_value="WH_01",
        query_id="QRY-1",
    )
    tracker.record(
        turn_index=2,
        item_key="warehouse_id",
        operation="INHERITED",
        provenance=ContextProvenance.DERIVED_FROM_VALIDATED_QUERY,
        new_value="WH_01",
    )
    tracker.record(
        turn_index=3,
        item_key="warehouse_id",
        operation="OVERRIDDEN",
        provenance=ContextProvenance.USER_CORRECTION,
        new_value="WH_02",
        old_value="WH_01",
        query_id="QRY-3",
    )

    narrative = tracker.explain_item_lineage("warehouse_id")
    assert "Lineage for 'warehouse_id':" in narrative
    assert "[Turn 1] CREATED (EXPLICIT_USER): 'WH_01'" in narrative
    assert "[Turn 2] INHERITED (DERIVED_FROM_VALIDATED_QUERY): 'WH_01'" in narrative
    assert "[Turn 3] OVERRIDDEN (USER_CORRECTION): 'WH_02'" in narrative


def test_lineage_event_immutability() -> None:
    ev = ContextLineageEvent(
        event_id="LIN-01",
        turn_index=1,
        item_key="sku_id",
        operation="CREATED",
        provenance=ContextProvenance.EXPLICIT_USER,
        new_value="SKU_01",
    )
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ev.new_value = "SKU_02"  # type: ignore # frozen


def test_lineage_tracker_with_initial_events() -> None:
    ev1 = ContextLineageEvent(
        event_id="LIN-01",
        turn_index=1,
        item_key="sku_id",
        operation="CREATED",
        provenance=ContextProvenance.EXPLICIT_USER,
        new_value="SKU_01",
    )
    tracker = ContextLineageTracker(events=[ev1])
    assert len(tracker.get_events()) == 1
    assert tracker.get_events()[0].event_id == "LIN-01"


def test_lineage_tracker_details_payload() -> None:
    tracker = ContextLineageTracker()
    ev = tracker.record(
        turn_index=2,
        item_key="channel_id",
        operation="INHERITED",
        provenance=ContextProvenance.DERIVED_FROM_VALIDATED_QUERY,
        new_value="ONLINE",
        details={"source_query": "QRY-100", "rule": "persistent_channel"},
    )
    assert ev.details["source_query"] == "QRY-100"
    assert ev.details["rule"] == "persistent_channel"


def test_lineage_tracker_pruned_and_expired_operations() -> None:
    tracker = ContextLineageTracker()
    ev1 = tracker.record(
        turn_index=3,
        item_key="channel_id",
        operation="PRUNED",
        provenance=ContextProvenance.SYSTEM_DEFAULT,
        new_value=None,
        old_value="ONLINE",
        details={"reason": "domain_switch_to_supplier"},
    )
    assert ev1.operation == "PRUNED"
    assert ev1.old_value == "ONLINE"

    ev2 = tracker.record(
        turn_index=4,
        item_key="temp_filter",
        operation="EXPIRED",
        provenance=ContextProvenance.SYSTEM_DEFAULT,
        new_value=None,
        old_value="active",
    )
    assert ev2.operation == "EXPIRED"


def test_lineage_tracker_conversation_id_filtering() -> None:
    tracker = ContextLineageTracker()
    ev_c1 = tracker.record(
        turn_index=1,
        item_key="sku_id",
        operation="CREATED",
        provenance=ContextProvenance.EXPLICIT_USER,
        new_value="SKU_01",
        conversation_id="conv-1",
    )
    ev_c2 = tracker.record(
        turn_index=1,
        item_key="sku_id",
        operation="CREATED",
        provenance=ContextProvenance.EXPLICIT_USER,
        new_value="SKU_99",
        conversation_id="conv-2",
    )

    c1_events = tracker.get_conversation_events("conv-1")
    c2_events = tracker.get_conversation_events("conv-2")

    assert len(c1_events) == 1
    assert c1_events[0].new_value == "SKU_01"
    assert c1_events[0].conversation_id == "conv-1"

    assert len(c2_events) == 1
    assert c2_events[0].new_value == "SKU_99"
    assert c2_events[0].conversation_id == "conv-2"

