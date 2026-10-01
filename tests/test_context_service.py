"""Unit tests for Conversation Context Service (Phase 7E)."""

from __future__ import annotations

from commerce_ai.context.memory import InMemoryConversationContextStore
from commerce_ai.context.schemas import EntityContextValue
from commerce_ai.context.service import ConversationContextService
from commerce_ai.copilot.enums import CopilotState, PlanExecutionStatus
from commerce_ai.copilot.schemas import (
    CopilotClarification,
    CopilotExecutionPlan,
    CopilotExecutionResult,
    CopilotInterpretation,
    CopilotRequest,
    CopilotResponse,
)
from commerce_ai.query_contracts.enums import BusinessDomain, QueryIntent


def test_service_get_or_create_fresh() -> None:
    svc = ConversationContextService()
    snap = svc.get_or_create_context("SESS-1", "CONV-1", user_id="U1", tenant_id="T1")
    assert snap.session_id == "SESS-1"
    assert snap.conversation_id == "CONV-1"
    assert snap.version == 1
    assert snap.session_context is not None
    assert snap.session_context.user_id == "U1"


def test_service_get_existing() -> None:
    svc = ConversationContextService()
    snap1 = svc.get_or_create_context("SESS-1", "CONV-1")
    snap1.entities.sku_id = "SKU_99"
    svc.store.save(snap1)

    snap2 = svc.get_or_create_context("SESS-1", "CONV-1")
    assert snap2.entities.sku_id == "SKU_99"


def test_service_resolve_request_entity_inheritance() -> None:
    svc = ConversationContextService()
    snap = svc.get_or_create_context("SESS-1", "CONV-1")
    snap.entities.sku_id = "SKU_01"
    snap.entities.warehouse_id = "WH_01"
    svc.store.save(snap)

    req, res = svc.resolve_request("What is its inventory?", "SESS-1", "CONV-1")
    assert isinstance(req, CopilotRequest)
    assert req.filters.get("sku_id") == "SKU_01"
    assert req.filters.get("warehouse_id") == "WH_01"
    assert res.inherited_entities.get("sku_id") == "SKU_01"


def test_service_resolve_request_advances_turn_and_version() -> None:
    svc = ConversationContextService()
    snap = svc.get_or_create_context("SESS-1", "CONV-1")
    assert snap.current_turn_index == 0

    req1, _ = svc.resolve_request("Show sales for SKU_01", "SESS-1", "CONV-1")
    snap1 = svc.store.get("SESS-1", "CONV-1")
    assert snap1 is not None
    assert snap1.current_turn_index == 1

    req2, _ = svc.resolve_request("What about WH_02?", "SESS-1", "CONV-1")
    snap2 = svc.store.get("SESS-1", "CONV-1")
    assert snap2 is not None
    assert snap2.current_turn_index == 2


def test_service_update_after_response_success() -> None:
    svc = ConversationContextService()
    req = CopilotRequest(request_id="REQ-1", question="Show sales for SKU_01")
    resp = CopilotResponse(
        response_id="RSP-1",
        request_id="REQ-1",
        state=CopilotState.COMPLETED,
        normalized_question="show sales for sku_01",
        interpretation=CopilotInterpretation(
            interpretation_id="INTRP-1",
            intent=QueryIntent.SALES_PERFORMANCE,
            domain=BusinessDomain.SALES,
            normalized_question="show sales for sku_01",
        ),
        execution_result=CopilotExecutionResult(
            result_id="RES-1",
            request_id="REQ-1",
            status=PlanExecutionStatus.COMPLETE,
            total_execution_time_ms=5.0,
            step_results=[],
        ),
    )

    updated_snap = svc.update_after_response("SESS-1", "CONV-1", req, resp)
    assert len(updated_snap.recent_queries) == 1
    assert updated_snap.recent_queries[0]["request_id"] == "REQ-1"
    assert len(updated_snap.result_refs) == 1


def test_service_update_after_response_clarification_required() -> None:
    svc = ConversationContextService()
    req = CopilotRequest(request_id="REQ-1", question="Show inventory")
    resp = CopilotResponse(
        response_id="RSP-1",
        request_id="REQ-1",
        state=CopilotState.CLARIFICATION_REQUIRED,
        normalized_question="show inventory",
        clarification=CopilotClarification(
            clarification_id="CLR-1",
            question="show inventory",
            reason="Please specify warehouse",
            missing_fields=["warehouse_id"],
        ),
    )

    updated_snap = svc.update_after_response("SESS-1", "CONV-1", req, resp)
    assert updated_snap.active_ambiguity is not None
    assert updated_snap.active_ambiguity.missing_fields == ["warehouse_id"]
    assert updated_snap.active_ambiguity.status == "PENDING"


def test_service_record_user_correction() -> None:
    svc = ConversationContextService()
    snap = svc.get_or_create_context("SESS-1", "CONV-1")
    snap.entities.warehouse_id = "WH_01"
    svc.store.save(snap)

    event = svc.record_user_correction(
        session_id="SESS-1",
        conversation_id="CONV-1",
        target_key="warehouse_id",
        new_value="WH_02",
        reason="User corrected warehouse",
    )
    assert event.previous_value == "WH_01"
    assert event.new_value == "WH_02"

    refreshed = svc.store.get("SESS-1", "CONV-1")
    assert refreshed is not None
    assert refreshed.entities.warehouse_id == "WH_02"
    assert len(refreshed.corrections) == 1


def test_service_reset_conversation() -> None:
    svc = ConversationContextService()
    svc.get_or_create_context("SESS-1", "CONV-1")
    assert svc.store.get("SESS-1", "CONV-1") is not None

    deleted = svc.reset_conversation("SESS-1", "CONV-1")
    assert deleted is True
    assert svc.store.get("SESS-1", "CONV-1") is None


def test_service_export_audit_trail() -> None:
    svc = ConversationContextService()
    svc.get_or_create_context("SESS-1", "CONV-1")
    svc.resolve_request("Show sales for SKU_01 in WH_01", "SESS-1", "CONV-1")

    trail = svc.export_audit_trail("SESS-1", "CONV-1")
    assert trail["session_id"] == "SESS-1"
    assert trail["conversation_id"] == "CONV-1"
    assert "active_entities" in trail
    assert "lineage_events" in trail
    assert "governance" in trail
