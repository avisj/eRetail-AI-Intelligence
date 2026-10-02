"""Unit tests for Phase 7F Deterministic Capability Routing Engine."""

from __future__ import annotations

import pytest

from commerce_ai.knowledge.enums import QueryCapability
from commerce_ai.knowledge.routing import classify_query_capability


def test_routing_pure_data_queries() -> None:
    data_queries = [
        "Show sales for SKU_001 in warehouse WH_01",
        "What were sales last month?",
        "What is our gross margin by channel?",
        "Show inventory position for WH_02",
        "What is the demand forecast for the next 14 days?",
        "Show return rate by customer segment",
    ]
    for q in data_queries:
        assert classify_query_capability(q) == QueryCapability.DATA_QUERY, f"Failed on: {q}"


def test_routing_pure_knowledge_queries() -> None:
    knowledge_queries = [
        "What is our replenishment policy?",
        "Explain the warehouse receiving SOP",
        "What are our standard return policies?",
        "Define safety stock formula and calculation rules",
        "What are the compliance procedures for quarantine items?",
        "Show the configuration guide for warehouse rebalancing",
        "What does our policy say regarding supplier lead times?",
    ]
    for q in knowledge_queries:
        assert classify_query_capability(q) == QueryCapability.KNOWLEDGE_QUERY, f"Failed on: {q}"


def test_routing_hybrid_queries() -> None:
    hybrid_queries = [
        "Inventory risk is high for SKU_001. What does our policy say should happen?",
        "Current return rate is 12%, what is our return threshold policy?",
        "Stockout risk detected for WH_01; what is our approved replenishment procedure?",
        "Sales dropped last month. What are our promotional discount guidelines?",
    ]
    for q in hybrid_queries:
        assert classify_query_capability(q) == QueryCapability.HYBRID_QUERY, f"Failed on: {q}"


def test_routing_unsupported_queries() -> None:
    unsupported_queries = [
        "What is the weather tomorrow in Chicago?",
        "Who won the soccer world cup final?",
        "What are the presidential election results?",
        "Write a poem about warehouse forklifts",
        "How do I bake chocolate chip cookies?",
    ]
    for q in unsupported_queries:
        assert classify_query_capability(q) == QueryCapability.UNSUPPORTED_QUERY, f"Failed on: {q}"


def test_routing_ambiguous_short_queries() -> None:
    ambiguous_queries = [
        "",
        "   ",
        "what?",
        "why",
        "how",
        "status",
        "inventory",
        "policy?",
    ]
    for q in ambiguous_queries:
        assert classify_query_capability(q) == QueryCapability.AMBIGUOUS_QUERY, f"Failed on: {q}"
