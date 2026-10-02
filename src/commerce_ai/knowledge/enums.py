"""Controlled Business Knowledge / RAG Enums (Phase 7F).

Defines controlled vocabularies for business document types, document lifecycle
statuses, knowledge domains, retrieval methods, provenance types, conflict severities,
and capability routing categories.
"""

from __future__ import annotations

from enum import Enum


class DocumentType(str, Enum):
    """Controlled taxonomy of business document types."""
    POLICY = "POLICY"
    SOP = "SOP"
    BUSINESS_RULE = "BUSINESS_RULE"
    GLOSSARY = "GLOSSARY"
    PRODUCT_DOCUMENTATION = "PRODUCT_DOCUMENTATION"
    PROCESS_DOCUMENTATION = "PROCESS_DOCUMENTATION"
    CONFIGURATION_GUIDE = "CONFIGURATION_GUIDE"
    WAREHOUSE_GUIDE = "WAREHOUSE_GUIDE"
    RETURN_POLICY = "RETURN_POLICY"
    REPLENISHMENT_POLICY = "REPLENISHMENT_POLICY"
    OTHER = "OTHER"


class DocumentStatus(str, Enum):
    """Document version lifecycle status."""
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    SUPERSEDED = "SUPERSEDED"
    ARCHIVED = "ARCHIVED"


class KnowledgeDomain(str, Enum):
    """Business domains aligned with platform architecture."""
    SALES = "SALES"
    FINANCIAL = "FINANCIAL"
    INVENTORY = "INVENTORY"
    DEMAND = "DEMAND"
    FORECASTING = "FORECASTING"
    RETURNS = "RETURNS"
    OPERATIONS = "OPERATIONS"
    BUSINESS_IMPACT = "BUSINESS_IMPACT"
    RECOMMENDATIONS = "RECOMMENDATIONS"
    DECISIONS = "DECISIONS"
    DATA_QUALITY = "DATA_QUALITY"
    CROSS_DOMAIN = "CROSS_DOMAIN"
    GENERAL = "GENERAL"


class RetrievalMethod(str, Enum):
    """Accurately records the retrieval algorithm used without fabricating semantic claims."""
    LEXICAL_TOKEN_MATCH = "LEXICAL_TOKEN_MATCH"
    EXACT_KEYWORD_MATCH = "EXACT_KEYWORD_MATCH"
    HYBRID_SEARCH = "HYBRID_SEARCH"


class KnowledgeProvenanceType(str, Enum):
    """Explicit evidence source classification separating data from document knowledge."""
    DATA_DERIVED = "DATA_DERIVED"
    DOCUMENT_DERIVED = "DOCUMENT_DERIVED"
    MIXED = "MIXED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ConflictSeverity(str, Enum):
    """Severity of policy or knowledge contradiction."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    BLOCKING = "BLOCKING"


class ConflictStatus(str, Enum):
    """Resolution status of a detected knowledge conflict."""
    DETECTED = "DETECTED"
    UNRESOLVED = "UNRESOLVED"
    RESOLVED = "RESOLVED"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


class QueryCapability(str, Enum):
    """Deterministic capability routing categories for inbound questions."""
    DATA_QUERY = "DATA_QUERY"
    KNOWLEDGE_QUERY = "KNOWLEDGE_QUERY"
    HYBRID_QUERY = "HYBRID_QUERY"
    UNSUPPORTED_QUERY = "UNSUPPORTED_QUERY"
    AMBIGUOUS_QUERY = "AMBIGUOUS_QUERY"
