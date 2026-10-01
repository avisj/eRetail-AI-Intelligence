"""Controlled Taxonomies and Enumerations for Business Explanations (Phase 7D).

Defines immutable, controlled vocabulary for:
- ExplanationType: Functional classification of business explanations
- ExplanationStatus: Execution and evidence sufficiency state
- ProvenanceType: Origin of evidence items
- EvidenceType: Empirical nature of the underlying data
- ExplanationConfidence: Deterministic confidence levels based on evidence support
- ClaimCategory: Classification of individual findings and assertions
"""

from __future__ import annotations

from enum import Enum


class ExplanationType(str, Enum):
    """Controlled catalog of business explanation types."""
    KPI_EXPLANATION = "KPI_EXPLANATION"
    TREND_EXPLANATION = "TREND_EXPLANATION"
    BREAKDOWN_EXPLANATION = "BREAKDOWN_EXPLANATION"
    COMPARISON_EXPLANATION = "COMPARISON_EXPLANATION"
    WHY_EXPLANATION = "WHY_EXPLANATION"
    MULTI_DOMAIN_EXPLANATION = "MULTI_DOMAIN_EXPLANATION"
    RISK_EXPLANATION = "RISK_EXPLANATION"
    FORECAST_EXPLANATION = "FORECAST_EXPLANATION"
    RETURN_EXPLANATION = "RETURN_EXPLANATION"
    INVENTORY_EXPLANATION = "INVENTORY_EXPLANATION"
    FINANCIAL_EXPLANATION = "FINANCIAL_EXPLANATION"
    OPERATIONAL_REVIEW_EXPLANATION = "OPERATIONAL_REVIEW_EXPLANATION"
    DATA_QUALITY_EXPLANATION = "DATA_QUALITY_EXPLANATION"
    INSUFFICIENT_DATA_EXPLANATION = "INSUFFICIENT_DATA_EXPLANATION"
    UNAVAILABLE_EXPLANATION = "UNAVAILABLE_EXPLANATION"


class ExplanationStatus(str, Enum):
    """Lifecycle status of the generated explanation."""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"


class ProvenanceType(str, Enum):
    """Pedigree of the evidence supporting an explanation."""
    USER_PROVIDED = "USER_PROVIDED"
    PHASE_7A_OBSERVED = "PHASE_7A_OBSERVED"
    PHASE_7B_DERIVED = "PHASE_7B_DERIVED"
    PHASE_7C_ORCHESTRATED = "PHASE_7C_ORCHESTRATED"
    MODEL_BASED = "MODEL_BASED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class EvidenceType(str, Enum):
    """Empirical nature of an evidence observation."""
    DIRECT_OBSERVED = "DIRECT_OBSERVED"
    DETERMINISTIC_DERIVED = "DETERMINISTIC_DERIVED"
    STATISTICAL_DERIVED = "STATISTICAL_DERIVED"
    MODEL_PROJECTED = "MODEL_PROJECTED"
    HYPOTHETICAL = "HYPOTHETICAL"


class ExplanationConfidence(str, Enum):
    """Deterministic evidence support tier.
    
    HIGH: Direct observed metric with complete supporting data.
    MEDIUM: Derived comparison or multi-step synthesis supported by complete upstream metrics.
    LOW: Limited evidence, small sample size, or partial data warnings.
    INSUFFICIENT: Required evidence unavailable or calculation failed.
    """
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class ClaimCategory(str, Enum):
    """Classification of individual business claims and findings."""
    OBSERVED_FACT = "OBSERVED_FACT"
    DERIVED_METRIC = "DERIVED_METRIC"
    COMPARISON_STATEMENT = "COMPARISON_STATEMENT"
    TREND_STATEMENT = "TREND_STATEMENT"
    BREAKDOWN_STATEMENT = "BREAKDOWN_STATEMENT"
    RISK_STATEMENT = "RISK_STATEMENT"
    FORECAST_STATEMENT = "FORECAST_STATEMENT"
    GOVERNANCE_NOTE = "GOVERNANCE_NOTE"
    LIMITATION_NOTE = "LIMITATION_NOTE"
    INSUFFICIENCY_NOTE = "INSUFFICIENCY_NOTE"
