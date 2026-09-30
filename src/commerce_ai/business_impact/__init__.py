"""Business Impact & Opportunity Quantification Module (Phase 6F).

Quantifies observed and estimated financial and operational exposures across
cross-domain commercial signals, providing auditable measurement without prescription.
"""

from __future__ import annotations

from commerce_ai.business_impact.schemas import (
    BusinessImpactConfig,
    BusinessImpactPortfolioSummary,
    BusinessImpactRecord,
    BusinessImpactResult,
    CalculationStatus,
    ImpactCategory,
    ImpactConfidence,
    ImpactType,
)
from commerce_ai.business_impact.quantification import (
    calculate_physical_capital_exposure,
    deduplicate_portfolio_exposure,
    generate_impact_id,
    get_economic_dimension,
    quantify_signal_impact,
)
from commerce_ai.business_impact.service import BusinessImpactService

__all__ = [
    "BusinessImpactConfig",
    "BusinessImpactPortfolioSummary",
    "BusinessImpactRecord",
    "BusinessImpactResult",
    "BusinessImpactService",
    "CalculationStatus",
    "ImpactCategory",
    "ImpactConfidence",
    "ImpactType",
    "calculate_physical_capital_exposure",
    "deduplicate_portfolio_exposure",
    "generate_impact_id",
    "get_economic_dimension",
    "quantify_signal_impact",
]
