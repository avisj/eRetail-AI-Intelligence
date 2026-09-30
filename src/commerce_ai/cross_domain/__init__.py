"""Cross-Domain Business Intelligence Module (Phase 6E).

Unified analytical intelligence layer connecting sales, revenue, gross margin,
operational economics, inventory positions, demand velocity, stockouts, forecasts,
returns, replenishment, and warehouse dynamics.
"""

from __future__ import annotations

from commerce_ai.cross_domain.schemas import (
    CrossDomainBusinessRecord,
    CrossDomainDimension,
    CrossDomainIntelligenceConfig,
    CrossDomainIntelligenceResult,
    CrossDomainPortfolioSummary,
    CrossDomainSignal,
    SignalCategory,
    SignalSeverity,
)
from commerce_ai.cross_domain.signals import (
    generate_cross_domain_signals,
    generate_signal_id,
)
from commerce_ai.cross_domain.service import CrossDomainIntelligenceService

__all__ = [
    "CrossDomainBusinessRecord",
    "CrossDomainDimension",
    "CrossDomainIntelligenceConfig",
    "CrossDomainIntelligenceResult",
    "CrossDomainPortfolioSummary",
    "CrossDomainSignal",
    "CrossDomainIntelligenceService",
    "SignalCategory",
    "SignalSeverity",
    "generate_cross_domain_signals",
    "generate_signal_id",
]
