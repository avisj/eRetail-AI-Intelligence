"""Decision Intelligence & Action Planning (Phase 6H).

Provides structured, auditable decision packages, candidate options, multi-dimensional trade-offs,
and qualitative risk evaluations for human business reviewers.

Core guarantees:
- approval_required=True, execution_allowed=False
- selected_option=None (strictly no automated winner selection)
- Complete machine-readable evidence and provenance traceability
- Non-causal, conditional explainability without LLMs
"""

from commerce_ai.decision_intelligence.options import DecisionOptionGenerator
from commerce_ai.decision_intelligence.risks import DecisionRiskEvaluator
from commerce_ai.decision_intelligence.schemas import (
    DecisionBusinessContext,
    DecisionEntityContext,
    DecisionIntelligenceConfig,
    DecisionIntelligenceResult,
    DecisionOption,
    DecisionOptionType,
    DecisionPackage,
    DecisionRiskFlag,
    DecisionStatus,
    DecisionTraceability,
    DecisionTradeOff,
    DecisionTradeOffDimension,
    InformationAvailability,
    OptionReversibility,
    RequiredDecisionInformation,
    RiskFlagType,
    RiskSeverity,
    generate_decision_id,
    generate_info_id,
    generate_option_id,
    generate_risk_id,
    generate_tradeoff_id,
)
from commerce_ai.decision_intelligence.service import DecisionIntelligenceService
from commerce_ai.decision_intelligence.templates import (
    format_decision_summary,
    format_decision_title,
)
from commerce_ai.decision_intelligence.tradeoffs import DecisionTradeOffEvaluator

__all__ = [
    "DecisionBusinessContext",
    "DecisionEntityContext",
    "DecisionIntelligenceConfig",
    "DecisionIntelligenceResult",
    "DecisionIntelligenceService",
    "DecisionOption",
    "DecisionOptionGenerator",
    "DecisionOptionType",
    "DecisionPackage",
    "DecisionRiskEvaluator",
    "DecisionRiskFlag",
    "DecisionStatus",
    "DecisionTraceability",
    "DecisionTradeOff",
    "DecisionTradeOffDimension",
    "DecisionTradeOffEvaluator",
    "InformationAvailability",
    "OptionReversibility",
    "RequiredDecisionInformation",
    "RiskFlagType",
    "RiskSeverity",
    "format_decision_summary",
    "format_decision_title",
    "generate_decision_id",
    "generate_info_id",
    "generate_option_id",
    "generate_risk_id",
    "generate_tradeoff_id",
]
