"""Recommendations & Decision Engine (Phase 4B).

Provides deterministic, explainable replenishment recommendations, order quantity sizing,
supply constraints, and draft Purchase Order Proposal generation.
"""

from commerce_ai.recommendations.schemas import (
    UrgencyLevel,
    ReplenishmentConstraint,
    ProposalStatus,
    CostDataStatus,
    ReplenishmentConfig,
    ReplenishmentRecommendation,
    ReplenishmentResult,
    PurchaseOrderLineProposal,
    PurchaseOrderProposal,
    PurchaseOrderProposalResult,
    TransferPriority,
    TransferConstraint,
    WarehouseTransferRecommendation,
    WarehouseRebalancingConfig,
    WarehouseRebalancingResult,
)
from commerce_ai.recommendations.replenishment import (
    determine_urgency,
    generate_deterministic_rationale,
    solve_entity_replenishment,
    ReplenishmentSolver,
)
from commerce_ai.recommendations.purchase_orders import (
    determine_proposal_urgency,
    determine_proposal_expected_delivery_date,
    generate_deterministic_proposal_id,
    generate_proposal_rationale,
    PurchaseOrderProposalService,
)
from commerce_ai.recommendations.warehouse_rebalancing import (
    determine_transfer_priority,
    generate_deterministic_transfer_id,
    generate_transfer_rationale,
    WarehouseRebalancingService,
)

__all__ = [
    "UrgencyLevel",
    "ReplenishmentConstraint",
    "ProposalStatus",
    "CostDataStatus",
    "ReplenishmentConfig",
    "ReplenishmentRecommendation",
    "ReplenishmentResult",
    "PurchaseOrderLineProposal",
    "PurchaseOrderProposal",
    "PurchaseOrderProposalResult",
    "TransferPriority",
    "TransferConstraint",
    "WarehouseTransferRecommendation",
    "WarehouseRebalancingConfig",
    "WarehouseRebalancingResult",
    "determine_urgency",
    "generate_deterministic_rationale",
    "solve_entity_replenishment",
    "ReplenishmentSolver",
    "determine_proposal_urgency",
    "determine_proposal_expected_delivery_date",
    "generate_deterministic_proposal_id",
    "generate_proposal_rationale",
    "PurchaseOrderProposalService",
    "determine_transfer_priority",
    "generate_deterministic_transfer_id",
    "generate_transfer_rationale",
    "WarehouseRebalancingService",
]

