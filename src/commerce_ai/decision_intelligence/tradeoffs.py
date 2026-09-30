"""Deterministic Trade-Off Modeling for Decision Intelligence (Phase 6H).

Constructs objective, multi-dimensional trade-off evaluations for candidate decision options.
Strict requirements:
- Strictly non-causal, conditional language ("may", "could", "depends on")
- Absolute prohibition against definitive claims ("will happen", "guarantees")
- Transparent evaluation across dimensions: INVENTORY_CAPITAL, SERVICE_LEVEL, HOLDING_COST,
  OPERATIONAL_WORKLOAD, MARGIN_PROTECTION, DATA_INTEGRITY
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from commerce_ai.decision_intelligence.schemas import (
    DecisionOption,
    DecisionOptionType,
    DecisionTradeOff,
    DecisionTradeOffDimension,
    generate_tradeoff_id,
)
from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendation,
    RecommendationType,
)


class DecisionTradeOffEvaluator:
    """Evaluates multi-dimensional trade-offs for candidate decision options."""

    def evaluate_tradeoffs(
        self,
        decision_id: str,
        options: List[DecisionOption],
        rec: BusinessRecommendation,
    ) -> List[DecisionTradeOff]:
        """Construct deterministic trade-offs for each option in a decision package."""
        tradeoffs: List[DecisionTradeOff] = []
        evidence_ids = [e.source_id for e in rec.evidence if e.source_id]

        for opt in options:
            t_list = self._evaluate_option_tradeoffs(decision_id, opt, rec, evidence_ids)
            tradeoffs.extend(t_list)

        return tradeoffs

    def _evaluate_option_tradeoffs(
        self,
        decision_id: str,
        option: DecisionOption,
        rec: BusinessRecommendation,
        evidence_ids: List[str],
    ) -> List[DecisionTradeOff]:
        opt_type = option.option_type
        t_list: List[DecisionTradeOff] = []

        if opt_type == DecisionOptionType.REPLENISHMENT_REVIEW:
            t_list.append(
                DecisionTradeOff(
                    tradeoff_id=generate_tradeoff_id(decision_id, option.option_id, DecisionTradeOffDimension.SERVICE_LEVEL),
                    decision_id=decision_id,
                    option_id=option.option_id,
                    dimension=DecisionTradeOffDimension.SERVICE_LEVEL,
                    positive_effect="May restore order fill rates and reduce customer stockout occurrences upon stock arrival.",
                    negative_effect="Incurs purchase capital expenditure and commits warehouse inbound receiving bandwidth.",
                    uncertainty="Depends on actual supplier delivery lead time and unconstrained customer demand velocity.",
                    evidence_ids=evidence_ids,
                )
            )
            t_list.append(
                DecisionTradeOff(
                    tradeoff_id=generate_tradeoff_id(decision_id, option.option_id, DecisionTradeOffDimension.INVENTORY_CAPITAL),
                    decision_id=decision_id,
                    option_id=option.option_id,
                    dimension=DecisionTradeOffDimension.INVENTORY_CAPITAL,
                    positive_effect="May protect revenue generation across high-performing sales channels.",
                    negative_effect="May tie up working capital in inventory holding if demand decelerates.",
                    uncertainty="Depends on demand stability and cash flow availability.",
                    evidence_ids=evidence_ids,
                )
            )

        elif opt_type == DecisionOptionType.PURCHASE_ORDER_REVIEW:
            t_list.append(
                DecisionTradeOff(
                    tradeoff_id=generate_tradeoff_id(decision_id, option.option_id, DecisionTradeOffDimension.OPERATIONAL_WORKLOAD),
                    decision_id=decision_id,
                    option_id=option.option_id,
                    dimension=DecisionTradeOffDimension.OPERATIONAL_WORKLOAD,
                    positive_effect="May realize freight efficiencies through multi-SKU purchase order consolidation.",
                    negative_effect="Requires procurement review, budget sign-off, and vendor transmission effort.",
                    uncertainty="Depends on supplier minimum order quantities (MOQ) and purchasing authorization thresholds.",
                    evidence_ids=evidence_ids,
                )
            )

        elif opt_type == DecisionOptionType.TRANSFER_REVIEW:
            t_list.append(
                DecisionTradeOff(
                    tradeoff_id=generate_tradeoff_id(decision_id, option.option_id, DecisionTradeOffDimension.INVENTORY_CAPITAL),
                    decision_id=decision_id,
                    option_id=option.option_id,
                    dimension=DecisionTradeOffDimension.INVENTORY_CAPITAL,
                    positive_effect="May relieve destination inventory deficit utilizing existing network surplus without new vendor spend.",
                    negative_effect="May incur inter-warehouse freight transit expenses and source warehouse handling costs.",
                    uncertainty="Depends on carrier route rates and destination receiving dock capacity.",
                    evidence_ids=evidence_ids,
                )
            )

        elif opt_type == DecisionOptionType.INVENTORY_REVIEW:
            t_list.append(
                DecisionTradeOff(
                    tradeoff_id=generate_tradeoff_id(decision_id, option.option_id, DecisionTradeOffDimension.HOLDING_COST),
                    decision_id=decision_id,
                    option_id=option.option_id,
                    dimension=DecisionTradeOffDimension.HOLDING_COST,
                    positive_effect="May identify physical discrepancies, damaged units, or slow-moving stock eligible for disposition.",
                    negative_effect="Consumes warehouse staff time for physical counting and inventory reconciliation.",
                    uncertainty="Depends on cycle counting accuracy and warehouse labor availability.",
                    evidence_ids=evidence_ids,
                )
            )

        elif opt_type == DecisionOptionType.RETURN_REVIEW:
            t_list.append(
                DecisionTradeOff(
                    tradeoff_id=generate_tradeoff_id(decision_id, option.option_id, DecisionTradeOffDimension.RETURN_RATE),
                    decision_id=decision_id,
                    option_id=option.option_id,
                    dimension=DecisionTradeOffDimension.RETURN_RATE,
                    positive_effect="May identify recurring product defects or packaging issues to curb future return rate drag.",
                    negative_effect="Does not recover historical return processing costs already incurred.",
                    uncertainty="Depends on whether returns stem from product quality issues or customer preference shifts.",
                    evidence_ids=evidence_ids,
                )
            )

        elif opt_type == DecisionOptionType.MARGIN_REVIEW:
            t_list.append(
                DecisionTradeOff(
                    tradeoff_id=generate_tradeoff_id(decision_id, option.option_id, DecisionTradeOffDimension.MARGIN_PROTECTION),
                    decision_id=decision_id,
                    option_id=option.option_id,
                    dimension=DecisionTradeOffDimension.MARGIN_PROTECTION,
                    positive_effect="May highlight supplier pricing discrepancies or excessive logistical fees compressing margin.",
                    negative_effect="Does not provide immediate margin expansion without contract renegotiation.",
                    uncertainty="Depends on supplier contract flexibility and vendor negotiation power.",
                    evidence_ids=evidence_ids,
                )
            )

        elif opt_type == DecisionOptionType.DATA_REVIEW:
            t_list.append(
                DecisionTradeOff(
                    tradeoff_id=generate_tradeoff_id(decision_id, option.option_id, DecisionTradeOffDimension.DATA_INTEGRITY),
                    decision_id=decision_id,
                    option_id=option.option_id,
                    dimension=DecisionTradeOffDimension.DATA_INTEGRITY,
                    positive_effect="May restore algorithmic coverage and ensure downstream solver reliability.",
                    negative_effect="Requires manual data inspection and cross-system reconciliation.",
                    uncertainty="Depends on master data availability in originating enterprise records.",
                    evidence_ids=evidence_ids,
                )
            )

        elif opt_type == DecisionOptionType.DEFER_DECISION:
            t_list.append(
                DecisionTradeOff(
                    tradeoff_id=generate_tradeoff_id(decision_id, option.option_id, DecisionTradeOffDimension.OPERATIONAL_WORKLOAD),
                    decision_id=decision_id,
                    option_id=option.option_id,
                    dimension=DecisionTradeOffDimension.OPERATIONAL_WORKLOAD,
                    positive_effect="Allows deliberate observation and avoids premature capital or labor commitment.",
                    negative_effect="May allow operational conditions (stockouts, return rates, holding costs) to persist or deteriorate.",
                    uncertainty="Depends on demand rate changes and market volatility during the deferral period.",
                    evidence_ids=evidence_ids,
                )
            )

        elif opt_type == DecisionOptionType.NO_CHANGE:
            t_list.append(
                DecisionTradeOff(
                    tradeoff_id=generate_tradeoff_id(decision_id, option.option_id, DecisionTradeOffDimension.INVENTORY_CAPITAL),
                    decision_id=decision_id,
                    option_id=option.option_id,
                    dimension=DecisionTradeOffDimension.INVENTORY_CAPITAL,
                    positive_effect="Maintains current operational stability and avoids immediate financial outlay.",
                    negative_effect="Rejects recommendation rationale; underlying risk factors remain unaddressed.",
                    uncertainty="Depends on whether external demand and operational conditions remain stationary.",
                    evidence_ids=evidence_ids,
                )
            )

        else:
            t_list.append(
                DecisionTradeOff(
                    tradeoff_id=generate_tradeoff_id(decision_id, option.option_id, DecisionTradeOffDimension.OPERATIONAL_WORKLOAD),
                    decision_id=decision_id,
                    option_id=option.option_id,
                    dimension=DecisionTradeOffDimension.OPERATIONAL_WORKLOAD,
                    positive_effect="Provides human oversight before committing organizational resources.",
                    negative_effect="Requires reviewer time and analytical validation.",
                    uncertainty="Depends on reviewer expertise and available telemetry.",
                    evidence_ids=evidence_ids,
                )
            )

        return t_list
