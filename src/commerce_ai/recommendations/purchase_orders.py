"""Purchase Order Proposal Engine (Phase 4B-2).

Converts approved replenishment recommendations into consolidated draft Purchase Order Proposals
grouped by (supplier_id, warehouse_id, currency) without creating or submitting real purchase orders.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import hashlib
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from commerce_ai.recommendations.schemas import (
    CostDataStatus,
    ProposalStatus,
    PurchaseOrderLineProposal,
    PurchaseOrderProposal,
    PurchaseOrderProposalResult,
    ReplenishmentRecommendation,
    ReplenishmentResult,
    UrgencyLevel,
)

# Relative ranking for deterministic urgency propagation
URGENCY_PRECEDENCE = {
    UrgencyLevel.CRITICAL.value: 4,
    UrgencyLevel.HIGH.value: 3,
    UrgencyLevel.MEDIUM.value: 2,
    UrgencyLevel.LOW.value: 1,
}


def determine_proposal_urgency(lines: List[PurchaseOrderLineProposal]) -> str:
    """Propagate the highest urgency level among constituent lines.

    Precedence order: CRITICAL > HIGH > MEDIUM > LOW.
    """
    if not lines:
        return UrgencyLevel.LOW.value
    highest_score = max(URGENCY_PRECEDENCE.get(line.urgency, 1) for line in lines)
    for name, score in URGENCY_PRECEDENCE.items():
        if score == highest_score:
            return name
    return UrgencyLevel.LOW.value


def determine_proposal_expected_delivery_date(
    lines: List[PurchaseOrderLineProposal],
) -> Optional[str]:
    """Determine the consolidated dock delivery date across constituent lines.

    Returns the maximum delivery date string (ISO YYYY-MM-DD) among lines with known dates,
    or None if no line has an expected delivery date.
    """
    dates = [line.expected_delivery_date for line in lines if line.expected_delivery_date is not None]
    if not dates:
        return None
    return max(dates)


def generate_deterministic_proposal_id(
    supplier_id: str,
    warehouse_id: str,
    currency: Optional[str],
    source_recommendation_ids: List[str],
    proposal_date: Optional[str] = None,
) -> str:
    """Generate a reproducible, deterministic proposal ID without random UUIDs.

    Derives a stable cryptographic short-digest over sorted recommendation IDs,
    the grouping key, and the proposal reference date.
    """
    sorted_ids = sorted(source_recommendation_ids)
    date_part = proposal_date or "NODATE"
    curr_part = currency or "NOCURR"
    seed = f"{supplier_id}|{warehouse_id}|{curr_part}|{date_part}|{','.join(sorted_ids)}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8].upper()
    curr_suffix = f"_{currency}" if currency else ""
    return f"PROP_{supplier_id}_{warehouse_id}{curr_suffix}_{digest}"


def generate_proposal_rationale(
    supplier_id: str,
    warehouse_id: str,
    lines: List[PurchaseOrderLineProposal],
    total_quantity: int,
    urgency: str,
) -> str:
    """Build a deterministic, explainable rationale for the consolidated proposal."""
    lines_count = len(lines)
    skus_count = len({line.sku_id for line in lines})
    rec_word = "recommendation" if lines_count == 1 else "recommendations"
    sku_word = "SKU" if skus_count == 1 else "SKUs"

    return (
        f"Draft PO proposal groups {lines_count} replenishment {rec_word} for supplier {supplier_id} "
        f"and warehouse {warehouse_id}. Total proposed quantity is {total_quantity} units across "
        f"{skus_count} {sku_word}. The highest replenishment urgency is {urgency}."
    )


class PurchaseOrderProposalService:
    """High-level service orchestrating draft Purchase Order Proposal generation.

    Strictly produces DRAFT proposals requiring human approval.
    Does NOT submit, approve, or transmit orders to external OMS/ERP systems.
    """

    def generate(
        self,
        replenishment_input: Union[ReplenishmentResult, List[ReplenishmentRecommendation], pd.DataFrame],
        proposal_date: Optional[Union[date, str, datetime]] = None,
        products: Optional[pd.DataFrame] = None,
        suppliers: Optional[pd.DataFrame] = None,
        default_currency: Optional[str] = None,
    ) -> PurchaseOrderProposalResult:
        """Convert replenishment recommendations into consolidated draft PO proposals.

        Args:
            replenishment_input: Phase 4B-1 ReplenishmentResult, list of ReplenishmentRecommendation,
                or recommendations DataFrame.
            proposal_date: Optional reference date for proposal creation and lead-time arrival projection.
                If None, expected_delivery_date is set to None.
            products: Optional product master DataFrame (for product_name, currency lookup).
            suppliers: Optional supplier master DataFrame.
            default_currency: Optional fallback currency. If None and catalog has no currency,
                currency remains None without silent assumptions.

        Returns:
            PurchaseOrderProposalResult containing grouped draft proposals and audit categorizations.
        """
        # 1. Normalize input recommendations
        recommendations: List[ReplenishmentRecommendation] = []
        if isinstance(replenishment_input, ReplenishmentResult):
            recommendations = replenishment_input.recommendations
        elif isinstance(replenishment_input, list):
            recommendations = replenishment_input
        elif isinstance(replenishment_input, pd.DataFrame):
            if not replenishment_input.empty:
                for _, row in replenishment_input.iterrows():
                    d = row.to_dict()
                    # Filter out NaN values for optional fields
                    clean_d = {k: v for k, v in d.items() if not pd.isna(v)}
                    recommendations.append(ReplenishmentRecommendation(**clean_d))

        # 2. Normalize proposal date
        prop_dt: Optional[date] = None
        if isinstance(proposal_date, (datetime, pd.Timestamp)):
            prop_dt = proposal_date.date()
        elif isinstance(proposal_date, date):
            prop_dt = proposal_date
        elif isinstance(proposal_date, str) and proposal_date.strip():
            try:
                prop_dt = date.fromisoformat(proposal_date.strip())
            except ValueError:
                prop_dt = None

        prop_date_str = prop_dt.isoformat() if prop_dt is not None else None

        # 3. Build product master lookups (product_name, currency)
        product_names: Dict[str, str] = {}
        product_currencies: Dict[str, str] = {}
        if products is not None and not products.empty:
            if "sku_id" in products.columns and "product_name" in products.columns:
                product_names = products.set_index("sku_id")["product_name"].dropna().to_dict()
            if "sku_id" in products.columns and "currency" in products.columns:
                product_currencies = products.set_index("sku_id")["currency"].dropna().to_dict()

        # 4. Filter and categorize recommendations
        actionable_candidates: List[Tuple[ReplenishmentRecommendation, PurchaseOrderLineProposal]] = []
        excluded_missing_supplier: List[ReplenishmentRecommendation] = []
        excluded_zero_quantity: List[ReplenishmentRecommendation] = []
        non_actionable_recommendations: List[ReplenishmentRecommendation] = []

        for rec in recommendations:
            # Check if replenishment was required
            if not rec.recommendation_required:
                non_actionable_recommendations.append(rec)
                continue

            # Check if order quantity is positive
            if rec.recommended_order_qty <= 0:
                excluded_zero_quantity.append(rec)
                non_actionable_recommendations.append(rec)
                continue

            # Check if supplier ID is present
            supp_id = rec.supplier_id
            if not supp_id or not str(supp_id).strip():
                excluded_missing_supplier.append(rec)
                continue

            supp_id_str = str(supp_id).strip()

            # Resolve currency
            curr = rec.currency or product_currencies.get(rec.sku_id) or default_currency
            curr_clean = str(curr).strip().upper() if curr and str(curr).strip() else None

            # Resolve product name
            prod_name = product_names.get(rec.sku_id)

            # Calculate expected delivery date if proposal_date and lead_time are available
            expected_delivery_dt: Optional[str] = None
            if prop_dt is not None and rec.lead_time_days is not None and rec.lead_time_days > 0.0:
                lead_days = int(math.ceil(rec.lead_time_days))
                expected_delivery_dt = (prop_dt + timedelta(days=lead_days)).isoformat()

            # Calculate line valuation
            line_val: Optional[float] = None
            if rec.unit_cost is not None and rec.unit_cost >= 0.0:
                line_val = round(float(rec.recommended_order_qty * rec.unit_cost), 2)

            source_rec_id = rec.effective_recommendation_id

            line_prop = PurchaseOrderLineProposal(
                sku_id=rec.sku_id,
                warehouse_id=rec.warehouse_id,
                supplier_id=supp_id_str,
                product_name=prod_name,
                quantity=rec.recommended_order_qty,
                unit_cost=rec.unit_cost,
                estimated_line_value=line_val,
                currency=curr_clean,
                lead_time_days=rec.lead_time_days,
                expected_delivery_date=expected_delivery_dt,
                urgency=rec.urgency,
                risk_category=rec.risk_category,
                source_recommendation_id=source_rec_id,
                rationale=rec.rationale,
            )

            actionable_candidates.append((rec, line_prop))

        # 5. Group actionable lines by (supplier_id, warehouse_id, currency)
        grouped_candidates: Dict[Tuple[str, str, Optional[str]], List[Tuple[ReplenishmentRecommendation, PurchaseOrderLineProposal]]] = {}
        for rec, line in actionable_candidates:
            group_key = (line.supplier_id, line.warehouse_id, line.currency)
            if group_key not in grouped_candidates:
                grouped_candidates[group_key] = []
            grouped_candidates[group_key].append((rec, line))

        # 6. Build PurchaseOrderProposal for each group
        proposals: List[PurchaseOrderProposal] = []
        for (supp_id, wh_id, curr), pair_list in grouped_candidates.items():
            lines = [pair[1] for pair in pair_list]
            recs = [pair[0] for pair in pair_list]

            total_qty = sum(line.quantity for line in lines)
            source_rec_ids = [line.source_recommendation_id for line in lines]

            # Calculate financial valuation and cost completeness
            available_values = [line.estimated_line_value for line in lines if line.estimated_line_value is not None]
            if len(available_values) == len(lines) and len(lines) > 0 and curr is not None:
                total_val = round(sum(available_values), 2)
                cost_status = CostDataStatus.COMPLETE_COST_DATA.value
            elif len(available_values) > 0:
                total_val = round(sum(available_values), 2)
                cost_status = CostDataStatus.PARTIAL_COST_DATA.value
            else:
                total_val = None
                cost_status = CostDataStatus.NO_COST_DATA.value

            # If currency is missing, even with unit costs, financial valuation is partial/incomplete
            if curr is None and cost_status == CostDataStatus.COMPLETE_COST_DATA.value:
                cost_status = CostDataStatus.PARTIAL_COST_DATA.value

            # Urgency propagation (highest rank among lines)
            prop_urgency = determine_proposal_urgency(lines)

            # Consolidated delivery date
            prop_deliv_date = determine_proposal_expected_delivery_date(lines)

            # Union of constraints applied across constituent recommendations
            all_constraints: Set[str] = set()
            for r in recs:
                all_constraints.update(r.constraints_applied)
            sorted_constraints = sorted(list(all_constraints))

            # Deterministic proposal ID
            proposal_id = generate_deterministic_proposal_id(
                supplier_id=supp_id,
                warehouse_id=wh_id,
                currency=curr,
                source_recommendation_ids=source_rec_ids,
                proposal_date=prop_date_str,
            )

            # Deterministic rationale
            rationale = generate_proposal_rationale(
                supplier_id=supp_id,
                warehouse_id=wh_id,
                lines=lines,
                total_quantity=total_qty,
                urgency=prop_urgency,
            )

            proposal = PurchaseOrderProposal(
                proposal_id=proposal_id,
                supplier_id=supp_id,
                warehouse_id=wh_id,
                proposal_date=prop_date_str,
                currency=curr,
                lines=lines,
                total_quantity=total_qty,
                total_value=total_val,
                cost_data_status=cost_status,
                expected_delivery_date=prop_deliv_date,
                urgency=prop_urgency,
                status=ProposalStatus.DRAFT.value,
                approval_required=True,
                source_recommendation_ids=source_rec_ids,
                rationale=rationale,
                constraints_applied=sorted_constraints,
            )
            proposals.append(proposal)

        # 7. Overall portfolio cost data status
        overall_cost_status: str
        if not proposals:
            overall_cost_status = CostDataStatus.NO_COST_DATA.value
        elif all(p.cost_data_status == CostDataStatus.COMPLETE_COST_DATA.value for p in proposals):
            overall_cost_status = CostDataStatus.COMPLETE_COST_DATA.value
        elif any(p.cost_data_status in (CostDataStatus.COMPLETE_COST_DATA.value, CostDataStatus.PARTIAL_COST_DATA.value) for p in proposals):
            overall_cost_status = CostDataStatus.PARTIAL_COST_DATA.value
        else:
            overall_cost_status = CostDataStatus.NO_COST_DATA.value

        total_proposed_qty = sum(p.total_quantity for p in proposals)
        available_prop_values = [p.total_value for p in proposals if p.total_value is not None]
        total_proposed_val = round(sum(available_prop_values), 2) if available_prop_values else None

        summary = {
            "proposal_count": len(proposals),
            "actionable_line_count": len(actionable_candidates),
            "excluded_missing_supplier_count": len(excluded_missing_supplier),
            "excluded_zero_quantity_count": len(excluded_zero_quantity),
            "non_actionable_count": len(non_actionable_recommendations),
            "total_proposed_quantity": total_proposed_qty,
            "total_proposed_value": total_proposed_val,
            "overall_cost_data_status": overall_cost_status,
            "proposals_by_urgency": {
                UrgencyLevel.CRITICAL.value: sum(1 for p in proposals if p.urgency == UrgencyLevel.CRITICAL.value),
                UrgencyLevel.HIGH.value: sum(1 for p in proposals if p.urgency == UrgencyLevel.HIGH.value),
                UrgencyLevel.MEDIUM.value: sum(1 for p in proposals if p.urgency == UrgencyLevel.MEDIUM.value),
                UrgencyLevel.LOW.value: sum(1 for p in proposals if p.urgency == UrgencyLevel.LOW.value),
            },
        }

        return PurchaseOrderProposalResult(
            proposals=proposals,
            actionable_line_count=len(actionable_candidates),
            excluded_missing_supplier=excluded_missing_supplier,
            excluded_zero_quantity=excluded_zero_quantity,
            non_actionable_recommendations=non_actionable_recommendations,
            total_proposed_quantity=total_proposed_qty,
            total_proposed_value=total_proposed_val,
            proposal_count=len(proposals),
            cost_data_status=overall_cost_status,
            summary=summary,
        )
