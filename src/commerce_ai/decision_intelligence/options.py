"""Deterministic Decision Options Generator for Decision Intelligence (Phase 6H).

Generates candidate decision options strictly grounded in evidence from Phase 6G recommendations.
Guarantees:
- Strictly adheres to the controlled decision option taxonomy
- NEVER selects a winning or optimal option (selected_option remains None)
- Does not recommend autonomous execution, purchase order submission, transfer execution, or pricing changes
- Only introduces transfer or replenishment options when backed by deterministic upstream engine output
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from commerce_ai.decision_intelligence.schemas import (
    DecisionOption,
    DecisionOptionType,
    OptionReversibility,
    generate_option_id,
)
from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendation,
    RecommendationType,
)


class DecisionOptionGenerator:
    """Deterministic generator of permissible decision options for human review."""

    def generate_options(
        self,
        decision_id: str,
        rec: BusinessRecommendation,
    ) -> List[DecisionOption]:
        """Generate permissible decision options based on recommendation type and evidence."""
        rec_type = rec.recommendation_type
        op = rec.operational_context
        fin = rec.financial_context

        options: List[DecisionOption] = []

        if rec_type == RecommendationType.REPLENISHMENT_REVIEW:
            options.extend(self._build_replenishment_options(decision_id, rec))
        elif rec_type == RecommendationType.PURCHASE_ORDER_REVIEW:
            options.extend(self._build_purchase_order_options(decision_id, rec))
        elif rec_type == RecommendationType.WAREHOUSE_REBALANCING_REVIEW:
            options.extend(self._build_transfer_options(decision_id, rec))
        elif rec_type in (
            RecommendationType.SLOW_MOVING_INVENTORY_REVIEW,
            RecommendationType.HIGH_VALUE_INVENTORY_REVIEW,
            RecommendationType.INVENTORY_REVIEW,
        ):
            options.extend(self._build_inventory_options(decision_id, rec))
        elif rec_type == RecommendationType.STOCKOUT_REVIEW:
            options.extend(self._build_stockout_options(decision_id, rec))
        elif rec_type in (
            RecommendationType.RETURN_REVIEW,
            RecommendationType.RETURN_ROOT_CAUSE_REVIEW,
        ):
            options.extend(self._build_return_options(decision_id, rec))
        elif rec_type == RecommendationType.MARGIN_REVIEW:
            options.extend(self._build_margin_options(decision_id, rec))
        elif rec_type == RecommendationType.DATA_QUALITY_REVIEW:
            options.extend(self._build_data_quality_options(decision_id, rec))
        elif rec_type == RecommendationType.FORECAST_REVIEW:
            options.extend(self._build_forecast_options(decision_id, rec))
        else:
            options.extend(self._build_generic_options(decision_id, rec))

        return options

    # -------------------------------------------------------------------------
    # Option Builders by Domain Classification
    # -------------------------------------------------------------------------

    def _build_replenishment_options(
        self, decision_id: str, rec: BusinessRecommendation
    ) -> List[DecisionOption]:
        op = rec.operational_context
        fin = rec.financial_context
        qty = op.recommended_order_qty if op and op.recommended_order_qty is not None else rec.recommended_quantity
        rop = op.reorder_point if op and op.reorder_point is not None else "ROP"
        pos = op.inventory_position if op and op.inventory_position is not None else "current"

        val_str = f" (${fin.proposed_purchase_value:,.2f})" if fin and fin.proposed_purchase_value else ""

        opt_review = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.REPLENISHMENT_REVIEW),
            option_type=DecisionOptionType.REPLENISHMENT_REVIEW,
            title="Review and Validate Replenishment Sizing",
            description=(
                f"Review the solved order sizing of {qty} units{val_str} against inventory position "
                f"({pos}) and reorder point ({rop}). Does not place or submit a purchase order."
            ),
            supporting_evidence=rec.evidence,
            expected_effect=(
                "May prevent stock depletion and support product availability; may require capital outlay "
                "and warehouse receiving capacity."
            ),
            financial_context=fin,
            operational_context=op,
            risks=["Requires supplier confirmation of lead time and minimum order quantity."],
            dependencies=["Supplier contract review", "Purchasing budget approval"],
            reversibility=OptionReversibility.MEDIUM,
            requires_external_action=True,
            recommended_by_engine=False,
        )

        opt_defer = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEFER_DECISION),
            option_type=DecisionOptionType.DEFER_DECISION,
            title="Defer Replenishment Decision",
            description="Defer ordering until the next demand observation cycle or inventory snapshot.",
            supporting_evidence=[],
            expected_effect=(
                "May preserve near-term cash flow; could increase risk of stockout if customer demand accelerates."
            ),
            financial_context=fin,
            operational_context=op,
            risks=["Potential stockout risk if lead time is long."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_no_change = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.NO_CHANGE),
            option_type=DecisionOptionType.NO_CHANGE,
            title="Maintain Current Inventory Without Ordering",
            description="Reject the replenishment recommendation and maintain existing stock levels.",
            supporting_evidence=[],
            expected_effect=(
                "Avoids purchase capital expenditure; could result in unmet customer orders if stock exhausts."
            ),
            financial_context=fin,
            operational_context=op,
            risks=["Unmitigated stockout exposure."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        return [opt_review, opt_defer, opt_no_change]

    def _build_purchase_order_options(
        self, decision_id: str, rec: BusinessRecommendation
    ) -> List[DecisionOption]:
        fin = rec.financial_context
        op = rec.operational_context
        val_str = f"${fin.proposed_purchase_value:,.2f}" if fin and fin.proposed_purchase_value else "estimated value"
        qty_str = f"{op.recommended_order_qty} units" if op and op.recommended_order_qty else "consolidated volume"

        opt_review = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.PURCHASE_ORDER_REVIEW),
            option_type=DecisionOptionType.PURCHASE_ORDER_REVIEW,
            title="Review Draft Purchase Order Proposal",
            description=(
                f"Review consolidated purchase order proposal covering {qty_str} with estimated valuation of {val_str}. "
                f"Does not transmit or submit order to supplier or ERP."
            ),
            supporting_evidence=rec.evidence,
            expected_effect=(
                "May satisfy multi-SKU replenishment needs with consolidated freight; requires procurement authorization."
            ),
            financial_context=fin,
            operational_context=op,
            risks=["Requires purchasing budget authorization and supplier order confirmation."],
            dependencies=["Procurement budget approval", "Supplier terms verification"],
            reversibility=OptionReversibility.MEDIUM,
            requires_external_action=True,
            recommended_by_engine=False,
        )

        opt_defer = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEFER_DECISION),
            option_type=DecisionOptionType.DEFER_DECISION,
            title="Defer Purchase Order Review",
            description="Defer PO proposal bundle review to the subsequent procurement cycle.",
            supporting_evidence=[],
            expected_effect=(
                "May allow consolidation of additional line items; could delay receiving schedule."
            ),
            financial_context=fin,
            operational_context=op,
            risks=["Potential delay in inventory replenishment."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_no_change = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.NO_CHANGE),
            option_type=DecisionOptionType.NO_CHANGE,
            title="Reject Purchase Order Proposal (No Change)",
            description="Dismiss draft proposal bundle without issuing purchase orders.",
            supporting_evidence=[],
            expected_effect="Maintains current procurement commitments without new capital deployment.",
            financial_context=fin,
            operational_context=op,
            risks=["Underlying SKU replenishment requirements remain unaddressed."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        return [opt_review, opt_defer, opt_no_change]

    def _build_transfer_options(
        self, decision_id: str, rec: BusinessRecommendation
    ) -> List[DecisionOption]:
        op = rec.operational_context
        fin = rec.financial_context
        src = op.source_warehouse if op and op.source_warehouse else "surplus facility"
        dest = op.destination_warehouse if op and op.destination_warehouse else "deficit facility"
        qty = op.transfer_quantity if op and op.transfer_quantity is not None else rec.recommended_quantity

        opt_review = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.TRANSFER_REVIEW),
            option_type=DecisionOptionType.TRANSFER_REVIEW,
            title="Review Inter-Warehouse Stock Transfer",
            description=(
                f"Review proposed transfer of {qty} units from {src} to {dest}. "
                f"Does not initiate logistics movement or freight booking."
            ),
            supporting_evidence=rec.evidence,
            expected_effect=(
                "May relieve deficit at receiving location using surplus stock; may incur freight cost and handling overhead."
            ),
            financial_context=fin,
            operational_context=op,
            risks=["Route transit capacity and warehouse receiving dock availability must be verified."],
            dependencies=["Freight carrier scheduling", "Warehouse bay clearance"],
            reversibility=OptionReversibility.MEDIUM,
            requires_external_action=True,
            recommended_by_engine=False,
        )

        opt_defer = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEFER_DECISION),
            option_type=DecisionOptionType.DEFER_DECISION,
            title="Defer Transfer Decision",
            description="Postpone transfer review until local demand trends confirm persistent deficit.",
            supporting_evidence=[],
            expected_effect="Preserves freight budget; could allow destination stockout to persist.",
            financial_context=fin,
            operational_context=op,
            risks=["Continued stock deficit at destination."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_no_change = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.NO_CHANGE),
            option_type=DecisionOptionType.NO_CHANGE,
            title="Cancel Transfer Proposal (No Change)",
            description="Maintain stock at source facility without inter-warehouse transfer.",
            supporting_evidence=[],
            expected_effect="Eliminates transfer freight costs; destination must rely on standard replenishment.",
            financial_context=fin,
            operational_context=op,
            risks=["Persistent imbalance across regional facilities."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        return [opt_review, opt_defer, opt_no_change]

    def _build_inventory_options(
        self, decision_id: str, rec: BusinessRecommendation
    ) -> List[DecisionOption]:
        op = rec.operational_context
        fin = rec.financial_context

        opt_inv = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.INVENTORY_REVIEW),
            option_type=DecisionOptionType.INVENTORY_REVIEW,
            title="Conduct Operational Inventory Audit",
            description="Audit physical inventory counts, storage locations, and carrying parameters.",
            supporting_evidence=rec.evidence,
            expected_effect=(
                "May verify physical inventory accuracy and identify disposition or storage reallocation options."
            ),
            financial_context=fin,
            operational_context=op,
            risks=["Requires warehouse cycle counting labor."],
            dependencies=["Warehouse floor audit access"],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        options = [opt_inv]

        # ONLY expose transfer option if a valid transfer recommendation or transfer quantity exists!
        # Strictly adheres to Section 12: "Do not invent a transfer recommendation."
        if op and op.transfer_quantity is not None and op.transfer_quantity > 0:
            src = op.source_warehouse or "surplus warehouse"
            dest = op.destination_warehouse or "deficit warehouse"
            opt_tr = DecisionOption(
                option_id=generate_option_id(decision_id, DecisionOptionType.TRANSFER_REVIEW),
                option_type=DecisionOptionType.TRANSFER_REVIEW,
                title="Review Warehouse Transfer Opportunity",
                description=(
                    f"Evaluate transfer of {op.transfer_quantity} units from {src} to {dest} based on existing transfer output."
                ),
                supporting_evidence=rec.evidence,
                expected_effect="May alleviate local holding excess by rebalancing stock to deficit location.",
                financial_context=fin,
                operational_context=op,
                risks=["Incurs transit freight and handling costs."],
                dependencies=["Logistics carrier confirmation"],
                reversibility=OptionReversibility.MEDIUM,
                requires_external_action=True,
                recommended_by_engine=False,
            )
            options.append(opt_tr)

        opt_defer = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEFER_DECISION),
            option_type=DecisionOptionType.DEFER_DECISION,
            title="Defer Inventory Disposition Decision",
            description="Postpone inventory adjustments pending subsequent sales velocity observations.",
            supporting_evidence=[],
            expected_effect="Avoids operational disruption; inventory carrying cost continues to accrue.",
            financial_context=fin,
            operational_context=op,
            risks=["Holding cost accumulation on slow-moving capital."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_no_change = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.NO_CHANGE),
            option_type=DecisionOptionType.NO_CHANGE,
            title="Maintain Current Inventory Strategy (No Change)",
            description="Keep current inventory holding profile unchanged.",
            supporting_evidence=[],
            expected_effect="No change to operational workflow or catalog disposition.",
            financial_context=fin,
            operational_context=op,
            risks=["Working capital remains tied up in existing stock."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        options.extend([opt_defer, opt_no_change])
        return options

    def _build_stockout_options(
        self, decision_id: str, rec: BusinessRecommendation
    ) -> List[DecisionOption]:
        op = rec.operational_context
        fin = rec.financial_context
        options: List[DecisionOption] = []

        # Only expose replenishment option if deterministic engine output exists!
        has_replenish = op and op.recommended_order_qty is not None and op.recommended_order_qty > 0
        has_transfer = op and op.transfer_quantity is not None and op.transfer_quantity > 0

        if has_replenish:
            opt_rep = DecisionOption(
                option_id=generate_option_id(decision_id, DecisionOptionType.REPLENISHMENT_REVIEW),
                option_type=DecisionOptionType.REPLENISHMENT_REVIEW,
                title="Review Replenishment Sizing to Mitigate Stockout",
                description=(
                    f"Evaluate proposed replenishment order of {op.recommended_order_qty} units to restore stock coverage."
                ),
                supporting_evidence=rec.evidence,
                expected_effect="May eliminate stockout and restore order fulfillment capability upon arrival.",
                financial_context=fin,
                operational_context=op,
                risks=["Subject to supplier lead time; will not resolve immediate same-day deficit."],
                dependencies=["Supplier lead time confirmation"],
                reversibility=OptionReversibility.MEDIUM,
                requires_external_action=True,
                recommended_by_engine=False,
            )
            options.append(opt_rep)

        if has_transfer:
            opt_tr = DecisionOption(
                option_id=generate_option_id(decision_id, DecisionOptionType.TRANSFER_REVIEW),
                option_type=DecisionOptionType.TRANSFER_REVIEW,
                title="Review Transfer from Surplus Warehouse to Mitigate Stockout",
                description=(
                    f"Evaluate rapid stock transfer of {op.transfer_quantity} units from {op.source_warehouse} to resolve stockout."
                ),
                supporting_evidence=rec.evidence,
                expected_effect="May expedite replenishment using internal network inventory with shorter transit time.",
                financial_context=fin,
                operational_context=op,
                risks=["Incurs transit shipping costs."],
                dependencies=["Expedited freight scheduling"],
                reversibility=OptionReversibility.MEDIUM,
                requires_external_action=True,
                recommended_by_engine=False,
            )
            options.append(opt_tr)

        # If neither replenishment nor transfer solver output exists:
        if not has_replenish and not has_transfer:
            opt_dem = DecisionOption(
                option_id=generate_option_id(decision_id, DecisionOptionType.DEMAND_REVIEW),
                option_type=DecisionOptionType.DEMAND_REVIEW,
                title="Review Demand Run-Rate & Lost Sales Impact",
                description=(
                    "Investigate historical demand patterns and customer substitution during the stockout period."
                ),
                supporting_evidence=rec.evidence,
                expected_effect="May refine unconstrained demand estimates and calibrate future safety stock buffers.",
                financial_context=fin,
                operational_context=op,
                risks=["Does not directly replenish stock; informational audit only."],
                dependencies=[],
                reversibility=OptionReversibility.HIGH,
                requires_external_action=False,
                recommended_by_engine=False,
            )
            opt_rev = DecisionOption(
                option_id=generate_option_id(decision_id, DecisionOptionType.REVIEW_ONLY),
                option_type=DecisionOptionType.REVIEW_ONLY,
                title="Review Operational Stockout Status",
                description="Acknowledge and monitor stockout duration without initiating unsized orders.",
                supporting_evidence=rec.evidence,
                expected_effect="Ensures operational visibility while awaiting solved replenishment proposal.",
                financial_context=fin,
                operational_context=op,
                risks=["Stockout persists until formal replenishment is solved."],
                dependencies=[],
                reversibility=OptionReversibility.HIGH,
                requires_external_action=False,
                recommended_by_engine=False,
            )
            options.extend([opt_dem, opt_rev])

        opt_defer = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEFER_DECISION),
            option_type=DecisionOptionType.DEFER_DECISION,
            title="Defer Stockout Decision",
            description="Defer decision pending upcoming replenishment batch run.",
            supporting_evidence=[],
            expected_effect="Allows next scheduled solver run to evaluate network stock.",
            financial_context=fin,
            operational_context=op,
            risks=["Potential revenue loss during deferral window."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )
        options.append(opt_defer)
        return options

    def _build_return_options(
        self, decision_id: str, rec: BusinessRecommendation
    ) -> List[DecisionOption]:
        op = rec.operational_context
        fin = rec.financial_context

        opt_ret = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.RETURN_REVIEW),
            option_type=DecisionOptionType.RETURN_REVIEW,
            title="Review Product Return Drivers & Processing",
            description="Audit return disposition logs, customer feedback, and warehouse inspection notes.",
            supporting_evidence=rec.evidence,
            expected_effect="May identify handling defects, packaging damage, or operational processing bottlenecks.",
            financial_context=fin,
            operational_context=op,
            risks=["Requires manual inspection of return warehouse triage logs."],
            dependencies=["Access to return processing logs"],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_dem = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEMAND_REVIEW),
            option_type=DecisionOptionType.DEMAND_REVIEW,
            title="Audit Catalog Specifications & Sizing Feedback",
            description="Review online product listings, size charts, and specification accuracy against return reasons.",
            supporting_evidence=rec.evidence,
            expected_effect="May identify listing ambiguities that contribute to customer mismatch.",
            financial_context=fin,
            operational_context=op,
            risks=["Does not alter physical product."],
            dependencies=["Merchandising catalog access"],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_data = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DATA_REVIEW),
            option_type=DecisionOptionType.DATA_REVIEW,
            title="Review Return Reason Categorization Quality",
            description="Audit accuracy and consistency of customer return reason categorization codes.",
            supporting_evidence=rec.evidence,
            expected_effect="May improve telemetry granularity for future return root-cause analysis.",
            financial_context=fin,
            operational_context=op,
            risks=["Data hygiene task only."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_defer = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEFER_DECISION),
            option_type=DecisionOptionType.DEFER_DECISION,
            title="Defer Return Investigation",
            description="Postpone review until additional return transactions accrue.",
            supporting_evidence=[],
            expected_effect="Accumulates more statistical sample before committing review resources.",
            financial_context=fin,
            operational_context=op,
            risks=["Elevated return processing costs continue during deferral."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        return [opt_ret, opt_dem, opt_data, opt_defer]

    def _build_margin_options(
        self, decision_id: str, rec: BusinessRecommendation
    ) -> List[DecisionOption]:
        op = rec.operational_context
        fin = rec.financial_context

        opt_mar = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.MARGIN_REVIEW),
            option_type=DecisionOptionType.MARGIN_REVIEW,
            title="Review Product Cost Structure & Operating Margins",
            description=(
                "Examine wholesale acquisition costs, supplier contract terms, and operational fee deductions. "
                "Does NOT recommend or alter consumer catalog prices."
            ),
            supporting_evidence=rec.evidence,
            expected_effect="May identify component cost inflation or unallocated fee overhead.",
            financial_context=fin,
            operational_context=op,
            risks=["Requires procurement and vendor agreement review."],
            dependencies=["Supplier contract data access"],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_inv = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.INVENTORY_REVIEW),
            option_type=DecisionOptionType.INVENTORY_REVIEW,
            title="Review Holding Duration & Carrying Cost Drag",
            description="Evaluate whether prolonged inventory holding is compressing realized product margin.",
            supporting_evidence=rec.evidence,
            expected_effect="May identify carrying cost drag on product profitability.",
            financial_context=fin,
            operational_context=op,
            risks=["Informational review only."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_data = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DATA_REVIEW),
            option_type=DecisionOptionType.DATA_REVIEW,
            title="Review Cost Completeness & Allocation",
            description="Verify completeness of unit cost, freight, and return processing cost attribution.",
            supporting_evidence=rec.evidence,
            expected_effect="May clarify true unit economics and eliminate data estimation gaps.",
            financial_context=fin,
            operational_context=op,
            risks=["Data audit task."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_defer = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEFER_DECISION),
            option_type=DecisionOptionType.DEFER_DECISION,
            title="Defer Margin Review",
            description="Defer margin investigation to the scheduled quarterly category review.",
            supporting_evidence=[],
            expected_effect="Maintains current operational focus; margin compression persists.",
            financial_context=fin,
            operational_context=op,
            risks=["Continued margin drag."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        return [opt_mar, opt_inv, opt_data, opt_defer]

    def _build_data_quality_options(
        self, decision_id: str, rec: BusinessRecommendation
    ) -> List[DecisionOption]:
        op = rec.operational_context
        fin = rec.financial_context

        opt_data = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DATA_REVIEW),
            option_type=DecisionOptionType.DATA_REVIEW,
            title="Remediate Missing Master Data Attributes",
            description=(
                "Update missing fields in the master catalog or system of record to restore algorithmic coverage."
            ),
            supporting_evidence=rec.evidence,
            expected_effect="May restore algorithmic calculations and enable downstream solver recommendations.",
            financial_context=fin,
            operational_context=op,
            risks=["Requires data entry verification in originating ERP/PIM."],
            dependencies=["Master data maintenance access"],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=True,
            recommended_by_engine=False,
        )

        opt_defer = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEFER_DECISION),
            option_type=DecisionOptionType.DEFER_DECISION,
            title="Defer Data Remediation",
            description="Defer master data updates to scheduled bulk synchronization cycle.",
            supporting_evidence=[],
            expected_effect="Postpones manual maintenance; downstream calculations remain suppressed or estimated.",
            financial_context=fin,
            operational_context=op,
            risks=["Downstream solver coverage remains impaired."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        return [opt_data, opt_defer]

    def _build_forecast_options(
        self, decision_id: str, rec: BusinessRecommendation
    ) -> List[DecisionOption]:
        op = rec.operational_context
        fin = rec.financial_context

        opt_dem = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEMAND_REVIEW),
            option_type=DecisionOptionType.DEMAND_REVIEW,
            title="Review Forward Demand Forecast Coverage",
            description="Evaluate forecast projections against warehouse stock horizons and lead times.",
            supporting_evidence=rec.evidence,
            expected_effect="May identify forward stock coverage gaps before they manifest as operational stockouts.",
            financial_context=fin,
            operational_context=op,
            risks=["Subject to forecasting model error and prediction intervals."],
            dependencies=["Forecasting service model outputs"],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_defer = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEFER_DECISION),
            option_type=DecisionOptionType.DEFER_DECISION,
            title="Defer Forecast Coverage Adjustment",
            description="Postpone action until subsequent forecast retraining cycle.",
            supporting_evidence=[],
            expected_effect="Awaits more recent demand signals before adjusting coverage plans.",
            financial_context=fin,
            operational_context=op,
            risks=["May miss lead-time ordering window if demand increases."],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        return [opt_dem, opt_defer]

    def _build_generic_options(
        self, decision_id: str, rec: BusinessRecommendation
    ) -> List[DecisionOption]:
        fin = rec.financial_context
        op = rec.operational_context

        opt_rev = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.REVIEW_ONLY),
            option_type=DecisionOptionType.REVIEW_ONLY,
            title="Review Operational Context",
            description="Conduct operational review of current entity status.",
            supporting_evidence=rec.evidence,
            expected_effect="Improves situational awareness of operational conditions.",
            financial_context=fin,
            operational_context=op,
            risks=[],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        opt_defer = DecisionOption(
            option_id=generate_option_id(decision_id, DecisionOptionType.DEFER_DECISION),
            option_type=DecisionOptionType.DEFER_DECISION,
            title="Defer Decision",
            description="Defer decision to subsequent monitoring cycle.",
            supporting_evidence=[],
            expected_effect="Maintains current operations.",
            financial_context=fin,
            operational_context=op,
            risks=[],
            dependencies=[],
            reversibility=OptionReversibility.HIGH,
            requires_external_action=False,
            recommended_by_engine=False,
        )

        return [opt_rev, opt_defer]
