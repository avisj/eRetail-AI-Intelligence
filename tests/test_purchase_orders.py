"""Unit and Integration Tests for Purchase Order Proposal Engine (Phase 4B-2)."""

from datetime import date
import pandas as pd
import pytest
from pydantic import ValidationError

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
from commerce_ai.recommendations.purchase_orders import (
    determine_proposal_urgency,
    determine_proposal_expected_delivery_date,
    generate_deterministic_proposal_id,
    generate_proposal_rationale,
    PurchaseOrderProposalService,
)


def _make_sample_rec(
    sku_id: str = "SKU_001",
    warehouse_id: str = "WH_01",
    supplier_id: str = "SUPP_A",
    recommended_order_qty: int = 100,
    recommendation_required: bool = True,
    unit_cost: float = 25.0,
    currency: str = "USD",
    lead_time_days: float = 7.0,
    urgency: str = "MEDIUM",
    risk_category: str = "UNDERSTOCK",
    constraints_applied: list = None,
) -> ReplenishmentRecommendation:
    """Helper to create a valid ReplenishmentRecommendation for testing."""
    return ReplenishmentRecommendation(
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        supplier_id=supplier_id,
        recommendation_required=recommendation_required,
        urgency=urgency,
        risk_category=risk_category,
        on_hand=50,
        reserved=10,
        on_order=0,
        net_inventory_position=40,
        forecast_daily_demand=10.0,
        lead_time_days=lead_time_days,
        reorder_point=80.0,
        target_stock_level=140.0,
        shortfall_qty=40,
        required_qty=100,
        recommended_order_qty=recommended_order_qty,
        unit_cost=unit_cost,
        estimated_order_cost=recommended_order_qty * unit_cost if unit_cost is not None else None,
        currency=currency,
        rationale="Sample test replenishment rationale.",
        constraints_applied=constraints_applied or [],
        status="COMPLETED",
    )


class TestPurchaseOrderProposalEngine:
    """Comprehensive test suite covering all 20 required edge cases for Phase 4B-2."""

    def test_single_recommendation_to_draft_po(self):
        """Case 1: Single recommendation produces exactly one draft PO proposal."""
        rec = _make_sample_rec(
            sku_id="SKU_1",
            warehouse_id="WH_1",
            supplier_id="SUPP_1",
            recommended_order_qty=100,
            unit_cost=50.0,
            currency="USD",
            lead_time_days=10.0,
            urgency="HIGH",
        )
        service = PurchaseOrderProposalService()
        result = service.generate([rec], proposal_date="2024-03-01")

        assert result.proposal_count == 1
        assert len(result.proposals) == 1
        prop = result.proposals[0]

        assert prop.supplier_id == "SUPP_1"
        assert prop.warehouse_id == "WH_1"
        assert prop.currency == "USD"
        assert prop.status == ProposalStatus.DRAFT.value
        assert prop.approval_required is True
        assert prop.total_quantity == 100
        assert prop.total_value == 5000.0
        assert prop.cost_data_status == CostDataStatus.COMPLETE_COST_DATA.value
        assert prop.expected_delivery_date == "2024-03-11"  # 2024-03-01 + 10 days
        assert prop.urgency == UrgencyLevel.HIGH.value
        assert len(prop.lines) == 1
        assert prop.lines[0].sku_id == "SKU_1"

    def test_multiple_skus_same_supplier_warehouse_consolidated(self):
        """Case 2: Multiple SKUs for same supplier and warehouse are consolidated into one proposal."""
        recs = [
            _make_sample_rec(sku_id="SKU_A", supplier_id="SUPP_1", warehouse_id="WH_1", recommended_order_qty=100, unit_cost=10.0),
            _make_sample_rec(sku_id="SKU_B", supplier_id="SUPP_1", warehouse_id="WH_1", recommended_order_qty=200, unit_cost=20.0),
            _make_sample_rec(sku_id="SKU_C", supplier_id="SUPP_1", warehouse_id="WH_1", recommended_order_qty=50, unit_cost=30.0),
        ]
        service = PurchaseOrderProposalService()
        result = service.generate(recs, proposal_date="2024-01-10")

        assert result.proposal_count == 1
        prop = result.proposals[0]
        assert prop.supplier_id == "SUPP_1"
        assert prop.warehouse_id == "WH_1"
        assert len(prop.lines) == 3
        assert prop.total_quantity == 350
        assert prop.total_value == (100 * 10.0) + (200 * 20.0) + (50 * 30.0)  # 1000 + 4000 + 1500 = 6500.0
        assert set(prop.source_recommendation_ids) == {
            "REC_SKU_A_WH_1",
            "REC_SKU_B_WH_1",
            "REC_SKU_C_WH_1",
        }

    def test_different_suppliers_separate_proposals(self):
        """Case 3: Recommendations with different suppliers form distinct proposals."""
        recs = [
            _make_sample_rec(sku_id="SKU_1", supplier_id="SUPP_ALPHA", warehouse_id="WH_1", recommended_order_qty=50),
            _make_sample_rec(sku_id="SKU_2", supplier_id="SUPP_BETA", warehouse_id="WH_1", recommended_order_qty=75),
        ]
        service = PurchaseOrderProposalService()
        result = service.generate(recs)

        assert result.proposal_count == 2
        suppliers = {p.supplier_id for p in result.proposals}
        assert suppliers == {"SUPP_ALPHA", "SUPP_BETA"}

    def test_different_warehouses_separate_proposals(self):
        """Case 4: Recommendations with different warehouses form distinct proposals."""
        recs = [
            _make_sample_rec(sku_id="SKU_1", supplier_id="SUPP_1", warehouse_id="WH_EAST", recommended_order_qty=50),
            _make_sample_rec(sku_id="SKU_1", supplier_id="SUPP_1", warehouse_id="WH_WEST", recommended_order_qty=80),
        ]
        service = PurchaseOrderProposalService()
        result = service.generate(recs)

        assert result.proposal_count == 2
        warehouses = {p.warehouse_id for p in result.proposals}
        assert warehouses == {"WH_EAST", "WH_WEST"}

    def test_different_currencies_separate_proposals(self):
        """Case 5: Recommendations with different currencies must never be merged into one proposal."""
        recs = [
            _make_sample_rec(sku_id="SKU_USD", supplier_id="SUPP_INTL", warehouse_id="WH_1", currency="USD", recommended_order_qty=40),
            _make_sample_rec(sku_id="SKU_EUR", supplier_id="SUPP_INTL", warehouse_id="WH_1", currency="EUR", recommended_order_qty=60),
        ]
        service = PurchaseOrderProposalService()
        result = service.generate(recs)

        assert result.proposal_count == 2
        currencies = {p.currency for p in result.proposals}
        assert currencies == {"USD", "EUR"}

    def test_missing_supplier_excluded(self):
        """Case 6: Recommendations missing supplier ID are excluded into excluded_missing_supplier."""
        rec_valid = _make_sample_rec(sku_id="SKU_VALID", supplier_id="SUPP_1", recommended_order_qty=50)
        rec_no_supp = _make_sample_rec(sku_id="SKU_NO_SUPP", supplier_id=None, recommended_order_qty=50)
        rec_empty_supp = _make_sample_rec(sku_id="SKU_EMPTY_SUPP", supplier_id="", recommended_order_qty=50)

        service = PurchaseOrderProposalService()
        result = service.generate([rec_valid, rec_no_supp, rec_empty_supp])

        assert result.proposal_count == 1
        assert len(result.excluded_missing_supplier) == 2
        excluded_skus = {r.sku_id for r in result.excluded_missing_supplier}
        assert excluded_skus == {"SKU_NO_SUPP", "SKU_EMPTY_SUPP"}

    def test_recommendation_not_required_excluded_from_po_lines(self):
        """Case 7: recommendation_required=False is excluded and kept in non_actionable_recommendations."""
        rec_not_req = _make_sample_rec(sku_id="SKU_NOT_REQ", recommendation_required=False, recommended_order_qty=0)
        rec_valid = _make_sample_rec(sku_id="SKU_REQ", recommendation_required=True, recommended_order_qty=100)

        service = PurchaseOrderProposalService()
        result = service.generate([rec_not_req, rec_valid])

        assert result.proposal_count == 1
        assert result.actionable_line_count == 1
        assert len(result.non_actionable_recommendations) == 1
        assert result.non_actionable_recommendations[0].sku_id == "SKU_NOT_REQ"

    def test_recommended_order_qty_zero_excluded(self):
        """Case 8: recommended_order_qty=0 is excluded and kept in excluded_zero_quantity."""
        rec_zero_qty = _make_sample_rec(sku_id="SKU_ZERO", recommendation_required=True, recommended_order_qty=0)
        service = PurchaseOrderProposalService()
        result = service.generate([rec_zero_qty])

        assert result.proposal_count == 0
        assert len(result.excluded_zero_quantity) == 1
        assert result.excluded_zero_quantity[0].sku_id == "SKU_ZERO"

    def test_missing_unit_cost_handled_safely(self):
        """Case 9: Missing unit cost sets estimated_line_value to None without inventing price."""
        rec_no_cost = _make_sample_rec(sku_id="SKU_NO_COST", unit_cost=None, recommended_order_qty=80)
        service = PurchaseOrderProposalService()
        result = service.generate([rec_no_cost])

        assert result.proposal_count == 1
        prop = result.proposals[0]
        assert prop.lines[0].unit_cost is None
        assert prop.lines[0].estimated_line_value is None
        assert prop.total_value is None
        assert prop.cost_data_status == CostDataStatus.NO_COST_DATA.value

    def test_partial_cost_data_clearly_flagged(self):
        """Case 10: Proposal with some costing and some missing is flagged PARTIAL_COST_DATA."""
        recs = [
            _make_sample_rec(sku_id="SKU_WITH_COST", unit_cost=25.0, recommended_order_qty=100),
            _make_sample_rec(sku_id="SKU_WITHOUT_COST", unit_cost=None, recommended_order_qty=50),
        ]
        service = PurchaseOrderProposalService()
        result = service.generate(recs)

        assert result.proposal_count == 1
        prop = result.proposals[0]
        assert prop.cost_data_status == CostDataStatus.PARTIAL_COST_DATA.value
        assert prop.total_value == 2500.0  # Sum of available values only
        assert result.cost_data_status == CostDataStatus.PARTIAL_COST_DATA.value

    def test_missing_currency_does_not_invent_default(self):
        """Case 11: Missing currency does not silently default to USD; flags financial incomplete."""
        rec_no_curr = _make_sample_rec(sku_id="SKU_NO_CURR", currency=None, unit_cost=20.0, recommended_order_qty=50)
        service = PurchaseOrderProposalService()
        result = service.generate([rec_no_curr])

        assert result.proposal_count == 1
        prop = result.proposals[0]
        assert prop.currency is None
        # With missing currency, valuation is flagged as partial/incomplete
        assert prop.cost_data_status == CostDataStatus.PARTIAL_COST_DATA.value

    def test_missing_lead_time_sets_delivery_date_none(self):
        """Case 12: Missing lead time sets expected_delivery_date to None without guessing."""
        rec_no_lt = _make_sample_rec(sku_id="SKU_NO_LT", lead_time_days=None, recommended_order_qty=50)
        service = PurchaseOrderProposalService()
        result = service.generate([rec_no_lt], proposal_date="2024-05-01")

        assert result.proposal_count == 1
        prop = result.proposals[0]
        assert prop.lines[0].expected_delivery_date is None
        assert prop.expected_delivery_date is None

    def test_multiple_urgency_levels_propagates_highest(self):
        """Case 13: Urgency propagation selects the highest rank (CRITICAL > HIGH > MEDIUM > LOW)."""
        recs = [
            _make_sample_rec(sku_id="SKU_MED", urgency=UrgencyLevel.MEDIUM.value),
            _make_sample_rec(sku_id="SKU_CRIT", urgency=UrgencyLevel.CRITICAL.value),
            _make_sample_rec(sku_id="SKU_LOW", urgency=UrgencyLevel.LOW.value),
        ]
        service = PurchaseOrderProposalService()
        result = service.generate(recs)

        assert result.proposal_count == 1
        prop = result.proposals[0]
        assert prop.urgency == UrgencyLevel.CRITICAL.value

        # High vs Medium
        recs_hm = [
            _make_sample_rec(sku_id="SKU_MED", urgency=UrgencyLevel.MEDIUM.value),
            _make_sample_rec(sku_id="SKU_HIGH", urgency=UrgencyLevel.HIGH.value),
        ]
        result_hm = service.generate(recs_hm)
        assert result_hm.proposals[0].urgency == UrgencyLevel.HIGH.value

    def test_deterministic_proposal_ids(self):
        """Case 14: Proposal IDs are 100% deterministic and reproducible across repeated runs."""
        recs = [
            _make_sample_rec(sku_id="SKU_1", supplier_id="SUPP_X", warehouse_id="WH_1", currency="USD"),
            _make_sample_rec(sku_id="SKU_2", supplier_id="SUPP_X", warehouse_id="WH_1", currency="USD"),
        ]
        service = PurchaseOrderProposalService()
        res1 = service.generate(recs, proposal_date="2024-06-01")
        res2 = service.generate(recs, proposal_date="2024-06-01")

        assert res1.proposals[0].proposal_id == res2.proposals[0].proposal_id
        assert "PROP_SUPP_X_WH_1_USD_" in res1.proposals[0].proposal_id

    def test_deterministic_proposal_output_and_rationale(self):
        """Case 15: Exact identical proposals and rationales generated on repeated calls."""
        recs = [
            _make_sample_rec(sku_id="SKU_1", supplier_id="SUPP_A", warehouse_id="WH_1", recommended_order_qty=100, urgency="HIGH"),
            _make_sample_rec(sku_id="SKU_2", supplier_id="SUPP_A", warehouse_id="WH_1", recommended_order_qty=200, urgency="MEDIUM"),
        ]
        service = PurchaseOrderProposalService()
        res1 = service.generate(recs, proposal_date="2024-06-01")
        res2 = service.generate(recs, proposal_date="2024-06-01")

        prop1 = res1.proposals[0]
        prop2 = res2.proposals[0]

        assert prop1.to_dict() == prop2.to_dict()
        expected_rationale = (
            "Draft PO proposal groups 2 replenishment recommendations for supplier SUPP_A and warehouse WH_1. "
            "Total proposed quantity is 300 units across 2 SKUs. The highest replenishment urgency is HIGH."
        )
        assert prop1.rationale == expected_rationale

    def test_source_recommendation_traceability(self):
        """Case 16: Source recommendation IDs are linked 1:1 to lines and listed on proposal."""
        rec1 = _make_sample_rec(sku_id="SKU_1", warehouse_id="WH_1")
        rec1.recommendation_id = "REC_CUSTOM_001"
        rec2 = _make_sample_rec(sku_id="SKU_2", warehouse_id="WH_1")
        rec2.recommendation_id = "REC_CUSTOM_002"

        service = PurchaseOrderProposalService()
        result = service.generate([rec1, rec2])

        prop = result.proposals[0]
        assert prop.lines[0].source_recommendation_id == "REC_CUSTOM_001"
        assert prop.lines[1].source_recommendation_id == "REC_CUSTOM_002"
        assert prop.source_recommendation_ids == ["REC_CUSTOM_001", "REC_CUSTOM_002"]

    def test_multiple_proposals_in_portfolio(self):
        """Case 17: Portfolio with diverse suppliers, warehouses, and currencies generates distinct POs."""
        recs = [
            _make_sample_rec(sku_id="S1", supplier_id="SUPP_1", warehouse_id="WH_A", currency="USD", recommended_order_qty=10),
            _make_sample_rec(sku_id="S2", supplier_id="SUPP_1", warehouse_id="WH_B", currency="USD", recommended_order_qty=20),
            _make_sample_rec(sku_id="S3", supplier_id="SUPP_2", warehouse_id="WH_A", currency="USD", recommended_order_qty=30),
            _make_sample_rec(sku_id="S4", supplier_id="SUPP_2", warehouse_id="WH_A", currency="EUR", recommended_order_qty=40),
        ]
        service = PurchaseOrderProposalService()
        result = service.generate(recs)

        assert result.proposal_count == 4
        assert result.actionable_line_count == 4
        assert result.total_proposed_quantity == 100

        # DataFrame exports
        proposals_df = result.to_dataframe()
        assert len(proposals_df) == 4
        assert "proposal_id" in proposals_df.columns
        assert "total_quantity" in proposals_df.columns

        lines_df = result.lines_to_dataframe()
        assert len(lines_df) == 4
        assert "proposal_id" in lines_df.columns
        assert "sku_id" in lines_df.columns

    def test_schema_rejects_negative_quantities(self):
        """Case 18: Pydantic schemas reject negative or zero quantities."""
        with pytest.raises(ValidationError):
            PurchaseOrderLineProposal(
                sku_id="SKU_1",
                warehouse_id="WH_1",
                supplier_id="SUPP_1",
                quantity=-5,  # Invalid negative quantity
                urgency="HIGH",
                risk_category="UNDERSTOCK",
                source_recommendation_id="REC_1",
                rationale="test",
            )

        with pytest.raises(ValidationError):
            PurchaseOrderProposal(
                proposal_id="PROP_1",
                supplier_id="SUPP_1",
                warehouse_id="WH_1",
                total_quantity=-10,  # Invalid negative total
                cost_data_status="NO_COST_DATA",
                urgency="HIGH",
                rationale="test",
            )

    def test_empty_recommendation_portfolio(self):
        """Case 19: Empty recommendation list gracefully returns empty result bundle."""
        service = PurchaseOrderProposalService()
        result = service.generate([])

        assert result.proposal_count == 0
        assert result.actionable_line_count == 0
        assert result.total_proposed_quantity == 0
        assert result.total_proposed_value is None
        assert result.cost_data_status == CostDataStatus.NO_COST_DATA.value
        assert result.to_dataframe().empty
        assert result.lines_to_dataframe().empty

    def test_human_approval_flag_always_true_and_draft_status(self):
        """Case 20: Every proposal strictly enforces status=DRAFT and approval_required=True."""
        recs = [
            _make_sample_rec(sku_id="SKU_A", supplier_id="SUPP_1", urgency="CRITICAL"),
            _make_sample_rec(sku_id="SKU_B", supplier_id="SUPP_2", urgency="LOW"),
        ]
        service = PurchaseOrderProposalService()
        result = service.generate(recs)

        for prop in result.proposals:
            assert prop.status == ProposalStatus.DRAFT.value
            assert prop.approval_required is True

    def test_end_to_end_from_replenishment_result(self):
        """Verify seamless integration consuming Phase 4B-1 ReplenishmentResult object."""
        replenishment_bundle = ReplenishmentResult(
            recommendations=[
                _make_sample_rec(sku_id="SKU_10", supplier_id="SUPP_X", warehouse_id="WH_1", recommended_order_qty=200, unit_cost=15.0),
                _make_sample_rec(sku_id="SKU_20", supplier_id="SUPP_X", warehouse_id="WH_1", recommended_order_qty=300, unit_cost=25.0),
                _make_sample_rec(sku_id="SKU_DORM", supplier_id="SUPP_X", warehouse_id="WH_1", recommendation_required=False, recommended_order_qty=0),
            ],
            summary={"total_entities_evaluated": 3},
        )

        products_df = pd.DataFrame([
            {"sku_id": "SKU_10", "product_name": "Ergonomic Desk", "currency": "USD"},
            {"sku_id": "SKU_20", "product_name": "Mesh Chair", "currency": "USD"},
        ])

        service = PurchaseOrderProposalService()
        result = service.generate(
            replenishment_bundle,
            proposal_date=date(2024, 4, 1),
            products=products_df,
        )

        assert result.proposal_count == 1
        prop = result.proposals[0]
        assert prop.total_quantity == 500
        assert prop.total_value == (200 * 15.0) + (300 * 25.0)  # 3000 + 7500 = 10500.0
        assert prop.lines[0].product_name == "Ergonomic Desk"
        assert prop.lines[1].product_name == "Mesh Chair"
        assert len(result.non_actionable_recommendations) == 1
