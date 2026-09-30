"""Unit and Integration Tests for Return Intervention / Cost-Utility Engine (Phase 5C-3)."""

from datetime import date
import pytest
import numpy as np
import pandas as pd

from commerce_ai.returns.schemas import (
    FinancialInputStatus,
    InterventionDecision,
    InterventionPolicyConfig,
    InterventionRecommendationCode,
    PortfolioInterventionSummary,
    ReturnInterventionRecommendation,
    ReturnRiskBand,
    ReturnRiskResult,
)
from commerce_ai.returns.intervention import (
    AVAILABLE_FINANCIAL_FIELDS,
    UNAVAILABLE_FINANCIAL_FIELDS,
    generate_intervention_id,
    generate_intervention_rationale,
    evaluate_order_line,
    evaluate_order_lines,
    summarize_portfolio,
    portfolio_breakdown_to_dataframe,
    ReturnInterventionService,
)


class TestFinancialInputAvailability:
    """Validate explicit financial field availability assessment."""

    def test_available_and_unavailable_field_definitions(self):
        assert "quantity" in AVAILABLE_FINANCIAL_FIELDS["sales"]
        assert "revenue" in AVAILABLE_FINANCIAL_FIELDS["sales"]
        assert "unit_cost" in AVAILABLE_FINANCIAL_FIELDS["products"]

        assert "return_shipping_cost" in UNAVAILABLE_FINANCIAL_FIELDS
        assert "handling_cost" in UNAVAILABLE_FINANCIAL_FIELDS
        assert "inspection_cost" in UNAVAILABLE_FINANCIAL_FIELDS
        assert "restocking_cost" in UNAVAILABLE_FINANCIAL_FIELDS
        assert "salvage_value" in UNAVAILABLE_FINANCIAL_FIELDS
        assert "processing_cost" in UNAVAILABLE_FINANCIAL_FIELDS
        assert "refund_amount" in UNAVAILABLE_FINANCIAL_FIELDS


class TestSingleLineInterventionEvaluation:
    """Test suite covering single order-line cost-utility evaluations and edge cases."""

    def test_sufficient_financial_inputs(self):
        # 1. Complete financial inputs including return costs
        record = {
            "sale_id": "SALE_101",
            "order_id": "ORD_501",
            "sku_id": "SKU_PRO_01",
            "warehouse_id": "WH_01",
            "channel_id": "CH_AMZ",
            "quantity": 2,
            "unit_price": 100.0,
            "revenue": 200.0,
            "unit_cost": 50.0,
            "return_shipping_cost": 15.0,
            "handling_cost": 5.0,
            "calibrated_return_probability": 0.30,
            "risk_band": "MEDIUM",
        }
        rec = evaluate_order_line(record)

        assert rec.financial_status == FinancialInputStatus.SUFFICIENT
        assert rec.estimated_product_cost == 100.0  # 2 * 50
        assert rec.estimated_gross_margin == 100.0  # 200 - 100
        assert rec.estimated_return_cost == 20.0    # 15 + 5
        assert rec.estimated_return_impact == 220.0 # 200 + 20
        assert rec.expected_return_exposure == 66.0 # 0.30 * 220.0
        assert rec.decision == InterventionDecision.INTERVENTION_INDICATED
        assert rec.recommendation_code == InterventionRecommendationCode.REVIEW_RETURN_RISK

    def test_missing_return_cost_inputs_non_strict(self):
        # 2a. Missing return costs under default non-strict policy (revenue proxy)
        record = {
            "sale_id": "SALE_102",
            "order_id": "ORD_502",
            "sku_id": "SKU_PRO_02",
            "warehouse_id": "WH_01",
            "channel_id": "CH_DIR",
            "quantity": 1,
            "unit_price": 150.0,
            "revenue": 150.0,
            "unit_cost": 75.0,
            "calibrated_return_probability": 0.25,
        }
        rec = evaluate_order_line(record)

        assert rec.financial_status == FinancialInputStatus.PARTIAL
        assert rec.estimated_return_cost is None
        assert rec.estimated_return_impact == 150.0
        assert rec.expected_return_exposure == 37.5  # 0.25 * 150
        assert "return_shipping_cost" in rec.missing_inputs
        assert rec.decision == InterventionDecision.INTERVENTION_INDICATED

    def test_missing_return_cost_inputs_strict(self):
        # 2b. Missing return costs under strict policy (requires explicit return cost)
        policy = InterventionPolicyConfig(require_return_costs=True)
        record = {
            "sale_id": "SALE_103",
            "order_id": "ORD_503",
            "sku_id": "SKU_PRO_03",
            "warehouse_id": "WH_01",
            "channel_id": "CH_DIR",
            "quantity": 1,
            "revenue": 150.0,
            "unit_cost": 75.0,
            "calibrated_return_probability": 0.30,
        }
        rec = evaluate_order_line(record, policy=policy)

        assert rec.decision == InterventionDecision.INSUFFICIENT_DATA
        assert rec.recommendation_code == InterventionRecommendationCode.INSUFFICIENT_FINANCIAL_INPUTS
        assert "Return probability is available, but return-specific cost inputs are missing." in rec.rationale
        assert rec.expected_return_exposure is None

    def test_missing_unit_cost(self):
        # 3. Missing unit cost
        record = {
            "sale_id": "SALE_104",
            "order_id": "ORD_504",
            "sku_id": "SKU_PRO_04",
            "quantity": 1,
            "revenue": 100.0,
            "calibrated_return_probability": 0.30,
        }
        rec = evaluate_order_line(record)

        assert rec.estimated_product_cost is None
        assert rec.estimated_gross_margin is None
        assert "unit_cost" in rec.missing_inputs

    def test_missing_probability(self):
        # 4. Missing probability
        record = {
            "sale_id": "SALE_105",
            "order_id": "ORD_505",
            "sku_id": "SKU_PRO_05",
            "quantity": 1,
            "revenue": 100.0,
            "unit_cost": 40.0,
            "calibrated_return_probability": None,
        }
        rec = evaluate_order_line(record)

        assert rec.decision == InterventionDecision.INSUFFICIENT_DATA
        assert rec.recommendation_code == InterventionRecommendationCode.INSUFFICIENT_PROBABILITY_DATA
        assert "Calibrated return probability is missing or invalid" in rec.rationale

    def test_invalid_probability_out_of_bounds(self):
        # 5. Invalid probability (< 0 or > 1)
        for bad_p in [-0.05, 1.25]:
            record = {
                "sale_id": "SALE_106",
                "order_id": "ORD_506",
                "sku_id": "SKU_PRO_06",
                "quantity": 1,
                "revenue": 100.0,
                "unit_cost": 40.0,
                "calibrated_return_probability": bad_p,
            }
            rec = evaluate_order_line(record)
            assert rec.decision == InterventionDecision.INSUFFICIENT_DATA
            assert rec.recommendation_code == InterventionRecommendationCode.DATA_QUALITY_ERROR
            assert "Invalid probability value" in rec.rationale

    def test_zero_quantity(self):
        # 6. Zero quantity
        record = {
            "sale_id": "SALE_107",
            "order_id": "ORD_507",
            "sku_id": "SKU_PRO_07",
            "quantity": 0,
            "unit_price": 50.0,
            "calibrated_return_probability": 0.25,
        }
        rec = evaluate_order_line(record)
        assert rec.decision == InterventionDecision.INSUFFICIENT_DATA
        assert rec.recommendation_code == InterventionRecommendationCode.DATA_QUALITY_ERROR
        assert "Invalid order quantity" in rec.rationale

    def test_negative_quantity(self):
        # 7. Negative quantity
        record = {
            "sale_id": "SALE_108",
            "order_id": "ORD_508",
            "sku_id": "SKU_PRO_08",
            "quantity": -3,
            "unit_price": 50.0,
            "calibrated_return_probability": 0.25,
        }
        rec = evaluate_order_line(record)
        assert rec.decision == InterventionDecision.INSUFFICIENT_DATA
        assert rec.recommendation_code == InterventionRecommendationCode.DATA_QUALITY_ERROR
        assert "Invalid order quantity" in rec.rationale

    def test_zero_revenue(self):
        # 8. Zero revenue (e.g. 100% promotion discount)
        record = {
            "sale_id": "SALE_109",
            "order_id": "ORD_509",
            "sku_id": "SKU_PRO_09",
            "quantity": 1,
            "unit_price": 50.0,
            "revenue": 0.0,
            "unit_cost": 0.0,
            "calibrated_return_probability": 0.40,
        }
        rec = evaluate_order_line(record)
        assert rec.revenue == 0.0
        assert rec.expected_return_exposure == 0.0
        assert rec.decision == InterventionDecision.NO_INTERVENTION_INDICATED

    def test_negative_margin(self):
        # 9. Negative gross margin (unit_cost > selling_price/revenue)
        record = {
            "sale_id": "SALE_110",
            "order_id": "ORD_510",
            "sku_id": "SKU_PRO_10",
            "quantity": 1,
            "unit_price": 30.0,
            "revenue": 30.0,
            "unit_cost": 50.0,  # margin = -$20
            "calibrated_return_probability": 0.35,
        }
        rec = evaluate_order_line(record)
        assert rec.estimated_gross_margin == -20.0
        assert rec.decision == InterventionDecision.REVIEW_REQUIRED
        assert rec.recommendation_code == InterventionRecommendationCode.NEGATIVE_MARGIN_REVIEW
        assert "negative gross margin" in rec.rationale.lower()

    def test_high_probability_high_exposure(self):
        # 10. High probability + High exposure (>= high_exposure_threshold = $75)
        record = {
            "sale_id": "SALE_111",
            "order_id": "ORD_511",
            "sku_id": "SKU_HIGH_01",
            "quantity": 2,
            "revenue": 300.0,
            "unit_cost": 100.0,
            "calibrated_return_probability": 0.35,  # exposure = 0.35 * 300 = $105 >= $75
        }
        rec = evaluate_order_line(record)
        assert rec.decision == InterventionDecision.INTERVENTION_INDICATED
        assert rec.recommendation_code == InterventionRecommendationCode.REVIEW_HIGH_VALUE_RETURN_EXPOSURE
        assert "exceeds the high-value policy threshold" in rec.rationale

    def test_high_probability_low_exposure(self):
        # 11. High probability + Low exposure (< minimum_expected_exposure = $25)
        record = {
            "sale_id": "SALE_112",
            "order_id": "ORD_512",
            "sku_id": "SKU_LOW_EXP",
            "quantity": 1,
            "revenue": 20.0,
            "unit_cost": 5.0,
            "calibrated_return_probability": 0.40,  # exposure = 0.40 * 20 = $8.0 < $25
        }
        rec = evaluate_order_line(record)
        assert rec.decision == InterventionDecision.NO_INTERVENTION_INDICATED
        assert rec.recommendation_code == InterventionRecommendationCode.NO_INTERVENTION_INDICATED
        assert "below the economic intervention threshold" in rec.rationale

    def test_low_probability_high_exposure(self):
        # 12. Low probability (< minimum_probability = 0.20) + High potential exposure
        record = {
            "sale_id": "SALE_113",
            "order_id": "ORD_513",
            "sku_id": "SKU_EXPENSIVE",
            "quantity": 1,
            "revenue": 1000.0,
            "unit_cost": 400.0,
            "calibrated_return_probability": 0.05,  # prob 5% < 20%
        }
        rec = evaluate_order_line(record)
        assert rec.decision == InterventionDecision.NO_INTERVENTION_INDICATED
        assert rec.recommendation_code == InterventionRecommendationCode.NO_INTERVENTION_INDICATED
        assert "Return probability is below the configured intervention threshold." in rec.rationale

    def test_low_probability_low_exposure(self):
        # 13. Low probability + Low exposure
        record = {
            "sale_id": "SALE_114",
            "order_id": "ORD_514",
            "sku_id": "SKU_CHEAP",
            "quantity": 1,
            "revenue": 15.0,
            "unit_cost": 5.0,
            "calibrated_return_probability": 0.04,
        }
        rec = evaluate_order_line(record)
        assert rec.decision == InterventionDecision.NO_INTERVENTION_INDICATED
        assert rec.recommendation_code == InterventionRecommendationCode.NO_INTERVENTION_INDICATED

    def test_insufficient_financial_inputs(self):
        # 14. Completely missing financial revenue and unit price
        record = {
            "sale_id": "SALE_115",
            "order_id": "ORD_515",
            "sku_id": "SKU_UNKNOWN_FIN",
            "calibrated_return_probability": 0.35,
        }
        rec = evaluate_order_line(record)
        assert rec.decision == InterventionDecision.INSUFFICIENT_DATA
        assert rec.recommendation_code == InterventionRecommendationCode.INSUFFICIENT_FINANCIAL_INPUTS
        assert "Financial inputs are insufficient" in rec.rationale

    def test_partial_financial_inputs(self):
        # 15. Partial financial inputs
        record = {
            "sale_id": "SALE_116",
            "order_id": "ORD_516",
            "sku_id": "SKU_PARTIAL",
            "quantity": 1,
            "revenue": 100.0,
            "unit_cost": 50.0,
            "calibrated_return_probability": 0.25,
        }
        rec = evaluate_order_line(record)
        assert rec.financial_status == FinancialInputStatus.PARTIAL
        assert "return_shipping_cost" in rec.missing_inputs

    def test_deterministic_ids(self):
        # 16. Deterministic IDs
        rec1 = {
            "sale_id": "SALE_999",
            "order_id": "ORD_999",
            "sku_id": "SKU_TEST",
            "calibrated_return_probability": 0.25,
        }
        id1 = generate_intervention_id("SALE_999", "SKU_TEST", "ORD_999", 0.25)
        id2 = generate_intervention_id("SALE_999", "SKU_TEST", "ORD_999", 0.25)
        id3 = generate_intervention_id("SALE_999", "SKU_TEST", "ORD_999", 0.35)

        assert id1 == id2
        assert id1 != id3
        assert id1.startswith("INTV_SKU_TEST_")

    def test_deterministic_rationale(self):
        # 17. Deterministic rationale
        r1 = generate_intervention_rationale(
            decision=InterventionDecision.NO_INTERVENTION_INDICATED,
            code=InterventionRecommendationCode.NO_INTERVENTION_INDICATED,
            calibrated_probability=0.08,
        )
        assert r1 == "Return probability is below the configured intervention threshold."

    def test_policy_configuration(self):
        # 18. Policy configuration
        custom_policy = InterventionPolicyConfig(
            policy_version="custom-v2.0",
            minimum_probability=0.40,
            minimum_expected_exposure=100.0,
            high_exposure_threshold=200.0,
        )
        record = {
            "sale_id": "SALE_117",
            "order_id": "ORD_517",
            "sku_id": "SKU_POLICY",
            "quantity": 1,
            "revenue": 200.0,
            "unit_cost": 80.0,
            "calibrated_return_probability": 0.30,  # 30% < 40% policy cutoff
        }
        rec = evaluate_order_line(record, policy=custom_policy)
        assert rec.policy_version == "custom-v2.0"
        assert rec.decision == InterventionDecision.NO_INTERVENTION_INDICATED

    def test_as_of_date_leakage_protection(self):
        # 19. as_of_date leakage protection
        record = {
            "sale_id": "SALE_FUTURE",
            "order_id": "ORD_FUTURE",
            "sku_id": "SKU_FUTURE",
            "date": "2026-05-01",
            "revenue": 100.0,
            "unit_cost": 50.0,
            "calibrated_return_probability": 0.35,
        }
        rec = evaluate_order_line(record, as_of_date="2025-12-31")
        assert rec.decision == InterventionDecision.INSUFFICIENT_DATA
        assert rec.recommendation_code == InterventionRecommendationCode.DATA_QUALITY_ERROR
        assert "future leakage rejected" in rec.rationale

    def test_duplicate_handling_in_batch(self):
        # 20. Duplicate handling
        batch = [
            {
                "sale_id": "SALE_DUP",
                "order_id": "ORD_1",
                "sku_id": "SKU_1",
                "quantity": 1,
                "revenue": 100.0,
                "unit_cost": 50.0,
                "calibrated_return_probability": 0.30,
            },
            {
                "sale_id": "SALE_DUP",  # duplicate
                "order_id": "ORD_1",
                "sku_id": "SKU_1",
                "quantity": 1,
                "revenue": 100.0,
                "unit_cost": 50.0,
                "calibrated_return_probability": 0.30,
            },
        ]
        recs = evaluate_order_lines(batch)
        assert len(recs) == 2
        assert recs[0].decision == InterventionDecision.INTERVENTION_INDICATED
        assert recs[1].decision == InterventionDecision.INSUFFICIENT_DATA
        assert recs[1].recommendation_code == InterventionRecommendationCode.DATA_QUALITY_ERROR
        assert "Duplicate sale_id detected" in recs[1].rationale

    def test_invalid_risk_band_handled_gracefully(self):
        # 25. Invalid or unclassified risk band
        record = {
            "sale_id": "SALE_RB",
            "order_id": "ORD_RB",
            "sku_id": "SKU_RB",
            "quantity": 1,
            "revenue": 100.0,
            "unit_cost": 50.0,
            "calibrated_return_probability": 0.30,
            "risk_band": "INVALID_RISK_BAND_NAME",
        }
        rec = evaluate_order_line(record)
        assert rec.risk_band == "INVALID_RISK_BAND_NAME"
        assert rec.decision == InterventionDecision.INTERVENTION_INDICATED


class TestPortfolioAggregation:
    """Test suite for portfolio-level summaries and multi-dimensional rollups."""

    def test_portfolio_aggregation_counts_and_metrics(self):
        # 21. Portfolio aggregation
        recs = [
            ReturnInterventionRecommendation(
                intervention_id="INTV_1",
                sale_id="S1",
                order_id="O1",
                sku_id="SKU_A",
                warehouse_id="WH_1",
                channel_id="CH_AMZ",
                calibrated_return_probability=0.30,
                revenue=100.0,
                estimated_return_impact=100.0,
                expected_return_exposure=30.0,
                financial_status=FinancialInputStatus.PARTIAL,
                decision=InterventionDecision.INTERVENTION_INDICATED,
                recommendation_code=InterventionRecommendationCode.REVIEW_RETURN_RISK,
                rationale="Elevated risk",
                policy_version="v1.0",
                generated_at="2026-09-30T00:00:00",
            ),
            ReturnInterventionRecommendation(
                intervention_id="INTV_2",
                sale_id="S2",
                order_id="O2",
                sku_id="SKU_B",
                warehouse_id="WH_2",
                channel_id="CH_DIR",
                calibrated_return_probability=0.05,
                revenue=50.0,
                estimated_return_impact=50.0,
                expected_return_exposure=2.5,
                financial_status=FinancialInputStatus.SUFFICIENT,
                decision=InterventionDecision.NO_INTERVENTION_INDICATED,
                recommendation_code=InterventionRecommendationCode.NO_INTERVENTION_INDICATED,
                rationale="Low risk",
                policy_version="v1.0",
                generated_at="2026-09-30T00:00:00",
            ),
            ReturnInterventionRecommendation(
                intervention_id="INTV_3",
                sale_id="S3",
                order_id="O3",
                sku_id="SKU_A",
                warehouse_id="WH_1",
                channel_id="CH_AMZ",
                calibrated_return_probability=None,
                revenue=None,  # missing revenue
                estimated_return_impact=None,
                expected_return_exposure=None,
                financial_status=FinancialInputStatus.INSUFFICIENT,
                decision=InterventionDecision.INSUFFICIENT_DATA,
                recommendation_code=InterventionRecommendationCode.INSUFFICIENT_FINANCIAL_INPUTS,
                rationale="Missing inputs",
                policy_version="v1.0",
                generated_at="2026-09-30T00:00:00",
            ),
        ]

        summary = summarize_portfolio(recs)
        assert summary.total_order_lines_analyzed == 3
        assert summary.sufficient_financial_records == 1
        assert summary.partial_financial_records == 1
        assert summary.insufficient_financial_records == 1

        assert summary.intervention_indicated_count == 1
        assert summary.no_intervention_count == 1
        assert summary.insufficient_data_count == 1
        assert summary.review_required_count == 0

        assert summary.total_revenue_represented == 150.0
        assert summary.total_estimated_return_impact == 150.0
        assert summary.total_expected_return_exposure == 32.5

    def test_sku_channel_warehouse_aggregation_breakdown(self):
        # 22. Breakdown tables by SKU, Channel, Warehouse
        recs = [
            ReturnInterventionRecommendation(
                intervention_id="INTV_1",
                sale_id="S1",
                order_id="O1",
                sku_id="SKU_ALPHA",
                warehouse_id="WH_EAST",
                channel_id="CH_AMZ",
                calibrated_return_probability=0.35,
                revenue=200.0,
                estimated_return_impact=200.0,
                expected_return_exposure=70.0,
                financial_status=FinancialInputStatus.PARTIAL,
                decision=InterventionDecision.INTERVENTION_INDICATED,
                recommendation_code=InterventionRecommendationCode.REVIEW_RETURN_RISK,
                rationale="High",
                policy_version="v1.0",
                generated_at="2026-09-30T00:00:00",
            ),
            ReturnInterventionRecommendation(
                intervention_id="INTV_2",
                sale_id="S2",
                order_id="O2",
                sku_id="SKU_ALPHA",
                warehouse_id="WH_EAST",
                channel_id="CH_AMZ",
                calibrated_return_probability=0.10,
                revenue=100.0,
                estimated_return_impact=100.0,
                expected_return_exposure=10.0,
                financial_status=FinancialInputStatus.PARTIAL,
                decision=InterventionDecision.NO_INTERVENTION_INDICATED,
                recommendation_code=InterventionRecommendationCode.NO_INTERVENTION_INDICATED,
                rationale="Low",
                policy_version="v1.0",
                generated_at="2026-09-30T00:00:00",
            ),
        ]
        summary = summarize_portfolio(recs)
        df_sku = portfolio_breakdown_to_dataframe(summary, dimension="sku")
        assert len(df_sku) == 1
        row_alpha = df_sku.iloc[0]
        assert row_alpha["sku_id"] == "SKU_ALPHA"
        assert row_alpha["order_line_count"] == 2
        assert row_alpha["intervention_indicated_count"] == 1
        assert row_alpha["no_intervention_count"] == 1
        assert row_alpha["total_revenue"] == 300.0
        assert row_alpha["total_expected_exposure"] == 80.0
        assert row_alpha["mean_calibrated_probability"] == 0.225

    def test_missing_values_not_treated_as_zero(self):
        # 23. Missing values must NOT be aggregated as 0.0
        recs = [
            ReturnInterventionRecommendation(
                intervention_id="INTV_NO_FIN",
                sale_id="S_NF",
                order_id="O_NF",
                sku_id="SKU_NF",
                warehouse_id="WH_NF",
                channel_id="CH_NF",
                financial_status=FinancialInputStatus.INSUFFICIENT,
                decision=InterventionDecision.INSUFFICIENT_DATA,
                recommendation_code=InterventionRecommendationCode.INSUFFICIENT_FINANCIAL_INPUTS,
                rationale="No fin",
                policy_version="v1.0",
                generated_at="2026-09-30T00:00:00",
            )
        ]
        summary = summarize_portfolio(recs)
        assert summary.total_revenue_represented is None
        assert summary.total_estimated_return_impact is None
        assert summary.total_expected_return_exposure is None

    def test_empty_dataset(self):
        # 24. Empty dataset
        summary = summarize_portfolio([])
        assert summary.total_order_lines_analyzed == 0
        assert summary.total_revenue_represented is None

        recs = evaluate_order_lines([])
        assert recs == []

        df_empty = portfolio_breakdown_to_dataframe(summary, dimension="channel")
        assert df_empty.empty


class TestReturnInterventionServiceIntegration:
    """Integration test suite for the ReturnInterventionService class."""

    def test_service_evaluate_dataframe(self):
        service = ReturnInterventionService()

        df_in = pd.DataFrame([
            {
                "sale_id": "S_INT_1",
                "order_id": "O_INT_1",
                "sku_id": "SKU_INT_1",
                "warehouse_id": "WH_01",
                "channel_id": "CH_AMZ",
                "quantity": 1,
                "unit_price": 200.0,
                "revenue": 200.0,
                "unit_cost": 80.0,
                "calibrated_return_probability": 0.35,
                "risk_band": "MEDIUM",
            },
            {
                "sale_id": "S_INT_2",
                "order_id": "O_INT_2",
                "sku_id": "SKU_INT_2",
                "warehouse_id": "WH_02",
                "channel_id": "CH_DIR",
                "quantity": 1,
                "unit_price": 50.0,
                "revenue": 50.0,
                "unit_cost": 20.0,
                "calibrated_return_probability": 0.05,
                "risk_band": "VERY_LOW",
            },
        ])

        df_out = service.evaluate_dataframe(df_in)
        assert len(df_out) == 2
        assert "intervention_id" in df_out.columns
        assert "decision" in df_out.columns
        assert "expected_return_exposure" in df_out.columns

        assert df_out.iloc[0]["decision"] == InterventionDecision.INTERVENTION_INDICATED.value
        assert df_out.iloc[1]["decision"] == InterventionDecision.NO_INTERVENTION_INDICATED.value

        recs = service.evaluate_order_lines(df_in)
        summary = service.summarize_portfolio(recs)
        assert summary.total_order_lines_analyzed == 2
        assert summary.intervention_indicated_count == 1
        assert summary.no_intervention_count == 1
