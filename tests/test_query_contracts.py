"""Unit and schema tests for Business Query Contracts (Phase 7B).

Tests:
- Enum values and taxonomies (QueryIntent, BusinessDomain, MetricIdentifier, etc.)
- BusinessQueryFilter creation, defaults, list deduplication, normalization
- TimeRangeContract chronology, presets, as_of_date bounding
- ComparisonContract definitions
- GovernanceMetadata immutability and default assertions (read_only=True, execution_allowed=False)
- ConfidenceRequirement and EvidenceRequirement contracts
- BusinessQueryContract creation, serialization, Pydantic round-trip, deterministic IDs
- Canonical question templates discovery and instantiation
"""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    BusinessGrain,
    ComparisonType,
    ContractPriority,
    MetricIdentifier,
    OutputGrain,
    PlanStatus,
    ProvenanceLevel,
    QueryIntent,
    RequestedOutput,
    TimeGranularity,
    TimeRangePreset,
)
from commerce_ai.query_contracts.schemas import (
    BusinessQueryContract,
    BusinessQueryFilter,
    ComparisonContract,
    ConfidenceRequirement,
    ContractGovernance,
    EvidenceRequirement,
    TimeRangeContract,
)
from commerce_ai.query_contracts.service import QueryContractService
from commerce_ai.query_contracts.templates import (
    CANONICAL_TEMPLATES,
    find_template_by_id,
    get_canonical_templates,
    match_template,
)


class TestQueryContractEnums:
    def test_query_intent_coverage(self):
        """Verify all 28 canonical intents are defined."""
        intents = list(QueryIntent)
        assert len(intents) == 28

    def test_canonical_28_intents_complete_list(self):
        """Test that every single required intent from Point 2 exists in QueryIntent."""
        expected_28 = {
            "SALES_PERFORMANCE", "REVENUE_ANALYSIS", "MARGIN_ANALYSIS",
            "PROFITABILITY_ANALYSIS", "UNIT_ECONOMICS_ANALYSIS", "INVENTORY_STATUS",
            "INVENTORY_RISK", "STOCKOUT_ANALYSIS", "SLOW_MOVING_INVENTORY",
            "HIGH_VALUE_INVENTORY", "DEMAND_ANALYSIS", "DEMAND_TREND",
            "ABC_XYZ_ANALYSIS", "FORECAST_ANALYSIS", "FORECAST_ACCURACY",
            "FORECAST_BIAS", "RETURN_ANALYSIS", "RETURN_ANOMALY_ANALYSIS",
            "RETURN_RISK", "RETURN_REASON_ANALYSIS", "REPLENISHMENT_REVIEW",
            "PURCHASE_ORDER_REVIEW", "WAREHOUSE_REBALANCING_REVIEW",
            "BUSINESS_IMPACT_ANALYSIS", "RECOMMENDATION_REVIEW", "DECISION_REVIEW",
            "DATA_QUALITY_ANALYSIS", "MULTI_DOMAIN_ANALYSIS",
        }
        actual = {i.value for i in QueryIntent}
        assert actual == expected_28

    def test_canonical_12_domains_complete_list(self):
        """Verify exactly the 12 approved business domains exist (Point 3)."""
        expected_12 = {
            "SALES", "FINANCIAL", "INVENTORY", "DEMAND", "FORECASTING",
            "RETURNS", "OPERATIONS", "BUSINESS_IMPACT", "RECOMMENDATIONS",
            "DECISIONS", "DATA_QUALITY", "CROSS_DOMAIN",
        }
        actual = {d.value for d in BusinessDomain}
        assert actual == expected_12

    def test_canonical_metrics_complete_list(self):
        """Verify all canonical metrics from Point 4 exist."""
        required_metrics = [
            "GROSS_REVENUE", "NET_REVENUE", "AOV", "UNITS", "ORDERS",
            "GROSS_MARGIN", "GROSS_MARGIN_PERCENT", "KNOWN_CONTRIBUTION_MARGIN",
            "FINAL_CONTRIBUTION_MARGIN", "COST_COMPLETENESS",
            "INVENTORY_UNITS", "INVENTORY_VALUE", "STOCKOUT_EXPOSURE",
            "SLOW_MOVING_EXPOSURE", "HIGH_VALUE_EXPOSURE", "DAYS_OF_SUPPLY",
            "DAILY_DEMAND", "DEMAND_TREND", "VELOCITY", "INTERMITTENCY",
            "ABC_CLASS", "XYZ_CLASS",
            "FORECAST_VALUE", "FORECAST_ACCURACY", "FORECAST_BIAS", "FORECAST_ERROR",
            "RETURN_RATE", "RETURN_COUNT", "RETURNED_UNITS", "RETURN_REVENUE_EXPOSURE",
            "RETURN_RISK", "RETURN_ANOMALY_COUNT",
            "GROSS_SIGNAL_EXPOSURE", "DEDUPLICATED_EXPOSURE", "DEDUPLICATED_PHYSICAL_CAPITAL_EXPOSURE",
            "RECOMMENDATION_COUNT",
            "DECISION_PACKAGE_COUNT", "PENDING_REVIEW_COUNT", "CONFLICT_COUNT", "INSUFFICIENT_INFORMATION_COUNT",
        ]
        actual_metrics = {m.value for m in MetricIdentifier}
        for rm in required_metrics:
            assert rm in actual_metrics, f"Missing required canonical metric: {rm}"

    def test_canonical_business_grain_vs_time_granularity_independent(self):
        """Verify business grain and time granularity are separate and independent (Point 5)."""
        # Business grains
        expected_grains = {
            "PORTFOLIO", "SKU", "SKU_WAREHOUSE", "SKU_CHANNEL",
            "WAREHOUSE", "CHANNEL", "CATEGORY", "BRAND", "SUPPLIER",
            "DAY", "WEEK", "MONTH",
        }
        actual_grains = {g.value for g in BusinessGrain}
        assert actual_grains == expected_grains

        # Time granularities
        expected_time_gran = {"NONE", "DAY", "WEEK", "MONTH", "QUARTER"}
        actual_time_gran = {t.value for t in TimeGranularity}
        assert actual_time_gran == expected_time_gran

        # Test simultaneous independence in a contract
        service = QueryContractService()
        contract, val = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            requested_grain=BusinessGrain.SKU,
            time_granularity=TimeGranularity.DAY,
            as_of_date="2026-06-30",
        )
        assert val.is_valid is True
        assert contract.requested_grain == BusinessGrain.SKU
        assert contract.time_granularity == TimeGranularity.DAY

    def test_requested_output_canonical_set(self):
        """Verify canonical requested output types (Point 6)."""
        expected_outputs = {"KPI", "TIME_SERIES", "BREAKDOWN", "TABLE", "DETAIL", "INSIGHT_EVIDENCE"}
        actual_outputs = {o.value for o in RequestedOutput}
        assert actual_outputs == expected_outputs

    def test_provenance_level_includes_insufficient_data(self):
        """Verify provenance levels include INSUFFICIENT_DATA (Point 16)."""
        expected_prov = {"DIRECT_OBSERVED", "DETERMINISTIC_DERIVED", "MODEL_BASED", "INSUFFICIENT_DATA"}
        actual_prov = {p.value for p in ProvenanceLevel}
        assert actual_prov == expected_prov


class TestQueryFilterContract:
    def test_filter_defaults(self):
        f = BusinessQueryFilter()
        assert f.sku_id is None
        assert f.currency is None
        assert f.limit is None

    def test_filter_list_deduplication(self):
        f = BusinessQueryFilter(sku_ids=["SKU_01", "SKU_02", "SKU_01", "  SKU_03  "])
        assert f.sku_ids == ["SKU_01", "SKU_02", "SKU_03"]

    def test_filter_single_string_coercion_for_list(self):
        f = BusinessQueryFilter(channel_ids="ONLINE")  # type: ignore
        assert f.channel_ids == ["ONLINE"]

    def test_filter_immutability(self):
        f = BusinessQueryFilter(sku_id="SKU_01")
        with pytest.raises(ValidationError):
            f.sku_id = "SKU_02"  # type: ignore

    def test_filter_forbids_extra_attributes(self):
        with pytest.raises(ValidationError):
            BusinessQueryFilter(arbitrary_field="malicious")  # type: ignore

    def test_all_canonical_filters_supported(self):
        """Verify all filters from Point 7 are supported with extra='forbid'."""
        f = BusinessQueryFilter(
            sku_ids=["SKU_01", "SKU_02"],
            warehouse_ids=["WH_01"],
            channel_ids=["ONLINE"],
            category_ids=["CAT_A"],
            brands=["BRAND_X"],
            supplier_ids=["SUP_01"],
            velocity_tier="FAST",
            abc_class="A",
            xyz_class="X",
            currency="USD",
        )
        assert f.sku_ids == ["SKU_01", "SKU_02"]
        assert f.warehouse_ids == ["WH_01"]
        assert f.channel_ids == ["ONLINE"]
        assert f.category_ids == ["CAT_A"]
        assert f.brands == ["BRAND_X"]
        assert f.supplier_ids == ["SUP_01"]
        assert f.velocity_tier == "FAST"
        assert f.abc_class == "A"
        assert f.xyz_class == "X"
        assert f.currency == "USD"


class TestTimeAndComparisonContract:
    def test_time_range_valid_chronology(self):
        tr = TimeRangeContract(
            start_date="2026-01-01",
            end_date="2026-06-30",
            as_of_date="2026-06-30",
        )
        assert tr.start_date == "2026-01-01"
        assert tr.end_date == "2026-06-30"
        assert tr.as_of_date == "2026-06-30"

    def test_time_range_rejects_end_date_exceeding_as_of_date(self):
        with pytest.raises(ValidationError, match="cannot exceed point-in-time as_of_date"):
            TimeRangeContract(
                start_date="2026-01-01",
                end_date="2026-07-01",
                as_of_date="2026-06-30",
            )

    def test_time_range_rejects_start_after_end(self):
        with pytest.raises(ValidationError, match="cannot be after end_date"):
            TimeRangeContract(
                start_date="2026-07-01",
                end_date="2026-06-30",
                as_of_date="2026-07-15",
            )

    def test_comparison_contract_defaults(self):
        comp = ComparisonContract()
        assert comp.comparison_type == ComparisonType.NONE
        assert comp.baseline_start_date is None

    def test_comparison_contract_custom(self):
        comp = ComparisonContract(
            comparison_type=ComparisonType.CUSTOM_COMPARISON,
            baseline_start_date="2025-01-01",
            baseline_end_date="2025-06-30",
            description="Prior year H1 baseline",
        )
        assert comp.comparison_type == ComparisonType.CUSTOM_COMPARISON
        assert comp.baseline_start_date == "2025-01-01"


class TestGovernanceAndEvidenceRequirements:
    def test_governance_strict_read_only(self):
        gov = ContractGovernance()
        assert gov.read_only is True
        assert gov.execution_allowed is False
        assert gov.action_execution is False

    def test_confidence_requirements_defaults(self):
        conf = ConfidenceRequirement()
        assert conf.allow_estimated is True
        assert conf.allow_model_based is True
        assert conf.required_provenance is None

    def test_evidence_requirements(self):
        ev = EvidenceRequirement(
            required_metrics=["gross_margin", "revenue"],
            required_sources=["commerce_ai.financial"],
            require_source_ids=True,
        )
        assert len(ev.required_metrics) == 2
        assert ev.require_source_ids is True


class TestBusinessQueryContractSerialization:
    def test_contract_roundtrip_pydantic(self):
        service = QueryContractService()
        contract, val = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            as_of_date="2026-06-30",
            currency="USD",
            requested_grain=OutputGrain.PORTFOLIO,
            requested_output=RequestedOutput.KPI,
        )
        assert val.is_valid is True
        assert contract.query_id.startswith("QRY-")

        # Serialize to JSON and deserialize back
        raw_json = contract.model_dump_json()
        reconstructed = BusinessQueryContract.model_validate_json(raw_json)
        assert reconstructed.query_id == contract.query_id
        assert reconstructed.intent == contract.intent
        assert reconstructed.governance.read_only is True
        assert reconstructed.required_tools == ["get_sales_summary"]

    def test_deterministic_query_id(self):
        """Ensure identical parameters produce identical deterministic query IDs."""
        service = QueryContractService()
        c1, _ = service.build_contract(
            intent=QueryIntent.MARGIN_ANALYSIS,
            as_of_date="2026-06-30",
            currency="USD",
        )
        c2, _ = service.build_contract(
            intent=QueryIntent.MARGIN_ANALYSIS,
            as_of_date="2026-06-30",
            currency="USD",
        )
        assert c1.query_id == c2.query_id

    def test_different_parameters_produce_different_query_ids(self):
        service = QueryContractService()
        c1, _ = service.build_contract(
            intent=QueryIntent.MARGIN_ANALYSIS,
            currency="USD",
        )
        c2, _ = service.build_contract(
            intent=QueryIntent.MARGIN_ANALYSIS,
            currency="EUR",
        )
        assert c1.query_id != c2.query_id


class TestCanonicalTemplates:
    def test_templates_catalog_size(self):
        tpls = get_canonical_templates()
        assert len(tpls) >= 20

    def test_find_template_by_id(self):
        tpl = find_template_by_id("TPL_SALES_PERFORMANCE")
        assert tpl is not None
        assert tpl.intent == QueryIntent.SALES_PERFORMANCE
        assert "get_sales_summary" in tpl.expected_tools

    def test_match_template_by_question(self):
        tpl = match_template("How are sales performing?")
        assert tpl is not None
        assert tpl.template_id == "TPL_SALES_PERFORMANCE"

    def test_build_contract_from_template(self):
        service = QueryContractService()
        contract, val = service.build_from_template(
            template_id="TPL_STOCKOUT_RISK",
            overrides={"as_of_date": "2026-06-30", "currency": "USD"},
        )
        assert val.is_valid is True
        assert contract.intent == QueryIntent.STOCKOUT_ANALYSIS
        assert contract.domain == BusinessDomain.INVENTORY
        assert "get_stockout_risk" in contract.required_tools
