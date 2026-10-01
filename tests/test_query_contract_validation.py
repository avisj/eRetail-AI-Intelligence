"""Validation test suite for Business Query Contracts (Phase 7B).

Tests:
- Unknown intent rejection
- Unknown metric rejection
- Incompatible metric for intent rejection
- Domain and intent mismatch rejection
- Unsupported grain for intent rejection
- Incompatible dimension combinations rejection
- Temporal anti-leakage rejection
- SQL, code, and path traversal injection rejection
- Governance mutation and execution request rejection
- Action execution keyword detection and rejection
- Unregistered tool name rejection
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from commerce_ai.query_contracts.enums import (
    BusinessDimension,
    BusinessDomain,
    MetricIdentifier,
    OutputGrain,
    QueryIntent,
    RequestedOutput,
)
from commerce_ai.query_contracts.schemas import (
    BusinessQueryContract,
    BusinessQueryFilter,
    ContractGovernance,
    TimeRangeContract,
)
from commerce_ai.query_contracts.service import QueryContractService
from commerce_ai.query_contracts.validation import validate_contract


@pytest.fixture
def service() -> QueryContractService:
    return QueryContractService()


class TestContractValidationRules:
    def test_valid_contract_passes(self, service: QueryContractService):
        contract, val = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            as_of_date="2026-06-30",
            currency="USD",
        )
        assert val.is_valid is True
        assert len(val.errors) == 0

    def test_unknown_intent_string_raises(self, service: QueryContractService):
        with pytest.raises(ValueError, match="Unknown or invalid query intent"):
            service.build_contract(intent="MAKE_ARBITRARY_PREDICTION")

    def test_unknown_metric_string_raises(self, service: QueryContractService):
        with pytest.raises(ValueError, match="Unknown or invalid metric identifier"):
            service.build_contract(
                intent=QueryIntent.SALES_PERFORMANCE,
                metrics=["NON_EXISTENT_METRIC"],
            )

    def test_metric_incompatible_with_intent_fails_validation(self, service: QueryContractService):
        # INVENTORY_VALUE is an inventory metric, not valid for SALES_PERFORMANCE
        contract, val = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            metrics=[MetricIdentifier.INVENTORY_VALUE],
        )
        assert val.is_valid is False
        codes = [e.code for e in val.errors]
        assert "METRIC_INTENT_INCOMPATIBLE" in codes

    def test_domain_intent_mismatch_fails_validation(self, service: QueryContractService):
        # SALES_PERFORMANCE belongs to SALES domain, not INVENTORY
        contract, val = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            domain=BusinessDomain.INVENTORY,
        )
        assert val.is_valid is False
        codes = [e.code for e in val.errors]
        assert "DOMAIN_INTENT_MISMATCH" in codes

    def test_unsupported_grain_fails_validation(self, service: QueryContractService):
        # DECISION_REVIEW does not support DAY grain (it is portfolio/entity review)
        contract, val = service.build_contract(
            intent=QueryIntent.DECISION_REVIEW,
            requested_grain=OutputGrain.DAY,
        )
        assert val.is_valid is False
        codes = [e.code for e in val.errors]
        assert "UNSUPPORTED_GRAIN_FOR_INTENT" in codes

    def test_incompatible_dimension_combination_fails(self, service: QueryContractService):
        # DATE and MONTH in the same query is an incompatible duplicate grain
        contract, val = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            dimensions=[BusinessDimension.DATE, BusinessDimension.MONTH],
        )
        assert val.is_valid is False
        codes = [e.code for e in val.errors]
        assert "DIMENSION_INVALID" in codes

    def test_temporal_leakage_detected(self):
        # end_date > as_of_date
        with pytest.raises(ValidationError, match="cannot exceed point-in-time as_of_date"):
            TimeRangeContract(
                start_date="2026-01-01",
                end_date="2026-08-01",
                as_of_date="2026-06-30",
            )

    def test_unregistered_tool_fails_validation(self, service: QueryContractService):
        contract, _ = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            required_tools=["execute_custom_backdoor_query"],
        )
        val = service.validate(contract)
        assert val.is_valid is False
        codes = [e.code for e in val.errors]
        assert "UNREGISTERED_TOOL_REQUESTED" in codes

    def test_action_execution_keywords_in_explanation_rejected(self, service: QueryContractService):
        prohibited_texts = [
            "Execute PO for supplier",
            "Please create po now",
            "Transfer stock between warehouses",
            "Change price for SKU_01",
            "Delete old inventory records",
        ]
        for text in prohibited_texts:
            contract, val = service.build_contract(
                intent=QueryIntent.SALES_PERFORMANCE,
                explanation_context=text,
            )
            assert val.is_valid is False
            codes = [e.code for e in val.errors]
            assert "ACTION_EXECUTION_ATTEMPT_REJECTED" in codes


class TestInjectionPrevention:
    def test_sql_injection_in_filter_rejected(self):
        with pytest.raises(ValidationError, match="Disallowed expression detected"):
            BusinessQueryFilter(sku_id="SKU_01'; DROP TABLE products; --")

    def test_sql_union_in_filter_rejected(self):
        with pytest.raises(ValidationError, match="Disallowed expression detected"):
            BusinessQueryFilter(channel_id="ONLINE' UNION SELECT * FROM users --")

    def test_code_execution_in_filter_rejected(self):
        with pytest.raises(ValidationError, match="Disallowed expression detected"):
            BusinessQueryFilter(brand="__import__('os').system('ls')")

    def test_path_traversal_in_filter_rejected(self):
        with pytest.raises(ValidationError, match="Disallowed expression detected"):
            BusinessQueryFilter(warehouse_id="../../etc/passwd")


class TestGovernanceEnforcement:
    def test_read_only_mutation_rejected(self, service: QueryContractService):
        contract, _ = service.build_contract(intent=QueryIntent.SALES_PERFORMANCE)
        # Manually alter governance using object.__setattr__ to bypass frozen Pydantic
        mutated_gov = ContractGovernance(read_only=False, execution_allowed=False)
        object.__setattr__(contract, "governance", mutated_gov)

        val = service.validate(contract)
        assert val.is_valid is False
        codes = [e.code for e in val.errors]
        assert "GOVERNANCE_MUTATION_FORBIDDEN" in codes

    def test_execution_allowed_rejected(self, service: QueryContractService):
        contract, _ = service.build_contract(intent=QueryIntent.SALES_PERFORMANCE)
        mutated_gov = ContractGovernance(read_only=True, execution_allowed=True)
        object.__setattr__(contract, "governance", mutated_gov)

        val = service.validate(contract)
        assert val.is_valid is False
        codes = [e.code for e in val.errors]
        assert "GOVERNANCE_EXECUTION_FORBIDDEN" in codes

    def test_action_execution_flag_rejected(self, service: QueryContractService):
        contract, _ = service.build_contract(intent=QueryIntent.SALES_PERFORMANCE)
        mutated_gov = ContractGovernance(read_only=True, execution_allowed=False, action_execution=True)
        object.__setattr__(contract, "governance", mutated_gov)

        val = service.validate(contract)
        assert val.is_valid is False
        codes = [e.code for e in val.errors]
        assert "GOVERNANCE_ACTION_FORBIDDEN" in codes


class TestFilterAndAdvisoryValidation:
    def test_empty_filters_valid(self, service: QueryContractService):
        contract, val = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            filters={},
        )
        assert val.is_valid is True

    def test_multi_attribute_filters_valid(self, service: QueryContractService):
        contract, val = service.build_contract(
            intent=QueryIntent.SALES_PERFORMANCE,
            filters={
                "sku_id": "SKU_01",
                "warehouse_id": "WH_01",
                "channel_id": "ONLINE",
                "velocity_tier": "FAST",
                "abc_class": "A",
                "xyz_class": "X",
                "limit": 100,
                "offset": 0,
            },
        )
        assert val.is_valid is True
        assert contract.filters.sku_id == "SKU_01"
        assert contract.filters.limit == 100

    def test_recommendation_review_advisory_warning_if_approval_unspecified(self, service: QueryContractService):
        contract, _ = service.build_contract(intent=QueryIntent.RECOMMENDATION_REVIEW)
        # Manually alter approval_required to False
        mutated_gov = ContractGovernance(read_only=True, execution_allowed=False, approval_required=False)
        object.__setattr__(contract, "governance", mutated_gov)

        val = service.validate(contract)
        assert val.is_valid is True  # Warning only, not an error
        warning_codes = [w.code for w in val.warnings]
        assert "APPROVAL_REQUIRED_ADVISORY" in warning_codes

    def test_multi_domain_empty_domains_warning(self, service: QueryContractService):
        contract, val = service.build_contract(
            intent=QueryIntent.MULTI_DOMAIN_ANALYSIS,
            domain=BusinessDomain.CROSS_DOMAIN,
            domains=[],
        )
        assert val.is_valid is True
        warning_codes = [w.code for w in val.warnings]
        assert "MULTI_DOMAIN_EMPTY_DOMAINS" in warning_codes

    def test_prohibited_ranking_tool_rejected(self, service: QueryContractService):
        """Verify that any ranking tool request is strictly rejected with PROHIBITED_RANKING_TOOL_REQUESTED (Point 8 & 20)."""
        contract, _ = service.build_contract(
            intent=QueryIntent.BUSINESS_IMPACT_ANALYSIS,
            required_tools=["get_opportunity_ranking"],
        )
        val = service.validate(contract)
        assert val.is_valid is False
        error_codes = [e.code for e in val.errors]
        assert "PROHIBITED_RANKING_TOOL_REQUESTED" in error_codes

    def test_data_quality_capability_unavailable_warning(self, service: QueryContractService):
        """Verify DATA_QUALITY_ANALYSIS produces CAPABILITY_UNAVAILABLE_IN_PHASE_7A warning (Point 13 & 20)."""
        contract, val = service.build_contract(intent=QueryIntent.DATA_QUALITY_ANALYSIS)
        assert val.is_valid is True
        assert any(w.code == "CAPABILITY_UNAVAILABLE_IN_PHASE_7A" for w in val.warnings)
        assert contract.required_tools == []

    def test_future_date_leakage_prevented(self, service: QueryContractService):
        """Verify forward-looking date leakage is rejected with TEMPORAL_LEAKAGE_DETECTED (Point 18 & 21-O)."""
        with pytest.raises(ValueError, match="cannot exceed point-in-time as_of_date"):
            TimeRangeContract(
                start_date="2026-01-01",
                end_date="2026-07-15",
                as_of_date="2026-06-30",
            )

