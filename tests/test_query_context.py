"""Unit tests for QueryContext and Context Validation (Phase 7A)."""

from __future__ import annotations

from datetime import date, datetime
import pytest
from pydantic import ValidationError

from commerce_ai.query_layer.context import QueryContext


class TestQueryContext:
    def test_query_context_initialization_defaults(self):
        ctx = QueryContext()
        assert ctx.as_of_date is None
        assert ctx.start_date is None
        assert ctx.end_date is None
        assert ctx.currency is None
        assert ctx.time_grain == "daily"
        assert ctx.sku_ids is None
        assert ctx.warehouse_ids is None
        assert ctx.channel_ids is None
        assert ctx.to_filter_dict() == {}

    def test_query_context_date_normalization_string(self):
        ctx = QueryContext(
            as_of_date="2026-06-30T15:30:00Z",
            start_date="2026-01-01",
            end_date="2026-06-30",
        )
        assert ctx.iso_as_of_date == "2026-06-30"
        assert ctx.iso_start_date == "2026-01-01"
        assert ctx.iso_end_date == "2026-06-30"
        assert ctx.effective_end_date == "2026-06-30"

    def test_query_context_date_normalization_date_and_datetime(self):
        d_as_of = datetime(2026, 6, 30, 12, 0, 0)
        d_start = date(2026, 1, 1)
        d_end = date(2026, 5, 31)

        ctx = QueryContext(as_of_date=d_as_of, start_date=d_start, end_date=d_end)
        assert ctx.iso_as_of_date == "2026-06-30"
        assert ctx.iso_start_date == "2026-01-01"
        assert ctx.iso_end_date == "2026-05-31"
        assert ctx.effective_end_date == "2026-05-31"

    def test_query_context_chronology_validation_start_after_end_raises(self):
        with pytest.raises(ValidationError) as exc:
            QueryContext(start_date="2026-07-01", end_date="2026-06-01")
        assert "start_date" in str(exc.value)

    def test_query_context_chronology_validation_start_after_as_of_raises(self):
        with pytest.raises(ValidationError) as exc:
            QueryContext(as_of_date="2026-05-01", start_date="2026-06-01")
        assert "cannot be after point-in-time as_of_date" in str(exc.value)

    def test_query_context_effective_end_date_capped_by_as_of_date(self):
        # Even if requested end_date is December, point-in-time as_of_date caps it at June
        ctx = QueryContext(as_of_date="2026-06-30", end_date="2026-12-31")
        assert ctx.effective_end_date == "2026-06-30"

    def test_query_context_effective_end_date_when_only_one_specified(self):
        ctx1 = QueryContext(as_of_date="2026-06-30")
        assert ctx1.effective_end_date == "2026-06-30"

        ctx2 = QueryContext(end_date="2026-05-31")
        assert ctx2.effective_end_date == "2026-05-31"

    def test_query_context_effective_currency_uppercase(self):
        ctx = QueryContext(currency="  usd  ")
        assert ctx.effective_currency == "USD"

    def test_query_context_sku_id_single_and_list_normalization(self):
        ctx1 = QueryContext(sku_id="SKU_001")
        assert ctx1.sku_ids == ["SKU_001"]

        ctx2 = QueryContext(sku_id=["SKU_002", " SKU_001 ", "SKU_002"])
        assert ctx2.sku_ids == ["SKU_001", "SKU_002"]

    def test_query_context_warehouse_id_normalization(self):
        ctx = QueryContext(warehouse_id=["WH_02", "WH_01"])
        assert ctx.warehouse_ids == ["WH_01", "WH_02"]

    def test_query_context_channel_id_normalization(self):
        ctx = QueryContext(channel_id="CH_AMZ")
        assert ctx.channel_ids == ["CH_AMZ"]

    def test_query_context_category_and_brand_normalization(self):
        ctx = QueryContext(category_id=["Electronics", "Home"], brand="HyperTech")
        assert ctx.category_ids == ["Electronics", "Home"]
        assert ctx.brands == ["HyperTech"]

    def test_query_context_velocity_tier_and_abc_xyz_normalization(self):
        ctx = QueryContext(velocity_tier="FAST", abc_class=["A", "B"], xyz_class="X")
        assert ctx.velocity_tiers == ["FAST"]
        assert ctx.abc_classes == ["A", "B"]
        assert ctx.xyz_classes == ["X"]

    def test_query_context_to_filter_dict_contains_only_active_filters(self):
        ctx = QueryContext(
            as_of_date="2026-06-30",
            sku_id=["SKU_001"],
            currency="USD",
            limit=50,
        )
        f = ctx.to_filter_dict()
        assert f["as_of_date"] == "2026-06-30"
        assert f["sku_ids"] == ["SKU_001"]
        assert f["currency"] == "USD"
        assert f["limit"] == 50
        assert "channel_ids" not in f
        assert "start_date" not in f

    def test_query_context_extra_attributes_forbidden(self):
        with pytest.raises(ValidationError):
            QueryContext(arbitrary_field="illegal")
