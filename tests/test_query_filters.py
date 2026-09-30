"""Unit tests for Point-in-Time Safety, Currency Isolation, and Filtering (Phase 7A)."""

from __future__ import annotations

import pandas as pd
import pytest

from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.filters import (
    apply_context_filters,
    apply_date_range_filter,
    apply_point_in_time_filter,
    check_currency_isolation,
)


@pytest.fixture
def sample_sales_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"sale_id": "S1", "date": "2024-01-01", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_AMZ", "quantity": 10, "unit_price": 50.0, "currency": "USD"},
        {"sale_id": "S2", "date": "2024-02-15", "sku_id": "SKU_01", "warehouse_id": "WH_01", "channel_id": "CH_DIR", "quantity": 5, "unit_price": 50.0, "currency": "USD"},
        {"sale_id": "S3", "date": "2024-03-20", "sku_id": "SKU_02", "warehouse_id": "WH_02", "channel_id": "CH_AMZ", "quantity": 2, "unit_price": 100.0, "currency": "USD"},
        {"sale_id": "S4", "date": "2024-07-01", "sku_id": "SKU_02", "warehouse_id": "WH_02", "channel_id": "CH_DIR", "quantity": 8, "unit_price": 100.0, "currency": "USD"},
    ])


@pytest.fixture
def sample_products_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"sku_id": "SKU_01", "category_id": "Electronics", "brand": "HyperTech", "currency": "USD", "unit_cost": 30.0},
        {"sku_id": "SKU_02", "category_id": "Home", "brand": "CozyLiving", "currency": "USD", "unit_cost": 60.0},
    ])


class TestQueryFilters:
    def test_apply_point_in_time_filter_excludes_future_dates(self, sample_sales_df):
        # as_of_date is 2024-03-01 -> S3 (March 20) and S4 (July 01) must be strictly excluded
        filtered = apply_point_in_time_filter(sample_sales_df, date_col="date", as_of_date="2024-03-01")
        assert len(filtered) == 2
        assert set(filtered["sale_id"]) == {"S1", "S2"}
        assert all(filtered["date"] <= "2024-03-01")

    def test_apply_point_in_time_filter_exact_match_included(self, sample_sales_df):
        filtered = apply_point_in_time_filter(sample_sales_df, date_col="date", as_of_date="2024-03-20")
        assert len(filtered) == 3
        assert set(filtered["sale_id"]) == {"S1", "S2", "S3"}

    def test_apply_point_in_time_filter_datetime_dtype(self, sample_sales_df):
        df = sample_sales_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        filtered = apply_point_in_time_filter(df, date_col="date", as_of_date="2024-02-15")
        assert len(filtered) == 2
        assert set(filtered["sale_id"]) == {"S1", "S2"}

    def test_apply_point_in_time_filter_empty_df_safe(self):
        empty = pd.DataFrame(columns=["date", "sale_id"])
        res = apply_point_in_time_filter(empty, date_col="date", as_of_date="2024-01-01")
        assert res.empty

    def test_apply_date_range_filter_bounded_interval(self, sample_sales_df):
        # 2024-02-01 to 2024-04-01 -> should match S2 and S3
        filtered = apply_date_range_filter(
            sample_sales_df,
            date_col="date",
            start_date="2024-02-01",
            end_date="2024-04-01",
        )
        assert len(filtered) == 2
        assert set(filtered["sale_id"]) == {"S2", "S3"}

    def test_apply_date_range_filter_with_as_of_date_cap(self, sample_sales_df):
        # end_date is 2024-12-31, but as_of_date is 2024-02-28 -> S3 and S4 cannot leak
        filtered = apply_date_range_filter(
            sample_sales_df,
            date_col="date",
            start_date="2024-01-01",
            end_date="2024-12-31",
            as_of_date="2024-02-28",
        )
        assert len(filtered) == 2
        assert set(filtered["sale_id"]) == {"S1", "S2"}

    def test_currency_isolation_single_currency_passes(self, sample_sales_df):
        filtered, curr, is_isolated, err = check_currency_isolation(sample_sales_df, currency_col="currency")
        assert is_isolated is True
        assert curr == "USD"
        assert err is None
        assert len(filtered) == 4

    def test_currency_isolation_multi_currency_rejected_without_filter(self):
        multi_curr_df = pd.DataFrame([
            {"sale_id": "S1", "currency": "USD", "amount": 100},
            {"sale_id": "S2", "currency": "EUR", "amount": 90},
        ])
        filtered, curr, is_isolated, err = check_currency_isolation(multi_curr_df, currency_col="currency")
        assert is_isolated is False
        assert curr is None
        assert "Multi-currency dataset detected" in err
        assert "FX conversion are strictly prohibited" in err

    def test_currency_isolation_multi_currency_with_filter_succeeds(self):
        multi_curr_df = pd.DataFrame([
            {"sale_id": "S1", "currency": "USD", "amount": 100},
            {"sale_id": "S2", "currency": "EUR", "amount": 90},
        ])
        filtered, curr, is_isolated, err = check_currency_isolation(
            multi_curr_df,
            currency_col="currency",
            requested_currency="EUR",
        )
        assert is_isolated is True
        assert curr == "EUR"
        assert len(filtered) == 1
        assert filtered["sale_id"].iloc[0] == "S2"

    def test_apply_context_filters_sku_and_warehouse(self, sample_sales_df):
        ctx = QueryContext(sku_id="SKU_01", warehouse_id="WH_01")
        filtered, curr, is_isolated, err = apply_context_filters(sample_sales_df, context=ctx, date_col="date")
        assert is_isolated is True
        assert len(filtered) == 2
        assert set(filtered["sale_id"]) == {"S1", "S2"}

    def test_apply_context_filters_category_via_products_df(self, sample_sales_df, sample_products_df):
        # Filtering for category Home should only retain SKU_02 transactions
        ctx = QueryContext(category_id="Home")
        filtered, curr, is_isolated, err = apply_context_filters(
            sample_sales_df,
            context=ctx,
            date_col="date",
            products_df=sample_products_df,
        )
        assert is_isolated is True
        assert len(filtered) == 2
        assert set(filtered["sku_id"]) == {"SKU_02"}
