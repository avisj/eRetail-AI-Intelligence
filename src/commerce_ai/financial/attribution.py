"""Dimensional Profitability Attribution, SKU Profiles, and Attribution Service (Phase 6C).

Implements deterministic analytics for:
- Profitability attribution across 8 dimensional cuts:
    * SKU
    * Category
    * Brand
    * Channel
    * Warehouse
    * SKU x Channel
    * SKU x Warehouse
    * Channel x Warehouse
- Contribution gap calculation: margin_contribution_pct - revenue_contribution_pct
- SKU Profitability Profiles with driver classifications and reason codes
- Temporal profitability attribution (Daily, Weekly, Monthly)
- ProfitabilityAttributionService unifying all Phase 6C capabilities
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.financial.concentration import (
    calculate_margin_concentration,
    calculate_margin_contribution_curve,
)
from commerce_ai.financial.discount_analysis import (
    analyze_discount_impact,
)
from commerce_ai.financial.margin_drivers import (
    analyze_negative_and_low_margins,
    build_margin_waterfall,
    classify_margin_drivers,
    evaluate_sku_driver_classifications,
)
from commerce_ai.financial.revenue_margin import (
    audit_financial_data_quality,
    filter_sales_by_as_of_date,
)
from commerce_ai.financial.schemas import (
    AttributionReasonCode,
    CostModelConfig,
    DiscountImpactSummary,
    FinancialDataQualityReport,
    FinancialIntelligenceConfig,
    MarginClassification,
    MarginConcentrationResult,
    MarginContributionPoint,
    MarginDriverClassification,
    MarginDriverSegment,
    MarginWaterfall,
    ProfitabilityAttributionConfig,
    ProfitabilityAttributionRecord,
    ProfitabilityAttributionResult,
    SKUProfitabilityProfile,
    TimeGrain,
)
from commerce_ai.financial.unit_economics import compute_unit_economics_dataframe


# =====================================================================
# Dimensional Attribution Aggregations
# =====================================================================


def _safe_float(val: Any) -> float:
    if val is None or pd.isna(val):
        return 0.0
    return float(val)


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    res = float(numerator / denominator)
    return None if np.isnan(res) or np.isinf(res) else res


def compute_dimensional_profitability_attribution(
    df: pd.DataFrame,
    dimension: str,
    portfolio_gross_margin: float,
    portfolio_net_revenue: float,
    currency: str = "USD",
) -> List[ProfitabilityAttributionRecord]:
    """Compute profitability attribution records across a specific dimension.

    Args:
        df: Enriched transaction DataFrame.
        dimension: Dimensional cut (SKU, CATEGORY, BRAND, CHANNEL, WAREHOUSE,
                   SKU_CHANNEL, SKU_WAREHOUSE, CHANNEL_WAREHOUSE, OVERALL).
        portfolio_gross_margin: Total portfolio gross margin for contribution share.
        portfolio_net_revenue: Total portfolio net revenue for contribution share.
        currency: ISO currency code.

    Returns:
        List of ProfitabilityAttributionRecord sorted by gross_margin descending.
    """
    dim_upper = dimension.strip().upper()

    if df.empty:
        return []

    # Map dimension to group column(s)
    if dim_upper in ("OVERALL", "PORTFOLIO"):
        group_cols = None
    elif dim_upper == "SKU":
        group_cols = ["sku_id"]
    elif dim_upper == "CATEGORY":
        group_cols = ["category_id"]
    elif dim_upper == "BRAND":
        group_cols = ["brand"]
    elif dim_upper == "CHANNEL":
        group_cols = ["channel_id"]
    elif dim_upper == "WAREHOUSE":
        group_cols = ["warehouse_id"]
    elif dim_upper in ("SKU_CHANNEL", "SKU_X_CHANNEL"):
        group_cols = ["sku_id", "channel_id"]
        dim_upper = "SKU_CHANNEL"
    elif dim_upper in ("SKU_WAREHOUSE", "SKU_X_WAREHOUSE"):
        group_cols = ["sku_id", "warehouse_id"]
        dim_upper = "SKU_WAREHOUSE"
    elif dim_upper in ("CHANNEL_WAREHOUSE", "CHANNEL_X_WAREHOUSE"):
        group_cols = ["channel_id", "warehouse_id"]
        dim_upper = "CHANNEL_WAREHOUSE"
    else:
        raise ValueError(f"Unsupported profitability attribution dimension: {dimension}")

    records: List[ProfitabilityAttributionRecord] = []

    if group_cols is None:
        # Portfolio level
        rec_count = len(df)
        order_count = int(df["order_id"].nunique()) if "order_id" in df.columns else 0
        total_units = int(df["quantity"].sum()) if "quantity" in df.columns else 0
        gross_rev = _safe_float(df["gross_revenue"].sum()) if "gross_revenue" in df.columns else 0.0
        disc = _safe_float(df["discount"].sum()) if "discount" in df.columns else 0.0
        net_rev = _safe_float(df["net_revenue"].sum()) if "net_revenue" in df.columns else 0.0
        cogs = _safe_float(df["product_cost"].sum()) if "product_cost" in df.columns else 0.0
        gm = _safe_float(df["gross_margin"].sum()) if "gross_margin" in df.columns else (net_rev - cogs)
        var_cost = _safe_float(df["total_known_variable_cost"].sum()) if "total_known_variable_cost" in df.columns else 0.0
        known_cm = _safe_float(df["known_contribution_margin"].sum()) if "known_contribution_margin" in df.columns else (gm - var_cost)

        disc_rate = _safe_div(disc, gross_rev) or 0.0
        gm_pct = _safe_div(gm, net_rev)
        margin_pu = _safe_div(gm, total_units)
        rev_pu = _safe_div(net_rev, total_units)
        cost_pu = _safe_div(cogs, total_units)
        known_cm_pct = _safe_div(known_cm, net_rev)
        aov = _safe_div(net_rev, order_count)

        rev_share = 1.0 if net_rev > 0 else 0.0
        margin_share = 1.0 if gm != 0 else 0.0

        if gm > 0:
            m_class = MarginClassification.POSITIVE_MARGIN
        elif gm == 0:
            m_class = MarginClassification.ZERO_MARGIN
        else:
            m_class = MarginClassification.NEGATIVE_MARGIN

        record = ProfitabilityAttributionRecord(
            dimension="PORTFOLIO",
            segment_key="PORTFOLIO",
            record_count=rec_count,
            order_count=order_count,
            total_units=total_units,
            gross_revenue=round(gross_rev, 2),
            discount=round(disc, 2),
            discount_rate=round(disc_rate, 4),
            net_revenue=round(net_rev, 2),
            product_cost=round(cogs, 2),
            gross_margin=round(gm, 2),
            gross_margin_pct=round(gm_pct, 4) if gm_pct is not None else None,
            margin_per_unit=round(margin_pu, 2) if margin_pu is not None else None,
            revenue_per_unit=round(rev_pu, 2) if rev_pu is not None else None,
            product_cost_per_unit=round(cost_pu, 2) if cost_pu is not None else None,
            revenue_contribution_pct=round(rev_share, 4),
            margin_contribution_pct=round(margin_share, 4),
            contribution_gap=0.0,
            absolute_contribution_gap=0.0,
            margin_classification=m_class,
            known_variable_cost=round(var_cost, 2),
            known_contribution_margin=round(known_cm, 2),
            known_contribution_margin_pct=round(known_cm_pct, 4) if known_cm_pct is not None else None,
            average_order_value=round(aov, 2) if aov is not None else None,
            currency=currency,
        )
        return [record]

    # Pre-clean group columns in temporary view
    work_df = df.copy()
    for c in group_cols:
        if c not in work_df.columns:
            work_df[c] = "UNASSIGNED"
        else:
            work_df[c] = work_df[c].fillna("UNASSIGNED").astype(str).str.strip()
            work_df[c] = work_df[c].replace("", "UNASSIGNED")

    grouped = work_df.groupby(group_cols, sort=False)

    for keys, group in grouped:
        if isinstance(keys, tuple):
            segment_key = "|".join(str(k) for k in keys)
        else:
            segment_key = str(keys)

        rec_count = len(group)
        order_count = int(group["order_id"].nunique()) if "order_id" in group.columns else 0
        total_units = int(group["quantity"].sum()) if "quantity" in group.columns else 0
        gross_rev = _safe_float(group["gross_revenue"].sum()) if "gross_revenue" in group.columns else 0.0
        disc = _safe_float(group["discount"].sum()) if "discount" in group.columns else 0.0
        net_rev = _safe_float(group["net_revenue"].sum()) if "net_revenue" in group.columns else 0.0
        cogs = _safe_float(group["product_cost"].sum()) if "product_cost" in group.columns else 0.0
        gm = _safe_float(group["gross_margin"].sum()) if "gross_margin" in group.columns else (net_rev - cogs)
        var_cost = _safe_float(group["total_known_variable_cost"].sum()) if "total_known_variable_cost" in group.columns else 0.0
        known_cm = _safe_float(group["known_contribution_margin"].sum()) if "known_contribution_margin" in group.columns else (gm - var_cost)

        disc_rate = _safe_div(disc, gross_rev) or 0.0
        gm_pct = _safe_div(gm, net_rev)
        margin_pu = _safe_div(gm, total_units)
        rev_pu = _safe_div(net_rev, total_units)
        cost_pu = _safe_div(cogs, total_units)
        known_cm_pct = _safe_div(known_cm, net_rev)
        aov = _safe_div(net_rev, order_count)

        rev_share = round(net_rev / portfolio_net_revenue, 4) if portfolio_net_revenue > 0 else 0.0
        margin_share = round(gm / portfolio_gross_margin, 4) if portfolio_gross_margin != 0 else 0.0
        gap = round(margin_share - rev_share, 4)
        abs_gap = round(abs(gap), 4)

        if gm > 0:
            m_class = MarginClassification.POSITIVE_MARGIN
        elif gm == 0:
            m_class = MarginClassification.ZERO_MARGIN
        else:
            m_class = MarginClassification.NEGATIVE_MARGIN

        records.append(
            ProfitabilityAttributionRecord(
                dimension=dim_upper,
                segment_key=segment_key,
                record_count=rec_count,
                order_count=order_count,
                total_units=total_units,
                gross_revenue=round(gross_rev, 2),
                discount=round(disc, 2),
                discount_rate=round(disc_rate, 4),
                net_revenue=round(net_rev, 2),
                product_cost=round(cogs, 2),
                gross_margin=round(gm, 2),
                gross_margin_pct=round(gm_pct, 4) if gm_pct is not None else None,
                margin_per_unit=round(margin_pu, 2) if margin_pu is not None else None,
                revenue_per_unit=round(rev_pu, 2) if rev_pu is not None else None,
                product_cost_per_unit=round(cost_pu, 2) if cost_pu is not None else None,
                revenue_contribution_pct=round(rev_share, 4),
                margin_contribution_pct=round(margin_share, 4),
                contribution_gap=round(gap, 4),
                absolute_contribution_gap=round(abs_gap, 4),
                margin_classification=m_class,
                known_variable_cost=round(var_cost, 2),
                known_contribution_margin=round(known_cm, 2),
                known_contribution_margin_pct=round(known_cm_pct, 4) if known_cm_pct is not None else None,
                average_order_value=round(aov, 2) if aov is not None else None,
                currency=currency,
            )
        )

    # Sort descending by gross_margin, ties broken by net_revenue desc, segment_key asc
    records.sort(key=lambda r: (-(r.gross_margin or 0.0), -(r.net_revenue or 0.0), str(r.segment_key)))
    return records


# =====================================================================
# SKU Profitability Profiles
# =====================================================================


def compute_sku_profitability_profiles(
    df: pd.DataFrame,
    products_df: Optional[pd.DataFrame],
    config: ProfitabilityAttributionConfig,
    portfolio_gross_margin: float,
    portfolio_net_revenue: float,
    currency: str = "USD",
) -> List[SKUProfitabilityProfile]:
    """Compute comprehensive SKU profitability profiles with deterministic driver classifications.

    Args:
        df: Enriched transaction DataFrame.
        products_df: Product catalog metadata DataFrame.
        config: ProfitabilityAttributionConfig.
        portfolio_gross_margin: Total portfolio gross margin.
        portfolio_net_revenue: Total portfolio net revenue.
        currency: ISO currency code.

    Returns:
        List of SKUProfitabilityProfile sorted by gross_margin descending.
    """
    if df.empty:
        return []

    # Build catalog lookup maps
    prod_meta: Dict[str, Dict[str, Any]] = {}
    if products_df is not None and not products_df.empty and "sku_id" in products_df.columns:
        for _, prow in products_df.iterrows():
            sid = str(prow["sku_id"]).strip()
            prod_meta[sid] = {
                "product_name": str(prow.get("product_name", "")).strip() or None,
                "category_id": str(prow.get("category_id", "")).strip() or None,
                "brand": str(prow.get("brand", "")).strip() or None,
            }

    work_df = df.copy()
    work_df["sku_id"] = work_df["sku_id"].fillna("UNASSIGNED").astype(str).str.strip()

    grouped = work_df.groupby("sku_id", sort=False)
    profiles: List[SKUProfitabilityProfile] = []

    for sid, group in grouped:
        total_units = int(group["quantity"].sum()) if "quantity" in group.columns else 0
        order_count = int(group["order_id"].nunique()) if "order_id" in group.columns else 0
        gross_rev = _safe_float(group["gross_revenue"].sum()) if "gross_revenue" in group.columns else 0.0
        disc = _safe_float(group["discount"].sum()) if "discount" in group.columns else 0.0
        net_rev = _safe_float(group["net_revenue"].sum()) if "net_revenue" in group.columns else 0.0
        cogs = _safe_float(group["product_cost"].sum()) if "product_cost" in group.columns else 0.0
        gm = _safe_float(group["gross_margin"].sum()) if "gross_margin" in group.columns else (net_rev - cogs)
        var_cost = _safe_float(group["total_known_variable_cost"].sum()) if "total_known_variable_cost" in group.columns else 0.0
        known_cm = _safe_float(group["known_contribution_margin"].sum()) if "known_contribution_margin" in group.columns else (gm - var_cost)

        disc_rate = _safe_div(disc, gross_rev) or 0.0
        gm_pct = _safe_div(gm, net_rev)
        margin_pu = _safe_div(gm, total_units)
        rev_pu = _safe_div(net_rev, total_units)
        cost_pu = _safe_div(cogs, total_units)

        rev_share = round(net_rev / portfolio_net_revenue, 4) if portfolio_net_revenue > 0 else 0.0
        margin_share = round(gm / portfolio_gross_margin, 4) if portfolio_gross_margin != 0 else 0.0
        gap = round(margin_share - rev_share, 4)

        # Lookup catalog details
        cat_meta = prod_meta.get(sid, {})
        p_name = cat_meta.get("product_name") or (group["product_name"].iloc[0] if "product_name" in group.columns else None)
        c_id = cat_meta.get("category_id") or (group["category_id"].iloc[0] if "category_id" in group.columns else None)
        b_name = cat_meta.get("brand") or (group["brand"].iloc[0] if "brand" in group.columns else None)

        econ_status = group["economics_status"].iloc[0] if "economics_status" in group.columns else None
        comp_pct = _safe_float(group["cost_completeness_pct"].mean()) if "cost_completeness_pct" in group.columns else None

        prof = SKUProfitabilityProfile(
            sku_id=sid,
            product_name=str(p_name) if p_name is not None and str(p_name) != "nan" else None,
            category_id=str(c_id) if c_id is not None and str(c_id) != "nan" else None,
            brand=str(b_name) if b_name is not None and str(b_name) != "nan" else None,
            total_units=total_units,
            order_count=order_count,
            gross_revenue=round(gross_rev, 2),
            discount=round(disc, 2),
            net_revenue=round(net_rev, 2),
            product_cost=round(cogs, 2),
            gross_margin=round(gm, 2),
            gross_margin_pct=round(gm_pct, 4) if gm_pct is not None else None,
            margin_per_unit=round(margin_pu, 2) if margin_pu is not None else None,
            revenue_per_unit=round(rev_pu, 2) if rev_pu is not None else None,
            product_cost_per_unit=round(cost_pu, 2) if cost_pu is not None else None,
            discount_rate=round(disc_rate, 4),
            revenue_contribution_pct=round(rev_share, 4),
            margin_contribution_pct=round(margin_share, 4),
            contribution_gap=round(gap, 4),
            driver_classifications=[],
            reason_codes=[],
            economics_status=str(econ_status) if econ_status is not None else None,
            known_variable_cost=round(var_cost, 2),
            known_contribution_margin=round(known_cm, 2),
            cost_completeness_pct=round(comp_pct, 4) if comp_pct is not None else None,
            currency=currency,
        )

        eval_prof = evaluate_sku_driver_classifications(prof, config)
        profiles.append(eval_prof)

    profiles.sort(key=lambda p: (-(p.gross_margin or 0.0), -(p.net_revenue or 0.0), str(p.sku_id)))
    return profiles


# =====================================================================
# Temporal Attribution
# =====================================================================


def compute_temporal_profitability_attribution(
    df: pd.DataFrame,
    time_grain: TimeGrain = TimeGrain.MONTHLY,
    portfolio_gross_margin: float = 0.0,
    portfolio_net_revenue: float = 0.0,
    currency: str = "USD",
) -> List[ProfitabilityAttributionRecord]:
    """Compute temporal profitability attribution records over Daily, Weekly, or Monthly buckets.

    Args:
        df: Enriched transaction DataFrame.
        time_grain: TimeGrain enum (DAILY, WEEKLY, MONTHLY).
        portfolio_gross_margin: Portfolio gross margin.
        portfolio_net_revenue: Portfolio net revenue.
        currency: ISO currency code.

    Returns:
        List of ProfitabilityAttributionRecord sorted chronologically.
    """
    if df.empty or "date" not in df.columns:
        return []

    work_df = df.copy()
    work_df["parsed_date"] = pd.to_datetime(work_df["date"], errors="coerce")
    work_df = work_df.dropna(subset=["parsed_date"])

    if work_df.empty:
        return []

    if time_grain == TimeGrain.DAILY:
        work_df["time_bucket"] = work_df["parsed_date"].dt.strftime("%Y-%m-%d")
        dim_label = "DATE"
    elif time_grain == TimeGrain.WEEKLY:
        # ISO week: YYYY-Www
        work_df["time_bucket"] = work_df["parsed_date"].dt.strftime("%Y-W%V")
        dim_label = "WEEK"
    else:  # MONTHLY
        work_df["time_bucket"] = work_df["parsed_date"].dt.strftime("%Y-%m")
        dim_label = "MONTH"

    grouped = work_df.groupby("time_bucket", sort=True)
    records: List[ProfitabilityAttributionRecord] = []

    for bucket, group in grouped:
        bucket_str = str(bucket)
        rec_count = len(group)
        order_count = int(group["order_id"].nunique()) if "order_id" in group.columns else 0
        total_units = int(group["quantity"].sum()) if "quantity" in group.columns else 0
        gross_rev = _safe_float(group["gross_revenue"].sum()) if "gross_revenue" in group.columns else 0.0
        disc = _safe_float(group["discount"].sum()) if "discount" in group.columns else 0.0
        net_rev = _safe_float(group["net_revenue"].sum()) if "net_revenue" in group.columns else 0.0
        cogs = _safe_float(group["product_cost"].sum()) if "product_cost" in group.columns else 0.0
        gm = _safe_float(group["gross_margin"].sum()) if "gross_margin" in group.columns else (net_rev - cogs)
        var_cost = _safe_float(group["total_known_variable_cost"].sum()) if "total_known_variable_cost" in group.columns else 0.0
        known_cm = _safe_float(group["known_contribution_margin"].sum()) if "known_contribution_margin" in group.columns else (gm - var_cost)

        disc_rate = _safe_div(disc, gross_rev) or 0.0
        gm_pct = _safe_div(gm, net_rev)
        margin_pu = _safe_div(gm, total_units)
        rev_pu = _safe_div(net_rev, total_units)
        cost_pu = _safe_div(cogs, total_units)
        known_cm_pct = _safe_div(known_cm, net_rev)
        aov = _safe_div(net_rev, order_count)

        rev_share = round(net_rev / portfolio_net_revenue, 4) if portfolio_net_revenue > 0 else 0.0
        margin_share = round(gm / portfolio_gross_margin, 4) if portfolio_gross_margin != 0 else 0.0
        gap = round(margin_share - rev_share, 4)
        abs_gap = round(abs(gap), 4)

        if gm > 0:
            m_class = MarginClassification.POSITIVE_MARGIN
        elif gm == 0:
            m_class = MarginClassification.ZERO_MARGIN
        else:
            m_class = MarginClassification.NEGATIVE_MARGIN

        records.append(
            ProfitabilityAttributionRecord(
                dimension=dim_label,
                segment_key=bucket_str,
                record_count=rec_count,
                order_count=order_count,
                total_units=total_units,
                gross_revenue=round(gross_rev, 2),
                discount=round(disc, 2),
                discount_rate=round(disc_rate, 4),
                net_revenue=round(net_rev, 2),
                product_cost=round(cogs, 2),
                gross_margin=round(gm, 2),
                gross_margin_pct=round(gm_pct, 4) if gm_pct is not None else None,
                margin_per_unit=round(margin_pu, 2) if margin_pu is not None else None,
                revenue_per_unit=round(rev_pu, 2) if rev_pu is not None else None,
                product_cost_per_unit=round(cost_pu, 2) if cost_pu is not None else None,
                revenue_contribution_pct=round(rev_share, 4),
                margin_contribution_pct=round(margin_share, 4),
                contribution_gap=round(gap, 4),
                absolute_contribution_gap=round(abs(gap), 4),
                margin_classification=m_class,
                known_variable_cost=round(var_cost, 2),
                known_contribution_margin=round(known_cm, 2),
                known_contribution_margin_pct=round(known_cm_pct, 4) if known_cm_pct is not None else None,
                average_order_value=round(aov, 2) if aov is not None else None,
                currency=currency,
            )
        )

    records.sort(key=lambda r: str(r.segment_key))
    return records


# =====================================================================
# Main Profitability Attribution Service
# =====================================================================


class ProfitabilityAttributionService:
    """Enterprise service executing deterministic profitability attribution and margin driver analytics."""

    def __init__(self, config: Optional[ProfitabilityAttributionConfig] = None):
        self.config = config or ProfitabilityAttributionConfig()

    def _prepare_enriched_dataframe(
        self,
        sales_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date]] = None,
    ) -> Tuple[pd.DataFrame, FinancialDataQualityReport]:
        """Validate, filter, and enrich transaction sales data using Phase 6A/6B infrastructure."""
        effective_cutoff = as_of_date if as_of_date is not None else self.config.as_of_date

        fin_cfg = FinancialIntelligenceConfig(
            as_of_date=effective_cutoff,
            default_currency=self.config.default_currency,
            allow_multi_currency=self.config.allow_multi_currency,
        )

        dq_report = audit_financial_data_quality(sales_df, products_df, fin_cfg)

        filtered_sales = filter_sales_by_as_of_date(sales_df, effective_cutoff)

        cost_cfg = CostModelConfig(
            as_of_date=effective_cutoff,
            default_currency=self.config.default_currency,
            allow_multi_currency=self.config.allow_multi_currency,
        )

        enriched_df = compute_unit_economics_dataframe(
            sales_df=filtered_sales,
            products_df=products_df,
            config=cost_cfg,
        )

        return enriched_df, dq_report

    def compute_profitability_attribution(
        self,
        sales_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date]] = None,
        time_grain: TimeGrain = TimeGrain.MONTHLY,
    ) -> ProfitabilityAttributionResult:
        """Execute full suite of Phase 6C profitability attribution analyses.

        Args:
            sales_df: Historical sales transaction DataFrame.
            products_df: Product catalog metadata DataFrame.
            as_of_date: Point-in-time historical cutoff date.
            time_grain: Granularity for temporal attribution (DAILY, WEEKLY, MONTHLY).

        Returns:
            Comprehensive ProfitabilityAttributionResult container.
        """
        enriched_df, dq_report = self._prepare_enriched_dataframe(sales_df, products_df, as_of_date)
        currency = self.config.default_currency

        # 1. Portfolio attribution
        portfolio_records = compute_dimensional_profitability_attribution(
            enriched_df,
            dimension="PORTFOLIO",
            portfolio_gross_margin=1.0,  # not used at portfolio level
            portfolio_net_revenue=1.0,
            currency=currency,
        )
        if portfolio_records:
            portfolio_rec = portfolio_records[0]
        else:
            portfolio_rec = ProfitabilityAttributionRecord(
                dimension="PORTFOLIO",
                segment_key="PORTFOLIO",
                record_count=0,
                order_count=0,
                total_units=0,
                gross_revenue=0.0,
                discount=0.0,
                discount_rate=0.0,
                net_revenue=0.0,
                product_cost=0.0,
                gross_margin=0.0,
                gross_margin_pct=None,
                margin_per_unit=None,
                revenue_per_unit=None,
                product_cost_per_unit=None,
                revenue_contribution_pct=0.0,
                margin_contribution_pct=0.0,
                contribution_gap=0.0,
                absolute_contribution_gap=0.0,
                margin_classification=MarginClassification.ZERO_MARGIN,
                known_variable_cost=0.0,
                known_contribution_margin=0.0,
                currency=currency,
            )

        port_margin = portfolio_rec.gross_margin or 0.0
        port_rev = portfolio_rec.net_revenue or 0.0

        # 2. Dimensional attributions across 8 required cuts
        dimensions_to_run = [
            "SKU",
            "CATEGORY",
            "BRAND",
            "CHANNEL",
            "WAREHOUSE",
            "SKU_CHANNEL",
            "SKU_WAREHOUSE",
            "CHANNEL_WAREHOUSE",
        ]
        dim_attributions: Dict[str, List[ProfitabilityAttributionRecord]] = {}
        for dim in dimensions_to_run:
            dim_attributions[dim] = compute_dimensional_profitability_attribution(
                enriched_df,
                dimension=dim,
                portfolio_gross_margin=port_margin,
                portfolio_net_revenue=port_rev,
                currency=currency,
            )

        # 3. SKU Profitability Profiles
        sku_profiles = compute_sku_profitability_profiles(
            df=enriched_df,
            products_df=products_df,
            config=self.config,
            portfolio_gross_margin=port_margin,
            portfolio_net_revenue=port_rev,
            currency=currency,
        )

        # 4. Margin Concentration and Contribution Curve
        sku_recs = dim_attributions.get("SKU", [])
        margin_concentration = calculate_margin_concentration(
            segments=sku_recs,
            percentiles=self.config.concentration_percentiles,
            dimension="SKU",
        )
        contribution_curve = calculate_margin_contribution_curve(
            segments=sku_recs,
            rank_by="gross_margin",
        )

        # 5. Discount Impact Analysis & Buckets
        discount_impact = analyze_discount_impact(
            df_or_records=enriched_df,
            bucket_boundaries=self.config.discount_bucket_boundaries,
            currency=currency,
        )

        # 6. Margin Drivers Classification
        margin_drivers = classify_margin_drivers(sku_profiles, self.config)

        # 7. Portfolio Margin Waterfall
        waterfall = build_margin_waterfall(
            gross_revenue=portfolio_rec.gross_revenue or 0.0,
            discount=portfolio_rec.discount or 0.0,
            net_revenue=portfolio_rec.net_revenue or 0.0,
            product_cost=portfolio_rec.product_cost or 0.0,
            gross_margin=portfolio_rec.gross_margin or 0.0,
            known_variable_costs=portfolio_rec.known_variable_cost or 0.0,
            known_contribution_margin=portfolio_rec.known_contribution_margin,
            final_contribution_margin=None,  # Unobserved variable costs in canonical dataset
            currency=currency,
        )

        # 8. Temporal Attribution
        time_attributions = compute_temporal_profitability_attribution(
            df=enriched_df,
            time_grain=time_grain,
            portfolio_gross_margin=port_margin,
            portfolio_net_revenue=port_rev,
            currency=currency,
        )

        effective_as_of = str(as_of_date) if as_of_date else (str(self.config.as_of_date) if self.config.as_of_date else None)

        return ProfitabilityAttributionResult(
            portfolio_attribution=portfolio_rec,
            dimension_attributions=dim_attributions,
            sku_profiles=sku_profiles,
            margin_concentration=margin_concentration,
            contribution_curve=contribution_curve,
            discount_impact=discount_impact,
            margin_drivers=margin_drivers,
            margin_waterfall=waterfall,
            time_attributions=time_attributions,
            data_quality_report=dq_report,
            currency=currency,
            as_of_date=effective_as_of,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def get_sku_profiles(
        self,
        sales_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
    ) -> List[SKUProfitabilityProfile]:
        """Convenience accessor for SKU profitability profiles."""
        res = self.compute_profitability_attribution(sales_df, products_df)
        return res.sku_profiles

    def get_concentration_analysis(
        self,
        sales_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
    ) -> MarginConcentrationResult:
        """Convenience accessor for margin concentration tiers."""
        res = self.compute_profitability_attribution(sales_df, products_df)
        return res.margin_concentration

    def get_discount_impact(
        self,
        sales_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
    ) -> DiscountImpactSummary:
        """Convenience accessor for promotional discount impact summary."""
        res = self.compute_profitability_attribution(sales_df, products_df)
        return res.discount_impact

    def get_margin_drivers(
        self,
        sales_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
    ) -> List[MarginDriverSegment]:
        """Convenience accessor for classified margin driver segments."""
        res = self.compute_profitability_attribution(sales_df, products_df)
        return res.margin_drivers

    def get_margin_waterfall(
        self,
        sales_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
    ) -> MarginWaterfall:
        """Convenience accessor for executive margin waterfall."""
        res = self.compute_profitability_attribution(sales_df, products_df)
        return res.margin_waterfall

    def get_dimensional_attribution(
        self,
        dimension: str,
        sales_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
    ) -> List[ProfitabilityAttributionRecord]:
        """Convenience accessor for a single dimensional attribution slice."""
        res = self.compute_profitability_attribution(sales_df, products_df)
        dim_upper = dimension.strip().upper()
        if dim_upper in res.dimension_attributions:
            return res.dimension_attributions[dim_upper]
        if dim_upper in ("OVERALL", "PORTFOLIO"):
            return [res.portfolio_attribution]
        raise ValueError(f"Attribution dimension '{dimension}' not found in computed results.")

    def get_temporal_attribution(
        self,
        time_grain: TimeGrain,
        sales_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
    ) -> List[ProfitabilityAttributionRecord]:
        """Convenience accessor for temporal profitability attribution."""
        res = self.compute_profitability_attribution(sales_df, products_df, time_grain=time_grain)
        return res.time_attributions
