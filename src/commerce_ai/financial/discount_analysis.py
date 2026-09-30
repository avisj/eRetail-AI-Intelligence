"""Promotional Discount Impact and Bucket Distribution Analytics (Phase 6C).

Implements deterministic analysis of:
- Margin impact of promotional discounting:
    margin_before_discount = gross_revenue - product_cost
    margin_after_discount = net_revenue - product_cost
    discount_margin_impact = margin_after_discount - margin_before_discount == -discount
- Discount bucket distribution (0%, 0-5%, 5-10%, 10-20%, 20-30%, 30%+)
- Realized margin erosion across promotional tiers
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.financial.schemas import (
    DiscountBucketMetric,
    DiscountImpactSummary,
    RevenueMarginRecord,
)

DEFAULT_DISCOUNT_BUCKETS: List[Tuple[float, float, str]] = [
    (0.0, 0.0, "0%"),
    (0.0, 0.05, "0-5%"),
    (0.05, 0.10, "5-10%"),
    (0.10, 0.20, "10-20%"),
    (0.20, 0.30, "20-30%"),
    (0.30, 1.00, "30%+"),
]


def _extract_sales_columns(df_or_records: Union[pd.DataFrame, Sequence[Any]]) -> pd.DataFrame:
    """Normalize input into a DataFrame with standardized financial columns."""
    if isinstance(df_or_records, pd.DataFrame):
        df = df_or_records.copy()
    elif isinstance(df_or_records, Sequence):
        if len(df_or_records) == 0:
            df = pd.DataFrame()
        elif hasattr(df_or_records[0], "model_dump"):
            df = pd.DataFrame([r.model_dump() for r in df_or_records])
        elif isinstance(df_or_records[0], dict):
            df = pd.DataFrame(df_or_records)
        else:
            df = pd.DataFrame([r.__dict__ for r in df_or_records])
    else:
        df = pd.DataFrame()

    if df.empty:
        return pd.DataFrame(
            columns=["gross_revenue", "discount", "net_revenue", "product_cost", "quantity", "discount_rate"]
        )

    # Standardize column names
    col_map = {
        "calculated_gross_revenue": "gross_revenue",
        "calculated_net_revenue": "net_revenue",
    }
    for old_col, new_col in col_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    for c in ["gross_revenue", "discount", "net_revenue", "product_cost", "quantity"]:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Ensure quantity is int-compatible
    df["quantity"] = df["quantity"].astype(int)

    # Calculate discount_rate if not present or missing
    if "discount_rate" not in df.columns:
        df["discount_rate"] = np.where(
            df["gross_revenue"] > 0,
            df["discount"] / df["gross_revenue"],
            0.0,
        )
    else:
        df["discount_rate"] = pd.to_numeric(df["discount_rate"], errors="coerce").fillna(0.0)

    return df


def analyze_discount_buckets(
    df_or_records: Union[pd.DataFrame, Sequence[Any]],
    bucket_boundaries: Optional[List[Tuple[float, float, str]]] = None,
    currency: str = "USD",
) -> List[DiscountBucketMetric]:
    """Segment sales transactions into discount rate buckets and aggregate performance.

    Args:
        df_or_records: Sales DataFrame or sequence of transaction records.
        bucket_boundaries: List of (min_rate, max_rate, label) tuples.
        currency: ISO currency code.

    Returns:
        List of DiscountBucketMetric objects.
    """
    if bucket_boundaries is None:
        bucket_boundaries = DEFAULT_DISCOUNT_BUCKETS

    df = _extract_sales_columns(df_or_records)

    bucket_metrics: List[DiscountBucketMetric] = []

    if df.empty:
        for min_r, max_r, label in bucket_boundaries:
            bucket_metrics.append(
                DiscountBucketMetric(
                    bucket_label=label,
                    min_discount_rate=min_r,
                    max_discount_rate=max_r,
                    transaction_count=0,
                    total_units=0,
                    total_gross_revenue=0.0,
                    total_discount=0.0,
                    total_net_revenue=0.0,
                    total_product_cost=0.0,
                    total_gross_margin=0.0,
                    gross_margin_pct=None,
                    margin_per_unit=None,
                    discount_margin_impact=0.0,
                    currency=currency,
                )
            )
        return bucket_metrics

    rates = df["discount_rate"].values

    for min_r, max_r, label in bucket_boundaries:
        if min_r == 0.0 and max_r == 0.0:
            mask = rates <= 1e-6
        elif max_r >= 1.0 or max_r == float("inf"):
            mask = rates > min_r
        else:
            mask = (rates > min_r) & (rates <= max_r)

        subset = df[mask]
        tx_count = len(subset)
        total_units = int(subset["quantity"].sum()) if tx_count > 0 else 0
        total_gross_rev = float(subset["gross_revenue"].sum()) if tx_count > 0 else 0.0
        total_disc = float(subset["discount"].sum()) if tx_count > 0 else 0.0
        total_net_rev = float(subset["net_revenue"].sum()) if tx_count > 0 else 0.0
        total_cogs = float(subset["product_cost"].sum()) if tx_count > 0 else 0.0
        total_gm = total_net_rev - total_cogs
        gm_pct = (total_gm / total_net_rev) if total_net_rev != 0 else None
        gm_per_unit = (total_gm / total_units) if total_units > 0 else None
        # Invariant: discount_margin_impact = (net_rev - cogs) - (gross_rev - cogs) = -total_disc
        disc_impact = -total_disc

        bucket_metrics.append(
            DiscountBucketMetric(
                bucket_label=label,
                min_discount_rate=min_r,
                max_discount_rate=max_r,
                transaction_count=tx_count,
                total_units=total_units,
                total_gross_revenue=round(total_gross_rev, 2),
                total_discount=round(total_disc, 2),
                total_net_revenue=round(total_net_rev, 2),
                total_product_cost=round(total_cogs, 2),
                total_gross_margin=round(total_gm, 2),
                gross_margin_pct=round(gm_pct, 4) if gm_pct is not None else None,
                margin_per_unit=round(gm_per_unit, 2) if gm_per_unit is not None else None,
                discount_margin_impact=round(disc_impact, 2),
                currency=currency,
            )
        )

    return bucket_metrics


def analyze_discount_impact(
    df_or_records: Union[pd.DataFrame, Sequence[Any]],
    bucket_boundaries: Optional[List[Tuple[float, float, str]]] = None,
    currency: str = "USD",
) -> DiscountImpactSummary:
    """Perform portfolio-level discount impact analysis and bucket segmentation.

    Args:
        df_or_records: Sales DataFrame or sequence of transaction records.
        bucket_boundaries: Optional custom bucket boundaries.
        currency: ISO currency code.

    Returns:
        DiscountImpactSummary detailing pre/post discount margins and bucket breakdown.
    """
    df = _extract_sales_columns(df_or_records)

    total_gross_rev = float(df["gross_revenue"].sum()) if not df.empty else 0.0
    total_disc = float(df["discount"].sum()) if not df.empty else 0.0
    total_net_rev = float(df["net_revenue"].sum()) if not df.empty else 0.0
    total_cogs = float(df["product_cost"].sum()) if not df.empty else 0.0

    discount_rate = (total_disc / total_gross_rev) if total_gross_rev > 0 else 0.0
    margin_before_disc = total_gross_rev - total_cogs
    margin_after_disc = total_net_rev - total_cogs
    # Invariant: discount_margin_impact = margin_after_disc - margin_before_disc == -total_disc
    discount_margin_impact = -total_disc

    buckets = analyze_discount_buckets(df, bucket_boundaries=bucket_boundaries, currency=currency)

    return DiscountImpactSummary(
        total_gross_revenue=round(total_gross_rev, 2),
        total_discount=round(total_disc, 2),
        discount_rate=round(discount_rate, 4),
        total_net_revenue=round(total_net_rev, 2),
        margin_before_discount=round(margin_before_disc, 2),
        margin_after_discount=round(margin_after_disc, 2),
        discount_margin_impact=round(discount_margin_impact, 2),
        buckets=buckets,
        currency=currency,
    )
