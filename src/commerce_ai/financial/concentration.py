"""Margin Concentration and Cumulative Contribution Curve Analytics (Phase 6C).

Implements deterministic calculations for:
- Ordered cumulative margin contribution curves across ranked segments
- Portfolio margin and revenue concentration across configurable percentile tiers (Top 1%, 5%, 10%, 20%)
- Identification of disproportionate profit concentration
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from commerce_ai.financial.schemas import (
    MarginConcentrationResult,
    MarginConcentrationTier,
    MarginContributionPoint,
    ProfitabilityAttributionRecord,
)


def calculate_margin_contribution_curve(
    segments: Sequence[ProfitabilityAttributionRecord],
    rank_by: str = "gross_margin",
) -> List[MarginContributionPoint]:
    """Compute the ordered cumulative margin contribution curve.

    Args:
        segments: Sequence of attribution records (typically SKU-level).
        rank_by: Metric attribute to rank by ('gross_margin' or 'net_revenue').

    Returns:
        Ordered list of MarginContributionPoint with running cumulative percentages.
    """
    if not segments:
        return []

    # Sort descending by primary metric, ties broken by net_revenue desc, segment_key asc
    def sort_key(rec: ProfitabilityAttributionRecord):
        primary_val = getattr(rec, rank_by, 0.0) or 0.0
        sec_val = rec.net_revenue or 0.0
        return (-primary_val, -sec_val, str(rec.segment_key))

    sorted_segments = sorted(segments, key=sort_key)

    total_margin = sum((s.gross_margin or 0.0) for s in sorted_segments)
    total_revenue = sum((s.net_revenue or 0.0) for s in sorted_segments)

    cumulative_margin = 0.0
    cumulative_revenue = 0.0
    curve_points: List[MarginContributionPoint] = []

    for idx, seg in enumerate(sorted_segments, start=1):
        seg_margin = float(seg.gross_margin or 0.0)
        seg_rev = float(seg.net_revenue or 0.0)
        cumulative_margin += seg_margin
        cumulative_revenue += seg_rev

        margin_share = (seg_margin / total_margin) if total_margin != 0 else 0.0
        rev_share = (seg_rev / total_revenue) if total_revenue != 0 else 0.0
        cum_margin_share = (cumulative_margin / total_margin) if total_margin != 0 else 0.0
        cum_rev_share = (cumulative_revenue / total_revenue) if total_revenue != 0 else 0.0

        curve_points.append(
            MarginContributionPoint(
                rank=idx,
                segment_id=str(seg.segment_key),
                segment_dimension=str(seg.dimension),
                segment_margin=round(seg_margin, 2),
                segment_revenue=round(seg_rev, 2),
                margin_contribution_pct=round(margin_share, 4),
                revenue_contribution_pct=round(rev_share, 4),
                cumulative_margin_contribution_pct=round(cum_margin_share, 4),
                cumulative_revenue_contribution_pct=round(cum_rev_share, 4),
            )
        )

    return curve_points


def calculate_margin_concentration(
    segments: Sequence[ProfitabilityAttributionRecord],
    percentiles: Optional[List[float]] = None,
    dimension: str = "SKU",
) -> MarginConcentrationResult:
    """Calculate concentration metrics for configurable top percentile tiers.

    Args:
        segments: Sequence of attribution records.
        percentiles: List of fractional percentiles (default: [0.01, 0.05, 0.10, 0.20]).
        dimension: Dimension name of the evaluated segments.

    Returns:
        MarginConcentrationResult containing concentration tiers.
    """
    if percentiles is None:
        percentiles = [0.01, 0.05, 0.10, 0.20]

    n_segments = len(segments)
    if n_segments == 0:
        return MarginConcentrationResult(
            dimension=dimension,
            total_segments=0,
            total_portfolio_margin=0.0,
            total_portfolio_revenue=0.0,
            tiers=[],
        )

    # Sort descending by gross_margin, ties broken by net_revenue desc, segment_key asc
    def sort_key(rec: ProfitabilityAttributionRecord):
        return (-(rec.gross_margin or 0.0), -(rec.net_revenue or 0.0), str(rec.segment_key))

    sorted_segments = sorted(segments, key=sort_key)

    total_margin = sum((s.gross_margin or 0.0) for s in sorted_segments)
    total_revenue = sum((s.net_revenue or 0.0) for s in sorted_segments)

    tiers: List[MarginConcentrationTier] = []

    for p in percentiles:
        top_n = min(n_segments, max(1, math.ceil(n_segments * p)))
        top_subset = sorted_segments[:top_n]

        cum_margin = sum((s.gross_margin or 0.0) for s in top_subset)
        cum_revenue = sum((s.net_revenue or 0.0) for s in top_subset)

        cum_margin_pct = (cum_margin / total_margin) if total_margin != 0 else 0.0
        cum_revenue_pct = (cum_revenue / total_revenue) if total_revenue != 0 else 0.0

        pct_num = p * 100
        tier_label = f"Top {int(pct_num)}%" if pct_num == int(pct_num) else f"Top {pct_num:.1f}%"

        tiers.append(
            MarginConcentrationTier(
                percentile=p,
                tier_label=tier_label,
                segment_count=n_segments,
                top_n_count=top_n,
                cumulative_margin_contribution_pct=round(cum_margin_pct, 4),
                cumulative_revenue_contribution_pct=round(cum_revenue_pct, 4),
                cumulative_gross_margin=round(cum_margin, 2),
                cumulative_net_revenue=round(cum_revenue, 2),
            )
        )

    return MarginConcentrationResult(
        dimension=dimension,
        total_segments=n_segments,
        total_portfolio_margin=round(total_margin, 2),
        total_portfolio_revenue=round(total_revenue, 2),
        tiers=tiers,
    )
