"""Margin Driver Classifications, Negative/Low Margin Analysis, and Waterfall (Phase 6C).

Implements deterministic analysis for:
- Classification of segments into operational margin driver categories:
    * HIGH_REVENUE_LOW_MARGIN_SHARE
    * LOW_REVENUE_HIGH_MARGIN_SHARE
    * HIGH_DISCOUNT
    * NEGATIVE_MARGIN
    * LOW_MARGIN
    * HIGH_MARGIN_CONTRIBUTOR
    * HIGH_REVENUE_HIGH_MARGIN
    * LOW_REVENUE_LOW_MARGIN
- Negative and low margin root-cause flagging with deterministic reason codes
- Commercial portfolio margin waterfall progression from Gross Revenue to Contribution Margin
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd

from commerce_ai.financial.schemas import (
    AttributionReasonCode,
    MarginDriverClassification,
    MarginDriverSegment,
    MarginWaterfall,
    MarginWaterfallStage,
    ProfitabilityAttributionConfig,
    ProfitabilityAttributionRecord,
    SKUProfitabilityProfile,
)


def evaluate_sku_driver_classifications(
    profile: SKUProfitabilityProfile,
    config: ProfitabilityAttributionConfig,
) -> SKUProfitabilityProfile:
    """Evaluate and assign deterministic driver classifications and reason codes to an SKU profile.

    Args:
        profile: The SKUProfitabilityProfile to evaluate.
        config: Configuration containing thresholds.

    Returns:
        Updated SKUProfitabilityProfile with populated classifications and reason codes.
    """
    classes: List[MarginDriverClassification] = []
    reasons: List[AttributionReasonCode] = []

    gm = profile.gross_margin if profile.gross_margin is not None else 0.0
    rev = profile.net_revenue if profile.net_revenue is not None else 0.0
    gm_pct = profile.gross_margin_pct
    disc_rate = profile.discount_rate or 0.0
    rev_share = profile.revenue_contribution_pct or 0.0
    margin_share = profile.margin_contribution_pct or 0.0
    gap = profile.contribution_gap or 0.0

    # 1. Negative margin
    if gm < 0:
        classes.append(MarginDriverClassification.NEGATIVE_MARGIN)
        reasons.append(AttributionReasonCode.NEGATIVE_GROSS_MARGIN)

    # 2. Low margin
    if gm >= 0 and gm_pct is not None and gm_pct < config.low_margin_threshold:
        classes.append(MarginDriverClassification.LOW_MARGIN)
        reasons.append(AttributionReasonCode.LOW_GROSS_MARGIN_PERCENT)

    # 3. High discount
    if disc_rate >= config.high_discount_threshold:
        classes.append(MarginDriverClassification.HIGH_DISCOUNT)
        reasons.append(AttributionReasonCode.HIGH_DISCOUNT)

    # 4. High revenue, low margin share (contribution gap deficit)
    if gap <= -config.contribution_gap_threshold:
        classes.append(MarginDriverClassification.HIGH_REVENUE_LOW_MARGIN_SHARE)
        reasons.append(AttributionReasonCode.CONTRIBUTION_GAP_DEFICIT)
        if rev_share >= 0.01:
            reasons.append(AttributionReasonCode.HIGH_REVENUE_CONTRIBUTION)
        reasons.append(AttributionReasonCode.LOW_MARGIN_CONTRIBUTION)

    # 5. Low revenue, high margin share
    if gap >= config.contribution_gap_threshold:
        classes.append(MarginDriverClassification.LOW_REVENUE_HIGH_MARGIN_SHARE)

    # 6. High margin contributor (e.g. margin share >= 5%)
    if margin_share >= 0.05:
        classes.append(MarginDriverClassification.HIGH_MARGIN_CONTRIBUTOR)

    # 7. High revenue and high margin (both above healthy levels)
    if rev_share >= 0.02 and gm_pct is not None and gm_pct >= config.low_margin_threshold:
        classes.append(MarginDriverClassification.HIGH_REVENUE_HIGH_MARGIN)

    # 8. Low revenue and low margin
    if rev_share < 0.005 and gm_pct is not None and gm_pct < config.low_margin_threshold:
        classes.append(MarginDriverClassification.LOW_REVENUE_LOW_MARGIN)

    # Deduplicate preserving order
    dedup_classes = list(dict.fromkeys(classes))
    dedup_reasons = list(dict.fromkeys(reasons))

    profile.driver_classifications = dedup_classes
    profile.reason_codes = dedup_reasons
    return profile


def classify_margin_drivers(
    sku_profiles: Sequence[SKUProfitabilityProfile],
    config: ProfitabilityAttributionConfig,
) -> List[MarginDriverSegment]:
    """Extract individual classified driver segments from evaluated SKU profiles.

    Args:
        sku_profiles: Sequence of evaluated SKU profiles.
        config: Configuration containing thresholds.

    Returns:
        List of MarginDriverSegment records.
    """
    driver_segments: List[MarginDriverSegment] = []

    for prof in sku_profiles:
        eval_prof = evaluate_sku_driver_classifications(prof, config)

        for cl in eval_prof.driver_classifications:
            desc = _generate_driver_description(eval_prof, cl)
            driver_segments.append(
                MarginDriverSegment(
                    segment_dimension="SKU",
                    segment_key=eval_prof.sku_id,
                    driver_classification=cl,
                    gross_margin=eval_prof.gross_margin or 0.0,
                    net_revenue=eval_prof.net_revenue or 0.0,
                    gross_margin_pct=eval_prof.gross_margin_pct,
                    revenue_contribution_pct=eval_prof.revenue_contribution_pct,
                    margin_contribution_pct=eval_prof.margin_contribution_pct,
                    contribution_gap=eval_prof.contribution_gap,
                    reason_codes=eval_prof.reason_codes,
                    description=desc,
                )
            )

    return driver_segments


def analyze_negative_and_low_margins(
    segments: Sequence[Union[ProfitabilityAttributionRecord, SKUProfitabilityProfile]],
    config: ProfitabilityAttributionConfig,
) -> List[MarginDriverSegment]:
    """Identify segments with negative or low gross margins and assign deterministic reason codes.

    Args:
        segments: Sequence of attribution records or SKU profiles.
        config: Configuration containing low_margin_threshold.

    Returns:
        List of MarginDriverSegment sorted by gross_margin ascending.
    """
    flagged: List[MarginDriverSegment] = []

    for seg in segments:
        dim = getattr(seg, "dimension", None) or "SKU"
        key = getattr(seg, "segment_key", None) or getattr(seg, "sku_id", "")
        gm = float(seg.gross_margin or 0.0) if seg.gross_margin is not None else 0.0
        rev = float(seg.net_revenue or 0.0) if seg.net_revenue is not None else 0.0
        gm_pct = seg.gross_margin_pct
        rev_share = seg.revenue_contribution_pct
        margin_share = seg.margin_contribution_pct
        gap = seg.contribution_gap
        disc_rate = getattr(seg, "discount_rate", 0.0) or 0.0

        if gm < 0:
            reasons = [AttributionReasonCode.NEGATIVE_GROSS_MARGIN]
            if disc_rate >= config.high_discount_threshold:
                reasons.append(AttributionReasonCode.HIGH_DISCOUNT)
            if gap is not None and gap <= -config.contribution_gap_threshold:
                reasons.append(AttributionReasonCode.CONTRIBUTION_GAP_DEFICIT)

            flagged.append(
                MarginDriverSegment(
                    segment_dimension=dim,
                    segment_key=str(key),
                    driver_classification=MarginDriverClassification.NEGATIVE_MARGIN,
                    gross_margin=gm,
                    net_revenue=rev,
                    gross_margin_pct=gm_pct,
                    revenue_contribution_pct=rev_share,
                    margin_contribution_pct=margin_share,
                    contribution_gap=gap,
                    reason_codes=reasons,
                    description=f"Segment realized negative gross margin (${gm:,.2f}) where product cost exceeds realized net revenue.",
                )
            )
        elif gm_pct is not None and gm_pct < config.low_margin_threshold:
            reasons = [AttributionReasonCode.LOW_GROSS_MARGIN_PERCENT]
            if disc_rate >= config.high_discount_threshold:
                reasons.append(AttributionReasonCode.HIGH_DISCOUNT)
            if gap is not None and gap <= -config.contribution_gap_threshold:
                reasons.append(AttributionReasonCode.CONTRIBUTION_GAP_DEFICIT)

            flagged.append(
                MarginDriverSegment(
                    segment_dimension=dim,
                    segment_key=str(key),
                    driver_classification=MarginDriverClassification.LOW_MARGIN,
                    gross_margin=gm,
                    net_revenue=rev,
                    gross_margin_pct=gm_pct,
                    revenue_contribution_pct=rev_share,
                    margin_contribution_pct=margin_share,
                    contribution_gap=gap,
                    reason_codes=reasons,
                    description=f"Segment realized low gross margin ({gm_pct:.2%}), below policy threshold ({config.low_margin_threshold:.2%}).",
                )
            )

    flagged.sort(key=lambda x: (x.gross_margin, x.gross_margin_pct or 0.0))
    return flagged


def build_margin_waterfall(
    gross_revenue: float,
    discount: float,
    net_revenue: float,
    product_cost: float,
    gross_margin: float,
    known_variable_costs: float = 0.0,
    known_contribution_margin: Optional[float] = None,
    final_contribution_margin: Optional[float] = None,
    currency: str = "USD",
) -> MarginWaterfall:
    """Construct an executive margin waterfall progression from Gross Revenue to Contribution Margin.

    Args:
        gross_revenue: Billed sales value before discounts.
        discount: Promotional discounts granted.
        net_revenue: Realized transaction revenue.
        product_cost: Procurement / manufacturing product cost.
        gross_margin: Commercial gross margin.
        known_variable_costs: Tracked variable expenses (fulfillment, shipping, fees, returns).
        known_contribution_margin: Realized margin after known variable costs.
        final_contribution_margin: Final contribution margin (None if costs incomplete).
        currency: ISO currency code.

    Returns:
        MarginWaterfall containing ordered sequence of stages.
    """
    if known_contribution_margin is None:
        known_contribution_margin = gross_margin - known_variable_costs

    stages: List[MarginWaterfallStage] = []

    def get_pct(amt: Optional[float]) -> Optional[float]:
        if amt is None or gross_revenue == 0:
            return None
        return round(amt / gross_revenue, 4)

    # 1. Gross Revenue
    stages.append(
        MarginWaterfallStage(
            stage_name="Gross Revenue",
            stage_order=1,
            amount=round(gross_revenue, 2),
            percentage_of_gross_revenue=1.0 if gross_revenue > 0 else 0.0,
            stage_type="SUBTOTAL",
            description="Total billed transaction value before promotional discounts",
        )
    )

    # 2. Promotional Discounts
    stages.append(
        MarginWaterfallStage(
            stage_name="Promotional Discounts",
            stage_order=2,
            amount=round(-discount, 2),
            percentage_of_gross_revenue=get_pct(-discount),
            stage_type="DEDUCTION",
            description="Promotional price reductions deducted from gross revenue",
        )
    )

    # 3. Net Revenue
    stages.append(
        MarginWaterfallStage(
            stage_name="Net Revenue",
            stage_order=3,
            amount=round(net_revenue, 2),
            percentage_of_gross_revenue=get_pct(net_revenue),
            stage_type="SUBTOTAL",
            description="Realized sales revenue after promotional discounts",
        )
    )

    # 4. Product Cost / COGS
    stages.append(
        MarginWaterfallStage(
            stage_name="Product Cost / COGS",
            stage_order=4,
            amount=round(-product_cost, 2),
            percentage_of_gross_revenue=get_pct(-product_cost),
            stage_type="DEDUCTION",
            description="Standard procurement product cost of goods sold",
        )
    )

    # 5. Gross Margin
    stages.append(
        MarginWaterfallStage(
            stage_name="Gross Margin",
            stage_order=5,
            amount=round(gross_margin, 2),
            percentage_of_gross_revenue=get_pct(gross_margin),
            stage_type="SUBTOTAL",
            description="Commercial gross profit realized above procurement product cost",
        )
    )

    # 6. Known Variable Costs
    stages.append(
        MarginWaterfallStage(
            stage_name="Known Variable Costs",
            stage_order=6,
            amount=round(-known_variable_costs, 2),
            percentage_of_gross_revenue=get_pct(-known_variable_costs),
            stage_type="DEDUCTION",
            description="Aggregated known variable fulfillment, shipping, fee, and return costs",
        )
    )

    # 7. Known Contribution Margin
    stages.append(
        MarginWaterfallStage(
            stage_name="Known Contribution Margin",
            stage_order=7,
            amount=round(known_contribution_margin, 2),
            percentage_of_gross_revenue=get_pct(known_contribution_margin),
            stage_type="SUBTOTAL",
            description="Commercial profit remaining after deducting all known variable costs",
        )
    )

    # 8. Final Contribution Margin
    final_amt = round(final_contribution_margin, 2) if final_contribution_margin is not None else None
    stages.append(
        MarginWaterfallStage(
            stage_name="Final Contribution Margin",
            stage_order=8,
            amount=final_amt,
            percentage_of_gross_revenue=get_pct(final_amt),
            stage_type="RESULT",
            description="Audited final contribution margin (unavailable when cost components are incomplete)",
        )
    )

    return MarginWaterfall(
        stages=stages,
        gross_revenue=round(gross_revenue, 2),
        discount=round(discount, 2),
        net_revenue=round(net_revenue, 2),
        product_cost=round(product_cost, 2),
        gross_margin=round(gross_margin, 2),
        known_variable_costs=round(known_variable_costs, 2),
        known_contribution_margin=round(known_contribution_margin, 2),
        final_contribution_margin=final_amt,
        currency=currency,
    )


def _generate_driver_description(profile: SKUProfitabilityProfile, classification: MarginDriverClassification) -> str:
    """Generate a descriptive, factual explanation for a driver classification."""
    gm = profile.gross_margin or 0.0
    gm_pct_str = f"{profile.gross_margin_pct:.2%}" if profile.gross_margin_pct is not None else "N/A"
    rev_share_str = f"{profile.revenue_contribution_pct:.2%}" if profile.revenue_contribution_pct is not None else "N/A"
    margin_share_str = f"{profile.margin_contribution_pct:.2%}" if profile.margin_contribution_pct is not None else "N/A"
    gap_str = f"{profile.contribution_gap:.2%}" if profile.contribution_gap is not None else "N/A"

    if classification == MarginDriverClassification.NEGATIVE_MARGIN:
        return f"Realized negative gross margin (${gm:,.2f}); product costs exceed realized net revenue."
    if classification == MarginDriverClassification.LOW_MARGIN:
        return f"Realized low gross margin percentage ({gm_pct_str}) below policy threshold."
    if classification == MarginDriverClassification.HIGH_DISCOUNT:
        disc_rate_str = f"{profile.discount_rate:.2%}" if profile.discount_rate is not None else "N/A"
        return f"High promotional discount rate ({disc_rate_str}) eroding gross margin."
    if classification == MarginDriverClassification.HIGH_REVENUE_LOW_MARGIN_SHARE:
        return (
            f"Disproportionate margin erosion: generates {rev_share_str} of revenue "
            f"but only {margin_share_str} of margin (contribution gap: {gap_str})."
        )
    if classification == MarginDriverClassification.LOW_REVENUE_HIGH_MARGIN_SHARE:
        return (
            f"High-efficiency margin generator: generates {margin_share_str} of margin "
            f"with only {rev_share_str} of revenue (contribution gap: +{gap_str})."
        )
    if classification == MarginDriverClassification.HIGH_MARGIN_CONTRIBUTOR:
        return f"Major portfolio margin contributor ({margin_share_str} of total portfolio gross margin)."
    if classification == MarginDriverClassification.HIGH_REVENUE_HIGH_MARGIN:
        return f"Strong portfolio anchor: generates {rev_share_str} of revenue and {gm_pct_str} gross margin."
    if classification == MarginDriverClassification.LOW_REVENUE_LOW_MARGIN:
        return f"Low volume, low margin segment ({rev_share_str} revenue share, {gm_pct_str} margin)."

    return f"Segment categorized under {classification.value}."
