"""Return Intervention / Cost-Utility Engine (Phase 5C-3).

Evaluates whether an order line with an estimated/calibrated return probability
has sufficient financial/business evidence to justify an operational intervention:
- Order line -> Calibrated return probability -> Financial inputs ->
  Estimated return cost / impact -> Intervention economics -> Business policy -> Recommendation

Key Architectural Principles:
1. Informational / Analytical decision layer only (no PO creation, inventory edits, or autonomous actions).
2. Explicitly distinguishes:
   A. Directly observed financial values (quantity, unit_price, discount, revenue, unit_cost)
   B. Calculated financial values (product_cost, gross_margin, return_cost, return_impact, expected_exposure)
   C. Unavailable financial values (return_shipping_cost, handling_cost, inspection_cost, restocking_cost, salvage_value)
3. Distinguishes:
   - Sufficient financial evidence
   - Insufficient financial evidence
   - Economically meaningful intervention opportunity
   - No intervention indicated
4. Deterministic, non-causal rationales (no LLMs, no causal speculation).
5. Fully configurable, policy-versioned rules.
6. Portfolio-level aggregation without treating missing values as zero.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
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


# =====================================================================
# Financial Input Availability Assessment
# =====================================================================

AVAILABLE_FINANCIAL_FIELDS: Dict[str, List[str]] = {
    "sales": ["quantity", "unit_price", "discount", "revenue", "currency"],
    "products": ["unit_cost", "selling_price"],
}

UNAVAILABLE_FINANCIAL_FIELDS: List[str] = [
    "return_shipping_cost",
    "handling_cost",
    "inspection_cost",
    "restocking_cost",
    "salvage_value",
    "processing_cost",
    "refund_amount",
]


# =====================================================================
# Deterministic Identifiers & Rationales
# =====================================================================


def generate_intervention_id(
    sale_id: str,
    sku_id: str,
    order_id: str,
    calibrated_probability: Optional[float] = None,
    policy_version: str = "1.0.0-prototype",
    as_of_date: Optional[Union[str, date, datetime]] = None,
) -> str:
    """Generate a reproducible, deterministic intervention recommendation ID via SHA-256."""
    prob_str = f"{calibrated_probability:.4f}" if calibrated_probability is not None else "NOPROB"
    date_str = str(as_of_date) if as_of_date is not None else "NODATE"
    clean_sale = str(sale_id).strip()
    clean_order = str(order_id).strip()
    clean_sku = str(sku_id).strip()

    seed = f"{clean_sale}|{clean_order}|{clean_sku}|{prob_str}|{policy_version}|{date_str}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8].upper()
    prefix_sku = clean_sku.replace(" ", "_").replace(":", "_")
    return f"INTV_{prefix_sku}_{digest}"


def generate_intervention_rationale(
    decision: InterventionDecision,
    code: InterventionRecommendationCode,
    calibrated_probability: Optional[float] = None,
    expected_exposure: Optional[float] = None,
    estimated_gross_margin: Optional[float] = None,
    missing_inputs: Optional[List[str]] = None,
    policy: Optional[InterventionPolicyConfig] = None,
) -> str:
    """Generate a deterministic, non-causal human-readable rationale.

    Guarantees:
    - Strictly non-causal (does not speculate on customer intent or product defects).
    - Grounded entirely in empirical probability and financial cost-utility modeling.
    """
    pol = policy or InterventionPolicyConfig()
    missing_str = ", ".join(missing_inputs) if missing_inputs else "none"

    if code == InterventionRecommendationCode.DATA_QUALITY_ERROR:
        if missing_inputs:
            return f"Data quality validation failed: missing or invalid fields ({missing_str})."
        return "Data quality validation failed: record contains invalid or out-of-bounds attributes."

    if code == InterventionRecommendationCode.INSUFFICIENT_PROBABILITY_DATA:
        return "Calibrated return probability is missing or invalid; cannot evaluate return risk."

    if code == InterventionRecommendationCode.INSUFFICIENT_FINANCIAL_INPUTS:
        if missing_inputs and any(m in ("revenue", "quantity", "unit_cost", "unit_price") for m in missing_inputs):
            return "Financial inputs are insufficient to calculate expected return exposure."
        if missing_inputs and any("return" in m for m in missing_inputs):
            return "Return probability is available, but return-specific cost inputs are missing."
        return "Financial inputs are insufficient to calculate expected return exposure."

    if code == InterventionRecommendationCode.NEGATIVE_MARGIN_REVIEW:
        margin_val = f"${estimated_gross_margin:.2f}" if estimated_gross_margin is not None else "uncalculated"
        return (
            f"Product exhibits negative gross margin ({margin_val}); manual commercial "
            f"and return risk review required."
        )

    if decision == InterventionDecision.NO_INTERVENTION_INDICATED:
        if calibrated_probability is not None and calibrated_probability < pol.minimum_probability:
            return "Return probability is below the configured intervention threshold."
        if expected_exposure is not None and expected_exposure < pol.minimum_expected_exposure:
            p_pct = f"{calibrated_probability:.1%}" if calibrated_probability is not None else "N/A"
            return (
                f"Return probability is elevated ({p_pct}), but expected return exposure "
                f"(${expected_exposure:.2f}) is below the economic intervention threshold (${pol.minimum_expected_exposure:.2f})."
            )
        return "Economic exposure and probability are below configured intervention thresholds."

    if decision == InterventionDecision.INTERVENTION_INDICATED:
        p_pct = f"{calibrated_probability:.1%}" if calibrated_probability is not None else "N/A"
        exp_val = f"${expected_exposure:.2f}" if expected_exposure is not None else "$0.00"
        if code == InterventionRecommendationCode.REVIEW_HIGH_VALUE_RETURN_EXPOSURE:
            return (
                f"Return probability is elevated ({p_pct}) and expected return exposure ({exp_val}) "
                f"exceeds the high-value policy threshold (${pol.high_exposure_threshold:.2f})."
            )
        return (
            f"Return probability is elevated ({p_pct}) and expected return exposure ({exp_val}) "
            f"exceeds the configured policy threshold (${pol.minimum_expected_exposure:.2f})."
        )

    return "Analytical evaluation completed under configured cost-utility policy."


# =====================================================================
# Single-Line Order Evaluation Engine
# =====================================================================


def evaluate_order_line(
    record: Union[Dict[str, Any], pd.Series, ReturnRiskResult],
    policy: Optional[InterventionPolicyConfig] = None,
    as_of_date: Optional[Union[str, date, datetime]] = None,
) -> ReturnInterventionRecommendation:
    """Evaluate financial inputs and return probability for a single order line.

    Args:
        record: Mapping, Series, or ReturnRiskResult containing transaction and probability data.
        policy: Configurable intervention policy.
        as_of_date: Reference cutoff date for point-in-time leakage protection.

    Returns:
        Strongly typed ReturnInterventionRecommendation object.
    """
    pol = policy or InterventionPolicyConfig()
    gen_time = datetime.now(timezone.utc).isoformat()
    as_of_str = str(as_of_date) if as_of_date is not None else None

    # Normalize record access
    if isinstance(record, ReturnRiskResult):
        data = record.to_dict()
    elif isinstance(record, pd.Series):
        data = record.to_dict()
    elif isinstance(record, dict):
        data = dict(record)
    else:
        data = {}

    sale_id = str(data.get("sale_id", "")).strip()
    order_id = str(data.get("order_id", "")).strip()
    sku_id = str(data.get("sku_id", "")).strip()
    warehouse_id = str(data.get("warehouse_id", "UNKNOWN")).strip()
    channel_id = str(data.get("channel_id", "UNKNOWN")).strip()
    pred_date_raw = data.get("prediction_date", data.get("date"))
    pred_date = str(pred_date_raw).strip() if pred_date_raw is not None and not pd.isna(pred_date_raw) else None

    risk_band = data.get("risk_band")
    if hasattr(risk_band, "value"):
        risk_band = risk_band.value
    elif risk_band is not None and not pd.isna(risk_band):
        risk_band = str(risk_band).strip().upper()
    else:
        risk_band = None

    raw_prob = data.get("raw_probability")
    raw_p: Optional[float] = float(raw_prob) if raw_prob is not None and not pd.isna(raw_prob) else None

    cal_prob = data.get("calibrated_return_probability", data.get("calibrated_probability", data.get("predicted_probability")))
    cal_p: Optional[float] = float(cal_prob) if cal_prob is not None and not pd.isna(cal_prob) else None

    # Track missing fields
    missing_inputs: List[str] = []

    # 1. Critical Identifier Validation
    if not sale_id:
        missing_inputs.append("sale_id")
    if not order_id:
        missing_inputs.append("order_id")
    if not sku_id:
        missing_inputs.append("sku_id")

    if missing_inputs:
        inv_id = generate_intervention_id(
            sale_id=sale_id or "UNKNOWN",
            sku_id=sku_id or "UNKNOWN",
            order_id=order_id or "UNKNOWN",
            calibrated_probability=cal_p,
            policy_version=pol.policy_version,
            as_of_date=as_of_str,
        )
        return ReturnInterventionRecommendation(
            intervention_id=inv_id,
            sale_id=sale_id,
            order_id=order_id,
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            channel_id=channel_id,
            prediction_date=pred_date,
            calibrated_return_probability=cal_p,
            raw_probability=raw_p,
            risk_band=risk_band,
            financial_status=FinancialInputStatus.INSUFFICIENT_FINANCIAL_INPUTS,
            decision=InterventionDecision.INSUFFICIENT_DATA,
            recommendation_code=InterventionRecommendationCode.DATA_QUALITY_ERROR,
            rationale=generate_intervention_rationale(
                decision=InterventionDecision.INSUFFICIENT_DATA,
                code=InterventionRecommendationCode.DATA_QUALITY_ERROR,
                missing_inputs=missing_inputs,
                policy=pol,
            ),
            missing_inputs=missing_inputs,
            policy_version=pol.policy_version,
            as_of_date=as_of_str,
            generated_at=gen_time,
        )

    # 2. As-Of-Date Leakage Protection
    if as_of_date is not None and pred_date is not None:
        try:
            d_order = pd.to_datetime(pred_date).date()
            d_as_of = pd.to_datetime(as_of_date).date()
            if d_order > d_as_of:
                inv_id = generate_intervention_id(sale_id, sku_id, order_id, cal_p, pol.policy_version, as_of_str)
                return ReturnInterventionRecommendation(
                    intervention_id=inv_id,
                    sale_id=sale_id,
                    order_id=order_id,
                    sku_id=sku_id,
                    warehouse_id=warehouse_id,
                    channel_id=channel_id,
                    prediction_date=pred_date,
                    calibrated_return_probability=cal_p,
                    raw_probability=raw_p,
                    risk_band=risk_band,
                    financial_status=FinancialInputStatus.INSUFFICIENT_FINANCIAL_INPUTS,
                    decision=InterventionDecision.INSUFFICIENT_DATA,
                    recommendation_code=InterventionRecommendationCode.DATA_QUALITY_ERROR,
                    rationale=f"Transaction date ({pred_date}) violates point-in-time as_of_date ({as_of_str}); future leakage rejected.",
                    missing_inputs=["valid_historical_date"],
                    policy_version=pol.policy_version,
                    as_of_date=as_of_str,
                    generated_at=gen_time,
                )
        except Exception:
            pass

    # 3. Probability Validation
    if cal_p is None:
        missing_inputs.append("calibrated_return_probability")
    elif cal_p < -1e-6 or cal_p > 1.0 + 1e-6 or np.isnan(cal_p) or np.isinf(cal_p):
        inv_id = generate_intervention_id(sale_id, sku_id, order_id, None, pol.policy_version, as_of_str)
        return ReturnInterventionRecommendation(
            intervention_id=inv_id,
            sale_id=sale_id,
            order_id=order_id,
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            channel_id=channel_id,
            prediction_date=pred_date,
            calibrated_return_probability=None,
            raw_probability=raw_p,
            risk_band=risk_band,
            financial_status=FinancialInputStatus.INSUFFICIENT,
            decision=InterventionDecision.INSUFFICIENT_DATA,
            recommendation_code=InterventionRecommendationCode.DATA_QUALITY_ERROR,
            rationale=f"Invalid probability value ({cal_p}); return probability must be in [0.0, 1.0].",
            missing_inputs=["valid_calibrated_return_probability"],
            policy_version=pol.policy_version,
            as_of_date=as_of_str,
            generated_at=gen_time,
        )

    # 4. Quantity and Pricing Validations
    qty_raw = data.get("quantity")
    qty: Optional[int] = None
    if qty_raw is not None and not pd.isna(qty_raw):
        try:
            qty_int = int(qty_raw)
            if qty_int <= 0:
                inv_id = generate_intervention_id(sale_id, sku_id, order_id, cal_p, pol.policy_version, as_of_str)
                return ReturnInterventionRecommendation(
                    intervention_id=inv_id,
                    sale_id=sale_id,
                    order_id=order_id,
                    sku_id=sku_id,
                    warehouse_id=warehouse_id,
                    channel_id=channel_id,
                    prediction_date=pred_date,
                    calibrated_return_probability=cal_p,
                    raw_probability=raw_p,
                    risk_band=risk_band,
                    quantity=qty_int,
                    financial_status=FinancialInputStatus.INSUFFICIENT_FINANCIAL_INPUTS,
                    decision=InterventionDecision.INSUFFICIENT_DATA,
                    recommendation_code=InterventionRecommendationCode.DATA_QUALITY_ERROR,
                    rationale=f"Invalid order quantity ({qty_int}); quantity must be strictly positive.",
                    missing_inputs=["valid_positive_quantity"],
                    policy_version=pol.policy_version,
                    as_of_date=as_of_str,
                    generated_at=gen_time,
                )
            qty = qty_int
        except ValueError:
            missing_inputs.append("quantity")
    else:
        missing_inputs.append("quantity")

    unit_p_raw = data.get("unit_price")
    unit_p: Optional[float] = None
    if unit_p_raw is not None and not pd.isna(unit_p_raw):
        try:
            unit_p_flt = float(unit_p_raw)
            if unit_p_flt < 0.0 or np.isnan(unit_p_flt) or np.isinf(unit_p_flt):
                inv_id = generate_intervention_id(sale_id, sku_id, order_id, cal_p, pol.policy_version, as_of_str)
                return ReturnInterventionRecommendation(
                    intervention_id=inv_id,
                    sale_id=sale_id,
                    order_id=order_id,
                    sku_id=sku_id,
                    warehouse_id=warehouse_id,
                    channel_id=channel_id,
                    prediction_date=pred_date,
                    calibrated_return_probability=cal_p,
                    raw_probability=raw_p,
                    risk_band=risk_band,
                    quantity=qty,
                    unit_price=unit_p_flt,
                    financial_status=FinancialInputStatus.INSUFFICIENT_FINANCIAL_INPUTS,
                    decision=InterventionDecision.INSUFFICIENT_DATA,
                    recommendation_code=InterventionRecommendationCode.DATA_QUALITY_ERROR,
                    rationale=f"Invalid unit price (${unit_p_flt:.2f}); price cannot be negative.",
                    missing_inputs=["valid_non_negative_unit_price"],
                    policy_version=pol.policy_version,
                    as_of_date=as_of_str,
                    generated_at=gen_time,
                )
            unit_p = unit_p_flt
        except ValueError:
            pass

    disc_raw = data.get("discount", 0.0)
    disc: float = float(disc_raw) if disc_raw is not None and not pd.isna(disc_raw) else 0.0

    # 5. Financial Model: Revenue, Product Cost, Gross Margin
    rev_raw = data.get("revenue")
    rev: Optional[float] = None
    if rev_raw is not None and not pd.isna(rev_raw):
        try:
            rev_flt = float(rev_raw)
            if rev_flt < 0.0 or np.isnan(rev_flt) or np.isinf(rev_flt):
                inv_id = generate_intervention_id(sale_id, sku_id, order_id, cal_p, pol.policy_version, as_of_str)
                return ReturnInterventionRecommendation(
                    intervention_id=inv_id,
                    sale_id=sale_id,
                    order_id=order_id,
                    sku_id=sku_id,
                    warehouse_id=warehouse_id,
                    channel_id=channel_id,
                    prediction_date=pred_date,
                    calibrated_return_probability=cal_p,
                    raw_probability=raw_p,
                    risk_band=risk_band,
                    quantity=qty,
                    unit_price=unit_p,
                    revenue=rev_flt,
                    financial_status=FinancialInputStatus.INSUFFICIENT_FINANCIAL_INPUTS,
                    decision=InterventionDecision.INSUFFICIENT_DATA,
                    recommendation_code=InterventionRecommendationCode.DATA_QUALITY_ERROR,
                    rationale=f"Invalid revenue (${rev_flt:.2f}); revenue cannot be negative.",
                    missing_inputs=["valid_non_negative_revenue"],
                    policy_version=pol.policy_version,
                    as_of_date=as_of_str,
                    generated_at=gen_time,
                )
            rev = rev_flt
        except ValueError:
            pass
    elif qty is not None and unit_p is not None:
        rev = max(0.0, float(qty * unit_p - disc))
    else:
        missing_inputs.append("revenue")

    unit_cost_raw = data.get("unit_cost")
    unit_cost: Optional[float] = None
    if unit_cost_raw is not None and not pd.isna(unit_cost_raw):
        try:
            uc_flt = float(unit_cost_raw)
            if uc_flt < 0.0 or np.isnan(uc_flt) or np.isinf(uc_flt):
                inv_id = generate_intervention_id(sale_id, sku_id, order_id, cal_p, pol.policy_version, as_of_str)
                return ReturnInterventionRecommendation(
                    intervention_id=inv_id,
                    sale_id=sale_id,
                    order_id=order_id,
                    sku_id=sku_id,
                    warehouse_id=warehouse_id,
                    channel_id=channel_id,
                    prediction_date=pred_date,
                    calibrated_return_probability=cal_p,
                    raw_probability=raw_p,
                    risk_band=risk_band,
                    quantity=qty,
                    unit_price=unit_p,
                    revenue=rev,
                    unit_cost=uc_flt,
                    financial_status=FinancialInputStatus.INSUFFICIENT_FINANCIAL_INPUTS,
                    decision=InterventionDecision.INSUFFICIENT_DATA,
                    recommendation_code=InterventionRecommendationCode.DATA_QUALITY_ERROR,
                    rationale=f"Invalid unit cost (${uc_flt:.2f}); unit cost cannot be negative.",
                    missing_inputs=["valid_non_negative_unit_cost"],
                    policy_version=pol.policy_version,
                    as_of_date=as_of_str,
                    generated_at=gen_time,
                )
            unit_cost = uc_flt
        except ValueError:
            missing_inputs.append("unit_cost")
    else:
        missing_inputs.append("unit_cost")

    est_product_cost: Optional[float] = None
    est_gross_margin: Optional[float] = None
    if qty is not None and unit_cost is not None:
        est_product_cost = float(qty * unit_cost)
        if rev is not None:
            est_gross_margin = float(rev - est_product_cost)

    # 6. Return-Specific Cost Inputs
    ret_shipping = data.get("return_shipping_cost")
    ret_handling = data.get("handling_cost", data.get("processing_cost"))
    ret_restocking = data.get("restocking_cost", data.get("inspection_cost"))
    ret_direct = data.get("return_cost", data.get("estimated_return_cost"))

    explicit_ret_cost = 0.0
    has_ret_cost = False

    if ret_direct is not None and not pd.isna(ret_direct):
        explicit_ret_cost += float(ret_direct)
        has_ret_cost = True
    else:
        if ret_shipping is not None and not pd.isna(ret_shipping):
            explicit_ret_cost += float(ret_shipping)
            has_ret_cost = True
        elif pol.default_return_shipping_cost is not None:
            explicit_ret_cost += float(pol.default_return_shipping_cost)
            has_ret_cost = True

        if ret_handling is not None and not pd.isna(ret_handling):
            explicit_ret_cost += float(ret_handling)
            has_ret_cost = True
        elif pol.default_handling_cost is not None:
            explicit_ret_cost += float(pol.default_handling_cost)
            has_ret_cost = True

        if ret_restocking is not None and not pd.isna(ret_restocking):
            explicit_ret_cost += float(ret_restocking)
            has_ret_cost = True
        elif pol.default_restocking_cost is not None:
            explicit_ret_cost += float(pol.default_restocking_cost)
            has_ret_cost = True

    if not has_ret_cost:
        missing_inputs.extend(["return_shipping_cost", "handling_cost", "restocking_cost"])

    # 7. Financial Status Assessment
    has_core_economics = rev is not None and unit_cost is not None and qty is not None

    if not has_core_economics:
        financial_status = FinancialInputStatus.INSUFFICIENT
    elif has_ret_cost:
        financial_status = FinancialInputStatus.SUFFICIENT
    else:
        financial_status = FinancialInputStatus.PARTIAL

    # 8. Return Impact and Expected Exposure Modeling
    est_return_cost: Optional[float] = explicit_ret_cost if has_ret_cost else None
    est_return_impact: Optional[float] = None
    expected_exposure: Optional[float] = None

    if pol.require_return_costs and not has_ret_cost:
        # Strict mode requires explicit return cost
        est_return_impact = None
        expected_exposure = None
    elif rev is not None:
        if pol.return_cost_model == "gross_margin" and est_gross_margin is not None:
            margin_loss = max(0.0, est_gross_margin)
            est_return_impact = margin_loss + (est_return_cost or 0.0)
        else:
            # Default 'item_revenue' model: revenue refunded plus reverse logistics costs
            est_return_impact = rev + (est_return_cost or 0.0)

        if cal_p is not None and est_return_impact is not None:
            expected_exposure = float(cal_p * est_return_impact)

    # 9. Intervention Decision Logic
    inv_id = generate_intervention_id(sale_id, sku_id, order_id, cal_p, pol.policy_version, as_of_str)

    # Missing probability handling
    if cal_p is None:
        return ReturnInterventionRecommendation(
            intervention_id=inv_id,
            sale_id=sale_id,
            order_id=order_id,
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            channel_id=channel_id,
            prediction_date=pred_date,
            calibrated_return_probability=None,
            raw_probability=raw_p,
            risk_band=risk_band,
            quantity=qty,
            unit_price=unit_p,
            discount=disc,
            revenue=rev,
            unit_cost=unit_cost,
            estimated_product_cost=est_product_cost,
            estimated_gross_margin=est_gross_margin,
            estimated_return_cost=est_return_cost,
            estimated_return_impact=est_return_impact,
            expected_return_exposure=expected_exposure,
            financial_status=financial_status,
            decision=InterventionDecision.INSUFFICIENT_DATA,
            recommendation_code=InterventionRecommendationCode.INSUFFICIENT_PROBABILITY_DATA,
            rationale=generate_intervention_rationale(
                decision=InterventionDecision.INSUFFICIENT_DATA,
                code=InterventionRecommendationCode.INSUFFICIENT_PROBABILITY_DATA,
                missing_inputs=missing_inputs,
                policy=pol,
            ),
            missing_inputs=missing_inputs,
            policy_version=pol.policy_version,
            as_of_date=as_of_str,
            generated_at=gen_time,
        )

    # Insufficient financial inputs handling
    if expected_exposure is None:
        rec_code = InterventionRecommendationCode.INSUFFICIENT_FINANCIAL_INPUTS
        return ReturnInterventionRecommendation(
            intervention_id=inv_id,
            sale_id=sale_id,
            order_id=order_id,
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            channel_id=channel_id,
            prediction_date=pred_date,
            calibrated_return_probability=cal_p,
            raw_probability=raw_p,
            risk_band=risk_band,
            quantity=qty,
            unit_price=unit_p,
            discount=disc,
            revenue=rev,
            unit_cost=unit_cost,
            estimated_product_cost=est_product_cost,
            estimated_gross_margin=est_gross_margin,
            estimated_return_cost=est_return_cost,
            estimated_return_impact=est_return_impact,
            expected_return_exposure=None,
            financial_status=FinancialInputStatus.INSUFFICIENT_FINANCIAL_INPUTS if financial_status == FinancialInputStatus.INSUFFICIENT else financial_status,
            decision=InterventionDecision.INSUFFICIENT_DATA,
            recommendation_code=rec_code,
            rationale=generate_intervention_rationale(
                decision=InterventionDecision.INSUFFICIENT_DATA,
                code=rec_code,
                missing_inputs=missing_inputs,
                policy=pol,
            ),
            missing_inputs=missing_inputs,
            policy_version=pol.policy_version,
            as_of_date=as_of_str,
            generated_at=gen_time,
        )

    # Negative margin handling (edge case)
    if est_gross_margin is not None and est_gross_margin < 0.0:
        decision = InterventionDecision.REVIEW_REQUIRED
        code = InterventionRecommendationCode.NEGATIVE_MARGIN_REVIEW
        rationale = generate_intervention_rationale(
            decision=decision,
            code=code,
            calibrated_probability=cal_p,
            expected_exposure=expected_exposure,
            estimated_gross_margin=est_gross_margin,
            policy=pol,
        )
    # Probability below threshold
    elif cal_p < pol.minimum_probability:
        decision = InterventionDecision.NO_INTERVENTION_INDICATED
        code = InterventionRecommendationCode.NO_INTERVENTION_INDICATED
        rationale = generate_intervention_rationale(
            decision=decision,
            code=code,
            calibrated_probability=cal_p,
            expected_exposure=expected_exposure,
            policy=pol,
        )
    # Probability elevated, but exposure below economic threshold
    elif expected_exposure < pol.minimum_expected_exposure:
        decision = InterventionDecision.NO_INTERVENTION_INDICATED
        code = InterventionRecommendationCode.NO_INTERVENTION_INDICATED
        rationale = generate_intervention_rationale(
            decision=decision,
            code=code,
            calibrated_probability=cal_p,
            expected_exposure=expected_exposure,
            policy=pol,
        )
    # Probability elevated AND exposure meets or exceeds threshold
    else:
        decision = InterventionDecision.INTERVENTION_INDICATED
        if expected_exposure >= pol.high_exposure_threshold:
            code = InterventionRecommendationCode.REVIEW_HIGH_VALUE_RETURN_EXPOSURE
        else:
            code = InterventionRecommendationCode.REVIEW_RETURN_RISK
        rationale = generate_intervention_rationale(
            decision=decision,
            code=code,
            calibrated_probability=cal_p,
            expected_exposure=expected_exposure,
            policy=pol,
        )

    return ReturnInterventionRecommendation(
        intervention_id=inv_id,
        sale_id=sale_id,
        order_id=order_id,
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        channel_id=channel_id,
        prediction_date=pred_date,
        calibrated_return_probability=cal_p,
        raw_probability=raw_p,
        risk_band=risk_band,
        quantity=qty,
        unit_price=unit_p,
        discount=disc,
        revenue=rev,
        unit_cost=unit_cost,
        estimated_product_cost=est_product_cost,
        estimated_gross_margin=est_gross_margin,
        estimated_return_cost=est_return_cost,
        estimated_return_impact=est_return_impact,
        expected_return_exposure=expected_exposure,
        financial_status=financial_status,
        decision=decision,
        recommendation_code=code,
        rationale=rationale,
        missing_inputs=missing_inputs,
        policy_version=pol.policy_version,
        as_of_date=as_of_str,
        generated_at=gen_time,
    )


# =====================================================================
# Batch & Portfolio Aggregation Service
# =====================================================================


def evaluate_order_lines(
    records: Union[pd.DataFrame, Sequence[Union[Dict[str, Any], ReturnRiskResult]]],
    policy: Optional[InterventionPolicyConfig] = None,
    as_of_date: Optional[Union[str, date, datetime]] = None,
) -> List[ReturnInterventionRecommendation]:
    """Evaluate an entire batch or DataFrame of order lines with duplicate detection."""
    pol = policy or InterventionPolicyConfig()
    results: List[ReturnInterventionRecommendation] = []
    seen_sale_ids: Set[str] = set()

    if isinstance(records, pd.DataFrame):
        if records.empty:
            return []
        items = records.to_dict(orient="records")
    else:
        items = list(records)

    for item in items:
        # Check duplicate sale_id
        sale_id = str(item.get("sale_id", "") if isinstance(item, dict) else getattr(item, "sale_id", "")).strip()

        if sale_id and sale_id in seen_sale_ids:
            # Duplicate record within same evaluation batch
            sku_id = str(item.get("sku_id", "") if isinstance(item, dict) else getattr(item, "sku_id", "")).strip()
            order_id = str(item.get("order_id", "") if isinstance(item, dict) else getattr(item, "order_id", "")).strip()
            inv_id = generate_intervention_id(sale_id, sku_id, order_id, None, pol.policy_version, as_of_date)
            dup_rec = ReturnInterventionRecommendation(
                intervention_id=inv_id,
                sale_id=sale_id,
                order_id=order_id,
                sku_id=sku_id,
                warehouse_id=str(item.get("warehouse_id", "UNKNOWN") if isinstance(item, dict) else getattr(item, "warehouse_id", "UNKNOWN")),
                channel_id=str(item.get("channel_id", "UNKNOWN") if isinstance(item, dict) else getattr(item, "channel_id", "UNKNOWN")),
                financial_status=FinancialInputStatus.INSUFFICIENT_FINANCIAL_INPUTS,
                decision=InterventionDecision.INSUFFICIENT_DATA,
                recommendation_code=InterventionRecommendationCode.DATA_QUALITY_ERROR,
                rationale=f"Duplicate sale_id detected ({sale_id}); rejected to avoid duplicate intervention accounting.",
                missing_inputs=["unique_sale_id"],
                policy_version=pol.policy_version,
                as_of_date=str(as_of_date) if as_of_date is not None else None,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            results.append(dup_rec)
            continue

        if sale_id:
            seen_sale_ids.add(sale_id)

        rec = evaluate_order_line(item, policy=pol, as_of_date=as_of_date)
        results.append(rec)

    return results


def summarize_portfolio(
    recommendations: Sequence[Union[ReturnInterventionRecommendation, Dict[str, Any]]],
    policy_version: Optional[str] = None,
) -> PortfolioInterventionSummary:
    """Aggregate intervention recommendations into an executive portfolio summary.

    Guarantees:
    - Missing financial values are never aggregated as zero.
    - If all records lack a financial metric, the portfolio metric is None.
    - Provides multi-dimensional breakdowns by SKU, Channel, Warehouse, Risk Band, and Decision.
    """
    total_count = len(recommendations)
    if total_count == 0:
        return PortfolioInterventionSummary(
            total_order_lines_analyzed=0,
            policy_version=policy_version or "",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    sufficient_cnt = 0
    partial_cnt = 0
    insufficient_cnt = 0

    decision_counts: Dict[str, int] = {
        InterventionDecision.INTERVENTION_INDICATED.value: 0,
        InterventionDecision.NO_INTERVENTION_INDICATED.value: 0,
        InterventionDecision.REVIEW_REQUIRED.value: 0,
        InterventionDecision.INSUFFICIENT_DATA.value: 0,
    }

    revenues: List[float] = []
    impacts: List[float] = []
    exposures: List[float] = []

    # Breakdown groupings: dim -> key -> accumulator
    sku_grp: Dict[str, Dict[str, Any]] = {}
    ch_grp: Dict[str, Dict[str, Any]] = {}
    wh_grp: Dict[str, Dict[str, Any]] = {}
    rb_grp: Dict[str, Dict[str, Any]] = {}

    def _update_group(grp: Dict[str, Dict[str, Any]], key: str, r_dec: str, r_rev: Optional[float], r_exp: Optional[float], r_prob: Optional[float]):
        if key not in grp:
            grp[key] = {
                "order_line_count": 0,
                "intervention_indicated_count": 0,
                "no_intervention_count": 0,
                "review_required_count": 0,
                "insufficient_data_count": 0,
                "revenues": [],
                "exposures": [],
                "probabilities": [],
            }
        g = grp[key]
        g["order_line_count"] += 1
        if r_dec == InterventionDecision.INTERVENTION_INDICATED.value:
            g["intervention_indicated_count"] += 1
        elif r_dec == InterventionDecision.NO_INTERVENTION_INDICATED.value:
            g["no_intervention_count"] += 1
        elif r_dec == InterventionDecision.REVIEW_REQUIRED.value:
            g["review_required_count"] += 1
        else:
            g["insufficient_data_count"] += 1

        if r_rev is not None:
            g["revenues"].append(r_rev)
        if r_exp is not None:
            g["exposures"].append(r_exp)
        if r_prob is not None:
            g["probabilities"].append(r_prob)

    resolved_policy_version = policy_version or ""

    for rec in recommendations:
        if isinstance(rec, dict):
            d_val = rec.get("decision", InterventionDecision.INSUFFICIENT_DATA)
            dec_str = d_val.value if hasattr(d_val, "value") else str(d_val)
            f_val = rec.get("financial_status", FinancialInputStatus.INSUFFICIENT)
            f_str = f_val.value if hasattr(f_val, "value") else str(f_val)
            rev_val = rec.get("revenue")
            imp_val = rec.get("estimated_return_impact")
            exp_val = rec.get("expected_return_exposure")
            prob_val = rec.get("calibrated_return_probability")
            sku_val = str(rec.get("sku_id", "UNKNOWN"))
            ch_val = str(rec.get("channel_id", "UNKNOWN"))
            wh_val = str(rec.get("warehouse_id", "UNKNOWN"))
            rb_val = str(rec.get("risk_band") or "UNCLASSIFIED")
            if not resolved_policy_version:
                resolved_policy_version = str(rec.get("policy_version", ""))
        else:
            dec_str = rec.decision.value
            f_str = rec.financial_status.value
            rev_val = rec.revenue
            imp_val = rec.estimated_return_impact
            exp_val = rec.expected_return_exposure
            prob_val = rec.calibrated_return_probability
            sku_val = rec.sku_id or "UNKNOWN"
            ch_val = rec.channel_id or "UNKNOWN"
            wh_val = rec.warehouse_id or "UNKNOWN"
            rb_val = str(rec.risk_band) if rec.risk_band else "UNCLASSIFIED"
            if not resolved_policy_version:
                resolved_policy_version = rec.policy_version

        # Count decisions
        if dec_str in decision_counts:
            decision_counts[dec_str] += 1
        else:
            decision_counts[dec_str] = 1

        # Count financial status
        if f_str == FinancialInputStatus.SUFFICIENT.value:
            sufficient_cnt += 1
        elif f_str == FinancialInputStatus.PARTIAL.value:
            partial_cnt += 1
        else:
            insufficient_cnt += 1

        if rev_val is not None and not pd.isna(rev_val):
            revenues.append(float(rev_val))
        if imp_val is not None and not pd.isna(imp_val):
            impacts.append(float(imp_val))
        if exp_val is not None and not pd.isna(exp_val):
            exposures.append(float(exp_val))

        r_flt = float(rev_val) if rev_val is not None and not pd.isna(rev_val) else None
        e_flt = float(exp_val) if exp_val is not None and not pd.isna(exp_val) else None
        p_flt = float(prob_val) if prob_val is not None and not pd.isna(prob_val) else None

        _update_group(sku_grp, sku_val, dec_str, r_flt, e_flt, p_flt)
        _update_group(ch_grp, ch_val, dec_str, r_flt, e_flt, p_flt)
        _update_group(wh_grp, wh_val, dec_str, r_flt, e_flt, p_flt)
        _update_group(rb_grp, rb_val, dec_str, r_flt, e_flt, p_flt)

    def _finalize_group(grp: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for k, v in grp.items():
            revs = v["revenues"]
            exps = v["exposures"]
            probs = v["probabilities"]
            out[k] = {
                "order_line_count": v["order_line_count"],
                "intervention_indicated_count": v["intervention_indicated_count"],
                "no_intervention_count": v["no_intervention_count"],
                "review_required_count": v["review_required_count"],
                "insufficient_data_count": v["insufficient_data_count"],
                "total_revenue": round(float(sum(revs)), 2) if revs else None,
                "total_expected_exposure": round(float(sum(exps)), 2) if exps else None,
                "mean_calibrated_probability": round(float(np.mean(probs)), 4) if probs else None,
            }
        return out

    tot_rev = round(float(sum(revenues)), 2) if revenues else None
    tot_imp = round(float(sum(impacts)), 2) if impacts else None
    tot_exp = round(float(sum(exposures)), 2) if exposures else None

    return PortfolioInterventionSummary(
        total_order_lines_analyzed=total_count,
        sufficient_financial_records=sufficient_cnt,
        partial_financial_records=partial_cnt,
        insufficient_financial_records=insufficient_cnt,
        intervention_indicated_count=decision_counts[InterventionDecision.INTERVENTION_INDICATED.value],
        no_intervention_count=decision_counts[InterventionDecision.NO_INTERVENTION_INDICATED.value],
        review_required_count=decision_counts[InterventionDecision.REVIEW_REQUIRED.value],
        insufficient_data_count=decision_counts[InterventionDecision.INSUFFICIENT_DATA.value],
        total_revenue_represented=tot_rev,
        total_estimated_return_impact=tot_imp,
        total_expected_return_exposure=tot_exp,
        breakdown_by_sku=_finalize_group(sku_grp),
        breakdown_by_channel=_finalize_group(ch_grp),
        breakdown_by_warehouse=_finalize_group(wh_grp),
        breakdown_by_risk_band=_finalize_group(rb_grp),
        breakdown_by_decision=decision_counts,
        policy_version=resolved_policy_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def portfolio_breakdown_to_dataframe(
    summary: PortfolioInterventionSummary,
    dimension: str = "sku",
) -> pd.DataFrame:
    """Flatten a specific dimension breakdown from PortfolioInterventionSummary into a pandas DataFrame."""
    dim_lower = dimension.strip().lower()
    if dim_lower in ("sku", "sku_id"):
        data = summary.breakdown_by_sku
        col_name = "sku_id"
    elif dim_lower in ("channel", "channel_id"):
        data = summary.breakdown_by_channel
        col_name = "channel_id"
    elif dim_lower in ("warehouse", "warehouse_id"):
        data = summary.breakdown_by_warehouse
        col_name = "warehouse_id"
    elif dim_lower in ("risk_band", "band"):
        data = summary.breakdown_by_risk_band
        col_name = "risk_band"
    else:
        raise ValueError(f"Unsupported breakdown dimension: '{dimension}'. Use 'sku', 'channel', 'warehouse', or 'risk_band'.")

    if not data:
        return pd.DataFrame()

    rows = []
    for k, v in data.items():
        row = {col_name: k}
        row.update(v)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# =====================================================================
# Service Class
# =====================================================================


class ReturnInterventionService:
    """Unified service for evaluating return interventions and portfolio cost-utility."""

    def __init__(self, policy: Optional[InterventionPolicyConfig] = None):
        self.policy = policy or InterventionPolicyConfig()

    def evaluate_order_line(
        self,
        record: Union[Dict[str, Any], pd.Series, ReturnRiskResult],
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> ReturnInterventionRecommendation:
        """Evaluate a single order line under configured policy."""
        return evaluate_order_line(record, policy=self.policy, as_of_date=as_of_date)

    def evaluate_order_lines(
        self,
        records: Union[pd.DataFrame, Sequence[Union[Dict[str, Any], ReturnRiskResult]]],
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> List[ReturnInterventionRecommendation]:
        """Evaluate a batch or DataFrame of order lines."""
        return evaluate_order_lines(records, policy=self.policy, as_of_date=as_of_date)

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> pd.DataFrame:
        """Evaluate a DataFrame and return an enriched pandas DataFrame."""
        if df is None or df.empty:
            return pd.DataFrame()
        recs = self.evaluate_order_lines(df, as_of_date=as_of_date)
        return pd.DataFrame([r.to_dict() for r in recs])

    def summarize_portfolio(
        self,
        recommendations: Sequence[Union[ReturnInterventionRecommendation, Dict[str, Any]]],
    ) -> PortfolioInterventionSummary:
        """Aggregate evaluated recommendations into an executive portfolio summary."""
        return summarize_portfolio(recommendations, policy_version=self.policy.policy_version)
