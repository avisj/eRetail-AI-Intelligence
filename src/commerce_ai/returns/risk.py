"""Return Risk Layer & Policy Engine (Phase 5C-2B).

Classifies calibrated return probabilities into configurable business risk bands:
- VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
- Strict validation of non-overlapping probability intervals spanning [0.0, 1.0]
- Deterministic, non-causal statistical interpretations (no LLM, no causal speculation)
- Empirical validation of risk policies (observed return frequency vs average probability per band)
- Segment-level calibration auditing with sample size reliability guards

Architectural Principles:
- Informational output only (never creates POs, modifies inventory, or triggers autonomous actions).
- Raw and calibrated probabilities are explicitly preserved side-by-side.
- All threshold and boundary assignments are fully configurable and policy-versioned.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from commerce_ai.returns.schemas import (
    ReturnRiskBand,
    ReturnRiskResult,
    RiskBandBoundary,
    RiskBandEvaluation,
    RiskPolicyConfig,
)
from commerce_ai.returns.calibration import (
    compute_calibration_metrics,
    validate_probability_bounds,
)


DEFAULT_RISK_SEGMENT_DIMENSIONS: List[str] = [
    "channel_id",
    "warehouse_id",
    "velocity_tier",
    "is_cold_start_sku",
]


class ReturnRiskClassifier:
    """Configurable classifier mapping return probabilities to operational risk bands."""

    def __init__(self, policy: Optional[RiskPolicyConfig] = None):
        self.policy = policy or RiskPolicyConfig()

    def classify_probability(self, probability: float) -> ReturnRiskBand:
        """Map a single probability value in [0.0, 1.0] to a ReturnRiskBand.

        Args:
            probability: Estimated or calibrated return probability.

        Returns:
            ReturnRiskBand enum member.
        """
        if probability < -1e-6 or probability > 1.0 + 1e-6:
            raise ValueError(f"Probability must be in [0.0, 1.0], got {probability}")
        p = float(np.clip(probability, 0.0, 1.0))

        for b in self.policy.boundaries:
            # Last boundary is inclusive on upper bound
            if b.upper_bound >= 1.0 - 1e-6:
                if b.lower_bound <= p <= b.upper_bound:
                    return b.band
            else:
                if b.lower_bound <= p < b.upper_bound:
                    return b.band

        # Fallback to last boundary if numerical rounding near 1.0
        return self.policy.boundaries[-1].band

    def generate_interpretation(
        self,
        probability: float,
        risk_band: ReturnRiskBand,
    ) -> str:
        """Generate a deterministic, non-causal statistical interpretation string.

        Guarantees:
        - Strictly non-causal (does not claim customer intent or causal triggers).
        - Grounded entirely in empirical probability modeling.
        """
        pct = int(round(float(probability) * 100))
        band_str = risk_band.value if hasattr(risk_band, "value") else str(risk_band)
        return (
            f"Risk Band: {band_str}. Historical evaluation indicates approximately a "
            f"{pct}% modeled probability of return for records receiving this calibrated probability."
        )

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        raw_prob_col: str = "raw_probability",
        cal_prob_col: str = "calibrated_probability",
        model_name: str = "Unknown",
        calibration_method: str = "none",
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """Apply risk classification across an entire predictions DataFrame.

        Args:
            df: Predictions DataFrame containing metadata and probability columns.
            raw_prob_col: Column name containing uncalibrated model probabilities.
            cal_prob_col: Column name containing calibrated model probabilities.
            model_name: Name of the generating model.
            calibration_method: Calibration technique label.
            threshold: Decision threshold.

        Returns:
            DataFrame augmented with risk_band, risk_policy_version, and interpretation.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # Handle column naming flexibility
        r_col = raw_prob_col if raw_prob_col in df.columns else (
            "predicted_probability" if "predicted_probability" in df.columns else None
        )
        c_col = cal_prob_col if cal_prob_col in df.columns else r_col

        if r_col is None or c_col is None:
            raise ValueError(
                f"Missing probability columns in DataFrame: expected '{raw_prob_col}' or '{cal_prob_col}'"
            )

        df_out = df.copy()
        raw_probs = validate_probability_bounds(df_out[r_col].values, param_name="raw_probability")
        cal_probs = validate_probability_bounds(df_out[c_col].values, param_name="calibrated_probability")

        risk_bands: List[str] = []
        interpretations: List[str] = []

        for cp in cal_probs:
            band = self.classify_probability(cp)
            interp = self.generate_interpretation(cp, band)
            risk_bands.append(band.value)
            interpretations.append(interp)

        df_out["raw_probability"] = np.round(raw_probs, 4)
        df_out["calibrated_probability"] = np.round(cal_probs, 4)
        df_out["risk_band"] = risk_bands
        df_out["risk_policy_version"] = self.policy.policy_version
        df_out["calibration_method"] = calibration_method
        df_out["interpretation"] = interpretations
        df_out["threshold"] = round(float(threshold), 4)

        if "model_name" not in df_out.columns:
            df_out["model_name"] = model_name

        return df_out


# =====================================================================
# Risk Policy Audit & Validation Utilities
# =====================================================================


def evaluate_risk_policy(
    df_results: pd.DataFrame,
    band_col: str = "risk_band",
    cal_prob_col: str = "calibrated_probability",
    actual_col: str = "actual_target",
) -> List[RiskBandEvaluation]:
    """Audit empirical validity of risk bands against observed return frequencies.

    Args:
        df_results: DataFrame of classified risk results.
        band_col: Column containing assigned risk bands.
        cal_prob_col: Column containing calibrated probabilities.
        actual_col: Column containing ground truth binary return outcomes (0 or 1).

    Returns:
        List of RiskBandEvaluation objects.
    """
    if df_results is None or df_results.empty:
        return []
    if band_col not in df_results.columns or cal_prob_col not in df_results.columns:
        return []

    total_records = len(df_results)
    evaluations: List[RiskBandEvaluation] = []

    # Standard risk band order
    ordered_bands = [b.value for b in ReturnRiskBand]
    present_bands = list(df_results[band_col].unique())
    # Sort by standard order
    sorted_bands = [b for b in ordered_bands if b in present_bands] + [
        b for b in present_bands if b not in ordered_bands
    ]

    for band in sorted_bands:
        sub = df_results[df_results[band_col] == band]
        n = len(sub)
        pct = round(float((n / total_records) * 100), 2) if total_records > 0 else 0.0

        cal_p = sub[cal_prob_col].values.astype(float)
        mean_p = round(float(np.mean(cal_p)), 4) if n > 0 else 0.0

        if actual_col in sub.columns and not sub[actual_col].isna().all():
            y_act = sub[actual_col].dropna().values.astype(int)
            ret_cnt = int(np.sum(y_act == 1))
            obs_rate = round(float(ret_cnt / len(y_act)), 4) if len(y_act) > 0 else 0.0
            cal_err = round(float(abs(mean_p - obs_rate)), 4)
        else:
            ret_cnt = 0
            obs_rate = 0.0
            cal_err = 0.0

        evaluations.append(
            RiskBandEvaluation(
                risk_band=band,
                record_count=n,
                percentage_of_records=pct,
                actual_returned_count=ret_cnt,
                observed_return_rate=obs_rate,
                average_calibrated_probability=mean_p,
                calibration_error=cal_err,
            )
        )

    return evaluations


def risk_evaluations_to_dataframe(evaluations: List[RiskBandEvaluation]) -> pd.DataFrame:
    """Convert RiskBandEvaluation list into a flat pandas DataFrame."""
    if not evaluations:
        return pd.DataFrame()
    return pd.DataFrame([e.to_dict() for e in evaluations])


def evaluate_segment_calibration(
    df_results: pd.DataFrame,
    dimensions: Optional[List[str]] = None,
    cal_prob_col: str = "calibrated_probability",
    actual_col: str = "actual_target",
    min_sample_size: int = 30,
) -> pd.DataFrame:
    """Audit calibration performance across operational dimensions with sample size guards."""
    if df_results is None or df_results.empty:
        return pd.DataFrame()

    dims = dimensions or DEFAULT_RISK_SEGMENT_DIMENSIONS
    rows: List[Dict[str, Any]] = []

    for dim in dims:
        if dim not in df_results.columns:
            continue

        for val, group in df_results.groupby(dim, observed=False):
            n = len(group)
            if n == 0:
                continue

            cal_p = group[cal_prob_col].values.astype(float)
            mean_p = round(float(np.mean(cal_p)), 4)

            is_sufficient = bool(n >= min_sample_size)

            if actual_col in group.columns and not group[actual_col].isna().all():
                y_act = group[actual_col].dropna().values.astype(int)
                ret_cnt = int(np.sum(y_act == 1))
                obs_rate = round(float(ret_cnt / len(y_act)), 4) if len(y_act) > 0 else 0.0
                cal_err = round(float(abs(mean_p - obs_rate)), 4)

                # Compute ECE if sufficient sample
                if is_sufficient and len(np.unique(y_act)) > 1:
                    metrics = compute_calibration_metrics(y_act, cal_p, method_name=dim, n_bins=10)
                    brier = metrics.brier_score
                    ece = metrics.ece
                else:
                    brier = round(float(np.mean((cal_p - y_act) ** 2)), 4) if len(y_act) > 0 else None
                    ece = None
            else:
                ret_cnt = 0
                obs_rate = 0.0
                cal_err = 0.0
                brier = None
                ece = None

            rows.append({
                "dimension": dim,
                "segment_value": str(val),
                "sample_size": n,
                "positive_count": ret_cnt,
                "positive_rate": obs_rate,
                "mean_calibrated_probability": mean_p,
                "calibration_error": cal_err,
                "brier_score": brier,
                "ece": ece,
                "is_sufficient_sample": is_sufficient,
            })

    return pd.DataFrame(rows)
