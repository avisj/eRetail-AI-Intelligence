"""Deterministic Statistical Return Anomaly Detection Engine (Phase 5B).

Provides deterministic and statistical methods to identify unusual historical return behavior:
- Rolling mean and sample standard deviation Z-Scores
- Rolling median and Median Absolute Deviation (MAD) modified Z-scores
- Period-over-period delta and percentage change
- Multi-dimensional detection: SKU, Channel, Warehouse, SKU×Channel, SKU×Warehouse, Reason, Time Series
- Multi-tier severity classification: CRITICAL, HIGH, MEDIUM, LOW, NONE
- Zero-variance handling (BASELINE_SHIFT, NO_ANOMALY)
- Sample size guards (min_sold_units, min_return_count, min_history_periods)
- Deterministic SHA-256 identifiers and factual non-LLM rationales
- Anti-leakage chronological date filtering
"""

from __future__ import annotations

from datetime import date, datetime
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.returns.schemas import (
    AnomalyDirection,
    AnomalySeverity,
    AnomalyType,
    ReturnAnomaly,
    ReturnAnomalyConfig,
    ReturnAnomalyResult,
)
from commerce_ai.returns.analytics import filter_by_as_of_date


# =====================================================================
# Deterministic Identifiers & Rationale Generators
# =====================================================================


def generate_deterministic_anomaly_id(
    dimension: str,
    entity_id: str,
    period: Optional[str],
    anomaly_type: Union[str, AnomalyType],
    as_of_date: Optional[str] = None,
) -> str:
    """Generate a deterministic 16-character SHA-256 anomaly identifier.

    Hash format: {dimension}|{entity_id}|{period}|{anomaly_type}|{as_of_date}
    """
    type_str = anomaly_type.value if isinstance(anomaly_type, AnomalyType) else str(anomaly_type)
    raw = f"{dimension.strip().upper()}|{entity_id.strip()}|{str(period or '').strip()}|{type_str.strip()}|{str(as_of_date or '').strip()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"ANOM-{digest[:16]}"


def generate_anomaly_rationale(
    dimension: str,
    entity_id: str,
    metric_name: str,
    current_value: float,
    baseline_value: float,
    anomaly_type: AnomalyType,
    severity: AnomalySeverity,
    direction: AnomalyDirection,
    z_score: Optional[float] = None,
    period: Optional[str] = None,
    sample_size: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate an explainable, factual, deterministic rationale string with no LLM hallucinations."""
    p_str = f" in period '{period}'" if period else ""

    if anomaly_type == AnomalyType.INSUFFICIENT_SAMPLE:
        min_sold = metadata.get("min_sold_units", 30) if metadata else 30
        min_ret = metadata.get("min_return_count", 5) if metadata else 5
        return (
            f"Entity '{entity_id}' ({dimension}){p_str} has insufficient sample size "
            f"(sample_size={sample_size or 0}, required min_sold={min_sold}, min_return_count={min_ret}). "
            f"Reliable anomaly detection not possible."
        )

    if anomaly_type == AnomalyType.BASELINE_SHIFT:
        return (
            f"Entity '{entity_id}' ({dimension}){p_str} observed {metric_name} = {current_value:.4f}, "
            f"shifting from constant historical baseline {baseline_value:.4f} (baseline std = 0.0000). "
            f"Severity: {severity.value}."
        )

    if anomaly_type == AnomalyType.RETURN_REASON_SHIFT:
        delta = current_value - baseline_value
        return (
            f"Return reason '{entity_id}' share shifted by {delta:+.2%} "
            f"(current: {current_value:.2%}, baseline: {baseline_value:.2%}){p_str}. "
            f"Direction: {direction.value}, Severity: {severity.value}."
        )

    if anomaly_type in (AnomalyType.RETURN_VOLUME_SPIKE, AnomalyType.RETURN_VOLUME_DROP):
        z_str = f", z-score: {z_score:+.2f}" if z_score is not None else ""
        return (
            f"Entity '{entity_id}' ({dimension}){p_str} observed returned units {current_value:.0f} "
            f"vs baseline mean {baseline_value:.1f}{z_str}. "
            f"Direction: {direction.value}, Severity: {severity.value}."
        )

    if anomaly_type in (
        AnomalyType.CHANNEL_RETURN_SHIFT,
        AnomalyType.WAREHOUSE_RETURN_SHIFT,
        AnomalyType.SKU_CHANNEL_RETURN_SHIFT,
        AnomalyType.SKU_WAREHOUSE_RETURN_SHIFT,
    ):
        z_str = f", z-score: {z_score:+.2f}" if z_score is not None else ""
        return (
            f"{dimension} '{entity_id}'{p_str} observed return rate {current_value:.2%} "
            f"vs baseline {baseline_value:.2%}{z_str}. "
            f"Direction: {direction.value}, Severity: {severity.value}."
        )

    if anomaly_type in (AnomalyType.RETURN_RATE_SPIKE, AnomalyType.RETURN_RATE_DROP):
        z_str = f", z-score: {z_score:+.2f}" if z_score is not None else ""
        return (
            f"Entity '{entity_id}' ({dimension}){p_str} observed return rate {current_value:.2%} "
            f"vs baseline mean {baseline_value:.2%}{z_str}. "
            f"Direction: {direction.value}, Severity: {severity.value}."
        )

    return (
        f"Entity '{entity_id}' ({dimension}){p_str} {metric_name} = {current_value:.4f} "
        f"is consistent with baseline {baseline_value:.4f} (within normal variance, severity: NONE)."
    )


# =====================================================================
# Statistical Scoring & Classification Helpers
# =====================================================================


def determine_direction(current: float, baseline: float, tol: float = 1e-6) -> AnomalyDirection:
    """Classify deviation direction relative to baseline."""
    if current > baseline + tol:
        return AnomalyDirection.INCREASE
    if current < baseline - tol:
        return AnomalyDirection.DECREASE
    return AnomalyDirection.NONE


def determine_severity_from_z(
    abs_z: float,
    config: ReturnAnomalyConfig,
) -> AnomalySeverity:
    """Map absolute z-score to severity classification."""
    if abs_z >= config.critical_z_score:
        return AnomalySeverity.CRITICAL
    if abs_z >= config.high_z_score:
        return AnomalySeverity.HIGH
    if abs_z >= config.medium_z_score:
        return AnomalySeverity.MEDIUM
    if abs_z >= config.low_z_score:
        return AnomalySeverity.LOW
    return AnomalySeverity.NONE


def compute_z_score(
    current: float,
    baseline_mean: float,
    baseline_std: float,
    config: Optional[ReturnAnomalyConfig] = None,
) -> Tuple[Optional[float], AnomalySeverity, AnomalyType, AnomalyDirection]:
    """Compute standard sample Z-score and classify severity and anomaly type."""
    cfg = config or ReturnAnomalyConfig()
    direction = determine_direction(current, baseline_mean)

    if baseline_std < 1e-9:
        if abs(current - baseline_mean) < 1e-6:
            return 0.0, AnomalySeverity.NONE, AnomalyType.NO_ANOMALY, AnomalyDirection.NONE

        abs_diff = abs(current - baseline_mean)
        if abs_diff >= 0.20:
            severity = AnomalySeverity.CRITICAL
        elif abs_diff >= 0.10:
            severity = AnomalySeverity.HIGH
        elif abs_diff >= 0.05:
            severity = AnomalySeverity.MEDIUM
        else:
            severity = AnomalySeverity.LOW
        return None, severity, AnomalyType.BASELINE_SHIFT, direction

    z = float((current - baseline_mean) / baseline_std)
    abs_z = abs(z)
    severity = determine_severity_from_z(abs_z, cfg)

    if abs_z >= cfg.z_score_threshold:
        anomaly_type = AnomalyType.RETURN_RATE_SPIKE if z > 0 else AnomalyType.RETURN_RATE_DROP
    else:
        anomaly_type = AnomalyType.NO_ANOMALY

    return z, severity, anomaly_type, direction


def compute_mad_score(
    current: float,
    baseline_median: float,
    mad: float,
    config: Optional[ReturnAnomalyConfig] = None,
) -> Tuple[Optional[float], AnomalySeverity, AnomalyType, AnomalyDirection]:
    """Compute modified Z-score using Median Absolute Deviation (MAD)."""
    cfg = config or ReturnAnomalyConfig()
    direction = determine_direction(current, baseline_median)

    if mad < 1e-9:
        if abs(current - baseline_median) < 1e-6:
            return 0.0, AnomalySeverity.NONE, AnomalyType.NO_ANOMALY, AnomalyDirection.NONE

        abs_diff = abs(current - baseline_median)
        if abs_diff >= 0.20:
            severity = AnomalySeverity.CRITICAL
        elif abs_diff >= 0.10:
            severity = AnomalySeverity.HIGH
        elif abs_diff >= 0.05:
            severity = AnomalySeverity.MEDIUM
        else:
            severity = AnomalySeverity.LOW
        return None, severity, AnomalyType.BASELINE_SHIFT, direction

    modified_z = float(0.6745 * (current - baseline_median) / mad)
    abs_z = abs(modified_z)
    severity = determine_severity_from_z(abs_z, cfg)

    if abs_z >= cfg.z_score_threshold:
        anomaly_type = AnomalyType.RETURN_RATE_SPIKE if modified_z > 0 else AnomalyType.RETURN_RATE_DROP
    else:
        anomaly_type = AnomalyType.NO_ANOMALY

    return modified_z, severity, anomaly_type, direction


# =====================================================================
# Modular Anomaly Evaluator Functions
# =====================================================================


def detect_series_anomaly(
    current_value: float,
    baseline_values: Sequence[float],
    dimension: str,
    entity_id: str,
    metric_name: str = "return_rate",
    period: Optional[str] = None,
    as_of_date: Optional[str] = None,
    config: Optional[ReturnAnomalyConfig] = None,
    sample_size: int = 100,
    sample_return_count: int = 10,
    metadata: Optional[Dict[str, Any]] = None,
) -> ReturnAnomaly:
    """Evaluate an observed metric value against its historical baseline sequence.

    Handles zero-variance baselines, sample size guards, robust MAD, and deterministic reporting.
    """
    cfg = config or ReturnAnomalyConfig()
    meta = dict(metadata or {})
    meta["min_sold_units"] = cfg.min_sold_units
    meta["min_return_count"] = cfg.min_return_count
    meta["min_history_periods"] = cfg.min_history_periods

    # 1. Check sample size guards (only for rate metrics)
    if metric_name == "return_rate":
        if sample_size < cfg.min_sold_units or sample_return_count < cfg.min_return_count:
            anom_id = generate_deterministic_anomaly_id(
                dimension, entity_id, period, AnomalyType.INSUFFICIENT_SAMPLE, as_of_date
            )
            base_val = float(np.mean(baseline_values)) if baseline_values else 0.0
            rationale = generate_anomaly_rationale(
                dimension=dimension,
                entity_id=entity_id,
                metric_name=metric_name,
                current_value=current_value,
                baseline_value=base_val,
                anomaly_type=AnomalyType.INSUFFICIENT_SAMPLE,
                severity=AnomalySeverity.NONE,
                direction=AnomalyDirection.NONE,
                period=period,
                sample_size=sample_size,
                metadata=meta,
            )
            return ReturnAnomaly(
                anomaly_id=anom_id,
                dimension=dimension,
                entity_id=entity_id,
                anomaly_type=AnomalyType.INSUFFICIENT_SAMPLE,
                severity=AnomalySeverity.NONE,
                direction=AnomalyDirection.NONE,
                current_value=current_value,
                baseline_value=base_val,
                metric_name=metric_name,
                z_score=None,
                mad_score=None,
                percentage_change=None,
                sample_size=sample_size,
                is_sufficient_sample=False,
                period=period,
                as_of_date=as_of_date,
                rationale=rationale,
                metadata=meta,
            )

    # 2. Check historical periods guard
    if len(baseline_values) < cfg.min_history_periods:
        anom_id = generate_deterministic_anomaly_id(
            dimension, entity_id, period, AnomalyType.INSUFFICIENT_SAMPLE, as_of_date
        )
        base_val = float(np.mean(baseline_values)) if baseline_values else 0.0
        meta["insufficient_reason"] = f"Only {len(baseline_values)} history periods (minimum {cfg.min_history_periods})"
        rationale = (
            f"Entity '{entity_id}' ({dimension}) in period '{period or 'N/A'}' has only {len(baseline_values)} "
            f"historical periods (minimum required: {cfg.min_history_periods}). Reliable baseline cannot be established."
        )
        return ReturnAnomaly(
            anomaly_id=anom_id,
            dimension=dimension,
            entity_id=entity_id,
            anomaly_type=AnomalyType.INSUFFICIENT_SAMPLE,
            severity=AnomalySeverity.NONE,
            direction=AnomalyDirection.NONE,
            current_value=current_value,
            baseline_value=base_val,
            metric_name=metric_name,
            z_score=None,
            mad_score=None,
            percentage_change=None,
            sample_size=sample_size,
            is_sufficient_sample=False,
            period=period,
            as_of_date=as_of_date,
            rationale=rationale,
            metadata=meta,
        )

    # 3. Compute baseline statistics
    base_arr = np.array(baseline_values, dtype=float)
    dim_upper = dimension.strip().upper()

    if cfg.use_mad:
        med = float(np.median(base_arr))
        abs_devs = np.abs(base_arr - med)
        mad = float(np.median(abs_devs))
        score, severity, a_type, direction = compute_mad_score(current_value, med, mad, cfg)
        z_val = None
        mad_val = score
        baseline_ref = med
    else:
        mean = float(np.mean(base_arr))
        std = float(np.std(base_arr, ddof=1)) if len(base_arr) > 1 else 0.0
        score, severity, a_type, direction = compute_z_score(current_value, mean, std, cfg)
        z_val = score
        mad_val = None
        baseline_ref = mean

    # Refine anomaly type based on dimension and metric
    final_type = a_type
    if a_type in (AnomalyType.RETURN_RATE_SPIKE, AnomalyType.RETURN_RATE_DROP):
        if dim_upper == "CHANNEL":
            final_type = AnomalyType.CHANNEL_RETURN_SHIFT
        elif dim_upper == "WAREHOUSE":
            final_type = AnomalyType.WAREHOUSE_RETURN_SHIFT
        elif dim_upper == "SKU_CHANNEL":
            final_type = AnomalyType.SKU_CHANNEL_RETURN_SHIFT
        elif dim_upper == "SKU_WAREHOUSE":
            final_type = AnomalyType.SKU_WAREHOUSE_RETURN_SHIFT
        elif metric_name == "returned_units":
            final_type = AnomalyType.RETURN_VOLUME_SPIKE if direction == AnomalyDirection.INCREASE else AnomalyType.RETURN_VOLUME_DROP

    # Percentage change calculation
    pct_change = None
    if baseline_ref > 1e-9:
        pct_change = ((current_value - baseline_ref) / baseline_ref) * 100.0
    elif abs(current_value) < 1e-9 and abs(baseline_ref) < 1e-9:
        pct_change = 0.0

    anom_id = generate_deterministic_anomaly_id(dimension, entity_id, period, final_type, as_of_date)
    rationale = generate_anomaly_rationale(
        dimension=dimension,
        entity_id=entity_id,
        metric_name=metric_name,
        current_value=current_value,
        baseline_value=baseline_ref,
        anomaly_type=final_type,
        severity=severity,
        direction=direction,
        z_score=z_val if z_val is not None else mad_val,
        period=period,
        sample_size=sample_size,
        metadata=meta,
    )

    return ReturnAnomaly(
        anomaly_id=anom_id,
        dimension=dimension,
        entity_id=entity_id,
        anomaly_type=final_type,
        severity=severity,
        direction=direction,
        current_value=current_value,
        baseline_value=baseline_ref,
        metric_name=metric_name,
        z_score=z_val,
        mad_score=mad_val,
        percentage_change=pct_change,
        sample_size=sample_size,
        is_sufficient_sample=True,
        period=period,
        as_of_date=as_of_date,
        rationale=rationale,
        metadata=meta,
    )


def detect_reason_shift_anomaly(
    reason: str,
    current_units: int,
    current_total_units: int,
    baseline_units: int,
    baseline_total_units: int,
    period: Optional[str] = None,
    as_of_date: Optional[str] = None,
    config: Optional[ReturnAnomalyConfig] = None,
) -> ReturnAnomaly:
    """Detect disproportionate shifts in return reason share relative to historical baseline."""
    cfg = config or ReturnAnomalyConfig()
    meta = {
        "current_reason_units": current_units,
        "current_total_returned_units": current_total_units,
        "baseline_reason_units": baseline_units,
        "baseline_total_returned_units": baseline_total_units,
    }

    # Sample size guard for returns
    if current_total_units < cfg.min_return_count or baseline_total_units < cfg.min_return_count:
        anom_id = generate_deterministic_anomaly_id(
            "REASON", reason, period, AnomalyType.INSUFFICIENT_SAMPLE, as_of_date
        )
        curr_share = current_units / current_total_units if current_total_units > 0 else 0.0
        base_share = baseline_units / baseline_total_units if baseline_total_units > 0 else 0.0
        rationale = (
            f"Return reason '{reason}' in period '{period or 'N/A'}' has insufficient return volume "
            f"(current returns: {current_total_units}, baseline returns: {baseline_total_units}, "
            f"minimum required: {cfg.min_return_count})."
        )
        return ReturnAnomaly(
            anomaly_id=anom_id,
            dimension="REASON",
            entity_id=reason,
            anomaly_type=AnomalyType.INSUFFICIENT_SAMPLE,
            severity=AnomalySeverity.NONE,
            direction=AnomalyDirection.NONE,
            current_value=curr_share,
            baseline_value=base_share,
            metric_name="reason_share",
            z_score=None,
            mad_score=None,
            percentage_change=None,
            sample_size=current_total_units,
            is_sufficient_sample=False,
            period=period,
            as_of_date=as_of_date,
            rationale=rationale,
            metadata=meta,
        )

    curr_share = current_units / current_total_units
    base_share = baseline_units / baseline_total_units
    delta_share = curr_share - base_share
    abs_delta = abs(delta_share)
    direction = determine_direction(curr_share, base_share)

    if abs_delta >= cfg.reason_shift_threshold:
        anomaly_type = AnomalyType.RETURN_REASON_SHIFT
        if abs_delta >= 0.30:
            severity = AnomalySeverity.CRITICAL
        elif abs_delta >= 0.25:
            severity = AnomalySeverity.HIGH
        elif abs_delta >= 0.20:
            severity = AnomalySeverity.MEDIUM
        else:
            severity = AnomalySeverity.LOW
    else:
        anomaly_type = AnomalyType.NO_ANOMALY
        severity = AnomalySeverity.NONE
        direction = AnomalyDirection.NONE

    pct_change = ((curr_share - base_share) / base_share * 100.0) if base_share > 1e-9 else None
    anom_id = generate_deterministic_anomaly_id("REASON", reason, period, anomaly_type, as_of_date)
    rationale = generate_anomaly_rationale(
        dimension="REASON",
        entity_id=reason,
        metric_name="reason_share",
        current_value=curr_share,
        baseline_value=base_share,
        anomaly_type=anomaly_type,
        severity=severity,
        direction=direction,
        period=period,
        sample_size=current_total_units,
        metadata=meta,
    )

    return ReturnAnomaly(
        anomaly_id=anom_id,
        dimension="REASON",
        entity_id=reason,
        anomaly_type=anomaly_type,
        severity=severity,
        direction=direction,
        current_value=curr_share,
        baseline_value=base_share,
        metric_name="reason_share",
        z_score=None,
        mad_score=None,
        percentage_change=pct_change,
        sample_size=current_total_units,
        is_sufficient_sample=True,
        period=period,
        as_of_date=as_of_date,
        rationale=rationale,
        metadata=meta,
    )


# =====================================================================
# Main Service Class
# =====================================================================


class ReturnAnomalyService:
    """Service providing end-to-end, multi-dimensional, deterministic return anomaly detection."""

    def __init__(self, config: Optional[ReturnAnomalyConfig] = None):
        self.config = config or ReturnAnomalyConfig()

    def detect(
        self,
        returns: pd.DataFrame,
        sales: Optional[pd.DataFrame] = None,
        products: Optional[pd.DataFrame] = None,
        warehouses: Optional[pd.DataFrame] = None,
        channels: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        current_period: Optional[str] = None,
        baseline_periods: Optional[List[str]] = None,
        config: Optional[ReturnAnomalyConfig] = None,
        time_series_freq: str = "M",
        include_insufficient_sample: bool = False,
        include_normal: bool = False,
    ) -> ReturnAnomalyResult:
        """Execute multi-dimensional anomaly detection on customer returns.

        Args:
            returns: Returns transaction DataFrame.
            sales: Sales transactions DataFrame.
            products: Optional products DataFrame.
            warehouses: Optional warehouses DataFrame.
            channels: Optional channels DataFrame.
            as_of_date: Evaluation cutoff date for anti-leakage compliance.
            current_period: Explicit evaluation period (e.g. '2026-03'). Defaults to latest period.
            baseline_periods: Explicit baseline period list. Defaults to all preceding periods.
            config: Optional configuration overrides.
            time_series_freq: Aggregation frequency ('M', 'W', 'D'). Defaults to monthly ('M').
            include_insufficient_sample: Whether to include INSUFFICIENT_SAMPLE records in the output.
            include_normal: Whether to include NO_ANOMALY records in the output.

        Returns:
            ReturnAnomalyResult bundle containing detected anomalies, severity tallies, and DataFrame export.
        """
        cfg = config or self.config
        as_of_str = str(as_of_date) if as_of_date is not None else None

        # 1. Anti-leakage chronological filtering
        ret_df = filter_by_as_of_date(returns, "return_date", as_of_date)
        sales_df = filter_by_as_of_date(sales, "date", as_of_date) if sales is not None else pd.DataFrame()

        if ret_df.empty:
            return ReturnAnomalyResult(as_of_date=as_of_str, config=cfg)

        # 2. Add period columns
        ret_df = ret_df.copy()
        ret_df["period"] = pd.to_datetime(ret_df["return_date"]).dt.to_period(time_series_freq).astype(str)

        if not sales_df.empty and "date" in sales_df.columns:
            sales_df = sales_df.copy()
            sales_df["period"] = pd.to_datetime(sales_df["date"]).dt.to_period(time_series_freq).astype(str)

        # 3. Identify periods
        all_periods = set(ret_df["period"].unique())
        if not sales_df.empty and "period" in sales_df.columns:
            all_periods.update(sales_df["period"].unique())
        sorted_periods = sorted(list(all_periods))

        if not sorted_periods:
            return ReturnAnomalyResult(as_of_date=as_of_str, config=cfg)

        if current_period is not None:
            curr_p = str(current_period)
            base_p = baseline_periods if baseline_periods is not None else [p for p in sorted_periods if p < curr_p]
        else:
            if len(sorted_periods) < 2:
                # Need at least 2 periods to establish baseline vs current
                return ReturnAnomalyResult(as_of_date=as_of_str, config=cfg)
            curr_p = sorted_periods[-1]
            base_p = [p for p in sorted_periods if p < curr_p] if baseline_periods is None else baseline_periods

        detected_anomalies: List[ReturnAnomaly] = []

        # Helper to decide if an anomaly should be included
        def add_anomaly_if_qualifies(anomaly: ReturnAnomaly):
            if anomaly.anomaly_type == AnomalyType.NO_ANOMALY:
                if include_normal:
                    detected_anomalies.append(anomaly)
            elif anomaly.anomaly_type == AnomalyType.INSUFFICIENT_SAMPLE:
                if include_insufficient_sample:
                    detected_anomalies.append(anomaly)
            else:
                detected_anomalies.append(anomaly)

        # 4. Dimension 1: SKU Level
        self._detect_dimensional_anomalies(
            dim_name="SKU",
            group_keys=["sku_id"],
            ret_df=ret_df,
            sales_df=sales_df,
            curr_p=curr_p,
            base_p=base_p,
            as_of_str=as_of_str,
            cfg=cfg,
            add_fn=add_anomaly_if_qualifies,
        )

        # 5. Dimension 2: Channel Level
        if "channel_id" in ret_df.columns or (not sales_df.empty and "channel_id" in sales_df.columns):
            self._detect_dimensional_anomalies(
                dim_name="CHANNEL",
                group_keys=["channel_id"],
                ret_df=ret_df,
                sales_df=sales_df,
                curr_p=curr_p,
                base_p=base_p,
                as_of_str=as_of_str,
                cfg=cfg,
                add_fn=add_anomaly_if_qualifies,
            )

        # 6. Dimension 3: Warehouse Level
        if "warehouse_id" in ret_df.columns or (not sales_df.empty and "warehouse_id" in sales_df.columns):
            self._detect_dimensional_anomalies(
                dim_name="WAREHOUSE",
                group_keys=["warehouse_id"],
                ret_df=ret_df,
                sales_df=sales_df,
                curr_p=curr_p,
                base_p=base_p,
                as_of_str=as_of_str,
                cfg=cfg,
                add_fn=add_anomaly_if_qualifies,
            )

        # 7. Dimension 4: SKU × Channel Level
        if ("sku_id" in ret_df.columns and "channel_id" in ret_df.columns) or (
            not sales_df.empty and "sku_id" in sales_df.columns and "channel_id" in sales_df.columns
        ):
            self._detect_dimensional_anomalies(
                dim_name="SKU_CHANNEL",
                group_keys=["sku_id", "channel_id"],
                ret_df=ret_df,
                sales_df=sales_df,
                curr_p=curr_p,
                base_p=base_p,
                as_of_str=as_of_str,
                cfg=cfg,
                add_fn=add_anomaly_if_qualifies,
            )

        # 8. Dimension 5: SKU × Warehouse Level
        if ("sku_id" in ret_df.columns and "warehouse_id" in ret_df.columns) or (
            not sales_df.empty and "sku_id" in sales_df.columns and "warehouse_id" in sales_df.columns
        ):
            self._detect_dimensional_anomalies(
                dim_name="SKU_WAREHOUSE",
                group_keys=["sku_id", "warehouse_id"],
                ret_df=ret_df,
                sales_df=sales_df,
                curr_p=curr_p,
                base_p=base_p,
                as_of_str=as_of_str,
                cfg=cfg,
                add_fn=add_anomaly_if_qualifies,
            )

        # 9. Dimension 6: Return Reason Shift
        if "return_reason" in ret_df.columns:
            self._detect_reason_shifts(
                ret_df=ret_df,
                curr_p=curr_p,
                base_p=base_p,
                as_of_str=as_of_str,
                cfg=cfg,
                add_fn=add_anomaly_if_qualifies,
            )

        # 10. Dimension 7: Overall Time Series (Portfolio Level)
        self._detect_portfolio_time_series_anomaly(
            ret_df=ret_df,
            sales_df=sales_df,
            curr_p=curr_p,
            base_p=base_p,
            as_of_str=as_of_str,
            cfg=cfg,
            add_fn=add_anomaly_if_qualifies,
        )

        # 11. Compile tallies and result
        crit_count = sum(1 for a in detected_anomalies if a.severity == AnomalySeverity.CRITICAL)
        high_count = sum(1 for a in detected_anomalies if a.severity == AnomalySeverity.HIGH)
        med_count = sum(1 for a in detected_anomalies if a.severity == AnomalySeverity.MEDIUM)
        low_count = sum(1 for a in detected_anomalies if a.severity == AnomalySeverity.LOW)
        total_anom = crit_count + high_count + med_count + low_count

        return ReturnAnomalyResult(
            anomalies=detected_anomalies,
            total_anomalies=total_anom,
            critical_count=crit_count,
            high_count=high_count,
            medium_count=med_count,
            low_count=low_count,
            as_of_date=as_of_str,
            config=cfg,
        )

    def _detect_dimensional_anomalies(
        self,
        dim_name: str,
        group_keys: List[str],
        ret_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        curr_p: str,
        base_p: List[str],
        as_of_str: Optional[str],
        cfg: ReturnAnomalyConfig,
        add_fn: Any,
    ) -> None:
        """Aggregate and evaluate dimensional entities (e.g. SKU, Channel, Warehouse)."""
        # Ensure keys exist in ret_df
        valid_ret_keys = [k for k in group_keys if k in ret_df.columns]
        if len(valid_ret_keys) != len(group_keys):
            return

        # Period sales aggregation
        if not sales_df.empty and all(k in sales_df.columns for k in group_keys):
            sales_agg = sales_df.groupby(["period"] + group_keys)["quantity"].sum().reset_index()
            sales_agg.rename(columns={"quantity": "sold_units"}, inplace=True)
        else:
            sales_agg = pd.DataFrame(columns=["period"] + group_keys + ["sold_units"])

        # Period returns aggregation
        ret_agg = ret_df.groupby(["period"] + group_keys).agg(
            returned_units=("quantity", "sum"),
            return_count=("quantity", "count"),
        ).reset_index()

        # Merge sales and returns
        if not sales_agg.empty:
            merged = pd.merge(sales_agg, ret_agg, on=["period"] + group_keys, how="outer")
        else:
            merged = ret_agg.copy()
            merged["sold_units"] = 0

        merged["sold_units"] = merged["sold_units"].fillna(0).astype(int)
        merged["returned_units"] = merged["returned_units"].fillna(0).astype(int)
        merged["return_count"] = merged["return_count"].fillna(0).astype(int)
        merged["return_rate"] = np.where(
            merged["sold_units"] > 0,
            merged["returned_units"] / merged["sold_units"],
            0.0,
        )

        # Form unique entity keys
        if len(group_keys) == 1:
            merged["entity_id"] = merged[group_keys[0]].astype(str)
        else:
            merged["entity_id"] = merged[group_keys[0]].astype(str) + ":" + merged[group_keys[1]].astype(str)

        # Evaluate each unique entity present in current period
        curr_subset = merged[merged["period"] == curr_p]
        for _, curr_row in curr_subset.iterrows():
            ent_id = curr_row["entity_id"]
            curr_rate = float(curr_row["return_rate"])
            curr_sold = int(curr_row["sold_units"])
            curr_ret = int(curr_row["returned_units"])
            curr_count = int(curr_row["return_count"])

            # Historical baseline values
            base_subset = merged[(merged["entity_id"] == ent_id) & (merged["period"].isin(base_p))]
            base_rates = base_subset["return_rate"].tolist()
            base_volumes = base_subset["returned_units"].tolist()

            # 1. Rate anomaly evaluation
            rate_anom = detect_series_anomaly(
                current_value=curr_rate,
                baseline_values=base_rates,
                dimension=dim_name,
                entity_id=ent_id,
                metric_name="return_rate",
                period=curr_p,
                as_of_date=as_of_str,
                config=cfg,
                sample_size=curr_sold,
                sample_return_count=curr_count,
            )
            add_fn(rate_anom)

            # 2. Volume anomaly evaluation (if rate is not an extreme spike, or sales surged)
            if rate_anom.anomaly_type == AnomalyType.NO_ANOMALY or rate_anom.severity in (
                AnomalySeverity.NONE,
                AnomalySeverity.LOW,
            ):
                if len(base_volumes) >= cfg.min_history_periods:
                    mean_base_vol = float(np.mean(base_volumes))
                    if curr_ret >= cfg.volume_spike_multiplier * mean_base_vol and curr_ret > mean_base_vol + 5:
                        vol_anom = detect_series_anomaly(
                            current_value=float(curr_ret),
                            baseline_values=[float(v) for v in base_volumes],
                            dimension=dim_name,
                            entity_id=ent_id,
                            metric_name="returned_units",
                            period=curr_p,
                            as_of_date=as_of_str,
                            config=cfg,
                            sample_size=curr_sold,
                            sample_return_count=curr_count,
                        )
                        if vol_anom.anomaly_type in (
                            AnomalyType.RETURN_VOLUME_SPIKE,
                            AnomalyType.RETURN_VOLUME_DROP,
                        ):
                            add_fn(vol_anom)

    def _detect_reason_shifts(
        self,
        ret_df: pd.DataFrame,
        curr_p: str,
        base_p: List[str],
        as_of_str: Optional[str],
        cfg: ReturnAnomalyConfig,
        add_fn: Any,
    ) -> None:
        """Detect disproportionate shifts in return reason distributions."""
        reason_agg = ret_df.groupby(["period", "return_reason"])["quantity"].sum().reset_index()
        period_totals = ret_df.groupby("period")["quantity"].sum().to_dict()

        curr_tot = period_totals.get(curr_p, 0)
        base_tot = sum(period_totals.get(p, 0) for p in base_p)

        curr_reasons = reason_agg[reason_agg["period"] == curr_p]
        all_reasons = set(curr_reasons["return_reason"].unique())

        for reason in all_reasons:
            curr_units = int(curr_reasons[curr_reasons["return_reason"] == reason]["quantity"].sum())
            base_units = int(
                reason_agg[(reason_agg["period"].isin(base_p)) & (reason_agg["return_reason"] == reason)][
                    "quantity"
                ].sum()
            )

            anom = detect_reason_shift_anomaly(
                reason=reason,
                current_units=curr_units,
                current_total_units=curr_tot,
                baseline_units=base_units,
                baseline_total_units=base_tot,
                period=curr_p,
                as_of_date=as_of_str,
                config=cfg,
            )
            add_fn(anom)

    def _detect_portfolio_time_series_anomaly(
        self,
        ret_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        curr_p: str,
        base_p: List[str],
        as_of_str: Optional[str],
        cfg: ReturnAnomalyConfig,
        add_fn: Any,
    ) -> None:
        """Evaluate overall portfolio return rate in current period vs baseline periods."""
        # Total portfolio returns per period
        ret_port = ret_df.groupby("period")["quantity"].sum().to_dict()
        ret_counts = ret_df.groupby("period")["quantity"].count().to_dict()

        # Total portfolio sales per period
        if not sales_df.empty and "quantity" in sales_df.columns:
            sales_port = sales_df.groupby("period")["quantity"].sum().to_dict()
        else:
            sales_port = {}

        curr_ret = int(ret_port.get(curr_p, 0))
        curr_sold = int(sales_port.get(curr_p, 0))
        curr_count = int(ret_counts.get(curr_p, 0))
        curr_rate = curr_ret / curr_sold if curr_sold > 0 else 0.0

        base_rates = []
        for p in base_p:
            b_ret = ret_port.get(p, 0)
            b_sold = sales_port.get(p, 0)
            b_rate = b_ret / b_sold if b_sold > 0 else 0.0
            base_rates.append(b_rate)

        port_anom = detect_series_anomaly(
            current_value=curr_rate,
            baseline_values=base_rates,
            dimension="PORTFOLIO",
            entity_id="TOTAL_PORTFOLIO",
            metric_name="return_rate",
            period=curr_p,
            as_of_date=as_of_str,
            config=cfg,
            sample_size=curr_sold,
            sample_return_count=curr_count,
        )
        add_fn(port_anom)
