"""Unit and Integration Tests for Return Anomaly Detection (Phase 5B).

Covers all 30+ statistical and edge cases:
- High baseline return rate vs unexpected shift
- Sudden spikes and drops in return rate
- Volume spike with vs without rate spike
- Zero-variance baseline handling (BASELINE_SHIFT, NO_ANOMALY)
- Sample size guards (min_sold_units, min_return_count, min_history_periods)
- Return reason distribution shifts (increase, decrease, below threshold)
- Multi-dimensional shifts: SKU, Channel, Warehouse, SKU×Channel, SKU×Warehouse
- Severity tiers: CRITICAL, HIGH, MEDIUM, LOW, NONE
- Deviation directions: INCREASE, DECREASE, NONE
- Modified Z-Score using Median Absolute Deviation (MAD)
- Deterministic SHA-256 ID repeatability and uniqueness
- Deterministic rationale generation
- Anti-leakage chronological date filtering
- Edge cases: empty datasets, zero sales, dataframe export, filtering helpers
"""

from __future__ import annotations

import pandas as pd
import pytest

from commerce_ai.returns.schemas import (
    AnomalyDirection,
    AnomalySeverity,
    AnomalyType,
    ReturnAnomaly,
    ReturnAnomalyConfig,
    ReturnAnomalyResult,
)
from commerce_ai.returns.anomaly import (
    ReturnAnomalyService,
    compute_mad_score,
    compute_z_score,
    detect_reason_shift_anomaly,
    detect_series_anomaly,
    determine_direction,
    generate_anomaly_rationale,
    generate_deterministic_anomaly_id,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def default_config() -> ReturnAnomalyConfig:
    return ReturnAnomalyConfig(
        z_score_threshold=2.0,
        critical_z_score=3.0,
        high_z_score=2.5,
        medium_z_score=2.0,
        low_z_score=1.5,
        min_sold_units=30,
        min_return_count=5,
        min_history_periods=3,
        reason_shift_threshold=0.15,
        volume_spike_multiplier=2.0,
    )


# =====================================================================
# Test Cases 1-5: Statistical Shifts vs High Baseline Normalcy
# =====================================================================


def test_high_baseline_normal_behavior(default_config):
    """High baseline return rate (e.g. 25% for apparel shoes) with normal current period is NOT an anomaly."""
    # Historical baseline rates: 25%, 26%, 24%, 25% (mean=0.25, std=0.00816)
    baseline = [0.25, 0.26, 0.24, 0.25]
    current = 0.25  # Current period is also 25%

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_SHOES_01",
        config=default_config,
        sample_size=200,
        sample_return_count=50,
    )

    assert anomaly.anomaly_type == AnomalyType.NO_ANOMALY
    assert anomaly.severity == AnomalySeverity.NONE
    assert anomaly.direction == AnomalyDirection.NONE
    assert anomaly.z_score is not None
    assert abs(anomaly.z_score) < 0.1


def test_low_baseline_sudden_spike(default_config):
    """Low baseline return rate (e.g. 2% for electronics) with sudden surge to 8% triggers CRITICAL spike."""
    # Baseline: 2%, 2.1%, 1.9%, 2.0% (mean ~0.02, std ~0.0008)
    baseline = [0.020, 0.021, 0.019, 0.020]
    current = 0.080  # Jumps to 8% (4x baseline)

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_ELEC_01",
        config=default_config,
        sample_size=300,
        sample_return_count=24,
    )

    assert anomaly.anomaly_type == AnomalyType.RETURN_RATE_SPIKE
    assert anomaly.severity == AnomalySeverity.CRITICAL
    assert anomaly.direction == AnomalyDirection.INCREASE
    assert anomaly.z_score is not None and anomaly.z_score >= 3.0


def test_return_rate_drop(default_config):
    """Sudden drop in return rate triggers RETURN_RATE_DROP with DECREASE direction."""
    baseline = [0.20, 0.21, 0.19, 0.20]  # mean ~0.20, std ~0.008
    current = 0.05  # Drops significantly to 5%

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_APPAR_02",
        config=default_config,
        sample_size=200,
        sample_return_count=10,
    )

    assert anomaly.anomaly_type == AnomalyType.RETURN_RATE_DROP
    assert anomaly.severity == AnomalySeverity.CRITICAL
    assert anomaly.direction == AnomalyDirection.DECREASE
    assert anomaly.z_score is not None and anomaly.z_score <= -3.0


def test_volume_spike_with_rate_spike(default_config):
    """When both rate and units surge, rate spike is accurately captured."""
    baseline = [0.03, 0.032, 0.028, 0.03]
    current = 0.12

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_001",
        metric_name="return_rate",
        config=default_config,
        sample_size=500,
        sample_return_count=60,
    )

    assert anomaly.anomaly_type == AnomalyType.RETURN_RATE_SPIKE
    assert anomaly.severity == AnomalySeverity.CRITICAL


def test_volume_spike_without_rate_spike(default_config):
    """Sales and returns surge 5x proportionally: volume spikes while rate remains normal."""
    # Volume baseline: 10, 11, 9, 10 units (mean=10, std=0.816)
    baseline_vols = [10.0, 11.0, 9.0, 10.0]
    current_vol = 50.0  # 5x volume spike

    anomaly = detect_series_anomaly(
        current_value=current_vol,
        baseline_values=baseline_vols,
        dimension="SKU",
        entity_id="SKU_VIRAL_01",
        metric_name="returned_units",
        config=default_config,
        sample_size=1000,
        sample_return_count=50,
    )

    assert anomaly.anomaly_type == AnomalyType.RETURN_VOLUME_SPIKE
    assert anomaly.severity == AnomalySeverity.CRITICAL
    assert anomaly.direction == AnomalyDirection.INCREASE


# =====================================================================
# Test Cases 6-8: Zero-Variance Baseline Handling
# =====================================================================


def test_zero_variance_baseline_identical_current(default_config):
    """Zero variance baseline with identical current value yields NO_ANOMALY (z=0.0, severity=NONE)."""
    baseline = [0.10, 0.10, 0.10, 0.10]
    current = 0.10

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_CONST_01",
        config=default_config,
        sample_size=100,
        sample_return_count=10,
    )

    assert anomaly.anomaly_type == AnomalyType.NO_ANOMALY
    assert anomaly.severity == AnomalySeverity.NONE
    assert anomaly.direction == AnomalyDirection.NONE
    assert anomaly.z_score == 0.0


def test_zero_variance_baseline_upward_shift(default_config):
    """Zero variance baseline with different current value flags BASELINE_SHIFT (z=None, no division by zero)."""
    baseline = [0.10, 0.10, 0.10, 0.10]
    current = 0.25  # Shifts upward by 15 pp

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_CONST_02",
        config=default_config,
        sample_size=100,
        sample_return_count=25,
    )

    assert anomaly.anomaly_type == AnomalyType.BASELINE_SHIFT
    assert anomaly.direction == AnomalyDirection.INCREASE
    assert anomaly.severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL)
    assert anomaly.z_score is None  # Cannot divide by zero


def test_zero_variance_baseline_downward_shift(default_config):
    """Zero variance baseline shifting downward flags BASELINE_SHIFT with DECREASE direction."""
    baseline = [0.10, 0.10, 0.10, 0.10]
    current = 0.03  # Shifts downward by 7 pp

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_CONST_03",
        config=default_config,
        sample_size=300,
        sample_return_count=9,
    )

    assert anomaly.anomaly_type == AnomalyType.BASELINE_SHIFT
    assert anomaly.direction == AnomalyDirection.DECREASE
    assert anomaly.severity in (AnomalySeverity.LOW, AnomalySeverity.MEDIUM)
    assert anomaly.z_score is None


# =====================================================================
# Test Cases 9-11: Sample Size Guards
# =====================================================================


def test_insufficient_sold_units(default_config):
    """Sold units below min_sold_units flags INSUFFICIENT_SAMPLE and does not trigger false spike."""
    baseline = [0.05, 0.05, 0.05]
    current = 0.40  # 40% return rate on only 10 sold units

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_LOW_VOL",
        config=default_config,
        sample_size=10,  # Below min_sold_units=30
        sample_return_count=4,
    )

    assert anomaly.anomaly_type == AnomalyType.INSUFFICIENT_SAMPLE
    assert anomaly.severity == AnomalySeverity.NONE
    assert anomaly.is_sufficient_sample is False
    assert anomaly.z_score is None


def test_insufficient_return_count(default_config):
    """Return count below min_return_count flags INSUFFICIENT_SAMPLE."""
    baseline = [0.02, 0.02, 0.02]
    current = 0.05

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_FEW_RETURNS",
        config=default_config,
        sample_size=60,
        sample_return_count=3,  # Below min_return_count=5
    )

    assert anomaly.anomaly_type == AnomalyType.INSUFFICIENT_SAMPLE
    assert anomaly.severity == AnomalySeverity.NONE
    assert anomaly.is_sufficient_sample is False


def test_insufficient_history_periods(default_config):
    """Having fewer historical periods than min_history_periods flags INSUFFICIENT_SAMPLE."""
    baseline = [0.05]  # Only 1 period, minimum is 3
    current = 0.15

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_NEW_LAUNCH",
        config=default_config,
        sample_size=100,
        sample_return_count=15,
    )

    assert anomaly.anomaly_type == AnomalyType.INSUFFICIENT_SAMPLE
    assert anomaly.severity == AnomalySeverity.NONE
    assert anomaly.is_sufficient_sample is False


# =====================================================================
# Test Cases 12-15: Return Reason Shifts
# =====================================================================


def test_reason_shift_defective_surge(default_config):
    """Defective return reason share surging by +30 pp triggers RETURN_REASON_SHIFT (CRITICAL)."""
    # Baseline: 100 defective units out of 1000 total returns = 10%
    # Current: 200 defective units out of 500 total returns = 40% (delta = +30 pp)
    anomaly = detect_reason_shift_anomaly(
        reason="DEFECTIVE",
        current_units=200,
        current_total_units=500,
        baseline_units=100,
        baseline_total_units=1000,
        period="2026-03",
        config=default_config,
    )

    assert anomaly.anomaly_type == AnomalyType.RETURN_REASON_SHIFT
    assert anomaly.severity == AnomalySeverity.CRITICAL
    assert anomaly.direction == AnomalyDirection.INCREASE
    assert anomaly.current_value == pytest.approx(0.40, rel=1e-3)
    assert anomaly.baseline_value == pytest.approx(0.10, rel=1e-3)


def test_reason_shift_decrease(default_config):
    """Return reason share dropping significantly flags RETURN_REASON_SHIFT with DECREASE direction."""
    # Baseline: 500 wrong size units out of 1000 = 50%
    # Current: 100 wrong size units out of 500 = 20% (delta = -30 pp)
    anomaly = detect_reason_shift_anomaly(
        reason="WRONG_SIZE",
        current_units=100,
        current_total_units=500,
        baseline_units=500,
        baseline_total_units=1000,
        period="2026-03",
        config=default_config,
    )

    assert anomaly.anomaly_type == AnomalyType.RETURN_REASON_SHIFT
    assert anomaly.severity == AnomalySeverity.CRITICAL
    assert anomaly.direction == AnomalyDirection.DECREASE


def test_reason_shift_below_threshold(default_config):
    """Shift below reason_shift_threshold (e.g. +2 pp < 15 pp) yields NO_ANOMALY."""
    # Baseline: 200 out of 1000 = 20%
    # Current: 110 out of 500 = 22% (delta = +2 pp)
    anomaly = detect_reason_shift_anomaly(
        reason="COLOR_MISMATCH",
        current_units=110,
        current_total_units=500,
        baseline_units=200,
        baseline_total_units=1000,
        config=default_config,
    )

    assert anomaly.anomaly_type == AnomalyType.NO_ANOMALY
    assert anomaly.severity == AnomalySeverity.NONE
    assert anomaly.direction == AnomalyDirection.NONE


def test_reason_shift_insufficient_sample(default_config):
    """Total returns below min_return_count flags INSUFFICIENT_SAMPLE."""
    anomaly = detect_reason_shift_anomaly(
        reason="DEFECTIVE",
        current_units=2,
        current_total_units=3,  # Below min_return_count=5
        baseline_units=50,
        baseline_total_units=500,
        config=default_config,
    )

    assert anomaly.anomaly_type == AnomalyType.INSUFFICIENT_SAMPLE
    assert anomaly.severity == AnomalySeverity.NONE


# =====================================================================
# Test Cases 16-19: Multi-Dimensional Shifts
# =====================================================================


def test_channel_return_shift(default_config):
    """Channel return rate jump flags CHANNEL_RETURN_SHIFT."""
    baseline = [0.04, 0.042, 0.038, 0.04]  # mean ~0.04, std ~0.0016
    current = 0.12  # Spikes to 12%

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="CHANNEL",
        entity_id="CH_APP",
        config=default_config,
        sample_size=1000,
        sample_return_count=120,
    )

    assert anomaly.anomaly_type == AnomalyType.CHANNEL_RETURN_SHIFT
    assert anomaly.dimension == "CHANNEL"
    assert anomaly.severity == AnomalySeverity.CRITICAL


def test_warehouse_return_shift(default_config):
    """Warehouse return rate surge flags WAREHOUSE_RETURN_SHIFT."""
    baseline = [0.03, 0.032, 0.028, 0.03]
    current = 0.10

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="WAREHOUSE",
        entity_id="WH_WEST",
        config=default_config,
        sample_size=800,
        sample_return_count=80,
    )

    assert anomaly.anomaly_type == AnomalyType.WAREHOUSE_RETURN_SHIFT
    assert anomaly.dimension == "WAREHOUSE"
    assert anomaly.severity == AnomalySeverity.CRITICAL


def test_sku_channel_interaction_shift(default_config):
    """SKU × Channel interaction anomaly flags SKU_CHANNEL_RETURN_SHIFT."""
    baseline = [0.05, 0.052, 0.048, 0.05]
    current = 0.15

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU_CHANNEL",
        entity_id="SKU_001:CH_WEB",
        config=default_config,
        sample_size=200,
        sample_return_count=30,
    )

    assert anomaly.anomaly_type == AnomalyType.SKU_CHANNEL_RETURN_SHIFT
    assert anomaly.dimension == "SKU_CHANNEL"
    assert anomaly.severity == AnomalySeverity.CRITICAL


def test_sku_warehouse_interaction_shift(default_config):
    """SKU × Warehouse interaction anomaly flags SKU_WAREHOUSE_RETURN_SHIFT."""
    baseline = [0.04, 0.041, 0.039, 0.04]
    current = 0.12

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU_WAREHOUSE",
        entity_id="SKU_001:WH_EAST",
        config=default_config,
        sample_size=200,
        sample_return_count=24,
    )

    assert anomaly.anomaly_type == AnomalyType.SKU_WAREHOUSE_RETURN_SHIFT
    assert anomaly.dimension == "SKU_WAREHOUSE"
    assert anomaly.severity == AnomalySeverity.CRITICAL


# =====================================================================
# Test Cases 20-24: Severity Tiers Mapping
# =====================================================================


def test_severity_critical(default_config):
    """|z| >= 3.0 maps to CRITICAL."""
    z, sev, a_type, _ = compute_z_score(current=0.35, baseline_mean=0.20, baseline_std=0.04, config=default_config)
    assert sev == AnomalySeverity.CRITICAL
    assert z >= 3.0


def test_severity_high(default_config):
    """2.5 <= |z| < 3.0 maps to HIGH."""
    # z = (0.27 - 0.20) / 0.026 = 2.69
    z, sev, a_type, _ = compute_z_score(current=0.27, baseline_mean=0.20, baseline_std=0.026, config=default_config)
    assert sev == AnomalySeverity.HIGH
    assert 2.5 <= z < 3.0


def test_severity_medium(default_config):
    """2.0 <= |z| < 2.5 maps to MEDIUM."""
    # z = (0.245 - 0.20) / 0.02 = 2.25
    z, sev, a_type, _ = compute_z_score(current=0.245, baseline_mean=0.20, baseline_std=0.02, config=default_config)
    assert sev == AnomalySeverity.MEDIUM
    assert 2.0 <= z < 2.5


def test_severity_low(default_config):
    """1.5 <= |z| < 2.0 maps to LOW."""
    # z = (0.234 - 0.20) / 0.02 = 1.70
    z, sev, a_type, _ = compute_z_score(current=0.234, baseline_mean=0.20, baseline_std=0.02, config=default_config)
    assert sev == AnomalySeverity.LOW
    assert 1.5 <= z < 2.0


def test_severity_none(default_config):
    """|z| < 1.5 maps to NONE."""
    # z = (0.21 - 0.20) / 0.02 = 0.50
    z, sev, a_type, _ = compute_z_score(current=0.21, baseline_mean=0.20, baseline_std=0.02, config=default_config)
    assert sev == AnomalySeverity.NONE
    assert a_type == AnomalyType.NO_ANOMALY


# =====================================================================
# Test Cases 25-27: Deviation Directions
# =====================================================================


def test_direction_increase():
    assert determine_direction(current=0.25, baseline=0.20) == AnomalyDirection.INCREASE


def test_direction_decrease():
    assert determine_direction(current=0.15, baseline=0.20) == AnomalyDirection.DECREASE


def test_direction_none():
    assert determine_direction(current=0.20, baseline=0.20) == AnomalyDirection.NONE


# =====================================================================
# Test Cases 28-29: Robust MAD (Median Absolute Deviation)
# =====================================================================


def test_modified_z_score_with_mad():
    """Robust MAD detection correctly evaluates modified z-score in presence of outliers."""
    cfg = ReturnAnomalyConfig(use_mad=True, z_score_threshold=2.0)
    baseline = [0.05, 0.05, 0.06, 0.04, 0.25]  # Outlier 0.25 in baseline history
    current = 0.15

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_ROBUST_01",
        config=cfg,
        sample_size=100,
        sample_return_count=15,
    )

    assert anomaly.mad_score is not None
    assert anomaly.anomaly_type == AnomalyType.RETURN_RATE_SPIKE
    assert anomaly.severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL)


def test_mad_with_zero_mad():
    """When MAD is 0 (constant baseline), handles safely via BASELINE_SHIFT."""
    cfg = ReturnAnomalyConfig(use_mad=True)
    baseline = [0.05, 0.05, 0.05, 0.05]
    current = 0.18

    anomaly = detect_series_anomaly(
        current_value=current,
        baseline_values=baseline,
        dimension="SKU",
        entity_id="SKU_ROBUST_02",
        config=cfg,
        sample_size=100,
        sample_return_count=18,
    )

    assert anomaly.anomaly_type == AnomalyType.BASELINE_SHIFT
    assert anomaly.severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL)
    assert anomaly.mad_score is None


# =====================================================================
# Test Cases 30-31: Deterministic ID & Rationale Repeatability
# =====================================================================


def test_deterministic_anomaly_id():
    """ID generator produces repeatable SHA-256 digests and varies across inputs."""
    id1 = generate_deterministic_anomaly_id("SKU", "SKU_001", "2026-03", AnomalyType.RETURN_RATE_SPIKE, "2026-03-31")
    id2 = generate_deterministic_anomaly_id("SKU", "SKU_001", "2026-03", AnomalyType.RETURN_RATE_SPIKE, "2026-03-31")
    id3 = generate_deterministic_anomaly_id("SKU", "SKU_002", "2026-03", AnomalyType.RETURN_RATE_SPIKE, "2026-03-31")

    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("ANOM-")
    assert len(id1) == 21  # "ANOM-" (5 chars) + 16 hex chars


def test_deterministic_rationale():
    """Rationale strings contain factual metrics and deterministic phrasing without hallucinations."""
    rationale = generate_anomaly_rationale(
        dimension="SKU",
        entity_id="SKU_123",
        metric_name="return_rate",
        current_value=0.185,
        baseline_value=0.052,
        anomaly_type=AnomalyType.RETURN_RATE_SPIKE,
        severity=AnomalySeverity.CRITICAL,
        direction=AnomalyDirection.INCREASE,
        z_score=6.33,
        period="2026-03",
    )

    assert "SKU_123" in rationale
    assert "18.50%" in rationale
    assert "5.20%" in rationale
    assert "+6.33" in rationale
    assert "CRITICAL" in rationale


# =====================================================================
# Test Cases 32-35: End-to-End Pipeline, Anti-Leakage & Edge Cases
# =====================================================================


def test_anti_leakage_as_of_date():
    """Service strictly excludes returns and sales transactions occurring past as_of_date."""
    # Returns dataset with transactions in Jan, Feb, Mar, and leaked Apr
    returns_data = [
        {"return_id": "R1", "order_id": "O1", "sku_id": "SKU_01", "return_date": "2026-01-15", "quantity": 5, "return_reason": "DEFECTIVE", "channel_id": "CH1", "warehouse_id": "WH1"},
        {"return_id": "R2", "order_id": "O2", "sku_id": "SKU_01", "return_date": "2026-02-15", "quantity": 5, "return_reason": "DEFECTIVE", "channel_id": "CH1", "warehouse_id": "WH1"},
        {"return_id": "R3", "order_id": "O3", "sku_id": "SKU_01", "return_date": "2026-03-15", "quantity": 5, "return_reason": "DEFECTIVE", "channel_id": "CH1", "warehouse_id": "WH1"},
        {"return_id": "R4", "order_id": "O4", "sku_id": "SKU_01", "return_date": "2026-04-15", "quantity": 100, "return_reason": "DEFECTIVE", "channel_id": "CH1", "warehouse_id": "WH1"},  # Future
    ]
    sales_data = [
        {"date": "2026-01-10", "sku_id": "SKU_01", "quantity": 100, "channel_id": "CH1", "warehouse_id": "WH1"},
        {"date": "2026-02-10", "sku_id": "SKU_01", "quantity": 100, "channel_id": "CH1", "warehouse_id": "WH1"},
        {"date": "2026-03-10", "sku_id": "SKU_01", "quantity": 100, "channel_id": "CH1", "warehouse_id": "WH1"},
        {"date": "2026-04-10", "sku_id": "SKU_01", "quantity": 100, "channel_id": "CH1", "warehouse_id": "WH1"},  # Future
    ]

    service = ReturnAnomalyService()
    result = service.detect(
        returns=pd.DataFrame(returns_data),
        sales=pd.DataFrame(sales_data),
        as_of_date="2026-03-31",
    )

    # Apr transaction was cut off. Evaluated period should be 2026-03, not 2026-04
    for a in result.anomalies:
        assert a.period != "2026-04"


def test_empty_datasets_handled_gracefully():
    """Empty returns or sales DataFrames return clean result bundle with 0 anomalies."""
    service = ReturnAnomalyService()
    result = service.detect(
        returns=pd.DataFrame(),
        sales=pd.DataFrame(),
        as_of_date="2026-03-31",
    )

    assert isinstance(result, ReturnAnomalyResult)
    assert result.total_anomalies == 0
    assert len(result.anomalies) == 0
    df = result.to_dataframe()
    assert df.empty


def test_to_dataframe_export(default_config):
    """result.to_dataframe() contains all required columns and valid records."""
    anomaly = detect_series_anomaly(
        current_value=0.15,
        baseline_values=[0.02, 0.021, 0.019],
        dimension="SKU",
        entity_id="SKU_EXPORT_01",
        config=default_config,
    )
    res = ReturnAnomalyResult(
        anomalies=[anomaly],
        total_anomalies=1,
        critical_count=1,
        as_of_date="2026-03-31",
        config=default_config,
    )

    df = res.to_dataframe()
    assert not df.empty
    assert len(df) == 1
    assert "anomaly_id" in df.columns
    assert "dimension" in df.columns
    assert "severity" in df.columns
    assert "z_score" in df.columns
    assert df.iloc[0]["entity_id"] == "SKU_EXPORT_01"


def test_result_filtering_helpers(default_config):
    """Filter methods correctly isolate anomalies by severity, dimension, and type."""
    a1 = detect_series_anomaly(
        current_value=0.20,
        baseline_values=[0.02, 0.021, 0.019],
        dimension="SKU",
        entity_id="SKU_CRIT",
        config=default_config,
        sample_size=500,
        sample_return_count=100,
    )
    a2 = detect_series_anomaly(
        current_value=0.0245,
        baseline_values=[0.02, 0.022, 0.018],
        dimension="CHANNEL",
        entity_id="CH_MED",
        config=default_config,
        sample_size=500,
        sample_return_count=12,
    )
    res = ReturnAnomalyResult(
        anomalies=[a1, a2],
        total_anomalies=2,
        critical_count=1,
        high_count=0,
        medium_count=1,
        low_count=0,
        config=default_config,
    )

    crit_list = res.filter_by_severity(AnomalySeverity.CRITICAL)
    assert len(crit_list) == 1
    assert crit_list[0].entity_id == "SKU_CRIT"

    chan_list = res.filter_by_dimension("CHANNEL")
    assert len(chan_list) == 1
    assert chan_list[0].entity_id == "CH_MED"

    type_list = res.filter_by_type(AnomalyType.RETURN_RATE_SPIKE)
    assert len(type_list) == 1


def test_full_pipeline_multi_dimensional_detection():
    """Integration test verifying end-to-end multi-dimensional detection across periods."""
    # Construct 4 monthly periods: Jan, Feb, Mar (baseline), Apr (current spike)
    # SKU_A: baseline return rate 5%, surges to 25% in Apr
    # SKU_B: baseline return rate 10%, stays 10% in Apr (normal)
    # Reason: DEFECTIVE share surges in Apr
    sales_rows = []
    return_rows = []

    periods = [("2026-01-15", 500, 25), ("2026-02-15", 500, 25), ("2026-03-15", 500, 25), ("2026-04-15", 500, 125)]
    for ret_id, (dt, sold, ret) in enumerate(periods):
        p_str = dt[:7]
        sales_rows.append({"date": dt, "sku_id": "SKU_A", "quantity": sold, "channel_id": "CH_WEB", "warehouse_id": "WH_01"})
        sales_rows.append({"date": dt, "sku_id": "SKU_B", "quantity": sold, "channel_id": "CH_WEB", "warehouse_id": "WH_01"})

        # SKU_A returns
        reason = "DEFECTIVE" if p_str == "2026-04" else "WRONG_SIZE"
        for i in range(ret):
            return_rows.append({
                "return_id": f"R_{ret_id}_{i}",
                "order_id": f"O_{ret_id}_{i}",
                "sku_id": "SKU_A",
                "return_date": dt,
                "quantity": 1,
                "return_reason": reason,
                "channel_id": "CH_WEB",
                "warehouse_id": "WH_01",
            })
        # SKU_B returns (steady 50 units = 10%)
        for i in range(50):
            return_rows.append({
                "return_id": f"RB_{ret_id}_{i}",
                "order_id": f"OB_{ret_id}_{i}",
                "sku_id": "SKU_B",
                "return_date": dt,
                "quantity": 1,
                "return_reason": "WRONG_SIZE",
                "channel_id": "CH_WEB",
                "warehouse_id": "WH_01",
            })

    sales_df = pd.DataFrame(sales_rows)
    returns_df = pd.DataFrame(return_rows)

    service = ReturnAnomalyService(config=ReturnAnomalyConfig(min_sold_units=50, min_return_count=10, min_history_periods=3))
    result = service.detect(
        returns=returns_df,
        sales=sales_df,
        as_of_date="2026-04-30",
    )

    assert result.total_anomalies > 0
    sku_anomalies = result.filter_by_dimension("SKU")
    sku_a_anoms = [a for a in sku_anomalies if a.entity_id == "SKU_A"]
    sku_b_anoms = [a for a in sku_anomalies if a.entity_id == "SKU_B"]

    # SKU_A must be flagged as a spike
    assert len(sku_a_anoms) >= 1
    assert sku_a_anoms[0].anomaly_type in (AnomalyType.RETURN_RATE_SPIKE, AnomalyType.BASELINE_SHIFT)
    assert sku_a_anoms[0].severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL)

    # SKU_B has steady 10% return rate -> should NOT be flagged as an anomaly
    assert len(sku_b_anoms) == 0

    # Reason shift should be detected for DEFECTIVE
    reason_anomalies = result.filter_by_dimension("REASON")
    defective_anoms = [a for a in reason_anomalies if a.entity_id == "DEFECTIVE"]
    assert len(defective_anoms) >= 1
    assert defective_anoms[0].anomaly_type == AnomalyType.RETURN_REASON_SHIFT
    assert defective_anoms[0].direction == AnomalyDirection.INCREASE
