"""Unit tests for Return Risk Layer & Policy Engine (Phase 5C-2B)."""

import pytest
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
    ReturnProbabilityCalibrator,
)
from commerce_ai.returns.risk import (
    ReturnRiskClassifier,
    evaluate_risk_policy,
    evaluate_segment_calibration,
    risk_evaluations_to_dataframe,
)


class TestRiskPolicyConfig:
    """Test suite for RiskPolicyConfig schema and boundary validation."""

    def test_default_policy_boundaries(self):
        policy = RiskPolicyConfig()
        assert policy.policy_version == "1.0.0-prototype"
        assert len(policy.boundaries) == 5

        # Check default band sequence
        expected_bands = [
            ReturnRiskBand.VERY_LOW,
            ReturnRiskBand.LOW,
            ReturnRiskBand.MEDIUM,
            ReturnRiskBand.HIGH,
            ReturnRiskBand.VERY_HIGH,
        ]
        assert [b.band for b in policy.boundaries] == expected_bands

        # Starts at 0.0 and ends at 1.0
        assert policy.boundaries[0].lower_bound == 0.0
        assert policy.boundaries[-1].upper_bound == 1.0

    def test_custom_valid_policy(self):
        custom_boundaries = [
            RiskBandBoundary(band=ReturnRiskBand.LOW, lower_bound=0.0, upper_bound=0.30),
            RiskBandBoundary(band=ReturnRiskBand.MEDIUM, lower_bound=0.30, upper_bound=0.70),
            RiskBandBoundary(band=ReturnRiskBand.HIGH, lower_bound=0.70, upper_bound=1.0),
        ]
        policy = RiskPolicyConfig(
            policy_version="custom-2026-v2",
            boundaries=custom_boundaries,
            description="Custom 3-band operational policy",
        )
        assert policy.policy_version == "custom-2026-v2"
        assert len(policy.boundaries) == 3

    def test_policy_with_gap_rejected(self):
        gap_boundaries = [
            RiskBandBoundary(band=ReturnRiskBand.VERY_LOW, lower_bound=0.0, upper_bound=0.20),
            # Gap between 0.20 and 0.30
            RiskBandBoundary(band=ReturnRiskBand.HIGH, lower_bound=0.30, upper_bound=1.0),
        ]
        with pytest.raises(ValueError, match="Gap in risk boundaries"):
            RiskPolicyConfig(boundaries=gap_boundaries)

    def test_policy_with_overlap_rejected(self):
        overlap_boundaries = [
            RiskBandBoundary(band=ReturnRiskBand.VERY_LOW, lower_bound=0.0, upper_bound=0.35),
            # Overlap: 0.30 is before 0.35
            RiskBandBoundary(band=ReturnRiskBand.HIGH, lower_bound=0.30, upper_bound=1.0),
        ]
        with pytest.raises(ValueError, match="Overlapping risk boundaries"):
            RiskPolicyConfig(boundaries=overlap_boundaries)

    def test_policy_not_starting_at_zero_rejected(self):
        invalid_start = [
            RiskBandBoundary(band=ReturnRiskBand.LOW, lower_bound=0.05, upper_bound=1.0),
        ]
        with pytest.raises(ValueError, match="must start at 0.0"):
            RiskPolicyConfig(boundaries=invalid_start)

    def test_policy_not_ending_at_one_rejected(self):
        invalid_end = [
            RiskBandBoundary(band=ReturnRiskBand.LOW, lower_bound=0.0, upper_bound=0.90),
        ]
        with pytest.raises(ValueError, match="must end at 1.0"):
            RiskPolicyConfig(boundaries=invalid_end)


class TestReturnRiskClassifier:
    """Test suite for probability mapping and interpretation generation."""

    def test_boundary_classifications(self):
        classifier = ReturnRiskClassifier()

        # Exact boundary values
        # Default boundaries:
        # VERY_LOW: [0.00, 0.10)
        # LOW:      [0.10, 0.25)
        # MEDIUM:   [0.25, 0.50)
        # HIGH:     [0.50, 0.75)
        # VERY_HIGH:[0.75, 1.00]

        assert classifier.classify_probability(0.0) == ReturnRiskBand.VERY_LOW
        assert classifier.classify_probability(0.05) == ReturnRiskBand.VERY_LOW
        assert classifier.classify_probability(0.0999) == ReturnRiskBand.VERY_LOW

        assert classifier.classify_probability(0.10) == ReturnRiskBand.LOW
        assert classifier.classify_probability(0.20) == ReturnRiskBand.LOW
        assert classifier.classify_probability(0.2499) == ReturnRiskBand.LOW

        assert classifier.classify_probability(0.25) == ReturnRiskBand.MEDIUM
        assert classifier.classify_probability(0.40) == ReturnRiskBand.MEDIUM
        assert classifier.classify_probability(0.4999) == ReturnRiskBand.MEDIUM

        assert classifier.classify_probability(0.50) == ReturnRiskBand.HIGH
        assert classifier.classify_probability(0.60) == ReturnRiskBand.HIGH
        assert classifier.classify_probability(0.7499) == ReturnRiskBand.HIGH

        assert classifier.classify_probability(0.75) == ReturnRiskBand.VERY_HIGH
        assert classifier.classify_probability(0.90) == ReturnRiskBand.VERY_HIGH
        assert classifier.classify_probability(1.0) == ReturnRiskBand.VERY_HIGH

    def test_invalid_probability_out_of_bounds(self):
        classifier = ReturnRiskClassifier()

        with pytest.raises(ValueError, match="Probability must be in"):
            classifier.classify_probability(-0.01)

        with pytest.raises(ValueError, match="Probability must be in"):
            classifier.classify_probability(1.05)

    def test_non_causal_interpretation(self):
        classifier = ReturnRiskClassifier()
        interp = classifier.generate_interpretation(0.35, ReturnRiskBand.MEDIUM)

        assert "Risk Band: MEDIUM" in interp
        assert "35%" in interp
        assert "modeled probability of return" in interp
        # Must be non-causal (no speculation about customer mind, quality defects, or fraud)
        assert "because the customer" not in interp.lower()
        assert "dissatisfied" not in interp.lower()
        assert "fraud" not in interp.lower()

    def test_classify_dataframe(self):
        classifier = ReturnRiskClassifier()
        df = pd.DataFrame({
            "order_id": ["O1", "O2", "O3"],
            "sku": ["SKU1", "SKU2", "SKU3"],
            "raw_probability": [0.05, 0.45, 0.85],
            "calibrated_probability": [0.08, 0.35, 0.78],
        })

        res = classifier.classify_dataframe(
            df,
            model_name="TestLGBM",
            calibration_method="isotonic",
            threshold=0.30,
        )

        assert len(res) == 3
        assert "risk_band" in res.columns
        assert "risk_policy_version" in res.columns
        assert "interpretation" in res.columns
        assert "model_name" in res.columns
        assert "threshold" in res.columns

        assert res.iloc[0]["risk_band"] == ReturnRiskBand.VERY_LOW.value
        assert res.iloc[1]["risk_band"] == ReturnRiskBand.MEDIUM.value
        assert res.iloc[2]["risk_band"] == ReturnRiskBand.VERY_HIGH.value

        assert res.iloc[0]["risk_policy_version"] == "1.0.0-prototype"
        assert res.iloc[0]["model_name"] == "TestLGBM"
        assert res.iloc[0]["threshold"] == 0.30

    def test_classify_dataframe_missing_columns(self):
        classifier = ReturnRiskClassifier()
        df = pd.DataFrame({"some_col": [1, 2]})
        with pytest.raises(ValueError, match="Missing probability columns"):
            classifier.classify_dataframe(df)

    def test_classify_dataframe_empty(self):
        classifier = ReturnRiskClassifier()
        res = classifier.classify_dataframe(pd.DataFrame())
        assert res.empty


class TestRiskPolicyEvaluation:
    """Test suite for empirical risk band validation."""

    def test_evaluate_risk_policy_metrics(self):
        df_results = pd.DataFrame({
            "risk_band": [
                "VERY_LOW", "VERY_LOW", "LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "VERY_HIGH"
            ],
            "calibrated_probability": [0.05, 0.07, 0.15, 0.20, 0.35, 0.40, 0.60, 0.85],
            "actual_target": [0, 0, 0, 1, 0, 1, 1, 1],
        })

        evals = evaluate_risk_policy(df_results)
        assert len(evals) == 5

        # Check VERY_LOW: 2 records, 0 returns, mean prob = 0.06
        vl = next(e for e in evals if e.risk_band == "VERY_LOW")
        assert vl.record_count == 2
        assert vl.percentage_of_records == 25.0
        assert vl.actual_returned_count == 0
        assert vl.observed_return_rate == 0.0
        assert vl.average_calibrated_probability == 0.06
        assert vl.calibration_error == 0.06

        # Convert to DataFrame
        df_eval = risk_evaluations_to_dataframe(evals)
        assert len(df_eval) == 5
        assert "observed_return_rate" in df_eval.columns
        assert "calibration_error" in df_eval.columns

    def test_evaluate_risk_policy_empty(self):
        assert evaluate_risk_policy(pd.DataFrame()) == []
        assert risk_evaluations_to_dataframe([]) .empty


class TestSegmentCalibration:
    """Test suite for operational slice calibration auditing with sample size guards."""

    def test_evaluate_segment_calibration_with_sample_guard(self):
        # 40 records in Channel C1, 10 records in Channel C2
        np.random.seed(42)
        n1 = 40
        n2 = 10
        y1 = np.random.choice([0, 1], size=n1, p=[0.8, 0.2])
        p1 = np.random.uniform(0.1, 0.4, size=n1)

        y2 = np.random.choice([0, 1], size=n2, p=[0.9, 0.1])
        p2 = np.random.uniform(0.05, 0.25, size=n2)

        df = pd.DataFrame({
            "channel_id": ["C1"] * n1 + ["C2"] * n2,
            "calibrated_probability": np.concatenate([p1, p2]),
            "actual_target": np.concatenate([y1, y2]),
        })

        seg_eval = evaluate_segment_calibration(df, dimensions=["channel_id"], min_sample_size=30)
        assert len(seg_eval) == 2

        row_c1 = seg_eval[seg_eval["segment_value"] == "C1"].iloc[0]
        assert row_c1["sample_size"] == 40
        assert bool(row_c1["is_sufficient_sample"]) is True
        assert row_c1["ece"] is not None

        row_c2 = seg_eval[seg_eval["segment_value"] == "C2"].iloc[0]
        assert row_c2["sample_size"] == 10
        assert bool(row_c2["is_sufficient_sample"]) is False
        assert pd.isna(row_c2["ece"])  # Guard suppresses ECE on tiny samples (represented as NaN in pandas float series)


class TestEndToEndCalibrationAndRisk:
    """Integration test: raw probability -> calibrator -> calibrated prob -> risk band."""

    def test_end_to_end_pipeline(self):
        np.random.seed(42)
        n_val = 150
        n_test = 50

        # Val set to fit calibrator
        y_val = np.random.choice([0, 1], size=n_val, p=[0.85, 0.15])
        y_val[0] = 1
        y_val[1] = 0
        raw_val = np.random.uniform(0.1, 0.9, size=n_val)

        calibrator = ReturnProbabilityCalibrator(method="sigmoid")
        calibrator.fit(y_val, raw_val)

        # Test set to calibrate and classify
        y_test = np.random.choice([0, 1], size=n_test, p=[0.85, 0.15])
        raw_test = np.random.uniform(0.1, 0.9, size=n_test)

        cal_test = calibrator.calibrate(raw_test)
        assert len(cal_test) == n_test
        assert np.all((cal_test >= 0.0) & (cal_test <= 1.0))

        df_test = pd.DataFrame({
            "sale_id": [f"S{i}" for i in range(n_test)],
            "raw_probability": raw_test,
            "calibrated_probability": cal_test,
            "actual_target": y_test,
        })

        classifier = ReturnRiskClassifier()
        classified_df = classifier.classify_dataframe(
            df_test,
            model_name="EndToEndModel",
            calibration_method="sigmoid",
        )

        assert len(classified_df) == n_test
        assert set(classified_df["risk_band"].unique()).issubset({
            "VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"
        })

        evals = evaluate_risk_policy(classified_df)
        assert len(evals) > 0
