"""Unit tests for Return Probability Calibration Engine (Phase 5C-2B)."""

import pytest
import numpy as np
import pandas as pd

from commerce_ai.returns.schemas import (
    CalibrationBin,
    CalibrationMethod,
    CalibrationMetrics,
)
from commerce_ai.returns.calibration import (
    ReturnProbabilityCalibrator,
    compute_calibration_metrics,
    compute_reliability_curve,
    evaluate_and_compare_calibrations,
    validate_probability_bounds,
)


@pytest.fixture
def synthetic_val_test_data():
    """Create deterministic synthetic probability and target arrays."""
    np.random.seed(42)
    n_val = 200
    n_test = 100

    # Overconfident uncalibrated probabilities with 10% true positive rate
    y_val = np.random.choice([0, 1], size=n_val, p=[0.90, 0.10])
    # Ensure both classes present
    y_val[0] = 1
    y_val[1] = 0

    # Raw probabilities intentionally miscalibrated (e.g. shifted upwards)
    p_val = np.where(y_val == 1, np.random.uniform(0.3, 0.9, size=n_val), np.random.uniform(0.1, 0.6, size=n_val))

    y_test = np.random.choice([0, 1], size=n_test, p=[0.90, 0.10])
    y_test[0] = 1
    y_test[1] = 0
    p_test = np.where(y_test == 1, np.random.uniform(0.3, 0.9, size=n_test), np.random.uniform(0.1, 0.6, size=n_test))

    return y_val, p_val, y_test, p_test


class TestProbabilityValidation:
    """Test suite for probability boundary checking."""

    def test_valid_probabilities_pass(self):
        probs = [0.0, 0.25, 0.5, 0.75, 1.0]
        validated = validate_probability_bounds(probs)
        np.testing.assert_allclose(validated, probs)

    def test_negative_probability_rejected(self):
        with pytest.raises(ValueError, match="values must be within"):
            validate_probability_bounds([-0.05, 0.5])

    def test_greater_than_one_probability_rejected(self):
        with pytest.raises(ValueError, match="values must be within"):
            validate_probability_bounds([0.5, 1.05])

    def test_empty_probabilities(self):
        res = validate_probability_bounds([])
        assert len(res) == 0


class TestReliabilityCurve:
    """Test suite for reliability binning and calibration error curves."""

    def test_reliability_bins_structure(self, synthetic_val_test_data):
        y_val, p_val, _, _ = synthetic_val_test_data
        bins = compute_reliability_curve(y_val, p_val, n_bins=10)

        assert len(bins) == 10
        total_counted = sum(b.sample_count for b in bins)
        assert total_counted == len(y_val)

        for b in bins:
            assert 0.0 <= b.lower_bound < b.upper_bound <= 1.0
            if b.sample_count > 0:
                assert 0.0 <= b.mean_predicted_probability <= 1.0
                assert 0.0 <= b.observed_return_rate <= 1.0
                assert 0.0 <= b.absolute_error <= 1.0

    def test_empty_bins_handled_safely(self):
        # All probabilities concentrated in [0.2, 0.3]
        y_true = np.array([0, 0, 1, 0, 0])
        p_prob = np.array([0.22, 0.25, 0.28, 0.21, 0.29])

        bins = compute_reliability_curve(y_true, p_prob, n_bins=5)
        assert len(bins) == 5

        # Bins other than index 1 should be empty
        empty_bins = [b for b in bins if b.bin_index != 1]
        for eb in empty_bins:
            assert eb.sample_count == 0
            assert eb.mean_predicted_probability == 0.0
            assert eb.observed_return_rate == 0.0
            assert eb.absolute_error == 0.0

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError, match="Number of reliability bins must be >= 2"):
            compute_reliability_curve([0, 1], [0.1, 0.9], n_bins=1)


class TestCalibrationMetricsCalculation:
    """Test suite for calibration summary metrics (ECE, MCE, Brier, Log Loss)."""

    def test_calibration_metrics(self, synthetic_val_test_data):
        y_val, p_val, _, _ = synthetic_val_test_data
        metrics = compute_calibration_metrics(y_val, p_val, method_name="raw", n_bins=10)

        assert metrics.method == "raw"
        assert 0.0 <= metrics.brier_score <= 1.0
        assert metrics.log_loss >= 0.0
        assert 0.0 <= metrics.ece <= 1.0
        assert 0.0 <= metrics.mce <= 1.0
        assert metrics.mce >= metrics.ece  # MCE is always >= ECE
        assert 0.0 <= metrics.mean_predicted_probability <= 1.0
        assert 0.0 <= metrics.observed_positive_rate <= 1.0
        assert len(metrics.bins) == 10


class TestReturnProbabilityCalibrator:
    """Test suite for calibrator fitting and inference."""

    def test_sigmoid_calibrator_fit_and_predict(self, synthetic_val_test_data):
        y_val, p_val, _, p_test = synthetic_val_test_data

        calibrator = ReturnProbabilityCalibrator(method=CalibrationMethod.SIGMOID, random_state=42)
        calibrator.fit(y_val, p_val)

        assert calibrator.is_fitted
        p_cal_val = calibrator.predict_proba(p_val)
        p_cal_test = calibrator.predict_proba(p_test)

        assert len(p_cal_val) == len(p_val)
        assert len(p_cal_test) == len(p_test)
        assert np.all(p_cal_val >= 0.0) and np.all(p_cal_val <= 1.0)
        assert np.all(p_cal_test >= 0.0) and np.all(p_cal_test <= 1.0)

    def test_isotonic_calibrator_fit_and_predict(self, synthetic_val_test_data):
        y_val, p_val, _, p_test = synthetic_val_test_data

        calibrator = ReturnProbabilityCalibrator(method=CalibrationMethod.ISOTONIC, random_state=42)
        calibrator.fit(y_val, p_val)

        assert calibrator.is_fitted
        p_cal = calibrator.predict_proba(p_test)
        assert len(p_cal) == len(p_test)
        assert np.all(p_cal >= 0.0) and np.all(p_cal <= 1.0)

    def test_none_method_passthrough(self):
        calibrator = ReturnProbabilityCalibrator(method=CalibrationMethod.NONE)
        calibrator.fit([0, 1], [0.2, 0.8])

        raw = [0.1, 0.5, 0.9]
        calibrated = calibrator.predict_proba(raw)
        np.testing.assert_allclose(calibrated, raw)

    def test_predict_before_fit_raises(self):
        calibrator = ReturnProbabilityCalibrator(method=CalibrationMethod.SIGMOID)
        with pytest.raises(ValueError, match="must be fitted"):
            calibrator.predict_proba([0.5])

    def test_single_class_validation_raises(self):
        calibrator = ReturnProbabilityCalibrator(method=CalibrationMethod.SIGMOID)
        with pytest.raises(ValueError, match="at least two distinct classes"):
            calibrator.fit([0, 0, 0], [0.1, 0.2, 0.3])

    def test_empty_validation_raises(self):
        calibrator = ReturnProbabilityCalibrator(method=CalibrationMethod.SIGMOID)
        with pytest.raises(ValueError, match="empty validation data"):
            calibrator.fit([], [])

    def test_calibrate_dataframe(self):
        df = pd.DataFrame({
            "predicted_probability": [0.1, 0.5, 0.8],
            "order_id": ["O1", "O2", "O3"],
        })
        calibrator = ReturnProbabilityCalibrator(method=CalibrationMethod.SIGMOID)
        calibrator.fit([0, 1, 0, 1], [0.2, 0.4, 0.6, 0.8])

        res = calibrator.calibrate_dataframe(df)
        assert "calibrated_probability" in res.columns
        assert len(res) == 3


class TestCalibrationComparisonAndSelection:
    """Test suite for comparing calibration techniques."""

    def test_evaluate_and_compare_calibrations(self, synthetic_val_test_data):
        y_val, p_val, y_test, p_test = synthetic_val_test_data

        comparison = evaluate_and_compare_calibrations(
            y_val=y_val,
            p_val=p_val,
            y_test=y_test,
            p_test=p_test,
            candidate_methods=["raw", "sigmoid", "isotonic"],
            n_bins=10,
        )

        assert "calibrators" in comparison
        assert "validation_metrics" in comparison
        assert "test_metrics" in comparison
        assert "best_method_val" in comparison

        assert set(comparison["validation_metrics"].keys()) == {"raw", "sigmoid", "isotonic"}
        assert set(comparison["test_metrics"].keys()) == {"raw", "sigmoid", "isotonic"}
        assert comparison["best_method_val"] in {"raw", "sigmoid", "isotonic"}
