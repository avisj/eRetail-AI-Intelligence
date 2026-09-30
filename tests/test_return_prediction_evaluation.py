"""Unit tests for Return Risk ML Model Evaluation Engine (Phase 5C-2A)."""

import pytest
import numpy as np
import pandas as pd

from commerce_ai.returns.schemas import (
    ModelEvaluationMetrics,
    ThresholdAnalysisPoint,
    SegmentEvaluationMetric,
)
from commerce_ai.returns.prediction_evaluation import (
    compute_classification_metrics,
    evaluate_threshold_scan,
    threshold_analysis_to_dataframe,
    evaluate_segments,
    segments_to_dataframe,
    evaluate_model_pipeline,
)
from commerce_ai.returns.prediction_models import (
    LogisticRegressionReturnModel,
    LightGBMReturnModel,
)
from commerce_ai.returns.prediction_dataset import (
    ReturnPredictionDataset,
    ReturnPredictionQualityReport,
    TemporalSplitInfo,
)


@pytest.fixture
def synthetic_predictions():
    """Create deterministic ground truth and probability arrays."""
    np.random.seed(42)
    n = 100
    y_true = np.array([1] * 20 + [0] * 80)
    # Give positives higher average probabilities
    y_prob = np.concatenate([
        np.random.uniform(0.4, 0.9, size=20),
        np.random.uniform(0.05, 0.45, size=80),
    ])
    return y_true, y_prob


class TestClassificationMetrics:
    """Test suite for core metric calculations."""

    def test_metrics_calculation(self, synthetic_predictions):
        y_true, y_prob = synthetic_predictions
        metrics = compute_classification_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.50,
            split_name="validation",
        )

        assert metrics.split == "validation"
        assert metrics.row_count == 100
        assert metrics.positive_count == 20
        assert metrics.negative_count == 80
        assert metrics.positive_rate == 0.20

        # Quality metrics
        assert 0.5 < metrics.roc_auc <= 1.0
        assert 0.2 < metrics.pr_auc <= 1.0
        assert 0.0 <= metrics.brier_score <= 1.0

        # Classification counts
        cm = metrics.confusion_matrix
        assert cm["tp"] + cm["fn"] == 20
        assert cm["fp"] + cm["tn"] == 80
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0

    def test_empty_predictions(self):
        metrics = compute_classification_metrics([], [], split_name="empty")
        assert metrics.row_count == 0
        assert metrics.roc_auc is None
        assert metrics.brier_score is None

    def test_single_class_target(self):
        y_true = [0, 0, 0, 0]
        y_prob = [0.1, 0.2, 0.15, 0.3]
        metrics = compute_classification_metrics(y_true, y_prob)

        assert metrics.positive_count == 0
        assert metrics.roc_auc is None
        assert metrics.pr_auc is None
        assert metrics.brier_score is not None


class TestThresholdAnalysis:
    """Test suite for decision threshold scanning."""

    def test_evaluate_threshold_scan(self, synthetic_predictions):
        y_true, y_prob = synthetic_predictions
        thresholds = [0.20, 0.50, 0.80]
        points = evaluate_threshold_scan(y_true, y_prob, thresholds=thresholds)

        assert len(points) == 3
        # Recall should be non-increasing with higher threshold
        assert points[0].recall >= points[1].recall >= points[2].recall
        # False positives should be non-increasing with higher threshold
        assert points[0].fp >= points[1].fp >= points[2].fp

        for p in points:
            assert p.tp + p.fn == 20
            assert p.fp + p.tn == 80
            assert 0.0 <= p.precision <= 1.0
            assert 0.0 <= p.recall <= 1.0
            assert 0.0 <= p.f1_score <= 1.0

    def test_threshold_dataframe_export(self, synthetic_predictions):
        y_true, y_prob = synthetic_predictions
        points = evaluate_threshold_scan(y_true, y_prob)
        df = threshold_analysis_to_dataframe(points)

        assert not df.empty
        assert "threshold" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1_score" in df.columns


class TestSegmentEvaluation:
    """Test suite for slice-level evaluation."""

    def test_evaluate_segments_with_sample_size_guard(self):
        # Create 2 channels: one with 50 rows (sufficient), one with 10 rows (insufficient)
        n1, n2 = 50, 10
        df = pd.DataFrame({
            "channel_id": ["CH_LARGE"] * n1 + ["CH_SMALL"] * n2,
            "actual_target": [1] * 10 + [0] * 40 + [1] * 2 + [0] * 8,
            "predicted_probability": np.random.uniform(0.1, 0.6, size=n1 + n2),
        })

        results = evaluate_segments(df, dimensions=["channel_id"], min_sample_size=30)
        assert len(results) == 2

        large_res = next(r for r in results if r.segment_value == "CH_LARGE")
        small_res = next(r for r in results if r.segment_value == "CH_SMALL")

        assert large_res.is_sufficient_sample is True
        assert small_res.is_sufficient_sample is False
        assert large_res.sample_size == 50
        assert small_res.sample_size == 10

    def test_segments_to_dataframe(self):
        df = pd.DataFrame({
            "warehouse_id": ["WH_01"] * 40,
            "actual_target": [1] * 10 + [0] * 30,
            "predicted_probability": [0.4] * 40,
        })
        results = evaluate_segments(df, dimensions=["warehouse_id"])
        seg_df = segments_to_dataframe(results)

        assert len(seg_df) == 1
        assert seg_df["dimension"].iloc[0] == "warehouse_id"
        assert seg_df["sample_size"].iloc[0] == 40


class TestModelPipelineEvaluation:
    """Test end-to-end model evaluation across dataset splits."""

    def test_evaluate_model_pipeline(self):
        np.random.seed(42)
        n = 150
        X = pd.DataFrame({
            "num_feat": np.random.randn(n),
            "channel_id": np.random.choice(["A", "B"], size=n),
            "warehouse_id": np.random.choice(["W1", "W2"], size=n),
        })
        y = pd.Series(np.random.choice([0, 1], size=n, p=[0.8, 0.2]))
        # Ensure at least two classes in each split
        y.iloc[0] = 1
        y.iloc[1] = 0
        y.iloc[100] = 1
        y.iloc[101] = 0
        y.iloc[130] = 1
        y.iloc[131] = 0

        metadata = pd.DataFrame({
            "prediction_id": [f"P_{i}" for i in range(n)],
            "sale_id": [f"S_{i}" for i in range(n)],
            "order_id": [f"O_{i}" for i in range(n)],
            "sku_id": [f"SKU_{i%5}" for i in range(n)],
            "warehouse_id": X["warehouse_id"].values,
            "channel_id": X["channel_id"].values,
            "prediction_date": ["2024-01-01"] * n,
            "target_returned": y.values,
        })

        # Train (0..100), Val (100..125), Test (125..150)
        train_idx = list(range(0, 100))
        val_idx = list(range(100, 125))
        test_idx = list(range(125, 150))

        dataset = ReturnPredictionDataset(
            X=X,
            y=y,
            metadata=metadata,
            feature_definitions={},
            quality_report=ReturnPredictionQualityReport(),
            temporal_split_info=TemporalSplitInfo(),
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
        )

        model = LogisticRegressionReturnModel()
        X_tr, y_tr = dataset.get_train_data()
        X_va, y_va = dataset.get_val_data()
        model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

        eval_bundle = evaluate_model_pipeline(
            model=model,
            dataset=dataset,
            default_threshold=0.50,
            min_segment_sample_size=10,
        )

        assert "train_metrics" in eval_bundle
        assert "validation_metrics" in eval_bundle
        assert "val_threshold_scan" in eval_bundle
        assert "test_metrics" in eval_bundle
        assert "test_predictions" in eval_bundle
        assert "segment_metrics" in eval_bundle

        assert eval_bundle["train_metrics"].row_count == 100
        assert eval_bundle["validation_metrics"].row_count == 25
        assert eval_bundle["test_metrics"].row_count == 25

        # Check metadata update
        meta = eval_bundle["metadata"]
        assert meta.train_metrics is not None
        assert meta.validation_metrics is not None
        assert meta.test_metrics is not None
        assert meta.test_row_count == 25
