"""Return Risk ML Model Evaluation Engine (Phase 5C-2A).

Comprehensive evaluation routines for tabular return risk prediction:
- Train partition metrics: ROC-AUC, PR-AUC, Brier score
- Validation partition metrics: ROC-AUC, PR-AUC, Precision, Recall, F1, Brier score, Confusion Matrix
- Validation probability threshold analysis (0.10 to 0.90) for trade-off exploration
- Holdout test partition metrics: Final unbiased evaluation strictly without hyperparameter tuning
- Granular segment evaluation by SKU, Channel, Warehouse, Velocity Tier, and Cold-Start Status
- Sample size threshold guards identifying statistically weak evaluation slices
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from commerce_ai.returns.schemas import (
    ModelEvaluationMetrics,
    ReturnModelMetadata,
    ReturnPredictionOutputRecord,
    SegmentEvaluationMetric,
    ThresholdAnalysisPoint,
)
from commerce_ai.returns.prediction_dataset import ReturnPredictionDataset
from commerce_ai.returns.prediction_models import BaseReturnRiskModel


# Default candidate decision thresholds for threshold analysis
DEFAULT_THRESHOLDS_TO_SCAN: List[float] = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
]

DEFAULT_SEGMENT_DIMENSIONS: List[str] = [
    "sku_id",
    "warehouse_id",
    "channel_id",
    "velocity_tier",
    "is_cold_start_sku",
]


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray, Sequence[int]],
    y_prob: Union[pd.Series, np.ndarray, Sequence[float]],
    threshold: float = 0.50,
    split_name: str = "validation",
) -> ModelEvaluationMetrics:
    """Calculate classification performance and calibration metrics for a dataset split.

    Args:
        y_true: Ground truth binary targets (0 or 1).
        y_prob: Estimated probabilities of return in [0.0, 1.0].
        threshold: Decision threshold for discrete classification.
        split_name: Label identifier ('train', 'validation', or 'test').

    Returns:
        Populated ModelEvaluationMetrics instance.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    n_samples = len(y_true_arr)
    if n_samples == 0:
        return ModelEvaluationMetrics(
            split=split_name,
            threshold=threshold,
            row_count=0,
            positive_count=0,
            negative_count=0,
            positive_rate=0.0,
        )

    pos_count = int((y_true_arr == 1).sum())
    neg_count = int((y_true_arr == 0).sum())
    pos_rate = round(float(pos_count / n_samples), 4)

    # Probability calibration: Brier score
    brier = round(float(brier_score_loss(y_true_arr, y_prob_arr)), 4)

    # Ranking metrics (require at least 2 distinct classes)
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    if len(np.unique(y_true_arr)) > 1:
        try:
            roc_auc = round(float(roc_auc_score(y_true_arr, y_prob_arr)), 4)
        except ValueError:
            roc_auc = None
        try:
            pr_auc = round(float(average_precision_score(y_true_arr, y_prob_arr)), 4)
        except ValueError:
            pr_auc = None

    # Discrete classification at threshold
    y_pred = (y_prob_arr >= threshold).astype(int)
    prec = round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4)

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    cm_dict = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    return ModelEvaluationMetrics(
        split=split_name,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        precision=prec,
        recall=rec,
        f1_score=f1,
        threshold=round(float(threshold), 4),
        confusion_matrix=cm_dict,
        row_count=n_samples,
        positive_count=pos_count,
        negative_count=neg_count,
        positive_rate=pos_rate,
    )


def evaluate_threshold_scan(
    y_true: Union[pd.Series, np.ndarray, Sequence[int]],
    y_prob: Union[pd.Series, np.ndarray, Sequence[float]],
    thresholds: Optional[Sequence[float]] = None,
) -> List[ThresholdAnalysisPoint]:
    """Evaluate precision, recall, and F1 trade-offs across a range of decision thresholds."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    thresh_list = sorted(list(thresholds or DEFAULT_THRESHOLDS_TO_SCAN))

    n_samples = len(y_true_arr)
    points: List[ThresholdAnalysisPoint] = []

    for t in thresh_list:
        y_pred = (y_prob_arr >= t).astype(int)
        prec = round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4)
        rec = round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4)
        f1 = round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4)

        cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
        tn = int(cm[0, 0])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tp = int(cm[1, 1])

        pred_pos = tp + fp
        pred_pos_rate = round(float(pred_pos / n_samples), 4) if n_samples > 0 else 0.0

        points.append(
            ThresholdAnalysisPoint(
                threshold=round(float(t), 4),
                precision=prec,
                recall=rec,
                f1_score=f1,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                predicted_positive_count=pred_pos,
                predicted_positive_rate=pred_pos_rate,
            )
        )

    return points


def threshold_analysis_to_dataframe(points: List[ThresholdAnalysisPoint]) -> pd.DataFrame:
    """Convert threshold analysis scan points to a flat pandas DataFrame."""
    if not points:
        return pd.DataFrame()
    return pd.DataFrame([p.to_dict() for p in points])


def evaluate_segments(
    predictions_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    dimensions: Optional[List[str]] = None,
    min_sample_size: int = 30,
    threshold: float = 0.50,
) -> List[SegmentEvaluationMetric]:
    """Evaluate model performance across operational segments and demographic cohorts.

    Args:
        predictions_df: Structured predictions DataFrame with actual_target and predicted_probability.
        features_df: Optional auxiliary features DataFrame providing velocity_tier, is_cold_start, etc.
        dimensions: List of column names to slice by.
        min_sample_size: Minimum sample size required to consider evaluation statistically robust.
        threshold: Decision cutoff applied for precision, recall, and F1.

    Returns:
        List of SegmentEvaluationMetric objects.
    """
    if predictions_df is None or predictions_df.empty:
        return []

    # Merge features if provided
    combined = predictions_df.copy()
    if features_df is not None and not features_df.empty:
        for c in features_df.columns:
            if c not in combined.columns and len(features_df) == len(combined):
                combined[c] = features_df[c].values

    dims_to_check = dimensions or DEFAULT_SEGMENT_DIMENSIONS
    results: List[SegmentEvaluationMetric] = []

    for dim in dims_to_check:
        if dim not in combined.columns:
            continue

        grouped = combined.groupby(dim, observed=False)
        for val, group in grouped:
            n = len(group)
            if n == 0:
                continue

            y_t = pd.to_numeric(group["actual_target"], errors="coerce").dropna().values.astype(int)
            y_p = pd.to_numeric(group["predicted_probability"], errors="coerce").dropna().values.astype(float)

            if len(y_t) == 0 or len(y_p) == 0:
                continue

            pos_cnt = int((y_t == 1).sum())
            pos_rate = round(float(pos_cnt / n), 4)
            is_sufficient = bool(n >= min_sample_size)

            brier = round(float(brier_score_loss(y_t, y_p)), 4)

            roc_auc: Optional[float] = None
            pr_auc: Optional[float] = None
            if len(np.unique(y_t)) > 1:
                try:
                    roc_auc = round(float(roc_auc_score(y_t, y_p)), 4)
                except ValueError:
                    roc_auc = None
                try:
                    pr_auc = round(float(average_precision_score(y_t, y_p)), 4)
                except ValueError:
                    pr_auc = None

            y_pred = (y_p >= threshold).astype(int)
            prec = round(float(precision_score(y_t, y_pred, zero_division=0)), 4)
            rec = round(float(recall_score(y_t, y_pred, zero_division=0)), 4)
            f1 = round(float(f1_score(y_t, y_pred, zero_division=0)), 4)

            results.append(
                SegmentEvaluationMetric(
                    dimension=dim,
                    segment_value=str(val),
                    sample_size=n,
                    positive_count=pos_cnt,
                    positive_rate=pos_rate,
                    is_sufficient_sample=is_sufficient,
                    brier_score=brier,
                    roc_auc=roc_auc,
                    pr_auc=pr_auc,
                    precision=prec,
                    recall=rec,
                    f1_score=f1,
                )
            )

    return results


def segments_to_dataframe(metrics: List[SegmentEvaluationMetric]) -> pd.DataFrame:
    """Export segment evaluation metrics to a flat pandas DataFrame."""
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame([m.to_dict() for m in metrics])


def evaluate_model_pipeline(
    model: BaseReturnRiskModel,
    dataset: ReturnPredictionDataset,
    default_threshold: float = 0.50,
    thresholds_to_scan: Optional[List[float]] = None,
    min_segment_sample_size: int = 30,
) -> Dict[str, Any]:
    """Execute end-to-end evaluation pipeline across train, validation, and holdout test splits.

    Guarantees:
    - Train evaluated on ranking and calibration (ROC-AUC, PR-AUC, Brier score).
    - Validation evaluated on full metrics and threshold trade-off scan.
    - Test evaluated as unbiased final holdout without any tuning.
    - Segments evaluated with statistical sufficiency guards.

    Args:
        model: Trained BaseReturnRiskModel instance.
        dataset: ReturnPredictionDataset containing chronological splits.
        default_threshold: Reporting probability decision threshold (default 0.50).
        thresholds_to_scan: List of candidate thresholds for validation curve analysis.
        min_segment_sample_size: Minimum sample size threshold for reliable segment slicing.

    Returns:
        Dictionary containing metric summaries, threshold scans, predictions, and updated metadata.
    """
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_val_data()
    X_test, y_test = dataset.get_test_data()

    # 1. Train Evaluation
    p_train = model.predict_proba(X_train) if not X_train.empty else np.array([])
    train_metrics = compute_classification_metrics(
        y_true=y_train,
        y_prob=p_train,
        threshold=default_threshold,
        split_name="train",
    )

    # 2. Validation Evaluation
    p_val = model.predict_proba(X_val) if not X_val.empty else np.array([])
    val_metrics = compute_classification_metrics(
        y_true=y_val,
        y_prob=p_val,
        threshold=default_threshold,
        split_name="validation",
    )
    val_threshold_scan = (
        evaluate_threshold_scan(y_val, p_val, thresholds=thresholds_to_scan)
        if len(y_val) > 0
        else []
    )

    # 3. Test Evaluation (Holdout reporting only, zero tuning)
    p_test = model.predict_proba(X_test) if not X_test.empty else np.array([])
    test_metrics = compute_classification_metrics(
        y_true=y_test,
        y_prob=p_test,
        threshold=default_threshold,
        split_name="test",
    )

    # 4. Structured Predictions with Metadata for Test Set
    meta_test = (
        dataset.metadata.iloc[dataset.test_indices].copy().reset_index(drop=True)
        if dataset.test_indices and not dataset.metadata.empty
        else pd.DataFrame()
    )
    test_predictions = (
        model.predict_with_metadata(X_test.reset_index(drop=True), meta_test, threshold=default_threshold)
        if not X_test.empty and not meta_test.empty
        else pd.DataFrame()
    )

    # 5. Segment Evaluation on Test Set
    segment_metrics = (
        evaluate_segments(
            predictions_df=test_predictions,
            features_df=X_test.reset_index(drop=True),
            min_sample_size=min_segment_sample_size,
            threshold=default_threshold,
        )
        if not test_predictions.empty
        else []
    )

    # 6. Update Model Metadata
    if model.metadata is not None:
        model.metadata.train_metrics = train_metrics.to_dict()
        model.metadata.validation_metrics = val_metrics.to_dict()
        model.metadata.test_metrics = test_metrics.to_dict()
        model.metadata.test_row_count = len(X_test)
        model.metadata.threshold_used = default_threshold

    return {
        "model": model,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "val_threshold_scan": val_threshold_scan,
        "test_metrics": test_metrics,
        "test_predictions": test_predictions,
        "segment_metrics": segment_metrics,
        "metadata": model.metadata,
    }
