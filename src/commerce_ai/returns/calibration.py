"""Return Probability Calibration Engine (Phase 5C-2B).

Provides post-hoc probability calibration routines for binary return risk prediction:
1. Sigmoid Calibration (Platt Scaling):
   - Univariate logistic regression on log-odds / logits
   - Parametric, smooth, monotonic scaling
2. Isotonic Calibration:
   - Non-parametric isotonic regression fitting piece-wise constant non-decreasing step function
   - Flexible, distribution-free, monotonic probability mapping
3. Calibration Diagnostics & Metrics:
   - Expected Calibration Error (ECE)
   - Maximum Calibration Error (MCE)
   - Brier Score & Cross-Entropy / Log Loss
   - Reliability diagrams / Calibration curve bins
   - Ranking preservation audits (ROC-AUC / PR-AUC)

Strict Data Split Rules:
- Calibrators are fitted strictly on VALIDATION split data (never on holdout test data).
- The holdout test set is evaluated post-calibration as an unbiased final check.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from commerce_ai.returns.schemas import (
    CalibrationBin,
    CalibrationMethod,
    CalibrationMetrics,
)


def validate_probability_bounds(
    probabilities: Union[pd.Series, np.ndarray, Sequence[float]],
    param_name: str = "probability",
) -> np.ndarray:
    """Validate that all probability values lie strictly within [0.0, 1.0]."""
    p_arr = np.asarray(probabilities, dtype=float)
    if len(p_arr) == 0:
        return p_arr
    min_val = float(np.min(p_arr))
    max_val = float(np.max(p_arr))
    if min_val < -1e-6 or max_val > 1.0 + 1e-6:
        raise ValueError(
            f"Invalid {param_name}: values must be within [0.0, 1.0], got range [{min_val:.4f}, {max_val:.4f}]"
        )
    return np.clip(p_arr, 0.0, 1.0)


def compute_reliability_curve(
    y_true: Union[pd.Series, np.ndarray, Sequence[int]],
    y_prob: Union[pd.Series, np.ndarray, Sequence[float]],
    n_bins: int = 10,
) -> List[CalibrationBin]:
    """Partition predicted probabilities into uniform bins and compute empirical calibration metrics.

    Args:
        y_true: Ground truth binary outcomes (0 or 1).
        y_prob: Estimated probabilities of return in [0.0, 1.0].
        n_bins: Number of equal-width probability partitions across [0.0, 1.0].

    Returns:
        List of CalibrationBin instances documenting sample counts, confidences, and errors.
    """
    if n_bins < 2:
        raise ValueError(f"Number of reliability bins must be >= 2, got {n_bins}")

    y_t = np.asarray(y_true, dtype=int)
    y_p = validate_probability_bounds(y_prob, param_name="y_prob")

    n_samples = len(y_t)
    if n_samples == 0:
        return []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[CalibrationBin] = []

    for i in range(n_bins):
        low = float(bin_edges[i])
        high = float(bin_edges[i + 1])

        # Last bin is closed on both sides [low, high], others are [low, high)
        if i == n_bins - 1:
            mask = (y_p >= low) & (y_p <= high)
        else:
            mask = (y_p >= low) & (y_p < high)

        count = int(mask.sum())
        if count > 0:
            mean_pred = float(np.mean(y_p[mask]))
            obs_rate = float(np.mean(y_t[mask]))
            abs_err = float(abs(mean_pred - obs_rate))
        else:
            mean_pred = 0.0
            obs_rate = 0.0
            abs_err = 0.0

        bins.append(
            CalibrationBin(
                bin_index=i,
                lower_bound=round(low, 4),
                upper_bound=round(high, 4),
                sample_count=count,
                mean_predicted_probability=round(mean_pred, 4),
                observed_return_rate=round(obs_rate, 4),
                absolute_error=round(abs_err, 4),
            )
        )

    return bins


def compute_calibration_metrics(
    y_true: Union[pd.Series, np.ndarray, Sequence[int]],
    y_prob: Union[pd.Series, np.ndarray, Sequence[float]],
    method_name: str = "raw",
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Calculate comprehensive calibration and ranking metrics for a probability prediction vector.

    Args:
        y_true: Ground truth binary outcomes (0 or 1).
        y_prob: Estimated return probabilities in [0.0, 1.0].
        method_name: Descriptive label for calibration technique ('raw', 'sigmoid', 'isotonic').
        n_bins: Number of equal-width probability partitions for ECE/MCE computation.

    Returns:
        Populated CalibrationMetrics model instance.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = validate_probability_bounds(y_prob, param_name="y_prob")
    n_samples = len(y_t)

    if n_samples == 0:
        return CalibrationMetrics(
            method=method_name,
            brier_score=0.0,
            log_loss=0.0,
            ece=0.0,
            mce=0.0,
            mean_predicted_probability=0.0,
            observed_positive_rate=0.0,
            bins=[],
        )

    # Calibration error curves
    bins = compute_reliability_curve(y_t, y_p, n_bins=n_bins)

    # Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    ece = 0.0
    mce = 0.0
    for b in bins:
        if b.sample_count > 0:
            weight = b.sample_count / n_samples
            ece += weight * b.absolute_error
            if b.absolute_error > mce:
                mce = b.absolute_error

    # Brier score
    brier = float(brier_score_loss(y_t, y_p))

    # Log loss / Cross-Entropy
    y_p_clipped = np.clip(y_p, 1e-7, 1.0 - 1e-7)
    try:
        ce = float(log_loss(y_t, y_p_clipped, labels=[0, 1]))
    except Exception:
        ce = 0.0

    # Mean predicted vs observed rate
    mean_p = float(np.mean(y_p))
    obs_rate = float(np.mean(y_t))

    # Ranking metrics (to check ranking preservation)
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

    return CalibrationMetrics(
        method=method_name,
        brier_score=round(brier, 4),
        log_loss=round(ce, 4),
        ece=round(ece, 4),
        mce=round(mce, 4),
        mean_predicted_probability=round(mean_p, 4),
        observed_positive_rate=round(obs_rate, 4),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        bins=bins,
    )


# =====================================================================
# Probability Calibrator Class
# =====================================================================


class ReturnProbabilityCalibrator:
    """Post-hoc probability calibrator trained on validation probabilities."""

    def __init__(
        self,
        method: Union[CalibrationMethod, str] = CalibrationMethod.SIGMOID,
        random_state: int = 42,
    ):
        if isinstance(method, CalibrationMethod):
            self.method = method
        else:
            self.method = CalibrationMethod(str(method).lower())

        self.random_state = random_state
        self.is_fitted: bool = False
        self.calibrator_: Optional[Union[LogisticRegression, IsotonicRegression]] = None
        self.training_sample_count: int = 0
        self.positive_count: int = 0

    def fit(
        self,
        y_val: Union[pd.Series, np.ndarray, Sequence[int]],
        p_val: Union[pd.Series, np.ndarray, Sequence[float]],
    ) -> "ReturnProbabilityCalibrator":
        """Fit calibration mapping strictly using validation observations.

        Args:
            y_val: Validation ground truth binary targets (0 or 1).
            p_val: Validation raw estimated probabilities in [0.0, 1.0].

        Returns:
            Fitted ReturnProbabilityCalibrator instance.
        """
        y_arr = np.asarray(y_val, dtype=int)
        p_arr = validate_probability_bounds(p_val, param_name="p_val")

        if len(y_arr) == 0:
            raise ValueError("Cannot fit calibrator on empty validation data.")
        if len(y_arr) != len(p_arr):
            raise ValueError(
                f"Validation target length ({len(y_arr)}) does not match probability length ({len(p_arr)})."
            )
        if len(np.unique(y_arr)) < 2:
            raise ValueError("Validation target y_val must contain at least two distinct classes (0 and 1).")

        self.training_sample_count = len(y_arr)
        self.positive_count = int(np.sum(y_arr == 1))

        if self.method == CalibrationMethod.NONE:
            self.calibrator_ = None
            self.is_fitted = True
            return self

        elif self.method == CalibrationMethod.SIGMOID:
            # Platt scaling: Logistic Regression on log-odds
            p_clipped = np.clip(p_arr, 1e-7, 1.0 - 1e-7)
            z_val = logit(p_clipped).reshape(-1, 1)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                clf = LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    random_state=self.random_state,
                )
                clf.fit(z_val, y_arr)

            self.calibrator_ = clf
            self.is_fitted = True
            return self

        elif self.method == CalibrationMethod.ISOTONIC:
            # Isotonic Regression: non-parametric monotonic step mapping
            iso = IsotonicRegression(
                out_of_bounds="clip",
                y_min=0.0,
                y_max=1.0,
            )
            iso.fit(p_arr, y_arr)

            self.calibrator_ = iso
            self.is_fitted = True
            return self

        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")

    def predict_proba(
        self,
        p_raw: Union[pd.Series, np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Map raw probabilities to calibrated return probabilities in [0.0, 1.0].

        Args:
            p_raw: Raw probability predictions from base ML model.

        Returns:
            1D numpy array of calibrated probabilities.
        """
        p_arr = validate_probability_bounds(p_raw, param_name="p_raw")
        if len(p_arr) == 0:
            return np.array([], dtype=float)

        if not self.is_fitted and self.method != CalibrationMethod.NONE:
            raise ValueError("Calibrator must be fitted on validation data before calling predict_proba.")

        if self.method == CalibrationMethod.NONE or self.calibrator_ is None:
            return p_arr

        elif self.method == CalibrationMethod.SIGMOID:
            p_clipped = np.clip(p_arr, 1e-7, 1.0 - 1e-7)
            z_raw = logit(p_clipped).reshape(-1, 1)
            clf: LogisticRegression = self.calibrator_  # type: ignore
            probs = clf.predict_proba(z_raw)[:, 1]
            return np.clip(probs, 0.0, 1.0)

        elif self.method == CalibrationMethod.ISOTONIC:
            iso: IsotonicRegression = self.calibrator_  # type: ignore
            probs = iso.predict(p_arr)
            return np.clip(probs, 0.0, 1.0)

        else:
            return p_arr

    def calibrate(
        self,
        p_raw: Union[pd.Series, np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Alias for predict_proba.

        Args:
            p_raw: Raw probability predictions from base ML model.

        Returns:
            1D numpy array of calibrated probabilities.
        """
        return self.predict_proba(p_raw)

    def calibrate_dataframe(
        self,
        df: pd.DataFrame,
        raw_prob_col: str = "predicted_probability",
        output_col: str = "calibrated_probability",
    ) -> pd.DataFrame:
        """Append calibrated probability column to a predictions DataFrame."""
        if df is None or df.empty:
            return pd.DataFrame()
        if raw_prob_col not in df.columns:
            raise ValueError(f"DataFrame missing raw probability column: '{raw_prob_col}'")

        df_out = df.copy()
        raw_p = df_out[raw_prob_col].values
        cal_p = self.predict_proba(raw_p)
        df_out[output_col] = np.round(cal_p, 4)
        return df_out


# =====================================================================
# Calibration Selection Engine
# =====================================================================


def evaluate_and_compare_calibrations(
    y_val: Union[pd.Series, np.ndarray, Sequence[int]],
    p_val: Union[pd.Series, np.ndarray, Sequence[float]],
    y_test: Optional[Union[pd.Series, np.ndarray, Sequence[int]]] = None,
    p_test: Optional[Union[pd.Series, np.ndarray, Sequence[float]]] = None,
    candidate_methods: Optional[List[str]] = None,
    n_bins: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit candidate calibrators on validation data and compute validation and test metrics.

    Important Data Split Guarantee:
    - All calibrators are fitted strictly on (y_val, p_val).
    - If (y_test, p_test) is provided, it is evaluated strictly out-of-sample for final reporting.

    Returns:
        Dictionary containing:
        - "calibrators": Dict of fitted calibrators by method
        - "validation_metrics": Dict of CalibrationMetrics by method
        - "test_metrics": Dict of CalibrationMetrics by method (if test provided)
        - "best_method_val": Name of candidate method achieving lowest validation Brier score
    """
    methods = candidate_methods or ["raw", "sigmoid", "isotonic"]
    calibrators: Dict[str, ReturnProbabilityCalibrator] = {}
    val_metrics: Dict[str, CalibrationMetrics] = {}
    test_metrics: Dict[str, CalibrationMetrics] = {}

    y_val_arr = np.asarray(y_val, dtype=int)
    p_val_arr = validate_probability_bounds(p_val, param_name="p_val")

    # 1. Evaluate Raw (Uncalibrated)
    if "raw" in methods or "none" in methods:
        val_metrics["raw"] = compute_calibration_metrics(
            y_true=y_val_arr,
            y_prob=p_val_arr,
            method_name="raw",
            n_bins=n_bins,
        )
        raw_cal = ReturnProbabilityCalibrator(method=CalibrationMethod.NONE)
        raw_cal.fit(y_val_arr, p_val_arr)
        calibrators["raw"] = raw_cal

        if y_test is not None and p_test is not None:
            test_metrics["raw"] = compute_calibration_metrics(
                y_true=np.asarray(y_test, dtype=int),
                y_prob=validate_probability_bounds(p_test, param_name="p_test"),
                method_name="raw",
                n_bins=n_bins,
            )

    # 2. Sigmoid Calibration
    if "sigmoid" in methods:
        sig_cal = ReturnProbabilityCalibrator(method=CalibrationMethod.SIGMOID, random_state=random_state)
        sig_cal.fit(y_val_arr, p_val_arr)
        calibrators["sigmoid"] = sig_cal

        p_val_sig = sig_cal.predict_proba(p_val_arr)
        val_metrics["sigmoid"] = compute_calibration_metrics(
            y_true=y_val_arr,
            y_prob=p_val_sig,
            method_name="sigmoid",
            n_bins=n_bins,
        )

        if y_test is not None and p_test is not None:
            p_test_sig = sig_cal.predict_proba(p_test)
            test_metrics["sigmoid"] = compute_calibration_metrics(
                y_true=np.asarray(y_test, dtype=int),
                y_prob=p_test_sig,
                method_name="sigmoid",
                n_bins=n_bins,
            )

    # 3. Isotonic Calibration
    if "isotonic" in methods:
        iso_cal = ReturnProbabilityCalibrator(method=CalibrationMethod.ISOTONIC, random_state=random_state)
        iso_cal.fit(y_val_arr, p_val_arr)
        calibrators["isotonic"] = iso_cal

        p_val_iso = iso_cal.predict_proba(p_val_arr)
        val_metrics["isotonic"] = compute_calibration_metrics(
            y_true=y_val_arr,
            y_prob=p_val_iso,
            method_name="isotonic",
            n_bins=n_bins,
        )

        if y_test is not None and p_test is not None:
            p_test_iso = iso_cal.predict_proba(p_test)
            test_metrics["isotonic"] = compute_calibration_metrics(
                y_true=np.asarray(y_test, dtype=int),
                y_prob=p_test_iso,
                method_name="isotonic",
                n_bins=n_bins,
            )

    # Determine best method strictly on validation set based on lowest Brier score
    best_method = min(
        val_metrics.keys(),
        key=lambda m: val_metrics[m].brier_score,
    )

    return {
        "calibrators": calibrators,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_method_val": best_method,
    }
