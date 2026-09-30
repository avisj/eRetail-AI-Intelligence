"""Evaluation script for Phase 5C-2B Return Probability Calibration & Risk Layer.

Trains Logistic Regression and LightGBM models on canonical Train partition,
fits Sigmoid and Isotonic calibrators strictly on Validation partition,
and evaluates raw vs calibrated probabilities and risk bands on Holdout Test partition.
"""

import json
import time
import numpy as np
import pandas as pd

from commerce_ai.returns.prediction_dataset import ReturnPredictionDatasetBuilder
from commerce_ai.returns.prediction_models import (
    LogisticRegressionReturnModel,
    LightGBMReturnModel,
)
from commerce_ai.returns.schemas import (
    LogisticRegressionModelConfig,
    LightGBMModelConfig,
    RiskPolicyConfig,
)
from commerce_ai.returns.calibration import (
    ReturnProbabilityCalibrator,
    compute_calibration_metrics,
    evaluate_and_compare_calibrations,
)
from commerce_ai.returns.risk import (
    ReturnRiskClassifier,
    evaluate_risk_policy,
    evaluate_segment_calibration,
    risk_evaluations_to_dataframe,
)


def run_full_calibration_experiment():
    print("=" * 80)
    print("STARTING FULL CALIBRATION & RISK EVALUATION EXPERIMENT (PHASE 5C-2B)")
    print("=" * 80)

    # 1. Load Data
    t0 = time.time()
    print("Loading data from data/sample/...")
    sales = pd.read_csv("data/sample/sales.csv")
    returns = pd.read_csv("data/sample/returns.csv")
    products = pd.read_csv("data/sample/products.csv")
    warehouses = pd.read_csv("data/sample/warehouses.csv")
    channels = pd.read_csv("data/sample/channels.csv")
    print(f"Data loaded in {time.time() - t0:.2f}s.")

    # 2. Build Dataset
    t1 = time.time()
    print("Building canonical ReturnPredictionDataset...")
    builder = ReturnPredictionDatasetBuilder()
    ds = builder.build(
        sales=sales,
        returns=returns,
        products=products,
        warehouses=warehouses,
        channels=channels,
    )
    print(f"Dataset built in {time.time() - t1:.2f}s: X={ds.X.shape}, y={len(ds.y)}")

    # Extract splits
    train_idx = ds.train_indices
    val_idx = ds.val_indices
    test_idx = ds.test_indices

    X_train, y_train = ds.X.iloc[train_idx], ds.y.iloc[train_idx]
    X_val, y_val = ds.X.iloc[val_idx], ds.y.iloc[val_idx]
    X_test, y_test = ds.X.iloc[test_idx], ds.y.iloc[test_idx]

    meta_val = ds.metadata.iloc[val_idx].copy()
    meta_test = ds.metadata.iloc[test_idx].copy()

    # Add features used for slicing
    for col in ["channel_id", "warehouse_id", "velocity_tier", "is_cold_start_sku"]:
        meta_test[col] = X_test[col].values

    print(f"Split sizes: Train={len(X_train)} (pos={y_train.sum()} ({y_train.mean():.4f})), "
          f"Val={len(X_val)} (pos={y_val.sum()} ({y_val.mean():.4f})), "
          f"Test={len(X_test)} (pos={y_test.sum()} ({y_test.mean():.4f}))")

    results = {}

    # Define models to benchmark
    models = {
        "LogisticRegression": LogisticRegressionReturnModel(
            config=LogisticRegressionModelConfig(
                class_weight="balanced",
                max_iter=500,
                random_state=42,
            )
        ),
        "LightGBM": LightGBMReturnModel(
            config=LightGBMModelConfig(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                class_weight="balanced",
                random_state=42,
                early_stopping_rounds=10,
            )
        ),
    }

    classifier = ReturnRiskClassifier()

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"EVALUATING MODEL: {name}")
        print("=" * 60)

        # Train model
        tm0 = time.time()
        print(f"Fitting {name} on Train split...")
        if name == "LightGBM":
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)
        print(f"Fitted {name} in {time.time() - tm0:.2f}s.")

        # Predict raw probabilities
        p_val_raw = model.predict_proba(X_val)
        p_test_raw = model.predict_proba(X_test)

        # Fit Calibrators strictly on Validation
        print("Fitting Sigmoid and Isotonic calibrators strictly on Validation split...")
        cal_sigmoid = ReturnProbabilityCalibrator(method="sigmoid", random_state=42)
        cal_sigmoid.fit(y_val, p_val_raw)

        cal_isotonic = ReturnProbabilityCalibrator(method="isotonic", random_state=42)
        cal_isotonic.fit(y_val, p_val_raw)

        # Calibrate
        p_val_sig = cal_sigmoid.predict_proba(p_val_raw)
        p_test_sig = cal_sigmoid.predict_proba(p_test_raw)

        p_val_iso = cal_isotonic.predict_proba(p_val_raw)
        p_test_iso = cal_isotonic.predict_proba(p_test_raw)

        # Metrics computation
        metrics_val_raw = compute_calibration_metrics(y_val, p_val_raw, method_name="raw")
        metrics_val_sig = compute_calibration_metrics(y_val, p_val_sig, method_name="sigmoid")
        metrics_val_iso = compute_calibration_metrics(y_val, p_val_iso, method_name="isotonic")

        metrics_test_raw = compute_calibration_metrics(y_test, p_test_raw, method_name="raw")
        metrics_test_sig = compute_calibration_metrics(y_test, p_test_sig, method_name="sigmoid")
        metrics_test_iso = compute_calibration_metrics(y_test, p_test_iso, method_name="isotonic")

        # Format comparison table
        print("\n--- VALIDATION SPLIT CALIBRATION METRICS ---")
        val_summary = pd.DataFrame([
            metrics_val_raw.to_dict(),
            metrics_val_sig.to_dict(),
            metrics_val_iso.to_dict(),
        ])[["method", "brier_score", "log_loss", "ece", "mce", "mean_predicted_probability", "observed_positive_rate", "roc_auc", "pr_auc"]]
        print(val_summary.to_string(index=False))

        print("\n--- HOLDOUT TEST SPLIT CALIBRATION METRICS ---")
        test_summary = pd.DataFrame([
            metrics_test_raw.to_dict(),
            metrics_test_sig.to_dict(),
            metrics_test_iso.to_dict(),
        ])[["method", "brier_score", "log_loss", "ece", "mce", "mean_predicted_probability", "observed_positive_rate", "roc_auc", "pr_auc"]]
        print(test_summary.to_string(index=False))

        # Risk band evaluation on holdout test using best calibration method on validation (lowest Brier score)
        best_val_method = "sigmoid" if metrics_val_sig.brier_score <= metrics_val_iso.brier_score else "isotonic"
        best_test_cal_probs = p_test_sig if best_val_method == "sigmoid" else p_test_iso
        print(f"\nBest calibration method on Validation: {best_val_method}")

        df_test_eval = meta_test.copy()
        df_test_eval["raw_probability"] = p_test_raw
        df_test_eval["calibrated_probability"] = best_test_cal_probs
        df_test_eval["actual_target"] = y_test.values

        classified_test = classifier.classify_dataframe(
            df_test_eval,
            raw_prob_col="raw_probability",
            cal_prob_col="calibrated_probability",
            model_name=name,
            calibration_method=best_val_method,
        )

        risk_band_evals = evaluate_risk_policy(classified_test)
        df_risk_eval = risk_evaluations_to_dataframe(risk_band_evals)
        print(f"\n--- EMPIRICAL RISK BAND POLICY AUDIT (TEST SET - {name} with {best_val_method}) ---")
        print(df_risk_eval.to_string(index=False))

        # Segment Calibration on Holdout Test
        seg_eval = evaluate_segment_calibration(
            classified_test,
            dimensions=["channel_id", "warehouse_id", "velocity_tier", "is_cold_start_sku"],
            min_sample_size=30,
        )
        print(f"\n--- SEGMENT CALIBRATION AUDIT (TEST SET - {name}) ---")
        print(seg_eval[["dimension", "segment_value", "sample_size", "positive_rate", "mean_calibrated_probability", "calibration_error", "brier_score", "ece", "is_sufficient_sample"]].to_string(index=False))

        results[name] = {
            "validation_metrics": {
                "raw": metrics_val_raw.to_dict(),
                "sigmoid": metrics_val_sig.to_dict(),
                "isotonic": metrics_val_iso.to_dict(),
            },
            "test_metrics": {
                "raw": metrics_test_raw.to_dict(),
                "sigmoid": metrics_test_sig.to_dict(),
                "isotonic": metrics_test_iso.to_dict(),
            },
            "best_validation_method": best_val_method,
            "risk_band_evaluation": [e.to_dict() for e in risk_band_evals],
            "segment_calibration": seg_eval.to_dict(orient="records"),
            "raw_test_bins": [b.to_dict() for b in metrics_test_raw.bins],
            "sigmoid_test_bins": [b.to_dict() for b in metrics_test_sig.bins],
            "isotonic_test_bins": [b.to_dict() for b in metrics_test_iso.bins],
        }

    # Save results to JSON
    with open("scripts/calibration_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults successfully saved to scripts/calibration_experiment_results.json")
    print("=" * 80)


if __name__ == "__main__":
    run_full_calibration_experiment()
