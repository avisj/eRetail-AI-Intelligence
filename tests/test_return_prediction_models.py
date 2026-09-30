"""Unit tests for Supervised Return Prediction ML Models (Phase 5C-2A)."""

import pytest
import numpy as np
import pandas as pd

from commerce_ai.returns.prediction_models import (
    BaseReturnRiskModel,
    LogisticRegressionReturnModel,
    LightGBMReturnModel,
    verify_feature_matrix_safety,
)
from commerce_ai.returns.schemas import (
    LogisticRegressionModelConfig,
    LightGBMModelConfig,
    ReturnModelMetadata,
)


@pytest.fixture
def sample_data():
    """Create a deterministic synthetic dataset for unit tests."""
    np.random.seed(42)
    n = 200

    X = pd.DataFrame({
        "quantity": np.random.randint(1, 10, size=n),
        "unit_price": np.random.uniform(10.0, 100.0, size=n),
        "discount": np.random.uniform(0.0, 10.0, size=n),
        "discount_rate": np.random.uniform(0.0, 0.2, size=n),
        "revenue": np.random.uniform(10.0, 90.0, size=n),
        "channel_id": np.random.choice(["CH_ONLINE", "CH_POS", "CH_MARKETPLACE"], size=n),
        "warehouse_id": np.random.choice(["WH_01", "WH_02", "WH_03"], size=n),
        "category_id": np.random.choice(["CAT_APPAREL", "CAT_ELECTRONICS"], size=n),
        "brand": np.random.choice(["BrandA", "BrandB"], size=n),
        "unit_cost": np.random.uniform(5.0, 50.0, size=n),
        "selling_price": np.random.uniform(10.0, 100.0, size=n),
        "gross_margin_rate": np.random.uniform(0.1, 0.5, size=n),
        "velocity_tier": np.random.choice(["FAST", "MED", "SLOW"], size=n),
        "day_of_week": np.random.randint(0, 7, size=n),
        "day_of_month": np.random.randint(1, 29, size=n),
        "month": np.random.randint(1, 13, size=n),
        "quarter": np.random.randint(1, 5, size=n),
        "is_weekend": np.random.choice([0, 1], size=n),
        "hist_sku_return_rate": np.random.uniform(0.0, 0.2, size=n),
        "hist_sku_sold_units": np.random.randint(0, 1000, size=n),
        "hist_sku_returned_units": np.random.randint(0, 100, size=n),
        "hist_channel_return_rate": np.random.uniform(0.05, 0.15, size=n),
        "hist_warehouse_return_rate": np.random.uniform(0.05, 0.15, size=n),
        "hist_sku_channel_return_rate": np.random.uniform(0.0, 0.2, size=n),
        "hist_sku_warehouse_return_rate": np.random.uniform(0.0, 0.2, size=n),
        "is_cold_start_sku": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "is_cold_start_channel": np.zeros(n, dtype=int),
        "is_cold_start_combination": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
    })

    # Imbalanced targets (~10% positive)
    y = np.random.choice([0, 1], size=n, p=[0.90, 0.10])
    # Ensure at least some 1s and 0s
    y[0] = 1
    y[1] = 0

    metadata = pd.DataFrame({
        "prediction_id": [f"PRED-{i:04d}" for i in range(n)],
        "sale_id": [f"S_{i:04d}" for i in range(n)],
        "order_id": [f"ORD_{i:04d}" for i in range(n)],
        "sku_id": [f"SKU_{i % 10:02d}" for i in range(n)],
        "warehouse_id": X["warehouse_id"].values,
        "channel_id": X["channel_id"].values,
        "prediction_date": ["2024-03-01"] * n,
        "target_returned": y,
    })

    return X, pd.Series(y), metadata


class TestLogisticRegressionReturnModel:
    """Test suite for Logistic Regression baseline return risk model."""

    def test_fit_and_predict_proba(self, sample_data):
        X, y, _ = sample_data
        model = LogisticRegressionReturnModel()
        model.fit(X, y)

        assert model.is_fitted
        probs = model.predict_proba(X)

        assert len(probs) == len(X)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert isinstance(probs, np.ndarray)

    def test_predict_classes_at_threshold(self, sample_data):
        X, y, _ = sample_data
        model = LogisticRegressionReturnModel()
        model.fit(X, y)

        preds_50 = model.predict(X, threshold=0.50)
        preds_10 = model.predict(X, threshold=0.10)

        assert set(np.unique(preds_50)).issubset({0, 1})
        # A lower threshold should classify at least as many positives
        assert preds_10.sum() >= preds_50.sum()

    def test_class_imbalance_configuration(self, sample_data):
        X, y, _ = sample_data
        cfg_balanced = LogisticRegressionModelConfig(class_weight="balanced")
        m_balanced = LogisticRegressionReturnModel(config=cfg_balanced)
        m_balanced.fit(X, y)

        cfg_none = LogisticRegressionModelConfig(class_weight=None)
        m_none = LogisticRegressionReturnModel(config=cfg_none)
        m_none.fit(X, y)

        assert m_balanced.metadata.class_weight_strategy == "balanced"
        assert m_none.metadata.class_weight_strategy == "None"

        # Predictions should differ due to weighting
        probs_bal = m_balanced.predict_proba(X)
        probs_none = m_none.predict_proba(X)
        assert not np.allclose(probs_bal, probs_none)

    def test_deterministic_reproducibility(self, sample_data):
        X, y, _ = sample_data
        m1 = LogisticRegressionReturnModel(config=LogisticRegressionModelConfig(random_state=42))
        m1.fit(X, y)
        p1 = m1.predict_proba(X)

        m2 = LogisticRegressionReturnModel(config=LogisticRegressionModelConfig(random_state=42))
        m2.fit(X, y)
        p2 = m2.predict_proba(X)

        np.testing.assert_allclose(p1, p2, atol=1e-6)

    def test_leakage_column_rejection(self, sample_data):
        X, y, _ = sample_data
        X_leaked = X.copy()
        X_leaked["return_date"] = "2024-03-05"

        model = LogisticRegressionReturnModel()
        with pytest.raises(ValueError, match="Target leakage detected"):
            model.fit(X_leaked, y)

    def test_predict_without_fit_raises(self, sample_data):
        X, _, _ = sample_data
        model = LogisticRegressionReturnModel()
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X)

    def test_empty_training_data_raises(self):
        model = LogisticRegressionReturnModel()
        with pytest.raises(ValueError, match="empty training data"):
            model.fit(pd.DataFrame(), pd.Series(dtype=int))

    def test_single_class_target_raises(self, sample_data):
        X, _, _ = sample_data
        y_single = pd.Series([0] * len(X))
        model = LogisticRegressionReturnModel()
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            model.fit(X, y_single)

    def test_unseen_categorical_level_handling(self, sample_data):
        X, y, _ = sample_data
        model = LogisticRegressionReturnModel()
        model.fit(X, y)

        X_novel = X.copy()
        X_novel.loc[0, "channel_id"] = "NOVEL_UNKNOWN_CHANNEL"
        probs = model.predict_proba(X_novel)
        assert 0.0 <= probs[0] <= 1.0


class TestLightGBMReturnModel:
    """Test suite for LightGBM baseline return risk model."""

    def test_fit_and_predict_proba(self, sample_data):
        X, y, _ = sample_data
        model = LightGBMReturnModel()
        model.fit(X, y)

        assert model.is_fitted
        probs = model.predict_proba(X)

        assert len(probs) == len(X)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert isinstance(probs, np.ndarray)

    def test_class_imbalance_configuration(self, sample_data):
        X, y, _ = sample_data
        cfg_bal = LightGBMModelConfig(class_weight="balanced")
        m_bal = LightGBMReturnModel(config=cfg_bal)
        m_bal.fit(X, y)

        cfg_none = LightGBMModelConfig(class_weight=None)
        m_none = LightGBMReturnModel(config=cfg_none)
        m_none.fit(X, y)

        assert m_bal.metadata.class_weight_strategy == "balanced"
        assert m_none.metadata.class_weight_strategy == "None"

    def test_deterministic_reproducibility(self, sample_data):
        X, y, _ = sample_data
        m1 = LightGBMReturnModel(config=LightGBMModelConfig(random_state=42))
        m1.fit(X, y)
        p1 = m1.predict_proba(X)

        m2 = LightGBMReturnModel(config=LightGBMModelConfig(random_state=42))
        m2.fit(X, y)
        p2 = m2.predict_proba(X)

        np.testing.assert_allclose(p1, p2, atol=1e-6)

    def test_validation_early_stopping(self, sample_data):
        X, y, _ = sample_data
        X_tr, y_tr = X.iloc[:150], y.iloc[:150]
        X_val, y_val = X.iloc[150:], y.iloc[150:]

        cfg = LightGBMModelConfig(n_estimators=50, early_stopping_rounds=5, random_state=42)
        model = LightGBMReturnModel(config=cfg)
        model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)

        assert model.is_fitted
        p = model.predict_proba(X_val)
        assert len(p) == len(X_val)

    def test_leakage_column_rejection(self, sample_data):
        X, y, _ = sample_data
        X_leaked = X.copy()
        X_leaked["target_returned"] = 1

        model = LightGBMReturnModel()
        with pytest.raises(ValueError, match="Target leakage detected"):
            model.fit(X_leaked, y)

    def test_predict_with_metadata_schema(self, sample_data):
        X, y, meta = sample_data
        model = LightGBMReturnModel()
        model.fit(X, y)

        res = model.predict_with_metadata(X, meta, threshold=0.35)

        expected_cols = [
            "prediction_id",
            "sale_id",
            "order_id",
            "sku_id",
            "warehouse_id",
            "channel_id",
            "prediction_date",
            "actual_target",
            "predicted_probability",
            "predicted_class",
            "model_name",
            "threshold",
        ]
        assert list(res.columns) == expected_cols
        assert res["model_name"].iloc[0] == "LightGBM"
        assert res["threshold"].iloc[0] == 0.35
        assert np.all(res["predicted_probability"] >= 0.0)
        assert np.all(res["predicted_probability"] <= 1.0)
        assert set(res["predicted_class"]).issubset({0, 1})

    def test_model_metadata_serialization(self, sample_data):
        X, y, _ = sample_data
        model = LightGBMReturnModel()
        model.fit(X, y)

        meta_dict = model.metadata.to_dict()
        assert meta_dict["model_name"] == "LightGBM"
        assert meta_dict["feature_count"] == len(X.columns)
        assert meta_dict["training_row_count"] == len(X)
        assert "hyperparameters" in meta_dict
