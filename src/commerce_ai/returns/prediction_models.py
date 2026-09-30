"""Supervised Machine Learning Return Risk Prediction Models (Phase 5C-2A).

Implements reproducible baseline classifiers for return risk probability prediction:
1. Logistic Regression:
   - Full preprocessing pipeline (numeric imputation + standard scaling, categorical one-hot encoding)
   - Configurable L2 regularization and class imbalance weighting
   - Deterministic execution with fixed random state
2. LightGBM:
   - Gradient boosted decision trees with native categorical handling
   - Configurable hyperparameters, class imbalance weighting, and optional validation early stopping
   - Deterministic execution with fixed random state

Architectural Guarantees:
- Output is RETURN PROBABILITY in [0.0, 1.0].
- Strict target leakage audits rejecting forbidden columns before training or inference.
- All preprocessing transformations fitted exclusively on training data (no future or test leakage).
- Separate model features from business identifiers and transaction metadata.
- Fully serializable model metadata documenting training lineage and hyperparameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import lightgbm as lgb
try:
    from lightgbm import LGBMClassifier, early_stopping
    from lightgbm.basic import LightGBMError
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore
    early_stopping = None  # type: ignore
    LightGBMError = Exception  # type: ignore

from commerce_ai.returns.schemas import (
    LightGBMModelConfig,
    LogisticRegressionModelConfig,
    ReturnModelMetadata,
    ReturnPredictionOutputRecord,
)
from commerce_ai.returns.prediction_dataset import FORBIDDEN_LEAKAGE_COLUMNS


# Default categorical feature names in Phase 5C-1 feature matrix
DEFAULT_CATEGORICAL_COLUMNS: Set[str] = {
    "channel_id",
    "warehouse_id",
    "category_id",
    "brand",
    "velocity_tier",
}


def verify_feature_matrix_safety(
    X: pd.DataFrame,
    forbidden_columns: Optional[Set[str]] = None,
) -> None:
    """Verify that feature matrix contains no target leakage or forbidden columns."""
    if X is None or X.empty:
        return
    forbidden = forbidden_columns or FORBIDDEN_LEAKAGE_COLUMNS
    x_cols_lower = {str(c).lower() for c in X.columns}
    violations = x_cols_lower.intersection(forbidden)
    if violations:
        raise ValueError(
            f"Target leakage detected! Feature matrix contains forbidden columns: {violations}"
        )


# =====================================================================
# Base Return Risk Model Interface
# =====================================================================


class BaseReturnRiskModel(ABC):
    """Abstract base class for supervised return-risk probability classifiers."""

    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0.0",
        random_seed: int = 42,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.random_seed = random_seed
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []
        self.categorical_features_: List[str] = []
        self.numeric_features_: List[str] = []
        self.metadata: Optional[ReturnModelMetadata] = None

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "BaseReturnRiskModel":
        """Fit the model strictly using training data."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict return probability for each observation in [0.0, 1.0]."""
        pass

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Predict binary classification (0 or 1) at a specified decision threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def predict_with_metadata(
        self,
        X: pd.DataFrame,
        metadata: pd.DataFrame,
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """Generate structured prediction output combining metadata and predicted probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating predictions.")
        if len(X) != len(metadata):
            raise ValueError(
                f"Feature matrix length ({len(X)}) does not match metadata length ({len(metadata)})."
            )

        probabilities = self.predict_proba(X)
        classes = (probabilities >= threshold).astype(int)

        preds_df = pd.DataFrame(index=X.index)
        preds_df["prediction_id"] = (
            metadata["prediction_id"].values
            if "prediction_id" in metadata.columns
            else [f"PRED-{i}" for i in range(len(X))]
        )
        preds_df["sale_id"] = (
            metadata["sale_id"].astype(str).values
            if "sale_id" in metadata.columns
            else preds_df["prediction_id"].values
        )
        preds_df["order_id"] = (
            metadata["order_id"].astype(str).values
            if "order_id" in metadata.columns
            else "UNKNOWN"
        )
        preds_df["sku_id"] = (
            metadata["sku_id"].astype(str).values
            if "sku_id" in metadata.columns
            else (X["sku_id"].astype(str).values if "sku_id" in X.columns else "UNKNOWN")
        )
        preds_df["warehouse_id"] = (
            metadata["warehouse_id"].astype(str).values
            if "warehouse_id" in metadata.columns
            else (X["warehouse_id"].astype(str).values if "warehouse_id" in X.columns else "UNKNOWN")
        )
        preds_df["channel_id"] = (
            metadata["channel_id"].astype(str).values
            if "channel_id" in metadata.columns
            else (X["channel_id"].astype(str).values if "channel_id" in X.columns else "UNKNOWN")
        )
        preds_df["prediction_date"] = (
            metadata["prediction_date"].astype(str).values
            if "prediction_date" in metadata.columns
            else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )

        if "target_returned" in metadata.columns:
            preds_df["actual_target"] = pd.to_numeric(metadata["target_returned"], errors="coerce").fillna(0).astype(int).values
        elif "actual_target" in metadata.columns:
            preds_df["actual_target"] = pd.to_numeric(metadata["actual_target"], errors="coerce").fillna(0).astype(int).values
        else:
            preds_df["actual_target"] = None

        preds_df["predicted_probability"] = np.round(probabilities, 4)
        preds_df["predicted_class"] = classes
        preds_df["model_name"] = self.model_name
        preds_df["threshold"] = round(float(threshold), 4)

        return preds_df

    def _partition_features(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify categorical vs numeric feature subsets."""
        cat_cols = []
        num_cols = []
        for c in X.columns:
            if c in DEFAULT_CATEGORICAL_COLUMNS or X[c].dtype == "object" or isinstance(X[c].dtype, pd.CategoricalDtype):
                cat_cols.append(c)
            else:
                num_cols.append(c)
        return cat_cols, num_cols


# =====================================================================
# Logistic Regression Baseline Model
# =====================================================================


class LogisticRegressionReturnModel(BaseReturnRiskModel):
    """Logistic Regression baseline classifier for return-risk estimation."""

    def __init__(
        self,
        config: Optional[LogisticRegressionModelConfig] = None,
        model_version: str = "1.0.0",
    ):
        cfg = config or LogisticRegressionModelConfig()
        super().__init__(
            model_name="LogisticRegression",
            model_version=model_version,
            random_seed=cfg.random_state,
        )
        self.config = cfg
        self.pipeline_: Optional[Pipeline] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "LogisticRegressionReturnModel":
        """Fit preprocessing pipeline and Logistic Regression on training data."""
        if X_train is None or X_train.empty:
            raise ValueError("Cannot train LogisticRegressionReturnModel on empty training data.")

        verify_feature_matrix_safety(X_train)
        if X_val is not None:
            verify_feature_matrix_safety(X_val)

        y_train_arr = np.asarray(y_train).astype(int)
        if len(np.unique(y_train_arr)) < 2:
            raise ValueError("Training target y_train must contain at least 2 distinct classes (0 and 1).")

        self.feature_names_ = list(X_train.columns)
        self.categorical_features_, self.numeric_features_ = self._partition_features(X_train)

        # Build column transformer
        transformers = []
        if self.numeric_features_:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, self.numeric_features_))

        if self.categorical_features_:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipe, self.categorical_features_))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            clf_kwargs: Dict[str, Any] = {
                "C": self.config.C,
                "solver": self.config.solver,
                "max_iter": self.config.max_iter,
                "class_weight": self.config.class_weight,
                "random_state": self.config.random_state,
            }
            if self.config.penalty != "l2":
                clf_kwargs["penalty"] = self.config.penalty

            clf = LogisticRegression(**clf_kwargs)
            self.pipeline_ = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ])
            self.pipeline_.fit(X_train, y_train_arr)

        self.is_fitted = True

        # Build metadata
        pos_cnt = int((y_train_arr == 1).sum())
        neg_cnt = int((y_train_arr == 0).sum())
        val_rows = len(X_val) if X_val is not None else 0
        self.metadata = ReturnModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            training_timestamp=datetime.now(timezone.utc).isoformat(),
            feature_names=self.feature_names_,
            feature_count=len(self.feature_names_),
            training_row_count=len(X_train),
            validation_row_count=val_rows,
            test_row_count=0,
            positive_count=pos_cnt,
            negative_count=neg_cnt,
            class_weight_strategy=str(self.config.class_weight),
            hyperparameters=self.config.to_dict(),
            random_seed=self.random_seed,
            threshold_used=0.50,
        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict return probability in [0.0, 1.0]."""
        if not self.is_fitted or self.pipeline_ is None:
            raise ValueError("Model must be fitted before calling predict_proba.")
        verify_feature_matrix_safety(X)

        # Check required columns
        missing_cols = set(self.feature_names_) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Input feature matrix is missing required features: {missing_cols}")

        probs = self.pipeline_.predict_proba(X)
        return probs[:, 1]


# =====================================================================
# LightGBM Baseline Model
# =====================================================================


class LightGBMReturnModel(BaseReturnRiskModel):
    """LightGBM gradient boosted decision tree classifier for return-risk estimation."""

    def __init__(
        self,
        config: Optional[LightGBMModelConfig] = None,
        model_version: str = "1.0.0",
    ):
        if LGBMClassifier is None:  # pragma: no cover
            raise ImportError("lightgbm is required for LightGBMReturnModel but is not installed.")
        cfg = config or LightGBMModelConfig()
        super().__init__(
            model_name="LightGBM",
            model_version=model_version,
            random_seed=cfg.random_state,
        )
        self.config = cfg
        self.ordinal_encoder_: Optional[OrdinalEncoder] = None
        self.classifier_: Optional[LGBMClassifier] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "LightGBMReturnModel":
        """Fit ordinal encoder on categoricals and LightGBM classifier on training data."""
        if X_train is None or X_train.empty:
            raise ValueError("Cannot train LightGBMReturnModel on empty training data.")

        verify_feature_matrix_safety(X_train)
        if X_val is not None:
            verify_feature_matrix_safety(X_val)

        y_train_arr = np.asarray(y_train).astype(int)
        if len(np.unique(y_train_arr)) < 2:
            raise ValueError("Training target y_train must contain at least 2 distinct classes (0 and 1).")

        self.feature_names_ = list(X_train.columns)
        self.categorical_features_, self.numeric_features_ = self._partition_features(X_train)

        # Prepare categorical encoding
        X_tr_proc = X_train.copy()
        if self.categorical_features_:
            self.ordinal_encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32,
            )
            # Fit encoder strictly on X_train
            X_tr_proc[self.categorical_features_] = self.ordinal_encoder_.fit_transform(
                X_train[self.categorical_features_].astype(str)
            )

        self.classifier_ = LGBMClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_samples=self.config.min_child_samples,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
        )

        callbacks = []
        eval_set = None
        if (
            self.config.early_stopping_rounds is not None
            and X_val is not None
            and y_val is not None
            and not X_val.empty
            and early_stopping is not None
        ):
            X_val_proc = X_val.copy()
            if self.categorical_features_ and self.ordinal_encoder_ is not None:
                X_val_proc[self.categorical_features_] = self.ordinal_encoder_.transform(
                    X_val[self.categorical_features_].astype(str)
                )
            y_val_arr = np.asarray(y_val).astype(int)
            eval_set = [(X_val_proc, y_val_arr)]
            callbacks.append(early_stopping(stopping_rounds=self.config.early_stopping_rounds, verbose=False))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if eval_set:
                self.classifier_.fit(
                    X_tr_proc,
                    y_train_arr,
                    eval_set=eval_set,
                    categorical_feature=self.categorical_features_ if self.categorical_features_ else "auto",
                    callbacks=callbacks,
                )
            else:
                self.classifier_.fit(
                    X_tr_proc,
                    y_train_arr,
                    categorical_feature=self.categorical_features_ if self.categorical_features_ else "auto",
                )

        self.is_fitted = True

        # Build metadata
        pos_cnt = int((y_train_arr == 1).sum())
        neg_cnt = int((y_train_arr == 0).sum())
        val_rows = len(X_val) if X_val is not None else 0
        self.metadata = ReturnModelMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            training_timestamp=datetime.now(timezone.utc).isoformat(),
            feature_names=self.feature_names_,
            feature_count=len(self.feature_names_),
            training_row_count=len(X_train),
            validation_row_count=val_rows,
            test_row_count=0,
            positive_count=pos_cnt,
            negative_count=neg_cnt,
            class_weight_strategy=str(self.config.class_weight),
            hyperparameters=self.config.to_dict(),
            random_seed=self.random_seed,
            threshold_used=0.50,
        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict return probability in [0.0, 1.0]."""
        if not self.is_fitted or self.classifier_ is None:
            raise ValueError("Model must be fitted before calling predict_proba.")
        verify_feature_matrix_safety(X)

        missing_cols = set(self.feature_names_) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Input feature matrix is missing required features: {missing_cols}")

        X_proc = X[self.feature_names_].copy()
        if self.categorical_features_ and self.ordinal_encoder_ is not None:
            X_proc[self.categorical_features_] = self.ordinal_encoder_.transform(
                X_proc[self.categorical_features_].astype(str)
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            probs = self.classifier_.predict_proba(X_proc)
        return probs[:, 1]
