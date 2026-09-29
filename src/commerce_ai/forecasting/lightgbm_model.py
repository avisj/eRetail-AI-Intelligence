"""LightGBM Machine Learning Forecasting Model.

Implements tabular gradient boosted trees for demand forecasting leveraging
lag features, rolling statistics, calendar indicators, and categorical attributes.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

from commerce_ai.forecasting.base import (
    ForecastModel,
    ForecastOutput,
    ForecastMetadata,
)
from commerce_ai.analytics.calendar import build_calendar_features


class LightGBMForecastModel(ForecastModel):
    """Gradient Boosted Tree model for demand forecasting using LightGBM.

    Features:
        - Native handling of categorical and numerical tabular features.
        - Strict filtering on stockout-constrained periods (forecast_training_eligible).
        - Multi-step recursive forecasting when future covariates are not pre-computed.
        - Direct scoring when future features are provided.
        - Prediction intervals via empirical residual variance.
    """

    name: str = "LightGBM"
    version: str = "1.0.0"

    DEFAULT_PARAMS: Dict[str, Any] = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 1,
        "verbose": -1,
    }

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols or [
            "sku_id",
            "warehouse_id",
            "abc_class",
            "xyz_class",
            "intermittency_class",
        ]
        self.lags = lags or [1, 7, 14, 28, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30, 90]
        self.model_params = {**self.DEFAULT_PARAMS, **(params or {})}
        
        self.regressor: Optional[lgb.LGBMRegressor] = None
        self.feature_names_: List[str] = []
        self.residual_std_: float = 0.0
        self.last_date_: Optional[pd.Timestamp] = None
        self.train_history_: Optional[pd.DataFrame] = None
        self.target_col_: str = "units_sold"
        self.date_col_: str = "date"

    def _identify_candidate_features(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect feature columns available in the DataFrame."""
        if self.feature_cols is not None:
            return [col for col in self.feature_cols if col in df.columns]

        candidates = []
        # Lag features
        candidates.extend([c for c in df.columns if c.startswith("demand_lag_")])
        # Rolling features
        candidates.extend([c for c in df.columns if c.startswith("demand_rolling_")])
        # Trend features
        candidates.extend([c for c in df.columns if c.startswith("trend_") or c.startswith("demand_growth_")])
        # Calendar & temporal features
        calendar_keys = [
            "day_of_week", "day_of_month", "day_of_year", "week_of_year",
            "month", "quarter", "year", "is_weekend", "is_month_start", "is_month_end",
            "sin_day_of_year", "cos_day_of_year", "sin_day_of_week", "cos_day_of_week",
            "sin_month", "cos_month",
        ]
        candidates.extend([c for c in calendar_keys if c in df.columns])
        # Events & promotions
        event_keys = ["is_event_day", "is_promotion", "is_holiday", "is_major_event"]
        candidates.extend([c for c in event_keys if c in df.columns])
        # Categoricals
        candidates.extend([c for c in self.categorical_cols if c in df.columns])

        return sorted(list(set(candidates)))

    def fit(
        self,
        history: pd.DataFrame,
        target_col: str = "units_sold",
        date_col: str = "date",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "LightGBMForecastModel":
        """Train LightGBM regressor on historical demand features.

        Args:
            history: Feature-engineered historical DataFrame.
            target_col: Name of target demand column.
            date_col: Date column name.
            metadata: Optional additional metadata dictionary.

        Returns:
            self: Fitted LightGBMForecastModel instance.
        """
        start_time = time.perf_counter()
        self.target_col_ = target_col
        self.date_col_ = date_col
        self._extract_entity_ids(history)

        clean_history = history.copy()
        clean_history[date_col] = pd.to_datetime(clean_history[date_col])
        clean_history = clean_history.sort_values(date_col).reset_index(drop=True)

        excluded_count = 0
        if "forecast_training_eligible" in clean_history.columns:
            ineligible_mask = ~clean_history["forecast_training_eligible"].astype(bool)
            excluded_count = int(ineligible_mask.sum())
            clean_history = clean_history[~ineligible_mask].reset_index(drop=True)

        self.last_date_ = pd.to_datetime(clean_history[date_col].iloc[-1]) if not clean_history.empty else pd.Timestamp.now()
        self.train_history_ = clean_history.copy()

        # Identify features
        self.feature_names_ = self._identify_candidate_features(clean_history)
        if not self.feature_names_:
            # If no lag/calendar features exist, generate calendar features
            clean_history = build_calendar_features(clean_history, date_col=date_col)
            self.feature_names_ = self._identify_candidate_features(clean_history)

        # Drop rows where target is NaN
        valid_df = clean_history.dropna(subset=[target_col]).copy()
        if len(valid_df) < 5:
            # Fallback for ultra-short series
            self.regressor = None
            self.is_fitted = True
            self.residual_std_ = 0.0
            self._metadata = ForecastMetadata(
                model_name=self.name,
                model_version=self.version,
                training_observations=len(valid_df),
                excluded_constrained_observations=excluded_count,
                runtime_seconds=time.perf_counter() - start_time,
                status="INSUFFICIENT_HISTORY",
                error_message="Fewer than 5 valid observations to train LightGBM",
            )
            return self

        X = valid_df[self.feature_names_].copy()
        y = valid_df[target_col].astype(float).values

        # Ensure categoricals are category dtype
        actual_categoricals = []
        for cat in self.categorical_cols:
            if cat in X.columns:
                X[cat] = X[cat].astype("category")
                actual_categoricals.append(cat)

        self.regressor = lgb.LGBMRegressor(**self.model_params)
        self.regressor.fit(X, y, categorical_feature=actual_categoricals if actual_categoricals else "auto")

        # Compute empirical residual standard error for prediction intervals
        preds = np.maximum(0.0, self.regressor.predict(X))
        residuals = y - preds
        self.residual_std_ = float(np.std(residuals))

        self.is_fitted = True
        self._metadata = ForecastMetadata(
            model_name=self.name,
            model_version=self.version,
            training_observations=len(valid_df),
            excluded_constrained_observations=excluded_count,
            runtime_seconds=time.perf_counter() - start_time,
            status="SUCCESS",
        )
        return self

    def predict(
        self,
        horizon: int,
        future_features: Optional[pd.DataFrame] = None,
        confidence_level: Optional[float] = None,
    ) -> ForecastOutput:
        """Generate demand forecasts for the given horizon."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")

        future_dates = self._build_future_dates(self.last_date_, horizon)

        # Edge case: model fitted on insufficient history
        if self.regressor is None:
            mean_val = float(self.train_history_[self.target_col_].mean()) if self.train_history_ is not None and not self.train_history_.empty else 0.0
            forecasts = np.full(horizon, max(0.0, mean_val))
            return ForecastOutput(
                sku_id=self._sku_id,
                warehouse_id=self._warehouse_id,
                forecast_dates=future_dates,
                forecast_units=forecasts,
                model_name=self.name,
                model_version=self.version,
                metadata=self._metadata,
            )

        # Case 1: Pre-computed future features provided
        if future_features is not None and all(f in future_features.columns for f in self.feature_names_):
            X_future = future_features[self.feature_names_].copy()
            for cat in self.categorical_cols:
                if cat in X_future.columns:
                    X_future[cat] = X_future[cat].astype("category")
            raw_preds = self.regressor.predict(X_future[:horizon])
            forecasts = np.maximum(0.0, raw_preds)
        else:
            # Case 2: Recursive multi-step prediction
            forecasts = self._recursive_forecast(horizon, future_dates)

        # Prediction intervals
        lower_bound = None
        upper_bound = None
        if confidence_level is not None:
            conf = float(confidence_level)
            alpha = 1.0 - conf
            z_score = float(stats.norm.ppf(1.0 - alpha / 2.0))
            lower_bound = np.maximum(0.0, forecasts - z_score * self.residual_std_)
            upper_bound = np.maximum(0.0, forecasts + z_score * self.residual_std_)

        return ForecastOutput(
            sku_id=self._sku_id,
            warehouse_id=self._warehouse_id,
            forecast_dates=future_dates,
            forecast_units=forecasts,
            model_name=self.name,
            model_version=self.version,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            metadata=self._metadata,
        )

    def _recursive_forecast(self, horizon: int, future_dates: List[str]) -> np.ndarray:
        """Autoregressively generate predictions step-by-step."""
        preds: List[float] = []
        
        # Keep track of simulated demand series
        hist_df = self.train_history_.copy()
        if hist_df is None or hist_df.empty:
            return np.zeros(horizon)

        known_demands = hist_df[self.target_col_].tolist()
        
        # Build date framework for future dates
        date_df = pd.DataFrame({self.date_col_: pd.to_datetime(future_dates)})
        date_df = build_calendar_features(date_df, date_col=self.date_col_)

        for step_idx, cur_date_str in enumerate(future_dates):
            row_dict: Dict[str, Any] = {}
            cur_date = pd.to_datetime(cur_date_str)

            # 1. Categoricals
            for cat in self.categorical_cols:
                if cat in hist_df.columns:
                    row_dict[cat] = hist_df[cat].iloc[-1]

            # 2. Calendar features
            cal_row = date_df[date_df[self.date_col_] == cur_date]
            if not cal_row.empty:
                for col in cal_row.columns:
                    if col in self.feature_names_:
                        row_dict[col] = cal_row[col].iloc[0]

            # 3. Dynamic Lags
            total_demands = known_demands + preds
            for k in self.lags:
                col_name = f"demand_lag_{k}"
                if col_name in self.feature_names_:
                    if len(total_demands) >= k:
                        row_dict[col_name] = total_demands[-k]
                    else:
                        row_dict[col_name] = total_demands[0] if total_demands else 0.0

            # 4. Dynamic Rolling Means and Stds
            for w in self.rolling_windows:
                mean_col = f"demand_rolling_mean_{w}"
                std_col = f"demand_rolling_std_{w}"
                window_slice = total_demands[-w:] if len(total_demands) >= w else total_demands
                if mean_col in self.feature_names_:
                    row_dict[mean_col] = float(np.mean(window_slice)) if window_slice else 0.0
                if std_col in self.feature_names_:
                    row_dict[std_col] = float(np.std(window_slice)) if len(window_slice) > 1 else 0.0

            # 5. Trend indicators
            if "trend_7_vs_30" in self.feature_names_:
                m7 = row_dict.get("demand_rolling_mean_7", 1.0)
                m30 = row_dict.get("demand_rolling_mean_30", 1.0)
                row_dict["trend_7_vs_30"] = float(m7 / (m30 + 1e-4))
            if "trend_30_vs_90" in self.feature_names_:
                m30 = row_dict.get("demand_rolling_mean_30", 1.0)
                m90 = row_dict.get("demand_rolling_mean_90", 1.0)
                row_dict["trend_30_vs_90"] = float(m30 / (m90 + 1e-4))

            # Assemble DataFrame row
            row_df = pd.DataFrame([row_dict])
            # Ensure all feature_names exist
            for feat in self.feature_names_:
                if feat not in row_df.columns:
                    row_df[feat] = 0.0
            
            X_step = row_df[self.feature_names_].copy()
            for cat in self.categorical_cols:
                if cat in X_step.columns:
                    X_step[cat] = X_step[cat].astype("category")

            y_step = float(np.maximum(0.0, self.regressor.predict(X_step)[0]))
            preds.append(y_step)

        return np.array(preds, dtype=float)
