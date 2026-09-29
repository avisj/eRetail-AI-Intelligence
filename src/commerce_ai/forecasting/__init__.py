"""Forecasting Module for eRetail AI Intelligence Platform.

Exposes base interfaces, statistical and machine learning models,
backtesting, evaluation, selection, and the unified ForecastService.
"""

from commerce_ai.forecasting.base import (
    ForecastModel,
    ForecastRecord,
    ForecastMetadata,
    ForecastOutput,
)
from commerce_ai.forecasting.config import ForecastConfig, BacktestConfig
from commerce_ai.forecasting.datasets import (
    build_forecast_dataset,
    time_series_train_test_split,
    generate_rolling_origin_folds,
)
from commerce_ai.forecasting.baselines import (
    NaiveModel,
    SeasonalNaiveModel,
    MovingAverageModel,
)
from commerce_ai.forecasting.exponential_smoothing import ExponentialSmoothingModel
from commerce_ai.forecasting.croston import CrostonModel
from commerce_ai.forecasting.lightgbm_model import LightGBMForecastModel
from commerce_ai.forecasting.timesfm import TimesFMForecastModel
from commerce_ai.forecasting.backtesting import RollingOriginBacktester, BacktestResult
from commerce_ai.forecasting.evaluation import ForecastEvaluator
from commerce_ai.forecasting.selection import (
    ModelSelector,
    SelectionCriteria,
    ModelSelectionPolicy,
)
from commerce_ai.forecasting.service import ForecastService

__all__ = [
    "ForecastModel",
    "ForecastRecord",
    "ForecastMetadata",
    "ForecastOutput",
    "ForecastConfig",
    "BacktestConfig",
    "build_forecast_dataset",
    "time_series_train_test_split",
    "generate_rolling_origin_folds",
    "NaiveModel",
    "SeasonalNaiveModel",
    "MovingAverageModel",
    "ExponentialSmoothingModel",
    "CrostonModel",
    "LightGBMForecastModel",
    "TimesFMForecastModel",
    "RollingOriginBacktester",
    "BacktestResult",
    "ForecastEvaluator",
    "ModelSelector",
    "SelectionCriteria",
    "ModelSelectionPolicy",
    "ForecastService",
]
