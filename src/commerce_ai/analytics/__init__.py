"""Analytics and Demand Intelligence Layer.

Provides demand reconstruction, stockout masking, ABC/XYZ classification,
calendar & business event encoding, lag/rolling features, and forecasting metrics.
"""

from commerce_ai.analytics.demand import build_daily_demand
from commerce_ai.analytics.stockout import (
    StockoutConfig,
    detect_stockout_periods,
    mask_stockout_demand,
)
from commerce_ai.analytics.abc_xyz import (
    ABCConfig,
    XYZConfig,
    calculate_abc_classification,
    calculate_xyz_classification,
    calculate_abc_xyz_matrix,
)
from commerce_ai.analytics.calendar import (
    BusinessEvent,
    build_calendar_features,
    build_event_features,
    get_standard_ecommerce_events,
)
from commerce_ai.analytics.features import (
    build_lag_features,
    build_rolling_features,
    build_trend_features,
    calculate_intermittency_metrics,
    build_demand_features,
)
from commerce_ai.analytics.metrics import (
    mae,
    rmse,
    mape,
    wape,
    bias,
    evaluate_forecast,
)

__all__ = [
    "build_daily_demand",
    "StockoutConfig",
    "detect_stockout_periods",
    "mask_stockout_demand",
    "ABCConfig",
    "XYZConfig",
    "calculate_abc_classification",
    "calculate_xyz_classification",
    "calculate_abc_xyz_matrix",
    "BusinessEvent",
    "build_calendar_features",
    "build_event_features",
    "get_standard_ecommerce_events",
    "build_lag_features",
    "build_rolling_features",
    "build_trend_features",
    "calculate_intermittency_metrics",
    "build_demand_features",
    "mae",
    "rmse",
    "mape",
    "wape",
    "bias",
    "evaluate_forecast",
]
