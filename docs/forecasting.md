# Forecasting Engine Architecture & Model Catalog

## 1. Overview & Core Philosophy

The **Forecasting Engine** in the **eRetail AI Intelligence Platform** is responsible for transforming historical omnichannel demand signals, inventory availability, and promotional drivers into accurate out-of-sample forward projections.

### Foundational Principles:
1. **No Assumed Winner ("Compete Fairly")**: High-capacity deep learning and foundation models (e.g. Google TimesFM-3) must compete on equal terms against classical statistical baselines (Seasonal Naive, Moving Average, Holt-Winters, Croston) and tabular ML (LightGBM). Models are chosen strictly on out-of-sample performance, bias, latency, and operational stability.
2. **Strict Leakage Protection**: All feature pipelines, temporal aggregations, and rolling windows enforce historical cutoff dates. At time $T$, no signal beyond $T-1$ is ever utilized.
3. **Physical Retail Bounds**: Forecasted demand must never be negative ($\hat{y} \ge 0$).
4. **Demand ≠ Sales (Stockout Masking)**: The engine trains on **unconstrained customer demand** by filtering out periods where inventory stockouts artificially suppressed sales (`forecast_training_eligible == True`).

---

## 2. Universal Forecasting Interface (`ForecastModel`)

All forecasting algorithms adhere to an abstract base contract defined in `commerce_ai.forecasting.base`:

```python
class ForecastModel(ABC):
    @abstractmethod
    def fit(
        self,
        history: pd.DataFrame,
        target_col: str = "units_sold",
        date_col: str = "date",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ForecastModel": ...

    @abstractmethod
    def predict(
        self,
        horizon: int,
        future_features: Optional[pd.DataFrame] = None,
        confidence_level: Optional[float] = None,
    ) -> ForecastOutput: ...
```

### Standardized Output Contracts

- **`ForecastRecord`**: Atomic prediction entity representing a single SKU $\times$ Warehouse $\times$ Date tuple, complete with point forecast, confidence bounds, model version, and audit run ID.
- **`ForecastOutput`**: High-level result container holding vector forecasts, future date series, prediction intervals, and metadata. Provides `.to_records()` and `.to_dataframe()`.
- **`ForecastMetadata`**: Detailed operational telemetry tracking training observation count, constrained records excluded, model status (`SUCCESS`, `INSUFFICIENT_HISTORY`, `TIMESFM_UNAVAILABLE`, `FALLBACK`), runtime in seconds, and error codes.

---

## 3. Model Catalog

| Model | Class | Target Pattern | Mechanism | Prediction Intervals |
| :--- | :--- | :--- | :--- | :--- |
| **Naive** | `NaiveModel` | Benchmark | Forward projects last eligible observation: $\hat{y}_{t+h} = y_T$ | None |
| **Seasonal Naive** | `SeasonalNaiveModel` | Weekly retail cycles | Cyclical projection matching day of week: $\hat{y}_{t+h} = y_{T+h-7k}$ | Empirical variance |
| **Moving Average** | `MovingAverageModel` | Slow trend / noise | Rolling arithmetic mean over 7, 14, or 30 days | Empirical std |
| **Exponential Smoothing** | `ExponentialSmoothingModel` | Level, trend, season | Statsmodels Holt-Winters / Simple Exp Smoothing with level updates | Residual variance $z$-score |
| **Croston & SBA** | `CrostonModel` | Intermittent / Lumpy | Decouples non-zero transaction size ($z$) from inter-arrival intervals ($p$); applies SBA factor $(1 - \frac{\alpha}{2})\frac{z}{p}$ | Quantile bootstrap |
| **LightGBM** | `LightGBMForecastModel` | Multi-variate / complex | Gradient boosted decision trees using lags (1, 7, 14, 28, 30), rolling stats, trends, calendar flags, and categoricals | Empirical residual error |
| **TimesFM Adapter** | `TimesFMForecastModel` | Zero-shot foundation | Google TimesFM transformer pretrained on 100B+ time-series points | Model quantiles |

---

## 4. LightGBM Tabular Forecasting Architecture

The `LightGBMForecastModel` utilizes features engineered by Phase 2:
- **Lags**: $T-1, T-7, T-14, T-28, T-30$
- **Rolling Windows**: 7, 14, 30, and 90-day causal rolling mean and standard deviation (shifted by 1 day to prevent leakage).
- **Momentum Indicators**: $\text{trend}_{7/30}$, $\text{trend}_{30/90}$, 7-day growth, 30-day growth.
- **Calendar & Cyclical Signals**: Sine/cosine encoding of day of year, day of week, month, and holiday/promotion flags.
- **Categoricals**: `sku_id`, `warehouse_id`, `abc_class`, `xyz_class`, `intermittency_class`.

When forecasting out-of-sample without precomputed future lags, the model performs **autoregressive multi-step recursive forecasting**:
1. It updates calendar and date signals for $T+1$.
2. It recomputes lags and rolling statistics dynamically from historical records and prior step predictions.
3. It predicts step $T+1$, clips at $\ge 0$, appends to the series, and advances to step $T+2$ until horizon $H$ is satisfied.

---

## 5. Google TimesFM Foundation Model Adapter

The platform integrates Google TimesFM (Time Series Foundation Model) via `TimesFMForecastModel`. 

### Runtime Environment Discovery & Fallback
Google TimesFM v1.x depends on `paxml==1.4.0` and JAX TPU/GPU binaries which are only compiled for Linux x86_64 and Google Cloud TPUs. In environments where native dependencies are absent (such as Darwin / macOS arm64):
- The model detects dependency absence during instantiation (`_check_availability()`).
- It flags status as `TIMESFM_UNAVAILABLE` with error code `TIMESFM_DEPENDENCY_MISSING`.
- It does **not** fail the backtest or crash the pipeline.
- It provides a `mock_predictor` injection hook for full unit and integration test coverage.
- **Zero Fabrication**: If TimesFM cannot execute in the local runtime, benchmark tables truthfully mark it as `UNAVAILABLE` without manufacturing synthetic accuracy numbers.

---

## 6. High-Level `ForecastService`

The `ForecastService` orchestrates production inference:
- **`forecast_series(...)`**: Scores an individual series with validation, non-negative clipping, and audit metadata.
- **`forecast_batch(...)`**: Scans multi-series datasets, routes each SKU to its designated champion model (via `ModelSelectionPolicy`), and handles graceful baseline fallback if an advanced model encounters numerical anomalies.
