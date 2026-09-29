# Automated Forecasting Model Selection & Routing Policy

## 1. Overview & Business Objectives

In an enterprise ecommerce environment managing tens of thousands of SKUs across multiple fulfillment centers, **no single forecasting model dominates across all time series**.
- A complex gradient-boosted tree (LightGBM) often excels on high-velocity items with rich seasonal and promotional covariates.
- Classical Croston / SBA typically beats machine learning on sporadic, intermittent tail items where sparse transactions cause regression trees to overfit.
- A 7-day Seasonal Naive baseline frequently matches or exceeds complex models on stable consumer staples, at a tiny fraction of the computational and maintenance cost.

The **Model Selection Engine** (`commerce_ai.forecasting.selection`) automates the objective selection of champion models based on backtested performance, operational constraints, and demand segmentation.

---

## 2. Selection Framework & Criteria

The `ModelSelector` evaluates the standardized master benchmark table using configurable parameters defined in `SelectionCriteria`:

```python
@dataclass
class SelectionCriteria:
    primary_metric: str = "WAPE (%)"
    secondary_metric: str = "Bias (%)"
    max_failure_rate: float = 0.05
    max_runtime_seconds: Optional[float] = None
    prefer_simpler_baseline_if_within_pct: float = 2.0  # Parsimony margin (%)
```

### 2.1 The Principle of Parsimony (Occam's Razor for ML)
A common pitfall in production time-series platforms is deploying complex ML architectures for negligible marginal gains. If LightGBM achieves a $14.5\%$ WAPE while a 7-day Seasonal Naive achieves $15.2\%$ ($+0.7\%$ difference), the marginal accuracy improvement often fails to justify the added operational overhead (feature engineering pipelines, retraining pipelines, inference latency, drift monitoring).

The `prefer_simpler_baseline_if_within_pct` parameter implements this parsimony principle: if any robust baseline (Seasonal Naive, Moving Average, Exponential Smoothing) performs within the configured margin of the top model, the baseline is selected as the champion.

---

## 3. Segment-Level Champion Routing (`ModelSelectionPolicy`)

Instead of forcing a single global model onto diverse inventory categories, the selector builds a **hierarchical routing policy**:

```
[ Inbound Series: SKU_042 @ WH_01 ]
                │
                ▼
   Is SKU explicitly overridden? ──(Yes)──► Use SKU Champion
                │ (No)
                ▼
  Check Demand Segment Category
        ┌───────┼───────────────┐
        ▼       ▼               ▼
     Smooth   Intermittent    Erratic / Lumpy
        │       │               │
        ▼       ▼               ▼
    LightGBM  Croston (SBA)   Exp Smoothing / MA
```

### Segment Champion Rules:
1. **Smooth Demand (High Occurrence, Low CV)**:
   - Primary: `LightGBM` or `Seasonal Naive`
   - Justification: Regular daily cadence and seasonality benefit from multi-lag regression and day-of-week feature modeling.
2. **Intermittent Demand (Low Occurrence, High Zero-Ratio)**:
   - Primary: `Croston (SBA)`
   - Justification: Decouples demand size from arrival interval; deflates positive bias typical of standard regression on sparse zeros.
3. **Erratic Demand (High Occurrence, High CV)**:
   - Primary: `Moving Average (14d/30d)` or `Exponential Smoothing`
   - Justification: Dampens extreme day-to-day transaction volatility without chasing noisy spikes.
4. **Lumpy Demand (Low Occurrence, High CV)**:
   - Primary: `Croston (SBA)` with conservative smoothing parameters.

---

## 4. Integration with `ForecastService`

During production execution, `ForecastService.forecast_batch()` takes a `ModelSelectionPolicy`:

```python
service = ForecastService()
policy = selector.build_segment_policy(
    global_benchmark=benchmark_table,
    segment_eval_df=segment_evaluations,
    segment_col="intermittency_class",
)

# Batch forecast with automated model routing
forecast_records = service.forecast_batch(
    dataset=demand_features_df,
    horizon=30,
    policy=policy,
)
```

Each series is inspected at runtime:
1. Entity segment is extracted (`intermittency_class`).
2. The series is routed to its designated champion model.
3. If the champion model raises a numerical exception, `ForecastService` automatically intercepts the failure, logs the issue, and smoothly falls back to `NaiveModel` to guarantee unblocked downstream replenishment planning.
