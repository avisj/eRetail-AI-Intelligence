# Time-Series Model Evaluation & Hierarchical Backtesting

## 1. Overview & Evaluation Principles

Evaluating demand forecasting in ecommerce requires metrics that directly correlate with supply chain costs: stockouts (lost revenue) versus overstocks (holding cost and obsolescence). Standard machine learning metrics (such as MSE or unweighted MAPE) distort retail performance because they either penalize outlier volume disproportionately or divide by zero on intermittent demand.

The platform implements **Rolling-Origin Time-Series Cross-Validation** and a **Hierarchical Evaluation Engine** adhering to the following rules:
1. **Strict Temporal Causality**: No observation from future periods may leak into training, scaling, or feature engineering.
2. **Volume-Weighted Assessment**: WAPE is preferred over MAPE for inventory decision-making.
3. **Directional Risk Tracking**: Forecast Bias is monitored alongside accuracy to detect systematic inventory build-up or depletion.
4. **Resilient Backtesting**: Individual series or fold failures are isolated and logged without corrupting global benchmarks.

---

## 2. Core Evaluation Metrics

### 2.1 WAPE (Weighted Absolute Percentage Error)
The primary supply chain accuracy KPI:
$$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N |y_i|} \times 100\%$$
- **Why it matters**: Weights error by sales volume. A 10-unit error on a product selling 1,000 units matters less than a 10-unit error on a product selling 12 units.
- **Zero-Demand Safety**: If $\sum |y_i| = 0$, the metric safely returns $0.0\%$ if $\sum |\hat{y}_i| = 0$, preventing undefined operations.

### 2.2 Forecast Bias Percentage
Measures systematic over-forecasting or under-forecasting:
$$\text{Bias} = \frac{\sum_{i=1}^N (\hat{y}_i - y_i)}{\sum_{i=1}^N |y_i|} \times 100\%$$
- **$\text{Bias} > 0$ (Over-forecasting)**: Leads to excess inventory, bloated working capital, and storage holding costs.
- **$\text{Bias} < 0$ (Under-forecasting)**: Leads to stockouts, lost sales, and degraded customer SLA.
- **$\text{Bias} \approx 0$**: Balanced, unbiased forecast.

### 2.3 MAE & RMSE
- **MAE** ($\frac{1}{N}\sum |y_i - \hat{y}_i|$): Measures average unit magnitude of error in linear scale.
- **RMSE** ($\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$): Sensitive to large forecasting blowups, critical for safety stock calculation.

### 2.4 Prediction Interval Coverage
$$\text{Coverage} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(L_i \le y_i \le U_i) \times 100\%$$
Validates whether empirical $95\%$ confidence bounds actually contain the observed demand $95\%$ of the time.

---

## 3. Rolling-Origin Backtesting Architecture

Standard $k$-fold cross validation violates temporal causality. Instead, the platform uses **Rolling-Origin Cross-Validation** (Walk-Forward Validation):

```
Fold 1: [--- Min History ---] -> [ Forecast Window H ]
Fold 2: [--- Min History + Step ---] -> [ Forecast Window H ]
Fold 3: [--- Min History + 2*Step ---] -> [ Forecast Window H ]
```

### `RollingOriginBacktester` Workflow:
1. Slices the historical dataset into $K$ chronologically expanding training windows up to cutoff date $T_k$.
2. Generates out-of-sample test targets over $[T_k + 1, T_k + H]$.
3. Fits each candidate model strictly on data $\le T_k$.
4. Collects predictions and joins with unconstrained actual observations.
5. Captures granular execution diagnostics: runtime in seconds, exceptions, failure status, and parameter states.

---

## 4. Multi-Level Hierarchical Evaluation

The `ForecastEvaluator` analyzes forecast errors across retail hierarchies:
- **Global**: Overall platform error across all SKUs, warehouses, and folds.
- **Per Fold**: Temporal stability and error drift across consecutive time windows.
- **Per SKU & Warehouse**: Micro-level performance identifying problematic inventory nodes.
- **Per Intermittency Segment**: Comparing model efficacy across:
  - **Smooth**: Regular, frequent demand.
  - **Intermittent**: Sporadic demand occurrences with long zero-demand streaks.
  - **Erratic**: High variability in transaction sizes.
  - **Lumpy**: Both sporadic intervals and erratic transaction sizes.
- **Per ABC/XYZ Class**: Performance on revenue-critical Class A versus tail Class C items.

---

## 5. Master Benchmark Table Standard (Section 25)

The platform formats multi-model benchmark results into a standardized comparative schema:

| Model | MAE | RMSE | WAPE (%) | Bias (%) | Runtime (s) | Failures | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | 1.82 | 2.65 | 14.8 | +0.4 | 1.25 | 0 | SUCCESS |
| **Seasonal Naive** | 2.15 | 3.12 | 17.5 | -0.8 | 0.04 | 0 | SUCCESS |
| **Moving Average (7d)** | 2.30 | 3.28 | 18.7 | -1.2 | 0.02 | 0 | SUCCESS |
| **Exponential Smoothing** | 2.34 | 3.35 | 19.0 | +1.1 | 0.45 | 0 | SUCCESS |
| **Croston (SBA)** | 2.85 | 3.95 | 23.2 | -0.2 | 0.08 | 0 | SUCCESS |
| **Naive** | 2.90 | 4.10 | 23.6 | -0.1 | 0.01 | 0 | SUCCESS |
| **TimesFM** | N/A | N/A | N/A | N/A | 0.00 | 1 | UNAVAILABLE |
