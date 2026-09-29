# Feature Engineering & Demand Representation

## 1. Feature Grain & Representation

The demand feature store produces time-series features at the fundamental fulfillment grain:

$$\text{Feature Grain} = \text{Date} + \text{SKU ID} + \text{Warehouse ID}$$

Operating at this atomic level enables both local fulfillment center forecasting and hierarchical aggregation to regional hubs, product categories, or enterprise totals.

---

## 2. Strict Data Leakage Prevention

In time-series forecasting, **data leakage** occurs when information from the target date $T$ (or future $T+k$) is inadvertently included in features used to predict demand at $T$.

### The Invariance Rule
> Any feature representing context at Day $T$ must be computed **strictly on observations from Day $T-1$ or earlier**.

```
Past History: [ T - 30 ... T - 2 , T - 1 ] | Cutoff: [ Day T ]
------------------------------------------+-----------------
       Eligible for Feature Input         |   Target Y_T
```

### Mathematical Enforcement
1. **Lags**:
   $$\text{Lag}_k(T) = Y_{T - k}, \quad k \ge 1$$
   No lag with $k \le 0$ is ever computed.
2. **Rolling Statistics**:
   $$\text{RollingMean}_w(T) = \frac{1}{w} \sum_{i=1}^{w} Y_{T - i}$$
   In code, this is guaranteed by applying `.shift(1)` across the series **prior** to invoking `.rolling(window=w)`.

---

## 3. Feature Taxonomy

### 3.1 Historical Lag Features
Captures autoregressive patterns and weekly day-of-week persistence:
- `demand_lag_1`: Immediate prior day demand (daily momentum).
- `demand_lag_7`: Same-day demand from prior week (weekly seasonality).
- `demand_lag_14`: Same-day demand two weeks prior.
- `demand_lag_28`: 4-week cycle reference.
- `demand_lag_30`: 1-month seasonal reference.

### 3.2 Causally Shifted Rolling Windows
Captures short, medium, and long-term demand levels and volatility:
- `demand_rolling_mean_7` & `demand_rolling_std_7`: 1-week momentum and short-term volatility.
- `demand_rolling_mean_14` & `demand_rolling_std_14`: Bi-weekly smoothed baseline.
- `demand_rolling_mean_30` & `demand_rolling_std_30`: Monthly level and standard deviation.
- `demand_rolling_mean_90` & `demand_rolling_std_90`: Quarterly baseline for trend anchoring.

### 3.3 Momentum & Trend Ratios
Quantifies acceleration or deceleration while safeguarding against zero division:
- `trend_7_vs_30`: $\frac{\text{Mean}_7}{\text{Mean}_{30}}$ ($>1.0$ indicates demand acceleration; $<1.0$ indicates softening).
- `trend_30_vs_90`: $\frac{\text{Mean}_{30}}{\text{Mean}_{90}}$ (long-term structural trend indicator).
- `demand_growth_7d`: $\frac{\text{Lag}_1 - \text{Lag}_7}{\text{Lag}_7}$ (week-over-week velocity growth).
- `demand_growth_30d`: $\frac{\text{Lag}_1 - \text{Lag}_{30}}{\text{Lag}_{30}}$ (month-over-month velocity growth).

### 3.4 Calendar & Cyclical Features
Preserves calendar continuity and smooth trigonometric transitions:
- Temporal components: `year`, `month`, `quarter`, `week`, `day_of_week` (0=Mon, 6=Sun), `day_of_month`, `day_of_year`.
- Boundary markers: `is_weekend`, `is_month_start`, `is_month_end`, `is_quarter_start`, `is_quarter_end`.
- Cyclical Encodings:
  $$\sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right), \quad \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$$
  $$\sin\left(\frac{2\pi \cdot (\text{month} - 1)}{12}\right), \quad \cos\left(\frac{2\pi \cdot (\text{month} - 1)}{12}\right)$$
  Ensures that Sunday ($6$) and Monday ($0$), as well as December ($12$) and January ($1$), are geometrically adjacent in feature space.

### 3.5 Commercial Events & Holiday Markers
Incorporate promotional and macro shocks:
- `is_event`: Binary indicator (1 if date coincides with a major commercial/holiday event).
- `event_name` & `event_type`: e.g. "Black Friday" (Mega Sale), "Payday" (Payroll cycle).
- `event_impact_factor`: Historical demand multiplier (e.g. $3.5\times$ for Cyber Monday).

---

## 4. ABC-XYZ Portfolio Segmentation

Combining revenue contribution with demand predictability yields a 9-box operational matrix:

```
           X (Low CV <= 0.5)     Y (Med CV 0.5-1.0)     Z (High CV > 1.0)
       +-----------------------+----------------------+----------------------+
   A   |          AX           |          AY          |          AZ          |
 (Top  | High Value, Steady    | High Value, Moderate | High Value, Erratic  |
  80%) | Automated Replenish   | Buffer Stock Needed  | Strict Human Review  |
       +-----------------------+----------------------+----------------------+
   B   |          BX           |          BY          |          BZ          |
 (Next | Med Value, Steady     | Med Value, Variable  | Med Value, Lumpy     |
  15%) | Periodic Review       | Standard Safety Stk  | Reorder on Demand    |
       +-----------------------+----------------------+----------------------+
   C   |          CX           |          CY          |          CZ          |
 (Tail | Low Value, Steady     | Low Value, Moderate  | Low Value, Erratic   |
  5%)  | Bulk / High MOQ       | Min Batch Reorder    | Made-to-Order / Drop |
       +-----------------------+----------------------+----------------------+
```

### Computational Isolation
- **ABC**: Computed using revenue contribution across the catalog. Thresholds ($80\%, 95\%$) are fully configurable.
- **XYZ**: Computed using the Coefficient of Variation ($CV = \frac{\sigma}{\mu}$) calculated **exclusively on unconstrained, stockout-masked observations** (`forecast_training_eligible == True`) so inventory bottlenecks do not artificially inflate volatility scores.

---

## 5. Forecast Error Metrics

Implemented in `commerce_ai.analytics.metrics`:

| Metric | Formula | Primary Use Case & Characteristics |
| :--- | :--- | :--- |
| **MAE** | $\frac{1}{n} \sum \|y - \hat{y}\|$ | Direct volume error in natural units. Robust to outliers. |
| **RMSE** | $\sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$ | Penalizes large errors heavily; critical for capacity planning. |
| **MAPE** | $\frac{1}{n} \sum \frac{\|y - \hat{y}\|}{\|y\|} \times 100\%$ | Unitless relative error. Masked on zero-demand days to avoid $\infty$. |
| **WAPE** | $\frac{\sum \|y - \hat{y}\|}{\sum \|y\|} \times 100\%$ | **Gold standard retail metric**. Volume-weighted; robust to zero actuals. |
| **Bias** | $\frac{\sum (\hat{y} - y)}{\sum \|y\|} \times 100\%$ | Tracks systematic over-forecasting ($>0$) vs under-forecasting ($<0$). |
