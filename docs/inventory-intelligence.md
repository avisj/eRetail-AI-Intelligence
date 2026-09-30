# Inventory Intelligence Engine (Phase 4A)

## 1. Overview & Architectural Principles

The **Inventory Intelligence Engine** serves as the analytical foundation of the platform's Prescriptive Layer (Tier 4). It evaluates on-hand, on-order, and committed inventory positions, computes supplier lead-time uncertainties, determines statistical safety stocks and dynamic reorder points (ROP), and classifies forward-looking inventory risks across SKU-warehouse portfolios.

### Architectural Rules
1. **Deterministic-First**: All inventory math (safety stock, ROP, lead-time variance, days of supply) uses deterministic equations.
2. **Decoupled Risk and Recommendation**: Risk reflects current and future inventory condition along with empirical evidence. Recommendation (purchase order proposals and rebalancing) is handled in separate modules.
3. **Configurable Policy**: Service-level targets, Z-scores, lead-time fallback thresholds, and risk classification boundaries are configurable parameters rather than hard-coded constants.
4. **Transparent Diagnostic Evidence**: The engine never reduces inventory decisions to an opaque score. Every assessment exposes all underlying metrics (daily demand mean/std, lead time mean/std, days of supply, stockout hazard probability, and capital exposure).
5. **Strict Non-Hallucination for EOQ**: Economic Order Quantity (EOQ) is strictly optional. If holding cost or fixed ordering cost cannot be derived reliably from catalog data, the system falls back to order-up-to periodic review without inventing cost values.

---

## 2. Core Mathematical Formulations

### 2.1 Net Inventory Position
The effective inventory position available to satisfy future demand:
$$\text{Net Inventory Position} = \text{On-Hand (Available)} + \text{On-Order (Open Inbound)} - \text{Reserved (Allocated)}$$

Where:
- $\text{On-Hand}$ ($\text{available\_qty}$): Uncommitted, pickable physical units in the warehouse.
- $\text{On-Order}$ ($\text{in\_transit\_qty}$ or open PO lines): Scheduled supplier deliveries in status `PENDING` or `IN_TRANSIT`. Delivered, cancelled, or completed orders are strictly excluded to avoid double-counting.
- $\text{Reserved}$ ($\text{reserved\_qty}$): Units committed to pending customer orders awaiting fulfillment.
- $\text{Total Physical Inventory} = \text{On-Hand} + \text{Reserved} + \text{Damaged}$.

---

### 2.2 Supplier & SKU Lead-Time Profiling
For every delivered purchase order $i$:
$$\text{actual\_lead\_time}_i = \text{actual\_delivery\_date}_i - \text{order\_date}_i$$

Empirical metrics computed over sample size $N$:
- **Mean Lead Time**: $L = \frac{1}{N} \sum_{i=1}^N \text{actual\_lead\_time}_i$
- **Lead-Time Standard Deviation**: $\sigma_L = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (\text{actual\_lead\_time}_i - L)^2}$ (sample variance, $\text{ddof}=1$)
- **On-Time Delivery Rate (OTD %)**: $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{actual\_delivery\_date}_i \le \text{expected\_delivery\_date}_i)$

#### Point-in-Time Temporal Leakage Protection
When backtesting or performing historical inventory evaluation as of a specific snapshot date (`as_of_date`), purchase orders delivered *after* `as_of_date` are strictly excluded from completed lead-time profiling (`order_date <= as_of_date` and `actual_delivery_date <= as_of_date`). This prevents future supplier transit performance from leaking into historical replenishment decisions.

#### Hierarchical Fallback Hierarchy
When sample size $N < \text{min\_history}$ (default = 3):
1. Use SKU-specific delivered POs (if $N \ge \text{min\_history}$).
2. Fall back to the preferred supplier's aggregate delivered POs.
3. Fall back to `Supplier.average_lead_time_days` from the catalog master (with $\sigma_L = 0$ or default std).
4. Fall back to platform global default (`default_lead_time_days = 14.0`).

---

### 2.3 Statistical Safety Stock (Dual Uncertainty)
Replenishment lead times and daily customer demand both exhibit stochastic variance. The engine uses the standard convolution model for independent normal distributions:

$$SS = Z \times \sqrt{L \cdot \sigma_D^2 + D^2 \cdot \sigma_L^2}$$

Where:
- $Z = \Phi^{-1}(\text{Target Service Level})$ (Inverse standard normal CDF / probit function).
- $L$ = Average replenishment lead time in days.
- $\sigma_D$ = Standard deviation of daily demand (computed over historical clean, unconstrained sales to preserve empirical uncertainty even under flat forward forecasts).
- $D$ = Daily demand rate (forward forecast mean or historical clean fallback).
- $\sigma_L$ = Standard deviation of replenishment lead time in days.

#### Analytical Reductions
- When $\sigma_L = 0$ (constant lead time): $SS = Z \cdot \sigma_D \sqrt{L}$.
- When $\sigma_D = 0$ (constant demand): $SS = Z \cdot D \cdot \sigma_L$.
- When $D = 0$ or $L = 0$: $SS = 0.0$.
- When $\sigma_D = 0$ and $\sigma_L = 0$: $SS = 0.0$.

---

### 2.4 Dynamic Reorder Point (ROP) & Forecast Integration
$$\text{ROP} = (D_{\text{lead\_time}} \times L) + SS$$

Where $(D_{\text{lead\_time}} \times L)$ is expected lead-time demand, and $SS$ protects against variance spikes.

#### Forward Forecast for Lead-Time Demand (`use_forecast_for_rop = True`)
- When enabled (default), $D_{\text{lead\_time}}$ is calculated from the forward out-of-sample forecast mean ($\hat{D}_{\text{forecast}}$), capturing upcoming seasonality, promotional lifts, or trend shifts.
- To prevent artificial collapse of safety stock when models produce smooth or flat point forecasts, $\sigma_D$ continues to measure empirical historical demand volatility.
- Every `SafetyStockResult` exposes `demand_rate_source`:
  - `"FORECAST"`: Forward forecast mean used.
  - `"HISTORICAL"`: Historical clean mean used (`use_forecast_for_rop = False`).
  - `"HISTORICAL_FALLBACK"`: Forecast unavailable or empty; fell back to historical clean mean.

---

### 2.5 Optional Economic Order Quantity (EOQ)
If fixed ordering cost ($S$) and holding cost per unit per year ($H$) are known:
$$\text{EOQ} = \sqrt{\frac{2 \cdot D_{\text{annual}} \cdot S}{H}}$$

Where $H$ can be supplied directly or computed from annual holding rate $h$ ($H = \text{unit\_cost} \times h$). If either cost is missing, the engine returns `None` and bases target stock levels on periodic review coverage:
$$\text{Target Stock Level} = \text{ROP} + (D \times \text{review\_period\_days})$$

---

## 3. Inventory Risk & Health Classification

### 3.1 Days of Supply (DOS)
$$\text{DOS} = \frac{\text{On-Hand}}{\max(D_{\text{forecast}}, \epsilon)}$$

### 3.2 Projected Depletion & Runout Date
Day-by-day simulated on-hand trajectory over forecast horizon $H$:
$$\text{remaining\_stock}_t = \max\left(0, \text{On-Hand} - \sum_{i=1}^t \hat{d}_i\right)$$

The exact fractional day until stockout ($\text{days\_to\_runout}$) is interpolated at the moment cumulative demand surpasses on-hand inventory.

### 3.3 Statistical Stockout Hazard Score
Estimated probability that lead-time demand will exceed current net inventory position:
$$P(\text{Stockout}) = P(D_L > \text{Net Position}) = 1 - \Phi\left(\frac{\text{Net Position} - D \cdot L}{\sqrt{L \cdot \sigma_D^2 + D^2 \cdot \sigma_L^2}}\right)$$

The hazard evaluation uses the SKU's historical daily demand variance ($\sigma_D$) rather than the variance of forecast step points, preventing the hazard curve from collapsing into an unrealistic 0/1 step function when point forecasts are uniform.

### 3.4 Risk Classification Categories
The engine assigns each SKU-warehouse entity to one of five discrete states:

| Category | Definition & Criteria | Business Impact |
| :--- | :--- | :--- |
| **`CRITICAL_STOCKOUT`** | Non-dormant demand AND ($\text{On-Hand} \le 0$ OR $\text{DOS} < L$ OR $\text{days\_to\_runout} \le L$) | Immediate lost revenue; stock will deplete before normal replenishment arrives. |
| **`UNDERSTOCK`** | $\text{Net Position} < \text{ROP}$ (and not critical) | Service level breach risk; purchase order must be issued promptly. |
| **`HEALTHY`** | Active SKU with $\text{ROP} \le \text{Net Position}$ and $\text{DOS} \le \text{overstock\_threshold}$; OR Dormant SKU with $\text{On-Hand} \le 0$ | Balanced inventory; supply pipeline covers target service level. Inactive items without stock do not trigger false alerts. |
| **`OVERSTOCK`** | $\text{DOS} > \text{overstock\_threshold}$ (default: 60 days) with active sales | Excess working capital tied up; carrying cost drag. |
| **`DEAD_STOCK`** | $\text{On-Hand} > 0$ with dormant demand ($\le \text{min\_demand\_threshold}$) | Capital lockup; high obsolescence risk. |


---

## 4. Configurable Parameters & Defaults

### 4.1 ServiceLevelPolicy
Default 9-box service level matrix based on ABC (revenue) and XYZ (volatility) classifications:

| ABC / XYZ | X (Low Volatility) | Y (Moderate Volatility) | Z (High Volatility) |
| :---: | :---: | :---: | :---: |
| **A (High Revenue)** | 99% ($Z \approx 2.33$) | 95% ($Z \approx 1.64$) | 92% ($Z \approx 1.41$) |
| **B (Medium Revenue)**| 95% ($Z \approx 1.64$) | 92% ($Z \approx 1.41$) | 90% ($Z \approx 1.28$) |
| **C (Low Revenue)**   | 90% ($Z \approx 1.28$) | 88% ($Z \approx 1.17$) | 85% ($Z \approx 1.04$) |

*Default Fallback*: 95%. Overrides supported per segment or per specific SKU ID.

### 4.2 LeadTimeConfig
- `min_history`: Minimum delivered orders required to calculate sample variance (default: 3).
- `default_lead_time_days`: Global fallback lead time when supplier data is absent (default: 14.0 days).
- `default_lead_time_std_days`: Fallback standard deviation (default: 0.0 days).

### 4.3 RiskThresholdConfig
- `critical_dos_threshold`: 3.0 days.
- `overstock_dos_threshold`: 60.0 days.
- `dead_stock_history_days`: 90 days.
- `min_demand_threshold`: 0.01 units/day.
- `lead_time_buffer_factor`: 1.0.

---

## 5. Limitations & Future Extensions

1. **Independent Demand & Lead Times**: The primary safety stock formula assumes demand and lead time are statistically independent. If supplier lead times expand specifically during peak holiday surges, joint covariance can be incorporated in future iterations.
2. **Normal Lead-Time Demand**: Z-score multipliers assume normal distribution of cumulative lead-time demand. For highly erratic, intermittent items (Z-class), empirical quantile methods or Poisson-gamma compounds provide further accuracy.
3. **Multi-Echelon Network Effects**: Phase 4A analyzes each SKU-warehouse node independently. Multi-echelon pooling and inter-warehouse rebalancing will be introduced in Phase 4C.
