# Demand Intelligence & Stockout Analysis

## 1. The Core Principle: Observed Sales ≠ True Demand

A standard pitfall in retail analytics is mistaking **historical sales** for **true consumer demand**.

$$\text{Observed Sales} = \min(\text{True Demand}, \text{Available On-Hand Inventory})$$

Consider an example:
- **True Customer Demand**: 100 units
- **Available Warehouse Inventory**: 20 units
- **Recorded Sales**: 20 units

If a machine learning forecasting model is trained naively on the 20 recorded units, it will incorrectly predict that demand is declining. This creates a destructive operational spiral: lower forecast $\rightarrow$ lower purchase orders $\rightarrow$ further stockouts $\rightarrow$ permanent loss of market share.

The **Demand Intelligence Layer** reconstructs the true demand context by cross-referencing sales transactions against concurrent inventory positions.

---

## 2. Regularized Demand Grid

Historical transaction logs omit dates where zero sales occurred. Dropping zero-sales days corrupts statistical time-series properties.

The `build_daily_demand` engine resolves this by generating a regularized Cartesian product grid:

$$\text{Grid} = \text{Dates} \times \text{SKUs} \times \text{Warehouses}$$

- Days without sales are explicitly populated with `units_sold = 0` and `revenue = 0.0`.
- Inventory snapshots are forward-filled across the temporal grid to maintain continuous visibility of on-hand stock (`available_qty`, `reserved_qty`, `in_transit_qty`).

---

## 3. Stockout Detection & Running Streak Tracking

Stockout detection is governed by configurable rules defined in `StockoutConfig`:

| State | Condition | Interpretation |
| :--- | :--- | :--- |
| **Stockout** | `available_qty <= stockout_threshold` (default: 0) | Physical depletion; fulfillment impossible. |
| **Low Stock** | `available_qty <= low_stock_threshold` (default: 5) | Danger zone; potential order truncation. |
| **Low-Stock Exhaustion**| `is_low_stock` and `units_sold >= available_qty` | Sales consumed all remaining stock, capping demand. |

### Tracking Metrics
- `is_stockout`: Boolean flag indicating on-hand zero balance.
- `stockout_days`: Running cumulative counter of consecutive days spent in stockout.
- `stockout_event_id`: Unique persistent identifier across each contiguous stockout streak (`SO_{sku}_{wh}_{event_count}`), facilitating duration and frequency audits.

---

## 4. Demand Masking & Forecast Eligibility

The objective of demand masking is **not to delete data**, but to provide downstream modeling pipelines with explicit filtering criteria:

| Date | Sales | Available Stock | Stockout | Constraint Status | Forecast Training Eligible |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2025-01-01 | 45 | 250 | No | Unconstrained | **True** |
| 2025-01-02 | 50 | 200 | No | Unconstrained | **True** |
| 2025-01-03 | 12 | 12 | No (Low Stock) | **Constrained** (Exhausted) | **False** (Masked) |
| 2025-01-04 | 0 | 0 | **Yes** | **Constrained** (Stockout) | **False** (Masked) |
| 2025-01-05 | 0 | 0 | **Yes** | **Constrained** (Stockout) | **False** (Masked) |
| 2025-01-06 | 55 | 300 | No | Unconstrained | **True** |

- `is_demand_constrained`: Set to `True` when sales were throttled by inventory shortages.
- `forecast_training_eligible`: Inverted boolean flag. When training forecasting models (e.g. TimesFM-3, LightGBM), masking in-eligible observations prevents artificial zeroes from distorting baseline demand estimations.

---

## 5. Demand Intermittency & Predictability

Not all products exhibit smooth daily consumption patterns. Demand behavior is segmented into three archetypes based on the `demand_occurrence_rate` ($\frac{\text{Active Selling Days}}{\text{Total Days}}$):

```
[ Active Days >= 70% ]  -->  REGULAR DEMAND
                             Predictable, high-volume; suited for standard time-series models.

[ 25% <= Active Days < 70% ] --> INTERMITTENT DEMAND
                                 Lumpy demand; requires Poisson / Croston-style models.

[ Active Days < 25% ]   -->  HIGHLY INTERMITTENT (SPARSE)
                             Infrequent bursts; requires hurdle / zero-inflated modeling.
```

### Derived Intermittency Metrics
- `zero_demand_days`: Count of inactive days.
- `non_zero_demand_days`: Count of active sales days.
- `demand_occurrence_rate`: Proportion of time the SKU experiences positive demand.
- `average_nonzero_demand`: Average transaction batch size when demand occurs.
- `intermittency_class`: `Regular`, `Intermittent`, or `Highly Intermittent`.
