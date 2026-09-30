# Returns Intelligence Foundation (Phase 5A)

## 1. Overview & Purpose

The **Returns Intelligence Foundation** establishes a deterministic, factual, and explainable customer returns intelligence pipeline for the platform.

Unlike replenishment or inventory rebalancing, which plan and move future physical stock, returns intelligence analyzes historical and ongoing return behavior across multi-channel fulfillment operations to answer fundamental operational questions:
- *How many units were returned and what is the exact return rate?*
- *Which SKUs, sales channels, or fulfillment warehouses exhibit elevated return rates?*
- *What are the primary documented drivers (reasons) behind customer returns?*
- *How are return rates trending over time?*
- *What is the financial valuation of returned merchandise?*
- *Which SKU × Channel or SKU × Warehouse combinations require operational investigation?*

---

## 2. Actual Returns Data Grain & Schema

### 2.1 Returns Data Grain
Through repository inspection of `src/commerce_ai/data/schemas.py`, `src/commerce_ai/data/generators.py`, and `src/commerce_ai/data/validators.py`:
- **Primary Key**: `return_id`
- **Granularity**: **Return Event / Return Line Item**. Each individual record represents a single returned item event associated with a specific sales order (`order_id`), product (`sku_id`), fulfillment warehouse (`warehouse_id`), and originating sales channel (`channel_id`), processed on `return_date` with a specific positive `quantity` and documented `reason`.

### 2.2 Canonical Schema (`ReturnRecord`)

| Field Name | Type | Description | Mandatory | Validation Rules |
| :--- | :--- | :--- | :--- | :--- |
| `return_id` | `str` | Unique return event identifier | Yes | Non-empty, unique |
| `order_id` | `str` | Associated original sales order ID | Yes | Non-empty, references `sales.order_id` |
| `return_date` | `date` | Date return was physically processed | Yes | Valid calendar date |
| `sku_id` | `str` | Returned Stock Keeping Unit ID | Yes | Non-empty, references `products.sku_id` |
| `warehouse_id` | `str` | Receiving/destination warehouse facility ID | Yes | Non-empty, references `warehouses.warehouse_id` |
| `quantity` | `int` | Returned quantity in units | Yes | Strict positive integer ($> 0$) |
| `reason` | `str` | Documented return reason category | Yes | Non-empty string |
| `channel_id` | `str` | Originating sales channel ID | Yes | Non-empty, references `channels.channel_id` |

> [!NOTE]
> **Monetary Fields**: Notice that `returns.csv` does **not** contain an intrinsic `refund_amount` or `return_value` column. The monetary valuation of returned units is deterministically resolved by joining against sales transaction order lines (`sales.unit_price` / `sales.revenue / sales.quantity`) or product master catalog records (`products.selling_price` or `products.unit_cost`). If neither sales pricing nor product pricing is provided, monetary values are reported as `None` without fabricating numbers.

---

## 3. Return-Rate Definitions

The engine distinguishes and explicitly calculates three distinct return rates without conflation:

### 3.1 Unit Return Rate (Primary Benchmark)
The primary metric used across all dimensional aggregations:

$$\text{Unit Return Rate} = \frac{\text{Returned Units}}{\text{Sold Units}}$$

- If $\text{Sold Units} > 0$: Factual ratio $\in [0.0, \infty)$ (typically $\le 1.0$).
- If $\text{Sold Units} == 0$ and $\text{Returned Units} == 0$: Evaluates to $0.0$.
- If $\text{Sold Units} == 0$ and $\text{Returned Units} > 0$: Evaluates to `None` and triggers the descriptive flag `ZERO_SALES_RECORDED` (e.g. an order placed in a prior untracked window returned during the current evaluation window).

### 3.2 Order Return Rate
Measures the proportion of sales transactions resulting in at least one return event:

$$\text{Order Return Rate} = \frac{\text{Distinct Returned Orders}}{\text{Distinct Total Sales Orders}}$$

### 3.3 Revenue Return Rate
Quantifies the monetary proportion of gross revenue refunded or returned:

$$\text{Revenue Return Rate} = \frac{\text{Total Estimated Return Value}}{\text{Total Sales Revenue}}$$

---

## 4. Multi-Dimensional Analytics

Returns intelligence aggregates performance across five distinct operational dimensions:

### 4.1 SKU Analytics
Profiles individual product performance:
- `sold_units`, `returned_units`, `return_rate`
- `order_count`, `return_count`, `order_return_rate`
- `sales_value`, `return_value`, `revenue_return_rate`
- `top_return_reason` (mode category for this SKU)
- `return_rate_change` and `trend` (`INCREASING`, `DECREASING`, `STABLE`, `INSUFFICIENT_DATA`)

### 4.2 Channel Analytics
Evaluates return performance across customer acquisition channels (e.g. Marketplace, Direct Web, Retail, B2B).
- **Neutral Language Guard**: Channels are presented factually with comparative return rates; the engine **never** applies subjective labels like "best" or "worst".

### 4.3 Warehouse Analytics
Evaluates return volume received by fulfillment facilities.
- **No Causal Blame Invariant**: Warehouse metrics reflect the physical return handling facility; the engine **never** implies the warehouse caused the return, using objective statements such as *"higher return rate observed for orders fulfilled by warehouse WH-001"*.

### 4.4 SKU × Channel Cross-Tabulation
Identifies channel-specific return spikes (e.g. a product with an overall healthy 5% return rate that suffers a 25% return rate on a specific marketplace channel due to inaccurate imagery or sizing charts).

### 4.5 SKU × Warehouse Cross-Tabulation
Identifies localized fulfillment discrepancies or regional return concentrations.

---

## 5. Return Reason Distribution

The engine aggregates observed return reason categories directly from the data:
- `returned_units`: Total units returned under the reason.
- `return_count`: Total return events under the reason.
- `percentage_of_units`: Share of total returned units ($\sum = 100\%$).
- `percentage_of_returns`: Share of total return transactions ($\sum = 100\%$).
- `estimated_return_value`: Monetary valuation associated with the reason.

> [!IMPORTANT]
> **No Synthetic Normalization**: Only documented categories present in the data are processed (e.g. *"Defective Item"*, *"Wrong Size/Color"*, *"Late Delivery"*, *"Item Not as Described"*, *"Customer Regret"*). Categories are never invented.

---

## 6. Time-Series Trends & Period Comparison

- **Periodic Bucketing**: Supports daily (`D`), weekly (`W`), and monthly (`M`) aggregation.
- **Period-over-Period Delta**:
  $$\Delta \text{Return Rate} = \text{Return Rate}_{\text{current period}} - \text{Return Rate}_{\text{previous period}}$$
- **Trend Classification**:
  - `INCREASING`: $\Delta \text{Return Rate} \ge +\text{increasing\_rate\_delta\_threshold}$ ($+0.05$)
  - `DECREASING`: $\Delta \text{Return Rate} \le -\text{increasing\_rate\_delta\_threshold}$ ($-0.05$)
  - `STABLE`: $|\Delta \text{Return Rate}| < 0.05$
  - `INSUFFICIENT_DATA`: Prior period history is unavailable or below sample threshold.

---

## 7. Sample Size Controls & Investigation Flags

### 7.1 Sample Size Reliability Guard
A high return rate computed over an insignificant sales volume (e.g. 1 return out of 2 sales = 50%) is statistically unreliable.
- **Configurable Threshold**: `min_sold_units_threshold` (default: 30 units).
- Items below this threshold are marked `is_sufficient_sample = False` and receive the flag `INSUFFICIENT_SAMPLE`.
- High-rate warnings are suppressed for insufficient samples to prevent false operational alarms.

### 7.2 Deterministic Investigation Flags
The engine emits factual audit flags without black-box ML:

| Flag Identifier | Trigger Condition | Operational Meaning |
| :--- | :--- | :--- |
| `HIGH_RETURN_RATE` | $\text{return\_rate} \ge \text{high\_return\_rate\_threshold}$ ($15\%$) and sufficient sample | Elevated return proportion warrants catalog or quality review |
| `HIGH_RETURN_VOLUME` | $\text{returned\_units} \ge \text{high\_return\_volume\_threshold}$ ($50$ units) | High physical volume impacting processing capacity |
| `RETURN_RATE_INCREASING` | $\Delta \text{Return Rate} \ge +5\%$ vs previous period | Emerging quality or customer mismatch issue |
| `INSUFFICIENT_SAMPLE` | $\text{sold\_units} < 30$ | Metric is statistically preliminary |
| `NO_RETURN_DATA` | $\text{sold\_units} > 0$ and $\text{returned\_units} == 0$ | Zero recorded customer returns |
| `ZERO_SALES_RECORDED` | $\text{sold\_units} == 0$ and $\text{returned\_units} > 0$ | Lagged return from order prior to evaluation window |

---

## 8. Anti-Leakage & As-Of Date Guarantee

To maintain mathematical validity for downstream forecasting and future predictive models:
- Every query accepts an optional `as_of_date`.
- All returns with $\text{return\_date} > \text{as\_of\_date}$ are **strictly excluded**.
- All sales with $\text{date} > \text{as\_of\_date}$ are **strictly excluded**.
- Lookahead leakage into past evaluation periods is prevented.

---

## 9. Data Quality Auditing

The engine performs comprehensive integrity checks exposed in `ReturnsDataQualityReport`:
1. **Missing Identifiers**: Detects null/empty `sku_id`, `return_date`, and `order_id`.
2. **Domain Range Violations**: Flags non-positive quantities ($\le 0$).
3. **Duplicate Detection**: Identifies duplicated `return_id` primary keys.
4. **Referential Integrity**: Checks foreign keys against `products` (`sku_id`), `warehouses` (`warehouse_id`), and `channels` (`channel_id`).
5. **No Destructive Dropping**: Flawed records are documented in the audit log rather than silently discarded.

---

## 10. Limitations & Future Roadmap

### Current Limitations (Phase 5A)
- Returns valuation is derived from order lines or product catalog masters; return shipping costs, restocking fees, and refurbishment expenses are not modeled.
- Disposition status (e.g. returned to stock, refurbished, liquidated, destroyed) is not captured in the raw data contract.

### Future Roadmap (Subsequent Phases)
- **Phase 5B**: Return anomaly detection (statistical variance & seasonality de-trending).
- **Phase 5C**: Pre-purchase return-risk prediction (supervised ML).
- **Phase 5D**: Return financial intelligence & profit margin drag modeling.
- **Phase 5E**: AI explanation and corrective recommendation agents.
