# Cost Breakdown & True Unit Economics (Phase 6B)

## 1. Executive Purpose & Scope

The Unit Economics module establishes an auditable, deterministic framework for measuring commercial contribution margins and cost structures across the eRetail network. It bridges top-line gross margin (established in Phase 6A) with multi-component variable direct expenses, including fulfillment freight, payment gateway processing, packaging materials, warehouse pick/pack labor, and customer return allowances.

> **CRITICAL ARCHITECTURAL BOUNDARY:**
>
> The current canonical dataset does not contain shipping, payment processing, packaging, warehouse handling, or return-processing costs. Therefore final contribution margin cannot be treated as accounting-level profitability.
>
> Product cost is derived from catalog standard procurement pricing (`unit_cost` in `products.csv`) rather than transactional FIFO/LIFO lot accounting COGS.

All calculations in Phase 6B are deterministic, reproducible, and mathematically grounded in source or explicitly configured assumption parameters.

---

## 2. Supported Cost Components & Availability Model

Phase 6B models seven distinct direct cost components:

| Component Key | Component Description | Canonical Status | Canonical Source Type |
|---|---|---|---|
| `PRODUCT_COST` | Standard supplier procurement cost of goods sold | `AVAILABLE_ESTIMATED` | `CATALOG_ESTIMATE` (`products.csv:unit_cost`) |
| `SHIPPING_COST` | Outbound customer fulfillment freight | `UNAVAILABLE` | `UNAVAILABLE` |
| `PAYMENT_PROCESSING_COST` | Merchant acquirer transaction fees & interchange | `UNAVAILABLE` | `UNAVAILABLE` |
| `PACKAGING_COST` | Shipping boxes, poly mailers, bubble wrap, labels | `UNAVAILABLE` | `UNAVAILABLE` |
| `WAREHOUSE_HANDLING_COST` | Fulfillment center pick, pack, and sortation labor | `UNAVAILABLE` | `UNAVAILABLE` |
| `RETURN_PROCESSING_COST` | Reverse logistics shipping & inspection fees | `UNAVAILABLE` | `UNAVAILABLE` |
| `OTHER_VARIABLE_COST` | Channel marketplace referral or incidental fees | `UNAVAILABLE` | `UNAVAILABLE` |

### Multi-Tiered Cost Injection:
While canonical data contains only `PRODUCT_COST`, the engine supports multi-tiered cost injection without schema alterations:
1. **Transaction-level data**: Explicit column fields in transactional sales feeds (`shipping_cost`, `payment_processing_cost`, `packaging_cost`, etc.) $\rightarrow$ `AVAILABLE_SOURCE` / `SOURCE_DATA`.
2. **Channel-level defaults**: Injected via `CostModelConfig.channel_shipping_costs` or `channel_payment_processing_rates` $\rightarrow$ `AVAILABLE_ASSUMED` / `CONFIGURED_ASSUMPTION`.
3. **Warehouse-level defaults**: Injected via `CostModelConfig.warehouse_handling_cost_per_unit` $\rightarrow$ `AVAILABLE_ASSUMED` / `CONFIGURED_ASSUMPTION`.
4. **Network-wide assumptions**: Injected via parametric per-unit or per-order settings $\rightarrow$ `AVAILABLE_ASSUMED` / `CONFIGURED_ASSUMPTION`.

---

## 3. Mathematical Formulation

### 3.1 Revenue & Product Cost
$$ \text{gross\_revenue} = \text{quantity} \times \text{unit\_price} $$
$$ \text{net\_revenue} = \text{gross\_revenue} - \text{discount} $$
$$ \text{product\_cost} = \text{quantity} \times \text{unit\_cost} $$
$$ \text{gross\_margin} = \text{net\_revenue} - \text{product\_cost} $$
$$ \text{gross\_margin\_pct} = \begin{cases} \frac{\text{gross\_margin}}{\text{net\_revenue}} & \text{if } \text{net\_revenue} > 0 \\ \text{None} & \text{otherwise} \end{cases} $$

### 3.2 Known Variable Costs
$$ \text{total\_known\_variable\_cost} = \sum_{c \in \text{Available Variable Costs}} \text{cost}_c $$
If no variable cost components are observed or configured, $\text{total\_known\_variable\_cost} = \$0.00$.

### 3.3 Contribution Margin & Calculability Rules
To prevent misleading profitability claims when cost data is incomplete, Phase 6B separates **known contribution margin** from **final contribution margin**:

1. **Known Contribution Margin**:
   $$ \text{known\_contribution\_margin} = \text{net\_revenue} - \text{product\_cost} - \text{total\_known\_variable\_cost} $$
   $$ \text{known\_contribution\_margin\_pct} = \begin{cases} \frac{\text{known\_contribution\_margin}}{\text{net\_revenue}} & \text{if } \text{net\_revenue} > 0 \\ \text{None} & \text{otherwise} \end{cases} $$

2. **Final Contribution Margin**:
   $$ \text{contribution\_margin} = \begin{cases} \text{known\_contribution\_margin} & \text{if all required components are available} \\ \text{None} & \text{if any required component is missing} \end{cases} $$
   $$ \text{contribution\_margin\_status} = \begin{cases} \text{CALCULABLE} & \text{if all required components are available} \\ \text{INSUFFICIENT\_COST\_DATA} & \text{if any required component is missing} \\ \text{NOT\_CALCULABLE} & \text{if revenue or cost inputs are invalid} \end{cases} $$

---

## 4. Cost Completeness & Economics Status

### 4.1 Cost Completeness Formula
Let $R$ be the set of configured required cost components (e.g. `[PRODUCT_COST, SHIPPING_COST, PAYMENT_PROCESSING_COST, PACKAGING_COST, WAREHOUSE_HANDLING_COST]`), and let $A$ be the set of available components for the transaction:

$$ \text{cost\_completeness\_pct} = \frac{|R \cap A|}{|R|} \times 100.0 $$

In the canonical dataset where only `PRODUCT_COST` is observed against 5 required components:
$$ \text{cost\_completeness\_pct} = \frac{1}{5} \times 100.0 = 20.0\% $$

### 4.2 Unit Economics Status
- **`FULLY_CALCULABLE`**: Valid revenue and 100% of required cost components observed.
- **`PARTIALLY_CALCULABLE`**: Product cost observed, but one or more required variable cost components are missing.
- **`INSUFFICIENT_COST_DATA`**: Product procurement cost is missing in catalog master.
- **`INVALID_DATA`**: Negative quantity, negative price, or malformed currency.

---

## 5. Aggregation Methodology

Unit economics metrics are aggregated across 11 dimensional cuts:
- `OVERALL` (Network totals)
- `SKU`, `CATEGORY`, `BRAND`
- `CHANNEL`, `WAREHOUSE`
- Cross-slices: `SKU_CHANNEL`, `SKU_WAREHOUSE`, `CHANNEL_WAREHOUSE`
- Temporal: `DATE` (Daily), `WEEK` (Weekly), `MONTH` (Monthly)

### Ratio Integrity Principles:
- Ratios are **never** computed as the average of row percentages.
- $\text{aggregate\_gross\_margin\_pct} = \frac{\sum \text{gross\_margin}}{\sum \text{net\_revenue}}$
- $\text{aggregate\_known\_contribution\_margin\_pct} = \frac{\sum \text{known\_contribution\_margin}}{\sum \text{net\_revenue}}$
- Contribution margin for an aggregate slice is marked `CALCULABLE` **only** when 100% of rows in that slice are `FULLY_CALCULABLE`. If any rows have missing required costs, slice contribution margin evaluates to `None` with status `INSUFFICIENT_COST_DATA`.

---

## 6. Descriptive Margin Erosion Analysis

Identifies structural margin dilution across products and nodes without black-box ML or subjective labels:
1. **Negative Margin SKUs**: Products generating commercial losses ($\text{gross\_margin} < \$0$).
2. **Low Margin SKUs**: Products performing below configured margin threshold (default $< 20\%$).
3. **High Discount SKUs**: Products where promotional discounts exceed $25\%$ of gross revenue.
4. **Low Margin Channels & Warehouses**: Facilities performing below the network average gross margin percentage.
5. **Margin Contribution Mismatches**: Segments where revenue share significantly outpaces gross margin share (e.g. generating 10% of revenue but only 2% of margin, diluting portfolio economics).

---

## 7. Data Quality Audit & Point-in-Time Anti-Leakage

### Validation Guards:
- Missing or blank SKU identifiers
- Negative quantities, prices, discounts, or costs
- Missing currency codes or currency discrepancies
- Duplicate `sale_id` records
- Chronological future date leakage relative to `as_of_date`

### Point-in-Time Isolation:
When `as_of_date` is configured, all transactions occurring after `as_of_date` are strictly excluded from calculations, ensuring leak-free backtesting and point-in-time financial truth.

---

## 8. Limitations & Future Integration Points

1. **Transactional Shipping & Fulfillment**: Outbound carrier freight tables (e.g. carrier rate sheets, dimensional zone weight) will be connected when carrier invoice feeds are introduced.
2. **Merchant Acquirer Settlement**: Payment gateway fee schedules vary by payment method (credit card, BNPL, PayPal, debit); currently modeled parametrically by channel.
3. **Return Cost Integration**: Customer return reverse logistics will integrate with Phase 5A returns telemetry to attribute return restocking labor and freight fees to specific product lines.
