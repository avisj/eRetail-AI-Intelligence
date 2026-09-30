# Financial Intelligence Foundation (Phase 6A)

## 1. Executive Purpose & Scope

The Financial Intelligence Foundation establishes the foundational financial analytics layer for the eRetail AI Intelligence platform. It provides deterministic, reproducible, and mathematically sound calculations for transactional revenue, standard procurement cost of goods sold, and gross margin across product, channel, warehouse, and temporal dimensions.

> **CRITICAL ARCHITECTURAL BOUNDARY:**
>
> `estimated_cogs` is based on the available product-level unit_cost and should not automatically be interpreted as accounting COGS.
>
> Phase 6A provides descriptive financial analytics. It does not optimize pricing, recommend discounts, forecast profit, or execute financial actions.

All calculations in Phase 6A are deterministic, traceable directly to source records, and operate without heuristics, ML predictions, or black-box estimation.

---

## 2. Canonical Source Field Inventory

The Phase 6A engine integrates data across four core commerce datasets:

### `sales.csv` (773,555 observed transactions)
| Field | Data Type | Role in Financial Analytics |
|---|---|---|
| `sale_id` | String | Unique transactional line identifier |
| `order_id` | String | Commercial sales order ID (used for distinct order counts and AOV) |
| `date` / `sale_date` | Date/String | Transaction timestamp (YYYY-MM-DD), basis for time intelligence & anti-leakage filtering |
| `sku_id` | String | Foreign key to Product Master catalog |
| `warehouse_id` | String | Fulfillment node identifier |
| `channel_id` | String | Commercial selling channel identifier |
| `quantity` | Integer | Physical units sold |
| `unit_price` | Float | Unit selling price charged to customer |
| `discount` | Float | Observed promotional discount granted on the order line |
| `revenue` | Float | Directly observed source net revenue recorded in transaction feed |
| `currency` | String | ISO-4217 currency denomination (`USD` canonical) |

### `products.csv` (500 observed SKUs)
| Field | Data Type | Role in Financial Analytics |
|---|---|---|
| `sku_id` | String | Primary product identifier |
| `product_name` | String | Human-readable product descriptor |
| `category_id` | String | Product category classification hierarchy |
| `brand` | String | Product brand classification |
| `unit_cost` | Float | Standard supplier procurement unit cost |
| `selling_price` | Float | Catalog list price |
| `currency` | String | Denomination (`USD`) |
| `preferred_supplier_id` | String | Primary procurement vendor |
| `velocity_tier` | String | Inventory movement velocity |

### `channels.csv` (4 selling channels)
- `CH_AMZ` (Amazon)
- `CH_SHOPIFY` (Shopify Direct-to-Consumer)
- `CH_TIKTOK` (TikTok Shop)
- `CH_WALMART` (Walmart Marketplace)

### `warehouses.csv` (5 fulfillment centers)
- `WH_CENTRAL`, `WH_EAST`, `WH_NORTH`, `WH_SOUTH`, `WH_WEST`

---

## 3. Financial Definitions & Mathematical Formulation

### 3.1 Gross Revenue
$$ \text{gross\_revenue} = \text{quantity} \times \text{unit\_price} $$
Calculated for all transactions where both `quantity` $\ge 0$ and `unit_price` $\ge 0$.

### 3.2 Promotional Discount & Discount Rate
$$ \text{discount} = \text{observed discount (or 0.0 if omitted)} $$
$$ \text{discount\_rate} = \begin{cases} \frac{\text{discount}}{\text{gross\_revenue}} & \text{if } \text{gross\_revenue} > 0 \\ 0.0 & \text{if } \text{gross\_revenue} = 0 \text{ and } \text{discount} = 0 \\ \text{None} & \text{otherwise} \end{cases} $$

### 3.3 Net Realized Revenue
$$ \text{calculated\_net\_revenue} = \text{gross\_revenue} - \text{discount} $$

### 3.4 Estimated COGS (Cost of Goods Sold)
$$ \text{estimated\_cogs} = \text{quantity} \times \text{unit\_cost} $$
*Note:* Named `estimated_cogs` because `unit_cost` originates from the standard supplier master rather than transaction-specific FIFO/LIFO inventory layers. If `unit_cost` is unavailable for a SKU, `estimated_cogs` is set to `None` (never assumed as zero) and the record receives a `PARTIAL` financial status.

### 3.5 Gross Margin & Gross Margin Percentage
$$ \text{gross\_margin} = \text{calculated\_net\_revenue} - \text{estimated\_cogs} $$
$$ \text{gross\_margin\_percentage} = \begin{cases} \frac{\text{gross\_margin}}{\text{calculated\_net\_revenue}} & \text{if } \text{calculated\_net\_revenue} > 0 \\ \text{None} & \text{if } \text{calculated\_net\_revenue} \le 0 \end{cases} $$

Zero-division protection prevents division by zero or negative revenue distortions, returning `None` instead of `NaN` or `Inf`.

---

## 4. Observed vs Calculated Values & Source Reconciliation

To maintain audit integrity, the engine never overwrites source transaction revenue. Instead, it computes an audit reconciliation:

$$ \text{revenue\_variance} = \text{calculated\_net\_revenue} - \text{source\_revenue} $$

### Reconciliation Status Rules:
1. **`MATCH`**: $|\text{revenue\_variance}| \le \text{minor\_variance\_tolerance}$ (default: $\$0.02$).
2. **`MINOR_VARIANCE`**: $\text{minor\_variance\_tolerance} < |\text{revenue\_variance}| \le \text{material\_variance\_tolerance}$ (default: $\$1.00$).
3. **`MATERIAL_VARIANCE`**: $|\text{revenue\_variance}| > \text{material\_variance\_tolerance}$ ($\ge \$1.00$).
4. **`UNAVAILABLE`**: Source revenue is null or unobserved.

---

## 5. Financial Data Statuses & Polarities

Each line item is classified into a strictly typed `FinancialDataStatus`:
- **`SUFFICIENT`**: Both valid revenue components and product `unit_cost` are observed. Full margin is computable.
- **`PARTIAL`**: Valid revenue is computable, but product `unit_cost` is missing in catalog. COGS and margin are uncomputable (`None`).
- **`INSUFFICIENT`**: Core revenue inputs (`quantity` or `unit_price`) are missing or null.
- **`INVALID`**: Negative quantities, negative prices, negative discounts, or negative costs detected.

Each computed margin is classified into `MarginClassification`:
- **`POSITIVE_MARGIN`**: $\text{gross\_margin} > 0$.
- **`ZERO_MARGIN`**: $\text{gross\_margin} = 0$.
- **`NEGATIVE_MARGIN`**: $\text{gross\_margin} < 0$.
- **`UNAVAILABLE`**: Cost or revenue unavailable.

---

## 6. Aggregation Methodology & Ratio Integrity

The aggregation engine aggregates across 11 dimensional cuts:
1. **Overall Portfolio**
2. **By SKU**
3. **By Category**
4. **By Brand**
5. **By Channel**
6. **By Warehouse**
7. **SKU × Channel**
8. **SKU × Warehouse**
9. **Channel × Warehouse**
10. **By Date**
11. **By Month**

### Ratio Integrity Principles:
- **Never average row-level percentages**: Aggregate gross margin percentage is strictly:
  $$ \text{gross\_margin\_percentage} = \frac{\sum \text{gross\_margin}}{\sum \text{calculated\_net\_revenue}} $$
- **Distinct Order Counting**: `order_count` counts unique `order_id` values, while `record_count` tracks line item volume.
- **Average Order Value (AOV)**:
  $$ \text{average\_order\_value} = \frac{\sum \text{calculated\_net\_revenue}}{\text{distinct orders}} $$
- **Contribution Shares**:
  $$ \text{revenue\_contribution} = \frac{\text{segment\_net\_revenue}}{\text{portfolio\_total\_net\_revenue}} $$
  $$ \text{margin\_contribution} = \frac{\text{segment\_gross\_margin}}{\text{portfolio\_total\_gross\_margin}} $$
- **Missing values are not treated as zero**: If all records in a segment lack cost or revenue, the sum evaluates to `None` rather than 0.

---

## 7. Time Intelligence

Supports temporal slicing at daily, weekly, and monthly levels:
- **Daily**: Indexed by `YYYY-MM-DD`.
- **Weekly**: Indexed by `YYYY-Www` (calendar week).
- **Monthly**: Indexed by `YYYY-MM`.

Monthly metrics calculate gross revenue, net revenue, discounts, estimated COGS, gross margin, margin %, order counts, order-line counts, units sold, and average order value across consecutive calendar months.

---

## 8. Neutral Analytical Rankings

Rankings use purely descriptive, neutral analytical labels:
- `highest_revenue` / `lowest_revenue`
- `highest_margin` / `lowest_margin`
- `highest_units` / `lowest_units`
- `highest_margin_percentage` / `lowest_margin_percentage`
- `negative_margin` (segments with total gross margin $< 0$, ordered most negative first)

No subjective labels (e.g. "good", "bad", "underperforming") are assigned.

---

## 9. Negative / Zero Margin Analysis

Examines transactions and SKUs generating negative gross margin:
- Tracks order lines, units, gross revenue, estimated COGS, and commercial loss ($\sum |\text{gross\_margin}|$).
- Component breakdowns classify loss drivers into measurable components:
  1. `cost_exceeds_price`: Unit procurement cost exceeds list price even without discounts.
  2. `discount_exceeds_margin`: List price exceeds cost, but promotional discounts exceed the spread.
  3. `both_factors`: High cost combined with active promotional discounting.

---

## 10. Data Quality Audit & Anti-Leakage Compliance

### Validation Checks:
- Missing SKU identifier
- Missing, zero, or negative quantity
- Missing or negative unit price
- Missing or negative discount
- Missing or negative source revenue
- Missing or negative unit procurement cost
- Missing or blank currency
- Duplicate `sale_id` records
- Chronological future date leakage relative to `as_of_date`
- Currency discrepancies against `default_currency` (`USD`)

### Point-in-Time Anti-Leakage Filtering:
When `as_of_date` is supplied, all sales records dated after `as_of_date` are strictly excluded prior to analytical aggregation, guaranteeing reproducible historical backtesting.

---

## 11. Deterministic Identifiers

All generated entities use deterministic SHA-256 digests:
- Line Record: `FIN-REC-{sha256(sale_id|sku_id|order_id|date|currency)[:16]}`
- Dimension Segment: `FIN-SEG-{sha256(dimension|segment_key|as_of_date|currency)[:16]}`
- Portfolio Summary: `FIN-PORT-{sha256(record_count|as_of_date|currency)[:16]}`

---

## 12. Known Limitations & Future Architecture

1. **Standard Cost vs Lot Accounting**: `unit_cost` reflects catalog procurement pricing. Transaction-specific lot batch costing and warehouse-specific landing fees will be modeled in Phase 6B.
2. **Promotional Optimization**: Phase 6A computes historical discounts and discount rates; discount policy optimization belongs to Phase 6C.
3. **Multi-Currency FX**: The current canonical dataset is 100% `USD`. Multi-currency conversion tables will be integrated when cross-border feeds are introduced.
4. **Returns Impact**: Gross margin in Phase 6A reflects shipped gross revenue. Integration with Phase 5A returns to produce Net Margin after Return Allowances will occur in subsequent phases.
