# Phase 6D — Cost Completeness & Operational Economics

## Executive Summary

Phase 6D advances the financial intelligence capabilities of the **eRetail AI Intelligence** platform from commercial gross margin to a rigorous, auditable **Operational Economics** framework.

> [!IMPORTANT]
> **Notice**: This is an analytical operational-cost model, not accounting software or an accounting ledger. It provides deterministic managerial economic analysis rather than financial reporting ledgers.

While Phases 6A through 6C focused on gross revenue, discounts, procurement product cost (COGS), commercial gross margin, and profitability attribution, Phase 6D provides the analytical operational cost model that calculates order-level and unit-level contribution margin:

$$\text{Gross Revenue} \longrightarrow \text{Net Revenue} \longrightarrow \text{Product Cost} \longrightarrow \text{Gross Margin} \longrightarrow \text{Operational Costs} \longrightarrow \text{Contribution Margin}$$

Phase 6D answers the foundational executive questions:
1. **What is the operational contribution margin after fulfillment freight, payment fees, packaging, handling labor, and reverse logistics?**
2. **Which cost components are directly observed from source records, which are estimated from catalog baselines, which are derived from configured operational assumptions, and which are currently unobserved?**
3. **How complete is our operational cost data across transactions, orders, and sales channels?**
4. **How are multi-line order expenses (e.g. flat order shipping and fixed payment fees) allocated across order lines without duplication or artificial inflation?**
5. **What is the empirical impact of actual reverse logistics events on order profitability when fee structures are modeled?**

All metrics are deterministic, factual calculations. No black-box machine learning, generative AI, LLMs, or autonomous business actions are utilized.

---

## 1. Operational Cost Taxonomy & Provenance

To prevent silent errors, hidden subsidies, and artificial profit inflation, Phase 6D enforces a strict provenance taxonomy for all 7 standard cost components:

| Cost Component (`CostComponent`) | Natural Grain (`CostGrain`) | Description |
| :--- | :--- | :--- |
| `PRODUCT_COST` | `UNIT` / `TRANSACTION` | Standard procurement cost of goods sold (COGS) |
| `SHIPPING_COST` | `TRANSACTION` / `ORDER` | Outbound freight and carrier delivery expenses |
| `PAYMENT_PROCESSING_COST` | `TRANSACTION` / `ORDER` | Payment gateway transaction rate fees and fixed authorization fees |
| `PACKAGING_COST` | `UNIT` / `ORDER` | Outer boxes, poly mailers, dunnage, thermal labels, and packing materials |
| `WAREHOUSE_HANDLING_COST` | `UNIT` / `ORDER` | Fulfillment center pick, pack, and handling labor |
| `RETURN_PROCESSING_COST` | `RETURN_EVENT` / `TRANSACTION` | Reverse logistics shipping, return inspection, and restocking handling |
| `OTHER_VARIABLE_COST` | `UNIT` / `TRANSACTION` | Miscellaneous variable fees, channel listing commissions, or royalties |

### 1.1 Provenance Hierarchy (`CostSourceType`)

Every cost entry is tagged with its definitive origin:
1. **`SOURCE_DATA` (Observed / Source Costs)**: The cost is observed directly from operational feeds or transaction tables (e.g. carrier invoices or gateway settlements).
2. **`CATALOG_ESTIMATE` (Catalog Estimates)**: The cost is derived from verified master catalog tables (e.g. standard product procurement costs in `products.csv`).
3. **`CONFIGURED_ASSUMPTION` (Configured Assumptions)**: The cost is derived from explicit, configurable operational assumptions (e.g. parametric shipping rate tables or payment gateway schedules).
4. **`UNAVAILABLE` (Unavailable Costs)**: The cost is unobserved and unconfigured.

### 1.2 Strict Separation of Observed Zero ($0.00) vs Missing (`None`)

A critical data governance rule in Phase 6D:
- **Observed Zero Cost (`$0.00`)**: Indicates an affirmative operational condition where no fee was incurred (e.g., promotional free freight verified in source data).
- **Missing Cost (`None`)**: Indicates an unobserved data gap (e.g., packaging material cost is not tracked in the source feed). Marked as `status = UNAVAILABLE` and `source_type = UNAVAILABLE`. Missing costs are **never** silently coerced to zero in completeness auditing.

### 1.3 Return Event Occurrence vs Monetary Processing Cost
Observed return records in `returns.csv` (e.g. `return_id`, `quantity`, `reason`) represent return event occurrences, not monetary return-processing costs. Therefore, if there is no source monetary return-processing fee and no configured return-processing fee rate, `RETURN_PROCESSING_COST` remains strictly `UNAVAILABLE`.

Zero return events cannot be treated as evidence that the business's return-processing cost is known to be $0.00 unless the configured operational cost model explicitly defines a monetary return-processing rate policy.

---

## 2. Cost Completeness Architecture

To govern financial reporting readiness, Phase 6D introduces an objective **Cost Completeness Model**.

### 2.1 Completeness Formulation
For any configured set of required cost components $R = \{c_1, c_2, \dots, c_k\}$:

$$\text{cost\_completeness\_pct} = \left(\frac{\sum_{c \in R} \mathbb{I}(c \text{ satisfies completeness policy})}{|R|}\right) \times 100$$

Where satisfaction is governed by `OperationalEconomicsConfig`:
- **Strict Source Only (`strict_source_only = True`)**: Only `SOURCE_DATA` satisfies completeness.
- **Standard Governance (`strict_source_only = False`)**:
  - `SOURCE_DATA` always satisfies completeness.
  - `CATALOG_ESTIMATE` satisfies completeness if `allow_estimated_costs_for_completeness = True` (default: `True`).
  - `CONFIGURED_ASSUMPTION` satisfies completeness if `allow_assumed_costs_for_completeness = True` (default: `False`).

### 2.2 Cost Completeness Statuses (`CostCompletenessStatus`)
- **`COMPLETE`**: $\text{cost\_completeness\_pct} \ge 100.0\%$ (all required cost components satisfied).
- **`PARTIALLY_COMPLETE`**: $0.0\% < \text{cost\_completeness\_pct} < 100.0\%$.
- **`INSUFFICIENT_DATA`**: $\text{cost\_completeness\_pct} == 0.0\%$.

### 2.3 Two-Tier Contribution Margin Governance
To ensure that operational economics metrics remain auditable and prevent under-costed profit overstatements:
1. **Known Contribution Margin (`known_contribution_margin`)**:
   $$\text{known\_contribution\_margin} = \text{net\_revenue} - \text{product\_cost} - \sum \text{available\_variable\_costs}$$
   Calculable whenever net revenue and product cost are observed. Missing costs are treated as 0 in this interim subtotal, with full disclosure in data completeness reporting.
2. **Final Contribution Margin (`final_contribution_margin`)**:
   $$\text{final\_contribution\_margin} = \begin{cases} \text{known\_contribution\_margin}, & \text{if } \text{cost\_completeness\_pct} \ge \text{min\_completeness\_threshold} \\ \text{None}, & \text{otherwise} \end{cases}$$
   When incomplete, `final_contribution_margin_status` is explicitly set to `INSUFFICIENT_COST_DATA`.

---

## 3. Multi-Line Order Handling & Mathematical Invariance

Orders frequently contain multiple line items. Order-level expenses—such as a flat outbound shipping charge (e.g. \$4.50 per order) or a fixed payment authorization fee (e.g. \$0.30 per order)—must not be duplicated across order lines.

### 3.1 Allocation Methodology
For order line $j$ belonging to order $O$:
$$\text{line\_allocated\_cost}_j = \text{order\_cost}_O \times w_j$$

Where allocation weight $w_j$ is determined by `order_cost_allocation_method`:
1. **`NET_REVENUE` (Default)**:
   $$w_j = \frac{\text{net\_revenue}_j}{\sum_{k \in O} \text{net\_revenue}_k} \quad (\text{if } \sum \text{net\_revenue} > 0)$$
   Fallback to quantity weighting if order net revenue $\le 0$, then equal split.
2. **`QUANTITY`**:
   $$w_j = \frac{\text{quantity}_j}{\sum_{k \in O} \text{quantity}_k} \quad (\text{if } \sum \text{quantity} > 0)$$
3. **`EQUAL`**:
   $$w_j = \frac{1}{|O|}$$

### 3.2 Invariance Invariant
$$\sum_{j \in O} \text{line\_allocated\_cost}_j \equiv \text{order\_cost}_O$$

This guarantees that:
- Order-level costs are never multiplied per line.
- The sum of transaction line economics exactly matches the order-level economics down to the cent.
- Distinct order counts and average order values remain statistically and mathematically exact.

---

## 4. Reverse Logistics & Return Cost Integration

Unlike models that ignore returns or use predictive machine learning estimates, Phase 6D integrates actual observed return events when monetary fee policies are modeled:

1. **Returns Dataset Matching**:
   Sales transactions are joined to `returns_df` on `(order_id, sku_id)` (or `order_id`).
2. **Observed Return Lines (Under Fee Policy)**:
   If an order line had verified returned units and a return fee assumption is configured:
   $$\text{return\_cost} = (\text{return\_cost\_per\_returned\_unit} \times \text{returned\_units}) + (\text{return\_cost\_per\_return\_event} \times \text{events})$$
   Tagged as `CONFIGURED_ASSUMPTION` (or `SOURCE_DATA` if observed in source data).
3. **Verified Non-Returned Lines (Under Fee Policy)**:
   If a return fee policy is configured and an order line experienced **no** return events in `returns_df`, return cost is **\$0.00** and tagged as `CONFIGURED_ASSUMPTION`.
4. **Unconfigured Fee Policy**:
   If no return processing fee assumption is configured and no source return fee exists, return cost is marked as `None` / `UNAVAILABLE` across all rows.
5. **Anti-Leakage Protection**:
   Returns occurring after `as_of_date` are filtered out during historical point-in-time analyses.

---

## 5. Canonical Dataset Empirical Benchmark Findings

The Phase 6D engine was benchmarked on the complete canonical synthetic eRetail dataset:
- **Sales Transactions**: 773,555 order lines
- **Product Catalog**: 500 SKUs
- **Fulfillment Warehouses**: 5 facilities
- **Commercial Channels**: 4 channels
- **Returns Repository**: 62,630 return records

### 5.1 Baseline Commercial Performance (Phase 6A/6B Traceability)
- **Total Physical Units Sold**: 2,171,966 units
- **Total Sales Orders**: 773,555 orders
- **Gross Revenue**: \$279,544,322.20
- **Promotional Discounts**: \$4,580,417.24 (1.64% discount rate)
- **Net Realized Revenue**: \$274,963,904.96
- **Product Procurement Cost (COGS)**: \$137,313,272.23
- **Gross Commercial Margin**: \$137,650,632.73 (**50.06%**)

### 5.2 Benchmark Run 1: Raw Canonical Data (Strict Governance, No Assumptions)
In the raw canonical dataset, only product procurement cost (`CATALOG_ESTIMATE`) is available. Outbound freight, payment processing, packaging, warehouse handling labor, and return processing cost are unobserved in the source data and have no configured assumptions:

| Metric | Empirical Value | Governance State |
| :--- | :--- | :--- |
| Total Known Variable Costs | \$0.00 | Missing costs not assumed as zero |
| Known Contribution Margin | \$137,650,632.73 | 50.06% of Net Revenue |
| Final Contribution Margin | `None` | **Blocked by Completeness Gate** |
| Final Contribution Margin Status | `INSUFFICIENT_COST_DATA` | Compliant with financial policy |
| Cost Completeness | **16.67%** | **1 of 6 required components available (`PRODUCT_COST`)** |
| Available Components | `['PRODUCT_COST']` | Catalog estimate from `products.csv` |
| Unavailable Components | `['SHIPPING_COST', 'PAYMENT_PROCESSING_COST', 'PACKAGING_COST', 'WAREHOUSE_HANDLING_COST', 'RETURN_PROCESSING_COST', 'OTHER_VARIABLE_COST']` | Unobserved in source data |
| Execution Runtime (773k rows) | **6.50 seconds** | Fully vectorized execution |

### 5.3 Benchmark Run 2: Analytical Scenario Simulation (Configured Assumptions)
> [!NOTE]
> The configured-assumption benchmark is an analytical operational-cost simulation demonstrating model behavior under explicit assumptions. It is NOT presented as actual business profitability.

Applying realistic operational fulfillment assumptions:
- Outbound shipping: \$4.50 flat per order + \$0.50 per unit
- Payment processing: 2.50% rate + \$0.30 fixed fee per order
- Packaging materials: \$0.75 per order
- Warehouse handling labor: \$1.20 per unit
- Reverse logistics fee: \$8.50 per returned unit

| Economic Stage | Aggregate Amount (\$) | % of Net Revenue | Provenance / Methodology |
| :--- | :--- | :--- | :--- |
| **Net Realized Revenue** | **\$274,963,904.96** | **100.00%** | `SOURCE_DATA` |
| Product Procurement Cost | -\$137,313,272.23 | 49.94% | `CATALOG_ESTIMATE` |
| **Commercial Gross Margin** | **\$137,650,632.73** | **50.06%** | Baseline Commercial Margin |
| Outbound Shipping Cost | -\$4,566,980.50 | 1.66% | Multi-line allocated order freight |
| Payment Processing Fees | -\$7,106,164.12 | 2.58% | Gateway rate + allocated fixed fees |
| Packaging Materials Cost | -\$580,166.25 | 0.21% | Allocated packaging box & dunnage |
| Warehouse Handling Labor | -\$2,606,359.20 | 0.95% | Facility pick/pack unit labor |
| Return Reverse Logistics | -\$990,930.00 | 0.36% | Verified returned units (\$8.50/unit) |
| **Total Operational Costs** | **-\$15,850,600.07** | **5.76%** | Sum of operational variable expenses |
| **Final Contribution Margin** | **\$121,800,032.66** | **44.30%** | **Populated & Fully Complete** |
| Cost Completeness | **100.00%** | `COMPLETE` | All required components satisfied under simulation policy |
| Execution Runtime (773k rows) | **7.28 seconds** | High-performance pandas vectorization |

### 5.4 Order-Level Aggregation Invariance Verification
- Distinct Orders Aggregated: **773,555 orders**
- Order Net Revenue Sum: **\$274,963,904.96** (matches line total to \$0.00 drift)
- Order Shipping Cost Sum: **\$4,566,980.50** (matches line allocated total to \$0.00 drift)
- Order Final Contribution Margin Sum: **\$121,800,032.66** (matches line total to \$0.00 drift)
- Computation Time: **8.47 seconds** across 773,555 multi-line orders.

### 5.5 Channel & Warehouse Operational Economics Breakdown

#### Commercial Channels
| Channel ID | Net Revenue (\$) | Product Cost (\$) | Variable Costs (\$) | Contribution Margin (\$) | CM % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `CH_AMZ` (Amazon) | \$123,532,522.60 | \$61,685,499.81 | \$7,117,240.31 | \$54,729,782.48 | 44.30% |
| `CH_DIR` (Direct) | \$68,794,994.48 | \$34,334,051.46 | \$3,966,598.46 | \$30,494,344.56 | 44.33% |
| `CH_FLK` (Flipkart) | \$41,436,049.46 | \$20,719,517.59 | \$2,387,273.39 | \$18,329,258.48 | 44.24% |
| `CH_MYN` (Myntra) | \$41,200,338.42 | \$20,574,203.37 | \$2,379,487.91 | \$18,246,647.14 | 44.29% |

#### Fulfillment Warehouses
| Warehouse ID | Net Revenue (\$) | Product Cost (\$) | Variable Costs (\$) | Contribution Margin (\$) | CM % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `WH_CENTRAL_01` | \$55,053,395.10 | \$27,495,338.93 | \$3,172,541.93 | \$24,385,514.24 | 44.29% |
| `WH_EAST_01` | \$54,976,590.36 | \$27,451,417.18 | \$3,169,924.26 | \$24,355,248.92 | 44.30% |
| `WH_NORTH_01` | \$54,936,977.36 | \$27,429,549.72 | \$3,163,665.73 | \$24,343,761.91 | 44.31% |
| `WH_SOUTH_01` | \$55,118,931.33 | \$27,533,517.42 | \$3,174,237.33 | \$24,411,176.58 | 44.29% |
| `WH_WEST_01` | \$54,878,010.81 | \$27,403,448.98 | \$3,170,230.82 | \$24,304,331.01 | 44.29% |

---

## 6. Architecture & Service Orchestration

Phase 6D is encapsulated in `OperationalEconomicsService`, offering five core programmatic capabilities:
1. `calculate_transaction_economics`: Vectorized transaction line computation.
2. `calculate_order_economics`: Order-level economics aggregating true order costs.
3. `calculate_portfolio_economics`: Network-level executive summary and cost completeness audit.
4. `calculate_segment_economics`: Dimensional aggregation (SKU, Category, Brand, Channel, Warehouse, cross-slices).
5. `calculate_temporal_economics`: Time series aggregations across daily, weekly, monthly, quarterly, and yearly grains.
6. `calculate_cost_completeness`: Dedicated standalone audit of cost availability and provenance.

---

## 7. Quality Assurance & Regression Protection

The implementation is verified by **42 unit and integration tests** in `tests/test_operational_economics.py` validating conditions A through AK.

The repository test suite stands at:
- **506 passing tests** (464 baseline tests + 42 Phase 6D tests)
- **0 test failures**
- **0 linting or syntax errors**
- **2 pre-existing statsmodels warnings** (Holt-Winters divide by zero in constant series, preserved without modification)
