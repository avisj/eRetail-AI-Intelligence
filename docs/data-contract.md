# Vendor-Neutral Ecommerce Data Contract

## 1. Scope & Objective

The **Standard Commerce Data Contract** defines the canonical schema, types, constraints, and relational rules that all incoming ecommerce datasets must satisfy before entering the platform's analytical or predictive layers.

By enforcing an invariant, vendor-neutral contract, the platform decouples all intelligence pipelines (forecasting, stockout detection, replenishment, agent reasoning) from external OMS, WMS, and ERP proprietary structures.

---

## 2. Core Entities & Relational Map

The contract defines eight core domain entities:

```
                  +---------------+
                  |   Suppliers   |
                  +---------------+
                          | (1:N)
                          v
+---------------+ (1:N) +---------------+ (N:1) +---------------+
|   Channels    | <---- |     Sales     | ----> |   Products    |
+---------------+       +---------------+       +---------------+
        ^                       | (1:N)                 ^
        |                       v                       |
        |               +---------------+               |
        +-------------- |    Returns    |               |
                        +---------------+               |
                                                        |
+---------------+ (1:N) +---------------+ (N:1)         |
|  Warehouses   | <---- |   Inventory   | --------------+
+---------------+       +---------------+               |
        ^                                               |
        | (1:N)         +---------------+ (N:1)         |
        +-------------- |   Purchases   | --------------+
                        +---------------+
```

---

## 3. Entity Specifications

### 3.1 Sales Transactions (`sales`)
Transactional record of confirmed customer orders.
* **Grain**: 1 row per order item (Order ID + SKU ID).
* **Primary Key**: `sale_id`
* **Foreign Keys**: `sku_id -> products.sku_id`, `warehouse_id -> warehouses.warehouse_id`, `channel_id -> channels.channel_id`
* **Invariant**: `quantity > 0`, `unit_price >= 0`, `discount >= 0`, `revenue >= 0`

### 3.2 Inventory Snapshots (`inventory`)
Daily periodic snapshot of stock status per SKU per fulfillment node.
* **Grain**: 1 row per date per SKU per warehouse.
* **Composite Primary Key**: `(snapshot_date, sku_id, warehouse_id)`
* **Foreign Keys**: `sku_id -> products.sku_id`, `warehouse_id -> warehouses.warehouse_id`
* **Invariant**: `available_qty >= 0`, `reserved_qty >= 0`, `in_transit_qty >= 0`, `damaged_qty >= 0`

### 3.3 Product Catalog (`products`)
Master catalog defining SKU specifications, cost, and baseline pricing.
* **Grain**: 1 row per SKU.
* **Primary Key**: `sku_id`
* **Invariant**: `unit_cost >= 0`, `selling_price >= 0`

### 3.4 Warehouses / Fulfillment Centers (`warehouses`)
Physical or third-party fulfillment nodes holding stock.
* **Grain**: 1 row per facility.
* **Primary Key**: `warehouse_id`
* **Optional Fields**: `capacity_units` (allows systems without physical volume data to integrate smoothly)

### 3.5 Purchase Orders (`purchases`)
Inbound inventory procurement orders placed with suppliers.
* **Grain**: 1 row per purchase order line.
* **Primary Key**: `purchase_order_id`
* **Foreign Keys**: `sku_id -> products.sku_id`, `supplier_id -> suppliers.supplier_id`, `warehouse_id -> warehouses.warehouse_id`
* **Invariant**: `quantity > 0`, `unit_cost >= 0`, `expected_delivery_date >= order_date`

### 3.6 Customer Returns (`returns`)
Post-purchase product return and reverse logistics records.
* **Grain**: 1 row per return event.
* **Primary Key**: `return_id`
* **Foreign Keys**: `order_id -> sales.order_id`, `sku_id -> products.sku_id`, `warehouse_id -> warehouses.warehouse_id`, `channel_id -> channels.channel_id`
* **Invariant**: `quantity > 0`, `return_date >= order date`

### 3.7 Sales Channels (`channels`)
Sales channels, online storefronts, marketplaces, or retail outlets.
* **Grain**: 1 row per channel.
* **Primary Key**: `channel_id`

### 3.8 Suppliers (`suppliers`)
Vendors and manufacturers supplying stock.
* **Grain**: 1 row per supplier.
* **Primary Key**: `supplier_id`
* **Invariant**: `average_lead_time_days > 0`, `minimum_order_quantity >= 1`

---

## 4. Contract Enforcement & Data Quality Rules

1. **Identifier Stability**: All entities use immutable string identifiers (`sku_id`, `warehouse_id`, etc.) rather than descriptive names for joins.
2. **Referential Integrity**: Every foreign key reference in `sales`, `inventory`, `purchases`, and `returns` must exist in its respective parent master dataset.
3. **No Silent Type Coercion**: Data types must strictly conform to schemas. Ingestion pipelines fail fast or isolate defective records in quarantine rather than silently mutating data.
4. **Extensibility**: All schemas allow arbitrary additive optional fields (e.g. metadata tags, localized descriptions, channel specific fees) without breaking core pipelines.
