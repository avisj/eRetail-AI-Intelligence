# Data Dictionary: Standard Commerce Schema

This document provides the field-level specification, datatypes, nullability, constraints, and descriptions for all entities in the Standard Commerce Data Contract.

---

## 1. Sales Transactions (`sales`)

| Field Name | Type | Nullable | Constraints | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sale_id` | String | No | Unique PK | Globally unique identifier for the transaction record | `"SALE_0001024"` |
| `order_id` | String | No | Non-empty | Order identifier (can group multiple line items) | `"ORD_89201"` |
| `date` | Date / DateTime | No | ISO 8601 | Timestamp or date when the order was confirmed | `"2025-06-15"` |
| `sku_id` | String | No | FK -> Products | Product SKU identifier | `"SKU_ELEC_0042"` |
| `warehouse_id` | String | No | FK -> Warehouses | Fulfillment node originating the shipment | `"WH_CENTRAL_01"` |
| `channel_id` | String | No | FK -> Channels | Sales channel originating the order | `"CH_AMAZON"` |
| `quantity` | Integer | No | > 0 | Units purchased | `2` |
| `unit_price` | Float | No | >= 0.0 | Unit list or gross price | `49.99` |
| `discount` | Float | No | >= 0.0 | Total discount applied to this line item | `5.00` |
| `revenue` | Float | No | >= 0.0 | Net realized revenue: `(qty * unit_price) - discount` | `94.98` |
| `currency` | String | No | 3-letter ISO | Currency code | `"USD"` |

---

## 2. Inventory Snapshots (`inventory`)

| Field Name | Type | Nullable | Constraints | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `snapshot_date` | Date | No | ISO 8601 | End-of-day date for the snapshot | `"2025-06-15"` |
| `sku_id` | String | No | FK -> Products | Product SKU identifier | `"SKU_ELEC_0042"` |
| `warehouse_id` | String | No | FK -> Warehouses | Warehouse facility identifier | `"WH_CENTRAL_01"` |
| `available_qty` | Integer | No | >= 0 | Stock on hand ready for allocation/pick | `120` |
| `reserved_qty` | Integer | No | >= 0 | Stock allocated to unfulfilled orders | `15` |
| `in_transit_qty` | Integer | No | >= 0 | Inbound purchase or transfer stock in transit | `50` |
| `damaged_qty` | Integer | No | >= 0 | Quarantined, damaged, or unfulfillable units | `2` |

---

## 3. Product Catalog (`products`)

| Field Name | Type | Nullable | Constraints | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sku_id` | String | No | Unique PK | Product Stock Keeping Unit | `"SKU_ELEC_0042"` |
| `product_name` | String | No | Non-empty | Descriptive commercial title | `"Wireless Noise-Canceling Headphones"`|
| `category_id` | String | No | Non-empty | Product category or hierarchy tier | `"Electronics"` |
| `brand` | String | Yes | None | Brand manufacturer name | `"AcousticPro"` |
| `unit_cost` | Float | No | >= 0.0 | Landed cost of goods sold (COGS) | `24.50` |
| `selling_price`| Float | No | >= 0.0 | Standard retail selling price (MSRP) | `49.99` |
| `currency` | String | No | 3-letter ISO | Currency code | `"USD"` |

---

## 4. Warehouses (`warehouses`)

| Field Name | Type | Nullable | Constraints | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `warehouse_id` | String | No | Unique PK | Facility code | `"WH_CENTRAL_01"` |
| `warehouse_name`| String| No | Non-empty | Facility descriptive name | `"Dallas Central Fulfillment Hub"` |
| `city` | String | No | Non-empty | Facility city location | `"Dallas"` |
| `state` | String | No | Non-empty | State or regional territory | `"TX"` |
| `country` | String | No | 2 or 3 letter | Country code / name | `"USA"` |
| `capacity_units`| Integer| Yes| >= 0 | Total storage capacity in standard units | `500000` |

---

## 5. Purchase Orders (`purchases`)

| Field Name | Type | Nullable | Constraints | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `purchase_order_id`| String | No | Non-empty | PO number or line identifier | `"PO_2025_0041"` |
| `order_date` | Date | No | ISO 8601 | Date PO was issued to supplier | `"2025-05-10"` |
| `sku_id` | String | No | FK -> Products | Ordered product SKU | `"SKU_ELEC_0042"` |
| `supplier_id` | String | No | FK -> Suppliers| Vendor supplying the goods | `"SUPP_AUDIO_01"` |
| `warehouse_id` | String | No | FK -> Warehouses| Receiving warehouse | `"WH_CENTRAL_01"` |
| `quantity` | Integer | No | > 0 | Total units ordered | `500` |
| `unit_cost` | Float | No | >= 0.0 | Negotiated purchase unit cost | `24.50` |
| `expected_delivery_date` | Date | No | >= `order_date` | Projected inbound arrival date | `"2025-05-25"` |
| `actual_delivery_date` | Date | Yes| None | Actual dock receipt date (null if pending) | `"2025-05-26"` |
| `status` | String | No | PENDING, IN_TRANSIT, DELIVERED, CANCELLED | Order fulfillment state | `"DELIVERED"` |

---

## 6. Customer Returns (`returns`)

| Field Name | Type | Nullable | Constraints | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `return_id` | String | No | Unique PK | Return record identifier | `"RET_90214"` |
| `order_id` | String | No | Non-empty | Original customer order reference | `"ORD_89201"` |
| `return_date` | Date | No | ISO 8601 | Date return request was logged | `"2025-06-20"` |
| `sku_id` | String | No | FK -> Products | Returned SKU identifier | `"SKU_ELEC_0042"` |
| `warehouse_id` | String | No | FK -> Warehouses| Receiving reverse-logistics facility | `"WH_CENTRAL_01"` |
| `quantity` | Integer | No | > 0 | Units returned | `1` |
| `reason` | String | No | Non-empty | Reason categorized (e.g. Defective, Wrong Size) | `"Customer Dissatisfaction"` |
| `channel_id` | String | No | FK -> Channels | Originating sales channel | `"CH_AMAZON"` |

---

## 7. Sales Channels (`channels`)

| Field Name | Type | Nullable | Constraints | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `channel_id` | String | No | Unique PK | Standard channel key | `"CH_AMAZON"` |
| `channel_name` | String | No | Non-empty | Name of the platform or channel | `"Amazon US"` |
| `channel_type` | String | No | Marketplace, Direct, B2B, Retail | Structural classification | `"Marketplace"` |

---

## 8. Suppliers (`suppliers`)

| Field Name | Type | Nullable | Constraints | Description | Example |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `supplier_id` | String | No | Unique PK | Vendor unique identifier | `"SUPP_AUDIO_01"` |
| `supplier_name` | String | No | Non-empty | Legal supplier trading name | `"Sonic Wave Manufacturing"` |
| `average_lead_time_days` | Integer | No | > 0 | Expected manufacturing & transit days | `14` |
| `minimum_order_quantity` | Integer | No | >= 1 | Minimum batch size per purchase order | `100` |
