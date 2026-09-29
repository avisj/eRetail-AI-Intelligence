# Data Directory

This directory contains data used by the AI Commerce Intelligence Platform.

## Directory Structure

- `raw/`: Unprocessed incoming raw files, batch dumps, or ingestion drops.
- `processed/`: Validated, normalized, and transformed datasets ready for modeling or analytics.
- `sample/`: Generated synthetic datasets used for local development, testing, and validation.

## Standard Entities

1. `sales.csv`: Historical transactional sales records.
2. `inventory.csv`: Daily inventory snapshots per SKU and warehouse.
3. `products.csv`: Product catalog metadata and pricing.
4. `warehouses.csv`: Warehouse facility locations and capacities.
5. `purchases.csv`: Purchase orders placed with suppliers and delivery tracking.
6. `returns.csv`: Customer return events linked to original sales orders.
7. `channels.csv`: Sales channels (e.g. Amazon, Direct Website, Wholesale).
8. `suppliers.csv`: Vendor master data with lead times and minimum order quantities.

## Data Governance & Privacy

- **No company/customer data**: Only synthetic data is permitted in this repository.
- **Git safety**: Large datasets in `raw/` and `processed/` are excluded via `.gitignore`.
