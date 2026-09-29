"""Data Validation and Quality Assurance Engine.

Implements structured validation for standard commerce datasets:
- Schema and required column enforcement
- Null and blank ID detection
- Duplicate primary key detection
- Date format and chronological consistency
- Quantity and monetary value range validation
- Cross-entity foreign key referential integrity
- Inventory consistency rules
- Comprehensive data quality reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd


@dataclass
class ValidationErrorDetail:
    """Specific validation issue with context."""

    entity: str
    issue: str
    field_name: Optional[str] = None
    row_index: Optional[int] = None
    invalid_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "field": self.field_name,
            "row_index": self.row_index,
            "issue": self.issue,
            "invalid_value": str(self.invalid_value) if self.invalid_value is not None else None,
        }


@dataclass
class ValidationResult:
    """Structured validation outcome."""

    valid: bool = True
    errors: List[ValidationErrorDetail] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def add_error(
        self,
        entity: str,
        issue: str,
        field_name: Optional[str] = None,
        row_index: Optional[int] = None,
        invalid_value: Optional[Any] = None,
    ) -> None:
        self.valid = False
        self.errors.append(
            ValidationErrorDetail(
                entity=entity,
                issue=issue,
                field_name=field_name,
                row_index=row_index,
                invalid_value=invalid_value,
            )
        )

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge another ValidationResult into this one."""
        self.valid = self.valid and other.valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.statistics.update(other.statistics)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "statistics": self.statistics,
        }

    def summary(self) -> str:
        status = "PASSED" if self.valid else "FAILED"
        return f"Validation {status}: {len(self.errors)} error(s), {len(self.warnings)} warning(s)."


# Required columns specification
REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "sales": [
        "sale_id",
        "order_id",
        "date",
        "sku_id",
        "warehouse_id",
        "channel_id",
        "quantity",
        "unit_price",
        "discount",
        "revenue",
        "currency",
    ],
    "inventory": [
        "snapshot_date",
        "sku_id",
        "warehouse_id",
        "available_qty",
        "reserved_qty",
        "in_transit_qty",
        "damaged_qty",
    ],
    "products": [
        "sku_id",
        "product_name",
        "category_id",
        "unit_cost",
        "selling_price",
        "currency",
    ],
    "warehouses": [
        "warehouse_id",
        "warehouse_name",
        "city",
        "state",
        "country",
    ],
    "purchases": [
        "purchase_order_id",
        "order_date",
        "sku_id",
        "supplier_id",
        "warehouse_id",
        "quantity",
        "unit_cost",
        "expected_delivery_date",
        "status",
    ],
    "returns": [
        "return_id",
        "order_id",
        "return_date",
        "sku_id",
        "warehouse_id",
        "quantity",
        "reason",
        "channel_id",
    ],
    "channels": [
        "channel_id",
        "channel_name",
        "channel_type",
    ],
    "suppliers": [
        "supplier_id",
        "supplier_name",
        "average_lead_time_days",
        "minimum_order_quantity",
    ],
}

PRIMARY_KEYS: Dict[str, List[str]] = {
    "sales": ["sale_id"],
    "inventory": ["snapshot_date", "sku_id", "warehouse_id"],
    "products": ["sku_id"],
    "warehouses": ["warehouse_id"],
    "purchases": ["purchase_order_id"],
    "returns": ["return_id"],
    "channels": ["channel_id"],
    "suppliers": ["supplier_id"],
}


class CommerceDataValidator:
    """Comprehensive validator for ecommerce datasets."""

    def __init__(self, max_error_samples: int = 50):
        self.max_error_samples = max_error_samples

    def validate_entity(self, entity_name: str, df: pd.DataFrame) -> ValidationResult:
        """Validate a single entity dataframe."""
        result = ValidationResult()
        result.statistics[f"{entity_name}_rows"] = len(df)
        result.statistics[f"{entity_name}_cols"] = len(df.columns)

        if entity_name not in REQUIRED_COLUMNS:
            result.add_warning(f"No schema rule registered for entity '{entity_name}'. Skipping schema check.")
            return result

        # 1. Missing columns
        required = REQUIRED_COLUMNS[entity_name]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            result.add_error(
                entity=entity_name,
                issue=f"Missing required columns: {missing_cols}",
            )
            # Cannot proceed with deep column checks if columns are missing
            return result

        # 2. Null or blank primary keys
        pk_cols = PRIMARY_KEYS.get(entity_name, [])
        for pk in pk_cols:
            null_mask = df[pk].isna() | (df[pk].astype(str).str.strip() == "")
            null_count = null_mask.sum()
            if null_count > 0:
                bad_indices = df.index[null_mask][:5].tolist()
                result.add_error(
                    entity=entity_name,
                    field_name=pk,
                    issue=f"Found {null_count} null or blank primary key values in column '{pk}'",
                    row_index=bad_indices[0] if bad_indices else None,
                )

        # 3. Duplicate primary keys
        if pk_cols:
            dup_mask = df.duplicated(subset=pk_cols, keep=False)
            dup_count = dup_mask.sum()
            if dup_count > 0:
                bad_indices = df.index[dup_mask][:5].tolist()
                result.add_error(
                    entity=entity_name,
                    issue=f"Found {dup_count} duplicate primary key records on {pk_cols}",
                    row_index=bad_indices[0] if bad_indices else None,
                )

        # 4. Entity-specific domain validations
        if entity_name == "sales":
            self._validate_sales_domain(df, result)
        elif entity_name == "inventory":
            self._validate_inventory_domain(df, result)
        elif entity_name == "products":
            self._validate_products_domain(df, result)
        elif entity_name == "warehouses":
            self._validate_warehouses_domain(df, result)
        elif entity_name == "purchases":
            self._validate_purchases_domain(df, result)
        elif entity_name == "returns":
            self._validate_returns_domain(df, result)
        elif entity_name == "suppliers":
            self._validate_suppliers_domain(df, result)

        return result

    def _validate_sales_domain(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Domain validations for sales."""
        # Quantity > 0
        bad_qty = df[df["quantity"] <= 0]
        if not bad_qty.empty:
            result.add_error(
                entity="sales",
                field_name="quantity",
                issue=f"Found {len(bad_qty)} sales records with non-positive quantity",
                row_index=bad_qty.index[0],
                invalid_value=bad_qty["quantity"].iloc[0],
            )

        # Unit price >= 0
        bad_price = df[df["unit_price"] < 0]
        if not bad_price.empty:
            result.add_error(
                entity="sales",
                field_name="unit_price",
                issue=f"Found {len(bad_price)} sales records with negative unit_price",
                row_index=bad_price.index[0],
                invalid_value=bad_price["unit_price"].iloc[0],
            )

        # Revenue >= 0
        bad_rev = df[df["revenue"] < 0]
        if not bad_rev.empty:
            result.add_error(
                entity="sales",
                field_name="revenue",
                issue=f"Found {len(bad_rev)} sales records with negative revenue",
                row_index=bad_rev.index[0],
                invalid_value=bad_rev["revenue"].iloc[0],
            )

        # Date validation
        try:
            pd.to_datetime(df["date"], errors="raise")
        except Exception as e:
            result.add_error(
                entity="sales",
                field_name="date",
                issue=f"Invalid date format in sales records: {e}",
            )

    def _validate_inventory_domain(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Domain validations for inventory."""
        for qty_col in ["available_qty", "reserved_qty", "in_transit_qty", "damaged_qty"]:
            bad = df[df[qty_col] < 0]
            if not bad.empty:
                result.add_error(
                    entity="inventory",
                    field_name=qty_col,
                    issue=f"Found {len(bad)} inventory snapshots with negative {qty_col}",
                    row_index=bad.index[0],
                    invalid_value=bad[qty_col].iloc[0],
                )

        try:
            pd.to_datetime(df["snapshot_date"], errors="raise")
        except Exception as e:
            result.add_error(
                entity="inventory",
                field_name="snapshot_date",
                issue=f"Invalid date format in inventory snapshots: {e}",
            )

    def _validate_products_domain(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Domain validations for products."""
        bad_cost = df[df["unit_cost"] < 0]
        if not bad_cost.empty:
            result.add_error(
                entity="products",
                field_name="unit_cost",
                issue=f"Found {len(bad_cost)} products with negative unit_cost",
                row_index=bad_cost.index[0],
                invalid_value=bad_cost["unit_cost"].iloc[0],
            )

        bad_price = df[df["selling_price"] < 0]
        if not bad_price.empty:
            result.add_error(
                entity="products",
                field_name="selling_price",
                issue=f"Found {len(bad_price)} products with negative selling_price",
                row_index=bad_price.index[0],
                invalid_value=bad_price["selling_price"].iloc[0],
            )

    def _validate_warehouses_domain(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Domain validations for warehouses."""
        if "capacity_units" in df.columns:
            bad_cap = df[df["capacity_units"].notna() & (df["capacity_units"] < 0)]
            if not bad_cap.empty:
                result.add_error(
                    entity="warehouses",
                    field_name="capacity_units",
                    issue=f"Found {len(bad_cap)} warehouses with negative capacity_units",
                    row_index=bad_cap.index[0],
                    invalid_value=bad_cap["capacity_units"].iloc[0],
                )

    def _validate_purchases_domain(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Domain validations for purchases."""
        bad_qty = df[df["quantity"] <= 0]
        if not bad_qty.empty:
            result.add_error(
                entity="purchases",
                field_name="quantity",
                issue=f"Found {len(bad_qty)} purchase orders with non-positive quantity",
                row_index=bad_qty.index[0],
                invalid_value=bad_qty["quantity"].iloc[0],
            )

        # Dates check: expected_delivery_date >= order_date
        try:
            order_dt = pd.to_datetime(df["order_date"])
            exp_dt = pd.to_datetime(df["expected_delivery_date"])
            bad_dates = df[exp_dt < order_dt]
            if not bad_dates.empty:
                result.add_error(
                    entity="purchases",
                    issue=f"Found {len(bad_dates)} purchase orders where expected_delivery_date is earlier than order_date",
                    row_index=bad_dates.index[0],
                )
        except Exception as e:
            result.add_error(
                entity="purchases",
                issue=f"Date parsing failed in purchases: {e}",
            )

    def _validate_returns_domain(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Domain validations for returns."""
        bad_qty = df[df["quantity"] <= 0]
        if not bad_qty.empty:
            result.add_error(
                entity="returns",
                field_name="quantity",
                issue=f"Found {len(bad_qty)} returns with non-positive quantity",
                row_index=bad_qty.index[0],
                invalid_value=bad_qty["quantity"].iloc[0],
            )

    def _validate_suppliers_domain(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Domain validations for suppliers."""
        bad_lead = df[df["average_lead_time_days"] <= 0]
        if not bad_lead.empty:
            result.add_error(
                entity="suppliers",
                field_name="average_lead_time_days",
                issue=f"Found {len(bad_lead)} suppliers with lead time <= 0",
                row_index=bad_lead.index[0],
                invalid_value=bad_lead["average_lead_time_days"].iloc[0],
            )

        bad_moq = df[df["minimum_order_quantity"] < 1]
        if not bad_moq.empty:
            result.add_error(
                entity="suppliers",
                field_name="minimum_order_quantity",
                issue=f"Found {len(bad_moq)} suppliers with MOQ < 1",
                row_index=bad_moq.index[0],
                invalid_value=bad_moq["minimum_order_quantity"].iloc[0],
            )

    def validate_relational_integrity(self, datasets: Dict[str, pd.DataFrame]) -> ValidationResult:
        """Validate foreign-key references across related dataframes."""
        result = ValidationResult()

        products_df = datasets.get("products")
        warehouses_df = datasets.get("warehouses")
        channels_df = datasets.get("channels")
        suppliers_df = datasets.get("suppliers")
        sales_df = datasets.get("sales")
        inventory_df = datasets.get("inventory")
        purchases_df = datasets.get("purchases")
        returns_df = datasets.get("returns")

        valid_skus: Set[str] = set(products_df["sku_id"].astype(str)) if products_df is not None and "sku_id" in products_df.columns else set()
        valid_whs: Set[str] = set(warehouses_df["warehouse_id"].astype(str)) if warehouses_df is not None and "warehouse_id" in warehouses_df.columns else set()
        valid_channels: Set[str] = set(channels_df["channel_id"].astype(str)) if channels_df is not None and "channel_id" in channels_df.columns else set()
        valid_suppliers: Set[str] = set(suppliers_df["supplier_id"].astype(str)) if suppliers_df is not None and "supplier_id" in suppliers_df.columns else set()

        # Helper to check FK
        def check_fk(df: Optional[pd.DataFrame], entity: str, col: str, valid_set: Set[str], parent_entity: str) -> None:
            if df is None or col not in df.columns or not valid_set:
                return
            actual_keys = df[col].astype(str)
            orphan_mask = ~actual_keys.isin(valid_set)
            orphan_count = orphan_mask.sum()
            if orphan_count > 0:
                orphans = actual_keys[orphan_mask].unique()[:5].tolist()
                result.add_error(
                    entity=entity,
                    field_name=col,
                    issue=f"Found {orphan_count} records in '{entity}' with orphan '{col}' not in {parent_entity}. Sample: {orphans}",
                )

        # Sales FKs
        if sales_df is not None:
            check_fk(sales_df, "sales", "sku_id", valid_skus, "products")
            check_fk(sales_df, "sales", "warehouse_id", valid_whs, "warehouses")
            check_fk(sales_df, "sales", "channel_id", valid_channels, "channels")

        # Inventory FKs
        if inventory_df is not None:
            check_fk(inventory_df, "inventory", "sku_id", valid_skus, "products")
            check_fk(inventory_df, "inventory", "warehouse_id", valid_whs, "warehouses")

        # Purchases FKs
        if purchases_df is not None:
            check_fk(purchases_df, "purchases", "sku_id", valid_skus, "products")
            check_fk(purchases_df, "purchases", "warehouse_id", valid_whs, "warehouses")
            check_fk(purchases_df, "purchases", "supplier_id", valid_suppliers, "suppliers")

        # Returns FKs
        if returns_df is not None:
            check_fk(returns_df, "returns", "sku_id", valid_skus, "products")
            check_fk(returns_df, "returns", "warehouse_id", valid_whs, "warehouses")
            check_fk(returns_df, "returns", "channel_id", valid_channels, "channels")

        return result

    def validate_all(self, datasets: Dict[str, pd.DataFrame]) -> ValidationResult:
        """Validate all datasets for schema, domain rules, and relational integrity."""
        total_result = ValidationResult()

        for entity_name, df in datasets.items():
            res = self.validate_entity(entity_name, df)
            total_result.merge(res)

        rel_result = self.validate_relational_integrity(datasets)
        total_result.merge(rel_result)

        return total_result


def generate_data_quality_report(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Generate a comprehensive data-quality summary report across all datasets.

    Reports:
    - Rows & Columns per dataset
    - Missing values count & percentages
    - Duplicate records count
    - Invalid records found by validator
    - Date range across time-series datasets
    - Unique SKUs, Warehouses, Channels
    - Total sales quantity & Total revenue
    """
    validator = CommerceDataValidator()
    validation_res = validator.validate_all(datasets)

    report: Dict[str, Any] = {
        "status": "VALID" if validation_res.valid else "INVALID",
        "total_errors": len(validation_res.errors),
        "total_warnings": len(validation_res.warnings),
        "entities": {},
    }

    # Per-entity summary
    for name, df in datasets.items():
        missing_count = int(df.isna().sum().sum())
        pk = PRIMARY_KEYS.get(name, [])
        duplicates = int(df.duplicated(subset=pk).sum()) if pk and all(p in df.columns for p in pk) else 0

        report["entities"][name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "missing_values": missing_count,
            "duplicate_primary_keys": duplicates,
        }

    # Date ranges
    date_ranges: Dict[str, Dict[str, str]] = {}
    if "sales" in datasets and "date" in datasets["sales"].columns and not datasets["sales"].empty:
        s_dates = pd.to_datetime(datasets["sales"]["date"])
        date_ranges["sales"] = {
            "min_date": str(s_dates.min().date()),
            "max_date": str(s_dates.max().date()),
            "total_days": int((s_dates.max() - s_dates.min()).days) + 1,
        }

    if "inventory" in datasets and "snapshot_date" in datasets["inventory"].columns and not datasets["inventory"].empty:
        inv_dates = pd.to_datetime(datasets["inventory"]["snapshot_date"])
        date_ranges["inventory"] = {
            "min_date": str(inv_dates.min().date()),
            "max_date": str(inv_dates.max().date()),
            "total_days": int((inv_dates.max() - inv_dates.min()).days) + 1,
        }

    report["date_ranges"] = date_ranges

    # Unique Entities
    unique_skus = 0
    if "products" in datasets and "sku_id" in datasets["products"].columns:
        unique_skus = int(datasets["products"]["sku_id"].nunique())
    elif "sales" in datasets and "sku_id" in datasets["sales"].columns:
        unique_skus = int(datasets["sales"]["sku_id"].nunique())

    unique_warehouses = 0
    if "warehouses" in datasets and "warehouse_id" in datasets["warehouses"].columns:
        unique_warehouses = int(datasets["warehouses"]["warehouse_id"].nunique())

    unique_channels = 0
    if "channels" in datasets and "channel_id" in datasets["channels"].columns:
        unique_channels = int(datasets["channels"]["channel_id"].nunique())

    report["dimensions"] = {
        "unique_skus": unique_skus,
        "unique_warehouses": unique_warehouses,
        "unique_channels": unique_channels,
    }

    # Aggregate Sales KPIs
    if "sales" in datasets and not datasets["sales"].empty:
        s_df = datasets["sales"]
        tot_qty = int(s_df["quantity"].sum()) if "quantity" in s_df.columns else 0
        tot_rev = float(round(s_df["revenue"].sum(), 2)) if "revenue" in s_df.columns else 0.0
        avg_order_val = float(round(tot_rev / s_df["order_id"].nunique(), 2)) if "order_id" in s_df.columns and s_df["order_id"].nunique() > 0 else 0.0

        report["sales_summary"] = {
            "total_sales_quantity": tot_qty,
            "total_revenue": tot_rev,
            "average_order_value": avg_order_val,
            "total_transactions": len(s_df),
            "currency": s_df["currency"].iloc[0] if "currency" in s_df.columns and not s_df.empty else "USD",
        }

    # Include validation errors in report
    report["validation_errors"] = [e.to_dict() for e in validation_res.errors[:20]]
    if len(validation_res.errors) > 20:
        report["validation_errors_truncated"] = len(validation_res.errors) - 20

    return report
