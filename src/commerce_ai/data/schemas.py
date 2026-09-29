"""Standard Commerce Data Contract - Pydantic Schemas.

Defines canonical Pydantic models for the 8 core ecommerce entities:
1. Sales (SaleRecord)
2. Inventory (InventorySnapshot)
3. Products (Product)
4. Warehouses (Warehouse)
5. Purchases (PurchaseOrder)
6. Returns (ReturnRecord)
7. Channels (Channel)
8. Suppliers (Supplier)
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Union, Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _validate_identifier(v: str, field_name: str) -> str:
    """Validate that an ID string is non-empty and stripped."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return v.strip()


class BaseCommerceModel(BaseModel):
    """Base model allowing extra fields for extensibility."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class SaleRecord(BaseCommerceModel):
    """Sales transaction record representing a line item in an order."""

    sale_id: str = Field(..., description="Unique transaction line ID")
    order_id: str = Field(..., description="Order ID linking transaction lines")
    date: Union[dt.date, dt.datetime] = Field(..., description="Transaction date or timestamp")
    sku_id: str = Field(..., description="Stock Keeping Unit ID")
    warehouse_id: str = Field(..., description="Fulfillment warehouse ID")
    channel_id: str = Field(..., description="Sales channel ID")
    quantity: int = Field(..., gt=0, description="Quantity sold, must be positive")
    unit_price: float = Field(..., ge=0.0, description="Unit sales price, non-negative")
    discount: float = Field(default=0.0, ge=0.0, description="Discount amount, non-negative")
    revenue: float = Field(..., ge=0.0, description="Net realized revenue")
    currency: str = Field(default="USD", min_length=3, max_length=3, description="ISO 3-letter currency")

    @field_validator("sale_id", "order_id", "sku_id", "warehouse_id", "channel_id")
    @classmethod
    def check_non_empty_ids(cls, v: str, info) -> str:
        return _validate_identifier(v, info.field_name)

    @field_validator("currency")
    @classmethod
    def uppercase_currency(cls, v: str) -> str:
        return v.strip().upper()


class InventorySnapshot(BaseCommerceModel):
    """Daily inventory snapshot for a specific SKU and warehouse."""

    snapshot_date: dt.date = Field(..., description="Date of inventory snapshot")
    sku_id: str = Field(..., description="Stock Keeping Unit ID")
    warehouse_id: str = Field(..., description="Warehouse ID")
    available_qty: int = Field(..., ge=0, description="Available on-hand pickable quantity")
    reserved_qty: int = Field(default=0, ge=0, description="Quantity reserved for orders")
    in_transit_qty: int = Field(default=0, ge=0, description="Quantity inbound in transit")
    damaged_qty: int = Field(default=0, ge=0, description="Damaged or quarantined quantity")

    @field_validator("sku_id", "warehouse_id")
    @classmethod
    def check_non_empty_ids(cls, v: str, info) -> str:
        return _validate_identifier(v, info.field_name)

    @property
    def total_physical_qty(self) -> int:
        """Physical inventory present in warehouse (available + reserved + damaged)."""
        return self.available_qty + self.reserved_qty + self.damaged_qty


class Product(BaseCommerceModel):
    """Product catalog master record."""

    sku_id: str = Field(..., description="Unique product SKU")
    product_name: str = Field(..., min_length=1, description="Commercial product name")
    category_id: str = Field(..., min_length=1, description="Category name or ID")
    brand: Optional[str] = Field(default=None, description="Brand name")
    unit_cost: float = Field(..., ge=0.0, description="Landed cost of goods sold")
    selling_price: float = Field(..., ge=0.0, description="Standard retail selling price")
    currency: str = Field(default="USD", min_length=3, max_length=3, description="ISO currency code")
    description: Optional[str] = Field(default=None, description="Optional product description")
    status: Optional[str] = Field(default="ACTIVE", description="Product status (ACTIVE, DISCONTINUED)")

    @field_validator("sku_id", "product_name", "category_id")
    @classmethod
    def check_non_empty_strings(cls, v: str, info) -> str:
        return _validate_identifier(v, info.field_name)

    @field_validator("currency")
    @classmethod
    def uppercase_currency(cls, v: str) -> str:
        return v.strip().upper()


class Warehouse(BaseCommerceModel):
    """Fulfillment warehouse facility master record."""

    warehouse_id: str = Field(..., description="Unique warehouse identifier")
    warehouse_name: str = Field(..., min_length=1, description="Warehouse facility name")
    city: str = Field(..., min_length=1, description="Facility city")
    state: str = Field(..., min_length=1, description="Facility state or province")
    country: str = Field(..., min_length=2, description="Facility country")
    capacity_units: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional maximum storage capacity in units",
    )

    @field_validator("warehouse_id", "warehouse_name", "city", "state", "country")
    @classmethod
    def check_non_empty_strings(cls, v: str, info) -> str:
        return _validate_identifier(v, info.field_name)


class PurchaseOrder(BaseCommerceModel):
    """Inbound purchase order line placed with a supplier."""

    purchase_order_id: str = Field(..., description="Purchase order identifier")
    order_date: dt.date = Field(..., description="Date purchase order was issued")
    sku_id: str = Field(..., description="Ordered SKU ID")
    supplier_id: str = Field(..., description="Vendor supplier ID")
    warehouse_id: str = Field(..., description="Receiving warehouse ID")
    quantity: int = Field(..., gt=0, description="Ordered quantity, must be positive")
    unit_cost: float = Field(..., ge=0.0, description="Purchased unit cost")
    expected_delivery_date: dt.date = Field(..., description="Estimated delivery date")
    actual_delivery_date: Optional[dt.date] = Field(default=None, description="Actual dock receipt date")
    status: str = Field(
        default="PENDING",
        description="PO state: PENDING, IN_TRANSIT, DELIVERED, CANCELLED",
    )

    @field_validator("purchase_order_id", "sku_id", "supplier_id", "warehouse_id", "status")
    @classmethod
    def check_non_empty_fields(cls, v: str, info) -> str:
        return _validate_identifier(v, info.field_name)

    @model_validator(mode="after")
    def validate_dates(self) -> PurchaseOrder:
        if self.expected_delivery_date < self.order_date:
            raise ValueError(
                f"expected_delivery_date ({self.expected_delivery_date}) cannot be earlier than order_date ({self.order_date})"
            )
        if self.actual_delivery_date is not None and self.actual_delivery_date < self.order_date:
            raise ValueError(
                f"actual_delivery_date ({self.actual_delivery_date}) cannot be earlier than order_date ({self.order_date})"
            )
        return self


class ReturnRecord(BaseCommerceModel):
    """Customer return record."""

    return_id: str = Field(..., description="Unique return event identifier")
    order_id: str = Field(..., description="Associated sales order ID")
    return_date: dt.date = Field(..., description="Date return was processed")
    sku_id: str = Field(..., description="Returned SKU ID")
    warehouse_id: str = Field(..., description="Receiving warehouse ID")
    quantity: int = Field(..., gt=0, description="Returned quantity, must be positive")
    reason: str = Field(..., min_length=1, description="Categorized return reason")
    channel_id: str = Field(..., description="Originating sales channel ID")

    @field_validator("return_id", "order_id", "sku_id", "warehouse_id", "reason", "channel_id")
    @classmethod
    def check_non_empty_strings(cls, v: str, info) -> str:
        return _validate_identifier(v, info.field_name)


class Channel(BaseCommerceModel):
    """Sales channel master record."""

    channel_id: str = Field(..., description="Unique channel identifier")
    channel_name: str = Field(..., min_length=1, description="Display name of the channel")
    channel_type: str = Field(..., min_length=1, description="Classification (Marketplace, Direct, B2B, Retail)")

    @field_validator("channel_id", "channel_name", "channel_type")
    @classmethod
    def check_non_empty_strings(cls, v: str, info) -> str:
        return _validate_identifier(v, info.field_name)


class Supplier(BaseCommerceModel):
    """Supplier / vendor master record."""

    supplier_id: str = Field(..., description="Unique supplier identifier")
    supplier_name: str = Field(..., min_length=1, description="Supplier corporate name")
    average_lead_time_days: int = Field(..., gt=0, description="Average lead time in calendar days")
    minimum_order_quantity: int = Field(default=1, ge=1, description="Minimum order quantity (MOQ)")

    @field_validator("supplier_id", "supplier_name")
    @classmethod
    def check_non_empty_strings(cls, v: str, info) -> str:
        return _validate_identifier(v, info.field_name)
