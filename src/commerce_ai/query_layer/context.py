"""Query Context specification for Business Intelligence Tool & Dashboard Layer (Phase 7A).

Provides a centralized, validated QueryContext object representing parameters
passed to read-only queries by Dashboards and eRetail Copilot:
- Temporal constraints: as_of_date (point-in-time safety), start_date, end_date, time_grain
- Commercial dimensions: sku_id, warehouse_id, channel_id, category_id, brand, supplier_id
- Analytical segments: velocity_tier, abc_class, xyz_class
- Currency constraint: strict currency isolation
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _normalize_date_str(val: Optional[Union[str, date, datetime]]) -> Optional[str]:
    """Convert input date/datetime/string into standard ISO YYYY-MM-DD string."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date().isoformat()
    if isinstance(val, date):
        return val.isoformat()
    # If string, extract date portion (first 10 chars if YYYY-MM-DD...)
    val_str = str(val).strip()
    if len(val_str) >= 10:
        return val_str[:10]
    return val_str


def _normalize_str_list(val: Optional[Union[str, Sequence[str], Set[str]]]) -> Optional[List[str]]:
    """Normalize a string or collection of strings into a sorted list of unique stripped strings."""
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip()
        return [v] if v else None
    result = {str(item).strip() for item in val if str(item).strip()}
    return sorted(list(result)) if result else None


class QueryContext(BaseModel):
    """Execution context and filter specification for Business Intelligence queries."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    as_of_date: Optional[Union[str, date, datetime]] = Field(
        default=None,
        description="Point-in-time reference date. All queries strictly enforce records <= as_of_date.",
    )
    start_date: Optional[Union[str, date, datetime]] = Field(
        default=None,
        description="Start of analysis window (inclusive).",
    )
    end_date: Optional[Union[str, date, datetime]] = Field(
        default=None,
        description="End of analysis window (inclusive). Cannot exceed as_of_date.",
    )
    currency: Optional[str] = Field(
        default=None,
        description="Mandatory or explicit currency code (e.g. 'USD') for currency isolation.",
    )

    # Dimensional Filters (Single string or list of strings)
    sku_id: Optional[Union[str, List[str]]] = Field(default=None, description="SKU identifier(s)")
    warehouse_id: Optional[Union[str, List[str]]] = Field(default=None, description="Warehouse identifier(s)")
    channel_id: Optional[Union[str, List[str]]] = Field(default=None, description="Sales channel identifier(s)")
    category_id: Optional[Union[str, List[str]]] = Field(default=None, description="Product category identifier(s)")
    brand: Optional[Union[str, List[str]]] = Field(default=None, description="Brand name(s)")
    supplier_id: Optional[Union[str, List[str]]] = Field(default=None, description="Supplier identifier(s)")

    # Analytical Segment Filters
    velocity_tier: Optional[Union[str, List[str]]] = Field(default=None, description="Velocity tier(s) (FAST, SLOW, etc.)")
    abc_class: Optional[Union[str, List[str]]] = Field(default=None, description="ABC revenue classification(s)")
    xyz_class: Optional[Union[str, List[str]]] = Field(default=None, description="XYZ variability classification(s)")

    # Query Configuration
    time_grain: str = Field(default="daily", description="Time granularity: daily, weekly, monthly")
    limit: Optional[int] = Field(default=None, description="Pagination / row limit")
    offset: Optional[int] = Field(default=None, description="Pagination offset")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary caller metadata")

    @model_validator(mode="after")
    def validate_dates_and_currencies(self) -> QueryContext:
        """Validate date chronology, point-in-time boundaries, and currency casing."""
        iso_as_of = _normalize_date_str(self.as_of_date)
        iso_start = _normalize_date_str(self.start_date)
        iso_end = _normalize_date_str(self.end_date)

        if iso_start and iso_end and iso_start > iso_end:
            raise ValueError(f"start_date ({iso_start}) cannot be after end_date ({iso_end})")

        if iso_as_of and iso_start and iso_start > iso_as_of:
            raise ValueError(f"start_date ({iso_start}) cannot be after point-in-time as_of_date ({iso_as_of})")

        return self

    # Normalized Accessors
    @property
    def iso_as_of_date(self) -> Optional[str]:
        return _normalize_date_str(self.as_of_date)

    @property
    def iso_start_date(self) -> Optional[str]:
        return _normalize_date_str(self.start_date)

    @property
    def iso_end_date(self) -> Optional[str]:
        return _normalize_date_str(self.end_date)

    @property
    def effective_end_date(self) -> Optional[str]:
        """Calculates effective upper date bound respecting as_of_date anti-leakage boundary."""
        as_of = self.iso_as_of_date
        end = self.iso_end_date
        if as_of and end:
            return min(as_of, end)
        return as_of or end

    @property
    def effective_currency(self) -> Optional[str]:
        return self.currency.strip().upper() if self.currency else None

    @property
    def sku_ids(self) -> Optional[List[str]]:
        return _normalize_str_list(self.sku_id)

    @property
    def warehouse_ids(self) -> Optional[List[str]]:
        return _normalize_str_list(self.warehouse_id)

    @property
    def channel_ids(self) -> Optional[List[str]]:
        return _normalize_str_list(self.channel_id)

    @property
    def category_ids(self) -> Optional[List[str]]:
        return _normalize_str_list(self.category_id)

    @property
    def brands(self) -> Optional[List[str]]:
        return _normalize_str_list(self.brand)

    @property
    def supplier_ids(self) -> Optional[List[str]]:
        return _normalize_str_list(self.supplier_id)

    @property
    def velocity_tiers(self) -> Optional[List[str]]:
        return _normalize_str_list(self.velocity_tier)

    @property
    def abc_classes(self) -> Optional[List[str]]:
        return _normalize_str_list(self.abc_class)

    @property
    def xyz_classes(self) -> Optional[List[str]]:
        return _normalize_str_list(self.xyz_class)

    def to_filter_dict(self) -> Dict[str, Any]:
        """Produce clean dictionary of active filters applied in this context."""
        active: Dict[str, Any] = {}
        if self.iso_as_of_date:
            active["as_of_date"] = self.iso_as_of_date
        if self.iso_start_date:
            active["start_date"] = self.iso_start_date
        if self.effective_end_date:
            active["end_date"] = self.effective_end_date
        if self.effective_currency:
            active["currency"] = self.effective_currency
        if self.sku_ids:
            active["sku_ids"] = self.sku_ids
        if self.warehouse_ids:
            active["warehouse_ids"] = self.warehouse_ids
        if self.channel_ids:
            active["channel_ids"] = self.channel_ids
        if self.category_ids:
            active["category_ids"] = self.category_ids
        if self.brands:
            active["brands"] = self.brands
        if self.supplier_ids:
            active["supplier_ids"] = self.supplier_ids
        if self.velocity_tiers:
            active["velocity_tiers"] = self.velocity_tiers
        if self.abc_classes:
            active["abc_classes"] = self.abc_classes
        if self.xyz_classes:
            active["xyz_classes"] = self.xyz_classes
        if self.time_grain != "daily":
            active["time_grain"] = self.time_grain
        if self.limit is not None:
            active["limit"] = self.limit
        if self.offset is not None:
            active["offset"] = self.offset
        return active
