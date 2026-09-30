"""Supplier and SKU Lead Time Profiling Engine.

Extracts empirical delivery durations, variance, and on-time performance metrics
from historical purchase orders, with configurable fallback to supplier master contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.inventory.schemas import LeadTimeMetrics


@dataclass
class LeadTimeConfig:
    """Configuration for lead-time calculation and fallback thresholds."""

    min_history: int = 3  # Minimum delivered orders needed for empirical estimation
    default_lead_time_days: float = 14.0  # Fallback if neither PO history nor supplier master exists
    default_lead_time_std_days: float = 0.0  # Fallback std dev when history is insufficient


def calculate_po_lead_times(
    purchases: pd.DataFrame,
    as_of_date: Optional[Union[str, date, datetime]] = None,
) -> pd.DataFrame:
    """Compute empirical delivery duration and on-time status for delivered purchase orders.

    Point-in-time rules when as_of_date is provided:
        1. Purchase orders with order_date > as_of_date are excluded.
        2. Deliveries with actual_delivery_date > as_of_date are excluded.
        3. Only orders completed on or before as_of_date contribute to empirical lead time.

    Args:
        purchases: Purchase orders DataFrame containing:
            ['purchase_order_id', 'order_date', 'actual_delivery_date', 'supplier_id']
            and optionally ['sku_id', 'expected_delivery_date', 'status'].
        as_of_date: Optional cutoff date for point-in-time calculation.

    Returns:
        pd.DataFrame: Cleaned DataFrame of delivered POs with:
            ['purchase_order_id', 'supplier_id', 'sku_id', 'order_date',
             'actual_delivery_date', 'expected_delivery_date',
             'actual_lead_time_days', 'is_on_time']
    """
    if purchases is None or purchases.empty:
        return pd.DataFrame(
            columns=[
                "purchase_order_id",
                "supplier_id",
                "sku_id",
                "order_date",
                "actual_delivery_date",
                "expected_delivery_date",
                "actual_lead_time_days",
                "is_on_time",
            ]
        )

    df = purchases.copy()

    # Required columns check
    req = {"order_date", "actual_delivery_date", "supplier_id"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"Purchases DataFrame missing required columns: {missing}")

    # Standardize dates
    df["order_date_dt"] = pd.to_datetime(df["order_date"], errors="coerce").dt.normalize()
    df["actual_delivery_dt"] = pd.to_datetime(df["actual_delivery_date"], errors="coerce").dt.normalize()

    # Filter to orders that have been delivered
    delivered = df[df["actual_delivery_dt"].notna() & df["order_date_dt"].notna()].copy()
    if delivered.empty:
        return pd.DataFrame()

    # Point-in-time cutoff filtering to prevent future leakage
    if as_of_date is not None:
        cutoff = pd.to_datetime(as_of_date).normalize()
        # Exclude POs ordered after cutoff
        delivered = delivered[delivered["order_date_dt"] <= cutoff]
        # Exclude POs delivered after cutoff (in-transit as of cutoff)
        delivered = delivered[delivered["actual_delivery_dt"] <= cutoff]
        if delivered.empty:
            return pd.DataFrame()

    # Calculate actual duration in days
    lead_days = (delivered["actual_delivery_dt"] - delivered["order_date_dt"]).dt.days
    # Discard non-causal records where delivery occurred before order date
    valid_mask = lead_days >= 0
    delivered = delivered[valid_mask].copy()
    delivered["actual_lead_time_days"] = (
        delivered["actual_delivery_dt"] - delivered["order_date_dt"]
    ).dt.days.astype(float)

    # Calculate on-time status if expected_delivery_date is available
    if "expected_delivery_date" in delivered.columns:
        delivered["expected_delivery_dt"] = pd.to_datetime(
            delivered["expected_delivery_date"], errors="coerce"
        ).dt.normalize()
        valid_exp = delivered["expected_delivery_dt"].notna()
        is_ot_series = delivered["actual_delivery_dt"] <= delivered["expected_delivery_dt"]
        delivered["is_on_time"] = [
            bool(is_ot_series.iloc[i]) if valid_exp.iloc[i] else None
            for i in range(len(delivered))
        ]
    else:
        delivered["expected_delivery_dt"] = pd.NaT
        delivered["is_on_time"] = None

    if "sku_id" not in delivered.columns:
        delivered["sku_id"] = None

    if "purchase_order_id" not in delivered.columns:
        delivered["purchase_order_id"] = [f"PO_{i}" for i in range(len(delivered))]

    res_cols = [
        "purchase_order_id",
        "supplier_id",
        "sku_id",
        "order_date",
        "actual_delivery_date",
        "expected_delivery_date",
        "actual_lead_time_days",
        "is_on_time",
    ]
    return delivered[res_cols].reset_index(drop=True)


def profile_supplier_lead_times(
    purchases: Optional[pd.DataFrame] = None,
    suppliers: Optional[pd.DataFrame] = None,
    config: Optional[LeadTimeConfig] = None,
    as_of_date: Optional[Union[str, date, datetime]] = None,
) -> Dict[str, LeadTimeMetrics]:
    """Profile empirical lead-time parameters for every supplier.

    Args:
        purchases: Optional historical purchase orders.
        suppliers: Optional master suppliers catalog (with average_lead_time_days).
        config: LeadTimeConfig specifying min_history threshold.
        as_of_date: Optional point-in-time reference date.

    Returns:
        Dict[str, LeadTimeMetrics]: Keyed by supplier_id.
    """
    cfg = config or LeadTimeConfig()
    profiles: Dict[str, LeadTimeMetrics] = {}

    # Build catalog map if suppliers DataFrame is provided
    catalog_map: Dict[str, float] = {}
    if suppliers is not None and not suppliers.empty:
        if "supplier_id" in suppliers.columns and "average_lead_time_days" in suppliers.columns:
            catalog_map = suppliers.set_index("supplier_id")["average_lead_time_days"].to_dict()

    # Process delivered historical orders
    po_df = calculate_po_lead_times(purchases, as_of_date=as_of_date) if purchases is not None else pd.DataFrame()

    if not po_df.empty:
        for supp_id, grp in po_df.groupby("supplier_id"):
            supp_id_str = str(supp_id)
            n_obs = len(grp)
            lead_times = grp["actual_lead_time_days"].values

            on_time_col = grp["is_on_time"].dropna()
            otd_rate = float(on_time_col.mean()) if not on_time_col.empty else None

            if n_obs >= cfg.min_history:
                mean_lt = float(np.mean(lead_times))
                std_lt = float(np.std(lead_times, ddof=1)) if n_obs > 1 else 0.0
                profiles[supp_id_str] = LeadTimeMetrics(
                    supplier_id=supp_id_str,
                    sample_size=n_obs,
                    mean_lead_time_days=max(0.0, mean_lt),
                    std_lead_time_days=max(0.0, std_lt),
                    min_lead_time_days=float(np.min(lead_times)),
                    max_lead_time_days=float(np.max(lead_times)),
                    on_time_delivery_rate=otd_rate,
                    is_fallback=False,
                )
            else:
                # History is insufficient: use supplier master average or fallback
                fallback_mean = float(catalog_map.get(supp_id_str, cfg.default_lead_time_days))
                profiles[supp_id_str] = LeadTimeMetrics(
                    supplier_id=supp_id_str,
                    sample_size=n_obs,
                    mean_lead_time_days=fallback_mean,
                    std_lead_time_days=cfg.default_lead_time_std_days,
                    min_lead_time_days=float(np.min(lead_times)) if n_obs > 0 else None,
                    max_lead_time_days=float(np.max(lead_times)) if n_obs > 0 else None,
                    on_time_delivery_rate=otd_rate,
                    is_fallback=True,
                    fallback_reason=f"Insufficient history ({n_obs} < {cfg.min_history})",
                )

    # For suppliers in master catalog with 0 PO history
    for supp_id, avg_days in catalog_map.items():
        supp_id_str = str(supp_id)
        if supp_id_str not in profiles:
            profiles[supp_id_str] = LeadTimeMetrics(
                supplier_id=supp_id_str,
                sample_size=0,
                mean_lead_time_days=float(avg_days),
                std_lead_time_days=cfg.default_lead_time_std_days,
                is_fallback=True,
                fallback_reason="No PO delivery history",
            )

    return profiles


def profile_sku_lead_times(
    purchases: Optional[pd.DataFrame] = None,
    products: Optional[pd.DataFrame] = None,
    suppliers: Optional[pd.DataFrame] = None,
    config: Optional[LeadTimeConfig] = None,
    as_of_date: Optional[Union[str, date, datetime]] = None,
) -> Dict[str, LeadTimeMetrics]:
    """Profile empirical lead times at the SKU level with hierarchical fallback.

    Fallback hierarchy:
        1. SKU-specific empirical PO delivery history (if sample size >= min_history)
        2. Supplier empirical PO delivery history
        3. Supplier master catalog average_lead_time_days
        4. Global default lead time (default_lead_time_days)

    Args:
        purchases: Historical purchase orders.
        products: Product catalog containing 'sku_id' and 'preferred_supplier_id'.
        suppliers: Supplier catalog containing 'supplier_id' and 'average_lead_time_days'.
        config: LeadTimeConfig specifying thresholds.
        as_of_date: Optional point-in-time reference date.

    Returns:
        Dict[str, LeadTimeMetrics]: Keyed by sku_id.
    """
    cfg = config or LeadTimeConfig()
    supplier_profiles = profile_supplier_lead_times(purchases, suppliers, cfg, as_of_date=as_of_date)

    # Map SKU to preferred supplier
    sku_supplier_map: Dict[str, str] = {}
    if products is not None and not products.empty:
        if "sku_id" in products.columns and "preferred_supplier_id" in products.columns:
            sku_supplier_map = products.dropna(subset=["preferred_supplier_id"]).set_index("sku_id")["preferred_supplier_id"].astype(str).to_dict()

    po_df = calculate_po_lead_times(purchases, as_of_date=as_of_date) if purchases is not None else pd.DataFrame()
    sku_profiles: Dict[str, LeadTimeMetrics] = {}

    # 1. Evaluate SKUs with delivered PO records
    if not po_df.empty and "sku_id" in po_df.columns:
        sku_grouped = po_df.dropna(subset=["sku_id"]).groupby("sku_id")
        for sku_id, grp in sku_grouped:
            sku_id_str = str(sku_id)
            n_obs = len(grp)
            supp_id = str(grp["supplier_id"].iloc[0]) if "supplier_id" in grp.columns else sku_supplier_map.get(sku_id_str, "UNKNOWN")
            lead_times = grp["actual_lead_time_days"].values

            on_time_col = grp["is_on_time"].dropna()
            otd_rate = float(on_time_col.mean()) if not on_time_col.empty else None

            if n_obs >= cfg.min_history:
                mean_lt = float(np.mean(lead_times))
                std_lt = float(np.std(lead_times, ddof=1)) if n_obs > 1 else 0.0
                sku_profiles[sku_id_str] = LeadTimeMetrics(
                    supplier_id=supp_id,
                    sku_id=sku_id_str,
                    sample_size=n_obs,
                    mean_lead_time_days=max(0.0, mean_lt),
                    std_lead_time_days=max(0.0, std_lt),
                    min_lead_time_days=float(np.min(lead_times)),
                    max_lead_time_days=float(np.max(lead_times)),
                    on_time_delivery_rate=otd_rate,
                    is_fallback=False,
                )
            else:
                # Hierarchical fallback to supplier profile
                supp_metric = supplier_profiles.get(supp_id)
                if supp_metric is not None:
                    sku_profiles[sku_id_str] = LeadTimeMetrics(
                        supplier_id=supp_id,
                        sku_id=sku_id_str,
                        sample_size=n_obs,
                        mean_lead_time_days=supp_metric.mean_lead_time_days,
                        std_lead_time_days=supp_metric.std_lead_time_days,
                        min_lead_time_days=float(np.min(lead_times)) if n_obs > 0 else None,
                        max_lead_time_days=float(np.max(lead_times)) if n_obs > 0 else None,
                        on_time_delivery_rate=otd_rate if otd_rate is not None else supp_metric.on_time_delivery_rate,
                        is_fallback=True,
                        fallback_reason=f"Insufficient SKU history ({n_obs} < {cfg.min_history}); inherited from supplier {supp_id}",
                    )
                else:
                    sku_profiles[sku_id_str] = LeadTimeMetrics(
                        supplier_id=supp_id,
                        sku_id=sku_id_str,
                        sample_size=n_obs,
                        mean_lead_time_days=cfg.default_lead_time_days,
                        std_lead_time_days=cfg.default_lead_time_std_days,
                        is_fallback=True,
                        fallback_reason=f"Default fallback ({n_obs} < {cfg.min_history})",
                    )

    # 2. Add SKUs from product catalog with no PO records
    if products is not None and not products.empty and "sku_id" in products.columns:
        for sku_id in products["sku_id"].unique():
            sku_id_str = str(sku_id)
            if sku_id_str not in sku_profiles:
                supp_id = sku_supplier_map.get(sku_id_str, "UNKNOWN")
                supp_metric = supplier_profiles.get(supp_id)
                if supp_metric is not None:
                    sku_profiles[sku_id_str] = LeadTimeMetrics(
                        supplier_id=supp_id,
                        sku_id=sku_id_str,
                        sample_size=0,
                        mean_lead_time_days=supp_metric.mean_lead_time_days,
                        std_lead_time_days=supp_metric.std_lead_time_days,
                        on_time_delivery_rate=supp_metric.on_time_delivery_rate,
                        is_fallback=True,
                        fallback_reason=f"No SKU PO history; inherited from supplier {supp_id}",
                    )
                else:
                    sku_profiles[sku_id_str] = LeadTimeMetrics(
                        supplier_id=supp_id,
                        sku_id=sku_id_str,
                        sample_size=0,
                        mean_lead_time_days=cfg.default_lead_time_days,
                        std_lead_time_days=cfg.default_lead_time_std_days,
                        is_fallback=True,
                        fallback_reason="No PO delivery history and supplier unknown",
                    )

    return sku_profiles
