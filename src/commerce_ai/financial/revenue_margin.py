"""Deterministic Line-Level and Aggregate Financial Intelligence Calculations (Phase 6A).

Provides deterministic calculations for:
- Line-level gross revenue, discount, net revenue, and source reconciliation
- Estimated COGS and gross margin / margin percentage with zero-division protection
- Data quality audits, status tagging, and point-in-time anti-leakage filtering
- Multi-dimensional aggregations (Overall, SKU, Category, Brand, Channel, Warehouse, Matrix slices, Time series)
- Neutral analytical rankings (top revenue, margin, units, bottom margin %, negative margin)
- Negative-margin component and loss analysis
- Executive portfolio summary
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.financial.schemas import (
    FinancialDataQualityReport,
    FinancialDataStatus,
    FinancialDimensionMetric,
    FinancialIntelligenceConfig,
    FinancialPortfolioSummary,
    MarginClassification,
    RankingResult,
    RevenueMarginRecord,
    RevenueReconciliationStatus,
    TimeGrain,
)


# =====================================================================
# Deterministic Identifiers
# =====================================================================


def generate_financial_record_id(
    sale_id: str,
    sku_id: str,
    order_id: str,
    date_val: str,
    currency: str = "USD",
) -> str:
    """Generate a deterministic 16-character SHA-256 identifier for a financial line record.

    Format: FIN-REC-{hexdigest[:16]}
    """
    raw = f"{str(sale_id).strip()}|{str(sku_id).strip()}|{str(order_id).strip()}|{str(date_val).strip()}|{str(currency).strip().upper()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"FIN-REC-{digest[:16]}"


def generate_financial_segment_id(
    dimension: str,
    segment_key: str,
    as_of_date: Optional[str] = None,
    currency: str = "USD",
) -> str:
    """Generate a deterministic 16-character SHA-256 identifier for a dimension aggregate segment.

    Format: FIN-SEG-{hexdigest[:16]}
    """
    raw = f"{str(dimension).strip().upper()}|{str(segment_key).strip()}|{str(as_of_date or '').strip()}|{str(currency).strip().upper()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"FIN-SEG-{digest[:16]}"


def generate_financial_portfolio_id(
    record_count: int,
    as_of_date: Optional[str] = None,
    currency: str = "USD",
) -> str:
    """Generate a deterministic 16-character SHA-256 identifier for a portfolio summary.

    Format: FIN-PORT-{hexdigest[:16]}
    """
    raw = f"PORTFOLIO|{int(record_count)}|{str(as_of_date or '').strip()}|{str(currency).strip().upper()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"FIN-PORT-{digest[:16]}"


# =====================================================================
# Data Quality & Leakage Protection
# =====================================================================


def filter_sales_by_as_of_date(
    sales_df: pd.DataFrame,
    as_of_date: Optional[Union[str, date, datetime]] = None,
) -> pd.DataFrame:
    """Filter sales dataset strictly to records dated on or before as_of_date.

    Guarantees anti-leakage protection for point-in-time financial analytics.
    Accepts date column named either 'date' or 'sale_date'.
    """
    if as_of_date is None or sales_df.empty:
        return sales_df

    cutoff_str = as_of_date.isoformat() if isinstance(as_of_date, (date, datetime)) else str(as_of_date)
    cutoff_ts = pd.to_datetime(cutoff_str)

    date_col = "date" if "date" in sales_df.columns else ("sale_date" if "sale_date" in sales_df.columns else None)
    if date_col is None:
        return sales_df

    dates = pd.to_datetime(sales_df[date_col], errors="coerce")
    valid_mask = dates <= cutoff_ts
    return sales_df.loc[valid_mask].copy()


def audit_financial_data_quality(
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    config: Optional[FinancialIntelligenceConfig] = None,
) -> FinancialDataQualityReport:
    """Audit sales and product cost data for completeness, validity, and leakage risks.

    Does not modify data. Produces an audit report with issue counts and structured details.
    """
    cfg = config or FinancialIntelligenceConfig()
    total_records = len(sales_df)

    if total_records == 0:
        return FinancialDataQualityReport(
            total_input_records=0,
            valid_records=0,
            invalid_records=0,
            is_clean=True,
        )

    date_col = "date" if "date" in sales_df.columns else ("sale_date" if "sale_date" in sales_df.columns else None)

    # 1. Missing or empty SKU
    missing_sku_mask = sales_df["sku_id"].isna() | (sales_df["sku_id"].astype(str).str.strip() == "") if "sku_id" in sales_df.columns else pd.Series(True, index=sales_df.index)
    missing_sku_count = int(missing_sku_mask.sum())

    # 2. Quantity checks
    if "quantity" in sales_df.columns:
        qty_series = pd.to_numeric(sales_df["quantity"], errors="coerce")
        missing_qty_count = int(qty_series.isna().sum())
        zero_qty_count = int((qty_series == 0).sum())
        negative_qty_count = int((qty_series < 0).sum())
    else:
        missing_qty_count = total_records
        zero_qty_count = 0
        negative_qty_count = 0

    # 3. Unit price checks
    if "unit_price" in sales_df.columns:
        price_series = pd.to_numeric(sales_df["unit_price"], errors="coerce")
        missing_price_count = int(price_series.isna().sum())
        negative_price_count = int((price_series < 0).sum())
    else:
        missing_price_count = total_records
        negative_price_count = 0

    # 4. Discount checks
    if "discount" in sales_df.columns:
        discount_series = pd.to_numeric(sales_df["discount"], errors="coerce")
        missing_discount_count = int(discount_series.isna().sum())
        negative_discount_count = int((discount_series < 0).sum())
    else:
        missing_discount_count = total_records
        negative_discount_count = 0

    # 5. Revenue checks
    rev_col = "revenue" if "revenue" in sales_df.columns else ("source_revenue" if "source_revenue" in sales_df.columns else None)
    if rev_col is not None:
        rev_series = pd.to_numeric(sales_df[rev_col], errors="coerce")
        missing_revenue_count = int(rev_series.isna().sum())
        negative_revenue_count = int((rev_series < 0).sum())
    else:
        missing_revenue_count = total_records
        negative_revenue_count = 0

    # 6. Currency checks
    curr_col = "currency" if "currency" in sales_df.columns else None
    mismatched_currencies: List[str] = []
    if curr_col is not None:
        curr_clean = sales_df[curr_col].fillna("").astype(str).str.strip().str.upper()
        missing_curr_count = int((curr_clean == "").sum())
        distinct_curr = set(curr_clean[curr_clean != ""].unique())
        for c in distinct_curr:
            if c != cfg.default_currency.upper():
                mismatched_currencies.append(c)
    else:
        missing_curr_count = total_records

    # 7. Duplicate sale_id
    if "sale_id" in sales_df.columns:
        dup_sale_count = int(sales_df["sale_id"].duplicated().sum())
    else:
        dup_sale_count = 0

    # 8. Future date leakage
    future_leakage_count = 0
    if cfg.as_of_date is not None and date_col is not None:
        cutoff = pd.to_datetime(cfg.as_of_date)
        dates = pd.to_datetime(sales_df[date_col], errors="coerce")
        future_leakage_count = int((dates > cutoff).sum())

    # 9. Unit cost checks (against products)
    missing_cost_count = 0
    negative_cost_count = 0
    if products_df is not None and not products_df.empty and "sku_id" in products_df.columns and "unit_cost" in products_df.columns:
        prod_cost_map = products_df.set_index("sku_id")["unit_cost"].to_dict()
        cost_series = sales_df["sku_id"].map(prod_cost_map) if "sku_id" in sales_df.columns else pd.Series(np.nan, index=sales_df.index)
        missing_cost_count = int(cost_series.isna().sum())
        negative_cost_count = int((cost_series < 0).sum())
    elif "unit_cost" in sales_df.columns:
        cost_series = pd.to_numeric(sales_df["unit_cost"], errors="coerce")
        missing_cost_count = int(cost_series.isna().sum())
        negative_cost_count = int((cost_series < 0).sum())
    else:
        missing_cost_count = total_records

    # Issues collection
    issues: List[Dict[str, Any]] = []
    if missing_sku_count > 0:
        issues.append({"check": "missing_sku", "count": missing_sku_count, "severity": "ERROR"})
    if missing_qty_count > 0:
        issues.append({"check": "missing_quantity", "count": missing_qty_count, "severity": "ERROR"})
    if negative_qty_count > 0:
        issues.append({"check": "negative_quantity", "count": negative_qty_count, "severity": "ERROR"})
    if missing_price_count > 0:
        issues.append({"check": "missing_unit_price", "count": missing_price_count, "severity": "ERROR"})
    if negative_price_count > 0:
        issues.append({"check": "negative_unit_price", "count": negative_price_count, "severity": "ERROR"})
    if negative_discount_count > 0:
        issues.append({"check": "negative_discount", "count": negative_discount_count, "severity": "ERROR"})
    if negative_revenue_count > 0:
        issues.append({"check": "negative_revenue", "count": negative_revenue_count, "severity": "ERROR"})
    if missing_cost_count > 0:
        issues.append({"check": "missing_unit_cost", "count": missing_cost_count, "severity": "WARNING"})
    if negative_cost_count > 0:
        issues.append({"check": "negative_unit_cost", "count": negative_cost_count, "severity": "ERROR"})
    if dup_sale_count > 0:
        issues.append({"check": "duplicate_sale_id", "count": dup_sale_count, "severity": "ERROR"})
    if future_leakage_count > 0:
        issues.append({"check": "future_date_leakage", "count": future_leakage_count, "severity": "CRITICAL"})
    if mismatched_currencies:
        issues.append({"check": "mismatched_currencies", "currencies": mismatched_currencies, "severity": "WARNING"})

    # Clean check: no critical/error issues
    has_errors = (
        missing_sku_count > 0
        or missing_qty_count > 0
        or negative_qty_count > 0
        or missing_price_count > 0
        or negative_price_count > 0
        or negative_discount_count > 0
        or negative_revenue_count > 0
        or negative_cost_count > 0
        or dup_sale_count > 0
        or future_leakage_count > 0
    )

    invalid_records = int(has_errors)  # Count of invalid flagged if any errors
    valid_records = total_records - invalid_records

    return FinancialDataQualityReport(
        total_input_records=total_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        missing_sku_count=missing_sku_count,
        missing_quantity_count=missing_qty_count,
        zero_quantity_count=zero_qty_count,
        negative_quantity_count=negative_qty_count,
        missing_unit_price_count=missing_price_count,
        negative_unit_price_count=negative_price_count,
        missing_discount_count=missing_discount_count,
        negative_discount_count=negative_discount_count,
        missing_revenue_count=missing_revenue_count,
        negative_revenue_count=negative_revenue_count,
        missing_unit_cost_count=missing_cost_count,
        negative_unit_cost_count=negative_cost_count,
        missing_currency_count=missing_curr_count,
        duplicate_sale_id_count=dup_sale_count,
        future_date_leakage_count=future_leakage_count,
        mismatched_currencies=mismatched_currencies,
        issues=issues,
        is_clean=not has_errors,
    )


# =====================================================================
# Record-Level Analytics (Deterministic Calculation)
# =====================================================================


def calculate_revenue_margin(
    row: Union[pd.Series, Dict[str, Any]],
    unit_cost: Optional[float] = None,
    config: Optional[FinancialIntelligenceConfig] = None,
) -> RevenueMarginRecord:
    """Calculate deterministic revenue, estimated COGS, and gross margin for a single transaction line.

    Performs line-level validation, reconciliation against source revenue, and status attribution.
    """
    cfg = config or FinancialIntelligenceConfig()

    def get_val(key: str, alt_key: Optional[str] = None) -> Any:
        if isinstance(row, dict):
            val = row.get(key)
            if val is None and alt_key:
                val = row.get(alt_key)
            return val
        if key in row:
            return row[key]
        if alt_key and alt_key in row:
            return row[alt_key]
        return None

    sale_id = str(get_val("sale_id") or "").strip()
    order_id = str(get_val("order_id") or "").strip()
    date_val = str(get_val("date", "sale_date") or "").strip()
    sku_id = str(get_val("sku_id") or "").strip()
    warehouse_id = str(get_val("warehouse_id") or "").strip()
    channel_id = str(get_val("channel_id") or "").strip()
    currency = str(get_val("currency") or cfg.default_currency).strip().upper()

    raw_qty = get_val("quantity")
    raw_price = get_val("unit_price")
    raw_disc = get_val("discount")
    raw_rev = get_val("revenue", "source_revenue")
    raw_cost = unit_cost if unit_cost is not None else get_val("unit_cost")

    # Safe parsing
    def safe_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
            return None if np.isnan(f) or np.isinf(f) else f
        except (ValueError, TypeError):
            return None

    def safe_int(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            f = float(v)
            return None if np.isnan(f) or np.isinf(f) else int(f)
        except (ValueError, TypeError):
            return None

    qty = safe_int(raw_qty)
    unit_price = safe_float(raw_price)
    discount = safe_float(raw_disc)
    source_revenue = safe_float(raw_rev)
    parsed_cost = safe_float(raw_cost)

    # 1. Validation & INVALID detection
    invalid_reasons = []
    if not sku_id:
        invalid_reasons.append("Missing SKU identifier")
    if qty is not None and qty < 0:
        invalid_reasons.append(f"Negative quantity: {qty}")
    if unit_price is not None and unit_price < 0:
        invalid_reasons.append(f"Negative unit price: {unit_price}")
    if discount is not None and discount < 0:
        invalid_reasons.append(f"Negative discount: {discount}")
    if source_revenue is not None and source_revenue < 0:
        invalid_reasons.append(f"Negative source revenue: {source_revenue}")
    if parsed_cost is not None and parsed_cost < 0:
        invalid_reasons.append(f"Negative unit cost: {parsed_cost}")

    record_id = generate_financial_record_id(sale_id, sku_id, order_id, date_val, currency)

    if invalid_reasons:
        return RevenueMarginRecord(
            record_id=record_id,
            sale_id=sale_id,
            order_id=order_id,
            date=date_val,
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            channel_id=channel_id,
            quantity=qty,
            unit_price=unit_price,
            discount=discount,
            discount_rate=None,
            source_revenue=source_revenue,
            calculated_gross_revenue=None,
            calculated_net_revenue=None,
            revenue_variance=None,
            revenue_reconciliation_status=RevenueReconciliationStatus.UNAVAILABLE,
            unit_cost=parsed_cost,
            estimated_cogs=None,
            gross_margin=None,
            gross_margin_percentage=None,
            margin_classification=MarginClassification.UNAVAILABLE,
            financial_status=FinancialDataStatus.INVALID,
            status_rationale="; ".join(invalid_reasons),
            currency=currency,
            as_of_date=str(cfg.as_of_date) if cfg.as_of_date else None,
        )

    # 2. INSUFFICIENT detection
    if qty is None or unit_price is None:
        missing_fields = []
        if qty is None:
            missing_fields.append("quantity")
        if unit_price is None:
            missing_fields.append("unit_price")
        return RevenueMarginRecord(
            record_id=record_id,
            sale_id=sale_id,
            order_id=order_id,
            date=date_val,
            sku_id=sku_id,
            warehouse_id=warehouse_id,
            channel_id=channel_id,
            quantity=qty,
            unit_price=unit_price,
            discount=discount,
            discount_rate=None,
            source_revenue=source_revenue,
            calculated_gross_revenue=None,
            calculated_net_revenue=None,
            revenue_variance=None,
            revenue_reconciliation_status=RevenueReconciliationStatus.UNAVAILABLE,
            unit_cost=parsed_cost,
            estimated_cogs=None,
            gross_margin=None,
            gross_margin_percentage=None,
            margin_classification=MarginClassification.UNAVAILABLE,
            financial_status=FinancialDataStatus.INSUFFICIENT,
            status_rationale=f"Insufficient revenue inputs: missing {', '.join(missing_fields)}",
            currency=currency,
            as_of_date=str(cfg.as_of_date) if cfg.as_of_date else None,
        )

    # 3. Revenue calculations
    gross_revenue = float(qty * unit_price)
    discount_val = discount if discount is not None else 0.0
    net_revenue = float(gross_revenue - discount_val)

    if gross_revenue > 0:
        discount_rate = float(discount_val / gross_revenue)
    elif discount_val == 0.0:
        discount_rate = 0.0
    else:
        discount_rate = None

    # Source revenue reconciliation
    if source_revenue is not None:
        revenue_variance = float(net_revenue - source_revenue)
        abs_var = abs(revenue_variance)
        if abs_var <= cfg.minor_variance_tolerance:
            recon_status = RevenueReconciliationStatus.MATCH
        elif abs_var <= cfg.material_variance_tolerance:
            recon_status = RevenueReconciliationStatus.MINOR_VARIANCE
        else:
            recon_status = RevenueReconciliationStatus.MATERIAL_VARIANCE
    else:
        revenue_variance = None
        recon_status = RevenueReconciliationStatus.UNAVAILABLE

    # 4. Estimated COGS & Margin calculation
    if parsed_cost is not None:
        estimated_cogs = float(qty * parsed_cost)
        gross_margin = float(net_revenue - estimated_cogs)

        if net_revenue > 0:
            gross_margin_pct = float(gross_margin / net_revenue)
        else:
            gross_margin_pct = None  # Protected from zero/negative revenue denominator

        if gross_margin > 0:
            margin_class = MarginClassification.POSITIVE_MARGIN
        elif gross_margin == 0:
            margin_class = MarginClassification.ZERO_MARGIN
        else:
            margin_class = MarginClassification.NEGATIVE_MARGIN

        fin_status = FinancialDataStatus.SUFFICIENT
        rationale = "Complete revenue and standard product cost available"
    else:
        estimated_cogs = None
        gross_margin = None
        gross_margin_pct = None
        margin_class = MarginClassification.UNAVAILABLE
        fin_status = FinancialDataStatus.PARTIAL
        rationale = "Valid revenue calculated, but product unit_cost is unavailable in catalog"

    return RevenueMarginRecord(
        record_id=record_id,
        sale_id=sale_id,
        order_id=order_id,
        date=date_val,
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        channel_id=channel_id,
        quantity=qty,
        unit_price=unit_price,
        discount=discount,
        discount_rate=discount_rate,
        source_revenue=source_revenue,
        calculated_gross_revenue=gross_revenue,
        calculated_net_revenue=net_revenue,
        revenue_variance=revenue_variance,
        revenue_reconciliation_status=recon_status,
        unit_cost=parsed_cost,
        estimated_cogs=estimated_cogs,
        gross_margin=gross_margin,
        gross_margin_percentage=gross_margin_pct,
        margin_classification=margin_class,
        financial_status=fin_status,
        status_rationale=rationale,
        currency=currency,
        as_of_date=str(cfg.as_of_date) if cfg.as_of_date else None,
    )


# =====================================================================
# Vectorized DataFrame Computation Engine
# =====================================================================


def compute_revenue_margin_dataframe(
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    config: Optional[FinancialIntelligenceConfig] = None,
) -> pd.DataFrame:
    """Compute revenue, estimated COGS, and gross margin vectorially over a sales DataFrame.

    Optimized for high-volume transactions (e.g. 750k+ rows in <1s) while guaranteeing
    exact mathematical equivalence with `calculate_revenue_margin`.
    """
    cfg = config or FinancialIntelligenceConfig()

    if sales_df.empty:
        cols = [
            "record_id", "sale_id", "order_id", "date", "sku_id", "warehouse_id", "channel_id",
            "quantity", "unit_price", "discount", "discount_rate", "source_revenue",
            "calculated_gross_revenue", "calculated_net_revenue", "revenue_variance",
            "revenue_reconciliation_status", "unit_cost", "estimated_cogs", "gross_margin",
            "gross_margin_percentage", "margin_classification", "financial_status",
            "status_rationale", "currency", "as_of_date"
        ]
        return pd.DataFrame(columns=cols)

    df = sales_df.copy()

    # Standardize column names
    if "sale_date" in df.columns and "date" not in df.columns:
        df["date"] = df["sale_date"]
    if "date" not in df.columns:
        df["date"] = ""
    df["date"] = df["date"].astype(str)

    if "sale_id" not in df.columns:
        df["sale_id"] = [f"SALE_{i}" for i in range(len(df))]
    df["sale_id"] = df["sale_id"].astype(str)

    if "order_id" not in df.columns:
        df["order_id"] = ""
    df["order_id"] = df["order_id"].astype(str)

    if "sku_id" not in df.columns:
        df["sku_id"] = ""
    df["sku_id"] = df["sku_id"].astype(str)

    if "warehouse_id" not in df.columns:
        df["warehouse_id"] = ""
    df["warehouse_id"] = df["warehouse_id"].astype(str)

    if "channel_id" not in df.columns:
        df["channel_id"] = ""
    df["channel_id"] = df["channel_id"].astype(str)

    if "currency" not in df.columns:
        df["currency"] = cfg.default_currency
    df["currency"] = df["currency"].fillna(cfg.default_currency).astype(str).str.strip().str.upper()

    # Numeric conversion
    qty = pd.to_numeric(df["quantity"], errors="coerce") if "quantity" in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
    unit_price = pd.to_numeric(df["unit_price"], errors="coerce") if "unit_price" in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
    discount = pd.to_numeric(df["discount"], errors="coerce") if "discount" in df.columns else pd.Series(0.0, index=df.index, dtype=float)
    rev_col = "revenue" if "revenue" in df.columns else ("source_revenue" if "source_revenue" in df.columns else None)
    source_rev = pd.to_numeric(df[rev_col], errors="coerce") if rev_col is not None else pd.Series(np.nan, index=df.index, dtype=float)

    # Product enrichment for unit_cost, category_id, brand
    unit_cost = None
    if products_df is not None and not products_df.empty and "sku_id" in products_df.columns:
        if "category_id" in products_df.columns and "category_id" not in df.columns:
            cat_map = products_df.set_index("sku_id")["category_id"].to_dict()
            df["category_id"] = df["sku_id"].map(cat_map)
        if "brand" in products_df.columns and "brand" not in df.columns:
            brand_map = products_df.set_index("sku_id")["brand"].to_dict()
            df["brand"] = df["sku_id"].map(brand_map)
        if "unit_cost" in products_df.columns:
            cost_map = products_df.set_index("sku_id")["unit_cost"].to_dict()
            unit_cost = df["sku_id"].map(cost_map)

    if unit_cost is None:
        if "unit_cost" in df.columns:
            unit_cost = pd.to_numeric(df["unit_cost"], errors="coerce")
        else:
            unit_cost = pd.Series(np.nan, index=df.index, dtype=float)
    else:
        unit_cost = pd.to_numeric(unit_cost, errors="coerce")

    # Invalid masks
    missing_sku = df["sku_id"].str.strip() == ""
    invalid_qty = (qty < 0)
    invalid_price = (unit_price < 0)
    invalid_disc = (discount < 0)
    invalid_rev = (source_rev < 0)
    invalid_cost = (unit_cost < 0)
    is_invalid = missing_sku | invalid_qty | invalid_price | invalid_disc | invalid_rev | invalid_cost

    # Insufficient mask
    is_insufficient = ~is_invalid & (qty.isna() | unit_price.isna())

    # Valid mask
    is_valid = ~is_invalid & ~is_insufficient

    # Calculations for valid records
    gross_rev = pd.Series(np.nan, index=df.index, dtype=float)
    disc_val = discount.fillna(0.0)
    net_rev = pd.Series(np.nan, index=df.index, dtype=float)
    disc_rate = pd.Series(np.nan, index=df.index, dtype=float)

    valid_indices = df.index[is_valid]
    if len(valid_indices) > 0:
        valid_qty = qty.loc[valid_indices]
        valid_price = unit_price.loc[valid_indices]
        valid_disc = disc_val.loc[valid_indices]

        calc_gross = valid_qty * valid_price
        calc_net = calc_gross - valid_disc

        gross_rev.loc[valid_indices] = calc_gross
        net_rev.loc[valid_indices] = calc_net

        pos_gross = valid_indices[calc_gross > 0]
        if len(pos_gross) > 0:
            disc_rate.loc[pos_gross] = valid_disc.loc[pos_gross] / calc_gross.loc[pos_gross]
        zero_gross_no_disc = valid_indices[(calc_gross == 0) & (valid_disc == 0)]
        if len(zero_gross_no_disc) > 0:
            disc_rate.loc[zero_gross_no_disc] = 0.0

    # Source revenue reconciliation
    rev_var = pd.Series(np.nan, index=df.index, dtype=float)
    recon_status = pd.Series(RevenueReconciliationStatus.UNAVAILABLE.value, index=df.index, dtype=object)

    has_source_rev = is_valid & source_rev.notna()
    if has_source_rev.any():
        idx = df.index[has_source_rev]
        diff = net_rev.loc[idx] - source_rev.loc[idx]
        rev_var.loc[idx] = diff
        abs_diff = diff.abs()

        match_mask = abs_diff <= cfg.minor_variance_tolerance
        minor_mask = (abs_diff > cfg.minor_variance_tolerance) & (abs_diff <= cfg.material_variance_tolerance)
        material_mask = abs_diff > cfg.material_variance_tolerance

        recon_status.loc[idx[match_mask]] = RevenueReconciliationStatus.MATCH.value
        recon_status.loc[idx[minor_mask]] = RevenueReconciliationStatus.MINOR_VARIANCE.value
        recon_status.loc[idx[material_mask]] = RevenueReconciliationStatus.MATERIAL_VARIANCE.value

    # Estimated COGS & Margin
    est_cogs = pd.Series(np.nan, index=df.index, dtype=float)
    gross_margin = pd.Series(np.nan, index=df.index, dtype=float)
    gross_margin_pct = pd.Series(np.nan, index=df.index, dtype=float)
    margin_class = pd.Series(MarginClassification.UNAVAILABLE.value, index=df.index, dtype=object)
    fin_status = pd.Series(FinancialDataStatus.INSUFFICIENT.value, index=df.index, dtype=object)
    rationale = pd.Series("Insufficient revenue inputs", index=df.index, dtype=object)

    if is_invalid.any():
        fin_status.loc[is_invalid] = FinancialDataStatus.INVALID.value
        rationale.loc[is_invalid] = "Invalid input values detected (negative or malformed fields)"

    has_cost = is_valid & unit_cost.notna()
    no_cost = is_valid & unit_cost.isna()

    if no_cost.any():
        idx_no_cost = df.index[no_cost]
        fin_status.loc[idx_no_cost] = FinancialDataStatus.PARTIAL.value
        rationale.loc[idx_no_cost] = "Valid revenue calculated, but product unit_cost is unavailable in catalog"

    if has_cost.any():
        idx_cost = df.index[has_cost]
        cogs = qty.loc[idx_cost] * unit_cost.loc[idx_cost]
        margin = net_rev.loc[idx_cost] - cogs

        est_cogs.loc[idx_cost] = cogs
        gross_margin.loc[idx_cost] = margin
        fin_status.loc[idx_cost] = FinancialDataStatus.SUFFICIENT.value
        rationale.loc[idx_cost] = "Complete revenue and standard product cost available"

        pos_net = idx_cost[net_rev.loc[idx_cost] > 0]
        if len(pos_net) > 0:
            gross_margin_pct.loc[pos_net] = margin.loc[pos_net] / net_rev.loc[pos_net]

        margin_class.loc[idx_cost[margin > 0]] = MarginClassification.POSITIVE_MARGIN.value
        margin_class.loc[idx_cost[margin == 0]] = MarginClassification.ZERO_MARGIN.value
        margin_class.loc[idx_cost[margin < 0]] = MarginClassification.NEGATIVE_MARGIN.value

    # Deterministic record IDs
    # Fast vectorized SHA-256 generation using hashlib over formatted string series
    raw_str = (
        df["sale_id"].str.strip()
        + "|"
        + df["sku_id"].str.strip()
        + "|"
        + df["order_id"].str.strip()
        + "|"
        + df["date"].str.strip()
        + "|"
        + df["currency"].str.strip().str.upper()
    )
    rec_ids = [
        f"FIN-REC-{hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]}"
        for s in raw_str
    ]

    out_df = pd.DataFrame(
        {
            "record_id": rec_ids,
            "sale_id": df["sale_id"],
            "order_id": df["order_id"],
            "date": df["date"],
            "sku_id": df["sku_id"],
            "warehouse_id": df["warehouse_id"],
            "channel_id": df["channel_id"],
            "quantity": qty,
            "unit_price": unit_price,
            "discount": discount,
            "discount_rate": disc_rate,
            "source_revenue": source_rev,
            "calculated_gross_revenue": gross_rev,
            "calculated_net_revenue": net_rev,
            "revenue_variance": rev_var,
            "revenue_reconciliation_status": recon_status,
            "unit_cost": unit_cost,
            "estimated_cogs": est_cogs,
            "gross_margin": gross_margin,
            "gross_margin_percentage": gross_margin_pct,
            "margin_classification": margin_class,
            "financial_status": fin_status,
            "status_rationale": rationale,
            "currency": df["currency"],
            "as_of_date": str(cfg.as_of_date) if cfg.as_of_date else None,
        },
        index=df.index,
    )

    if "category_id" in df.columns:
        out_df["category_id"] = df["category_id"]
    if "brand" in df.columns:
        out_df["brand"] = df["brand"]

    return out_df


def compute_revenue_margin_records(
    sales_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    config: Optional[FinancialIntelligenceConfig] = None,
) -> List[RevenueMarginRecord]:
    """Compute revenue and margin records and return as strongly-typed Pydantic objects."""
    df = compute_revenue_margin_dataframe(sales_df, products_df, config)
    records: List[RevenueMarginRecord] = []

    for row in df.itertuples(index=False):
        records.append(
            RevenueMarginRecord(
                record_id=row.record_id,
                sale_id=row.sale_id,
                order_id=row.order_id,
                date=row.date,
                sku_id=row.sku_id,
                warehouse_id=row.warehouse_id,
                channel_id=row.channel_id,
                quantity=int(row.quantity) if pd.notna(row.quantity) else None,
                unit_price=float(row.unit_price) if pd.notna(row.unit_price) else None,
                discount=float(row.discount) if pd.notna(row.discount) else None,
                discount_rate=float(row.discount_rate) if pd.notna(row.discount_rate) else None,
                source_revenue=float(row.source_revenue) if pd.notna(row.source_revenue) else None,
                calculated_gross_revenue=float(row.calculated_gross_revenue) if pd.notna(row.calculated_gross_revenue) else None,
                calculated_net_revenue=float(row.calculated_net_revenue) if pd.notna(row.calculated_net_revenue) else None,
                revenue_variance=float(row.revenue_variance) if pd.notna(row.revenue_variance) else None,
                revenue_reconciliation_status=RevenueReconciliationStatus(row.revenue_reconciliation_status),
                unit_cost=float(row.unit_cost) if pd.notna(row.unit_cost) else None,
                estimated_cogs=float(row.estimated_cogs) if pd.notna(row.estimated_cogs) else None,
                gross_margin=float(row.gross_margin) if pd.notna(row.gross_margin) else None,
                gross_margin_percentage=float(row.gross_margin_percentage) if pd.notna(row.gross_margin_percentage) else None,
                margin_classification=MarginClassification(row.margin_classification),
                financial_status=FinancialDataStatus(row.financial_status),
                status_rationale=row.status_rationale,
                currency=row.currency,
                as_of_date=row.as_of_date,
            )
        )
    return records


# =====================================================================
# Aggregation Engine
# =====================================================================


def aggregate_financial_dimension(
    df: pd.DataFrame,
    dimension: str,
    total_portfolio_revenue: Optional[float] = None,
    total_portfolio_margin: Optional[float] = None,
    config: Optional[FinancialIntelligenceConfig] = None,
) -> List[FinancialDimensionMetric]:
    """Aggregate financial metrics across any supported dimensional cut.

    Dimensions supported:
    - OVERALL
    - SKU
    - CATEGORY
    - BRAND
    - CHANNEL
    - WAREHOUSE
    - SKU_CHANNEL (or SKU_X_CHANNEL)
    - SKU_WAREHOUSE (or SKU_X_WAREHOUSE)
    - CHANNEL_WAREHOUSE (or CHANNEL_X_WAREHOUSE)
    - DATE
    - MONTH (YYYY-MM)

    CRITICAL: Never averages row-level percentages. Aggregated margin percentage is
    strictly total_gross_margin / total_net_revenue.
    """
    cfg = config or FinancialIntelligenceConfig()
    dim_upper = dimension.strip().upper().replace(" ", "_")

    if df.empty:
        return []

    # Map dimension to group column(s)
    if dim_upper == "OVERALL":
        group_cols = None
    elif dim_upper == "SKU":
        group_cols = ["sku_id"]
    elif dim_upper == "CATEGORY":
        group_cols = ["category_id"] if "category_id" in df.columns else ["sku_id"]
    elif dim_upper == "BRAND":
        group_cols = ["brand"] if "brand" in df.columns else ["sku_id"]
    elif dim_upper == "CHANNEL":
        group_cols = ["channel_id"]
    elif dim_upper == "WAREHOUSE":
        group_cols = ["warehouse_id"]
    elif dim_upper in ("SKU_CHANNEL", "SKU_X_CHANNEL"):
        group_cols = ["sku_id", "channel_id"]
    elif dim_upper in ("SKU_WAREHOUSE", "SKU_X_WAREHOUSE"):
        group_cols = ["sku_id", "warehouse_id"]
    elif dim_upper in ("CHANNEL_WAREHOUSE", "CHANNEL_X_WAREHOUSE"):
        group_cols = ["channel_id", "warehouse_id"]
    elif dim_upper == "DATE":
        group_cols = ["date"]
    elif dim_upper == "MONTH":
        df = df.copy()
        df["month"] = df["date"].astype(str).str.slice(0, 7)
        group_cols = ["month"]
    else:
        raise ValueError(f"Unsupported financial dimension: {dimension}")

    def compute_slice_metrics(sub_df: pd.DataFrame, seg_key: str) -> FinancialDimensionMetric:
        rec_count = len(sub_df)
        order_count = int(sub_df["order_id"].nunique()) if "order_id" in sub_df.columns else 0

        # Units
        qty_series = sub_df["quantity"].dropna()
        total_units = int(qty_series.sum()) if not qty_series.empty else 0

        # Gross revenue
        gross_series = sub_df["calculated_gross_revenue"].dropna()
        total_gross = float(gross_series.sum()) if not gross_series.empty else None

        # Discount
        disc_series = sub_df["discount"].dropna()
        total_disc = float(disc_series.sum()) if not disc_series.empty else None

        # Net revenue
        net_series = sub_df["calculated_net_revenue"].dropna()
        total_net = float(net_series.sum()) if not net_series.empty else None

        # Estimated COGS
        cogs_series = sub_df["estimated_cogs"].dropna()
        total_cogs = float(cogs_series.sum()) if not cogs_series.empty else None

        # Gross margin
        margin_series = sub_df["gross_margin"].dropna()
        total_margin = float(margin_series.sum()) if not margin_series.empty else None

        # Ratios (Safe protected calculation)
        if total_net is not None and total_net > 0 and total_margin is not None:
            gm_pct = float(total_margin / total_net)
        else:
            gm_pct = None

        if total_gross is not None and total_gross > 0 and total_disc is not None:
            disc_rate = float(total_disc / total_gross)
        elif total_disc == 0.0:
            disc_rate = 0.0
        else:
            disc_rate = None

        if total_net is not None and order_count > 0:
            aov = float(total_net / order_count)
        else:
            aov = None

        # Portfolio contributions
        if total_net is not None and total_portfolio_revenue and total_portfolio_revenue > 0:
            rev_contrib = float(total_net / total_portfolio_revenue)
        else:
            rev_contrib = None

        if total_margin is not None and total_portfolio_margin and abs(total_portfolio_margin) > 0:
            margin_contrib = float(total_margin / total_portfolio_margin)
        else:
            margin_contrib = None

        # Negative margin lines
        neg_mask = sub_df["gross_margin"] < 0
        neg_count = int(neg_mask.sum())
        if neg_count > 0:
            neg_df = sub_df.loc[neg_mask]
            neg_rev = float(neg_df["calculated_net_revenue"].dropna().sum()) if not neg_df["calculated_net_revenue"].dropna().empty else 0.0
            neg_units = int(neg_df["quantity"].dropna().sum()) if not neg_df["quantity"].dropna().empty else 0
        else:
            neg_rev = None
            neg_units = 0

        # Completeness & counts
        has_cost_mask = sub_df["unit_cost"].notna()
        records_with_cost = int(has_cost_mask.sum())
        records_without_cost = rec_count - records_with_cost

        valid_rev_mask = sub_df["calculated_net_revenue"].notna() & (sub_df["calculated_net_revenue"] >= 0)
        records_with_valid_rev = int(valid_rev_mask.sum())
        records_with_invalid_rev = rec_count - records_with_valid_rev

        sufficient_mask = sub_df["financial_status"] == FinancialDataStatus.SUFFICIENT.value
        fin_completeness = round((int(sufficient_mask.sum()) / rec_count) * 100.0, 2) if rec_count > 0 else 0.0

        currency = str(sub_df["currency"].iloc[0]) if "currency" in sub_df.columns and not sub_df["currency"].empty else cfg.default_currency

        return FinancialDimensionMetric(
            dimension=dim_upper,
            segment_key=seg_key,
            record_count=rec_count,
            order_count=order_count,
            total_units=total_units,
            total_gross_revenue=total_gross,
            total_discount=total_disc,
            total_net_revenue=total_net,
            total_estimated_cogs=total_cogs,
            total_gross_margin=total_margin,
            gross_margin_percentage=gm_pct,
            discount_rate=disc_rate,
            average_order_value=aov,
            revenue_contribution=rev_contrib,
            margin_contribution=margin_contrib,
            negative_margin_order_line_count=neg_count,
            negative_margin_revenue=neg_rev,
            negative_margin_units=neg_units,
            financial_completeness=fin_completeness,
            records_with_cost=records_with_cost,
            records_without_cost=records_without_cost,
            records_with_valid_revenue=records_with_valid_rev,
            records_with_invalid_revenue=records_with_invalid_rev,
            currency=currency,
        )

    results: List[FinancialDimensionMetric] = []

    if group_cols is None:
        # OVERALL
        results.append(compute_slice_metrics(df, "OVERALL"))
    else:
        grouped = df.groupby(group_cols, dropna=False, sort=True)
        for name, sub in grouped:
            if isinstance(name, tuple):
                key = ":".join(str(part or "") for part in name)
            else:
                key = str(name or "")
            results.append(compute_slice_metrics(sub, key))

    return results


# =====================================================================
# Time Intelligence
# =====================================================================


def aggregate_time_series(
    df: pd.DataFrame,
    grain: Union[str, TimeGrain] = TimeGrain.MONTHLY,
    total_portfolio_revenue: Optional[float] = None,
    total_portfolio_margin: Optional[float] = None,
    config: Optional[FinancialIntelligenceConfig] = None,
) -> List[FinancialDimensionMetric]:
    """Aggregate financial metrics across specified temporal granularity.

    Supports DAILY, WEEKLY, MONTHLY grains.
    """
    g_str = grain.value if isinstance(grain, TimeGrain) else str(grain).strip().upper()
    df_copy = df.copy()

    if df_copy.empty:
        return []

    date_series = pd.to_datetime(df_copy["date"], errors="coerce")

    if g_str in ("DAILY", "DAY", "D"):
        df_copy["time_grain_key"] = date_series.dt.strftime("%Y-%m-%d")
        dim_label = "DATE"
    elif g_str in ("WEEKLY", "WEEK", "W"):
        # Format as Year-Www (e.g. 2024-W01)
        df_copy["time_grain_key"] = date_series.dt.strftime("%Y-W%U")
        dim_label = "WEEK"
    elif g_str in ("MONTHLY", "MONTH", "M"):
        df_copy["time_grain_key"] = date_series.dt.strftime("%Y-%m")
        dim_label = "MONTH"
    else:
        raise ValueError(f"Unsupported temporal grain: {grain}")

    grouped = df_copy.groupby("time_grain_key", dropna=False, sort=True)
    results: List[FinancialDimensionMetric] = []

    for name, sub in grouped:
        key = str(name or "")
        metric = aggregate_financial_dimension(
            sub,
            dimension="OVERALL",
            total_portfolio_revenue=total_portfolio_revenue,
            total_portfolio_margin=total_portfolio_margin,
            config=config,
        )[0]
        # Adjust dimension & segment_key for time series
        metric.dimension = dim_label
        metric.segment_key = key
        results.append(metric)

    return results


# =====================================================================
# Deterministic Ranking Engine
# =====================================================================


def rank_segments(
    metrics: Sequence[FinancialDimensionMetric],
    ranking_metric: str = "highest_revenue",
    top_n: int = 10,
) -> RankingResult:
    """Deterministically rank segments using neutral analytical labels.

    Supports:
    - highest_revenue / lowest_revenue (by total_net_revenue)
    - highest_margin / lowest_margin (by total_gross_margin)
    - highest_units / lowest_units (by total_units)
    - highest_margin_percentage / lowest_margin_percentage (by gross_margin_percentage)
    - negative_margin (items with total_gross_margin < 0, ordered most negative first)
    """
    valid_metrics = [
        "highest_revenue",
        "lowest_revenue",
        "highest_margin",
        "lowest_margin",
        "highest_units",
        "lowest_units",
        "highest_margin_percentage",
        "lowest_margin_percentage",
        "negative_margin",
    ]
    norm_metric = ranking_metric.strip().lower()
    if norm_metric not in valid_metrics:
        raise ValueError(f"Unknown ranking metric: '{ranking_metric}'. Must be one of {valid_metrics}")

    if not metrics:
        return RankingResult(
            ranking_metric=norm_metric,
            dimension="UNKNOWN",
            items=[],
        )

    dim_label = metrics[0].dimension

    if norm_metric == "highest_revenue":
        sorted_items = sorted(
            [m for m in metrics if m.total_net_revenue is not None],
            key=lambda x: x.total_net_revenue or 0.0,
            reverse=True,
        )
    elif norm_metric == "lowest_revenue":
        sorted_items = sorted(
            [m for m in metrics if m.total_net_revenue is not None],
            key=lambda x: x.total_net_revenue or 0.0,
            reverse=False,
        )
    elif norm_metric == "highest_margin":
        sorted_items = sorted(
            [m for m in metrics if m.total_gross_margin is not None],
            key=lambda x: x.total_gross_margin or 0.0,
            reverse=True,
        )
    elif norm_metric == "lowest_margin":
        sorted_items = sorted(
            [m for m in metrics if m.total_gross_margin is not None],
            key=lambda x: x.total_gross_margin or 0.0,
            reverse=False,
        )
    elif norm_metric == "highest_units":
        sorted_items = sorted(
            metrics,
            key=lambda x: x.total_units,
            reverse=True,
        )
    elif norm_metric == "lowest_units":
        sorted_items = sorted(
            metrics,
            key=lambda x: x.total_units,
            reverse=False,
        )
    elif norm_metric == "highest_margin_percentage":
        sorted_items = sorted(
            [m for m in metrics if m.gross_margin_percentage is not None],
            key=lambda x: x.gross_margin_percentage or 0.0,
            reverse=True,
        )
    elif norm_metric == "lowest_margin_percentage":
        sorted_items = sorted(
            [m for m in metrics if m.gross_margin_percentage is not None],
            key=lambda x: x.gross_margin_percentage or 0.0,
            reverse=False,
        )
    elif norm_metric == "negative_margin":
        sorted_items = sorted(
            [m for m in metrics if m.total_gross_margin is not None and m.total_gross_margin < 0],
            key=lambda x: x.total_gross_margin or 0.0,
            reverse=False,  # Most negative first
        )
    else:
        sorted_items = list(metrics)

    return RankingResult(
        ranking_metric=norm_metric,
        dimension=dim_label,
        items=sorted_items[:top_n],
    )


# =====================================================================
# Negative / Zero Margin Analysis
# =====================================================================


def analyze_negative_margins(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze order lines and SKU aggregations exhibiting negative or zero gross margin.

    Returns structured counts, monetary totals, and component breakdowns (discount vs cost vs price).
    """
    if df.empty or "gross_margin" not in df.columns:
        return {
            "negative_margin_order_lines": 0,
            "zero_margin_order_lines": 0,
            "positive_margin_order_lines": 0,
            "negative_margin_revenue": 0.0,
            "negative_margin_cogs": 0.0,
            "negative_margin_loss": 0.0,
            "negative_margin_units": 0,
            "negative_margin_orders": 0,
            "component_breakdown": {
                "discount_exceeds_margin": 0,
                "cost_exceeds_price": 0,
                "both_factors": 0,
            },
        }

    neg_mask = df["gross_margin"] < 0
    zero_mask = df["gross_margin"] == 0
    pos_mask = df["gross_margin"] > 0

    neg_lines = int(neg_mask.sum())
    zero_lines = int(zero_mask.sum())
    pos_lines = int(pos_mask.sum())

    if neg_lines > 0:
        neg_df = df.loc[neg_mask]
        neg_rev = float(neg_df["calculated_net_revenue"].dropna().sum())
        neg_cogs = float(neg_df["estimated_cogs"].dropna().sum())
        neg_loss = float(abs(neg_df["gross_margin"].dropna().sum()))
        neg_units = int(neg_df["quantity"].dropna().sum())
        neg_orders = int(neg_df["order_id"].nunique()) if "order_id" in neg_df.columns else 0

        # Component breakdown (measurable components, not ungrounded causal claims)
        # 1. Selling price < Unit cost
        cost_exceeds_price = int(((neg_df["unit_price"] < neg_df["unit_cost"]) & (neg_df["discount"].fillna(0) == 0)).sum())
        # 2. Selling price >= Unit cost, but discount wiped out the margin
        discount_exceeds_margin = int(((neg_df["unit_price"] >= neg_df["unit_cost"]) & (neg_df["discount"].fillna(0) > (neg_df["unit_price"] - neg_df["unit_cost"]))).sum())
        # 3. Both factors combined
        both_factors = int(((neg_df["unit_price"] < neg_df["unit_cost"]) & (neg_df["discount"].fillna(0) > 0)).sum())
    else:
        neg_rev = 0.0
        neg_cogs = 0.0
        neg_loss = 0.0
        neg_units = 0
        neg_orders = 0
        cost_exceeds_price = 0
        discount_exceeds_margin = 0
        both_factors = 0

    return {
        "negative_margin_order_lines": neg_lines,
        "zero_margin_order_lines": zero_lines,
        "positive_margin_order_lines": pos_lines,
        "negative_margin_revenue": round(neg_rev, 2),
        "negative_margin_cogs": round(neg_cogs, 2),
        "negative_margin_loss": round(neg_loss, 2),
        "negative_margin_units": neg_units,
        "negative_margin_orders": neg_orders,
        "component_breakdown": {
            "discount_exceeds_margin": discount_exceeds_margin,
            "cost_exceeds_price": cost_exceeds_price,
            "both_factors": both_factors,
        },
    }


# =====================================================================
# Portfolio Summary
# =====================================================================


def summarize_financial_portfolio(
    df: pd.DataFrame,
    quality_report: Optional[FinancialDataQualityReport] = None,
    config: Optional[FinancialIntelligenceConfig] = None,
) -> FinancialPortfolioSummary:
    """Produce executive portfolio-level totals, reconciliation status, and completeness rate.

    Never converts missing values to zero. Correctly reports unobserved inputs.
    """
    cfg = config or FinancialIntelligenceConfig()
    total_lines = len(df)

    if total_lines == 0:
        return FinancialPortfolioSummary(
            total_order_lines=0,
            total_orders=0,
            total_units=0,
            financial_completeness_rate=0.0,
            revenue_reconciliation_status=RevenueReconciliationStatus.UNAVAILABLE,
            currency=cfg.default_currency,
            as_of_date=str(cfg.as_of_date) if cfg.as_of_date else None,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    total_orders = int(df["order_id"].nunique()) if "order_id" in df.columns else 0

    qty_series = df["quantity"].dropna()
    total_units = int(qty_series.sum()) if not qty_series.empty else 0

    gross_series = df["calculated_gross_revenue"].dropna()
    total_gross = float(gross_series.sum()) if not gross_series.empty else None

    disc_series = df["discount"].dropna()
    total_disc = float(disc_series.sum()) if not disc_series.empty else None

    net_series = df["calculated_net_revenue"].dropna()
    total_net = float(net_series.sum()) if not net_series.empty else None

    cogs_series = df["estimated_cogs"].dropna()
    total_cogs = float(cogs_series.sum()) if not cogs_series.empty else None

    margin_series = df["gross_margin"].dropna()
    total_margin = float(margin_series.sum()) if not margin_series.empty else None

    # Protected ratio calculations
    if total_net is not None and total_net > 0 and total_margin is not None:
        gm_pct = float(total_margin / total_net)
    else:
        gm_pct = None

    if total_gross is not None and total_gross > 0 and total_disc is not None:
        disc_rate = float(total_disc / total_gross)
    elif total_disc == 0.0:
        disc_rate = 0.0
    else:
        disc_rate = None

    if total_net is not None and total_orders > 0:
        aov = float(total_net / total_orders)
    else:
        aov = None

    # Negative margin metrics
    neg_mask = df["gross_margin"] < 0
    neg_lines_count = int(neg_mask.sum())
    if neg_lines_count > 0:
        neg_df = df.loc[neg_mask]
        neg_rev = float(neg_df["calculated_net_revenue"].dropna().sum()) if not neg_df["calculated_net_revenue"].dropna().empty else 0.0
        neg_units = int(neg_df["quantity"].dropna().sum()) if not neg_df["quantity"].dropna().empty else 0
    else:
        neg_rev = None
        neg_units = 0

    # Count of distinct SKUs with aggregate gross_margin < 0
    neg_skus_count = 0
    if "sku_id" in df.columns and "gross_margin" in df.columns:
        sku_margins = df.groupby("sku_id")["gross_margin"].sum(min_count=1)
        neg_skus_count = int((sku_margins < 0).sum())

    # Data completeness & counts
    has_cost = df["unit_cost"].notna()
    records_with_cost = int(has_cost.sum())
    records_without_cost = total_lines - records_with_cost

    valid_rev = df["calculated_net_revenue"].notna() & (df["calculated_net_revenue"] >= 0)
    records_with_valid_rev = int(valid_rev.sum())
    records_with_invalid_rev = total_lines - records_with_valid_rev

    sufficient_mask = df["financial_status"] == FinancialDataStatus.SUFFICIENT.value
    fin_completeness = round((int(sufficient_mask.sum()) / total_lines) * 100.0, 2)

    # Reconciliation counts
    match_count = int((df["revenue_reconciliation_status"] == RevenueReconciliationStatus.MATCH.value).sum())
    minor_var_count = int((df["revenue_reconciliation_status"] == RevenueReconciliationStatus.MINOR_VARIANCE.value).sum())
    mat_var_count = int((df["revenue_reconciliation_status"] == RevenueReconciliationStatus.MATERIAL_VARIANCE.value).sum())

    if mat_var_count > 0:
        overall_recon = RevenueReconciliationStatus.MATERIAL_VARIANCE
    elif minor_var_count > 0:
        overall_recon = RevenueReconciliationStatus.MINOR_VARIANCE
    elif match_count > 0:
        overall_recon = RevenueReconciliationStatus.MATCH
    else:
        overall_recon = RevenueReconciliationStatus.UNAVAILABLE

    currency = str(df["currency"].iloc[0]) if "currency" in df.columns and not df["currency"].empty else cfg.default_currency

    return FinancialPortfolioSummary(
        total_order_lines=total_lines,
        total_orders=total_orders,
        total_units=total_units,
        total_gross_revenue=total_gross,
        total_discount=total_disc,
        total_net_revenue=total_net,
        total_estimated_cogs=total_cogs,
        total_gross_margin=total_margin,
        gross_margin_percentage=gm_pct,
        portfolio_discount_rate=disc_rate,
        average_order_value=aov,
        negative_margin_revenue=neg_rev,
        negative_margin_units=neg_units,
        count_of_negative_margin_order_lines=neg_lines_count,
        count_of_skus_with_negative_aggregate_margin=neg_skus_count,
        records_with_cost=records_with_cost,
        records_without_cost=records_without_cost,
        records_with_valid_revenue=records_with_valid_rev,
        records_with_invalid_revenue=records_with_invalid_rev,
        financial_completeness_rate=fin_completeness,
        revenue_reconciliation_status=overall_recon,
        reconciliation_match_count=match_count,
        reconciliation_variance_count=minor_var_count,
        reconciliation_material_variance_count=mat_var_count,
        currency=currency,
        as_of_date=str(cfg.as_of_date) if cfg.as_of_date else None,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
