"""Returns Intelligence Service (Phase 5A).

Coordinates the end-to-end Returns Intelligence pipeline:
Data Quality Audit -> Anti-leakage Chronological Filtering -> Valuation Enrichment ->
Dimensional Metric Aggregations -> Reason Breakdown -> Time Series -> Investigation Flags.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from commerce_ai.returns.schemas import (
    InvestigationFlag,
    ReturnMetricRecord,
    ReturnReasonBreakdown,
    ReturnsConfig,
    ReturnsDataQualityReport,
    ReturnsIntelligenceResult,
    ReturnsSummary,
    ReturnTimeSeriesPoint,
)
from commerce_ai.returns.analytics import (
    assess_returns_data_quality,
    compute_dimensional_metrics,
    compute_reason_breakdowns,
    compute_time_series_points,
    enrich_returns_with_monetary_value,
    filter_by_as_of_date,
)


class ReturnsIntelligenceService:
    """Service providing comprehensive, factual, deterministic returns intelligence."""

    def __init__(self, config: Optional[ReturnsConfig] = None):
        self.config = config or ReturnsConfig()

    def analyze(
        self,
        returns: pd.DataFrame,
        sales: Optional[pd.DataFrame] = None,
        products: Optional[pd.DataFrame] = None,
        warehouses: Optional[pd.DataFrame] = None,
        channels: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[ReturnsConfig] = None,
        previous_period_days: Optional[int] = None,
        time_series_freq: str = "M",
    ) -> ReturnsIntelligenceResult:
        """Run the end-to-end returns intelligence pipeline.

        Args:
            returns: Customer returns dataset (raw or loaded).
            sales: Historical sales transactions for return rate calculation.
            products: Optional product master for valuation and validation.
            warehouses: Optional warehouse master for referential audit.
            channels: Optional sales channel master for referential audit.
            as_of_date: Reference evaluation cutoff date for anti-leakage compliance.
            config: Optional ReturnsConfig overriding instance configuration.
            previous_period_days: Optional lookback window (e.g. 30, 90 days) for period-over-period trend analysis.
            time_series_freq: Frequency bucket for time-series points ('D', 'W', 'M').

        Returns:
            ReturnsIntelligenceResult containing dimensional metrics, reasons, time series, and audit reports.
        """
        cfg = config or self.config

        # 1. Audit Data Quality on raw input before filtering
        data_quality_report = assess_returns_data_quality(
            returns_df=returns,
            products_df=products,
            warehouses_df=warehouses,
            channels_df=channels,
            known_reasons=cfg.known_reasons,
        )

        # 2. Apply Chronological Anti-Leakage Filtering
        ret_df = filter_by_as_of_date(returns, "return_date", as_of_date)
        sales_df = filter_by_as_of_date(sales, "date", as_of_date) if sales is not None else pd.DataFrame()

        # Handle Empty Returns Dataset
        if ret_df.empty:
            summary = ReturnsSummary(
                total_sold_units=int(sales_df["quantity"].sum()) if not sales_df.empty and "quantity" in sales_df.columns else 0,
                total_returned_units=0,
                overall_unit_return_rate=0.0 if not sales_df.empty and sales_df["quantity"].sum() > 0 else None,
                total_sales_value=float(sales_df["revenue"].sum()) if not sales_df.empty and "revenue" in sales_df.columns else None,
                total_return_value=0.0,
                overall_revenue_return_rate=0.0 if not sales_df.empty and sales_df.get("revenue", pd.Series(dtype=float)).sum() > 0 else None,
                total_return_events=0,
                total_orders=int(sales_df["order_id"].nunique()) if not sales_df.empty and "order_id" in sales_df.columns else 0,
                overall_order_return_rate=0.0 if not sales_df.empty and sales_df.get("order_id", pd.Series(dtype=str)).nunique() > 0 else None,
                top_return_reason=None,
                as_of_date=str(as_of_date) if as_of_date is not None else None,
            )
            return ReturnsIntelligenceResult(
                summary=summary,
                sku_metrics=[],
                channel_metrics=[],
                warehouse_metrics=[],
                reason_metrics=[],
                time_series=[],
                sku_channel_metrics=[],
                sku_warehouse_metrics=[],
                investigation_flags={},
                data_quality=data_quality_report,
                as_of_date=str(as_of_date) if as_of_date is not None else None,
            )

        # 3. Partition Previous Period for Trend Stability if requested
        prev_returns_df: Optional[pd.DataFrame] = None
        prev_sales_df: Optional[pd.DataFrame] = None

        if previous_period_days is not None and previous_period_days > 0 and as_of_date is not None:
            if isinstance(as_of_date, str):
                cur_end = datetime.fromisoformat(as_of_date).date()
            elif isinstance(as_of_date, datetime):
                cur_end = as_of_date.date()
            else:
                cur_end = as_of_date

            cur_start = cur_end - timedelta(days=previous_period_days)
            prev_start = cur_start - timedelta(days=previous_period_days)

            # Previous period slices
            if not ret_df.empty and "return_date" in ret_df.columns:
                r_dates = pd.to_datetime(ret_df["return_date"]).dt.date
                prev_returns_df = ret_df[(r_dates >= prev_start) & (r_dates < cur_start)].copy()
                ret_df = ret_df[(r_dates >= cur_start) & (r_dates <= cur_end)].copy()

            if not sales_df.empty and "date" in sales_df.columns:
                s_dates = pd.to_datetime(sales_df["date"]).dt.date
                prev_sales_df = sales_df[(s_dates >= prev_start) & (s_dates < cur_start)].copy()
                sales_df = sales_df[(s_dates >= cur_start) & (s_dates <= cur_end)].copy()

        # 4. Enrich Returns with Monetary Valuation
        ret_enriched = enrich_returns_with_monetary_value(
            returns_df=ret_df,
            sales_df=sales_df,
            products_df=products,
        )

        prev_ret_enriched = None
        if prev_returns_df is not None and not prev_returns_df.empty:
            prev_ret_enriched = enrich_returns_with_monetary_value(
                returns_df=prev_returns_df,
                sales_df=prev_sales_df,
                products_df=products,
            )

        # 5. Dimensional Aggregations
        sku_metrics = compute_dimensional_metrics(
            dimension_name="SKU",
            group_cols=["sku_id"],
            returns_df=ret_enriched,
            sales_df=sales_df,
            previous_period_returns_df=prev_ret_enriched,
            previous_period_sales_df=prev_sales_df,
            config=cfg,
        )

        channel_metrics = compute_dimensional_metrics(
            dimension_name="CHANNEL",
            group_cols=["channel_id"],
            returns_df=ret_enriched,
            sales_df=sales_df,
            previous_period_returns_df=prev_ret_enriched,
            previous_period_sales_df=prev_sales_df,
            config=cfg,
        )

        warehouse_metrics = compute_dimensional_metrics(
            dimension_name="WAREHOUSE",
            group_cols=["warehouse_id"],
            returns_df=ret_enriched,
            sales_df=sales_df,
            previous_period_returns_df=prev_ret_enriched,
            previous_period_sales_df=prev_sales_df,
            config=cfg,
        )

        sku_channel_metrics = compute_dimensional_metrics(
            dimension_name="SKU_CHANNEL",
            group_cols=["sku_id", "channel_id"],
            returns_df=ret_enriched,
            sales_df=sales_df,
            previous_period_returns_df=prev_ret_enriched,
            previous_period_sales_df=prev_sales_df,
            config=cfg,
        )

        sku_warehouse_metrics = compute_dimensional_metrics(
            dimension_name="SKU_WAREHOUSE",
            group_cols=["sku_id", "warehouse_id"],
            returns_df=ret_enriched,
            sales_df=sales_df,
            previous_period_returns_df=prev_ret_enriched,
            previous_period_sales_df=prev_sales_df,
            config=cfg,
        )

        # 6. Reasons and Time Series
        tot_ret_units = int(ret_enriched["quantity"].sum())
        tot_ret_events = len(ret_enriched)

        reason_metrics = compute_reason_breakdowns(
            returns_df=ret_enriched,
            total_returned_units=tot_ret_units,
            total_return_events=tot_ret_events,
        )

        time_series = compute_time_series_points(
            returns_df=ret_enriched,
            sales_df=sales_df,
            freq=time_series_freq,
        )

        # 7. Executive Portfolio Summary
        tot_sold_units = int(sales_df["quantity"].sum()) if not sales_df.empty and "quantity" in sales_df.columns else 0
        overall_unit_rate = round(tot_ret_units / tot_sold_units, 4) if tot_sold_units > 0 else None

        sales_rev_sum = sales_df["revenue"].sum(min_count=1) if not sales_df.empty and "revenue" in sales_df.columns else np.nan
        tot_sales_val = round(float(sales_rev_sum), 2) if pd.notna(sales_rev_sum) else None

        ret_val_sum = ret_enriched["estimated_return_value"].sum(min_count=1) if "estimated_return_value" in ret_enriched.columns else np.nan
        tot_ret_val = round(float(ret_val_sum), 2) if pd.notna(ret_val_sum) else None

        overall_rev_rate = (
            round(tot_ret_val / tot_sales_val, 4) if (tot_sales_val and tot_sales_val > 0 and tot_ret_val is not None) else None
        )

        tot_orders = sales_df["order_id"].nunique() if not sales_df.empty and "order_id" in sales_df.columns else 0
        ret_orders_cnt = ret_enriched["order_id"].nunique() if "order_id" in ret_enriched.columns else 0
        overall_order_rate = round(ret_orders_cnt / tot_orders, 4) if tot_orders > 0 else None

        top_reason_val = None
        if "reason" in ret_enriched.columns and not ret_enriched["reason"].dropna().empty:
            top_reason_val = str(ret_enriched["reason"].mode().iloc[0])

        summary = ReturnsSummary(
            total_sold_units=tot_sold_units,
            total_returned_units=tot_ret_units,
            overall_unit_return_rate=overall_unit_rate,
            total_sales_value=tot_sales_val,
            total_return_value=tot_ret_val,
            overall_revenue_return_rate=overall_rev_rate,
            total_return_events=tot_ret_events,
            total_orders=tot_orders,
            overall_order_return_rate=overall_order_rate,
            top_return_reason=top_reason_val,
            as_of_date=str(as_of_date) if as_of_date is not None else None,
        )

        # 8. Consolidate Investigation Flags
        investigation_map: Dict[str, List[str]] = {}
        all_metrics_lists = [sku_metrics, channel_metrics, warehouse_metrics, sku_channel_metrics, sku_warehouse_metrics]
        for m_list in all_metrics_lists:
            for rec in m_list:
                for flg in rec.investigation_flags:
                    if flg not in investigation_map:
                        investigation_map[flg] = []
                    item_desc = f"{rec.dimension}:{rec.key}"
                    if item_desc not in investigation_map[flg]:
                        investigation_map[flg].append(item_desc)

        return ReturnsIntelligenceResult(
            summary=summary,
            sku_metrics=sku_metrics,
            channel_metrics=channel_metrics,
            warehouse_metrics=warehouse_metrics,
            reason_metrics=reason_metrics,
            time_series=time_series,
            sku_channel_metrics=sku_channel_metrics,
            sku_warehouse_metrics=sku_warehouse_metrics,
            investigation_flags=investigation_map,
            data_quality=data_quality_report,
            as_of_date=str(as_of_date) if as_of_date is not None else None,
        )
