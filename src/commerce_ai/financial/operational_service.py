"""Operational Economics Service (Phase 6D).

Provides the high-level orchestration interface for:
- Transaction-level operational economics & cost completeness auditing
- Multi-line order operational economics without cost duplication
- Dimensional operational economics aggregations (SKU, Channel, Warehouse, cross-slices)
- Temporal operational economics analysis (daily, weekly, monthly, quarterly, yearly)
- Network-level portfolio operational economics summaries
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd

from commerce_ai.financial.schemas import (
    CostCompletenessReport,
    OperationalEconomicsConfig,
    OperationalEconomicsRecord,
    OperationalEconomicsSegment,
    OperationalEconomicsSummary,
    OrderOperationalEconomicsRecord,
    TimeGrain,
)
from commerce_ai.financial.operational_economics import (
    aggregate_operational_dimension,
    aggregate_operational_time_series,
    calculate_cost_completeness_report,
    calculate_operational_economics_record,
    compute_operational_economics_dataframe,
    compute_order_operational_economics_dataframe,
    resolve_transaction_operational_costs,
    summarize_operational_portfolio,
)


class OperationalEconomicsService:
    """Service orchestrating deterministic operational economics and cost completeness."""

    def __init__(self, config: Optional[OperationalEconomicsConfig] = None):
        self.config = config or OperationalEconomicsConfig()

    def _resolve_config(
        self,
        config: Optional[OperationalEconomicsConfig] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> OperationalEconomicsConfig:
        cfg = config or self.config
        if as_of_date is not None:
            as_of_str = as_of_date.isoformat() if hasattr(as_of_date, "isoformat") else str(as_of_date)
            cfg = cfg.model_copy(update={"as_of_date": as_of_str})
        return cfg

    def calculate_transaction_economics(
        self,
        sales: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[OperationalEconomicsConfig] = None,
        include_records: bool = False,
    ) -> Union[pd.DataFrame, List[OperationalEconomicsRecord]]:
        """Calculate line-level operational economics across all sales transactions."""
        cfg = self._resolve_config(config, as_of_date)
        calc_df = compute_operational_economics_dataframe(
            sales_df=sales,
            products_df=products,
            returns_df=returns,
            config=cfg,
        )

        if not include_records:
            return calc_df

        # Convert to Pydantic objects if explicitly requested
        records: List[OperationalEconomicsRecord] = []
        for _, row in calc_df.iterrows():
            record = calculate_operational_economics_record(
                row=row,
                unit_cost=row.get("unit_product_cost"),
                config=cfg,
            )
            records.append(record)
        return records

    def calculate_order_economics(
        self,
        sales: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[OperationalEconomicsConfig] = None,
    ) -> pd.DataFrame:
        """Calculate true order-level operational economics without multi-line cost duplication."""
        cfg = self._resolve_config(config, as_of_date)
        line_df = compute_operational_economics_dataframe(
            sales_df=sales,
            products_df=products,
            returns_df=returns,
            config=cfg,
        )
        return compute_order_operational_economics_dataframe(line_df, cfg)

    def calculate_portfolio_economics(
        self,
        sales: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[OperationalEconomicsConfig] = None,
    ) -> OperationalEconomicsSummary:
        """Calculate network-level portfolio operational economics and cost completeness report."""
        cfg = self._resolve_config(config, as_of_date)
        calc_df = compute_operational_economics_dataframe(
            sales_df=sales,
            products_df=products,
            returns_df=returns,
            config=cfg,
        )
        return summarize_operational_portfolio(calc_df, cfg)

    def calculate_segment_economics(
        self,
        sales: pd.DataFrame,
        dimension: str,
        products: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[OperationalEconomicsConfig] = None,
    ) -> List[OperationalEconomicsSegment]:
        """Aggregate operational economics along an analytical dimension."""
        cfg = self._resolve_config(config, as_of_date)
        calc_df = compute_operational_economics_dataframe(
            sales_df=sales,
            products_df=products,
            returns_df=returns,
            config=cfg,
        )
        return aggregate_operational_dimension(calc_df, dimension, cfg)

    def calculate_temporal_economics(
        self,
        sales: pd.DataFrame,
        time_grain: Union[str, TimeGrain] = TimeGrain.MONTHLY,
        products: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[OperationalEconomicsConfig] = None,
    ) -> List[OperationalEconomicsSegment]:
        """Aggregate operational economics along temporal buckets."""
        cfg = self._resolve_config(config, as_of_date)
        calc_df = compute_operational_economics_dataframe(
            sales_df=sales,
            products_df=products,
            returns_df=returns,
            config=cfg,
        )
        return aggregate_operational_time_series(calc_df, time_grain, cfg)

    def calculate_cost_completeness(
        self,
        sales: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[OperationalEconomicsConfig] = None,
    ) -> CostCompletenessReport:
        """Audit cost completeness across the sales portfolio."""
        summary = self.calculate_portfolio_economics(
            sales=sales,
            products=products,
            returns=returns,
            as_of_date=as_of_date,
            config=config,
        )
        return summary.cost_completeness_report
