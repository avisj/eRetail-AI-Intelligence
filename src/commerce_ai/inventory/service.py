"""Inventory Intelligence Service.

Coordinates the end-to-end Phase 4A inventory intelligence pipeline:
Demand Grid -> Stockout Masking -> ABC/XYZ -> Forecast Ingestion ->
Inventory Position -> Lead Time Profiling -> Safety Stock / ROP -> Inventory Risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from commerce_ai.analytics.demand import build_daily_demand
from commerce_ai.analytics.stockout import StockoutConfig, mask_stockout_demand
from commerce_ai.analytics.abc_xyz import ABCConfig, XYZConfig, calculate_abc_xyz_matrix
from commerce_ai.forecasting.service import ForecastService
from commerce_ai.forecasting.base import ForecastOutput

from commerce_ai.inventory.schemas import (
    InventoryPositionRecord,
    LeadTimeMetrics,
    SafetyStockResult,
    InventoryRiskRecord,
)
from commerce_ai.inventory.position import calculate_inventory_positions
from commerce_ai.inventory.lead_time import LeadTimeConfig, profile_sku_lead_times
from commerce_ai.inventory.safety_stock import (
    ServiceLevelPolicy,
    calculate_safety_stock_result,
)
from commerce_ai.inventory.risk import RiskThresholdConfig, assess_sku_inventory_risk


@dataclass
class InventoryAnalysisResult:
    """Consolidated outputs from the Inventory Intelligence pipeline."""

    positions: pd.DataFrame
    safety_stocks: pd.DataFrame
    risks: pd.DataFrame
    lead_times: pd.DataFrame
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "positions_count": len(self.positions),
            "safety_stocks_count": len(self.safety_stocks),
            "risks_count": len(self.risks),
        }


class InventoryService:
    """Unified service orchestrating inventory position, lead times, safety stocks, and risks."""

    def __init__(
        self,
        forecast_service: Optional[ForecastService] = None,
        service_level_policy: Optional[ServiceLevelPolicy] = None,
        lead_time_config: Optional[LeadTimeConfig] = None,
        risk_config: Optional[RiskThresholdConfig] = None,
        stockout_config: Optional[StockoutConfig] = None,
        use_forecast_for_rop: bool = True,
    ):
        self.forecast_service = forecast_service or ForecastService()
        self.service_level_policy = service_level_policy or ServiceLevelPolicy()
        self.lead_time_config = lead_time_config or LeadTimeConfig()
        self.risk_config = risk_config or RiskThresholdConfig()
        self.stockout_config = stockout_config or StockoutConfig()
        self.use_forecast_for_rop = use_forecast_for_rop

    def analyze_portfolio(
        self,
        sales: pd.DataFrame,
        inventory: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        suppliers: Optional[pd.DataFrame] = None,
        purchases: Optional[pd.DataFrame] = None,
        forecasts: Optional[pd.DataFrame] = None,
        forecast_bundle: Optional[pd.DataFrame] = None,
        horizon: int = 30,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        skus: Optional[List[str]] = None,
        warehouse_ids: Optional[List[str]] = None,
        use_forecast_for_rop: Optional[bool] = None,
    ) -> InventoryAnalysisResult:
        """Run the end-to-end Inventory Intelligence pipeline across a portfolio.

        Args:
            sales: Historical sales transactions.
            inventory: Inventory snapshot history.
            products: Product catalog master.
            suppliers: Supplier master.
            purchases: Inbound purchase orders (both delivered and open).
            forecasts: Optional pre-computed forecasts (ForecastRecord DataFrame).
                If None, forecasts are automatically generated via ForecastService.
            forecast_bundle: Optional alias for forecasts.
            horizon: Forecast evaluation horizon in days.
            as_of_date: Optional cutoff reference date for inventory evaluation.
            skus: Optional subset of SKUs to restrict to.
            warehouse_ids: Optional subset of warehouses to restrict to.
            use_forecast_for_rop: Optional override specifying whether to use
                mean forecast demand (True) or historical mean demand (False) for ROP.

        Returns:
            InventoryAnalysisResult containing positions, safety stocks, risks, and summary metrics.
        """
        # 1. Build regularized daily demand grid & apply stockout masking
        daily_demand = build_daily_demand(
            sales=sales,
            inventory=inventory,
            products=products,
            warehouses=None,
            skus=skus,
            warehouse_ids=warehouse_ids,
            end_date=as_of_date,
        )
        masked_demand = mask_stockout_demand(daily_demand, self.stockout_config)

        # 2. Portfolio ABC/XYZ Segmentation
        sku_segments, _, _ = calculate_abc_xyz_matrix(masked_demand)
        segment_map = sku_segments.set_index("sku_id")["abc_xyz_class"].to_dict() if not sku_segments.empty else {}

        # 3. Forecast Generation or Ingestion
        active_forecasts = forecasts if forecasts is not None else forecast_bundle
        if active_forecasts is None:
            forecast_df = self.forecast_service.forecast_batch(
                dataset=masked_demand,
                horizon=horizon,
            )
        else:
            forecast_df = active_forecasts.copy()

        # 4. Inventory Position Accounting
        positions_df = calculate_inventory_positions(
            snapshots=inventory,
            purchases=purchases,
            as_of_date=as_of_date,
        )
        if skus is not None:
            positions_df = positions_df[positions_df["sku_id"].isin(skus)]
        if warehouse_ids is not None:
            positions_df = positions_df[positions_df["warehouse_id"].isin(warehouse_ids)]

        # 5. Lead Time Profiling (point-in-time filtered using as_of_date)
        sku_lead_times = profile_sku_lead_times(
            purchases=purchases,
            products=products,
            suppliers=suppliers,
            config=self.lead_time_config,
            as_of_date=as_of_date,
        )
        lead_time_rows = [lt.to_dict() for lt in sku_lead_times.values()]
        lead_times_df = pd.DataFrame(lead_time_rows) if lead_time_rows else pd.DataFrame()

        # 6. Product master pricing/cost lookup
        prod_costs: Dict[str, float] = {}
        prod_prices: Dict[str, float] = {}
        if products is not None and not products.empty:
            if "sku_id" in products.columns and "unit_cost" in products.columns:
                prod_costs = products.set_index("sku_id")["unit_cost"].to_dict()
            if "sku_id" in products.columns and "selling_price" in products.columns:
                prod_prices = products.set_index("sku_id")["selling_price"].to_dict()

        # 7. Compute Safety Stock, ROP, and Inventory Risk per (SKU, Warehouse)
        safety_stock_records: List[SafetyStockResult] = []
        risk_records: List[InventoryRiskRecord] = []

        # Iterate over all identified (sku, warehouse) positions
        for _, pos_row in positions_df.iterrows():
            s_id = str(pos_row["sku_id"])
            w_id = str(pos_row["warehouse_id"])
            on_hand = int(pos_row["on_hand"])
            net_pos = int(pos_row["net_position"])

            # Resolve segment and target service level
            seg = segment_map.get(s_id, "BY")
            service_lvl = self.service_level_policy.get_service_level(sku_id=s_id, segment=seg)

            # Historical clean demand stats
            sku_wh_hist = masked_demand[
                (masked_demand["sku_id"] == s_id) & (masked_demand["warehouse_id"] == w_id)
            ]
            clean_hist = sku_wh_hist[sku_wh_hist["forecast_training_eligible"]]
            if not clean_hist.empty and "units_sold" in clean_hist.columns:
                hist_demand_mean = float(clean_hist["units_sold"].mean())
                hist_demand_std = float(clean_hist["units_sold"].std(ddof=1)) if len(clean_hist) > 1 else 0.0
            else:
                hist_demand_mean = float(sku_wh_hist["units_sold"].mean()) if not sku_wh_hist.empty else 0.0
                hist_demand_std = float(sku_wh_hist["units_sold"].std(ddof=1)) if len(sku_wh_hist) > 1 else 0.0

            # Lead time metrics
            lt_metric = sku_lead_times.get(s_id)
            mean_lt = lt_metric.mean_lead_time_days if lt_metric else self.lead_time_config.default_lead_time_days
            std_lt = lt_metric.std_lead_time_days if lt_metric else self.lead_time_config.default_lead_time_std_days

            # Costs
            u_cost = prod_costs.get(s_id, 0.0)
            s_price = prod_prices.get(s_id, u_cost * 1.5)

            # Extract forecasted demand sequence for this entity
            if (
                forecast_df is not None
                and not forecast_df.empty
                and "sku_id" in forecast_df.columns
                and "warehouse_id" in forecast_df.columns
            ):
                fc_subset = forecast_df[
                    (forecast_df["sku_id"] == s_id) & (forecast_df["warehouse_id"] == w_id)
                ]
            else:
                fc_subset = pd.DataFrame()

            if not fc_subset.empty and "forecast_units" in fc_subset.columns:
                fc_units = fc_subset["forecast_units"].values
                fc_dates = fc_subset["forecast_date"].astype(str).tolist() if "forecast_date" in fc_subset.columns else None
                valid_fc = (len(fc_units) > 0) and not np.isnan(fc_units).all() and (np.nanmean(fc_units) >= 0.0)
            else:
                fc_units = np.full(horizon, max(0.0, hist_demand_mean))
                fc_dates = None
                valid_fc = False

            # FIX 4: Resolve demand rate for ROP (forecast rate vs historical fallback)
            use_fc_rop = self.use_forecast_for_rop if use_forecast_for_rop is None else use_forecast_for_rop
            if use_fc_rop and valid_fc:
                d_rate = float(np.nanmean(fc_units))
                demand_source = "FORECAST"
            else:
                d_rate = max(0.0, hist_demand_mean)
                demand_source = "HISTORICAL_FALLBACK" if use_fc_rop else "HISTORICAL"

            # Compute Safety Stock and ROP
            ss_res = calculate_safety_stock_result(
                sku_id=s_id,
                warehouse_id=w_id,
                daily_demand_mean=max(0.0, d_rate),
                daily_demand_std=max(0.0, hist_demand_std),
                lead_time_mean_days=max(0.0, mean_lt),
                lead_time_std_days=max(0.0, std_lt),
                service_level=service_lvl,
                unit_cost=u_cost if u_cost > 0 else None,
                demand_rate_source=demand_source,
            )
            safety_stock_records.append(ss_res)

            # Assess Inventory Risk (passing historical clean demand std for hazard calculation)
            risk_rec = assess_sku_inventory_risk(
                sku_id=s_id,
                warehouse_id=w_id,
                on_hand=on_hand,
                net_position=net_pos,
                daily_forecast=fc_units,
                lead_time_mean_days=mean_lt,
                lead_time_std_days=std_lt,
                safety_stock=ss_res.safety_stock,
                reorder_point=ss_res.reorder_point,
                target_stock_level=ss_res.target_stock_level,
                unit_cost=u_cost,
                selling_price=s_price,
                forecast_dates=fc_dates,
                daily_demand_std=max(0.0, hist_demand_std),
                config=self.risk_config,
            )
            risk_records.append(risk_rec)

        # Convert to DataFrames
        ss_df = pd.DataFrame([r.to_dict() for r in safety_stock_records])
        risks_df = pd.DataFrame([r.to_dict() for r in risk_records])

        # Summary KPIs
        category_counts = risks_df["risk_category"].value_counts().to_dict() if not risks_df.empty else {}
        total_excess_capital = float(risks_df["excess_capital"].sum()) if not risks_df.empty else 0.0
        total_lost_revenue_risk = float(risks_df["lost_revenue_risk"].sum()) if not risks_df.empty else 0.0

        summary = {
            "total_series_analyzed": len(positions_df),
            "risk_categories": category_counts,
            "total_excess_capital_at_risk": round(total_excess_capital, 2),
            "total_stockout_revenue_at_risk": round(total_lost_revenue_risk, 2),
        }

        return InventoryAnalysisResult(
            positions=positions_df,
            safety_stocks=ss_df,
            risks=risks_df,
            lead_times=lead_times_df,
            summary=summary,
        )
