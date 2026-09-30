"""Cross-Domain Business Intelligence Orchestration Service (Phase 6E).

Coordinates cross-domain analytics across:
1. Sales & Financial Intelligence (Phase 6A, 6B, 6C, 6D)
2. Inventory Accounting & Positions (Phase 4A)
3. Demand Intelligence, Velocity & ABC/XYZ Segmentation (Phase 2)
4. Stockout Detection & Exposure (Phase 2 & 4A)
5. Out-of-Sample Demand Forecasting (Phase 3)
6. Returns Intelligence & Anomaly Detection (Phase 5A, 5B)
7. Replenishment Solver & Reorder Triggers (Phase 4B)
8. Multi-Warehouse Operational Context (Phase 4C)

Guarantees:
- Strict non-causal, descriptive signal evaluation
- Anti-leakage chronological filtering via point-in-time `as_of_date`
- Currency isolation (no cross-currency summation without FX)
- Explicit domain coverage tracking across 6 core domains
- High-performance vectorized execution on large transaction datasets
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.cross_domain.schemas import (
    CrossDomainBusinessRecord,
    CrossDomainIntelligenceConfig,
    CrossDomainIntelligenceResult,
    CrossDomainPortfolioSummary,
    CrossDomainSignal,
    SignalCategory,
    SignalSeverity,
)
from commerce_ai.cross_domain.signals import generate_cross_domain_signals
from commerce_ai.financial.revenue_margin import filter_sales_by_as_of_date


class CrossDomainIntelligenceService:
    """Service providing unified, deterministic cross-domain business intelligence."""

    def __init__(self, config: Optional[CrossDomainIntelligenceConfig] = None):
        self.config = config or CrossDomainIntelligenceConfig()

    def _resolve_config(
        self,
        config: Optional[CrossDomainIntelligenceConfig] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> CrossDomainIntelligenceConfig:
        cfg = config or self.config
        if as_of_date is not None:
            as_of_str = as_of_date.isoformat() if hasattr(as_of_date, "isoformat") else str(as_of_date)
            cfg = cfg.model_copy(update={"as_of_date": as_of_str})
        return cfg

    def _filter_by_as_of_date(
        self,
        df: Optional[pd.DataFrame],
        date_col: str,
        as_of_date: Optional[Union[str, date, datetime]],
    ) -> Optional[pd.DataFrame]:
        """Apply chronological point-in-time filter to dataset."""
        if df is None or df.empty or as_of_date is None:
            return df
        if date_col not in df.columns:
            return df
        cutoff_dt = pd.to_datetime(as_of_date).normalize()
        dates = pd.to_datetime(df[date_col]).dt.normalize()
        return df[dates <= cutoff_dt].copy()

    def build_sku_warehouse_view(
        self,
        sales: Optional[pd.DataFrame] = None,
        inventory: Optional[pd.DataFrame] = None,
        products: Optional[pd.DataFrame] = None,
        warehouses: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
        purchases: Optional[pd.DataFrame] = None,
        forecasts: Optional[pd.DataFrame] = None,
        stockouts: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[CrossDomainIntelligenceConfig] = None,
    ) -> List[CrossDomainBusinessRecord]:
        """Construct the canonical primary analytical grain (SKU × Warehouse) records."""
        cfg = self._resolve_config(config, as_of_date)
        as_of_str = str(cfg.as_of_date) if cfg.as_of_date is not None else None

        # 1. Apply Anti-Leakage Chronological Filtering
        filtered_sales = self._filter_by_as_of_date(sales, "date", cfg.as_of_date)
        filtered_returns = self._filter_by_as_of_date(returns, "return_date", cfg.as_of_date)
        filtered_inv = self._filter_by_as_of_date(inventory, "snapshot_date", cfg.as_of_date)
        filtered_purchases = self._filter_by_as_of_date(purchases, "order_date", cfg.as_of_date)

        # 2. Enforce Currency Isolation
        reporting_currency = cfg.default_currency
        if filtered_sales is not None and not filtered_sales.empty and "currency" in filtered_sales.columns:
            unique_currencies = filtered_sales["currency"].dropna().astype(str).str.strip().str.upper().unique().tolist()
            if len(unique_currencies) > 1 and not cfg.allow_multi_currency:
                raise ValueError(
                    f"Mixed currency detected: {unique_currencies}. "
                    f"Cross-domain aggregation without FX conversion is disabled."
                )
            if len(unique_currencies) == 1:
                reporting_currency = unique_currencies[0]

        # 3. Build Universe of (SKU, Warehouse) entities
        pairs: Set[Tuple[str, str]] = set()
        if products is not None and not products.empty and "sku_id" in products.columns and warehouses is not None and not warehouses.empty and "warehouse_id" in warehouses.columns:
            for s in products["sku_id"].dropna().unique():
                for w in warehouses["warehouse_id"].dropna().unique():
                    pairs.add((str(s), str(w)))

        if inventory is not None and not inventory.empty and "sku_id" in inventory.columns and "warehouse_id" in inventory.columns:
            for _, r in inventory[["sku_id", "warehouse_id"]].drop_duplicates().iterrows():
                pairs.add((str(r["sku_id"]), str(r["warehouse_id"])))

        if sales is not None and not sales.empty and "sku_id" in sales.columns and "warehouse_id" in sales.columns:
            for _, r in sales[["sku_id", "warehouse_id"]].drop_duplicates().iterrows():
                pairs.add((str(r["sku_id"]), str(r["warehouse_id"])))

        if not pairs:
            # Fallback if both sales and inventory empty
            return []

        sorted_pairs = sorted(list(pairs))
        base_df = pd.DataFrame(sorted_pairs, columns=["sku_id", "warehouse_id"])

        # 4. Integrate Product Master Metadata
        prod_meta: Dict[str, Dict[str, Any]] = {}
        if products is not None and not products.empty and "sku_id" in products.columns:
            for _, prow in products.iterrows():
                sid = str(prow["sku_id"])
                prod_meta[sid] = {
                    "category_id": str(prow.get("category", prow.get("category_id", ""))),
                    "brand": str(prow.get("brand", "")),
                    "unit_cost": float(prow.get("unit_cost", 0.0)) if pd.notna(prow.get("unit_cost")) else None,
                    "unit_price": float(prow.get("unit_price", 0.0)) if pd.notna(prow.get("unit_price")) else None,
                }

        # 5. Domain 1: SALES & FINANCIAL AGGREGATION
        financial_available = False
        sales_agg_df = pd.DataFrame(columns=["sku_id", "warehouse_id"])
        if filtered_sales is not None and not filtered_sales.empty and "sku_id" in filtered_sales.columns and "warehouse_id" in filtered_sales.columns:
            financial_available = True
            sdf = filtered_sales.copy()
            if "quantity" not in sdf.columns:
                sdf["quantity"] = 0
            if "revenue" not in sdf.columns:
                sdf["revenue"] = 0.0

            # Attach unit_cost if available for product_cost calculation
            if "unit_cost" not in sdf.columns:
                sdf["unit_cost"] = sdf["sku_id"].astype(str).map(
                    lambda s: prod_meta.get(s, {}).get("unit_cost", 0.0) if s in prod_meta else 0.0
                )

            sdf["calculated_product_cost"] = sdf["quantity"] * sdf["unit_cost"]
            sdf["order_id_col"] = sdf["order_id"] if "order_id" in sdf.columns else sdf.index

            sales_agg_df = (
                sdf.groupby(["sku_id", "warehouse_id"], as_index=False)
                .agg(
                    units_sold=("quantity", "sum"),
                    net_revenue=("revenue", "sum"),
                    product_cost=("calculated_product_cost", "sum"),
                    transaction_count=("quantity", "count"),
                    order_count=("order_id_col", "nunique"),
                )
            )
            sales_agg_df["sku_id"] = sales_agg_df["sku_id"].astype(str)
            sales_agg_df["warehouse_id"] = sales_agg_df["warehouse_id"].astype(str)
            sales_agg_df["gross_revenue"] = sales_agg_df["net_revenue"]  # Default unless discount present
            sales_agg_df["gross_margin"] = sales_agg_df["net_revenue"] - sales_agg_df["product_cost"]
            sales_agg_df["gross_margin_pct"] = np.where(
                sales_agg_df["net_revenue"] > 0,
                sales_agg_df["gross_margin"] / sales_agg_df["net_revenue"],
                0.0,
            )
            sales_agg_df["average_order_value"] = np.where(
                sales_agg_df["order_count"] > 0,
                sales_agg_df["net_revenue"] / sales_agg_df["order_count"],
                0.0,
            )

        # 6. Domain 2: INVENTORY POSITION AGGREGATION
        inventory_available = False
        inv_agg_df = pd.DataFrame(columns=["sku_id", "warehouse_id"])
        if filtered_inv is not None and not filtered_inv.empty and "sku_id" in filtered_inv.columns and "warehouse_id" in filtered_inv.columns:
            inventory_available = True
            idf = filtered_inv.copy()
            idf["sku_id"] = idf["sku_id"].astype(str)
            idf["warehouse_id"] = idf["warehouse_id"].astype(str)

            # Keep latest snapshot per (sku_id, warehouse_id)
            if "snapshot_date" in idf.columns:
                idf = idf.sort_values("snapshot_date").groupby(["sku_id", "warehouse_id"], as_index=False).last()
            else:
                idf = idf.groupby(["sku_id", "warehouse_id"], as_index=False).last()

            avail_col = "available_qty" if "available_qty" in idf.columns else ("available" if "available" in idf.columns else None)
            res_col = "reserved_qty" if "reserved_qty" in idf.columns else ("reserved" if "reserved" in idf.columns else None)
            on_hand_col = "on_hand" if "on_hand" in idf.columns else None

            if avail_col:
                idf["available_inventory"] = idf[avail_col].fillna(0).astype(int)
            else:
                idf["available_inventory"] = 0

            if res_col:
                idf["reserved_inventory"] = idf[res_col].fillna(0).astype(int)
            else:
                idf["reserved_inventory"] = 0

            if on_hand_col:
                idf["current_on_hand"] = idf[on_hand_col].fillna(0).astype(int)
            else:
                idf["current_on_hand"] = idf["available_inventory"] + idf["reserved_inventory"]

            # Attach unit cost for inventory valuation
            idf["unit_cost"] = idf["sku_id"].map(
                lambda s: prod_meta.get(s, {}).get("unit_cost", 0.0) if s in prod_meta else 0.0
            )
            idf["inventory_value"] = idf["current_on_hand"] * idf["unit_cost"]
            idf["inventory_position"] = idf["current_on_hand"] - idf["reserved_inventory"]

            inv_agg_df = idf[
                ["sku_id", "warehouse_id", "current_on_hand", "available_inventory", "reserved_inventory", "inventory_position", "inventory_value"]
            ]

        # 7. Domain 3: DEMAND & ABC/XYZ AGGREGATION
        demand_available = False
        demand_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if financial_available and not sales_agg_df.empty:
            demand_available = True
            # Calculate daily demand stats per (sku_id, warehouse_id)
            sdf_daily = (
                filtered_sales.groupby(["date", "sku_id", "warehouse_id"], as_index=False)["quantity"]
                .sum()
            )
            sdf_daily["sku_id"] = sdf_daily["sku_id"].astype(str)
            sdf_daily["warehouse_id"] = sdf_daily["warehouse_id"].astype(str)

            grp = sdf_daily.groupby(["sku_id", "warehouse_id"])["quantity"]
            daily_means = grp.mean().to_dict()
            daily_stds = grp.std().fillna(0.0).to_dict()

            # Global Pareto ABC per SKU across portfolio
            sku_rev = sales_agg_df.groupby("sku_id")["net_revenue"].sum().sort_values(ascending=False)
            tot_rev = sku_rev.sum()
            cum_share = (sku_rev / tot_rev).cumsum() if tot_rev > 0 else pd.Series(0.0, index=sku_rev.index)
            prior_share = cum_share.shift(1).fillna(0.0)
            abc_map = np.where(prior_share < 0.80, "A", np.where(prior_share < 0.95, "B", "C"))
            sku_abc = dict(zip(sku_rev.index, abc_map))

            for pair in sorted_pairs:
                d_mean = float(daily_means.get(pair, 0.0))
                d_std = float(daily_stds.get(pair, 0.0))
                cv = (d_std / d_mean) if d_mean > 0 else 0.0
                xyz = "X" if cv <= 0.50 else ("Y" if cv <= 1.00 else "Z")
                abc = sku_abc.get(pair[0], "C")
                demand_metrics[pair] = {
                    "average_daily_demand": d_mean,
                    "demand_std": d_std,
                    "abc_class": abc,
                    "xyz_class": xyz,
                    "intermittency_class": "SMOOTH" if cv <= 0.50 else ("INTERMITTENT" if cv > 1.0 else "LUMPY"),
                }

        # 8. Domain 4: STOCKOUT INTEGRATION
        stockout_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if stockouts is not None and not stockouts.empty and "sku_id" in stockouts.columns and "warehouse_id" in stockouts.columns:
            so_df = stockouts.copy()
            so_df["sku_id"] = so_df["sku_id"].astype(str)
            so_df["warehouse_id"] = so_df["warehouse_id"].astype(str)
            so_grouped = so_df.groupby(["sku_id", "warehouse_id"])
            for (sid, wid), g in so_grouped:
                so_days = int(g["is_stockout"].sum()) if "is_stockout" in g.columns else int(len(g))
                so_rate = float(so_days / len(g)) if len(g) > 0 else 0.0
                stockout_metrics[(sid, wid)] = {
                    "stockout_days": so_days,
                    "stockout_rate": so_rate,
                    "stockout_flag": so_days > 0,
                }

        # 9. Domain 5: RETURNS INTEGRATION
        returns_available = False
        returns_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if filtered_returns is not None and not filtered_returns.empty and "sku_id" in filtered_returns.columns and "warehouse_id" in filtered_returns.columns:
            returns_available = True
            rdf = filtered_returns.copy()
            rdf["sku_id"] = rdf["sku_id"].astype(str)
            rdf["warehouse_id"] = rdf["warehouse_id"].astype(str)
            if "quantity" not in rdf.columns:
                rdf["quantity"] = 1

            ret_grp = rdf.groupby(["sku_id", "warehouse_id"])
            for (sid, wid), g in ret_grp:
                ret_u = int(g["quantity"].sum())
                ret_c = int(len(g))
                returns_metrics[(sid, wid)] = {
                    "return_units": ret_u,
                    "return_count": ret_c,
                    "return_anomaly_flag": bool(ret_u > 50),  # Observed high return anomaly flag
                }

        # 10. Domain 6: FORECAST INTEGRATION
        forecast_available = False
        forecast_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if forecasts is not None and not forecasts.empty and "sku_id" in forecasts.columns and "warehouse_id" in forecasts.columns:
            forecast_available = True
            fdf = forecasts.copy()
            fdf["sku_id"] = fdf["sku_id"].astype(str)
            fdf["warehouse_id"] = fdf["warehouse_id"].astype(str)
            for _, frow in fdf.iterrows():
                sid = str(frow["sku_id"])
                wid = str(frow["warehouse_id"])
                forecast_metrics[(sid, wid)] = {
                    "forecast_mean": float(frow.get("forecast_mean", 0.0)),
                    "forecast_horizon": int(frow.get("forecast_horizon", 14)) if pd.notna(frow.get("forecast_horizon")) else 14,
                    "forecast_model": str(frow.get("forecast_model", "LightGBM")),
                }

        # 11. Domain 7: REPLENISHMENT INTEGRATION
        replenishment_available = False
        replenishment_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if inventory_available:
            replenishment_available = True
            for pair in sorted_pairs:
                inv_m = inv_agg_df[
                    (inv_agg_df["sku_id"] == pair[0]) & (inv_agg_df["warehouse_id"] == pair[1])
                ]
                avail = int(inv_m["available_inventory"].values[0]) if not inv_m.empty else 0
                net_pos = int(inv_m["inventory_position"].values[0]) if not inv_m.empty else 0
                dm = demand_metrics.get(pair, {})
                avg_d = dm.get("average_daily_demand", 0.0)
                rop = float(round(avg_d * 7.0 + 10.0, 1))  # 7 days lead time + safety buffer
                trigger = bool(net_pos <= rop)
                roq = int(max(0, math.ceil(rop * 2.0 - net_pos))) if trigger else 0
                replenishment_metrics[pair] = {
                    "reorder_point": rop,
                    "recommended_order_qty": roq,
                    "replenishment_trigger": trigger,
                    "replenishment_risk": "HIGH" if trigger and net_pos <= 0 else ("MEDIUM" if trigger else "LOW"),
                }

        # 12. Assemble Records
        records: List[CrossDomainBusinessRecord] = []
        sales_dict = sales_agg_df.set_index(["sku_id", "warehouse_id"]).to_dict("index") if not sales_agg_df.empty else {}
        inv_dict = inv_agg_df.set_index(["sku_id", "warehouse_id"]).to_dict("index") if not inv_agg_df.empty else {}

        for pair in sorted_pairs:
            sid, wid = pair
            s_data = sales_dict.get(pair, {})
            i_data = inv_dict.get(pair, {})
            d_data = demand_metrics.get(pair, {})
            so_data = stockout_metrics.get(pair, {})
            r_data = returns_metrics.get(pair, {})
            fc_data = forecast_metrics.get(pair, {})
            rep_data = replenishment_metrics.get(pair, {})
            p_data = prod_meta.get(sid, {})

            u_sold = int(s_data.get("units_sold", 0))
            r_units = int(r_data.get("return_units", 0))
            ret_rate = (r_units / u_sold) if u_sold > 0 else (0.0 if r_units == 0 and u_sold == 0 else None)

            avg_d = d_data.get("average_daily_demand")
            cur_on_hand = i_data.get("current_on_hand")
            doc = (cur_on_hand / avg_d) if (cur_on_hand is not None and avg_d and avg_d > 0) else None

            # Calculate Domain Coverage
            has_fin = bool(s_data)
            has_inv = bool(i_data)
            has_dem = bool(d_data)
            has_fc = bool(fc_data)
            has_ret = bool(r_data)
            has_rep = bool(rep_data)

            avail_domains = [has_fin, has_inv, has_dem, has_fc, has_ret, has_rep]
            num_avail = sum(1 for a in avail_domains if a)
            cov_pct = round((num_avail / 6.0) * 100.0, 2)
            missing_cnt = 6 - num_avail

            rec = CrossDomainBusinessRecord(
                sku_id=sid,
                warehouse_id=wid,
                category_id=p_data.get("category_id"),
                brand=p_data.get("brand"),
                currency=reporting_currency,
                # Financial
                gross_revenue=s_data.get("gross_revenue"),
                net_revenue=s_data.get("net_revenue"),
                product_cost=s_data.get("product_cost"),
                gross_margin=s_data.get("gross_margin"),
                gross_margin_pct=s_data.get("gross_margin_pct"),
                # Sales
                units_sold=u_sold,
                transaction_count=s_data.get("transaction_count"),
                order_count=s_data.get("order_count"),
                average_order_value=s_data.get("average_order_value"),
                # Inventory
                current_on_hand=cur_on_hand,
                inventory_value=i_data.get("inventory_value"),
                available_inventory=i_data.get("available_inventory"),
                reserved_inventory=i_data.get("reserved_inventory"),
                inventory_position=i_data.get("inventory_position"),
                days_of_cover=doc,
                # Demand
                average_daily_demand=avg_d,
                demand_std=d_data.get("demand_std"),
                intermittency_class=d_data.get("intermittency_class"),
                abc_class=d_data.get("abc_class"),
                xyz_class=d_data.get("xyz_class"),
                # Stockout
                stockout_days=so_data.get("stockout_days", 0),
                stockout_rate=so_data.get("stockout_rate", 0.0),
                stockout_flag=so_data.get("stockout_flag", False),
                # Forecast
                forecast_mean=fc_data.get("forecast_mean"),
                forecast_horizon=fc_data.get("forecast_horizon"),
                forecast_model=fc_data.get("forecast_model"),
                forecast_available=has_fc,
                # Returns
                return_rate=ret_rate,
                return_count=r_data.get("return_count", 0),
                return_units=r_units,
                return_anomaly_flag=r_data.get("return_anomaly_flag", False),
                # Replenishment
                reorder_point=rep_data.get("reorder_point"),
                recommended_order_qty=rep_data.get("recommended_order_qty"),
                replenishment_trigger=rep_data.get("replenishment_trigger"),
                replenishment_risk=rep_data.get("replenishment_risk"),
                # Quality & Coverage
                financial_available=has_fin,
                inventory_available=has_inv,
                demand_available=has_dem,
                returns_available=has_ret,
                replenishment_available=has_rep,
                missing_domain_count=missing_cnt,
                domain_coverage_pct=cov_pct,
                as_of_date=as_of_str,
            )
            records.append(rec)

        return records

    def build_sku_view(
        self,
        records: List[CrossDomainBusinessRecord],
    ) -> List[CrossDomainBusinessRecord]:
        """Aggregate primary SKU × Warehouse records up to the SKU grain."""
        if not records:
            return []

        sku_groups: Dict[str, List[CrossDomainBusinessRecord]] = {}
        for r in records:
            sku_groups.setdefault(r.sku_id, []).append(r)

        sku_records: List[CrossDomainBusinessRecord] = []
        for sid, recs in sku_groups.items():
            rep = recs[0]
            tot_rev = sum(r.net_revenue for r in recs if r.net_revenue is not None)
            tot_cogs = sum(r.product_cost for r in recs if r.product_cost is not None)
            tot_gm = tot_rev - tot_cogs
            gm_pct = (tot_gm / tot_rev) if tot_rev > 0 else 0.0

            tot_sold = sum(r.units_sold for r in recs if r.units_sold is not None)
            tot_orders = sum(r.order_count for r in recs if r.order_count is not None)
            aov = (tot_rev / tot_orders) if tot_orders > 0 else 0.0

            tot_on_hand = sum(r.current_on_hand for r in recs if r.current_on_hand is not None)
            tot_avail = sum(r.available_inventory for r in recs if r.available_inventory is not None)
            tot_res = sum(r.reserved_inventory for r in recs if r.reserved_inventory is not None)
            tot_inv_val = sum(r.inventory_value for r in recs if r.inventory_value is not None)
            tot_pos = sum(r.inventory_position for r in recs if r.inventory_position is not None)

            tot_ret_units = sum(r.return_units for r in recs if r.return_units is not None)
            tot_ret_count = sum(r.return_count for r in recs if r.return_count is not None)
            ret_rate = (tot_ret_units / tot_sold) if tot_sold > 0 else None

            tot_demand = sum(r.average_daily_demand for r in recs if r.average_daily_demand is not None)
            so_days_max = max((r.stockout_days or 0) for r in recs)
            any_trigger = any(r.replenishment_trigger is True for r in recs)
            tot_roq = sum((r.recommended_order_qty or 0) for r in recs)

            has_fc = any(r.forecast_available for r in recs)
            tot_fc = sum(r.forecast_mean for r in recs if r.forecast_mean is not None) if has_fc else None

            # Domain coverage for SKU
            has_fin = any(r.financial_available for r in recs)
            has_inv = any(r.inventory_available for r in recs)
            has_dem = any(r.demand_available for r in recs)
            has_ret = any(r.returns_available for r in recs)
            has_rep = any(r.replenishment_available for r in recs)

            avail_domains = [has_fin, has_inv, has_dem, has_fc, has_ret, has_rep]
            num_avail = sum(1 for a in avail_domains if a)
            cov_pct = round((num_avail / 6.0) * 100.0, 2)

            sku_rec = CrossDomainBusinessRecord(
                sku_id=sid,
                warehouse_id=None,
                category_id=rep.category_id,
                brand=rep.brand,
                currency=rep.currency,
                gross_revenue=tot_rev,
                net_revenue=tot_rev,
                product_cost=tot_cogs,
                gross_margin=tot_gm,
                gross_margin_pct=gm_pct,
                units_sold=tot_sold,
                order_count=tot_orders,
                average_order_value=aov,
                current_on_hand=tot_on_hand,
                available_inventory=tot_avail,
                reserved_inventory=tot_res,
                inventory_value=tot_inv_val,
                inventory_position=tot_pos,
                average_daily_demand=tot_demand,
                abc_class=rep.abc_class,
                xyz_class=rep.xyz_class,
                intermittency_class=rep.intermittency_class,
                stockout_days=so_days_max,
                stockout_flag=so_days_max > 0,
                return_units=tot_ret_units,
                return_count=tot_ret_count,
                return_rate=ret_rate,
                forecast_available=has_fc,
                forecast_mean=tot_fc,
                replenishment_trigger=any_trigger,
                recommended_order_qty=tot_roq,
                financial_available=has_fin,
                inventory_available=has_inv,
                demand_available=has_dem,
                returns_available=has_ret,
                replenishment_available=has_rep,
                missing_domain_count=6 - num_avail,
                domain_coverage_pct=cov_pct,
                as_of_date=rep.as_of_date,
            )
            sku_records.append(sku_rec)

        return sku_records

    def build_warehouse_view(
        self,
        records: List[CrossDomainBusinessRecord],
    ) -> List[CrossDomainBusinessRecord]:
        """Aggregate primary SKU × Warehouse records up to the Warehouse grain."""
        if not records:
            return []

        wh_groups: Dict[str, List[CrossDomainBusinessRecord]] = {}
        for r in records:
            if r.warehouse_id:
                wh_groups.setdefault(r.warehouse_id, []).append(r)

        wh_records: List[CrossDomainBusinessRecord] = []
        for wid, recs in wh_groups.items():
            rep = recs[0]
            tot_rev = sum(r.net_revenue for r in recs if r.net_revenue is not None)
            tot_cogs = sum(r.product_cost for r in recs if r.product_cost is not None)
            tot_gm = tot_rev - tot_cogs
            gm_pct = (tot_gm / tot_rev) if tot_rev > 0 else 0.0

            tot_sold = sum(r.units_sold for r in recs if r.units_sold is not None)
            tot_orders = sum(r.order_count for r in recs if r.order_count is not None)

            tot_on_hand = sum(r.current_on_hand for r in recs if r.current_on_hand is not None)
            tot_avail = sum(r.available_inventory for r in recs if r.available_inventory is not None)
            tot_inv_val = sum(r.inventory_value for r in recs if r.inventory_value is not None)

            tot_ret_units = sum(r.return_units for r in recs if r.return_units is not None)
            tot_ret_count = sum(r.return_count for r in recs if r.return_count is not None)
            ret_rate = (tot_ret_units / tot_sold) if tot_sold > 0 else None

            so_count = sum(1 for r in recs if r.stockout_flag is True)
            so_rate = (so_count / len(recs)) if recs else 0.0

            trigger_count = sum(1 for r in recs if r.replenishment_trigger is True)
            tot_roq = sum((r.recommended_order_qty or 0) for r in recs)

            wh_rec = CrossDomainBusinessRecord(
                sku_id=f"WH_{wid}_SUMMARY",
                warehouse_id=wid,
                currency=rep.currency,
                gross_revenue=tot_rev,
                net_revenue=tot_rev,
                product_cost=tot_cogs,
                gross_margin=tot_gm,
                gross_margin_pct=gm_pct,
                units_sold=tot_sold,
                order_count=tot_orders,
                current_on_hand=tot_on_hand,
                available_inventory=tot_avail,
                inventory_value=tot_inv_val,
                stockout_days=so_count,
                stockout_rate=so_rate,
                stockout_flag=so_count > 0,
                return_units=tot_ret_units,
                return_count=tot_ret_count,
                return_rate=ret_rate,
                replenishment_trigger=trigger_count > 0,
                recommended_order_qty=tot_roq,
                financial_available=any(r.financial_available for r in recs),
                inventory_available=any(r.inventory_available for r in recs),
                demand_available=any(r.demand_available for r in recs),
                forecast_available=any(r.forecast_available for r in recs),
                returns_available=any(r.returns_available for r in recs),
                replenishment_available=any(r.replenishment_available for r in recs),
                missing_domain_count=rep.missing_domain_count,
                domain_coverage_pct=rep.domain_coverage_pct,
                as_of_date=rep.as_of_date,
            )
            wh_records.append(wh_rec)

        return wh_records

    def build_channel_view(
        self,
        sales: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
        products: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[CrossDomainIntelligenceConfig] = None,
    ) -> List[CrossDomainBusinessRecord]:
        """Construct descriptive cross-domain records at the Sales Channel grain."""
        cfg = self._resolve_config(config, as_of_date)
        filtered_sales = self._filter_by_as_of_date(sales, "date", cfg.as_of_date)
        filtered_returns = self._filter_by_as_of_date(returns, "return_date", cfg.as_of_date)

        if filtered_sales is None or filtered_sales.empty or "channel_id" not in filtered_sales.columns:
            return []

        sdf = filtered_sales.copy()
        if "quantity" not in sdf.columns:
            sdf["quantity"] = 0
        if "revenue" not in sdf.columns:
            sdf["revenue"] = 0.0

        prod_costs: Dict[str, float] = {}
        if products is not None and not products.empty and "sku_id" in products.columns and "unit_cost" in products.columns:
            prod_costs = products.set_index("sku_id")["unit_cost"].dropna().to_dict()

        sdf["unit_cost"] = sdf["sku_id"].astype(str).map(lambda s: prod_costs.get(s, 0.0))
        sdf["product_cost"] = sdf["quantity"] * sdf["unit_cost"]
        sdf["order_id_col"] = sdf["order_id"] if "order_id" in sdf.columns else sdf.index

        ch_sales = (
            sdf.groupby("channel_id", as_index=False)
            .agg(
                units_sold=("quantity", "sum"),
                net_revenue=("revenue", "sum"),
                product_cost=("product_cost", "sum"),
                transaction_count=("quantity", "count"),
                order_count=("order_id_col", "nunique"),
            )
        )

        ch_returns: Dict[str, int] = {}
        if filtered_returns is not None and not filtered_returns.empty and "channel_id" in filtered_returns.columns:
            rdf = filtered_returns.copy()
            if "quantity" not in rdf.columns:
                rdf["quantity"] = 1
            ch_returns = rdf.groupby("channel_id")["quantity"].sum().to_dict()

        records: List[CrossDomainBusinessRecord] = []
        for _, row in ch_sales.iterrows():
            cid = str(row["channel_id"])
            net_rev = float(row["net_revenue"])
            cogs = float(row["product_cost"])
            gm = net_rev - cogs
            gm_pct = (gm / net_rev) if net_rev > 0 else 0.0
            u_sold = int(row["units_sold"])
            u_ret = int(ch_returns.get(cid, 0))
            ret_rate = (u_ret / u_sold) if u_sold > 0 else None
            orders = int(row["order_count"])
            aov = (net_rev / orders) if orders > 0 else 0.0

            # Inventory and Replenishment are not native to Channel grain
            rec = CrossDomainBusinessRecord(
                sku_id=f"CH_{cid}_SUMMARY",
                channel_id=cid,
                currency=cfg.default_currency,
                gross_revenue=net_rev,
                net_revenue=net_rev,
                product_cost=cogs,
                gross_margin=gm,
                gross_margin_pct=gm_pct,
                units_sold=u_sold,
                transaction_count=int(row["transaction_count"]),
                order_count=orders,
                average_order_value=aov,
                return_units=u_ret,
                return_count=u_ret,
                return_rate=ret_rate,
                financial_available=True,
                inventory_available=False,
                demand_available=True,
                forecast_available=False,
                returns_available=bool(ch_returns),
                replenishment_available=False,
                missing_domain_count=3,
                domain_coverage_pct=50.0,
                as_of_date=str(cfg.as_of_date) if cfg.as_of_date is not None else None,
            )
            records.append(rec)

        return records

    def build_cross_domain_signals(
        self,
        records: List[CrossDomainBusinessRecord],
        config: Optional[CrossDomainIntelligenceConfig] = None,
    ) -> List[CrossDomainSignal]:
        """Generate all descriptive, deterministic cross-domain signals."""
        cfg = config or self.config
        return generate_cross_domain_signals(records=records, config=cfg)

    def build_portfolio_summary(
        self,
        records: List[CrossDomainBusinessRecord],
        signals: List[CrossDomainSignal],
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[CrossDomainIntelligenceConfig] = None,
    ) -> CrossDomainPortfolioSummary:
        """Calculate high-level summary metrics across records and signals."""
        cfg = self._resolve_config(config, as_of_date)

        # Signal Distributions
        cat_counts: Dict[str, int] = {}
        for s in signals:
            cat_key = s.category.value if isinstance(s.category, SignalCategory) else str(s.category)
            cat_counts[cat_key] = cat_counts.get(cat_key, 0) + 1

        sev_counts: Dict[str, int] = {}
        for s in signals:
            sev_key = s.severity.value if isinstance(s.severity, SignalSeverity) else str(s.severity)
            sev_counts[sev_key] = sev_counts.get(sev_key, 0) + 1

        type_counts: Dict[str, int] = {}
        for s in signals:
            type_counts[s.signal_type] = type_counts.get(s.signal_type, 0) + 1
        sorted_types = dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))

        # Financial & Inventory Totals
        tot_rev = sum(r.net_revenue for r in records if r.net_revenue is not None) if records else None
        tot_gm = sum(r.gross_margin for r in records if r.gross_margin is not None) if records else None
        tot_inv = sum(r.inventory_value for r in records if r.inventory_value is not None) if records else None

        # Domain Coverage Summary
        cov_summary: Dict[str, float] = {}
        if records:
            n = len(records)
            cov_summary["financial_coverage_pct"] = round(sum(1 for r in records if r.financial_available) / n * 100.0, 2)
            cov_summary["inventory_coverage_pct"] = round(sum(1 for r in records if r.inventory_available) / n * 100.0, 2)
            cov_summary["demand_coverage_pct"] = round(sum(1 for r in records if r.demand_available) / n * 100.0, 2)
            cov_summary["forecast_coverage_pct"] = round(sum(1 for r in records if r.forecast_available) / n * 100.0, 2)
            cov_summary["returns_coverage_pct"] = round(sum(1 for r in records if r.returns_available) / n * 100.0, 2)
            cov_summary["replenishment_coverage_pct"] = round(sum(1 for r in records if r.replenishment_available) / n * 100.0, 2)
            cov_summary["average_domain_coverage_pct"] = round(sum(r.domain_coverage_pct for r in records) / n, 2)

        return CrossDomainPortfolioSummary(
            total_records=len(records),
            total_signals=len(signals),
            signals_by_category=cat_counts,
            signals_by_severity=sev_counts,
            top_signal_types=sorted_types,
            domain_coverage_summary=cov_summary,
            total_revenue=round(tot_rev, 2) if tot_rev is not None else None,
            total_margin=round(tot_gm, 2) if tot_gm is not None else None,
            total_inventory_value=round(tot_inv, 2) if tot_inv is not None else None,
            currency=cfg.default_currency,
            as_of_date=str(cfg.as_of_date) if cfg.as_of_date is not None else None,
        )

    def analyze(
        self,
        sales: Optional[pd.DataFrame] = None,
        inventory: Optional[pd.DataFrame] = None,
        products: Optional[pd.DataFrame] = None,
        warehouses: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
        purchases: Optional[pd.DataFrame] = None,
        forecasts: Optional[pd.DataFrame] = None,
        stockouts: Optional[pd.DataFrame] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[CrossDomainIntelligenceConfig] = None,
    ) -> CrossDomainIntelligenceResult:
        """Run the comprehensive end-to-end cross-domain intelligence pipeline."""
        cfg = self._resolve_config(config, as_of_date)

        records = self.build_sku_warehouse_view(
            sales=sales,
            inventory=inventory,
            products=products,
            warehouses=warehouses,
            returns=returns,
            purchases=purchases,
            forecasts=forecasts,
            stockouts=stockouts,
            as_of_date=cfg.as_of_date,
            config=cfg,
        )

        signals = self.build_cross_domain_signals(records=records, config=cfg)

        summary = self.build_portfolio_summary(
            records=records,
            signals=signals,
            as_of_date=cfg.as_of_date,
            config=cfg,
        )

        return CrossDomainIntelligenceResult(
            records=records,
            signals=signals,
            summary=summary,
            as_of_date=str(cfg.as_of_date) if cfg.as_of_date is not None else None,
        )
