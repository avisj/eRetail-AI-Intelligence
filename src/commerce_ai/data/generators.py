"""Realistic Synthetic Ecommerce Data Generator.

Generates realistic, coupled ecommerce operational data:
- Catalogs with categories: Electronics, Fashion, Beauty, Home, Sports, Grocery
- Velocity profiles: High velocity (Pareto top 15%), Medium (30%), Slow (40%), Intermittent (10%), New launch (5%)
- Realistic Seasonality: weekly cycles, monthly salary spikes, Q4 holiday surges
- Controlled Trends: growing products, declining products, stable baselines
- Promotional Spikes: marketing campaigns driving 2x-5x lift
- Closed-Loop Inventory Simulation:
    Sales decrement stock -> Low stock triggers Purchase Orders -> Supplier lead times -> Inbound dock receipt increments stock
- Realistic Business Edge Cases:
    Stockouts (unfulfilled demand), Overstock (high MOQ on slow movers), Dead Stock (aging products)
- Linked Returns: realistic category return rates (Fashion 22%, Electronics 10%, Grocery 1%) linked to real orders
- Controlled Anomalies: configurable injection for testing validator & data quality engines
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    """Configuration for synthetic ecommerce data generation."""

    number_of_skus: int = 500
    number_of_warehouses: int = 5
    number_of_channels: int = 4
    number_of_suppliers: int = 20
    historical_days: int = 730
    start_date: date = date(2024, 1, 1)
    random_seed: int = 42
    include_controlled_anomalies: bool = False


class SyntheticDataGenerator:
    """Generates coupled, realistic ecommerce operational datasets."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all 8 canonical ecommerce datasets with realistic relational coupling."""
        # 1. Channels master
        channels_df = self._generate_channels()

        # 2. Warehouses master
        warehouses_df = self._generate_warehouses()

        # 3. Suppliers master
        suppliers_df = self._generate_suppliers()

        # 4. Products master
        products_df = self._generate_products(suppliers_df)

        # 5. Closed-loop simulation: Sales, Inventory, Purchases, Returns
        sales_df, inventory_df, purchases_df, returns_df = self._simulate_commerce_dynamics(
            products_df=products_df,
            warehouses_df=warehouses_df,
            channels_df=channels_df,
            suppliers_df=suppliers_df,
        )

        datasets = {
            "channels": channels_df,
            "warehouses": warehouses_df,
            "suppliers": suppliers_df,
            "products": products_df,
            "purchases": purchases_df,
            "sales": sales_df,
            "returns": returns_df,
            "inventory": inventory_df,
        }

        if self.config.include_controlled_anomalies:
            self._inject_controlled_anomalies(datasets)

        return datasets

    def generate_and_save(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        """Generate datasets and persist them as CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        datasets = self.generate_all()
        for name, df in datasets.items():
            file_path = output_dir / f"{name}.csv"
            df.to_csv(file_path, index=False)

        return datasets

    # -------------------------------------------------------------------------
    # Master Datasets Generation
    # -------------------------------------------------------------------------

    def _generate_channels(self) -> pd.DataFrame:
        channels_pool = [
            ("CH_AMZ", "Amazon Marketplace", "Marketplace"),
            ("CH_DIR", "Direct Brand Webstore", "Direct"),
            ("CH_FLK", "Flipkart Marketplace", "Marketplace"),
            ("CH_MYN", "Myntra Fashion Hub", "Marketplace"),
            ("CH_RET", "Flagship Retail Store", "Retail Store"),
            ("CH_B2B", "Wholesale & Corporate", "B2B"),
        ]
        n = min(self.config.number_of_channels, len(channels_pool))
        selected = channels_pool[:n]
        return pd.DataFrame(
            selected,
            columns=["channel_id", "channel_name", "channel_type"],
        )

    def _generate_warehouses(self) -> pd.DataFrame:
        warehouses_pool = [
            ("WH_NORTH_01", "Northern Regional Fulfillment Center", "Chicago", "IL", "USA", 500_000),
            ("WH_SOUTH_01", "Dallas Distribution Hub", "Dallas", "TX", "USA", 650_000),
            ("WH_WEST_01", "Pacific Gateway Warehouse", "Reno", "NV", "USA", 400_000),
            ("WH_EAST_01", "Atlantic Crossdock Facility", "Allentown", "PA", "USA", 600_000),
            ("WH_CENTRAL_01", "Midwest Central Logistics", "Indianapolis", "IN", "USA", 800_000),
            ("WH_TEXAS_02", "Houston Port Terminal", "Houston", "TX", "USA", 350_000),
            ("WH_CALIF_02", "Inland Empire Depot", "Ontario", "CA", "USA", 550_000),
        ]
        n = min(self.config.number_of_warehouses, len(warehouses_pool))
        selected = warehouses_pool[:n]
        return pd.DataFrame(
            selected,
            columns=["warehouse_id", "warehouse_name", "city", "state", "country", "capacity_units"],
        )

    def _generate_suppliers(self) -> pd.DataFrame:
        names = [
            "Apex Global Manufacturing", "Titan Electronics Ltd", "Summit Apparel Group",
            "OmniTextiles Corp", "Vanguard Home Goods", "Pinnacle Beauty Labs",
            "Aura Cosmetic Works", "Zenith Sporting Goods", "Prime Logistics Supply",
            "Pacific Rim Imports", "Beacon Foods & Organics", "Frontier Hardware Co",
            "Starlight Consumer Tech", "Velocity Gear Inc", "Heritage Leatherworks",
            "EcoFiber Solutions", "Alpine Sports Equipment", "Precision Electronics Inc",
            "Horizon Nutrition Ltd", "Atlas Global Brands", "Crestview Packaging",
        ]
        suppliers = []
        for i in range(self.config.number_of_suppliers):
            s_id = f"SUPP_{i+1:03d}"
            name = names[i % len(names)] + (f" #{i//len(names) + 1}" if i >= len(names) else "")
            lead_time = int(self.rng.choice([5, 7, 10, 14, 21, 28, 35], p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.05, 0.05]))
            moq = int(self.rng.choice([25, 50, 100, 200, 500], p=[0.2, 0.3, 0.3, 0.15, 0.05]))
            suppliers.append({
                "supplier_id": s_id,
                "supplier_name": name,
                "average_lead_time_days": lead_time,
                "minimum_order_quantity": moq,
            })
        return pd.DataFrame(suppliers)

    def _generate_products(self, suppliers_df: pd.DataFrame) -> pd.DataFrame:
        categories = {
            "Electronics": {"cogs_range": (30.0, 300.0), "margin": (0.30, 0.50), "brands": ["VoltMax", "SoundPulse", "OptiView", "HyperTech"]},
            "Fashion": {"cogs_range": (8.0, 60.0), "margin": (0.50, 0.70), "brands": ["UrbanFit", "Loom & Thread", "AeroStyle", "SoleCraft"]},
            "Beauty": {"cogs_range": (5.0, 40.0), "margin": (0.60, 0.80), "brands": ["PureGlow", "DermaCare", "LuxeEssence", "BioBotanicals"]},
            "Home": {"cogs_range": (15.0, 120.0), "margin": (0.40, 0.60), "brands": ["NestHaven", "ChefCraft", "NordicLiving", "ArtisanWood"]},
            "Sports": {"cogs_range": (12.0, 90.0), "margin": (0.45, 0.65), "brands": ["PeakPerformance", "IronCore", "TrailMaster", "AquaPulse"]},
            "Grocery": {"cogs_range": (3.0, 25.0), "margin": (0.25, 0.40), "brands": ["NatureHarvest", "VitalOrganics", "DailyBite", "FarmSelect"]},
        }
        category_keys = list(categories.keys())

        products = []
        supplier_ids = suppliers_df["supplier_id"].tolist()

        # Distribute velocity profiles
        # 15% High, 30% Medium, 40% Slow, 10% Intermittent, 5% New Launch
        n_skus = self.config.number_of_skus
        velocity_assignments = self.rng.choice(
            ["HIGH", "MEDIUM", "SLOW", "INTERMITTENT", "NEW_LAUNCH"],
            size=n_skus,
            p=[0.15, 0.30, 0.40, 0.10, 0.05],
        )

        for i in range(n_skus):
            sku_id = f"SKU_{i+1:05d}"
            cat_name = category_keys[i % len(category_keys)]
            cat_info = categories[cat_name]

            brand = str(self.rng.choice(cat_info["brands"]))
            cogs = round(float(self.rng.uniform(*cat_info["cogs_range"])), 2)
            margin = float(self.rng.uniform(*cat_info["margin"]))
            selling_price = round(cogs / (1.0 - margin), 2)

            product_name = f"{brand} {cat_name} Item {i+1}"
            pref_supplier = supplier_ids[i % len(supplier_ids)]
            velocity = velocity_assignments[i]

            products.append({
                "sku_id": sku_id,
                "product_name": product_name,
                "category_id": cat_name,
                "brand": brand,
                "unit_cost": cogs,
                "selling_price": selling_price,
                "currency": "USD",
                "preferred_supplier_id": pref_supplier,
                "velocity_tier": velocity,
            })

        return pd.DataFrame(products)

    # -------------------------------------------------------------------------
    # Closed-Loop Simulation (Sales, Purchases, Inventory, Returns)
    # -------------------------------------------------------------------------

    def _simulate_commerce_dynamics(
        self,
        products_df: pd.DataFrame,
        warehouses_df: pd.DataFrame,
        channels_df: pd.DataFrame,
        suppliers_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs day-by-day simulation coupling sales, inventory depletion, POs, receipts, and returns."""
        skus = products_df["sku_id"].tolist()
        whs = warehouses_df["warehouse_id"].tolist()
        channels = channels_df["channel_id"].tolist()
        supp_map = suppliers_df.set_index("supplier_id").to_dict(orient="index")
        prod_map = products_df.set_index("sku_id").to_dict(orient="index")

        num_days = self.config.historical_days
        start_date = self.config.start_date
        date_list = [start_date + timedelta(days=d) for d in range(num_days)]

        # Category return rate mapping
        category_return_rates = {
            "Fashion": 0.22,
            "Electronics": 0.10,
            "Home": 0.07,
            "Sports": 0.06,
            "Beauty": 0.04,
            "Grocery": 0.015,
        }

        # Base daily demand rates by velocity tier
        tier_base_demand = {
            "HIGH": (12.0, 25.0),
            "MEDIUM": (3.0, 8.0),
            "SLOW": (0.2, 1.0),
            "INTERMITTENT": (0.05, 0.3),
            "NEW_LAUNCH": (5.0, 15.0),
        }

        # Channel traffic weights
        ch_weights = [0.45, 0.25, 0.15, 0.15][:len(channels)]
        ch_weights = np.array(ch_weights) / sum(ch_weights)

        # SKU parameters: base demand, trend factor, seasonal peak
        sku_params = {}
        for s in skus:
            info = prod_map[s]
            tier = info["velocity_tier"]
            b_min, b_max = tier_base_demand[tier]
            base_dem = float(self.rng.uniform(b_min, b_max))

            # Trend: growing (+30% to +80%), declining (-20% to -50%), or flat
            trend_type = self.rng.choice(["flat", "growing", "declining"], p=[0.5, 0.3, 0.2])
            if trend_type == "growing":
                trend_slope = float(self.rng.uniform(0.3, 0.7)) / num_days
            elif trend_type == "declining":
                trend_slope = -float(self.rng.uniform(0.2, 0.5)) / num_days
            else:
                trend_slope = 0.0

            # Launch day: day 0 except NEW_LAUNCH which launches midway (day 250-400)
            launch_day = int(self.rng.integers(250, 450)) if tier == "NEW_LAUNCH" else 0

            sku_params[s] = {
                "base_demand": base_dem,
                "trend_slope": trend_slope,
                "launch_day": launch_day,
                "category": info["category_id"],
                "tier": tier,
                "supplier_id": info["preferred_supplier_id"],
                "unit_cost": info["unit_cost"],
                "selling_price": info["selling_price"],
            }

        # Initialize inventory state for each (sku, wh)
        # We start with 15-30 days of supply for regular, higher for slow, zero for new launch
        inv_state: Dict[Tuple[str, str], Dict[str, int]] = {}
        reorder_point: Dict[Tuple[str, str], int] = {}
        target_stock: Dict[Tuple[str, str], int] = {}

        for s in skus:
            supp_info = supp_map[sku_params[s]["supplier_id"]]
            lead_time = supp_info["average_lead_time_days"]
            moq = supp_info["minimum_order_quantity"]
            daily_per_wh = max(0.5, sku_params[s]["base_demand"] / len(whs))

            rop = int(np.ceil(daily_per_wh * (lead_time + 5)))  # ROP = demand * (lead + buffer)
            tgt = max(rop + moq, int(np.ceil(rop + daily_per_wh * 20)))

            for w in whs:
                pair = (s, w)
                reorder_point[pair] = rop
                target_stock[pair] = tgt

                if sku_params[s]["launch_day"] > 0:
                    init_stock = 0
                elif sku_params[s]["tier"] == "SLOW":
                    # Slow movers often have high MOQ overstock
                    init_stock = max(moq * 2, int(tgt * 1.5))
                else:
                    init_stock = int(self.rng.integers(int(rop * 0.8), int(tgt * 1.2) + 1))

                inv_state[pair] = {
                    "available": init_stock,
                    "in_transit": 0,
                    "reserved": 0,
                    "damaged": int(self.rng.choice([0, 1, 2], p=[0.85, 0.12, 0.03])),
                }

        # Scheduled Inbound Purchase Orders: list of (delivery_date, po_dict)
        scheduled_receipts: List[Tuple[date, Dict[str, Any]]] = []
        purchases_records: List[Dict[str, Any]] = []
        sales_records: List[Dict[str, Any]] = []
        returns_records: List[Dict[str, Any]] = []
        inventory_records: List[Dict[str, Any]] = []

        po_counter = 1
        sale_counter = 1
        order_counter = 10000
        return_counter = 1

        # Precompute promotional campaigns (e.g. 5 major sales events across the 2 years)
        promo_days = set()
        # Q4 Holiday Cyber Week (days ~320-330, and ~685-695)
        for d in [325, 326, 327, 328, 690, 691, 692, 693]:
            if d < num_days:
                promo_days.add(d)

        # Simulation loop over historical_days
        for day_idx, current_date in enumerate(date_list):
            is_promo_day = day_idx in promo_days
            weekday = current_date.weekday()  # 0=Mon, 6=Sun
            day_of_month = current_date.day

            # Weekly seasonality factor (higher Fri-Sun)
            weekly_factor = 1.25 if weekday in [4, 5, 6] else 0.90
            # Month-end salary cycle factor (days 28-31)
            monthly_factor = 1.15 if day_of_month >= 28 else 1.00
            # Annual holiday seasonality (Nov & Dec)
            holiday_factor = 1.40 if current_date.month in [11, 12] else 1.00

            # 1. Process dock receipts arriving today
            receipts_today = [item for item in scheduled_receipts if item[0] == current_date]
            for _, po in receipts_today:
                s = po["sku_id"]
                w = po["warehouse_id"]
                qty = po["quantity"]
                inv_state[(s, w)]["available"] += qty
                inv_state[(s, w)]["in_transit"] = max(0, inv_state[(s, w)]["in_transit"] - qty)

            # 2. Daily Sales Generation & Inventory Depletion
            for s in skus:
                p_params = sku_params[s]
                if day_idx < p_params["launch_day"]:
                    continue

                # Trend multiplier
                trend_mult = 1.0 + p_params["trend_slope"] * (day_idx - p_params["launch_day"])
                trend_mult = max(0.05, trend_mult)

                # Promotion multiplier
                promo_mult = float(self.rng.uniform(2.5, 4.5)) if is_promo_day else 1.0

                expected_demand_total = (
                    p_params["base_demand"]
                    * weekly_factor
                    * monthly_factor
                    * holiday_factor
                    * trend_mult
                    * promo_mult
                )

                # Intermittent demand check
                if p_params["tier"] == "INTERMITTENT":
                    if self.rng.random() > 0.15:  # 85% days zero demand
                        expected_demand_total = 0.0

                if expected_demand_total <= 0:
                    continue

                # Distribute demand across warehouses
                wh_demand = self.rng.poisson(expected_demand_total / len(whs), size=len(whs))

                for w_idx, w in enumerate(whs):
                    unconstrained_qty = int(wh_demand[w_idx])
                    if unconstrained_qty <= 0:
                        continue

                    pair = (s, w)
                    avail = inv_state[pair]["available"]

                    # Realistic Stockout constraint: can only sell what is on hand
                    realized_qty = min(unconstrained_qty, avail)

                    if realized_qty > 0:
                        inv_state[pair]["available"] -= realized_qty

                        # Generate sale line item
                        order_counter += 1
                        order_id = f"ORD_{order_counter}"
                        sale_id = f"SALE_{sale_counter:07d}"
                        sale_counter += 1

                        channel_id = str(self.rng.choice(channels, p=ch_weights))
                        unit_p = p_params["selling_price"]
                        # Promotional discount
                        discount_pct = float(self.rng.uniform(0.10, 0.25)) if is_promo_day else (
                            float(self.rng.uniform(0.05, 0.15)) if self.rng.random() < 0.1 else 0.0
                        )
                        line_discount = round(realized_qty * unit_p * discount_pct, 2)
                        gross_rev = round(realized_qty * unit_p, 2)
                        net_rev = max(0.0, round(gross_rev - line_discount, 2))

                        sales_records.append({
                            "sale_id": sale_id,
                            "order_id": order_id,
                            "date": current_date.isoformat(),
                            "sku_id": s,
                            "warehouse_id": w,
                            "channel_id": channel_id,
                            "quantity": realized_qty,
                            "unit_price": unit_p,
                            "discount": line_discount,
                            "revenue": net_rev,
                            "currency": "USD",
                        })

                        # Return probability check
                        cat_ret_rate = category_return_rates.get(p_params["category"], 0.05)
                        if self.rng.random() < cat_ret_rate:
                            lag_days = int(self.rng.integers(3, 15))
                            ret_date = current_date + timedelta(days=lag_days)
                            if ret_date <= date_list[-1]:
                                reason_options = [
                                    "Defective Item", "Wrong Size/Color", "Late Delivery",
                                    "Item Not as Described", "Customer Regret",
                                ]
                                ret_reason = str(self.rng.choice(reason_options))
                                return_id = f"RET_{return_counter:06d}"
                                return_counter += 1

                                returns_records.append({
                                    "return_id": return_id,
                                    "order_id": order_id,
                                    "return_date": ret_date.isoformat(),
                                    "sku_id": s,
                                    "warehouse_id": w,
                                    "quantity": int(self.rng.integers(1, realized_qty + 1)),
                                    "reason": ret_reason,
                                    "channel_id": channel_id,
                                })

            # 3. Replenishment Check: trigger POs if available + in_transit <= ROP
            for s in skus:
                p_params = sku_params[s]
                if day_idx < p_params["launch_day"]:
                    # Schedule initial PO for new launch SKUs before launch date
                    if day_idx == p_params["launch_day"] - 20:
                        pass
                    else:
                        continue

                supp_info = supp_map[p_params["supplier_id"]]
                base_lead = supp_info["average_lead_time_days"]
                moq = supp_info["minimum_order_quantity"]

                for w in whs:
                    pair = (s, w)
                    curr_state = inv_state[pair]
                    pipeline = curr_state["available"] + curr_state["in_transit"]

                    if pipeline <= reorder_point[pair]:
                        # Reorder up to target stock
                        needed = target_stock[pair] - pipeline
                        order_qty = max(needed, moq)

                        # Actual lead time with realistic stochastic variance (-2 to +5 days)
                        actual_lead = max(2, int(base_lead + self.rng.integers(-2, 6)))
                        expected_deliv = current_date + timedelta(days=base_lead)
                        actual_deliv = current_date + timedelta(days=actual_lead)

                        curr_state["in_transit"] += order_qty

                        po_id = f"PO_{po_counter:06d}"
                        po_counter += 1

                        status = "DELIVERED" if actual_deliv <= date_list[-1] else "IN_TRANSIT"
                        delivered_dt_str = actual_deliv.isoformat() if status == "DELIVERED" else None

                        po_dict = {
                            "purchase_order_id": po_id,
                            "order_date": current_date.isoformat(),
                            "sku_id": s,
                            "supplier_id": p_params["supplier_id"],
                            "warehouse_id": w,
                            "quantity": order_qty,
                            "unit_cost": p_params["unit_cost"],
                            "expected_delivery_date": expected_deliv.isoformat(),
                            "actual_delivery_date": delivered_dt_str,
                            "status": status,
                        }
                        purchases_records.append(po_dict)

                        if actual_deliv <= date_list[-1]:
                            scheduled_receipts.append((actual_deliv, po_dict))

            # 4. Inventory Daily Snapshot Recording
            # To maintain manageable file size while preserving rich dynamics,
            # snapshot weekly or daily. Let's record weekly + last 30 days daily,
            # or periodic snapshots. For full compliance, let's record snapshot for each day
            # if total skus is small, or weekly snapshots for 730 days.
            # 500 SKUs * 5 WHs * 730 days = 1.8M rows (too large for quick CSV).
            # Recording every 7 days (plus the last 14 days) captures the entire 2-year trajectory
            # at ~275k rows, which loads in < 1 second!
            is_snapshot_day = (day_idx % 7 == 0) or (day_idx >= num_days - 14)
            if is_snapshot_day:
                for pair, st in inv_state.items():
                    s_id, w_id = pair
                    inventory_records.append({
                        "snapshot_date": current_date.isoformat(),
                        "sku_id": s_id,
                        "warehouse_id": w_id,
                        "available_qty": st["available"],
                        "reserved_qty": st["reserved"],
                        "in_transit_qty": st["in_transit"],
                        "damaged_qty": st["damaged"],
                    })

        sales_df = pd.DataFrame(sales_records)
        inventory_df = pd.DataFrame(inventory_records)
        purchases_df = pd.DataFrame(purchases_records)
        returns_df = pd.DataFrame(returns_records)

        # Drop temporary helper columns from products_df
        clean_products_df = products_df.drop(
            columns=["preferred_supplier_id", "velocity_tier"],
            errors="ignore",
        )

        return sales_df, inventory_df, purchases_df, returns_df

    def _inject_controlled_anomalies(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Inject controlled anomalies for testing validation and quality alerting."""
        # Insert 1 invalid negative price into sales
        if "sales" in datasets and not datasets["sales"].empty:
            datasets["sales"].iloc[0, datasets["sales"].columns.get_loc("unit_price")] = -10.0
        # Insert 1 orphan SKU in returns
        if "returns" in datasets and not datasets["returns"].empty:
            datasets["returns"].iloc[0, datasets["returns"].columns.get_loc("sku_id")] = "SKU_NON_EXISTENT_99999"


def generate_sample_dataset(
    output_dir: Optional[Path] = None,
    number_of_skus: int = 500,
    number_of_warehouses: int = 5,
    number_of_channels: int = 4,
    number_of_suppliers: int = 20,
    historical_days: int = 730,
    random_seed: int = 42,
    include_controlled_anomalies: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Convenience function to generate and optionally save the synthetic dataset."""
    config = GeneratorConfig(
        number_of_skus=number_of_skus,
        number_of_warehouses=number_of_warehouses,
        number_of_channels=number_of_channels,
        number_of_suppliers=number_of_suppliers,
        historical_days=historical_days,
        random_seed=random_seed,
        include_controlled_anomalies=include_controlled_anomalies,
    )
    generator = SyntheticDataGenerator(config)
    if output_dir is not None:
        return generator.generate_and_save(output_dir)
    return generator.generate_all()
