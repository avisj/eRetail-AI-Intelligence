"""Integration and Unit Tests for InventoryService (Phase 4A End-to-End Pipeline)."""

from datetime import date, timedelta
import pandas as pd
import pytest

from commerce_ai.inventory.service import InventoryService, InventoryAnalysisResult
from commerce_ai.inventory.safety_stock import ServiceLevelPolicy
from commerce_ai.forecasting.service import ForecastService


@pytest.fixture
def ecommerce_portfolio_fixture():
    """Builds a miniature, consistent multi-entity dataset."""
    start_date = date(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(60)]

    # 1. Products
    products_df = pd.DataFrame([
        {"sku_id": "SKU_001", "product_name": "Premium Headphones", "category_id": "Electronics", "unit_cost": 50.0, "selling_price": 90.0, "preferred_supplier_id": "SUPP_A"},
        {"sku_id": "SKU_002", "product_name": "Organic Cotton Tee", "category_id": "Fashion", "unit_cost": 10.0, "selling_price": 25.0, "preferred_supplier_id": "SUPP_B"},
    ])

    # 2. Suppliers
    suppliers_df = pd.DataFrame([
        {"supplier_id": "SUPP_A", "supplier_name": "Alpha Tech", "average_lead_time_days": 10, "minimum_order_quantity": 20},
        {"supplier_id": "SUPP_B", "supplier_name": "Beta Apparel", "average_lead_time_days": 14, "minimum_order_quantity": 50},
    ])

    # 3. Purchases
    purchases_df = pd.DataFrame([
        # Delivered POs for SUPP_A
        {"purchase_order_id": "PO_1", "supplier_id": "SUPP_A", "sku_id": "SKU_001", "warehouse_id": "WH_01", "quantity": 100, "status": "DELIVERED", "order_date": "2024-01-05", "expected_delivery_date": "2024-01-15", "actual_delivery_date": "2024-01-16"},
        {"purchase_order_id": "PO_2", "supplier_id": "SUPP_A", "sku_id": "SKU_001", "warehouse_id": "WH_01", "quantity": 100, "status": "DELIVERED", "order_date": "2024-01-20", "expected_delivery_date": "2024-01-30", "actual_delivery_date": "2024-01-29"},
        {"purchase_order_id": "PO_3", "supplier_id": "SUPP_A", "sku_id": "SKU_001", "warehouse_id": "WH_01", "quantity": 100, "status": "DELIVERED", "order_date": "2024-02-05", "expected_delivery_date": "2024-02-15", "actual_delivery_date": "2024-02-15"},
        # Open PO for SKU_001 in transit
        {"purchase_order_id": "PO_4", "supplier_id": "SUPP_A", "sku_id": "SKU_001", "warehouse_id": "WH_01", "quantity": 50, "status": "IN_TRANSIT", "order_date": "2024-02-25", "expected_delivery_date": "2024-03-05", "actual_delivery_date": None},
    ])

    # 4. Sales
    sales_rows = []
    sale_id = 1
    for d in dates:
        # SKU_001 sells 5-10 units daily
        qty_1 = 8 if d.weekday() in [4, 5] else 5
        sales_rows.append({
            "sale_id": f"S_{sale_id}",
            "order_id": f"ORD_{sale_id}",
            "date": d.isoformat(),
            "sku_id": "SKU_001",
            "warehouse_id": "WH_01",
            "channel_id": "CH_WEB",
            "quantity": qty_1,
            "unit_price": 90.0,
            "revenue": qty_1 * 90.0,
        })
        sale_id += 1

        # SKU_002 sells 1-2 units daily
        qty_2 = 2 if d.day % 2 == 0 else 1
        sales_rows.append({
            "sale_id": f"S_{sale_id}",
            "order_id": f"ORD_{sale_id}",
            "date": d.isoformat(),
            "sku_id": "SKU_002",
            "warehouse_id": "WH_01",
            "channel_id": "CH_WEB",
            "quantity": qty_2,
            "unit_price": 25.0,
            "revenue": qty_2 * 25.0,
        })
        sale_id += 1

    sales_df = pd.DataFrame(sales_rows)

    # 5. Inventory Snapshots
    inv_rows = []
    for d in dates:
        # SKU_001 has 30 units on hand at the end
        inv_rows.append({
            "snapshot_date": d.isoformat(),
            "sku_id": "SKU_001",
            "warehouse_id": "WH_01",
            "available_qty": 30 if d == dates[-1] else 80,
            "reserved_qty": 5,
            "in_transit_qty": 50,
            "damaged_qty": 0,
        })
        # SKU_002 has 200 units on hand (potential overstock)
        inv_rows.append({
            "snapshot_date": d.isoformat(),
            "sku_id": "SKU_002",
            "warehouse_id": "WH_01",
            "available_qty": 200,
            "reserved_qty": 0,
            "in_transit_qty": 0,
            "damaged_qty": 0,
        })
    inventory_df = pd.DataFrame(inv_rows)

    return {
        "products": products_df,
        "suppliers": suppliers_df,
        "purchases": purchases_df,
        "sales": sales_df,
        "inventory": inventory_df,
    }


class TestInventoryService:
    def test_full_pipeline_integration(self, ecommerce_portfolio_fixture):
        """End-to-end integration test:

        Data -> Demand -> Forecast -> Inventory Position -> Lead Time -> Safety Stock -> ROP -> Inventory Risk
        """
        bundle = ecommerce_portfolio_fixture
        service = InventoryService()

        # Run analysis across portfolio
        result: InventoryAnalysisResult = service.analyze_portfolio(
            sales=bundle["sales"],
            inventory=bundle["inventory"],
            products=bundle["products"],
            suppliers=bundle["suppliers"],
            purchases=bundle["purchases"],
            horizon=14,
        )

        # 1. Assert positions
        assert not result.positions.empty
        assert len(result.positions) == 2
        sku1_pos = result.positions[result.positions["sku_id"] == "SKU_001"].iloc[0]
        # available: 30, open PO: 50, reserved: 5 -> net_position = 75
        assert sku1_pos["on_hand"] == 30
        assert sku1_pos["on_order"] == 50
        assert sku1_pos["reserved"] == 5
        assert sku1_pos["net_position"] == 75

        # 2. Assert lead times
        assert not result.lead_times.empty
        sku1_lt = result.lead_times[result.lead_times["sku_id"] == "SKU_001"].iloc[0]
        assert sku1_lt["sample_size"] == 3
        assert bool(sku1_lt["is_fallback"]) is False

        # 3. Assert safety stock & ROP
        assert not result.safety_stocks.empty
        assert len(result.safety_stocks) == 2
        for _, ss_row in result.safety_stocks.iterrows():
            assert ss_row["safety_stock"] >= 0.0
            assert ss_row["reorder_point"] >= ss_row["safety_stock"]
            assert ss_row["service_level"] > 0.0

        # 4. Assert inventory risks
        assert not result.risks.empty
        assert len(result.risks) == 2
        sku1_risk = result.risks[result.risks["sku_id"] == "SKU_001"].iloc[0]
        sku2_risk = result.risks[result.risks["sku_id"] == "SKU_002"].iloc[0]

        # SKU_002 has 200 units on hand selling ~1.5/day -> DOS > 100 days -> OVERSTOCK
        assert sku2_risk["risk_category"] == "OVERSTOCK"
        assert sku2_risk["days_of_supply"] > 60.0
        assert sku2_risk["excess_units"] > 0
        assert sku2_risk["excess_capital"] > 0

        # Evidence transparency
        assert "daily_demand_forecast_mean" in sku1_risk["underlying_metrics"]
        assert "lead_time_mean_days" in sku1_risk["underlying_metrics"]
        assert "reorder_point" in sku1_risk["underlying_metrics"]

        # 5. Assert summary KPIs
        assert "total_series_analyzed" in result.summary
        assert result.summary["total_series_analyzed"] == 2
        assert "total_excess_capital_at_risk" in result.summary
        assert result.summary["total_excess_capital_at_risk"] > 0.0

    def test_custom_service_level_policy_injection(self, ecommerce_portfolio_fixture):
        bundle = ecommerce_portfolio_fixture
        custom_policy = ServiceLevelPolicy(
            default_service_level=0.90,
            sku_overrides={"SKU_001": 0.999},
        )
        service = InventoryService(service_level_policy=custom_policy)

        result = service.analyze_portfolio(
            sales=bundle["sales"],
            inventory=bundle["inventory"],
            products=bundle["products"],
            suppliers=bundle["suppliers"],
            purchases=bundle["purchases"],
            horizon=7,
        )

        sku1_ss = result.safety_stocks[result.safety_stocks["sku_id"] == "SKU_001"].iloc[0]
        assert sku1_ss["service_level"] == 0.999

    def test_forecast_for_rop_and_rate_source(self, ecommerce_portfolio_fixture):
        """Fix 4: Verify use_forecast_for_rop uses forecast rate and reports correct demand_rate_source."""
        bundle = ecommerce_portfolio_fixture

        # 1. Default: use_forecast_for_rop = True
        service_fc = InventoryService(use_forecast_for_rop=True)
        result_fc = service_fc.analyze_portfolio(
            sales=bundle["sales"],
            inventory=bundle["inventory"],
            products=bundle["products"],
            suppliers=bundle["suppliers"],
            purchases=bundle["purchases"],
            horizon=7,
        )
        assert not result_fc.safety_stocks.empty
        sku1_ss = result_fc.safety_stocks[result_fc.safety_stocks["sku_id"] == "SKU_001"].iloc[0]
        assert sku1_ss["demand_rate_source"] == "FORECAST"
        # Safety stock should preserve historical demand variance (daily_demand_std > 0)
        assert sku1_ss["daily_demand_std"] > 0.0
        assert sku1_ss["safety_stock"] > 0.0

        # 2. Disabled: use_forecast_for_rop = False
        service_hist = InventoryService(use_forecast_for_rop=False)
        result_hist = service_hist.analyze_portfolio(
            sales=bundle["sales"],
            inventory=bundle["inventory"],
            products=bundle["products"],
            suppliers=bundle["suppliers"],
            purchases=bundle["purchases"],
            horizon=7,
        )
        sku1_hist_ss = result_hist.safety_stocks[result_hist.safety_stocks["sku_id"] == "SKU_001"].iloc[0]
        assert sku1_hist_ss["demand_rate_source"] == "HISTORICAL"

    def test_fallback_to_historical_demand_rate_source(self, ecommerce_portfolio_fixture):
        """Fix 4: When forecast is empty or missing, falls back to historical demand rate."""
        bundle = ecommerce_portfolio_fixture
        service = InventoryService(use_forecast_for_rop=True)

        # Pass empty forecast DataFrame
        result_empty_fc = service.analyze_portfolio(
            sales=bundle["sales"],
            inventory=bundle["inventory"],
            products=bundle["products"],
            suppliers=bundle["suppliers"],
            purchases=bundle["purchases"],
            forecast_bundle=pd.DataFrame(),
            horizon=7,
        )
        assert not result_empty_fc.safety_stocks.empty
        sku1_ss = result_empty_fc.safety_stocks[result_empty_fc.safety_stocks["sku_id"] == "SKU_001"].iloc[0]
        assert sku1_ss["demand_rate_source"] == "HISTORICAL_FALLBACK"
        assert sku1_ss["reorder_point"] > 0.0
        assert sku1_ss["safety_stock"] > 0.0

