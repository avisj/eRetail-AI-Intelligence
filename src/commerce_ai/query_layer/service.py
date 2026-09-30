"""Unified Query Layer Service (Phase 7A).

Provides a single, coordinated access point for dashboards and future eRetail Copilot:
- Injects underlying DataFrames and precomputed intelligence artifacts into read-only tools
- Enforces point-in-time constraints and currency isolation
- Exposes ToolRegistry for discovery and tool execution
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd

from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.registry import ToolRegistry, default_registry
from commerce_ai.query_layer.schemas import CalculationStatus, QueryMetadata, QueryResponse


class QueryLayerService:
    """Unified service orchestrating Business Intelligence Query Tools."""

    def __init__(
        self,
        sales_df: Optional[pd.DataFrame] = None,
        inventory_df: Optional[pd.DataFrame] = None,
        products_df: Optional[pd.DataFrame] = None,
        warehouses_df: Optional[pd.DataFrame] = None,
        returns_df: Optional[pd.DataFrame] = None,
        suppliers_df: Optional[pd.DataFrame] = None,
        replenishment_result: Optional[Any] = None,
        po_result: Optional[Any] = None,
        rebalancing_result: Optional[Any] = None,
        business_impact_result: Optional[Any] = None,
        recommendation_result: Optional[Any] = None,
        decision_result: Optional[Any] = None,
        forecast_output: Optional[Any] = None,
        forecast_df: Optional[pd.DataFrame] = None,
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.sales_df = sales_df if sales_df is not None else pd.DataFrame()
        self.inventory_df = inventory_df if inventory_df is not None else pd.DataFrame()
        self.products_df = products_df if products_df is not None else pd.DataFrame()
        self.warehouses_df = warehouses_df if warehouses_df is not None else pd.DataFrame()
        self.returns_df = returns_df if returns_df is not None else pd.DataFrame()
        self.suppliers_df = suppliers_df if suppliers_df is not None else pd.DataFrame()

        self.replenishment_result = replenishment_result
        self.po_result = po_result
        self.rebalancing_result = rebalancing_result
        self.business_impact_result = business_impact_result
        self.recommendation_result = recommendation_result
        self.decision_result = decision_result
        self.forecast_output = forecast_output
        self.forecast_df = forecast_df

        self.registry = registry or default_registry

    @classmethod
    def from_sample_data(
        cls,
        data_dir: Union[str, Path] = "data/sample",
        registry: Optional[ToolRegistry] = None,
    ) -> QueryLayerService:
        """Convenience loader for canonical sample directory."""
        p = Path(data_dir)
        sales = pd.read_csv(p / "sales.csv") if (p / "sales.csv").exists() else None
        inventory = pd.read_csv(p / "inventory.csv") if (p / "inventory.csv").exists() else None
        products = pd.read_csv(p / "products.csv") if (p / "products.csv").exists() else None
        warehouses = pd.read_csv(p / "warehouses.csv") if (p / "warehouses.csv").exists() else None
        returns = pd.read_csv(p / "returns.csv") if (p / "returns.csv").exists() else None
        suppliers = pd.read_csv(p / "suppliers.csv") if (p / "suppliers.csv").exists() else None

        return cls(
            sales_df=sales,
            inventory_df=inventory,
            products_df=products,
            warehouses_df=warehouses,
            returns_df=returns,
            suppliers_df=suppliers,
            registry=registry,
        )

    def execute_tool(
        self,
        tool_name: str,
        context: Optional[QueryContext] = None,
        **kwargs: Any,
    ) -> QueryResponse:
        """Execute a registered query tool dynamically by name."""
        ctx = context or QueryContext()
        handler = self.registry.get_tool(tool_name)
        if handler is None:
            meta = QueryMetadata(
                query_id=f"qry_unknown_{tool_name}",
                tool_name=tool_name,
                domain="unknown",
                generated_as_of=ctx.iso_as_of_date or "",
                calculation_status=CalculationStatus.ERROR,
            )
            return QueryResponse(
                query_id=meta.query_id,
                tool_name=tool_name,
                domain=meta.domain,
                status=CalculationStatus.ERROR,
                metadata=meta,
                error_message=f"Tool '{tool_name}' is not registered in ToolRegistry.",
            )

        # Inject known dataset parameters if not explicitly provided
        args: Dict[str, Any] = {"context": ctx}
        args.update(kwargs)

        # Sales tools
        if "sales_df" not in args:
            args["sales_df"] = self.sales_df
        if "products_df" not in args:
            args["products_df"] = self.products_df
        if "inventory_df" not in args:
            args["inventory_df"] = self.inventory_df
        if "returns_df" not in args:
            args["returns_df"] = self.returns_df
        if "suppliers_df" not in args:
            args["suppliers_df"] = self.suppliers_df

        # Intelligence artifacts
        if "replenishment_result" not in args:
            args["replenishment_result"] = self.replenishment_result
        if "po_result" not in args:
            args["po_result"] = self.po_result
        if "rebalancing_result" not in args:
            args["rebalancing_result"] = self.rebalancing_result
        if "impact_result" not in args:
            args["impact_result"] = self.business_impact_result
        if "recommendation_result" not in args:
            args["recommendation_result"] = self.recommendation_result
        if "decision_result" not in args:
            args["decision_result"] = self.decision_result
        if "forecast_output" not in args:
            args["forecast_output"] = self.forecast_output
        if "forecast_df" not in args:
            args["forecast_df"] = self.forecast_df

        # Inspect handler signature to filter valid arguments
        import inspect
        sig = inspect.signature(handler)
        valid_args = {}
        for param_name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                valid_args = args
                break
            if param_name in args:
                valid_args[param_name] = args[param_name]

        return handler(**valid_args)

    # -------------------------------------------------------------
    # Explicit Domain Dispatch Methods
    # -------------------------------------------------------------

    # Sales
    def get_sales_summary(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_sales_summary", context=context)

    def get_sales_trend(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_sales_trend", context=context)

    def get_sales_by_sku(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_sales_by_sku", context=context)

    def get_sales_by_channel(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_sales_by_channel", context=context)

    def get_sales_by_warehouse(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_sales_by_warehouse", context=context)

    def get_sales_by_category(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_sales_by_category", context=context)

    def get_sales_by_brand(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_sales_by_brand", context=context)

    # Financial
    def get_revenue_summary(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_revenue_summary", context=context)

    def get_margin_summary(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_margin_summary", context=context)

    def get_unit_economics(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_unit_economics", context=context)

    def get_margin_drivers(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_margin_drivers", context=context)

    def get_operational_economics(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_operational_economics", context=context)

    def get_profitability_attribution(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_profitability_attribution", context=context)

    # Inventory
    def get_inventory_summary(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_inventory_summary", context=context)

    def get_inventory_position(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_inventory_position", context=context)

    def get_stockout_risk(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_stockout_risk", context=context)

    def get_slow_moving_inventory(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_slow_moving_inventory", context=context)

    def get_high_value_inventory(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_high_value_inventory", context=context)

    def get_inventory_risk(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_inventory_risk", context=context)

    def get_inventory_risk_breakdown(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_inventory_risk", context=context)

    # Demand
    def get_demand_summary(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_demand_summary", context=context)

    def get_demand_trend(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_demand_trend", context=context)

    def get_demand_profile(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_demand_profile", context=context)

    def get_abc_xyz_distribution(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_abc_xyz_distribution", context=context)

    # Forecasting
    def get_forecast(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_forecast", context=context)

    def get_forecast_accuracy(self, context: Optional[QueryContext] = None, **kwargs: Any) -> QueryResponse:
        return self.execute_tool("get_forecast_accuracy", context=context, **kwargs)

    def get_forecast_bias(self, context: Optional[QueryContext] = None, **kwargs: Any) -> QueryResponse:
        return self.execute_tool("get_forecast_bias", context=context, **kwargs)

    def get_forecast_summary(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_forecast_summary", context=context)

    # Returns
    def get_return_summary(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_return_summary", context=context)

    def get_return_rate(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_return_rate", context=context)

    def get_return_trend(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_return_trend", context=context)

    def get_return_anomalies(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_return_anomalies", context=context)

    def get_return_risk(self, context: Optional[QueryContext] = None, **kwargs: Any) -> QueryResponse:
        return self.execute_tool("get_return_risk", context=context, **kwargs)

    def get_return_reason_breakdown(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_return_reason_breakdown", context=context)

    # Operations
    def get_replenishment_reviews(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_replenishment_reviews", context=context)

    def get_purchase_order_reviews(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_purchase_order_reviews", context=context)

    def get_warehouse_rebalancing_reviews(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_warehouse_rebalancing_reviews", context=context)

    # Business Impact
    def get_business_impact_summary(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_business_impact_summary", context=context)

    def get_business_impact_by_category(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_business_impact_by_category", context=context)

    def get_business_impact_by_type(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_business_impact_by_type", context=context)

    def get_physical_capital_exposure(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_physical_capital_exposure", context=context)

    # Recommendations
    def get_recommendation_summary(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_recommendation_summary", context=context)

    def get_recommendations(self, context: Optional[QueryContext] = None, **kwargs: Any) -> QueryResponse:
        return self.execute_tool("get_recommendations", context=context, **kwargs)

    def get_recommendation_list(self, context: Optional[QueryContext] = None, **kwargs: Any) -> QueryResponse:
        return self.execute_tool("get_recommendations", context=context, **kwargs)

    def get_recommendation_by_id(self, recommendation_id: str, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_recommendation_by_id", context=context, recommendation_id=recommendation_id)

    # Decisions
    def get_decision_summary(self, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_decision_summary", context=context)

    def get_decision_packages(self, context: Optional[QueryContext] = None, **kwargs: Any) -> QueryResponse:
        return self.execute_tool("get_decision_packages", context=context, **kwargs)

    def get_decision_list(self, context: Optional[QueryContext] = None, **kwargs: Any) -> QueryResponse:
        return self.execute_tool("get_decision_packages", context=context, **kwargs)

    def get_decision_by_id(self, decision_id: str, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_decision_package_by_id", context=context, decision_id=decision_id)

    def get_decision_package_by_id(self, decision_id: str, context: Optional[QueryContext] = None) -> QueryResponse:
        return self.execute_tool("get_decision_package_by_id", context=context, decision_id=decision_id)

    def to_copilot_declarations(self) -> List[Dict[str, Any]]:
        """Export all registered tools as Copilot function declarations."""
        return self.registry.to_copilot_declarations()
