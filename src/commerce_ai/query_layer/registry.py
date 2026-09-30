"""Tool Registry for Business Intelligence Query Layer (Phase 7A).

Provides controlled, read-only tool discovery and execution metadata.
This registry forms the formal foundation for:
1. Dashboard backend dispatching
2. Future eRetail Copilot agentic tool calling
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Union
from pydantic import BaseModel, ConfigDict, Field

from commerce_ai.query_layer.schemas import QueryDomain


class ToolMetadata(BaseModel):
    """Metadata describing a registered Business Intelligence Query Tool."""
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(description="Unique identifier of the tool (e.g. get_sales_summary)")
    display_name: str = Field(description="Human-readable tool title")
    description: str = Field(description="Comprehensive tool purpose for dashboards and Copilot reasoning")
    domain: str = Field(description="Business domain (sales, inventory, demand, financial, etc.)")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Expected context and input parameters")
    output_schema: str = Field(default="QueryResponse", description="Schema type of the result")
    read_only: bool = Field(default=True, description="Strictly True for all Phase 7A tools")
    supports_filters: List[str] = Field(default_factory=list, description="Supported filter dimensions")
    supports_as_of_date: bool = Field(default=True, description="Enforces point-in-time safety")
    supports_currency: bool = Field(default=True, description="Enforces currency isolation")
    source_engine: str = Field(description="Underlying commerce_ai service or computation")


class ToolRegistry:
    """Registry maintaining available Business Intelligence Query Tools."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        tool_name: str,
        handler: Callable[..., Any],
        metadata: ToolMetadata,
    ) -> None:
        """Register a read-only query tool with its handler and metadata."""
        if not metadata.read_only:
            raise ValueError(f"Tool '{tool_name}' must be read_only=True in Phase 7A.")
        self._registry[tool_name] = {
            "handler": handler,
            "metadata": metadata,
        }

    def get_tool(self, tool_name: str) -> Optional[Callable[..., Any]]:
        """Retrieve executable tool handler by name."""
        entry = self._registry.get(tool_name)
        return entry["handler"] if entry else None

    def get_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Retrieve tool metadata by name."""
        entry = self._registry.get(tool_name)
        return entry["metadata"] if entry else None

    def list_tools(self, domain: Optional[Union[str, QueryDomain]] = None) -> List[ToolMetadata]:
        """List metadata for all registered tools, optionally filtered by domain."""
        dom_filter = domain.value if isinstance(domain, QueryDomain) else (domain.lower() if domain else None)
        tools = [entry["metadata"] for entry in self._registry.values()]
        if dom_filter:
            tools = [t for t in tools if t.domain.lower() == dom_filter]
        return sorted(tools, key=lambda x: (x.domain, x.tool_name))

    def get_all_tool_names(self) -> List[str]:
        """Return sorted list of all registered tool names."""
        return sorted(list(self._registry.keys()))

    def to_copilot_declarations(self) -> List[Dict[str, Any]]:
        """Produce structured declarations for future eRetail Copilot tool calling."""
        declarations = []
        for meta in self.list_tools():
            declarations.append({
                "name": meta.tool_name,
                "description": meta.description,
                "parameters": {
                    "type": "object",
                    "properties": meta.input_schema.get("properties", {}),
                    "required": meta.input_schema.get("required", []),
                },
                "read_only": meta.read_only,
                "domain": meta.domain,
                "source_engine": meta.source_engine,
            })
        return declarations


# Global Default Registry
default_registry = ToolRegistry()


def register_standard_tools(registry: ToolRegistry) -> None:
    """Populate registry with standard Phase 7A read-only business tools."""
    from commerce_ai.query_layer import (
        sales,
        financial,
        inventory,
        demand,
        forecasting,
        returns,
        operations,
        impact,
        recommendations,
        decisions,
    )

    # 1. Sales Tools
    registry.register(
        tool_name="get_sales_summary",
        handler=sales.get_sales_summary,
        metadata=ToolMetadata(
            tool_name="get_sales_summary",
            display_name="Sales Revenue Summary",
            description="Computes executive commercial sales KPIs: Gross Revenue, Net Revenue, Units Sold, Orders, AOV.",
            domain=QueryDomain.SALES.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["sku_id", "warehouse_id", "channel_id", "category_id", "brand", "start_date", "end_date", "as_of_date", "currency"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.sales",
        ),
    )
    registry.register(
        tool_name="get_sales_trend",
        handler=sales.get_sales_trend,
        metadata=ToolMetadata(
            tool_name="get_sales_trend",
            display_name="Sales Revenue Trend",
            description="Generates net revenue and unit volume time series across daily, weekly, or monthly granularity.",
            domain=QueryDomain.SALES.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[TimeSeriesResult]",
            read_only=True,
            supports_filters=["sku_id", "warehouse_id", "channel_id", "start_date", "end_date", "as_of_date", "currency", "time_grain"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.sales",
        ),
    )
    registry.register(
        tool_name="get_sales_by_sku",
        handler=sales.get_sales_by_sku,
        metadata=ToolMetadata(
            tool_name="get_sales_by_sku",
            display_name="Sales by SKU",
            description="Calculates tabular sales performance broken down by SKU without subjective rankings.",
            domain=QueryDomain.SALES.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["sku_id", "warehouse_id", "channel_id", "start_date", "end_date", "as_of_date", "currency", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.sales",
        ),
    )
    registry.register(
        tool_name="get_sales_by_channel",
        handler=sales.get_sales_by_channel,
        metadata=ToolMetadata(
            tool_name="get_sales_by_channel",
            display_name="Sales by Channel",
            description="Calculates commercial revenue breakdown across sales channels (Direct, Amazon, Wholesale, etc.).",
            domain=QueryDomain.SALES.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[BreakdownResult]",
            read_only=True,
            supports_filters=["warehouse_id", "start_date", "end_date", "as_of_date", "currency"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.sales",
        ),
    )
    registry.register(
        tool_name="get_sales_by_warehouse",
        handler=sales.get_sales_by_warehouse,
        metadata=ToolMetadata(
            tool_name="get_sales_by_warehouse",
            display_name="Sales by Warehouse",
            description="Calculates commercial revenue breakdown across fulfillment distribution centers.",
            domain=QueryDomain.SALES.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[BreakdownResult]",
            read_only=True,
            supports_filters=["channel_id", "start_date", "end_date", "as_of_date", "currency"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.sales",
        ),
    )
    registry.register(
        tool_name="get_sales_by_category",
        handler=sales.get_sales_by_category,
        metadata=ToolMetadata(
            tool_name="get_sales_by_category",
            display_name="Sales by Category",
            description="Calculates commercial revenue breakdown across product catalog categories.",
            domain=QueryDomain.SALES.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[BreakdownResult]",
            read_only=True,
            supports_filters=["category_id", "brand", "start_date", "end_date", "as_of_date", "currency"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.sales",
        ),
    )
    registry.register(
        tool_name="get_sales_by_brand",
        handler=sales.get_sales_by_brand,
        metadata=ToolMetadata(
            tool_name="get_sales_by_brand",
            display_name="Sales by Brand",
            description="Calculates commercial revenue breakdown across product catalog brands.",
            domain=QueryDomain.SALES.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[BreakdownResult]",
            read_only=True,
            supports_filters=["category_id", "brand", "start_date", "end_date", "as_of_date", "currency"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.sales",
        ),
    )

    # 2. Financial Tools
    registry.register(
        tool_name="get_revenue_summary",
        handler=financial.get_revenue_summary,
        metadata=ToolMetadata(
            tool_name="get_revenue_summary",
            display_name="Financial Revenue & Reconciliation",
            description="Exposes Phase 6A reconciled revenue, discounts, and source variance.",
            domain=QueryDomain.FINANCIAL.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "channel_id"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.financial.FinancialIntelligenceService",
        ),
    )
    registry.register(
        tool_name="get_margin_summary",
        handler=financial.get_margin_summary,
        metadata=ToolMetadata(
            tool_name="get_margin_summary",
            display_name="Gross Margin & Profitability",
            description="Exposes Phase 6A gross margin, COGS, and gross margin percentage.",
            domain=QueryDomain.FINANCIAL.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "channel_id"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.financial.FinancialIntelligenceService",
        ),
    )
    registry.register(
        tool_name="get_unit_economics",
        handler=financial.get_unit_economics,
        metadata=ToolMetadata(
            tool_name="get_unit_economics",
            display_name="True Unit Economics & Variable Costs",
            description="Exposes Phase 6B unit economics: fulfillment, payment processing, packaging, handling, return fees, and known contribution margin.",
            domain=QueryDomain.FINANCIAL.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "channel_id"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.financial.UnitEconomicsService",
        ),
    )
    registry.register(
        tool_name="get_margin_drivers",
        handler=financial.get_margin_drivers,
        metadata=ToolMetadata(
            tool_name="get_margin_drivers",
            display_name="Commercial Margin Waterfall",
            description="Exposes Phase 6C margin waterfall stages and erosion drivers.",
            domain=QueryDomain.FINANCIAL.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[BreakdownResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.financial.margin_drivers",
        ),
    )
    registry.register(
        tool_name="get_operational_economics",
        handler=financial.get_operational_economics,
        metadata=ToolMetadata(
            tool_name="get_operational_economics",
            display_name="Operational Economics & Cost Completeness",
            description="Exposes Phase 6D operational cost allocations and comprehensive cost completeness report.",
            domain=QueryDomain.FINANCIAL.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.financial.OperationalEconomicsService",
        ),
    )
    registry.register(
        tool_name="get_profitability_attribution",
        handler=financial.get_profitability_attribution,
        metadata=ToolMetadata(
            tool_name="get_profitability_attribution",
            display_name="Profitability Attribution Profiles",
            description="Exposes Phase 6C SKU profitability profiles and contribution margin gaps.",
            domain=QueryDomain.FINANCIAL.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.financial.ProfitabilityAttributionService",
        ),
    )

    # 3. Inventory Tools
    registry.register(
        tool_name="get_inventory_summary",
        handler=inventory.get_inventory_summary,
        metadata=ToolMetadata(
            tool_name="get_inventory_summary",
            display_name="Inventory Summary & Valuation",
            description="Computes executive inventory KPIs: On-Hand, On-Order, Reserved, Net Position, and Valuation.",
            domain=QueryDomain.INVENTORY.value,
            input_schema={"properties": {"context": "QueryContext", "inventory_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "category_id", "brand"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.inventory.position",
        ),
    )
    registry.register(
        tool_name="get_inventory_position",
        handler=inventory.get_inventory_position,
        metadata=ToolMetadata(
            tool_name="get_inventory_position",
            display_name="Inventory Position Grid",
            description="Exposes detailed tabular stock positions across SKU x Warehouse combinations.",
            domain=QueryDomain.INVENTORY.value,
            input_schema={"properties": {"context": "QueryContext", "inventory_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.inventory.position",
        ),
    )
    registry.register(
        tool_name="get_stockout_risk",
        handler=inventory.get_stockout_risk,
        metadata=ToolMetadata(
            tool_name="get_stockout_risk",
            display_name="Stockout Risk Diagnostics",
            description="Identifies items at critical or high stockout risk with runout and days of supply metrics.",
            domain=QueryDomain.INVENTORY.value,
            input_schema={"properties": {"context": "QueryContext", "inventory_df": "DataFrame", "products_df": "DataFrame", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["as_of_date", "sku_id", "warehouse_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.inventory.risk",
        ),
    )
    registry.register(
        tool_name="get_slow_moving_inventory",
        handler=inventory.get_slow_moving_inventory,
        metadata=ToolMetadata(
            tool_name="get_slow_moving_inventory",
            display_name="Slow Moving Inventory & Excess",
            description="Identifies slow-moving inventory items and capital allocation exposure.",
            domain=QueryDomain.INVENTORY.value,
            input_schema={"properties": {"context": "QueryContext", "inventory_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.inventory",
        ),
    )
    registry.register(
        tool_name="get_high_value_inventory",
        handler=inventory.get_high_value_inventory,
        metadata=ToolMetadata(
            tool_name="get_high_value_inventory",
            display_name="High Value Inventory Concentration",
            description="Identifies inventory items holding significant working capital value.",
            domain=QueryDomain.INVENTORY.value,
            input_schema={"properties": {"context": "QueryContext", "inventory_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.inventory",
        ),
    )
    registry.register(
        tool_name="get_inventory_risk",
        handler=inventory.get_inventory_risk,
        metadata=ToolMetadata(
            tool_name="get_inventory_risk",
            display_name="Inventory Risk Tier Breakdown",
            description="Categorical breakdown of portfolio across Critical Stockout, Understock, Healthy, and Overstock tiers.",
            domain=QueryDomain.INVENTORY.value,
            input_schema={"properties": {"context": "QueryContext", "inventory_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[BreakdownResult]",
            read_only=True,
            supports_filters=["as_of_date", "sku_id", "warehouse_id"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.inventory.risk",
        ),
    )
    registry.register(
        tool_name="get_inventory_risk_breakdown",
        handler=inventory.get_inventory_risk,
        metadata=ToolMetadata(
            tool_name="get_inventory_risk_breakdown",
            display_name="Inventory Risk Tier Breakdown",
            description="Categorical breakdown of portfolio across Critical Stockout, Understock, Healthy, and Overstock tiers.",
            domain=QueryDomain.INVENTORY.value,
            input_schema={"properties": {"context": "QueryContext", "inventory_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[BreakdownResult]",
            read_only=True,
            supports_filters=["as_of_date", "sku_id", "warehouse_id"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.inventory.risk",
        ),
    )

    # 4. Demand Tools
    registry.register(
        tool_name="get_demand_summary",
        handler=demand.get_demand_summary,
        metadata=ToolMetadata(
            tool_name="get_demand_summary",
            display_name="Demand Volume & Breadth Summary",
            description="Exposes aggregate demand volume, daily demand velocity, and active SKU catalog breadth.",
            domain=QueryDomain.DEMAND.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "start_date", "end_date", "sku_id", "warehouse_id", "channel_id"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.analytics.demand",
        ),
    )
    registry.register(
        tool_name="get_demand_trend",
        handler=demand.get_demand_trend,
        metadata=ToolMetadata(
            tool_name="get_demand_trend",
            display_name="Demand Volume Trend",
            description="Generates demand unit volume time series across daily, weekly, or monthly grain.",
            domain=QueryDomain.DEMAND.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[TimeSeriesResult]",
            read_only=True,
            supports_filters=["as_of_date", "start_date", "end_date", "sku_id", "warehouse_id", "time_grain"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.analytics.demand",
        ),
    )
    registry.register(
        tool_name="get_demand_profile",
        handler=demand.get_demand_profile,
        metadata=ToolMetadata(
            tool_name="get_demand_profile",
            display_name="Demand Intermittency Profiles",
            description="Calculates SKU-level demand intermittency, ADI, CV2, and demand categorization.",
            domain=QueryDomain.DEMAND.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["as_of_date", "sku_id", "warehouse_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.analytics.features",
        ),
    )
    registry.register(
        tool_name="get_abc_xyz_distribution",
        handler=demand.get_abc_xyz_distribution,
        metadata=ToolMetadata(
            tool_name="get_abc_xyz_distribution",
            display_name="ABC-XYZ 9-Box Matrix",
            description="Categorical breakdown across ABC (revenue contribution) and XYZ (demand volatility) segments.",
            domain=QueryDomain.DEMAND.value,
            input_schema={"properties": {"context": "QueryContext", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[BreakdownResult]",
            read_only=True,
            supports_filters=["as_of_date", "sku_id", "warehouse_id"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.analytics.abc_xyz",
        ),
    )

    # 5. Forecasting Tools
    registry.register(
        tool_name="get_forecast",
        handler=forecasting.get_forecast,
        metadata=ToolMetadata(
            tool_name="get_forecast",
            display_name="Demand Forecast Projection",
            description="Queries forward-looking demand forecast time series without data fabrication.",
            domain=QueryDomain.FORECASTING.value,
            input_schema={"properties": {"context": "QueryContext", "forecast_df": "DataFrame", "forecast_output": "ForecastOutput"}},
            output_schema="QueryResponse[TimeSeriesResult]",
            read_only=True,
            supports_filters=["sku_id", "warehouse_id", "time_grain"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.forecasting",
        ),
    )
    registry.register(
        tool_name="get_forecast_accuracy",
        handler=forecasting.get_forecast_accuracy,
        metadata=ToolMetadata(
            tool_name="get_forecast_accuracy",
            display_name="Forecast Accuracy Metrics",
            description="Queries standard forecasting accuracy error metrics: WAPE, MAE, RMSE, MAPE.",
            domain=QueryDomain.FORECASTING.value,
            input_schema={"properties": {"context": "QueryContext", "actual_series": "list", "forecast_series": "list"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["sku_id", "warehouse_id"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.forecasting.evaluation",
        ),
    )
    registry.register(
        tool_name="get_forecast_bias",
        handler=forecasting.get_forecast_bias,
        metadata=ToolMetadata(
            tool_name="get_forecast_bias",
            display_name="Forecast Bias Direction",
            description="Evaluates forecast bias direction (over-forecasting vs under-forecasting).",
            domain=QueryDomain.FORECASTING.value,
            input_schema={"properties": {"context": "QueryContext", "actual_series": "list", "forecast_series": "list"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["sku_id", "warehouse_id"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.analytics.metrics",
        ),
    )
    registry.register(
        tool_name="get_forecast_summary",
        handler=forecasting.get_forecast_summary,
        metadata=ToolMetadata(
            tool_name="get_forecast_summary",
            display_name="Forecast Model Summary",
            description="Summarizes active forecasting model name and planning horizon length.",
            domain=QueryDomain.FORECASTING.value,
            input_schema={"properties": {"context": "QueryContext", "forecast_output": "ForecastOutput"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["sku_id", "warehouse_id"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.forecasting",
        ),
    )

    # 6. Returns Tools
    registry.register(
        tool_name="get_return_summary",
        handler=returns.get_return_summary,
        metadata=ToolMetadata(
            tool_name="get_return_summary",
            display_name="Customer Returns Summary",
            description="Computes executive return KPIs: return events, returned units, unit return rate %, revenue return rate %, returned value.",
            domain=QueryDomain.RETURNS.value,
            input_schema={"properties": {"context": "QueryContext", "returns_df": "DataFrame", "sales_df": "DataFrame", "products_df": "DataFrame"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "start_date", "end_date", "sku_id", "warehouse_id", "channel_id"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.returns.ReturnsIntelligenceService",
        ),
    )
    registry.register(
        tool_name="get_return_rate",
        handler=returns.get_return_rate,
        metadata=ToolMetadata(
            tool_name="get_return_rate",
            display_name="Return Rate Analytics",
            description="Exposes unit and revenue return rates with sample-size guard evaluation.",
            domain=QueryDomain.RETURNS.value,
            input_schema={"properties": {"context": "QueryContext", "returns_df": "DataFrame", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "sku_id", "warehouse_id", "channel_id"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.returns",
        ),
    )
    registry.register(
        tool_name="get_return_trend",
        handler=returns.get_return_trend,
        metadata=ToolMetadata(
            tool_name="get_return_trend",
            display_name="Customer Returns Trend",
            description="Generates customer return unit volume time series across daily, weekly, or monthly grain.",
            domain=QueryDomain.RETURNS.value,
            input_schema={"properties": {"context": "QueryContext", "returns_df": "DataFrame"}},
            output_schema="QueryResponse[TimeSeriesResult]",
            read_only=True,
            supports_filters=["as_of_date", "start_date", "end_date", "sku_id", "warehouse_id", "time_grain"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.returns",
        ),
    )
    registry.register(
        tool_name="get_return_anomalies",
        handler=returns.get_return_anomalies,
        metadata=ToolMetadata(
            tool_name="get_return_anomalies",
            display_name="Return Anomaly Diagnostics",
            description="Exposes Phase 5B statistical return anomalies with severity and factual rationales.",
            domain=QueryDomain.RETURNS.value,
            input_schema={"properties": {"context": "QueryContext", "returns_df": "DataFrame", "sales_df": "DataFrame"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["as_of_date", "sku_id", "warehouse_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.returns.anomaly",
        ),
    )
    registry.register(
        tool_name="get_return_risk",
        handler=returns.get_return_risk,
        metadata=ToolMetadata(
            tool_name="get_return_risk",
            display_name="Predictive Return Risk Scores",
            description="Exposes Phase 5C predictive return risk records and calibration tiers.",
            domain=QueryDomain.RETURNS.value,
            input_schema={"properties": {"context": "QueryContext", "return_risk_records": "list"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["sku_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.returns.risk",
        ),
    )
    registry.register(
        tool_name="get_return_reason_breakdown",
        handler=returns.get_return_reason_breakdown,
        metadata=ToolMetadata(
            tool_name="get_return_reason_breakdown",
            display_name="Return Reasons Breakdown",
            description="Categorical breakdown of return events across customer reasons (Defective, Wrong Size, etc.).",
            domain=QueryDomain.RETURNS.value,
            input_schema={"properties": {"context": "QueryContext", "returns_df": "DataFrame"}},
            output_schema="QueryResponse[BreakdownResult]",
            read_only=True,
            supports_filters=["as_of_date", "sku_id", "warehouse_id", "channel_id"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.returns",
        ),
    )

    # 7. Operations Tools
    registry.register(
        tool_name="get_replenishment_reviews",
        handler=operations.get_replenishment_reviews,
        metadata=ToolMetadata(
            tool_name="get_replenishment_reviews",
            display_name="Replenishment Review Candidates",
            description="Read-only view of Phase 4B-1 replenishment recommendations. Pure review access without execution.",
            domain=QueryDomain.OPERATIONS.value,
            input_schema={"properties": {"context": "QueryContext", "replenishment_result": "ReplenishmentResult"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["sku_id", "warehouse_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.recommendations.replenishment",
        ),
    )
    registry.register(
        tool_name="get_purchase_order_reviews",
        handler=operations.get_purchase_order_reviews,
        metadata=ToolMetadata(
            tool_name="get_purchase_order_reviews",
            display_name="Draft Purchase Order Proposal Reviews",
            description="Read-only view of Phase 4B-2 consolidated purchase order proposals. Pure review access without submission.",
            domain=QueryDomain.OPERATIONS.value,
            input_schema={"properties": {"context": "QueryContext", "po_result": "PurchaseOrderProposalResult"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["warehouse_id", "supplier_id", "currency", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.recommendations.purchase_orders",
        ),
    )
    registry.register(
        tool_name="get_warehouse_rebalancing_reviews",
        handler=operations.get_warehouse_rebalancing_reviews,
        metadata=ToolMetadata(
            tool_name="get_warehouse_rebalancing_reviews",
            display_name="Warehouse Stock Rebalancing Reviews",
            description="Read-only view of Phase 4C inter-warehouse stock transfer candidates. Pure review access without transfer execution.",
            domain=QueryDomain.OPERATIONS.value,
            input_schema={"properties": {"context": "QueryContext", "rebalancing_result": "WarehouseRebalancingResult"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["sku_id", "warehouse_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=False,
            source_engine="commerce_ai.recommendations.warehouse_rebalancing",
        ),
    )

    # 8. Business Impact Tools
    registry.register(
        tool_name="get_business_impact_summary",
        handler=impact.get_business_impact_summary,
        metadata=ToolMetadata(
            tool_name="get_business_impact_summary",
            display_name="Business Impact Portfolio Summary",
            description="Exposes Phase 6F quantified exposures: gross signal exposure, deduplicated exposure, and physical capital exposure.",
            domain=QueryDomain.IMPACT.value,
            input_schema={"properties": {"context": "QueryContext", "impact_result": "BusinessImpactResult"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.business_impact.BusinessImpactService",
        ),
    )
    registry.register(
        tool_name="get_business_impact_by_category",
        handler=impact.get_business_impact_by_category,
        metadata=ToolMetadata(
            tool_name="get_business_impact_by_category",
            display_name="Business Impact by Category",
            description="Exposes Phase 6F aggregate exposure breakdown across ImpactCategory (no entity rankings).",
            domain=QueryDomain.IMPACT.value,
            input_schema={"properties": {"context": "QueryContext", "impact_result": "BusinessImpactResult"}},
            output_schema="QueryResponse[BreakdownResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.business_impact",
        ),
    )
    registry.register(
        tool_name="get_business_impact_by_type",
        handler=impact.get_business_impact_by_type,
        metadata=ToolMetadata(
            tool_name="get_business_impact_by_type",
            display_name="Business Impact by Type",
            description="Exposes Phase 6F aggregate exposure breakdown across ImpactType (Observed Loss, At-Risk Revenue, etc.).",
            domain=QueryDomain.IMPACT.value,
            input_schema={"properties": {"context": "QueryContext", "impact_result": "BusinessImpactResult"}},
            output_schema="QueryResponse[BreakdownResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.business_impact",
        ),
    )
    registry.register(
        tool_name="get_physical_capital_exposure",
        handler=impact.get_physical_capital_exposure,
        metadata=ToolMetadata(
            tool_name="get_physical_capital_exposure",
            display_name="Physical Capital Exposure",
            description="Exposes Phase 6F deduplicated physical capital exposure KPI.",
            domain=QueryDomain.IMPACT.value,
            input_schema={"properties": {"context": "QueryContext", "impact_result": "BusinessImpactResult"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.business_impact",
        ),
    )

    # 9. Recommendation Tools
    registry.register(
        tool_name="get_recommendation_summary",
        handler=recommendations.get_recommendation_summary,
        metadata=ToolMetadata(
            tool_name="get_recommendation_summary",
            display_name="Business Recommendations Summary",
            description="Exposes Phase 6G recommendation counts, priority breakdown, and governance verification.",
            domain=QueryDomain.RECOMMENDATIONS.value,
            input_schema={"properties": {"context": "QueryContext", "recommendation_result": "BusinessRecommendationResult"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "channel_id"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.recommendations.business_recommendations",
        ),
    )
    registry.register(
        tool_name="get_recommendations",
        handler=recommendations.get_recommendations,
        metadata=ToolMetadata(
            tool_name="get_recommendations",
            display_name="Business Recommendations List",
            description="Exposes Phase 6G filterable recommendations (strictly DRAFT, approval_required=True, execution_allowed=False).",
            domain=QueryDomain.RECOMMENDATIONS.value,
            input_schema={"properties": {"context": "QueryContext", "recommendation_result": "BusinessRecommendationResult"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "channel_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.recommendations.business_recommendations",
        ),
    )
    registry.register(
        tool_name="get_recommendation_list",
        handler=recommendations.get_recommendations,
        metadata=ToolMetadata(
            tool_name="get_recommendation_list",
            display_name="Business Recommendations List",
            description="Exposes Phase 6G filterable recommendations (strictly DRAFT, approval_required=True, execution_allowed=False).",
            domain=QueryDomain.RECOMMENDATIONS.value,
            input_schema={"properties": {"context": "QueryContext", "recommendation_result": "BusinessRecommendationResult"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "channel_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.recommendations.business_recommendations",
        ),
    )
    registry.register(
        tool_name="get_recommendation_by_id",
        handler=recommendations.get_recommendation_by_id,
        metadata=ToolMetadata(
            tool_name="get_recommendation_by_id",
            display_name="Recommendation Inspection",
            description="Exposes full audit detail, rationale, and evidence lineage for a specific recommendation.",
            domain=QueryDomain.RECOMMENDATIONS.value,
            input_schema={"properties": {"context": "QueryContext", "recommendation_id": "str", "recommendation_result": "BusinessRecommendationResult"}},
            output_schema="QueryResponse[InsightResult]",
            read_only=True,
            supports_filters=["as_of_date"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.recommendations.business_recommendations",
        ),
    )

    # 10. Decision Tools
    registry.register(
        tool_name="get_decision_summary",
        handler=decisions.get_decision_summary,
        metadata=ToolMetadata(
            tool_name="get_decision_summary",
            display_name="Decision Packages Summary",
            description="Exposes Phase 6H decision packages summary, governance verification (selected_option=None), and risk breakdown.",
            domain=QueryDomain.DECISIONS.value,
            input_schema={"properties": {"context": "QueryContext", "decision_result": "DecisionIntelligenceResult"}},
            output_schema="QueryResponse[MetricResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.decision_intelligence.DecisionIntelligenceService",
        ),
    )
    registry.register(
        tool_name="get_decision_packages",
        handler=decisions.get_decision_packages,
        metadata=ToolMetadata(
            tool_name="get_decision_packages",
            display_name="Decision Packages List",
            description="Exposes Phase 6H filterable decision packages (strictly PENDING_REVIEW, selected_option=None).",
            domain=QueryDomain.DECISIONS.value,
            input_schema={"properties": {"context": "QueryContext", "decision_result": "DecisionIntelligenceResult"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.decision_intelligence",
        ),
    )
    registry.register(
        tool_name="get_decision_list",
        handler=decisions.get_decision_packages,
        metadata=ToolMetadata(
            tool_name="get_decision_list",
            display_name="Decision Packages List",
            description="Exposes Phase 6H filterable decision packages (strictly PENDING_REVIEW, selected_option=None).",
            domain=QueryDomain.DECISIONS.value,
            input_schema={"properties": {"context": "QueryContext", "decision_result": "DecisionIntelligenceResult"}},
            output_schema="QueryResponse[TableResult]",
            read_only=True,
            supports_filters=["as_of_date", "currency", "sku_id", "warehouse_id", "limit", "offset"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.decision_intelligence",
        ),
    )
    registry.register(
        tool_name="get_decision_package_by_id",
        handler=decisions.get_decision_package_by_id,
        metadata=ToolMetadata(
            tool_name="get_decision_package_by_id",
            display_name="Decision Package Inspection",
            description="Exposes detailed candidate options, trade-offs, and risk flags for a specific decision package.",
            domain=QueryDomain.DECISIONS.value,
            input_schema={"properties": {"context": "QueryContext", "decision_id": "str", "decision_result": "DecisionIntelligenceResult"}},
            output_schema="QueryResponse[TableResult, InsightResult]",
            read_only=True,
            supports_filters=["as_of_date"],
            supports_as_of_date=True,
            supports_currency=True,
            source_engine="commerce_ai.decision_intelligence",
        ),
    )


# Automatically populate default registry
register_standard_tools(default_registry)
