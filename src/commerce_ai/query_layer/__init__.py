"""Business Intelligence Tool & Dashboard Data Layer (Phase 7A).

Provides a unified, read-only, point-in-time safe, and currency-isolated query layer:
- QueryContext: Standardized parameter and temporal filter specification
- Schemas: Standard response contracts (MetricResult, TimeSeriesResult, BreakdownResult, TableResult, InsightResult, QueryResponse)
- Registry: ToolRegistry with complete metadata for dashboard routing and future eRetail Copilot tool execution
- Service: QueryLayerService orchestrating underlying datasets and intelligence outputs
- Tools: Modular read-only functions across Sales, Financial, Inventory, Demand, Forecasting, Returns, Operations, Impact, Recommendations, and Decisions.
"""

from __future__ import annotations

from commerce_ai.query_layer.context import QueryContext
from commerce_ai.query_layer.schemas import (
    BreakdownItem,
    BreakdownResult,
    CalculationStatus,
    EvidenceReference,
    InsightResult,
    MetricResult,
    QueryDomain,
    QueryMetadata,
    QueryResponse,
    TableResult,
    TimeSeriesPoint,
    TimeSeriesResult,
)
from commerce_ai.query_layer.registry import (
    ToolMetadata,
    ToolRegistry,
    default_registry,
)
from commerce_ai.query_layer.service import QueryLayerService

# Domain Tools
from commerce_ai.query_layer.sales import (
    get_sales_by_brand,
    get_sales_by_category,
    get_sales_by_channel,
    get_sales_by_sku,
    get_sales_by_warehouse,
    get_sales_summary,
    get_sales_trend,
)
from commerce_ai.query_layer.financial import (
    get_margin_drivers,
    get_margin_summary,
    get_operational_economics,
    get_profitability_attribution,
    get_revenue_summary,
    get_unit_economics,
)
from commerce_ai.query_layer.inventory import (
    get_high_value_inventory,
    get_inventory_position,
    get_inventory_risk,
    get_inventory_summary,
    get_slow_moving_inventory,
    get_stockout_risk,
)
from commerce_ai.query_layer.demand import (
    get_abc_xyz_distribution,
    get_demand_profile,
    get_demand_summary,
    get_demand_trend,
)
from commerce_ai.query_layer.forecasting import (
    get_forecast,
    get_forecast_accuracy,
    get_forecast_bias,
    get_forecast_summary,
)
from commerce_ai.query_layer.returns import (
    get_return_anomalies,
    get_return_rate,
    get_return_reason_breakdown,
    get_return_risk,
    get_return_summary,
    get_return_trend,
)
from commerce_ai.query_layer.operations import (
    get_purchase_order_reviews,
    get_replenishment_reviews,
    get_warehouse_rebalancing_reviews,
)
from commerce_ai.query_layer.impact import (
    get_business_impact_by_category,
    get_business_impact_by_type,
    get_business_impact_summary,
    get_physical_capital_exposure,
)
from commerce_ai.query_layer.recommendations import (
    get_recommendation_by_id,
    get_recommendation_summary,
    get_recommendations,
)
from commerce_ai.query_layer.decisions import (
    get_decision_package_by_id,
    get_decision_packages,
    get_decision_summary,
)

__all__ = [
    # Context & Schemas
    "QueryContext",
    "CalculationStatus",
    "QueryDomain",
    "QueryMetadata",
    "EvidenceReference",
    "MetricResult",
    "TimeSeriesPoint",
    "TimeSeriesResult",
    "BreakdownItem",
    "BreakdownResult",
    "TableResult",
    "InsightResult",
    "QueryResponse",
    # Registry & Service
    "ToolMetadata",
    "ToolRegistry",
    "default_registry",
    "QueryLayerService",
    # Sales
    "get_sales_summary",
    "get_sales_trend",
    "get_sales_by_sku",
    "get_sales_by_channel",
    "get_sales_by_warehouse",
    "get_sales_by_category",
    "get_sales_by_brand",
    # Financial
    "get_revenue_summary",
    "get_margin_summary",
    "get_unit_economics",
    "get_margin_drivers",
    "get_operational_economics",
    "get_profitability_attribution",
    # Inventory
    "get_inventory_summary",
    "get_inventory_position",
    "get_stockout_risk",
    "get_slow_moving_inventory",
    "get_high_value_inventory",
    "get_inventory_risk",
    # Demand
    "get_demand_summary",
    "get_demand_trend",
    "get_demand_profile",
    "get_abc_xyz_distribution",
    # Forecasting
    "get_forecast",
    "get_forecast_accuracy",
    "get_forecast_bias",
    "get_forecast_summary",
    # Returns
    "get_return_summary",
    "get_return_rate",
    "get_return_trend",
    "get_return_anomalies",
    "get_return_risk",
    "get_return_reason_breakdown",
    # Operations
    "get_replenishment_reviews",
    "get_purchase_order_reviews",
    "get_warehouse_rebalancing_reviews",
    # Impact
    "get_business_impact_summary",
    "get_business_impact_by_category",
    "get_business_impact_by_type",
    "get_physical_capital_exposure",
    # Recommendations
    "get_recommendation_summary",
    "get_recommendations",
    "get_recommendation_by_id",
    # Decisions
    "get_decision_summary",
    "get_decision_packages",
    "get_decision_package_by_id",
]
