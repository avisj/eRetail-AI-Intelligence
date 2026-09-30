# Business Intelligence Tool & Dashboard Layer (Phase 7A)

> [!IMPORTANT]
> **Core Architectural Principles**:
> Phase 7A is deterministic, read-only, source-engine orchestration, structured query responses, dashboard-ready, and Copilot-ready.
> Phase 7A is NOT an LLM, an agent, RAG, autonomous execution, PO execution, inventory transfer execution, or pricing automation.
> It exposes structured, point-in-time compliant query tools across all 10 platform domains without data mutation or competitive rankings.

---

## 1. Objective & Product Context

Phase 7A provides the foundational data and tool execution layer that powers:
1. **Executive Dashboards**: Point-in-time metrics, KPI cards, and trend visualizations.
2. **Analysis Workspace**: Slice-and-dice dimensional breakdowns, table exploration, and anomaly diagnostics.
3. **eRetail Copilot**: Machine-readable JSON function declarations enabling conversational AI agents to interrogate platform intelligence safely.
4. **Future REST / GraphQL APIs**: Standardized, deterministic response envelopes.

All calculations orchestrate existing platform services (Phases 1 through 6H) without duplication of business logic, formulas, or statistical algorithms.

---

## 2. Architecture & Pipeline Topology

The Query Layer sits between downstream consumers (Dashboards, Copilot, APIs) and upstream analytical engines:

    +-----------------------------------------------------------------+
    |                 Downstream Consumers                            |
    |   [eRetail Copilot]   [Executive Dashboards]   [Analysis API]   |
    +-----------------------------------------------------------------+
                                     |
                                     v
    +-----------------------------------------------------------------+
    |                Phase 7A: Query Layer Service                    |
    |                                                                 |
    |   [QueryContext]                 [ToolRegistry]                 |
    |     - as_of_date anti-leakage       - 46 canonical tools        |
    |     - currency isolation            - 3 registered aliases      |
    |     - dimensional filters           - 49 Copilot declarations   |
    |                                     - read-only verification    |
    |                                                                 |
    |   Standardized Response Contracts:                              |
    |     QueryResponse -> MetricResult | TimeSeriesResult |          |
    |                      BreakdownResult | TableResult | Insight    |
    +-----------------------------------------------------------------+
                                     |
        +----------------------------+----------------------------+
        |                            |                            |
        v                            v                            v
    [Commercial Domains]        [Operations Domains]     [Governance Domains]
    - Sales Intelligence        - Inventory Position     - Business Impact (6F)
    - Financial & Margins (6A)  - Replenishment (4B)     - Recommendations (6G)
    - Unit Economics (6B)       - Rebalancing (4C)       - Decision Support (6H)
    - Margin Drivers (6C)       - Demand & ABC/XYZ (2)   
    - Operational Econ (6D)     - Forecasting (3)        
    - Returns Analytics (5A-C)  - PO Proposals (4B-2)    

---

## 3. Core Governance Guarantees

### 3.1 Strict Read-Only Access
All 46 registered canonical query tools and 3 convenience aliases are certified read-only (`is_read_only=True`). The `ToolRegistry` explicitly rejects registration of mutating tools. No database updates, file modifications, PO creations, or warehouse transfer executions occur within Phase 7A.

### 3.2 Point-in-Time Anti-Leakage
Every query accepts a validated `QueryContext` containing an `as_of_date`. The filtering subsystem (`commerce_ai.query_layer.filters`):
- Restricts all historical dataframes strictly to `date <= as_of_date`.
- Caps any user-provided `end_date` at `as_of_date` to prevent forward-looking data leakage.
- Validates chronological sanity (`start_date <= end_date <= as_of_date`).

### 3.3 Currency Isolation
To prevent meaningless arithmetic across mixed currencies:
- Multi-currency datasets require explicit currency filtering in `QueryContext`.
- If unisolated currencies are detected, tools return a structured `CURRENCY_INCONSISTENCY` error rather than computing corrupted totals.
- Every monetary metric and breakdown item is explicitly tagged with its ISO-4217 currency code.

### 3.4 Zero Individual Entity Rankings
In alignment with platform governance rules, Phase 7A strictly prohibits "Top 10 opportunities", winner declarations, or competitive entity rankings. All outputs present:
- Portfolio aggregate distributions (by category, type, severity, status).
- Complete unranked entity tables with deterministic pagination (`limit`, `offset`).
- Non-causal, objective analytical metrics.

---

## 4. Standardized Response Contracts

All tools return a consistent [`QueryResponse`](file:///src/commerce_ai/query_layer/schemas.py) envelope containing:

| Field | Type | Description |
| :--- | :--- | :--- |
| `query_id` | `str` | Deterministic SHA-256 fingerprint of tool name and parameters. |
| `tool_name` | `str` | Registered name of the executing query tool. |
| `domain` | `QueryDomain` | Functional domain classification enum. |
| `status` | `CalculationStatus` | Execution state: `SUCCESS`, `PARTIAL`, `EMPTY`, `UNAVAILABLE`, `ERROR`. |
| `metadata` | `QueryMetadata` | Audit metadata including execution latency, data version, and provenance. |
| `metrics` | `List[MetricResult]` | Discrete KPI metrics designed for scorecard display. |
| `time_series` | `Optional[TimeSeriesResult]` | Temporal sequence for trend charts. |
| `breakdown` | `Optional[BreakdownResult]` | Categorical/dimensional segmentation items. |
| `table` | `Optional[TableResult]` | Tabular data with column schemas, rows, and pagination. |
| `insight` | `Optional[InsightResult]` | Structured factual observations with evidence references. |
| `error_message`| `Optional[str]` | Human-readable explanation in case of failure. |

---

## 5. Domain Query Tools Catalog

The platform registers **46 canonical query tools** across 10 functional domains, plus **3 convenience aliases**, providing **49 total Copilot tool declarations**:

### Domain 1: Sales & Commercial Intelligence (7 tools)
- `get_sales_summary`: Gross revenue, discounts, net revenue, units sold, order count, and AOV.
- `get_sales_trend`: Historical daily, weekly, or monthly net revenue time series.
- `get_sales_by_sku`: Paginated performance table across all active SKUs.
- `get_sales_by_channel`: Net revenue and unit distribution across sales channels.
- `get_sales_by_warehouse`: Commercial fulfilment distribution across fulfillment centers.
- `get_sales_by_category`: Revenue distribution across merchandise product categories.
- `get_sales_by_brand`: Revenue breakdown across portfolio brands.

### Domain 2: Financial Intelligence & Unit Economics (6 tools)
- `get_revenue_summary`: Portfolio revenue reconciliation and discount exposure.
- `get_margin_summary`: COGS, gross margin, gross margin %, and commercial loss transactions.
- `get_unit_economics`: Net revenue, product cost, shipping, return, and variable cost breakdown.
- `get_margin_drivers`: Multi-stage margin waterfall (Gross -> Net -> Gross Margin -> Contribution Margin).
- `get_operational_economics`: Variable fulfillment expenses and cost completeness audit scores.
- `get_profitability_attribution`: Price, volume, cost, and mix variance margin drivers.

### Domain 3: Inventory Position & Risk (6 tools)
- `get_inventory_summary`: Total stock on hand, units, inventory capital, and active SKU counts.
- `get_inventory_position`: Paginated stock position table with days-of-supply metrics.
- `get_stockout_risk`: Inventory items with low coverage or projected runout.
- `get_slow_moving_inventory`: Capital tied up in slow-moving and aged inventory tiers.
- `get_high_value_inventory`: Capital exposure in high-unit-cost merchandise.
- `get_inventory_risk_breakdown`: Aggregate distribution of inventory risk categories.

### Domain 4: Demand Intelligence (4 tools)
- `get_demand_summary`: Network demand velocity, daily average units, and peak volume.
- `get_demand_trend`: Historical unit demand volume aggregated by chosen time grain.
- `get_demand_profile`: Intermittency classification (ADI, CV2, Smooth, Intermittent, Lumpy, Erratic).
- `get_abc_xyz_distribution`: SKU count distribution across ABC revenue and XYZ variability segments.

### Domain 5: Forecasting & Accuracy (4 tools)
- `get_forecast`: Forward-looking demand forecast time series. Returns `UNAVAILABLE` if models not yet run.
- `get_forecast_accuracy`: Historical forecast accuracy metrics (MAE, RMSE, MAPE, WAPE).
- `get_forecast_bias`: Forecast tracking signal, mean error, and systematic over/under-bias direction.
- `get_forecast_summary`: Portfolio-level forecast summary metrics.

### Domain 6: Returns Intelligence (6 tools)
- `get_return_summary`: Return order count, returned units, return rate %, and return value exposure.
- `get_return_rate`: Unit and revenue return rates with benchmark comparisons.
- `get_return_trend`: Historical return volume and value trend over time.
- `get_return_anomalies`: Statistically anomalous return spikes and affected entities.
- `get_return_risk`: Risk scoring distribution across products and return categories.
- `get_return_reason_breakdown`: Aggregate categorical distribution of reported return reasons.

### Domain 7: Operational Economics & Proposals (3 tools)
- `get_replenishment_reviews`: Solved replenishment proposals and suggested order quantities.
- `get_purchase_order_reviews`: Draft purchase order proposals with supplier constraints.
- `get_warehouse_rebalancing_reviews`: Inter-facility inventory transfer proposals.

### Domain 8: Business Impact Quantification (4 tools)
- `get_business_impact_summary`: Gross signal exposure, deduplicated exposure, and physical capital exposure.
- `get_business_impact_by_category`: Quantified monetary exposure distributed across Impact Categories.
- `get_business_impact_by_type`: Exposure distribution across Impact Types (At-Risk Revenue, Capital Tied Up, etc.).
- `get_physical_capital_exposure`: Deduplicated physical inventory capital at risk across overlapping signals.

### Domain 9: Business Recommendations (Read-Only) (3 tools)
- `get_recommendation_summary`: Recommendation counts, priority distributions, and review requirements (100%).
- `get_recommendations`: Filterable tabular recommendation list with rule IDs and evidence.
- `get_recommendation_by_id`: Comprehensive recommendation detail package.

### Domain 10: Decision Intelligence & Support (Read-Only) (3 tools)
- `get_decision_summary`: Decision package counts, pending review status (100%), and unselected winners (100%).
- `get_decision_packages`: Paginated decision packages containing candidate options and trade-offs.
- `get_decision_package_by_id`: Full decision package detail including qualitative risk flags and evidence links.

### Registered Convenience Aliases (3 tools)
- `get_inventory_risk`: Alias pointing to canonical `get_inventory_risk_breakdown`.
- `get_recommendation_list`: Alias pointing to canonical `get_recommendations`.
- `get_decision_list`: Alias pointing to canonical `get_decision_packages`.

---

## 6. Integration Guide: eRetail Copilot

Every registered tool automatically generates a JSON Schema declaration compatible with LLM function calling via `registry.to_copilot_declarations()`:

    tool_def = {
        "name": "get_sales_summary",
        "description": "Retrieve executive-level sales and commercial KPIs...",
        "parameters": {
            "type": "object",
            "properties": {
                "as_of_date": {"type": "string", "description": "ISO cutoff date"},
                "currency": {"type": "string", "description": "ISO currency code"},
                "channel_id": {"type": "string", "description": "Channel filter"},
                "warehouse_id": {"type": "string", "description": "Warehouse filter"}
            },
            "required": []
        }
    }

When the Copilot model invokes a function, `QueryLayerService.execute_tool(name, context)` parses the arguments into a validated `QueryContext`, applies anti-leakage filters, executes the deterministic domain logic, and returns the machine-readable `QueryResponse`.

---

## 7. Verification & Quality Assurance

Phase 7A has undergone full verification:
- **73 dedicated unit and integration tests** across 4 test suites:
  - `tests/test_query_context.py`: Parameter validation, chronology, pagination, extra attribute rejection.
  - `tests/test_query_filters.py`: Anti-leakage date filtering, datetime handling, currency isolation.
  - `tests/test_query_tools.py`: All 46 canonical domain tools tested against schema contracts.
  - `tests/test_query_layer.py`: ToolRegistry, dynamic dispatch, aliases, copilot declarations, immutability.
- **780 total regression tests** passing across the full platform repository (Phases 1 through 7A).
- Zero automated mutations, zero ranked opportunities, zero causally fabricated claims.

---

## 8. Platform Roadmap (Phases 7A through 9)

The eRetail AI Intelligence evolution is sequenced across the following governed roadmap:

- **Phase 7A: Business Intelligence Tool & Dashboard Data Layer** (Current)
  - Deterministic query layer, point-in-time filtering, currency isolation, 46 canonical tools, 10 domains.
- **Phase 7B: Business Query Contracts**
  - Schema formalization, contract versioning, strict payload verification, API serialization.
- **Phase 7C: Copilot Reasoning & Orchestration**
  - Read-only tool orchestration, multi-turn plan synthesis, query selection and composition.
- **Phase 7D: Business Explanations**
  - Deterministic natural language synthesis grounded strictly in verified query evidence.
- **Phase 7E: Controlled Conversation Context**
  - Audit logging, state preservation, conversation memory, contextual guardrails.
- **Phase 7F: RAG / Business Knowledge**
  - Grounding in business policies, vendor contracts, operational SLAs, and domain definitions.
- **Phase 7G: Dashboard + UI Platform**
  - Web frontend, interactive KPI dashboards, Analysis Workspace, visual analytics.
- **Phase 8: Agentic Workflows**
  - Multi-step cross-functional workflow planning and human approval packaging.
- **Phase 9: Controlled AI Execution**
  - Gated action execution, rollback mechanisms, policy assertions, enterprise audit trails.

