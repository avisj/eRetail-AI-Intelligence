# Development Roadmap: AI Commerce Intelligence Platform

This roadmap defines the engineering milestones, sequencing, and capabilities across all development phases.

---

## Phase Matrix

| Phase | Milestone | Status | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Foundation & Architecture** | **Completed** | Project structure, clean architecture boundaries, documentation, virtual environment, and baseline configuration. |
| **Phase 1** | **Data Layer & Contracts** | **Completed** | Standard Commerce Data Contract (8 entities), Pydantic validation suite, realistic synthetic data generator, CSV loaders, quality reports, and automated test suite. |
| **Phase 2** | **Feature Engineering & Analytics** | **Completed** | Demand reconstruction grid, stockout detection & masking, ABC/XYZ portfolio classification, cyclical calendar & retail events, leakage-free lag/rolling features, intermittency modeling, and forecast error metrics. |
| **Phase 3** | **Forecasting Engine** | **Completed** | Universal `ForecastModel` interface, standardized `ForecastRecord`/`ForecastOutput` contracts, baselines (Naive, Seasonal Naive, Moving Average, Holt-Winters, Croston SBA), tabular ML (LightGBM), zero-shot foundation adapter (TimesFM with Darwin fallback), rolling-origin cross-validation backtester, hierarchical evaluation (WAPE, Bias, MAE, RMSE), automated model selection with parsimony margin, and unified `ForecastService`. |
| **Phase 4** | **Inventory & Decision Engines** | Planned | Probabilistic stockout prediction, multi-echelon safety stock calculation, automated replenishment & purchase order proposal generator, and warehouse rebalancing solver. |
| **Phase 5** | **AI Agent Ecosystem & Copilot** | Planned | Multi-agent framework (Data, Forecast, Inventory, Purchase, Warehouse, Returns, Profit, Knowledge, Orchestrator agents), RAG knowledge layer, and natural-language Copilot. |
| **Phase 6** | **Integrations & Safe Execution** | Planned | Pluggable connectors (Shopify, SAP, Oracle, eRetail, custom OMS/WMS), automated draft PO execution, guardrails, and role-based human-in-the-loop approval workflows. |

---

## Detailed Milestone Objectives

### Phase 0: Foundation & Clean Architecture (Delivered)
- Standalone, vendor-neutral directory structure.
- Separation of concerns between deterministic code, ML models, RAG, and LLM agents.
- Virtual environment setup and strict git safety rules.

### Phase 1: Data Contracts, Validation & Realistic Simulation (Delivered)
- Vendor-agnostic standard data contracts for 8 core ecommerce entities.
- Pydantic v2 validation models with strict schema enforcement.
- Realistic synthetic generator featuring weekly/monthly/annual seasonality, promotions, lead times, closed-loop sales-inventory-purchase coupling, and velocity tiers.
- Robust data loaders, structured validation result reporting, and exploratory notebook.

### Phase 2: Feature Engineering & Classical Analytics (Delivered)
- Dynamic ABC (revenue) and XYZ (demand volatility) classification.
- Stockout detection and demand masking (`forecast_training_eligible`).
- Calendar temporal flags, cyclical sine/cosine encoders, and retail promotional event markers.
- Leakage-free lag (1, 7, 14, 28, 30) and causal rolling window aggregations (7, 14, 30, 90).
- Demand intermittency categorization (Smooth, Intermittent, Erratic, Lumpy).

### Phase 3: Demand Forecasting Engine & Benchmarking (Delivered)
- Universal `ForecastModel` interface with standardized outputs (`ForecastRecord`, `ForecastOutput`, `ForecastMetadata`).
- Statistical & heuristic baselines: `NaiveModel`, `SeasonalNaiveModel` (7d), `MovingAverageModel` (7d, 14d, 30d).
- Parametric smoothing: `ExponentialSmoothingModel` (Holt-Winters with prediction intervals).
- Intermittent demand modeling: `CrostonModel` (Classic and Syntetos-Boylan Approximation SBA).
- Tabular Machine Learning: `LightGBMForecastModel` with autoregressive recursive multi-step forecasting and prediction intervals.
- Foundation Model Adapter: `TimesFMForecastModel` with environment discovery, graceful `TIMESFM_UNAVAILABLE` fallback, and unit-testable mock integration.
- Time-series cross-validation: `RollingOriginBacktester` with error isolation and runtime diagnostics.
- Hierarchical evaluation: `ForecastEvaluator` computing MAE, RMSE, zero-safe MAPE, volume-weighted WAPE, Bias %, and coverage across global, SKU, warehouse, and intermittency dimensions.
- Champion selection: `ModelSelector` with parsimony margin and `ModelSelectionPolicy` for segment-specific routing.
- High-level inference API: `ForecastService` supporting single-series and batch execution with automatic policy routing.

### Phase 4: Decision & Optimization Engine
- **Replenishment Solver**: Dynamic reorder points (ROP) and Economic Order Quantity (EOQ) incorporating lead-time uncertainty.
- **Purchase Order Engine**: Consolidated supplier PO drafting respecting minimum order quantities (MOQ) and supplier packing tiers.
- **Warehouse Rebalancer**: Linear programming / heuristic optimizer to minimize split shipments and rebalance regional stock.

### Phase 5: Autonomous Multi-Agent & Copilot System
- Autonomous domain agents equipped with deterministic tools.
- Orchestration engine to execute multi-step diagnostic investigations.
- RAG layer for organizational business rules, supplier terms, and commercial knowledge.
- Conversational Copilot for natural-language reporting and interactive scenario simulation.

### Phase 6: Enterprise Connectors & Action Execution
- Production connectors for major OMS/WMS ecosystems (Shopify, eRetail, SAP, NetSuite).
- Bidirectional execution APIs with human-in-the-loop approvals.
- Granular permissioning, execution simulation ("dry run" mode), and tamper-evident audit logging.
