# AI Commerce Intelligence Platform

A vendor-neutral commerce intelligence platform designed to turn operational retail data into demand visibility, forecasting, risk detection, recommendations, and decision support. The repository has evolved beyond the original foundational datasets into a layered intelligence stack with deterministic analytics, forecasting, recommendation generation, Copilot-style orchestration, business explanations, and governed knowledge retrieval.

The platform is built to answer questions such as:

- What is happening in sales, inventory, profitability, and returns?
- Why is stockout risk or margin pressure rising?
- What is likely to happen next for demand and forecast accuracy?
- What should the business review or act on?
- What evidence, policy, and reasoning support the recommendation?

---

## What is implemented today

This codebase now covers the full operational intelligence lifecycle, not just data validation and forecasting.

### 1. Data foundation and validation
- Canonical vendor-neutral commerce data contracts for sales, inventory, products, warehouses, purchases, returns, channels, and suppliers.
- Pydantic-based schema validation and integrity checks.
- Synthetic data generation for realistic retail scenarios.
- Deterministic loaders and dataset quality auditing.

### 2. Demand intelligence and analytics
- Daily demand reconstruction and inventory-time-series normalization.
- Stockout detection and demand masking logic.
- ABC/XYZ portfolio segmentation.
- Feature engineering for sales and inventory modeling.
- Forecast evaluation metrics and hierarchical benchmarking.

### 3. Forecasting engine
- Baseline models: naive, seasonal naive, moving average.
- Statistical models: exponential smoothing and Croston-based intermittent demand modeling.
- ML forecasting: LightGBM-based tabular forecasting.
- Foundation model integration for zero-shot/adapter-based time-series forecasting.
- Rolling-origin backtesting and model selection policy.

### 4. Query layer and BI orchestration
- Structured query tools for revenue, margin, inventory, returns, forecasting, operations, recommendations, and decisions.
- Query contract definitions for intent mapping and business logic routing.
- Tool registry and execution service for deterministic BI operations.
- Safe read-only execution context for dashboards and Copilot-style prompts.

### 5. Recommendations and decision support
- Replenishment recommendations.
- Purchase-order proposals and warehouse rebalancing logic.
- Business impact assessment and recommendation schemas.
- Decision intelligence packages for human review with governance-safe execution behavior.
- Risk flags, trade-off evaluation, and traceability.

### 6. Copilot reasoning and controlled conversation
- Deterministic question interpretation and task planning.
- Orchestration across multiple business domains.
- Context management for session continuity, entity inheritance, ambiguity handling, and correction tracking.
- Evidence-based response aggregation and execution isolation.

### 7. Knowledge, governance, and explanations
- Business knowledge ingestion and repository-based retrieval.
- Capability classification for hybrid data + knowledge workflows.
- Explanation generation that converts execution results into grounded business narratives.
- Governance controls to prevent fabricated operational recommendations.
- Auditability and provenance tracking across reasoning workflows.

---

## High-level software architecture

The platform follows a layered architecture with strict separation between data ingestion, analytics, intelligence, orchestration, and human-facing interpretation.

```text
+----------------------------------------------------------------------------------+
|                          Presentation / Interaction Layer                         |
|  Dashboards  |  Chat / Copilot  |  API / Webhooks / Internal Tools                |
+-------------------------------------------+--------------------------------------+
                                            |
                                            v
+----------------------------------------------------------------------------------+
|                           Orchestration & Reasoning Layer                           |
|  CopilotService  |  ContextService  |  QueryContractService  |  QueryLayerService   |
|  Interpret / Plan / Execute  |  Session Memory  |  Intent Routing  | Tool Registry |
+-------------------------------------------+--------------------------------------+
                                            |
                                            v
+----------------------------------------------------------------------------------+
|                              Intelligence & Decision Layer                          |
|  Demand Analytics  |  Forecasting  |  Financial Intelligence  |  Inventory       |
|  Recommendations   |  Decision Intelligence  |  Business Impact  | Explanations   |
+-------------------------------------------+--------------------------------------+
                                            |
                                            v
+----------------------------------------------------------------------------------+
|                         Domain Analytics & Calculation Layer                        |
|  Feature engineering  |  ABC/XYZ  |  Stockout logic  |  Return analytics         |
|  KPI derivations  |  Replenishment / PO / Rebalancing  |  Model evaluation         |
+-------------------------------------------+--------------------------------------+
                                            |
                                            v
+----------------------------------------------------------------------------------+
|                             Data Contract & Quality Layer                           |
|  Pydantic schemas  |  Validation  |  Data generation  |  Data loaders / auditors   |
+-------------------------------------------+--------------------------------------+
                                            |
                                            v
+----------------------------------------------------------------------------------+
|                             Integration / Source Layer                                |
|  CSV / Batch files  |  OMS / WMS / ERP / Warehouse systems  |  External adapters    |
+----------------------------------------------------------------------------------+
```

### Architectural principles
- Deterministic-first analytics for revenue, inventory, order, and operational calculations.
- Separate business logic from orchestration and from LLM-style conversational interpretation.
- Human-in-the-loop decision governance for recommendations and actions.
- Evidence-first outputs with provenance, traceability, and auditability.
- Vendor-neutral data contracts with adapter-ready integration patterns.

---

## Repository structure

```text
.
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── README.md
│   ├── processed/
│   ├── raw/
│   └── sample/
│       ├── README.md
│       ├── channels.csv
│       ├── inventory.csv
│       ├── products.csv
│       ├── purchases.csv
│       ├── returns.csv
│       ├── sales.csv
│       ├── suppliers.csv
│       └── warehouses.csv
│
├── docs/
│   ├── architecture.md
│   ├── business-explanations.md
│   ├── business-impact.md
│   ├── business-knowledge-rag.md
│   ├── business-recommendations.md
│   ├── controlled-conversation-context.md
│   ├── copilot-reasoning-orchestration.md
│   ├── cross-domain-intelligence.md
│   ├── data-contract.md
│   ├── data-dictionary.md
│   ├── decision-intelligence.md
│   ├── demand-intelligence.md
│   ├── development-roadmap.md
│   ├── feature-engineering.md
│   ├── financial-intelligence.md
│   ├── forecasting-model-selection.md
│   ├── forecasting.md
│   ├── inventory-intelligence.md
│   ├── model-evaluation.md
│   ├── operational-economics.md
│   ├── product-vision.md
│   ├── profitability-attribution.md
│   ├── purchase-order-proposals.md
│   ├── query-contracts.md
│   ├── query-layer.md
│   ├── replenishment-engine.md
│   ├── return-anomaly-detection.md
│   ├── return-intervention-engine.md
│   ├── return-prediction-dataset.md
│   ├── return-prediction-modeling.md
│   ├── return-risk-calibration.md
│   ├── returns-intelligence.md
│   ├── unit-economics.md
│   └── warehouse-rebalancing.md
│
├── notebooks/
│   ├── README.md
│   ├── 01_data_exploration.ipynb
│   ├── 02_demand_intelligence.ipynb
│   ├── 03_abc_xyz_analysis.ipynb
│   ├── 04_forecasting_baselines.ipynb
│   └── 05_forecasting_model_comparison.ipynb
│
├── scripts/
│   ├── benchmark_business_impact.py
│   ├── benchmark_business_recommendations.py
│   ├── benchmark_context.py
│   ├── benchmark_copilot.py
│   ├── benchmark_decision_intelligence.py
│   ├── benchmark_explanations.py
│   ├── benchmark_knowledge.py
│   ├── benchmark_query_contracts.py
│   ├── benchmark_query_layer.py
│   ├── calibration_experiment_results.json
│   ├── evaluate_calibration_risk.py
│   ├── generate_sample_data.py
│   └── run_validation.py
│
├── src/
│   └── commerce_ai/
│       ├── __init__.py
│       ├── agents/
│       ├── analytics/
│       │   ├── abc_xyz.py
│       │   ├── calendar.py
│       │   ├── demand.py
│       │   ├── features.py
│       │   ├── metrics.py
│       │   └── stockout.py
│       ├── api/
│       ├── business_impact/
│       │   ├── quantification.py
│       │   ├── schemas.py
│       │   └── service.py
│       ├── config/
│       │   ├── settings.py
│       │   └── __init__.py
│       ├── context/
│       │   ├── enums.py
│       │   ├── governance.py
│       │   ├── lineage.py
│       │   ├── memory.py
│       │   ├── resolver.py
│       │   ├── schemas.py
│       │   ├── scope.py
│       │   ├── service.py
│       │   └── __init__.py
│       ├── copilot/
│       │   ├── ambiguity.py
│       │   ├── decomposition.py
│       │   ├── dependencies.py
│       │   ├── enums.py
│       │   ├── execution.py
│       │   ├── governance.py
│       │   ├── interpreter.py
│       │   ├── normalization.py
│       │   ├── orchestration.py
│       │   ├── results.py
│       │   ├── schemas.py
│       │   ├── service.py
│       │   └── __init__.py
│       ├── cross_domain/
│       │   ├── schemas.py
│       │   ├── service.py
│       │   └── signals.py
│       ├── data/
│       │   ├── generators.py
│       │   ├── loaders.py
│       │   ├── schemas.py
│       │   ├── validators.py
│       │   └── __init__.py
│       ├── decision_intelligence/
│       │   ├── options.py
│       │   ├── risks.py
│       │   ├── schemas.py
│       │   ├── service.py
│       │   ├── templates.py
│       │   ├── tradeoffs.py
│       │   └── __init__.py
│       ├── explanations/
│       │   ├── auditor.py
│       │   ├── confidence.py
│       │   ├── evidence.py
│       │   ├── enums.py
│       │   ├── governance.py
│       │   ├── renderer.py
│       │   ├── schemas.py
│       │   ├── service.py
│       │   ├── templates.py
│       │   └── __init__.py
│       ├── financial/
│       ├── forecasting/
│       │   ├── backtesting.py
│       │   ├── baselines.py
│       │   ├── base.py
│       │   ├── config.py
│       │   ├── croston.py
│       │   ├── datasets.py
│       │   ├── evaluation.py
│       │   ├── exponential_smoothing.py
│       │   ├── lightgbm_model.py
│       │   ├── selection.py
│       │   ├── service.py
│       │   ├── timesfm.py
│       │   └── __init__.py
│       ├── knowledge/
│       ├── query_contracts/
│       │   ├── dimensions.py
│       │   ├── enums.py
│       │   ├── intent.py
│       │   ├── metrics.py
│       │   ├── planner.py
│       │   ├── schemas.py
│       │   ├── service.py
│       │   ├── templates.py
│       │   ├── tool_mapping.py
│       │   ├── validation.py
│       │   └── __init__.py
│       ├── query_layer/
│       │   ├── base.py
│       │   ├── context.py
│       │   ├── decisions.py
│       │   ├── demand.py
│       │   ├── filters.py
│       │   ├── financial.py
│       │   ├── forecasting.py
│       │   ├── impact.py
│       │   ├── inventory.py
│       │   ├── operations.py
│       │   ├── recommendations.py
│       │   ├── registry.py
│       │   ├── returns.py
│       │   ├── sales.py
│       │   ├── schemas.py
│       │   ├── service.py
│       │   └── __init__.py
│       ├── recommendations/
│       │   ├── business_recommendations.py
│       │   ├── business_rules.py
│       │   ├── business_schemas.py
│       │   ├── purchase_orders.py
│       │   ├── replenishment.py
│       │   ├── schemas.py
│       │   ├── warehouse_rebalancing.py
│       │   └── __init__.py
│       └── __init__.py
│
└── tests/
    ├── __init__.py
    ├── test_abc_xyz.py
    ├── test_backtesting.py
    ├── test_baselines.py
    ├── test_calendar.py
    ├── test_croston.py
    ├── test_demand.py
    ├── test_evaluation.py
    ├── test_exponential_smoothing.py
    ├── test_features.py
    ├── test_forecasting_base.py
    ├── test_forecasting_datasets.py
    ├── test_forecasting_service.py
    ├── test_generators.py
    ├── test_lightgbm.py
    ├── test_loaders.py
    ├── test_metrics.py
    ├── test_selection.py
    ├── test_schemas.py
    ├── test_stockout.py
    ├── test_timesfm.py
    ├── test_validators.py
    └── test_selection.py
```

---

## Execution flow and key runtime patterns

The application is structured around a consistent, deterministic reasoning flow:

1. Raw business data enters through the data layer.
2. Validation, schema checks, and data normalization ensure reliability.
3. Analytics functions produce metrics, demand grids, stockout indicators, and feature sets.
4. Forecasting modules evaluate time-series models and benchmark the best-performing approach.
5. Query-layer tools surface operational and financial intelligence to downstream callers.
6. Recommendations engines propose replenishment, PO, and rebalancing actions.
7. Decision intelligence packages those recommendations into review-ready decision packages.
8. Copilot reasoning interprets business questions and orchestrates the appropriate tool chain.
9. Context tracking preserves conversational memory and entity continuity.
10. Explanations and knowledge retrieval ground the final answer in evidence and policy.

This makes the platform suitable not only for analytics but also for conversational business intelligence and governed operational decision support.

---

## Getting started

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

```bash
cp .env.example .env
```

### Generate sample data

```bash
python scripts/generate_sample_data.py
```

### Validate data quality

```bash
python scripts/run_validation.py --dir data/sample
```

### Run notebooks

```bash
jupyter notebook notebooks/
```

### Run tests

```bash
pytest
```

---

## Current maturity

The repository is no longer a basic prototype. It already includes a strong set of implemented modules and patterns for:

- data validation and synthetic commerce datasets
- demand reconstruction and inventory health analysis
- forecast benchmarking and model selection
- business-impact quantification
- recommendation generation and review logic
- decision packaging and risk evaluation
- Copilot orchestration and intent interpretation
- contextual memory and conversation governance
- knowledge retrieval and explanation generation

This positions the project as a working commerce intelligence foundation with a clear path toward deeper automation, richer domain agents, and broader enterprise integration.

---

## Roadmap alignment

The project is aligned with a staged evolution:

- Phase 0–1: canonical data contracts and validation
- Phase 2: intelligence analytics and feature engineering
- Phase 3: forecasting and benchmarking
- Phase 4+: operational recommendations and supply actions
- Phase 5+: domain agents and autonomous workflow orchestration
- Phase 6+: decision support, review, and governance
- Phase 7+: conversational Copilot, business explanations, knowledge grounding, and context-aware reasoning

The current codebase already contains substantial real implementation in the later phases, especially around query orchestration, context management, recommendation logic, decision intelligence, and explanation services.

---

## Summary

The AI Commerce Intelligence Platform is now a multi-layer intelligence system for retail operations, combining deterministic analytics, forecasting, recommendation engines, and a governed AI reasoning layer. It is designed to support real business questions with evidence, traceability, and operational decision support rather than opaque model output alone.

This README reflects the actual state of the repository as implemented today, and it maps directly to the architecture and module structure present in the codebase.
