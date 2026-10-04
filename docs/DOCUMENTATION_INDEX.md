# Documentation Index

Complete reference of all documentation files organized by module and domain. Each document contains detailed specifications, implementation notes, and design decisions for its respective module.

## Quick Navigation by Category

- [Data & Contracts](#data--contracts)
- [Analytics & Demand Intelligence](#analytics--demand-intelligence)
- [Forecasting](#forecasting)
- [Financial Intelligence](#financial-intelligence)
- [Inventory Management](#inventory-management)
- [Returns & Risk Management](#returns--risk-management)
- [Recommendations & Operations](#recommendations--operations)
- [Query & Orchestration Layer](#query--orchestration-layer)
- [Copilot & Reasoning](#copilot--reasoning)
- [Business Intelligence & Explanations](#business-intelligence--explanations)
- [Architecture & Strategy](#architecture--strategy)

---

## Data & Contracts

| Module | Documentation | Purpose |
|--------|---------------|---------|
| **Data Contracts** | [data-contract.md](data-contract.md) | Vendor-neutral commerce data contracts, schema definitions, and validation framework |
| **Data Dictionary** | [data-dictionary.md](data-dictionary.md) | Complete reference of all data entities, fields, types, and relationships |
| **Query Contracts** | [query-contracts.md](query-contracts.md) | Intent mapping, business logic routing, and query contract definitions |

---

## Analytics & Demand Intelligence

| Module | Documentation | Purpose |
|--------|---------------|---------|
| **Demand Intelligence** | [demand-intelligence.md](demand-intelligence.md) | Daily demand reconstruction, inventory normalization, and demand masking logic |
| **Feature Engineering** | [feature-engineering.md](feature-engineering.md) | Feature extraction, transformation pipelines, and variable engineering for modeling |
| **ABC/XYZ Analysis** | Covered in [demand-intelligence.md](demand-intelligence.md) | Portfolio segmentation using Pareto and volatility analysis |
| **Stockout Detection** | Covered in [demand-intelligence.md](demand-intelligence.md) | Stockout identification logic and inventory constraint handling |

---

## Forecasting

| Module | Documentation | Purpose |
|--------|---------------|---------|
| **Forecasting Engine** | [forecasting.md](forecasting.md) | Overview of all forecasting models, selection strategy, and execution framework |
| **Model Selection** | [forecasting-model-selection.md](forecasting-model-selection.md) | Model evaluation, selection policy, and hyperparameter optimization |
| **Model Evaluation** | [model-evaluation.md](model-evaluation.md) | Forecast accuracy metrics, hierarchical benchmarking, and performance assessment |

---

## Financial Intelligence

| Module | Documentation | Purpose |
|--------|---------------|---------|
| **Financial Intelligence** | [financial-intelligence.md](financial-intelligence.md) | Revenue, margin, profitability analysis, and financial metrics |
| **Profitability Attribution** | [profitability-attribution.md](profitability-attribution.md) | Margin decomposition by product, channel, and warehouse |
| **Operational Economics** | [operational-economics.md](operational-economics.md) | Cost modeling, operational KPIs, and financial impact quantification |
| **Unit Economics** | [unit-economics.md](unit-economics.md) | Per-unit cost, margin, and contribution analysis |
| **Business Impact** | [business-impact.md](business-impact.md) | Impact quantification framework and business metric translation |

---

## Inventory Management

| Module | Documentation | Purpose |
|--------|---------------|---------|
| **Inventory Intelligence** | [inventory-intelligence.md](inventory-intelligence.md) | Inventory health, carrying costs, and inventory optimization strategies |
| **Replenishment Engine** | [replenishment-engine.md](replenishment-engine.md) | Automated replenishment recommendations with safety stock calculation |
| **Warehouse Rebalancing** | [warehouse-rebalancing.md](warehouse-rebalancing.md) | Inter-warehouse inventory redistribution and network optimization |
| **Purchase Order Proposals** | [purchase-order-proposals.md](purchase-order-proposals.md) | Vendor-based PO generation with lead times and MOQ constraints |

---

## Returns & Risk Management

| Module | Documentation | Purpose |
|--------|---------------|---------|
| **Returns Intelligence** | [returns-intelligence.md](returns-intelligence.md) | Return rate analysis, trend detection, and category risk profiling |
| **Return Prediction Dataset** | [return-prediction-dataset.md](return-prediction-dataset.md) | Feature engineering and dataset construction for return prediction models |
| **Return Prediction Modeling** | [return-prediction-modeling.md](return-prediction-modeling.md) | ML models, training pipelines, and return probability estimation |
| **Return Risk Calibration** | [return-risk-calibration.md](return-risk-calibration.md) | Risk scoring, calibration methodology, and reliability assessment |
| **Return Anomaly Detection** | [return-anomaly-detection.md](return-anomaly-detection.md) | Anomaly identification, root cause analysis, and alert generation |
| **Return Intervention Engine** | [return-intervention-engine.md](return-intervention-engine.md) | Decision logic for intervention strategies and customer retention actions |

---

## Recommendations & Operations

| Module | Documentation | Purpose |
|--------|---------------|---------|
| **Business Recommendations** | [business-recommendations.md](business-recommendations.md) | Recommendation generation, business rules, and review workflows |
| **Decision Intelligence** | [decision-intelligence.md](decision-intelligence.md) | Decision packaging, risk evaluation, trade-off analysis, and governance |

---

## Query & Orchestration Layer

| Module | Documentation | Purpose |
|--------|---------------|---------|
| **Query Layer** | [query-layer.md](query-layer.md) | Core query execution service for revenue, margin, inventory, and operations tools |

---

## Copilot & Reasoning

| Module | Documentation | Purpose |
|--------|---------------|---------|
| **Copilot Reasoning & Orchestration** | [copilot-reasoning-orchestration.md](copilot-reasoning-orchestration.md) | Question interpretation, task planning, execution orchestration, and result aggregation |
| **Controlled Conversation Context** | [controlled-conversation-context.md](controlled-conversation-context.md) | Session management, context tracking, entity inheritance, and ambiguity handling |

---

## Business Intelligence & Explanations

| Module | Documentation | Purpose |
|--------|---------------|---------|
| **Business Knowledge RAG** | [business-knowledge-rag.md](business-knowledge-rag.md) | Knowledge ingestion, retrieval, grounding, and hybrid data + knowledge workflows |
| **Business Explanations** | [business-explanations.md](business-explanations.md) | Evidence-based narrative generation and governance-safe explanation rendering |
| **Cross-Domain Intelligence** | [cross-domain-intelligence.md](cross-domain-intelligence.md) | Multi-domain signal integration, correlation analysis, and holistic intelligence synthesis |

---

## Architecture & Strategy

| Module | Documentation | Purpose |
|--------|---------------|---------|
| **Architecture** | [architecture.md](architecture.md) | High-level system design, layered architecture, and integration patterns |
| **Product Vision** | [product-vision.md](product-vision.md) | Strategic direction, long-term goals, and product roadmap alignment |
| **Development Roadmap** | [development-roadmap.md](development-roadmap.md) | Phased implementation plan, milestones, and future capabilities |

---

## Statistics

- **Total Documentation Files**: 39
- **Total Lines of Documentation**: 10,000+
- **Coverage**: All modules, features, and implementation details
- **Format**: Markdown (.md)
- **Last Updated**: Auto-generated from repository state

## Usage Guidelines

1. **Getting Started**: Begin with [architecture.md](architecture.md) for system overview
2. **Implementation Details**: Refer to module-specific documentation for design and code-level details
3. **Query Examples**: Check [query-layer.md](query-layer.md) for operational intelligence examples
4. **Integration**: Review [data-contract.md](data-contract.md) for data integration patterns
5. **AI Features**: See [copilot-reasoning-orchestration.md](copilot-reasoning-orchestration.md) for conversational AI details

---

## Document Relationships

```
Architecture.md
    ├── Data Contract.md
    │   ├── Data Dictionary.md
    │   └── Query Contracts.md
    ├── Analytics & Feature Engineering.md
    │   ├── Demand Intelligence.md
    │   └── Forecasting.md
    ├── Financial Intelligence.md
    │   ├── Profitability Attribution.md
    │   ├── Unit Economics.md
    │   └── Operational Economics.md
    ├── Inventory Intelligence.md
    │   ├── Replenishment Engine.md
    │   ├── Purchase Order Proposals.md
    │   └── Warehouse Rebalancing.md
    ├── Returns Intelligence.md
    │   ├── Return Prediction Dataset.md
    │   ├── Return Prediction Modeling.md
    │   ├── Return Risk Calibration.md
    │   ├── Return Anomaly Detection.md
    │   └── Return Intervention Engine.md
    ├── Query Layer.md
    ├── Copilot Reasoning & Orchestration.md
    │   ├── Controlled Conversation Context.md
    │   ├── Business Knowledge RAG.md
    │   ├── Business Explanations.md
    │   └── Cross-Domain Intelligence.md
    ├── Business Recommendations.md
    ├── Decision Intelligence.md
    ├── Product Vision.md
    └── Development Roadmap.md
```

---

## Key References in Source Code

Implementations corresponding to each documentation module can be found in:

- `src/commerce_ai/data/` - Data contracts and validation
- `src/commerce_ai/analytics/` - Demand and feature engineering
- `src/commerce_ai/forecasting/` - All forecasting models
- `src/commerce_ai/financial/` - Financial intelligence modules
- `src/commerce_ai/query_layer/` - Query execution and tool registry
- `src/commerce_ai/recommendations/` - Replenishment and recommendations
- `src/commerce_ai/copilot/` - Copilot reasoning and orchestration
- `src/commerce_ai/explanations/` - Business explanations and governance
- `src/commerce_ai/knowledge/` - Business knowledge and RAG
- `src/commerce_ai/context/` - Conversation context management
- `src/commerce_ai/decision_intelligence/` - Decision packaging and risk evaluation

