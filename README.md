# AI Commerce Intelligence Platform

A standalone, vendor-neutral intelligence platform for ecommerce, OMS, WMS, inventory, purchasing, sales, warehouse, and operational data.

> **Long-term vision**: Convert ecommerce operational data into forecasts, risks, opportunities, recommendations, and eventually executable actions.

---

## 1. What Has Been Implemented (Phase 0, Phase 1, Phase 2 & Phase 3)

This repository contains the foundation, data contracts, demand intelligence, and demand forecasting engine:
- **Vendor-Neutral Data Contract (Phase 0 + 1)**: Canonical schemas for 8 core ecommerce entities (`sales`, `inventory`, `products`, `warehouses`, `purchases`, `returns`, `channels`, `suppliers`).
- **Pydantic Validation & Ingestion (Phase 1)**: Strict schema validation, value range assertions, non-empty identifiers, chronological verification, and safe optional field extensibility.
- **Relational Integrity Auditor (Phase 1)**: Cross-entity foreign key referential integrity verification across all datasets.
- **Realistic Synthetic Data Generator (Phase 1)**: Coupled simulation with Pareto velocity tiers, seasonality, trends, promotional lift, category return rates, and closed-loop stockout & replenishment cycles.
- **Demand Reconstruction Grid (Phase 2)**: Regularized daily SKU-Warehouse time series explicitly populating zero-sales days and forward-filling inventory positions.
- **Stockout Detection & Demand Masking (Phase 2)**: Identification of stockout streaks (`is_stockout`, `stockout_days`, `stockout_event_id`) and demand masking (`is_demand_constrained`, `forecast_training_eligible`).
- **Dynamic ABC/XYZ Portfolio Segmentation (Phase 2)**: Configurable revenue Pareto contribution (ABC), demand coefficient of variation (XYZ), and 9-box operational segmentation matrix.
- **Leakage-Free Feature Engineering (Phase 2)**: Autoregressive lags ($1, 7, 14, 28, 30$), causally shifted rolling windows ($7, 14, 30, 90$), trend momentum ratios, cyclical trigonometric calendar encodings, and business events.
- **Forecasting Engine & Benchmarking (Phase 3)**:
  - Universal `ForecastModel` interface with standardized outputs (`ForecastRecord`, `ForecastOutput`, `ForecastMetadata`).
  - Baseline algorithms: `NaiveModel`, `SeasonalNaiveModel` (7d), `MovingAverageModel` (7d, 14d, 30d).
  - Parametric smoothing: `ExponentialSmoothingModel` (Holt-Winters with prediction intervals).
  - Intermittent demand modeling: `CrostonModel` (Classic and Syntetos-Boylan Approximation SBA).
  - Tabular Machine Learning: `LightGBMForecastModel` with autoregressive recursive multi-step forecasting and prediction intervals.
  - Zero-shot Foundation Adapter: `TimesFMForecastModel` with environment discovery, graceful `TIMESFM_UNAVAILABLE` status, and unit-testable mock predictor.
  - Time-series cross validation: `RollingOriginBacktester` with strict temporal cutoff and series-level error isolation.
  - Hierarchical evaluation: `ForecastEvaluator` computing MAE, RMSE, zero-safe MAPE, volume-weighted WAPE, Bias %, and coverage across global, SKU, warehouse, and intermittency dimensions.
  - Champion selection: `ModelSelector` with parsimony margin and `ModelSelectionPolicy` for segment-specific routing.
  - High-level inference API: `ForecastService` supporting single-series and batch execution with automatic policy routing.
- **Automated Test Suite**: 88 unit and integration tests passing with 100% clean test results.

---

## 2. Project Directory Structure

```text
.
├── .env.example              # Environment variables template
├── .gitignore                # Version control ignore list (no secrets, no large dumps)
├── README.md                 # Project documentation & guide
├── pyproject.toml            # Project packaging & pytest configuration
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── raw/                  # Ingestion drops (git-ignored)
│   ├── processed/            # Clean transformed datasets (git-ignored)
│   ├── sample/               # Generated synthetic sample CSV files
│   └── README.md
│
├── docs/
│   ├── product-vision.md     # 5-tier decision hierarchy & capability roadmap
│   ├── architecture.md       # Layered system architecture & SPI adapters
│   ├── data-contract.md      # Vendor-neutral canonical entity contracts
│   ├── data-dictionary.md    # Field-level dictionary and data constraints
│   ├── demand-intelligence.md# Daily demand grid and stockout masking logic
│   ├── feature-engineering.md# Lags, rolling statistics, and calendar features
│   ├── forecasting.md        # Forecasting models catalog & interfaces
│   ├── model-evaluation.md   # Backtesting framework & error metrics
│   ├── forecasting-model-selection.md # Automated model routing & policy
│   └── development-roadmap.md# Multi-phase engineering roadmap
│
├── notebooks/
│   ├── 01_data_exploration.ipynb          # End-to-end exploratory analysis notebook
│   ├── 02_demand_intelligence.ipynb      # Demand reconstruction & stockout masking
│   ├── 03_abc_xyz_analysis.ipynb          # Portfolio segmentation & 9-box matrix
│   ├── 04_forecasting_baselines.ipynb     # Heuristic & statistical baseline comparison
│   ├── 05_forecasting_model_comparison.ipynb # Multi-model backtesting & LightGBM benchmark
│   └── README.md
│
├── scripts/
│   ├── generate_sample_data.py   # CLI tool to generate synthetic datasets
│   └── run_validation.py         # CLI tool to audit data quality & schema integrity
│
├── src/
│   └── commerce_ai/
│       ├── __init__.py
│       ├── config/               # Pydantic environment configuration
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── data/                 # Data contract, validation, loading, generation
│       │   ├── __init__.py
│       │   ├── schemas.py
│       │   ├── validators.py
│       │   ├── loaders.py
│       │   └── generators.py
│       ├── analytics/            # Demand Intelligence & Feature Engineering (Phase 2)
│       │   ├── __init__.py
│       │   ├── demand.py         # Daily regularized demand grid
│       │   ├── stockout.py       # Stockout detection & demand masking
│       │   ├── abc_xyz.py        # ABC/XYZ segmentation & 9-box matrix
│       │   ├── calendar.py       # Calendar & business retail event features
│       │   ├── features.py       # Lags, causally shifted rolling windows, trends
│       │   └── metrics.py        # MAE, RMSE, zero-safe MAPE, WAPE, Bias
│       ├── forecasting/          # Forecasting Engine & Benchmarking (Phase 3)
│       │   ├── __init__.py
│       │   ├── base.py           # Universal ForecastModel, ForecastOutput, ForecastRecord
│       │   ├── config.py         # ForecastConfig, BacktestConfig
│       │   ├── datasets.py       # Time-series splits & rolling origin folds
│       │   ├── baselines.py      # Naive, Seasonal Naive, Moving Average
│       │   ├── exponential_smoothing.py # Holt-Winters & Simple Exp Smoothing
│       │   ├── croston.py        # Croston Classic & SBA intermittent models
│       │   ├── lightgbm_model.py # Tabular GBDT forecasting model
│       │   ├── timesfm.py        # Google TimesFM adapter & Darwin fallback
│       │   ├── backtesting.py    # Rolling-origin cross validation runner
│       │   ├── evaluation.py     # Hierarchical multi-level evaluation engine
│       │   ├── selection.py      # Automated champion selector & segment routing
│       │   └── service.py        # High-level ForecastService for production inference
│       ├── inventory/            # (Future Phase 4) Stockout, safety stock
│       ├── recommendations/      # (Future Phase 4) Replenishment, PO proposals
│       ├── agents/               # (Future Phase 5) Autonomous domain agents
│       ├── rag/                  # (Future Phase 5) Knowledge base & business docs
│       ├── integrations/         # (Future Phase 6) OMS/WMS adapters
│       └── api/                  # (Future Phase 6) FastAPI endpoints
│
└── tests/
    ├── __init__.py
    ├── test_schemas.py               # Pydantic schema validation tests
    ├── test_validators.py            # Validator and quality report tests
    ├── test_loaders.py               # CSV loader & error handling tests
    ├── test_generators.py            # Synthetic generator & reproducibility tests
    ├── test_demand.py                # Daily demand reconstruction tests
    ├── test_stockout.py              # Stockout detection & masking tests
    ├── test_abc_xyz.py               # ABC, XYZ, and matrix classification tests
    ├── test_calendar.py              # Calendar & retail event feature tests
    ├── test_features.py              # Lags, rolling window, leakage tests
    ├── test_metrics.py               # Forecast evaluation metric tests
    ├── test_forecasting_base.py      # Base contract & serialization tests
    ├── test_forecasting_datasets.py  # Splits & rolling origin fold tests
    ├── test_baselines.py             # Naive, Seasonal Naive, Moving Avg tests
    ├── test_exponential_smoothing.py # Holt-Winters & interval tests
    ├── test_croston.py               # Croston & SBA intermittent tests
    ├── test_lightgbm.py              # LightGBM fitting & recursive forecasting tests
    ├── test_timesfm.py               # TimesFM environment & mock tests
    ├── test_backtesting.py           # Multi-fold backtesting & error isolation tests
    ├── test_evaluation.py            # Hierarchical evaluation & table tests
    ├── test_selection.py             # ModelSelector & segment policy tests
    └── test_forecasting_service.py   # ForecastService single/batch inference tests
```

---

## 3. Getting Started

### 3.1 Python Environment Setup
Requires Python 3.9+ (Python 3.12 recommended).

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Configure Environment Variables

```bash
cp .env.example .env
```

---

## 4. Operational Commands

### 4.1 Generate Synthetic Data
To generate the full canonical dataset (500 SKUs, 5 Warehouses, 4 Channels, 20 Suppliers, 730 Days):

```bash
python scripts/generate_sample_data.py
```

Options:
```bash
python scripts/generate_sample_data.py --help
# Example: Quick run with 50 SKUs for 90 days
python scripts/generate_sample_data.py --skus 50 --warehouses 3 --days 90 --output data/sample
```

### 4.2 Run Data Quality & Validation Audit

```bash
python scripts/run_validation.py --dir data/sample
```

To output raw JSON for automated CI pipelines:
```bash
python scripts/run_validation.py --dir data/sample --json
```

### 4.3 Run Automated Tests

```bash
pytest
```

---

## 5. Launching Jupyter Notebooks

To run the exploratory data analysis notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
Or open `notebooks/01_data_exploration.ipynb` directly in VS Code / Cursor with the `.venv` kernel.

---

## 6. Current Limitations (Phase 0 + Phase 1)

- **Purely Local & Batch**: Operates on local CSV files without persistent relational databases or streaming message queues.
- **Synthetic Data Only**: Validated exclusively against synthetic coupled ecommerce data; no production systems or proprietary connectors are connected.
- **Heuristic Inventory Simulation**: Uses stochastic simulation rules (Poisson demand, base lead time variance) rather than live warehouse telemetry.
- **No Forecasting / ML Yet**: TimesFM-3 and ML models will be introduced in subsequent phases after feature engineering is completed.

---

## 7. Next Development Phase: Phase 3

**Phase 3 — Forecasting Engine**:
1. **Zero-Shot Foundation Model Adapter**: Integration with TimesFM-3 for multi-horizon demand forecasting.
2. **Tabular Machine Learning Baselines**: LightGBM time-series pipeline incorporating the Phase 2 feature dataset.
3. **Classical Statistical Baselines**: Historical moving averages, Holt-Winters, and Croston's method for intermittent series.
4. **Hierarchical Reconciliation & Backtesting**: Evaluating WAPE, RMSE, and bias across SKU and warehouse aggregation tiers.
