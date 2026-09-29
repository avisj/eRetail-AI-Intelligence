# Notebooks Directory

This directory contains research, exploratory data analysis (EDA), and experimental notebooks for the AI Commerce Intelligence Platform.

## Available Notebooks

- `01_data_exploration.ipynb`: Foundation exploratory data analysis covering the 8 core commerce datasets:
  - Ingestion and schema auditing
  - Dataset dimensions, column types, and nullity checks
  - Date ranges and temporal coverage
  - SKU, warehouse, and channel master catalog distributions
  - Daily sales revenue and volume time series
  - Top-selling SKUs (Pareto analysis)
  - Warehouse inventory distribution and on-hand levels
  - Return rates across product categories
  - Stockout detection and unconstrained demand analysis

## Running the Notebooks

1. Ensure the virtual environment is activated:
   ```bash
   source .venv/bin/activate
   ```
2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook notebooks/
   ```
3. Or open directly inside VS Code / your IDE.
