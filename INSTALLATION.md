# Installation & Setup Guide

Complete step-by-step instructions to clone, set up, and run the AI Commerce Intelligence Platform on your system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Clone the Repository](#clone-the-repository)
3. [Installation Steps](#installation-steps)
4. [Verification](#verification)
5. [Quick Start](#quick-start)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

---

## System Requirements

Before you begin, ensure your system meets these requirements:

### Minimum Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.9+ | Recommended: 3.10 or 3.11 |
| **pip** | 21.0+ | Package manager |
| **Git** | 2.20+ | For cloning the repository |
| **RAM** | 4 GB minimum | 8 GB recommended for full functionality |
| **Disk Space** | 2 GB | For repository + dependencies + sample data |
| **OS** | macOS, Linux, Windows | Any modern OS with Python support |

### Verify Your System

```bash
# Check Python version (should be 3.9+)
python3 --version

# Check pip version
pip3 --version

# Check Git version
git --version
```

---

## Clone the Repository

### Using HTTPS (recommended for most users)

```bash
git clone https://github.com/avisj/eRetail-AI-Intelligence.git
cd eRetail-AI-Intelligence
```

### Using SSH (if you have SSH keys configured)

```bash
git clone git@github.com:avisj/eRetail-AI-Intelligence.git
cd eRetail-AI-Intelligence
```

### Verify cloned repository

```bash
# List top-level files to confirm successful clone
ls -la

# You should see:
# - src/
# - scripts/
# - notebooks/
# - data/
# - tests/
# - docs/
# - requirements.txt
# - pyproject.toml
# - README.md
```

---

## Installation Steps

### Step 1: Create Python Virtual Environment

A virtual environment isolates project dependencies from your system Python.

#### On macOS/Linux:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Your prompt should now show (.venv) prefix
# Example: (.venv) user@machine:eRetail-AI-Intelligence $
```

#### On Windows (Command Prompt):

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Your prompt should now show (.venv) prefix
```

#### On Windows (PowerShell):

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2: Upgrade pip and Install Build Tools

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Upgrade setuptools and wheel
pip install --upgrade setuptools wheel
```

### Step 3: Install Python Dependencies

You can install using either `requirements.txt` or `pyproject.toml`:

#### Option A: Using requirements.txt (standard)

```bash
pip install -r requirements.txt
```

#### Option B: Using Poetry (if you prefer)

```bash
# Install Poetry if not already installed
pip install poetry

# Install dependencies using poetry
poetry install
```

### Step 4: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration (optional for local setup)
# On macOS/Linux:
nano .env

# On Windows (Notepad):
notepad .env

# Key environment variables (for reference):
# - Set any API keys for LLM integration if needed
# - Configure data paths if using non-default locations
```

### Step 5: Generate Sample Data

Sample data is pre-generated for local testing, but you can regenerate it:

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Generate sample commerce datasets
python scripts/generate_sample_data.py

# This creates CSV files in data/sample/ with:
# - sales.csv (transactional records)
# - inventory.csv (inventory snapshots)
# - products.csv (product catalog)
# - warehouses.csv (warehouse locations)
# - purchases.csv (purchase orders)
# - returns.csv (customer returns)
# - channels.csv (sales channels)
# - suppliers.csv (supplier master data)
```

### Step 6: Validate Installation

```bash
# Validate data schemas and integrity
python scripts/run_validation.py --dir data/sample

# Expected output:
# ✓ All schemas validated successfully
# ✓ Data integrity checks passed
# ✓ Ready for analytics
```

---

## Verification

Run these commands to verify your setup is complete and working:

### 1. Check Python Environment

```bash
# Should show .venv path
which python3  # macOS/Linux
where python   # Windows

# Should show packages in .venv
pip list | head -20
```

### 2. Import Core Modules

```python
python3 -c "
from commerce_ai.data.schemas import SalesRecord
from commerce_ai.analytics.demand import DemandAnalytics
from commerce_ai.forecasting.service import ForecastingService
print('✓ All core modules imported successfully')
"
```

### 3. Run Basic Data Validation

```bash
python3 << 'EOF'
from commerce_ai.data.loaders import load_csv
from commerce_ai.data.validators import validate_sales

# Load sample data
sales_df = load_csv('data/sample/sales.csv')

# Validate
is_valid = validate_sales(sales_df)
print(f"✓ Sample data validation: {'PASSED' if is_valid else 'FAILED'}")
print(f"  Loaded {len(sales_df)} sales records")
EOF
```

### 4. Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_demand.py -v

# Run with coverage report
pytest tests/ --cov=src/commerce_ai --cov-report=html
```

---

## Quick Start

After installation, here are quick ways to explore the platform:

### Option 1: Run Sample Notebooks

```bash
# Activate environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Start Jupyter
jupyter notebook notebooks/

# Open your browser to http://localhost:8888
# Start with: 01_data_exploration.ipynb
```

### Option 2: Run Benchmark Scripts

```bash
# Validate data contracts
python scripts/run_validation.py --dir data/sample

# Run demand intelligence benchmark
python scripts/benchmark_query_layer.py

# Run forecasting benchmark
python scripts/benchmark_query_layer.py --forecast
```

### Option 3: Interactive Python Shell

```bash
python3

# Inside Python:
from commerce_ai.data.loaders import load_csv
from commerce_ai.analytics.demand import DemandAnalytics

# Load data
sales = load_csv('data/sample/sales.csv')
inventory = load_csv('data/sample/inventory.csv')

# Run demand analytics
analytics = DemandAnalytics()
demand_grid = analytics.build_demand_grid(sales, inventory)

print(demand_grid.head())

exit()
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Python version error

```
Error: python3 version must be 3.9+
```

**Solution:**
```bash
# Check your Python version
python3 --version

# If needed, install Python 3.10+ from python.org
# Then recreate virtual environment:
rm -rf .venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. Virtual environment not activating

```
Error: command not found: python or module not found
```

**Solution:**
```bash
# On macOS/Linux, make sure you're using the correct path:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Verify activation (should show (.venv) in prompt)
which python  # macOS/Linux
where python  # Windows
```

#### 3. Dependencies installation fails

```
Error: pip error or package not found
```

**Solution:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Clear pip cache
pip cache purge

# Try installing again
pip install -r requirements.txt

# If specific package fails, install dependencies individually:
pip install numpy pandas scikit-learn lightgbm pydantic
```

#### 4. Missing data files

```
Error: FileNotFoundError: data/sample/sales.csv not found
```

**Solution:**
```bash
# Regenerate sample data
python scripts/generate_sample_data.py

# Verify files were created
ls -la data/sample/
```

#### 5. Jupyter notebook kernel issues

```
Error: ModuleNotFoundError when running notebook
```

**Solution:**
```bash
# Install Jupyter in the virtual environment
pip install jupyter jupyterlab

# Create kernel for virtual environment
python -m ipykernel install --user --name commerce-ai --display-name "Python (Commerce AI)"

# Restart Jupyter and select the new kernel
```

#### 6. Permission denied on scripts

```
Error: Permission denied: ./scripts/generate_sample_data.py
```

**Solution:**
```bash
# Make script executable (macOS/Linux)
chmod +x scripts/generate_sample_data.py

# Or run with python directly
python scripts/generate_sample_data.py
```

#### 7. Windows long path issues

```
Error: FileNotFoundError with long paths on Windows
```

**Solution:**
```bash
# Enable long paths in Windows (run as Administrator):
# Open PowerShell as Admin and run:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Then restart your terminal
```

---

## Next Steps

After successful installation:

1. **Read the Documentation**
   - Start with [docs/architecture.md](docs/architecture.md)
   - Review [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) for all modules

2. **Explore the Notebooks**
   - Run `01_data_exploration.ipynb` to understand data
   - Run `02_demand_intelligence.ipynb` for demand analysis
   - Run `03_abc_xyz_analysis.ipynb` for portfolio segmentation
   - Run `04_forecasting_baselines.ipynb` for forecasting intro
   - Run `05_forecasting_model_comparison.ipynb` for model evaluation

3. **Review Source Code**
   - Explore `src/commerce_ai/` directory structure
   - Each module has clear separation of concerns
   - Start with simpler modules like `analytics/` before moving to complex ones like `copilot/`

4. **Run Benchmarks**
   ```bash
   python scripts/benchmark_query_layer.py
   python scripts/benchmark_business_impact.py
   python scripts/benchmark_copilot.py
   ```

5. **Load Your Own Data**
   - Follow [data/README.md](data/README.md) format
   - Place CSVs in `data/raw/`
   - Use data loaders to validate schemas
   - See [docs/data-contract.md](docs/data-contract.md) for requirements

6. **Integrate with Your System**
   - Review [docs/query-layer.md](docs/query-layer.md) for API usage
   - Check [docs/data-contract.md](docs/data-contract.md) for integration patterns
   - See `src/commerce_ai/api/` for REST endpoint examples

---

## Getting Help

If you encounter issues:

1. **Check the Documentation**
   - See [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)
   - Review relevant module docs for detailed specs

2. **Check Existing Tests**
   - Browse `tests/` directory for usage examples
   - Run tests with verbose output: `pytest tests/ -vv`

3. **Review Notebooks**
   - Each notebook includes comments and examples
   - Great for learning how modules work together

4. **Open an Issue**
   - Provide: OS, Python version, error message, steps to reproduce
   - Include output of: `pip list` and `python --version`

---

## Quick Reference

### Commonly Used Commands

```bash
# Activate environment
source .venv/bin/activate

# Deactivate environment
deactivate

# Run all tests
pytest tests/

# Generate sample data
python scripts/generate_sample_data.py

# Start Jupyter
jupyter notebook notebooks/

# Run validation
python scripts/run_validation.py --dir data/sample

# Check dependencies
pip list

# Update all packages
pip install --upgrade -r requirements.txt
```

### Directory Structure After Installation

```
eRetail-AI-Intelligence/
├── .venv/                          # Virtual environment (created after Step 1)
├── data/
│   ├── sample/                     # Generated sample data
│   │   ├── sales.csv
│   │   ├── inventory.csv
│   │   ├── products.csv
│   │   └── ...
│   ├── raw/                        # For your raw data
│   └── processed/                  # For processed data
├── src/commerce_ai/                # Main package
├── scripts/                        # Utility scripts
├── notebooks/                      # Jupyter notebooks
├── tests/                          # Unit tests
├── docs/                           # Documentation
├── .env                            # Environment variables (created from .env.example)
├── requirements.txt                # Python dependencies
└── README.md                       # Main README
```

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| 1.0 | Oct 2026 | Initial installation guide |

---

Last updated: October 2026

For the latest version of this guide, visit: [INSTALLATION.md](INSTALLATION.md)
