# Phase 5B: Deterministic Return Anomaly Detection Engine

## 1. Overview & Architectural Scope

The **Return Anomaly Detection Engine** (Phase 5B) provides statistical, deterministic detection of unusual historical return patterns across retail operations. It is designed to answer the operational question:

> *"What customer return behavior is unusually different from its historical or peer baseline?"*

### Clear Boundary of Scope
- **What it does**: Evaluates historical returns and sales across dimensions (SKU, Channel, Warehouse, SKU×Channel, SKU×Warehouse, Reason, Portfolio Time Series) using rolling statistical distributions (Mean/Std Z-Score and Median/MAD Modified Z-Score) to flag unexpected spikes, drops, and distribution shifts.
- **What it does NOT do**: It does **not** predict whether a future order will be returned (future return probability belongs to predictive ML in Phase 5C). It does not use LLMs, agents, RAG, fraud detection, customer profiling, or autonomous purchasing/refund decisions.

---

## 2. Statistical Formulation & Detection Logic

### 2.1 Standard Rolling Z-Score
For an entity with current metric $x$ and historical baseline values $\{x_1, x_2, \dots, x_N\}$:
$$\mu = \frac{1}{N} \sum_{i=1}^N x_i$$
$$\sigma = \sqrt{\frac{1}{N - 1} \sum_{i=1}^N (x_i - \mu)^2}$$
$$z = \frac{x - \mu}{\sigma}$$

### 2.2 Robust Modified Z-Score (Median Absolute Deviation)
When outliers are present in the historical baseline, the engine provides robust estimation via Median Absolute Deviation:
$$\tilde{x} = \text{median}(\{x_1, \dots, x_N\})$$
$$\text{MAD} = \text{median}(\{|x_i - \tilde{x}|\})$$
$$M = \frac{0.6745 \cdot (x - \tilde{x})}{\text{MAD}}$$

### 2.3 Baseline Level vs. Anomaly Distinction
A high return rate does **not** inherently constitute an anomaly:
- An apparel footwear SKU with a consistent 25% return rate ($\mu = 0.25, \sigma = 0.01$) observed at 25% in the current period produces $z = 0.0$ (`NO_ANOMALY`, severity `NONE`).
- An electronics SKU with a typical 2% return rate ($\mu = 0.02, \sigma = 0.002$) jumping to 8% produces $z = +30.0$ (`RETURN_RATE_SPIKE`, severity `CRITICAL`).

### 2.4 Zero-Variance Handling
When historical variance is zero ($\sigma = 0$ or $\text{MAD} = 0$):
- If $x = \mu$: The metric is perfectly steady ($z = 0.0$, `NO_ANOMALY`, severity `NONE`).
- If $x \ne \mu$: Division by zero is prevented ($z = \text{None}$). The event is classified as `BASELINE_SHIFT` with severity determined by absolute deviation:
  - $|\Delta| \ge 0.20 \rightarrow \text{CRITICAL}$
  - $0.10 \le |\Delta| < 0.20 \rightarrow \text{HIGH}$
  - $0.05 \le |\Delta| < 0.10 \rightarrow \text{MEDIUM}$
  - $|\Delta| < 0.05 \rightarrow \text{LOW}$

---

## 3. Sample Size Reliability Guards

Statistical return rate estimation requires adequate denominators to prevent noisy false positives. The engine enforces three strict sample guards:
1. `min_sold_units` (default: 30): If current period sold units $< 30$, rate spikes are suppressed, flagging `INSUFFICIENT_SAMPLE`.
2. `min_return_count` (default: 5): If return event count $< 5$, flags `INSUFFICIENT_SAMPLE`.
3. `min_history_periods` (default: 3): If fewer than 3 historical baseline periods exist, flags `INSUFFICIENT_SAMPLE`.

---

## 4. Multi-Dimensional Taxonomy

### 4.1 Anomaly Types
| Anomaly Type | Description |
| :--- | :--- |
| `RETURN_RATE_SPIKE` | Return rate statistically surges above baseline ($z \ge +2.0$). |
| `RETURN_RATE_DROP` | Return rate statistically drops below baseline ($z \le -2.0$). |
| `RETURN_VOLUME_SPIKE` | Returned units surge above multiplier threshold even if sales grew proportionally. |
| `RETURN_VOLUME_DROP` | Returned units drop significantly below historical average. |
| `RETURN_REASON_SHIFT` | Disproportionate surge or collapse in reason share ($\ge 15\text{ pp}$). |
| `CHANNEL_RETURN_SHIFT` | Significant change in specific channel return rate. |
| `WAREHOUSE_RETURN_SHIFT` | Significant change in specific warehouse return rate. |
| `SKU_CHANNEL_RETURN_SHIFT` | Interaction anomaly isolated to a specific SKU on a specific channel. |
| `SKU_WAREHOUSE_RETURN_SHIFT` | Interaction anomaly isolated to a specific SKU fulfilled by a warehouse. |
| `BASELINE_SHIFT` | Shift from a previously zero-variance baseline. |
| `INSUFFICIENT_SAMPLE` | Sample size or history duration below statistical reliability threshold. |
| `NO_ANOMALY` | Metric is within expected historical variance. |

### 4.2 Severity Tiers
- `CRITICAL`: $|z| \ge 3.0$ (extreme outlier, $< 0.3\%$ probability under normality)
- `HIGH`: $2.5 \le |z| < 3.0$
- `MEDIUM`: $2.0 \le |z| < 2.5$
- `LOW`: $1.5 \le |z| < 2.0$
- `NONE`: $|z| < 1.5$

### 4.3 Deviation Direction
- `INCREASE`: Current metric exceeds baseline.
- `DECREASE`: Current metric is below baseline.
- `NONE`: Current metric matches baseline within tolerance.

---

## 5. Return Reason Distribution Shift

Reason shifts evaluate changes in the proportion of returns attributed to each category:
$$\text{Current Share} = \frac{\text{Reason Returned Units}_{\text{current}}}{\text{Total Returned Units}_{\text{current}}}$$
$$\text{Baseline Share} = \frac{\text{Reason Returned Units}_{\text{baseline}}}{\text{Total Returned Units}_{\text{baseline}}}$$
$$\Delta \text{Share} = \text{Current Share} - \text{Baseline Share}$$

If $|\Delta \text{Share}| \ge \text{reason\_shift\_threshold}$ (default: $0.15$ or $15\text{ pp}$), a `RETURN_REASON_SHIFT` anomaly is recorded.

---

## 6. Deterministic IDs & Factual Rationales

### 6.1 Deterministic SHA-256 Identification
Every detected anomaly receives an immutable, reproducible ID formatted as:
$$\text{ANOM-} + \text{SHA-256}(\text{Dimension} \parallel \text{Entity ID} \parallel \text{Period} \parallel \text{Anomaly Type} \parallel \text{As-Of Date})[0:16]$$

### 6.2 Factual Non-LLM Rationales
Rationales are strictly deterministic string templates populated with ground-truth numbers, avoiding hallucinated causal speculation. Example:
```text
Entity 'SKU_123' (SKU) in period '2026-03' observed return rate 18.50% vs baseline mean 5.20%, z-score: +6.33. Direction: INCREASE, Severity: CRITICAL.
```

---

## 7. Python Usage Examples

### 7.1 Basic Multi-Dimensional Anomaly Detection
```python
import pandas as pd
from commerce_ai.returns import ReturnAnomalyService, ReturnAnomalyConfig

# Initialize service with custom statistical thresholds
config = ReturnAnomalyConfig(
    z_score_threshold=2.0,
    critical_z_score=3.0,
    min_sold_units=30,
    min_return_count=5,
    min_history_periods=3,
    reason_shift_threshold=0.15,
)
service = ReturnAnomalyService(config=config)

# Run detection with strict anti-leakage evaluation cutoff
result = service.detect(
    returns=returns_df,
    sales=sales_df,
    as_of_date="2026-03-31",
    time_series_freq="M",
)

print(f"Total anomalies: {result.total_anomalies}")
print(f"Critical: {result.critical_count}, High: {result.high_count}")

# Export to flat DataFrame
anomalies_df = result.to_dataframe()
print(anomalies_df[["dimension", "entity_id", "anomaly_type", "severity", "rationale"]])
```

### 7.2 Filtering Anomalies
```python
from commerce_ai.returns import AnomalySeverity

# Filter critical issues requiring urgent investigation
critical_anomalies = result.filter_by_severity(AnomalySeverity.CRITICAL)

# Filter SKU-specific shifts
sku_anomalies = result.filter_by_dimension("SKU")

# Filter channel shifts
channel_anomalies = result.filter_by_dimension("CHANNEL")
```

---

## 8. Anti-Leakage & Reproducibility Guarantee

1. **Chronological Filtering**: All sales and returns records timestamped after `as_of_date` are strictly filtered prior to aggregation.
2. **Determinism**: Given identical input DataFrames and `as_of_date`, output anomaly IDs, z-scores, severities, and rationales are 100% invariant across runs and platforms.
