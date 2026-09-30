# Supervised Return Risk ML Modeling & Evaluation (Phase 5C-2A)

This document details the first supervised machine learning baselines for predicting order-line return risk, covering architecture, tabular formulation, class imbalance handling, reproducible model implementations, and chronological evaluation.

---

## 1. Why Return Prediction is a Tabular Classification Problem

Customer return prediction in omnichannel retail is fundamentally an **instance-level binary classification problem**, not a contiguous time series forecasting problem:
- **Decision Unit**: The observation unit is an individual transaction order line: $(order\_id, sku\_id)$ or $sale\_id$.
- **Feature Space**: The predictive signals are cross-sectional and heterogeneous: order attributes (quantity, price, discount, channel), product attributes (category, brand, cost, margin, velocity tier), temporal calendar context (day of week, month, seasonality), and point-in-time historical track records (historical return rate of the SKU, channel, and warehouse).
- **Target Formulation**: The ground-truth outcome is binary:
  $$\text{target\_returned} \in \{0, 1\}$$
  indicating whether any portion of the item line was returned within the policy return window ($\le 30$ calendar days).

---

## 2. Why TimesFM is NOT Used for Return Prediction

Google's **TimesFM** (Time Series Foundation Model) is a zero-shot foundational model designed for **univariate/multivariate time series forecasting** (e.g. predicting weekly SKU sales demand $y_{t+1}, \dots, y_{t+H}$ from contiguous historical demand sequences $y_1, \dots, y_t$).

TimesFM is not applicable to return risk prediction for five fundamental reasons:
1. **Grain Incompatibility**: TimesFM expects regularly spaced sequential arrays indexed by discrete time steps ($t \in \mathbb{Z}$). In contrast, return prediction operates on discrete, non-contiguous customer order events occurring asynchronously across hundreds of thousands of independent baskets.
2. **Tabular Conditioning**: Return risk depends heavily on static and relational order-line features (e.g. discount depth, item price, channel type, warehouse location). TimesFM does not condition predictions on heterogeneous tabular feature matrices.
3. **Target Type**: TimesFM outputs numeric continuous values (future demand quantities), whereas return risk estimation requires a calibrated probability distribution over a binary label: $P(\text{returned} = 1 \mid X) \in [0.0, 1.0]$.
4. **Severe Class Imbalance**: Foundation time-series models do not incorporate cost-sensitive loss functions or class-weighting mechanisms necessary to address 8.10% positive class rarity.
5. **Architectural Separation of Concerns**: In Commerce AI, TimesFM is reserved for demand forecasting (Phase 2). Return risk is cleanly isolated as a supervised tabular classification engine (Phase 5C).

---

## 3. Logistic Regression Baseline

The [`LogisticRegressionReturnModel`](file:///c:/Users/avijeet.jaiswal/Documents/GitHub/Project/eRetail-AI-Intelligence/src/commerce_ai/returns/prediction_models.py#L208-L318) provides a transparent, interpretable, linear probabilistic benchmark.

### Architecture & Preprocessing Pipeline
To guarantee reproducibility and strict separation between training and inference:
1. **Numeric Features (23 columns)**:
   - Median imputation via `SimpleImputer(strategy="median")`
   - Standardization to zero mean and unit variance via `StandardScaler()`
2. **Categorical Features (5 columns: `channel_id`, `warehouse_id`, `category_id`, `brand`, `velocity_tier`)**:
   - Missing level imputation via `SimpleImputer(strategy="constant", fill_value="UNKNOWN")`
   - Sparse-safe one-hot encoding via `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
3. **No Leakage Guarantee**: The entire `ColumnTransformer` is fitted exclusively on $X_{train}$. Unseen categories in validation or test splits are encoded as all zeros without raising errors.
4. **Regularization & Optimization**:
   - $L_2$ penalty with inverse regularization strength $C = 1.0$
   - L-BFGS quasi-Newton solver with deterministic random seed `random_state = 42`

---

## 4. LightGBM Baseline

The [`LightGBMReturnModel`](file:///c:/Users/avijeet.jaiswal/Documents/GitHub/Project/eRetail-AI-Intelligence/src/commerce_ai/returns/prediction_models.py#L321-L476) implements a conservative, gradient-boosted decision tree classifier capable of capturing non-linear interactions and threshold effects (e.g., steep discounts combined with certain apparel categories).

### Architecture & Categorical Handling
1. **Native Categorical Partitioning**:
   - Categorical features are encoded into integer codes using `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` fitted exclusively on $X_{train}$.
   - Features are passed to LightGBM with explicit `categorical_feature` specification, allowing the tree builder to find optimal histogram splits over categorical subsets without high-dimensional one-hot explosion.
2. **Conservative Hyperparameters**:
   - `n_estimators = 100`, `learning_rate = 0.05`, `num_leaves = 31`, `max_depth = 6`
   - `subsample = 0.8`, `colsample_bytree = 0.8`, `min_child_samples = 20`
3. **Early Stopping**:
   - Monitored strictly on the validation partition ($X_{val}, y_{val}$) with `stopping_rounds = 10`.
   - The holdout test set is never exposed to early stopping or hyperparameter feedback.

---

## 5. Class Imbalance Strategy

In the canonical dataset, returns represent **8.10%** of all sold order lines ($62,630$ returns vs. $710,925$ non-returns). Standard unweighted empirical risk minimization would result in models that trivially predict 0 for all items, achieving 91.90% nominal accuracy while failing completely at business risk detection.

### Implemented Strategy
- **Configurable Class Weighting**:
  - **Logistic Regression**: `class_weight="balanced"`, adjusting sample weights inversely proportional to class frequencies:
    $$w_c = \frac{N}{2 \cdot N_c}$$
  - **LightGBM**: `class_weight="balanced"`, ensuring the gradient and hessian contributions of positive examples are weighted proportionally to their rarity.
- **Explicit Metadata Documentation**: The weighting strategy is explicitly recorded in [`ReturnModelMetadata.class_weight_strategy`](file:///c:/Users/avijeet.jaiswal/Documents/GitHub/Project/eRetail-AI-Intelligence/src/commerce_ai/returns/schemas.py#L716).
- **Prohibition of Synthetic Sampling**: Synthetic oversampling techniques (such as SMOTE) are strictly avoided because synthetic interpolation in high-dimensional mixed categorical/numeric commerce data introduces artificial feature correlation artifacts.

---

## 6. Chronological Evaluation

Omnichannel retail processes are subject to non-stationary demand shifts, promotional calendar cycles, and product assortment updates. Standard $K$-fold cross-validation with random shuffling introduces severe temporal lookahead leakage.

### Split Methodology
Data is ordered chronologically by sale date and split into three contiguous partitions:
1. **Train Set (70%)**: Earliest chronological partition ($541,488$ rows; `2024-01-01` to `2025-06-03`). Used exclusively to fit preprocessing encoders, scalers, and model parameters.
2. **Validation Set (15%)**: Intermediate chronological partition ($116,033$ rows; `2025-06-03` to `2025-09-17`). Used for tuning early stopping, evaluating probability calibrations, and conducting threshold trade-off scans.
3. **Holdout Test Set (15%)**: Most recent chronological partition ($116,034$ rows; `2025-09-17` to `2025-12-30`). Used strictly for final unbiased reporting.

---

## 7. ROC-AUC vs. PR-AUC

In imbalanced classification ($P(y=1) \approx 0.081$), evaluating models solely by the Receiver Operating Characteristic (ROC-AUC) can be misleading:
- **ROC-AUC**: Measures the trade-off between True Positive Rate ($\text{Recall}$) and False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$). Because $\text{TN}$ is massive ($> 710,000$), a large influx of false positives results in only a tiny increase in $\text{FPR}$, artificially inflating ROC-AUC.
- **PR-AUC (Precision-Recall AUC)**: Evaluates Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) directly against Recall ($\frac{\text{TP}}{\text{TP} + \text{FN}}$). PR-AUC ignores true negatives and focuses exclusively on the quality of positive return identifications.
- **Baseline Expectation**: The baseline PR-AUC for a random classifier equals the positive class prevalence ($0.0810$). Both models achieve $\text{PR-AUC} \approx 0.160–0.175$, doubling the random baseline.

---

## 8. Why Accuracy is Not the Primary Metric

Accuracy ($\frac{\text{TP} + \text{TN}}{\text{Total}}$) is an unsuitable evaluation criterion for return prediction:
- A naive trivial model that predicts $\hat{y} = 0$ for every line item achieves an **accuracy of 91.90%**.
- However, this model identifies $0$ returns, generates $0$ risk alerts, and provides zero business utility.
- Therefore, primary model ranking relies on **PR-AUC**, **ROC-AUC**, and **Brier score**, while operational classification relies on **Precision**, **Recall**, and **F1** at business-relevant thresholds.

---

## 9. Brier Score and Probability Quality

The Brier score measures the mean squared error between the predicted probability $\hat{p}_i \in [0.0, 1.0]$ and the binary indicator $y_i \in \{0, 1\}$:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

- **Calibration Importance**: In supply chain and reverse logistics, downstream actions (such as flagging orders for physical inspection, offering sizing assistance, or delaying automated refunds) depend on calibrated probabilities, not binary decisions.
- **Current Baseline Performance**: Both models achieve a Brier score of $\approx 0.211–0.216$. Because balanced class weighting intentionally shifts output probabilities to increase sensitivity, formal probability calibration (e.g. Platt scaling, isotonic regression) will be addressed in Phase 5C-2B.

---

## 10. Why Test Data is Not Used for Model Selection

- The holdout test set represents future, unseen transactions (`2025-09-17` to `2025-12-30`).
- If hyperparameters, decision thresholds, or model choices were adjusted based on test set metrics, the test set would be compromised by optimization bias.
- All model exploration, threshold scanning, and parameter validation are performed strictly on the validation set.

---

## 11. Why Business Threshold Selection is Deferred

The default threshold of $0.50$ is used strictly for standardization and reporting. In enterprise ecommerce:
- A conservative business threshold (e.g. $0.65–0.75$) may be desirable if the intervention involves a costly human review or contact with the customer.
- An aggressive threshold (e.g. $0.35–0.45$) may be preferred if the intervention is low-cost and high-leverage (such as prompting an automated digital size recommendation or selecting closer fulfillment routing).
- Threshold selection requires a formal **cost-utility matrix** (cost of false positive vs. cost of false negative), which will be developed in the business risk recommendation layer.

---

## 12. Known Limitations

1. **Probability Distortion under Class Weighting**: Using `class_weight="balanced"` inflates the raw predicted probabilities above the empirical base rate ($8.10\%$), requiring downstream calibration.
2. **Cold-Start Slices**: For newly listed SKUs with fewer than 10 historical orders, model predictions rely heavily on category-level historical averages, yielding lower discriminative resolution.
3. **Static Catalog Information**: Product dimension attributes (e.g. detailed size charts, fabric stretch specifications) are currently summarized by category and brand rather than textual descriptions.
