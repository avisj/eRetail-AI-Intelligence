# BUSINESS EXPLANATIONS LAYER (PHASE 7D)

============================================================
1. ARCHITECTURE & LAYER BOUNDARIES
============================================================

The Business Explanations Layer (Phase 7D) is the deterministic business intelligence synthesis and narrative generation system of the eRetail AI Intelligence platform.

Phase 7D consumes the execution results and structured evidence emitted by the Copilot Reasoning & Orchestration Layer (Phase 7C), verifies mathematical and provenance grounding against the canonical Phase 7A Query Layer and Phase 7B Contracts, and emits fully auditable, non-sensational, governance-compliant business explanations.

    +-------------------------------------------------------------+
    |                     USER / COPILOT UI                       |
    |      "Why did margin decline?"  "How are sales performing?" |
    +-------------------------------------------------------------+
                                   |
                                   v
    +-------------------------------------------------------------+
    |          PHASE 7D: BUSINESS EXPLANATIONS (AUDITABLE)        |
    |  - Deterministic Narrative Synthesizers & Template Builders |
    |  - Semantic Claim Auditor & Value-Consistency Verifier      |
    |  - Multi-Dimensional Evidence Lineage & Provenance Tracker  |
    |  - Zero-Ranking & Zero-Ungrounded-Causality Enforcement    |
    |  - Multi-Domain Cross-Functional Contextual Synthesizer     |
    +-------------------------------------------------------------+
                                   |
                                   v
    +-------------------------------------------------------------+
    |         PHASE 7C: COPILOT REASONING & ORCHESTRATION         |
    |  - Intake Normalization & Security Filter                   |
    |  - Deterministic Intent & Template Interpreter              |
    |  - Multi-Step Plan Decomposition & Dependency Graph         |
    |  - Step Execution Coordinator & Failure Isolation           |
    +-------------------------------------------------------------+
                                   |
                                   v
    +-------------------------------------------------------------+
    |         PHASE 7B: BUSINESS QUERY CONTRACTS & PLANNING        |
    |  - 28 Validated Canonical Query Intents                     |
    |  - Deterministic Validation Pipeline & QueryPlan Synthesis  |
    +-------------------------------------------------------------+
                                   |
                                   v
    +-------------------------------------------------------------+
    |           PHASE 7A: QUERY LAYER TOOLS & REGISTRY             |
    |  - 46 Canonical Read-Only Tools & 3 Compatibility Aliases   |
    |  - Point-in-Time QueryContext & Currency Isolation          |
    +-------------------------------------------------------------+
                                   |
                                   v
    +-------------------------------------------------------------+
    |                CORE DOMAIN ENGINES (PHASES 1-6)             |
    |  Sales, Financial, Inventory, Demand, Forecasting, Returns, |
    |  Replenishment, Purchase Orders, Rebalancing, Impact        |
    +-------------------------------------------------------------+

Key Architectural Guarantees:
- Strict Upstream Grounding: Every finding in a BusinessExplanation links to one or more ExplanationEvidence items originating from executed Phase 7A tool responses.
- Complete Lineage Pointers: Every evidence item retains source_tool, source_step_id, source_engine, metric, value, unit, and currency.
- Strict Read-Only Execution: The explanation layer possesses zero mutation capabilities. It cannot generate purchase orders, create inventory transfers, modify prices, or alter platform state.
- Decision Neutrality: Explanations provide factual context and evaluation of options without selecting decision options or assigning subjective entity rankings.
- No LLM Hallucination Risk: All finding statements, summaries, and headlines are generated through deterministic, template-driven logic and verified by the semantic claim auditor.


============================================================
2. CONTROLLED EXPLANATION TAXONOMY
============================================================

Phase 7D defines 15 canonical ExplanationType classes covering the complete spectrum of enterprise retail inquiry categories:

    1. KPI_EXPLANATION: Single or multi-metric point-in-time performance evaluations.
    2. TREND_EXPLANATION: Directional analysis of time-series observations over planning periods.
    3. BREAKDOWN_EXPLANATION: Dimensional distributions across channels, warehouses, categories, or reasons.
    4. COMPARISON_EXPLANATION: Variance analysis across consecutive fiscal periods or baseline targets.
    5. WHY_EXPLANATION: Multi-step diagnostic investigations combining multi-tool evidence.
    6. MULTI_DOMAIN_EXPLANATION: Cross-functional synthesis across commercial, inventory, and operational boundaries.
    7. RISK_EXPLANATION: Financial capital exposure and operational vulnerability assessments.
    8. FORECAST_EXPLANATION: Forward-looking statistical and machine-learning demand projections.
    9. RETURN_EXPLANATION: Customer return rates, volume trends, and defect code analyses.
    10. INVENTORY_EXPLANATION: Stock positions, holding valuation, slow-moving items, and stockout vulnerability.
    11. FINANCIAL_EXPLANATION: Revenue, gross margin, profitability attribution, and operational economics.
    12. OPERATIONAL_REVIEW_EXPLANATION: Replenishment proposals, purchase orders, and inter-facility transfers.
    13. DATA_QUALITY_EXPLANATION: Audit trails, data completeness, and catalog telemetry health.
    14. INSUFFICIENT_DATA_EXPLANATION: Structured notification when input constraints lack underlying observations.
    15. UNAVAILABLE_EXPLANATION: Graceful fallback when requested capabilities or analytical models are unconfigured.


============================================================
3. SEMANTIC GROUNDING & CLAIM AUDITOR
============================================================

The ClaimAuditor (commerce_ai.explanations.auditor) performs exhaustive semantic validation on every generated BusinessExplanation prior to client emission.

Validation Checks Executed:
- CHECK_GOVERNANCE_INVARIANTS: Verifies all governance invariant flags (read_only=True, action_execution=False, execution_allowed=False, no_ranking_enforced=True, no_decision_selection_enforced=True, no_causality_invented=True, no_recommendation_generated=True).
- CHECK_ALL_CLAIMS_GROUNDED_IN_EVIDENCE: Every finding must have at least one valid supporting ExplanationEvidence reference.
- CHECK_VALUE_CONSISTENCY: Every explicit numeric value appearing in a finding statement (including currency, integer units, percentages, and negative impacts) must match the value or comparison_value of its linked evidence items within floating-point tolerance.
- CHECK_ZERO_UNGROUNDED_CAUSALITY: Prohibits ungrounded causal assertions (e.g., 'caused by', 'caused the decline', 'the root cause is') unless supported by formal econometric or causal models.
- CHECK_ZERO_RANKINGS_OR_LEADERBOARDS: Prohibits subjective entity rankings, leaderboards, top-N, best/worst performing labels, and winner/loser terminology.
- CHECK_FORECAST_HONESTY_AS_MODEL: Ensures forward-looking projections are framed as statistical estimates, expectations, or projections rather than empirical historical facts.


============================================================
4. DOMAIN SEMANTIC RULES & GOVERNANCE
============================================================

Financial Semantics:
- Gross Revenue vs Net Revenue: Never conflated. Net revenue explicitly accounts for promotional discounts and operational allowances.
- Gross Margin Rate vs Gross Profit: Rates are preserved as percentages; absolute gross profit is denominated in the isolated context currency.
- Catalog Unit Cost vs GAAP Accounting COGS: Unit catalog costs reflect supplier contract values and are never mischaracterized as GAAP FIFO/LIFO cost of goods sold.

Inventory & Operational Semantics:
- Position Hierarchy: On-hand inventory, available stock, and net inventory position are strictly differentiated.
- Risk Separation: Excess slow-moving holding risks and stockout runout vulnerabilities are reported as concurrent catalog exposures without inventing causal links between them.
- Operational Reviews: Replenishment proposals, purchase order recommendations, and warehouse transfers are framed as draft proposals requiring explicit human authorization.

Forecasting Honesty:
- Model Projections vs Empirical Facts: Forecasts explicitly state their model provenance (e.g., MovingAverage, LightGBM, TimesFM), forecast horizon, and baseline nature.
- Confidence Tiering: Forecast statements carry MEDIUM confidence to distinguish model extrapolation from observed historical transactions.

Returns & Anomaly Governance:
- Returns Separation: Customer dissatisfaction codes (sizing, defects, late delivery) are reported as observed co-occurrences without ungrounded causal assertions.
- Statistical Spikes: Anomalies are reported with their statistical baseline deviations rather than attributed to unmodeled operational failures.


============================================================
5. PROVENANCE & CONFIDENCE TIERS
============================================================

Provenance Tiers:
- USER_PROVIDED: Filter criteria, entity IDs, time boundaries, and context currencies supplied by the requester.
- PHASE_7A_OBSERVED: Direct empirical observations retrieved from canonical query tools.
- PHASE_7B_DERIVED: Deterministically calculated metrics (variances, shares of total, summary aggregates).
- PHASE_7C_ORCHESTRATED: Synthesized cross-step findings across multi-step execution plans.
- MODEL_BASED: Forward-looking predictions and statistical anomaly flags generated by trained models.
- INSUFFICIENT_DATA: Pointers indicating missing or incomplete data pipelines.

Evidence Support Confidence Levels:
- HIGH: Supported by multi-source empirical observations or deterministic calculations with full lineage.
- MEDIUM: Supported by single-source observations or statistical forecast model projections.
- LOW: Supported by partial data or degraded historical time horizons.
- INSUFFICIENT: Missing critical observations; prevents fabrication of synthetic metrics.


============================================================
6. RENDERING FORMATS
============================================================

Business explanations can be rendered in multiple standard formats:
- Structured Pydantic Container: BusinessExplanation model providing type-safe programmatic access for API endpoints and dashboard widgets.
- Formatted Text / Markdown: Human-readable markdown executive briefing featuring clean bulleted findings, risks, and limitations.
- Executive Briefing String: High-level executive headline and summary suitable for mobile notifications and conversational Copilot interfaces.
