# Business Query Contracts & Deterministic Planning (Phase 7B)

> [!IMPORTANT]
> Core Architectural Principles:
> Phase 7B is a deterministic contract and planning layer positioned directly above Phase 7A.
> Phase 7B answers: "WHAT BUSINESS QUESTION IS BEING ASKED?"
> Phase 7B does NOT contain an LLM, does NOT contain an AI agent, does NOT execute tools, does NOT execute business actions, and does NOT implement RAG.
> It defines strongly-typed business query contracts, validation rules, intent taxonomies, and unexecuted execution blueprints (QueryPlan) for future Phase 7C orchestration.

---

## 1. Purpose

The Business Query Contract layer provides the formal semantic boundary between conversational AI / Dashboards and the underlying deterministic platform intelligence. Without this layer, Large Language Models would be forced to guess arbitrary API calls, invent unsupported parameter combinations, or attempt dangerous execution mutations.

Phase 7B ensures that any business question—whether entered via a UI widget or phrased in natural language—must first resolve to a validated, immutable BusinessQueryContract before any tool execution can be considered.

---

## 2. Relationship to Phase 7A

Phase 7A established the platform's query layer:
- Phase 7A answers: "WHAT TOOLS DO WE HAVE?" (46 canonical read-only tools and 3 approved convenience aliases across 10 functional domains)
- Phase 7B answers: "WHAT BUSINESS QUESTION IS BEING ASKED?" (28 controlled intents, 12 business domains, and validated contracts)

The architectural sequence flows:

    User / Future eRetail Copilot
                  |
                  v
       Business Query Contract (Phase 7B)
                  |
                  v
           QueryPlan & ToolCalls (Phase 7B)
                  |
                  v
       Phase 7A Query Layer Execution (Phase 7A / Future 7C)
                  |
                  v
       Structured Evidence & Response Envelope
                  |
                  v
       Future Business Explanation Layer (Phase 7D)

---

## 3. BusinessQueryContract Schema

The primary contract represents a validated, serializable, immutable business question:

    contract = BusinessQueryContract(
        query_id="QRY-a1b2c3d4e5f67890",
        intent=QueryIntent.SALES_PERFORMANCE,
        domain=BusinessDomain.SALES,
        metrics=[MetricIdentifier.NET_REVENUE, MetricIdentifier.ORDERS],
        dimensions=[BusinessDimension.WAREHOUSE],
        filters=BusinessQueryFilter(currency="USD"),
        time_range=TimeRangeContract(as_of_date="2026-06-30"),
        requested_grain=BusinessGrain.WAREHOUSE,
        time_granularity=TimeGranularity.NONE,
        requested_output=RequestedOutput.BREAKDOWN,
        required_tools=["get_sales_by_warehouse"],
        governance=ContractGovernance(read_only=True, execution_allowed=False)
    )

Every contract carries a deterministic SHA-256 query ID, ensuring that identical semantic queries produce identical hashes.

---

## 4. Intent Taxonomy

The platform establishes 28 controlled canonical query intents:

1. SALES_PERFORMANCE: Top-level sales KPIs, volume, and commercial revenue.
2. REVENUE_ANALYSIS: Net and gross revenue reconciliation and trends.
3. MARGIN_ANALYSIS: COGS, gross margin, and margin ratios.
4. PROFITABILITY_ANALYSIS: Contribution margin and multi-factor profitability attribution.
5. UNIT_ECONOMICS_ANALYSIS: Variable fulfillment expenses, unit costs, and cost completeness.
6. INVENTORY_STATUS: Stock on hand, units, tied-up capital, and active SKUs.
7. INVENTORY_RISK: Network-wide inventory risk tier distributions.
8. STOCKOUT_ANALYSIS: Low stock coverage and projected stockout runouts.
9. SLOW_MOVING_INVENTORY: Aged and slow-moving capital exposure.
10. HIGH_VALUE_INVENTORY: High-unit-cost capital concentration.
11. DEMAND_ANALYSIS: Network demand velocity and intermittency profiles.
12. DEMAND_TREND: Historical unit demand progression.
13. ABC_XYZ_ANALYSIS: Revenue Pareto and variability segmentation.
14. FORECAST_ANALYSIS: Forward-looking demand predictions.
15. FORECAST_ACCURACY: Historical forecast precision (MAE, RMSE, WAPE).
16. FORECAST_BIAS: Tracking signals and systematic over/under-forecasting.
17. RETURN_ANALYSIS: Return order count, returned units, and return value.
18. RETURN_ANOMALY_ANALYSIS: Statistically anomalous return volume spikes.
19. RETURN_RISK: Products and categories with elevated return risk.
20. RETURN_REASON_ANALYSIS: Customer-reported return reason distribution.
21. REPLENISHMENT_REVIEW: Proposed replenishment orders awaiting human review.
22. PURCHASE_ORDER_REVIEW: Draft purchase orders awaiting human review.
23. WAREHOUSE_REBALANCING_REVIEW: Proposed inter-facility inventory transfers.
24. BUSINESS_IMPACT_ANALYSIS: Quantified gross, deduplicated, and capital exposure.
25. RECOMMENDATION_REVIEW: Phase 6G business recommendations pending approval.
26. DECISION_REVIEW: Phase 6H decision packages awaiting review.
27. DATA_QUALITY_ANALYSIS: Source cost completeness and audit metrics (capability unavailable in Phase 7A).
28. MULTI_DOMAIN_ANALYSIS: Structured multi-domain cross-functional queries.

---

## 5. Domain Taxonomy

Phase 7B establishes 12 business domains:
- SALES: Commercial sales, orders, and revenue channels
- FINANCIAL: Gross margin, unit economics, profitability
- INVENTORY: Positions, coverage, stockout risk, capital tiers
- DEMAND: Demand velocity, trend profiles, ABC/XYZ
- FORECASTING: Predictive demand models, accuracy, bias
- RETURNS: Return rates, anomalous spikes, root causes
- OPERATIONS: Replenishment review, purchase orders, rebalancing
- BUSINESS_IMPACT: Signal quantification, deduplication, capital exposure
- RECOMMENDATIONS: Candidate recommendation governance
- DECISIONS: Decision package trade-off governance
- DATA_QUALITY: Ingestion completeness and audit trails
- CROSS_DOMAIN: Multi-domain cross-functional analyses

---

## 6. Metric Taxonomy

All metrics are controlled enums mapping directly to Phase 7A / upstream engines:
- Commercial: GROSS_REVENUE, NET_REVENUE, AOV, UNITS, ORDERS
- Financial: GROSS_MARGIN, GROSS_MARGIN_PERCENT, KNOWN_CONTRIBUTION_MARGIN, FINAL_CONTRIBUTION_MARGIN, COST_COMPLETENESS
- Inventory: INVENTORY_UNITS, INVENTORY_VALUE, STOCKOUT_EXPOSURE, SLOW_MOVING_EXPOSURE, HIGH_VALUE_EXPOSURE, DAYS_OF_SUPPLY
- Demand: DAILY_DEMAND, DEMAND_TREND, VELOCITY, INTERMITTENCY, ABC_CLASS, XYZ_CLASS
- Forecast: FORECAST_VALUE, FORECAST_ACCURACY, FORECAST_BIAS, FORECAST_ERROR
- Returns: RETURN_RATE, RETURN_COUNT, RETURNED_UNITS, RETURN_REVENUE_EXPOSURE, RETURN_RISK, RETURN_ANOMALY_COUNT
- Impact: GROSS_SIGNAL_EXPOSURE, DEDUPLICATED_EXPOSURE, DEDUPLICATED_PHYSICAL_CAPITAL_EXPOSURE
- Governance: RECOMMENDATION_COUNT, DECISION_PACKAGE_COUNT, PENDING_REVIEW_COUNT, CONFLICT_COUNT, INSUFFICIENT_INFORMATION_COUNT

---

## 7. Dimension Taxonomy

Defines valid slice-and-dice dimensions: SKU, WAREHOUSE, CHANNEL, CATEGORY, BRAND, SUPPLIER, VELOCITY_TIER, ABC_CLASS, XYZ_CLASS, DATE, WEEK, MONTH, QUARTER.

Pairwise conflicts (e.g. requesting both DATE and MONTH in the same dimensional breakdown) are rejected deterministically during validation.

---

## 8. Time Contract

Controlled via TimeRangeContract:
- Relative presets: TODAY, LAST_7_DAYS, LAST_30_DAYS, LAST_90_DAYS, THIS_MONTH, PREVIOUS_MONTH, THIS_QUARTER, PREVIOUS_QUARTER, CUSTOM
- Anti-leakage invariant: end_date <= as_of_date is strictly enforced.
- Chronology invariant: start_date <= end_date is strictly enforced.

---

## 9. Comparison Contract

Defines requested comparative baselines: NONE, PREVIOUS_PERIOD, PREVIOUS_YEAR_PERIOD, CUSTOM_COMPARISON.
Phase 7B defines the requested comparison structure; Phase 7A / 7C executes the data retrieval.

---

## 10. Filter Contract

BusinessQueryFilter validates all filter parameters:
- Supports single and list filters: sku_id, warehouse_id, channel_id, category_id, brand, supplier_id, velocity_tier, abc_class, xyz_class, currency, limit, offset.
- Strict injection protection: RegEx filters reject SQL keywords (SELECT, DROP, UNION), scripting keywords (import, eval), and path traversal attempts (../).

---

## 11. Requested Grain vs Time Granularity

Phase 7B architecturally decouples the aggregation entity from the temporal reporting frequency:

1. BusinessGrain (Aggregation Entity):
   PORTFOLIO, SKU, SKU_WAREHOUSE, SKU_CHANNEL, WAREHOUSE, CHANNEL, CATEGORY, BRAND, SUPPLIER, DAY, WEEK, MONTH (time values retained for backward compatibility).

2. TimeGranularity (Temporal Frequency):
   NONE, DAY, WEEK, MONTH, QUARTER.

This separation allows contracts to express orthogonal multidimensional requests—such as daily trend series for a specific SKU (requested_grain=SKU, time_granularity=DAY).

---

## 12. Requested Output

Defines expected response format: KPI, TIME_SERIES, BREAKDOWN, TABLE, DETAIL, INSIGHT_EVIDENCE.
Phase 7B does NOT generate natural-language answers. INSIGHT_EVIDENCE requests structured evidence objects for later explanation layers.

---

## 13. Tool Mapping & Prohibition of Rankings

Deterministic mapping from (intent, dimensions, output) to Phase 7A canonical query tools:
- SALES_PERFORMANCE -> get_sales_summary (default), get_sales_trend (time-series), get_sales_by_sku (by SKU), etc.
- INVENTORY_RISK -> get_inventory_risk (canonical; alias get_inventory_risk_breakdown accepted)
- STOCKOUT_ANALYSIS -> get_stockout_risk
- REPLENISHMENT_REVIEW -> get_replenishment_reviews
- DATA_QUALITY_ANALYSIS -> [] (capability unavailable in Phase 7A registry; emits CAPABILITY_UNAVAILABLE_IN_PHASE_7A)

Strict Ranking Prohibition:
Under platform governance, no tool that computes subjective entity rankings (e.g. get_opportunity_ranking, top-N, winners, leaderboards) is ever mapped or permitted. Any contract requesting such tools is rejected with PROHIBITED_RANKING_TOOL_REQUESTED. Business Impact queries map exclusively to aggregate summary and exposure tools (get_business_impact_summary, get_business_impact_by_category, get_business_impact_by_type, get_physical_capital_exposure).

---

## 14. Multi-Domain Queries

Queries cutting across multiple business domains (e.g. "How are revenue, margin and inventory performing?") are represented as:
- intent = MULTI_DOMAIN_ANALYSIS
- domains = [FINANCIAL, INVENTORY]
- Deterministic tool resolution maps to the union of summary tools: [get_inventory_summary, get_margin_summary, get_revenue_summary].

---

## 15. WHY-Style Investigation Queries

Diagnostic questions (e.g. "Why is margin down?", "Why is inventory risk high?") produce multi-step investigation contracts:
- Margin Investigation:
  1. get_margin_summary (baseline margin KPIs)
  2. get_margin_drivers (waterfall variance analysis)
  3. get_sales_by_channel (commercial channel breakdown)
  4. get_sales_by_warehouse (fulfillment location breakdown)
- Inventory Risk Investigation:
  1. get_inventory_risk (risk tier distributions)
  2. get_inventory_summary (network stock overview)
  3. get_slow_moving_inventory (aging capital analysis)
  4. get_stockout_risk (stockout coverage analysis)

---

## 16. Action / Recommendation Questions

Action-oriented business questions ("What should we reorder?", "What purchase orders are planned?") are mapped strictly to informational review intents:
- REPLENISHMENT_REVIEW, PURCHASE_ORDER_REVIEW, WAREHOUSE_REBALANCING_REVIEW
- Governance flags guarantee: approval_required=True, execution_allowed=False, action_execution=False.
- No executable purchase orders or inventory transfers are created or dispatched.

---

## 17. QueryPlan

A deterministic, unexecuted execution plan (QueryPlan) containing:
- plan_id: Deterministic SHA-256 fingerprint (PLAN-<hash>)
- contract_id: Reference back to BusinessQueryContract (QRY-<hash>)
- steps: Ordered list of ToolCallStep instances with arguments and business purpose
- execution_order: Sequenced tool list (summaries first, deep dives secondary)
- dependencies: Step dependency graph
- status: PLANNED, or UNAVAILABLE for capabilities not exposed in Phase 7A

---

## 18. Validation Rules

Contracts are validated deterministically against 14 core rules:
1. Unknown intent rejection
2. Unknown metric rejection
3. Metric-intent incompatibility
4. Domain-intent mismatch
5. Unsupported aggregation grain
6. Temporal leakage (end_date > as_of_date)
7. Chronology ordering (start_date <= end_date)
8. Currency isolation enforcement
9. Incompatible dimension combinations
10. Action execution keyword rejection
11. Unregistered tool rejection
12. Ranking tool prohibition (PROHIBITED_RANKING_TOOL_REQUESTED)
13. SQL / code / path traversal injection rejection
14. Governance immutability assertion (read_only=True, execution_allowed=False)

---

## 19. Governance Guarantees

Every contract and plan strictly enforces:
- read_only = True
- execution_allowed = False
- action_execution = False
- Zero automated mutations, zero database writes, zero price changes.

---

## 20. Evidence Requirements

Contracts define what evidence is required from Phase 7A via EvidenceRequirement:
- Required metric identifiers
- Expected source engine namespaces
- Source record traceability (SIG-*, RULE-*, REC-*, DEC-*)

---

## 21. Provenance Standards

Contracts specify acceptable data pedigree via ConfidenceRequirement:
- DIRECT_OBSERVED: Raw transaction facts
- DETERMINISTIC_DERIVED: Exact accounting formulas
- MODEL_BASED: Statistical or ML models
- INSUFFICIENT_DATA: Explicit handling for missing or unobservable inputs

---

## 22. Unsupported Requests

Requests attempting mutations or non-standard actions are rejected with structured codes:
- ACTION_EXECUTION_ATTEMPT_REJECTED
- GOVERNANCE_MUTATION_FORBIDDEN
- GOVERNANCE_EXECUTION_FORBIDDEN
- UNREGISTERED_TOOL_REQUESTED
- PROHIBITED_RANKING_TOOL_REQUESTED

---

## 23. Deterministic Query IDs

Every contract receives a deterministic SHA-256 hash formatted as QRY-<hex> derived from its normalized canonical fields. Identical business questions yield identical query IDs.

---

## 24. Canonical Question Templates

The platform defines 24 standardized templates across all domains serving as reference test cases, documentation anchors, and contract blueprints.

---

## 25. Benchmark Performance

Representative contracts benchmarked on canonical synthetic datasets demonstrate sub-millisecond execution:
- Average contract validation latency: ~135.7 μs (0.136 ms)
- Average plan generation latency: ~69.9 μs (0.070 ms)
- 100% prohibited actions, ranking requests, and malicious injections blocked.

---

## 26. Testing & Quality Assurance

- 83 dedicated Phase 7B tests across 4 test suites:
  - tests/test_query_contracts.py (28 tests)
  - tests/test_query_contract_mapping.py (19 tests)
  - tests/test_query_contract_validation.py (24 tests)
  - tests/test_query_contract_planner.py (12 tests)
- 863 total regression tests passing across the repository with zero failures.

---

## 27. Phase 7C Integration

In Phase 7C (Copilot Reasoning & Orchestration), the QueryPlan produced by Phase 7B will be fed to an execution coordinator that invokes Phase 7A tools, aggregates response envelopes, and prepares structured evidence packages for Phase 7D natural language explanations.

---

## 28. Limitations & Non-Goals

Phase 7B explicitly does NOT:
- Implement an LLM reasoning loop or prompt parser
- Implement autonomous agent loops
- Execute tools or dispatch API calls
- Execute business actions (purchase orders, transfers, discounts)
- Compute subjective entity rankings (top-N, winners, leaderboards)
- Fabricate query tools for capabilities not present in Phase 7A (e.g. data quality analysis is explicitly marked unavailable)
- Implement RAG or vector databases
- Generate natural language text or summaries
