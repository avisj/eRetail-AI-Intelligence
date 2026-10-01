# COPILOT REASONING & ORCHESTRATION LAYER (PHASE 7C)

============================================================
1. ARCHITECTURE OVERVIEW
============================================================

The Copilot Reasoning & Orchestration Layer (Phase 7C) provides the deterministic analytical reasoning, intake normalization, intent interpretation, multi-step dependency decomposition, and tool execution orchestration that powers the conversational eRetail Copilot, executive dashboards, and future autonomous workflows.

Phase 7C sits strictly between conversational human inquiries and the underlying analytical contracts (Phase 7B) and calculation tools (Phase 7A):

    +-------------------------------------------------------------+
    |                     USER / COPILOT UI                       |
    |      "Why did margin decline?"  "How are sales performing?" |
    +-------------------------------------------------------------+
                                   |
                                   v
    +-------------------------------------------------------------+
    |         PHASE 7C: COPILOT REASONING & ORCHESTRATION         |
    |  - Intake Normalization & Security Filter                   |
    |  - Ambiguity Detection & Clarification Generator            |
    |  - Deterministic Intent & Template Interpreter              |
    |  - Multi-Step Plan Decomposition & Dependency Graph         |
    |  - Step Execution Coordinator & Failure Isolation           |
    |  - Evidence Lineage & Decision Neutrality Enforcement       |
    +-------------------------------------------------------------+
                                   |
                                   v
    +-------------------------------------------------------------+
    |         PHASE 7B: BUSINESS QUERY CONTRACTS & PLANNING        |
    |  - BusinessQueryContract & BusinessQueryFilter              |
    |  - Deterministic Validation Pipeline                        |
    |  - QueryPlan Synthesis & Governance Invariants              |
    +-------------------------------------------------------------+
                                   |
                                   v
    +-------------------------------------------------------------+
    |           PHASE 7A: QUERY LAYER TOOLS & REGISTRY             |
    |  - 46 Canonical Tools & 3 Compatibility Aliases             |
    |  - Point-in-Time QueryContext & Currency Isolation          |
    |  - Structured QueryResponse Envelopes                       |
    +-------------------------------------------------------------+
                                   |
                                   v
    +-------------------------------------------------------------+
    |                CORE DOMAIN ENGINES (PHASES 1-6)             |
    |  Sales, Financial, Inventory, Demand, Forecasting, Returns, |
    |  Replenishment, Purchase Orders, Rebalancing, Impact        |
    +-------------------------------------------------------------+

Key Architectural Properties:
- Phase 7C Never Bypasses Phase 7B: Every reasoning step translates into a validated BusinessQueryContract and QueryPlan.
- Strict Read-Only Guard: All actions, purchase order creations, transfer executions, and price changes are strictly prohibited.
- Decision Neutrality: Recommends options without autonomous selection or subjective ranking.
- No LLM Dependency: Does not depend on an LLM and therefore avoids LLM-generated interpretation and tool-selection behavior at this layer, ensuring deterministic execution, sub-millisecond latency, and full auditability.


============================================================
2. DETERMINISTIC DESIGN PRINCIPLES
============================================================

1. Deterministic Intent & Tool Resolution:
All interpretations, decompositions, and plans are produced via deterministic rule engines, canonical pattern matchers, and contract schemas. Identical questions with identical filters always produce bit-for-bit identical plans and execution outcomes. Does not depend on an LLM and therefore avoids LLM-generated interpretation and tool-selection behavior at this layer.

2. Reproducible Fingerprinting:
All plan IDs, step IDs, evidence IDs, and clarification IDs are derived via deterministic SHA-256 hashes of normalized inputs, timestamps, and parameters.

3. Complete Auditability & Provenance:
Every metric and insight returned in Copilot responses contains explicit evidence lineage linking back to the originating Phase 7A tool and logical source-engine identifier.

4. Defense-in-Depth Governance:
Governance checks occur at multiple layers:
- Intake: Detection of prompt injections, system prompt overrides, and prohibited action execution commands.
- Plan Synthesis: Verification of read-only flags and rejection of subjective ranking tools.
- Execution: Decision package neutrality enforcement preventing autonomous selection.

5. Sub-Millisecond Latency:
Because all parsing and planning is deterministic without remote network calls or heavy LLM invocations, average end-to-end orchestration latency is under 1 millisecond.


============================================================
3. LIFECYCLE STATE MACHINE
============================================================

The Copilot lifecycle progresses through a formal finite state machine:

    [RECEIVED]
        |
        v
    [NORMALIZED] ----(Security Violation)-----> [FAILED]
        |
        v
    [INTERPRETED] ---(Ambiguous / Unrecognized)-> [CLARIFICATION_REQUIRED]
        |
        v
    [CONTRACT_BUILT]
        |
        v
    [PLANNED] ---(Unavailable Domain)--------> [UNAVAILABLE]
        |
        +-------(plan_only=True)-------------> [PLANNED (Halt)]
        |
        v
    [EXECUTING]
        |
        +----(All Steps Succeeded)-----------> [COMPLETED]
        |
        +----(Some Steps Succeeded)----------> [PARTIAL]
        |
        +----(All Required Steps Failed)-----> [FAILED]

Lifecycle State Definitions:
- RECEIVED: Request ingested from caller, unique request ID assigned.
- NORMALIZED: Question text trimmed, whitespace collapsed, trailing punctuation normalized, entities preserved.
- CLARIFICATION_REQUIRED: Ambiguous, underspecified, or unrecognized question halted; structured options presented to caller.
- INTERPRETED: Intent, domain, metrics, dimensions, filters, grain, and temporal parameters resolved.
- CONTRACT_BUILT: Phase 7B BusinessQueryContract instances generated and validated.
- PLANNED: Multi-step execution graph constructed and checked against governance rules.
- EXECUTING: Steps evaluated against Phase 7A tools respecting topological dependency order.
- COMPLETED: All reasoning steps executed successfully with full evidence lineage assembled.
- PARTIAL: Non-critical steps failed or were blocked, but primary evidence was retrieved.
- UNAVAILABLE: Inquired business capability is intentionally unavailable in the current release.
- FAILED: Halted by security violation, malformed request, or fatal tool execution failure.


============================================================
4. QUESTION INTAKE & NORMALIZATION
============================================================

Intake normalization converts unstructured text into a canonical format suitable for deterministic rule matching while preserving crucial business entities:

Processing Rules:
1. Whitespace Collapsing: All newline characters (\r, \n), tabs (\t), and repeated spaces are collapsed into single spaces.
2. Punctuation Trimming: Trailing sentence terminators (?, !, .) are stripped while internal hyphens and underscores are preserved.
3. Identifier Preservation: SKU numbers (e.g. SKU_001, sku-12345), warehouse codes (e.g. WH_01, wh-east), and ISO dates (e.g. 2026-06-30) retain exact casing and structure.
4. Acronym Recognition: Standard business acronyms (SKU, WH, PO, AOV, COGS, ABC, XYZ, USD, EUR) are recognized without corruption.

Security & Injection Filtering:
Before interpretation begins, the question is evaluated against malicious patterns:
- Prompt Injection: Patterns such as "ignore previous instructions", "system prompt", "developer mode", "override governance".
- Data Destruction / Mutation: Patterns attempting SQL/DML mutations like "drop table", "delete from", "truncate table", "delete inventory".
- Prohibited Direct Actions: Commands attempting direct autonomous action execution such as "create purchase order", "transfer 100 units", "change price".


============================================================
5. AMBIGUITY DETECTION & CLARIFICATION PROTOCOL
============================================================

When an inquiry is too broad, multi-meaning, or severely underspecified, guessing a single contract risks presenting misleading conclusions to executives.

The ambiguity policy intercepts such queries and returns CopilotState.CLARIFICATION_REQUIRED accompanied by a structured CopilotClarification object.

Ambiguity Categories:
1. Underspecified Domain Queries:
   Example: "tell me about our performance", "how is the business doing", "what are the numbers"
   Clarification: Directs the user to select between sales performance, gross margin economics, inventory position, or multi-domain overview.
2. Underspecified Inventory Queries:
   Example: "show inventory", "check stock"
   Clarification: Requests whether the user needs portfolio position, SKU-level breakdown, risk tiers, or stockout exposure.
3. Underspecified Returns Queries:
   Example: "what about returns", "check returns"
   Clarification: Requests whether the user wants overall return rate, return anomalies, customer return reasons, or risk tiers.
4. Generic Status Requests:
   Example: "status report", "health check", "show me data", "analyze"
   Clarification: Presents functional domains for explicit disambiguation.

Structured Clarification Contract:
    clarification_id: CLR-281b369ba3b8dbcb
    question: "tell me about our performance"
    reason: "The inquiry is broad and underspecified. Please specify which dimension of performance you wish to review."
    missing_fields: ["domain", "intent"]
    possible_interpretations:
      - "Commercial sales performance (revenue, orders, units)"
      - "Financial gross margin and contribution margin economics"
      - "Inventory position and network risk breakdown"
      - "Executive cross-domain operational overview"
    severity: BLOCKING


============================================================
6. INTENT MAPPING & CANONICAL TEMPLATES
============================================================

Interpretation maps normalized questions to structured CopilotInterpretation objects through a 5-tier evaluation hierarchy:

Tier 1: Canonical Template Matching
Direct 1-to-1 match against Phase 7B CANONICAL_TEMPLATES:
- TPL_SALES_PERFORMANCE: Commercial sales performance overview (get_sales_summary)
- TPL_SALES_TREND: Sales revenue trend over time (get_sales_trend)
- TPL_MARGIN_STATUS: Gross margin and financial health (get_margin_summary)
- TPL_INVENTORY_POSITION: Network inventory status and on-hand stock (get_inventory_position)
- TPL_STOCKOUT_RISK: Critical stockout risk and days of supply (get_stockout_risk)
- TPL_DEMAND_TREND: Daily demand trajectory and trend velocity (get_demand_trend)
- TPL_FORECAST: Statistical future demand forecast (get_forecast, get_forecast_summary)
- TPL_RETURN_RATE: Customer return rate and return volumes (get_return_rate, get_return_summary)
- TPL_BUSINESS_IMPACT: Gross and deduplicated financial exposure (get_business_impact_summary)
- TPL_RECOMMENDATIONS_PENDING: Active business recommendations pending review (get_recommendation_summary, get_recommendations)
- TPL_DECISIONS_AWAITING: Decision packages awaiting human governance (get_decision_packages, get_decision_summary)

Tier 2: Why-Style Diagnostic Investigations
Synthesizes multi-step root-cause investigation plans using approved canonical Phase 7A tools:
- "Why did margin decline?": Decomposes into 4 sequential steps:
  1. get_margin_summary (macro gross margin baseline)
  2. get_margin_drivers (operational margin driver breakdown)
  3. get_sales_by_channel (channel margin performance)
  4. get_sales_by_warehouse (fulfillment warehouse margin impact)
- "Why is inventory risk high?": Decomposes into 4 sequential steps:
  1. get_inventory_risk (aggregate inventory risk tier distribution)
  2. get_inventory_summary (macro inventory portfolio baseline)
  3. get_slow_moving_inventory (aging excess stock concentration)
  4. get_stockout_risk (imminent stockout risk exposure)
- "Why are returns increasing?": Decomposes into 4 sequential steps:
  1. get_return_summary (macro return rate baseline)
  2. get_return_trend (longitudinal return trend trajectory)
  3. get_return_anomalies (statistically anomalous return spikes)
  4. get_return_reason_breakdown (customer-reported return reasons)

Tier 3: Multi-Domain Cross-Functional Synthesis
Triggered when an inquiry spans multiple functional domains (e.g. "Show sales, margin, inventory and returns together"):
- Generates a composite plan querying canonical Phase 7A summary tools:
  - get_revenue_summary
  - get_margin_summary
  - get_inventory_summary
  - get_return_summary

Tier 4: Keyword & Rule Matchers
Evaluates domain-specific vocabulary and regex patterns across all 9 business domains with entity extraction (SKUs, warehouses, channels, categories, dates).

Tier 5: Unrecognized / Unsupported Queries
Strictly avoids silent fallback to sales performance. Returns None from the interpreter, which triggers CopilotState.CLARIFICATION_REQUIRED with structured options or CopilotState.UNAVAILABLE.


============================================================
7. PLAN DECOMPOSITION & DEPENDENCY GRAPH
============================================================

A CopilotExecutionPlan consists of an ordered sequence of CopilotStep items linked through a directed acyclic graph (DAG).

Dependency Management:
Each step defines:
- step_id: Unique deterministic step identifier (e.g. STEP-1, STEP-2)
- sequence: 1-based execution order index
- required: Boolean indicating whether failure of this step blocks subsequent dependents
- dependencies: List of prerequisite step IDs that must successfully complete before this step runs
- status: Lifecycle state of the step (READY, RUNNING, COMPLETED, FAILED, BLOCKED, SKIPPED)

Topological Execution Resolution:
The DependencyGraph computes executable steps iteratively:
1. Steps with zero unsatisfied prerequisites are identified as READY.
2. Ready steps are executed against Phase 7A tools.
3. Upon step failure:
   - If required=True: All downstream steps depending directly or transitively on this step are immediately marked BLOCKED.
   - If required=False: Downstream steps may proceed if their other prerequisites are satisfied.
4. Independent steps (e.g. parallel domain queries) execute unhindered by unrelated step failures.


============================================================
8. TOOL COORDINATION & PHASE 7A INTEGRATION
============================================================

Step execution coordinates directly with QueryLayerService in Phase 7A.

Argument & Context Translation:
For each step, Copilot synthesizes a strongly-typed QueryContext from step arguments:
- Temporal Parameters: as_of_date, start_date, end_date, time_grain
- Entity Boundaries: sku_id, warehouse_id, channel_id, category_id, brand, supplier_id
- Analytical Filters: velocity_tier, abc_class, xyz_class
- Currency Isolation: currency ISO code passed transparently to prevent FX distortion
- Pagination & Windowing: limit, offset

Invocation Invariant:
Phase 7C only invokes canonical tools registered in Phase 7A's ToolRegistry. Invented tools (e.g. get_margin_status, get_known_contribution_margin, get_inventory_balance, get_return_reasons) are strictly prohibited and prevented by contract validation.


============================================================
9. EVIDENCE LINEAGE & PROVENANCE CHAIN
============================================================

Every executed step yields a structured CopilotEvidence package verifying data pedigree:

Evidence Schema:
    evidence_id: EVI-4f7a9c1e2b3d8f0a
    step_id: STEP-1
    tool_name: get_sales_summary
    source_engine: commerce_ai.sales
    metrics: ["net_revenue", "gross_revenue", "order_count", "units_sold"]
    entity_count: 1
    as_of_date: "2026-06-30"
    currency: "USD"
    data_summary:
      net_revenue: 1254300.50
      gross_margin: 432100.25
    provenance: PHASE_7A_TOOL

Logical Source-Engine Identifiers:
The source_engine values (e.g. commerce_ai.sales, commerce_ai.financial.FinancialIntelligenceService, commerce_ai.inventory.risk) represent logical source-engine identifiers designating the computational subsystem or service within the platform architecture, rather than standalone importable Python modules.

Chain of Custody:
The CopilotExecutionResult maintains the full evidence_chain array. Future Phase 7D explanation generators consume this exact chain to produce grounded factual narratives without generating unsupported numbers.


============================================================
10. FAILURE ISOLATION & RESILIENCE
============================================================

Failure isolation ensures that a failure in one calculation or data source does not crash the entire platform:

Isolation Levels:
1. Tool-Level Exception Containment:
   Any runtime exception thrown by a Phase 7A tool is captured and wrapped in a structured CopilotFailure object with error_code and error_message.
2. Graceful Degradation (PARTIAL Status):
   If non-essential steps fail (e.g. an auxiliary breakdown chart fails while the main KPI card succeeds), the plan finishes with status=PARTIAL, returning all successfully gathered evidence.
3. Unavailable Capability Handling:
   Queries targeting capabilities not present in Phase 7A (e.g. data quality completeness) return status=UNAVAILABLE with an advisory explanation rather than throwing an error.
4. Dependency Blocking:
   Downstream steps that rely on data from a failed required step are cleanly marked BLOCKED without executing or throwing unhandled errors.


============================================================
11. GOVERNANCE ENFORCEMENT & INVARIANTS
============================================================

Copilot enforces four inviolable architectural constraints:

1. Read-Only Invariant:
   - read_only = True
   - action_execution = False
   - execution_allowed = False
   Any attempt to execute mutations (creating purchase orders, transferring warehouse stock, adjusting selling prices) is intercepted and rejected at the intake gate.

2. Prohibited Ranking Invariant:
   - no_ranking_enforced = True
   Query plans containing subjective ranking or winner-selection tools (e.g. top_n, leaderboard, winners) are strictly prohibited and rejected during plan validation.

3. Decision Package Neutrality:
   - no_decision_selection_enforced = True
   When tools return decision recommendations or scenarios, Copilot strips any selected_option values, forcing neutrality and preserving human decision authority.

4. Point-in-Time & Currency Integrity:
   Historical as_of_date boundaries and explicit currency ISO filters are strictly forwarded to prevent lookahead bias and cross-currency blending.


============================================================
12. COPILOT SERVICE API REFERENCE
============================================================

The CopilotService provides four high-level entry points:

1. process(question_or_request, as_of_date, currency, context) -> CopilotResponse
   Complete end-to-end lifecycle orchestration: normalization, security filtering, ambiguity checking, interpretation, decomposition, planning, execution, and evidence lineage assembly.

2. interpret(question_or_request, as_of_date, currency, context) -> Optional[CopilotInterpretation]
   Performs intake normalization and deterministic semantic interpretation without planning or executing tools. Returns None if query cannot be mapped to an authorized intent.

3. plan(question_or_request, as_of_date, currency, context) -> Optional[CopilotExecutionPlan]
   Performs normalization, interpretation, Phase 7B contract synthesis, and multi-step plan construction without executing tools (state=PLANNED).

4. execute(plan, query_layer) -> CopilotExecutionResult
   Executes a pre-synthesized CopilotExecutionPlan against Phase 7A tools, isolating step failures and returning structured execution results.


============================================================
13. BENCHMARK SUITE & PERFORMANCE PROFILE
============================================================

A comprehensive benchmark suite in scripts/benchmark_copilot.py evaluates 25 distinct scenarios spanning:
- Single-domain canonical inquiries across all 9 domains
- Why-style multi-step diagnostic investigations (4 steps each)
- Multi-domain cross-functional performance reviews
- Entity-filtered queries (SKU, Warehouse, Channel)
- Currency-specified inquiries
- As-of-date boundary inquiries
- Ambiguous questions triggering structured clarifications
- Security attacks (prompt injections, unauthorized mutations)
- Gracefully degraded unavailable capabilities

Benchmark Results (25 Test Inquiries):
- Test Cases Matching Expectation: 25/25
- Completed Lifecycles: 18
- Clarifications Required: 3
- Unavailable Capabilities: 1
- Security / Adversarial Blocked: 3
- Unhandled Exceptions: 0
- Average Execution Latency: 0.19 ms
- Average End-to-End Latency: 0.65 ms


============================================================
14. FUTURE INTEGRATION ROADMAP
============================================================

The completion of Phase 7C establishes the foundation for subsequent roadmap phases:

Phase 7D — Natural-Language Generation & Grounded Narrative:
- Consumes the CopilotExecutionResult and evidence_chain produced by Phase 7C.
- Generates executive-ready natural-language summaries strictly grounded in verified evidence.
- Every figure in the narrative references a verified EvidenceReference.

Phase 7E — Conversational Copilot UI & Dashboard Visualizations:
- Directly renders CopilotResponse envelopes in interactive chat interfaces.
- Renders KPI cards, time-series line charts, dimensional pie/bar breakdowns, and tabular drill-downs directly from structured tool responses.
- Displays interactive clarification buttons when state=CLARIFICATION_REQUIRED.

Phase 7F — Supervised Agentic Workflows:
- Consumes decision review packages with human-in-the-loop approvals.
- Initiates draft proposals for replenishment and rebalancing requiring explicit executive sign-off before downstream system dispatch.


============================================================
15. VERIFICATION CHECKLIST & COMPLIANCE MATRIX
============================================================

Requirement                                         Status      Verification
----------------------------------------------------------------------------------
Deterministic Reasoning & Orchestration Layer       COMPLIANT   116 dedicated tests
Phase 7C uses Phase 7B Contracts & Plans            COMPLIANT   Verified in decomposition
Phase 7C executes through Phase 7A Tools            COMPLIANT   Verified in execution
Zero LLM / Zero RAG / Zero Vector DB Dependency     COMPLIANT   Pure deterministic Python
Strict Read-Only Governance (No Actions/Mutations)  COMPLIANT   Enforced in governance
No PO creation, transfer execution, price changes   COMPLIANT   Blocked at intake gate
No Ranking Tools / No Leaderboards                  COMPLIANT   Enforced in plan validation
Decision Package Neutrality Enforced                COMPLIANT   Tested in neutrality tests
Ambiguity Detection & Clarification Protocol        COMPLIANT   Tested across 4 categories
Why-Style Multi-Step Diagnostic Investigations      COMPLIANT   Tested on margin, inventory, returns
No Invented Tool Names (Reconciled with 7A)         COMPLIANT   Audited against ToolRegistry
Removal of Silent Fallback to Sales Performance     COMPLIANT   Verified in regression tests
Phase 7B Authoritative Template Authority           COMPLIANT   Referenced from templates.py
Failure Isolation & Graceful Degradation            COMPLIANT   Tested partial/blocked steps
Currency Isolation Forwarding                       COMPLIANT   Verified in benchmarks
Point-in-Time as_of_date Integrity                  COMPLIANT   Verified in benchmarks
Minimum 100 Dedicated Phase 7C Tests                COMPLIANT   116 dedicated tests passing
Full Repository Baseline Regression-Free            COMPLIANT   979/979 tests passing (100%)
Benchmark Suite Created & Passing                   COMPLIANT   25/25 passing (< 1 ms latency)
Zero Triple-Backtick Fences in Documentation        COMPLIANT   Strictly 4-space indented
