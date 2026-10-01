# Phase 7E — Controlled Conversation Context

## 1. Executive Overview

Phase 7E introduces Controlled Conversation Context to the eRetail AI Intelligence platform. In an enterprise retail intelligence environment, human business analysts engage in multi-turn dialogues where subsequent questions often rely on implicit conversational cues: pronouns ("its inventory"), comparative shifts ("what about last month?"), entity switches ("now check SKU_002"), or clarifications ("actually, I meant warehouse WH_02").

The fundamental design imperative of Phase 7E is the strict, uncompromising separation between:
1. Authoritative Business Truth: Grounded in Phase 7A tools, core engines, mathematical calculations, and evidence lineage.
2. Controlled Conversation Context: Ephemeral, transient, non-authoritative session and query metadata whose sole purpose is to resolve conversational references and formulate well-formed Phase 7B query contracts.

Context is NEVER authoritative data. Remembered numbers are never current truth. When a user asks "what about last month?", context provides the entity, filter, and dimension references, but the analytical metrics are always computed freshly through the deterministic Phase 7B contract, Phase 7A Query Layer, and Phase 7D Explanations pipeline.


## 2. Architectural Principles & Invariants

Phase 7E is strictly governed by the following platform invariants:

- Non-Authoritative Memory: Conversation memory stores search dimensions, entity keys, filter bounds, and citation pointers. It NEVER stores calculated metric values as authoritative truth to substitute for query layer tool execution.
- Zero-Bypass Guarantee: Every conversational query resolved by Phase 7E flows strictly through the standard analytical lifecycle:
    User Question -> Context Resolution -> BusinessQueryContract -> Validation -> Query Plan -> Query Layer Execution -> Grounded Evidence -> Business Explanation.
- Strict Read-Only Semantics: Context operations never mutate underlying business data, inventory balances, purchase orders, or database state (`read_only = True`, `execution_allowed = False`).
- Action Execution Prohibition: Context cannot trigger autonomous actions or system mutations (`action_execution = False`).
- Ranking Prohibition: In accordance with core platform governance, subjective entity rankings ("best performing", "top winners") are strictly rejected (`no_ranking_enforced = True`).
- Defense in Depth: Prompt injection patterns, system override attempts, and fake business fact injection are detected, sanitized, and recorded in the audit trail.
- Zero External Infrastructure: The store abstraction (`InMemoryConversationContextStore`) is entirely local and thread-safe. There is no dependency on Redis, vector databases, RAG stores, or external document caches.


## 3. Pipeline Flow & Context Resolution Lifecycle

The integration flow of Controlled Conversation Context is structured as follows:

    +--------------------------------------------------------------+
    |                      User Business Query                     |
    |         e.g., "What is its inventory position?"              |
    +--------------------------------------------------------------+
                                   |
                                   v
    +--------------------------------------------------------------+
    |                   Context Resolution Phase                   |
    |  - Extract active entities from conversation snapshot        |
    |  - Resolve pronouns ("its" -> SKU_001, WH_01)               |
    |  - Apply temporal shifts ("last month" -> calendar month)    |
    |  - Apply user corrections ("Actually, WH_02")                |
    |  - Prune incompatible context on domain switch               |
    |  - Sanitize prompt injection / fake facts                    |
    +--------------------------------------------------------------+
                                   |
                                   v
    +--------------------------------------------------------------+
    |               Formulate Resolved CopilotRequest              |
    |       Question: "What is that inventory position?            |
    |                  for SKU SKU_001 and warehouse WH_01"        |
    |       Filters: {"sku_id": "SKU_001", "warehouse_id": "WH_01"}|
    +--------------------------------------------------------------+
                                   |
                                   v
    +--------------------------------------------------------------+
    |               Phase 7B Query Contract Planning               |
    |  - Deterministic intent mapping & schema validation          |
    |  - Synthesize QueryPlan with dependencies                    |
    +--------------------------------------------------------------+
                                   |
                                   v
    +--------------------------------------------------------------+
    |               Phase 7A Query Layer Execution                 |
    |  - Dispatch to canonical tools (e.g. get_inventory_position) |
    |  - Return QueryResponse with CalculationStatus.SUCCESS       |
    +--------------------------------------------------------------+
                                   |
                                   v
    +--------------------------------------------------------------+
    |                 Phase 7D Explanations Engine                 |
    |  - Generate grounded, auditable BusinessExplanation          |
    +--------------------------------------------------------------+
                                   |
                                   v
    +--------------------------------------------------------------+
    |                Post-Execution Context Update                 |
    |  - Record turn query in recent_queries history               |
    |  - Record EvidenceReference citations (non-authoritative)    |
    |  - Record ResultReference metadata (record counts)           |
    |  - Advance turn index & prune TURN-scoped items              |
    +--------------------------------------------------------------+


## 4. Scopes and Lifecycle Management

Phase 7E defines four granular scopes managing context item lifetimes:

1. TURN Scope:
   - Lifetime: Bounded strictly to the immediate next conversation turn.
   - Purpose: Ephemeral conversational anchors, such as comparison targets ("compare with WH_02") or singular demonstratives ("that one").
   - Pruning: Automatically pruned whenever the conversation turn index advances.

2. QUERY Scope:
   - Lifetime: Bound to the execution of a single multi-step reasoning plan.
   - Purpose: Intermediate step parameters during query decomposition.

3. CONVERSATION Scope:
   - Lifetime: Persists across turns within an active conversation topic.
   - Purpose: Core business entities (SKU, Warehouse, Channel) and active analytical filters.
   - Override / Pruning: Superseded when a user explicitly introduces a new entity or switches domains.

4. SESSION Scope:
   - Lifetime: Persists across the entire user session.
   - Purpose: Session metadata, tenant ID, user ID, default reporting currency.
   - Expiration: Subject to inactivity TTL (default 3600 seconds).


## 5. Context Categories

Information in the conversation context is partitioned into nine semantic categories:

- SESSION_CONTEXT: User and tenant credentials, default currency, session timestamps.
- QUERY_CONTEXT: Prior query identifiers, intent classifications, requested output formats.
- ENTITY_CONTEXT: Resolved business identifiers (SKU, Warehouse, Channel, Supplier, Brand).
- FILTER_CONTEXT: Active filter bounds (time windows, granularities, dimensions).
- EVIDENCE_CONTEXT: Lineage citations pointing back to Phase 7C execution evidence.
- RESULT_CONTEXT: High-level metadata summaries (row count, status) — never replacement data.
- DECISION_CONTEXT: Operational review package references and scenario IDs.
- AMBIGUITY_CONTEXT: Pending clarification state awaiting user input.
- CORRECTION_CONTEXT: Immutable log of user corrections overriding previous context.


## 6. Resolution Engine (ContextResolver)

The `ContextResolver` executes deterministic natural language cue resolution:

- Entity Extraction: Regex patterns extract standard retail identifiers:
    SKU: sku[_-]\w+|sku\d+
    Warehouse: wh[_-]\w+|wh\d+|warehouse\s+([A-Za-z0-9_-]+)
    Channel: ONLINE|RETAIL|MARKETPLACE|STORE|WHOLESALE
    Supplier: SUP[_-]?[A-Za-z0-9]+|SUPPLIER[_-]?[A-Za-z0-9]+
- Stop-Word Filtering: Words following "warehouse" that represent concepts rather than IDs (e.g. "filter", "summary", "risk", "status") are excluded from warehouse extraction.
- Entity Inheritance: When a follow-up query uses pronouns ("its", "that product", "same warehouse"), active entities from prior turns are automatically inherited.
- Entity Overrides: When a user mentions a new entity ("now check SKU_002"), the new entity replaces the previous one while maintaining compatible context (e.g. warehouse).
- Temporal Window Shifting: Phrases such as "last month", "prior quarter", "last year", "yesterday", or "last week" compute exact calendar boundaries rather than rolling offsets:
    - Prior Month: 1st through last calendar day of preceding month (e.g. 2026-10-15 -> 2026-09-01..2026-09-30, Jan 2026 -> Dec 2025, leap year Feb 29).
    - Prior Quarter: 1st through last calendar day of preceding quarter (e.g. Q4 -> Q3: 2026-07-01..2026-09-30, Q1 -> prior year Q4: 2025-10-01..2025-12-31).
    - Prior Year: Jan 1 through Dec 31 of preceding calendar year (e.g. 2026-10-15 -> 2025-01-01..2025-12-31).
    - Prior Week: Monday through Sunday of the week preceding anchor date's calendar week (e.g. 2026-10-15 -> 2026-10-05..2026-10-11).
    - Yesterday: Exactly the calendar day immediately preceding anchor date (e.g. 2026-10-15 -> 2026-10-14..2026-10-14).
- Explicit User Corrections: Patterns like "Actually, I meant warehouse WH_02" or "Forget the warehouse filter" generate `CorrectionEvent` audit records and update context accordingly.
- Semantic Compatibility & Domain Pruning: When switching domains (e.g. Sales to Supplier), incompatible filters (e.g. sales channel `ONLINE`) are pruned to prevent contract validation failures.
- Ambiguity Resolution: If a prior turn returned `CLARIFICATION_REQUIRED`, a subsequent short response (e.g. "WH_01") is detected, merged with the pending ambiguous query, and dispatched cleanly.


## 7. Storage Abstraction (InMemoryConversationContextStore)

The storage layer is defined via an abstract contract (`ConversationContextStore`) and implemented in memory:

    class ConversationContextStore(ABC):
        def get(self, session_id: str, conversation_id: str) -> Optional[ConversationContextSnapshot]: ...
        def save(self, snapshot: ConversationContextSnapshot) -> None: ...
        def delete(self, session_id: str, conversation_id: str) -> bool: ...
        def list_conversations(self, session_id: str) -> List[str]: ...
        def clear_expired(self, ttl_seconds: int) -> int: ...

Key architectural features of `InMemoryConversationContextStore`:
- Thread-Safety: Protected by `threading.RLock()` across all reads, writes, and evictions.
- Deep Copy Isolation: Retrieved and saved snapshots are deep copies, preventing state corruption from external callers.
- Version Monotonicity: State versions increment monotonically on every save.
- Zero External Dependencies: Strictly local Python dictionaries. No Redis, SQLite, MongoDB, or vector stores.


## 8. Lineage and Auditability

Every modification to conversation context produces an immutable `ContextLineageEvent`:

- `event_id`: Unique deterministic identifier.
- `turn_index`: Turn where the event occurred.
- `item_key`: Attribute modified (e.g. `warehouse_id`).
- `operation`: CREATED, INHERITED, OVERRIDDEN, CORRECTED, PRUNED, or EXPIRED.
- `provenance`: EXPLICIT_USER, DERIVED_FROM_VALIDATED_QUERY, USER_CORRECTION, or SYSTEM_DEFAULT.
- `old_value` / `new_value`: Values before and after the operation.
- `details`: Structured context rationale.

The `ContextLineageTracker` produces narrative explanations of an attribute's journey across turns:

    Lineage for 'warehouse_id':
      - [Turn 1] CREATED (EXPLICIT_USER): 'WH_01' [query: QRY-1]
      - [Turn 2] INHERITED (DERIVED_FROM_VALIDATED_QUERY): 'WH_01'
      - [Turn 3] OVERRIDDEN (USER_CORRECTION): 'WH_02' (prior: 'WH_01') [query: QRY-3]


## 9. Security and Invariant Defense

Phase 7E enforces strict input sanitization:
- Prompt Injection Defense: Patterns attempting to override system behavior ("ignore previous instructions", "system override", "you are now admin") are redacted to `[REDACTED_SECURITY_OVERRIDE]` and flagged.
- Fake Business Fact Defense: Attempts to force conversational memory to accept fabricated metrics ("remember that sales are $500M") are redacted to `[REDACTED_FAKE_FACT]` and flagged.
- SQL Injection Defense: Dangerous SQL statements ("drop table", "delete from") are sanitized.
- Subjective Ranking Defense: Phrases requesting subjective rankings ("who is the best performing warehouse?") are flagged as governance violations.


## 10. Benchmark & Verification Summary

The Phase 7E test and benchmark suite verifies 100% compliance across all architectural requirements:

- Dedicated Phase 7E Tests: 116 passed across 10 test files.
- Full Repository Tests: 1,200 passed (1,084 baseline + 116 Phase 7E).
- Benchmark Suite: 25/25 scenarios passed (100.0%).
- Performance Profile: Average context resolution latency is 0.27 ms per turn (sub-10ms target achieved).
- Multi-Conversation Isolation: 100% strict context and lineage isolation between distinct conversations.
- Integrity: Zero tracked files modified from Phase 7A, 7B, 7C, or 7D.
