# Purchase Order Proposal Engine (Phase 4B-2)

## 1. Overview & Architectural Role

The **Purchase Order Proposal Engine** acts as the synthesis bridge between inventory intelligence / replenishment planning and external procurement operations. It ingests approved, actionable replenishment recommendations from the Replenishment Solver (Phase 4B-1) and consolidates them into orderable, supplier-facing **DRAFT Purchase Order Proposals**.

### 1.1 Architectural Flow
```
Sales Transactions
  ↓
Demand Intelligence (Reconstruction, Stockout Masking, ABC/XYZ)
  ↓
Forecasting Service (Statistical, Machine Learning, TimesFM)
  ↓
Inventory Intelligence Engine (Positions, Lead Times, Safety Stock, ROP, Risk)
  ↓
Replenishment Solver (Phase 4B-1: Sizing, MOQ, Pack Size, Safety Guards)
  ↓
PO Proposal Engine (Phase 4B-2: Grouping, Costing, Draft PO Assembly)  ← THIS MODULE
  ↓
Human Procurement Approval (Draft-Only Safety Guarantee)
  ↓
External OMS / ERP Integration (Future Lifecycle Phases)
```

### 1.2 Separation of Responsibilities
A fundamental design invariant of the platform is the strict separation between replenishment determination and purchase order formulation:

| Component | Core Operational Question | Key Output |
| :--- | :--- | :--- |
| **Phase 4B-1: Replenishment Solver** | *"How much inventory should be replenished for each SKU at each warehouse?"* | `ReplenishmentRecommendation` (Item-level unconstrained and constrained sizing) |
| **Phase 4B-2: PO Proposal Engine** | *"What draft vendor purchase-order proposals should represent that replenishment?"* | `PurchaseOrderProposal` (Supplier-grouped multi-line draft procurement documents) |

The PO Proposal Engine **never** recalculates safety stock, ROP, MOQ, pack sizes, or replenishment quantities. It faithfully translates Phase 4B-1 outputs into procurement-ready draft proposals.

---

## 2. Multi-Line Grouping Logic

Replenishment recommendations are consolidated into draft purchase orders using a strict three-dimensional grouping key:

$$\text{Grouping Key} = (\text{supplier\_id}, \text{warehouse\_id}, \text{currency})$$

### 2.1 Grouping Invariants
1. **Never Merge Suppliers**: Items fulfilled by different vendors are strictly segregated into independent PO proposals.
2. **Never Merge Warehouses**: Ship-to destination facilities are preserved 1:1; multi-warehouse shipments cannot be mingled onto a single receiving dock order.
3. **Never Merge Currencies**: Items priced or contracted in different currency codes (e.g. USD, EUR, INR) form distinct financial proposals.
4. **Missing Currency Segregation**: Items without currency data are grouped together under `currency = None` without fabricating a default currency.

### 2.2 Consolidation Example
Consider three separate SKU recommendations:
- **SKU-A** $\to$ Supplier `SUPP_01`, Warehouse `WH_01`, Currency `USD`: 100 units @ $10.00
- **SKU-B** $\to$ Supplier `SUPP_01`, Warehouse `WH_01`, Currency `USD`: 200 units @ $20.00
- **SKU-C** $\to$ Supplier `SUPP_01`, Warehouse `WH_01`, Currency `USD`: 50 units @ $30.00

These automatically consolidate into:
```
Draft PO Proposal
├── Proposal ID: PROP_SUPP_01_WH_01_USD_A3B91C4D
├── Supplier: SUPP_01
├── Destination Warehouse: WH_01
├── Currency: USD
├── Total Quantity: 350 units
├── Total Value: $6,500.00
├── Lines:
│   ├── SKU-A: 100 units | Unit Cost: $10.00 | Line Value: $1,000.00
│   ├── SKU-B: 200 units | Unit Cost: $20.00 | Line Value: $4,000.00
│   └── SKU-C:  50 units | Unit Cost: $30.00 | Line Value: $1,500.00
├── Urgency: (Derived from highest line urgency)
├── Status: DRAFT
└── Approval Required: TRUE
```

---

## 3. Business Logic & Validation Rules

### 3.1 Supplier Validation
- A purchase order cannot exist without an identified vendor.
- Recommendations where `supplier_id` is missing, `None`, or empty are explicitly excluded from PO line creation.
- Excluded items are captured in `excluded_missing_supplier` for administrative auditing and procurement vendor assignment.

### 3.2 Actionable Recommendations Filtering
- Items with `recommendation_required = False` (e.g. healthy inventory above ROP or suppressed dormant items) are routed to `non_actionable_recommendations`.
- Items with `recommended_order_qty <= 0` are routed to `excluded_zero_quantity`.
- Only items where `recommendation_required = True` and `recommended_order_qty > 0` with a valid `supplier_id` generate PO proposal lines.

### 3.3 Financial Costing & Valuation Completeness
- **Line Value**:
  $$\text{estimated\_line\_value} = \text{quantity} \times \text{unit\_cost}$$
  If `unit_cost` is unavailable or `None`, `estimated_line_value = None`. Prices are never invented.
- **Proposal Total Value**:
  $$\text{total\_value} = \sum_{i \in \text{lines with cost}} \text{estimated\_line\_value}_i$$
- **Cost Data Status**:
  - `COMPLETE_COST_DATA`: All lines have valid `unit_cost` AND `currency` is explicit.
  - `PARTIAL_COST_DATA`: Some lines have cost data, but at least one line has `unit_cost = None` OR `currency = None`.
  - `NO_COST_DATA`: Zero lines have cost data.

### 3.4 Expected Delivery Date Projection
- Computed deterministically when reference `proposal_date` and supplier `lead_time_days` are available:
  $$\text{expected\_delivery\_date} = \text{proposal\_date} + \lceil \text{lead\_time\_days} \rceil$$
- **Consolidated Proposal Delivery Date**: The maximum delivery date across all constituent line items, representing full-order dock readiness.
- If lead time or proposal date is missing, delivery date is explicitly set to `None` without guessing.

### 3.5 Deterministic Urgency Propagation
Proposal-level urgency reflects the most urgent operational condition among its lines:

$$\text{Proposal Urgency} = \max_{\text{lines}} (\text{line urgency})$$

Where precedence is strictly ordered:
$$\text{CRITICAL} > \text{HIGH} > \text{MEDIUM} > \text{LOW}$$

For example, a proposal containing one `CRITICAL` line and three `MEDIUM` lines evaluates to `CRITICAL` urgency, alerting buyers to prioritize vendor execution.

---

## 4. Auditability, Traceability & Determinism

### 4.1 1:1 Source Traceability
- Every line item maintains a `source_recommendation_id` referencing the exact Phase 4B-1 recommendation.
- Every consolidated proposal exposes `source_recommendation_ids: List[str]`, enabling full backward lineage to the specific demand, forecast, safety stock, and inventory position that justified the purchase.

### 4.2 Deterministic Cryptographic IDs
Proposal IDs are generated using a deterministic SHA-256 hash digest over the grouping key, proposal date, and sorted source recommendation IDs:

$$\text{seed} = \text{supplier\_id} \,\|\, \text{warehouse\_id} \,\|\, \text{currency} \,\|\, \text{proposal\_date} \,\|\, \text{sorted(source\_ids)}$$
$$\text{proposal\_id} = \text{PROP\_}\{\text{supplier\_id}\}\_\{\text{warehouse\_id}\}[\_\{\text{currency}\}]\_\{\text{SHA256(seed)[:8]}\}$$

This guarantees 100% idempotence and reproducibility: executing the engine multiple times against the same recommendation set always yields identical proposal IDs and structures.

### 4.3 Deterministic Natural-Language Rationale
Proposals generate factual, template-driven natural language rationales (no LLMs):
> *"Draft PO proposal groups 3 replenishment recommendations for supplier SUPP-001 and warehouse WH-001. Total proposed quantity is 350 units across 3 SKUs. The highest replenishment urgency is HIGH."*

---

## 5. Draft-Only Safety & Human-in-the-Loop Model

The PO Proposal Engine is strictly non-autonomous:
- **`status`**: Set to `"DRAFT"` on every generated proposal.
- **`approval_required`**: Set to `True` without exception.
- **Zero External Side-Effects**: The engine never:
  - Calls external vendor or ERP APIs
  - Submits or issues real purchase orders
  - Modifies live inventory ledger balances
  - Mutates supplier catalog contracts

---

## 6. Service API & Integration

```python
from commerce_ai.recommendations.replenishment import ReplenishmentSolver
from commerce_ai.recommendations.purchase_orders import PurchaseOrderProposalService

# 1. Generate Replenishment Recommendations (Phase 4B-1)
solver = ReplenishmentSolver()
replenishment_result = solver.solve(
    inventory_analysis=inventory_analysis,
    products=products_df,
    suppliers=suppliers_df,
)

# 2. Generate Draft PO Proposals (Phase 4B-2)
po_service = PurchaseOrderProposalService()
proposal_result = po_service.generate(
    replenishment_input=replenishment_result,
    proposal_date="2024-03-01",
    products=products_df,
)

# Inspect consolidated proposals
print(f"Generated {proposal_result.proposal_count} draft proposals.")
for prop in proposal_result.proposals:
    print(f"PO: {prop.proposal_id} | Supplier: {prop.supplier_id} | Qty: {prop.total_quantity} | Value: {prop.total_value} {prop.currency}")
    print(f"  Urgency: {prop.urgency} | Status: {prop.status} | Human Approval: {prop.approval_required}")

# Flat DataFrame export for procurement review UI
proposals_summary_df = proposal_result.to_dataframe()
proposal_lines_df = proposal_result.lines_to_dataframe()
```

---

## 7. Limitations & Future OMS/ERP Integration

1. **Vendor Minimum Spend Aggregation**: Currently, individual line MOQs are enforced in Phase 4B-1. Supplier-level order value thresholds (e.g. $10,000 order minimum for free freight) and container volume constraints will be integrated during workflow policy optimization.
2. **ERP / OMS Connectors**: Subsequent integration phases will supply authenticated REST/EDI adapters (e.g. SAP, NetSuite, Shopify, Coupa) to transmit human-approved purchase orders.
3. **Approval Lifecycle Engine**: Future phases will implement human review actions (`APPROVE`, `REJECT`, `AMEND`) with electronic signature audit logs.
