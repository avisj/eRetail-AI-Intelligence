"""Canonical Benchmark Script for Business Recommendation Engine (Phase 6G).

Evaluates the deterministic Business Recommendation Engine against the canonical
enterprise synthetic dataset in data/sample/ (773,555 sales rows, 292,500 inventory rows).

Reports ONLY aggregate distributions:
- Total recommendations
- Recommendations by type
- Recommendations by priority
- Recommendations by status
- Recommendations by confidence
- Recommendations requiring human review
- Conflict count & types
- Insufficient-data count
- Linked signal count
- Linked impact count

DO NOT output:
- Top 10 recommendations / SKUs / Warehouses
- Ranked entities or portfolio opportunity leaderboards
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import pandas as pd

from commerce_ai.cross_domain.service import CrossDomainIntelligenceService
from commerce_ai.business_impact.service import BusinessImpactService
from commerce_ai.recommendations.business_recommendations import BusinessRecommendationEngine
from commerce_ai.recommendations.business_schemas import (
    BusinessRecommendationConfig,
    RecommendationPriority,
    RecommendationType,
)


def run_benchmark():
    data_dir = Path("data/sample")
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        sys.exit(1)

    print("============================================================")
    print("PHASE 6G -- BUSINESS RECOMMENDATION ENGINE")
    print("CANONICAL DATASET BENCHMARK")
    print("============================================================")
    print(f"Loading canonical datasets from {data_dir.resolve()}...")

    t0 = time.perf_counter()
    products_df = pd.read_csv(data_dir / "products.csv")
    warehouses_df = pd.read_csv(data_dir / "warehouses.csv")
    sales_df = pd.read_csv(data_dir / "sales.csv")
    inventory_df = pd.read_csv(data_dir / "inventory.csv")
    returns_df = pd.read_csv(data_dir / "returns.csv") if (data_dir / "returns.csv").exists() else None
    suppliers_df = pd.read_csv(data_dir / "suppliers.csv") if (data_dir / "suppliers.csv").exists() else None
    t_load = time.perf_counter() - t0

    print(f"Loaded:")
    print(f"  - Products:   {len(products_df):>10,d} rows")
    print(f"  - Warehouses: {len(warehouses_df):>10,d} rows")
    print(f"  - Sales:      {len(sales_df):>10,d} rows")
    print(f"  - Inventory:  {len(inventory_df):>10,d} rows")
    if returns_df is not None:
        print(f"  - Returns:    {len(returns_df):>10,d} rows")
    if suppliers_df is not None:
        print(f"  - Suppliers:  {len(suppliers_df):>10,d} rows")
    print(f"Data loading completed in {t_load:.2f}s\n")

    # Step 1: Run Cross-Domain Intelligence (Phase 6E)
    print("Step 1: Running Cross-Domain Intelligence (Phase 6E)...")
    t1 = time.perf_counter()
    cd_service = CrossDomainIntelligenceService()
    cd_result = cd_service.analyze(
        sales=sales_df,
        inventory=inventory_df,
        products=products_df,
        warehouses=warehouses_df,
        returns=returns_df,
    )
    t_cd = time.perf_counter() - t1
    print(f"Generated {len(cd_result.records):,d} primary records and {len(cd_result.signals):,d} signals in {t_cd:.2f}s\n")

    # Step 2: Run Business Impact Quantification (Phase 6F)
    print("Step 2: Running Business Impact Quantification (Phase 6F)...")
    t2 = time.perf_counter()
    impact_service = BusinessImpactService()
    impact_result = impact_service.quantify_portfolio(
        signals=cd_result.signals,
        records=cd_result.records,
    )
    t_impact = time.perf_counter() - t2
    print(f"Quantified {len(impact_result.impact_records):,d} impact records in {t_impact:.2f}s\n")

    # Step 3: Run Business Recommendation Engine (Phase 6G)
    print("Step 3: Running Business Recommendation Engine (Phase 6G)...")
    t3 = time.perf_counter()
    rec_engine = BusinessRecommendationEngine()
    rec_result = rec_engine.generate_recommendations(
        signals=cd_result.signals,
        impact_result=impact_result,
        cross_domain_records=cd_result.records,
        products_df=products_df,
        inventory_df=inventory_df,
        sales_df=sales_df,
        suppliers_df=suppliers_df,
        returns_df=returns_df,
    )
    t_rec = time.perf_counter() - t3
    total_time = t_cd + t_impact + t_rec

    print("============================================================")
    print("CANONICAL BENCHMARK RESULTS (PHASE 6G)")
    print("============================================================")
    print(f"Total Sales Rows Processed:             {len(sales_df):>15,d}")
    print(f"Total Inventory Snapshots:              {len(inventory_df):>15,d}")
    print(f"Total Primary Records (SKU x WH):       {len(cd_result.records):>15,d}")
    print(f"Total Cross-Domain Signals Evaluated:   {len(cd_result.signals):>15,d}")
    print(f"Total Business Impacts Linked:          {len(impact_result.impact_records):>15,d}")
    print(f"Total Recommendations Emitted:          {rec_result.total_recommendations:>15,d}")
    print(f"Recommendations Requiring Human Review: {rec_result.human_review_count:>15,d} (100.0%)")
    print(f"Detected Operational Conflicts:         {rec_result.conflict_count:>15,d}")
    print(f"Insufficient Data Recommendations:      {rec_result.insufficient_data_count:>15,d}")
    print(f"Linked Signal Evidences:                {rec_result.linked_signal_count:>15,d}")
    print(f"Linked Business Impact Evidences:       {rec_result.linked_impact_count:>15,d}")
    print(f"Phase 6E Runtime:                       {t_cd:>15.2f} s")
    print(f"Phase 6F Runtime:                       {t_impact:>15.2f} s")
    print(f"Phase 6G Runtime:                       {t_rec:>15.2f} s")
    print(f"Total Pipeline Runtime:                 {total_time:>15.2f} s")
    print("------------------------------------------------------------")

    print("\nRECOMMENDATION DISTRIBUTION BY TYPE:")
    for rec_type, count in sorted(rec_result.recommendations_by_type.items(), key=lambda x: x[1], reverse=True):
        pct = (count / rec_result.total_recommendations * 100.0) if rec_result.total_recommendations > 0 else 0.0
        print(f"  {rec_type:<35} {count:>8,d} ({pct:>5.1f}%)")

    print("\nRECOMMENDATION DISTRIBUTION BY PRIORITY:")
    for pri in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH, RecommendationPriority.MEDIUM, RecommendationPriority.LOW, RecommendationPriority.INFO]:
        count = rec_result.recommendations_by_priority.get(pri.value, 0)
        pct = (count / rec_result.total_recommendations * 100.0) if rec_result.total_recommendations > 0 else 0.0
        print(f"  {pri.value:<15} {count:>8,d} ({pct:>5.1f}%)")

    print("\nRECOMMENDATION DISTRIBUTION BY STATUS:")
    for status, count in sorted(rec_result.recommendations_by_status.items()):
        pct = (count / rec_result.total_recommendations * 100.0) if rec_result.total_recommendations > 0 else 0.0
        print(f"  {status:<15} {count:>8,d} ({pct:>5.1f}%)")

    print("\nRECOMMENDATION DISTRIBUTION BY CONFIDENCE PROVENANCE:")
    for conf, count in sorted(rec_result.recommendations_by_confidence.items(), key=lambda x: x[1], reverse=True):
        pct = (count / rec_result.total_recommendations * 100.0) if rec_result.total_recommendations > 0 else 0.0
        print(f"  {conf:<25} {count:>8,d} ({pct:>5.1f}%)")

    if rec_result.conflicts:
        print("\nOPERATIONAL CONFLICT BREAKDOWN:")
        conflict_types: dict[str, int] = {}
        for c in rec_result.conflicts:
            conflict_types[c.conflict_type.value] = conflict_types.get(c.conflict_type.value, 0) + 1
        for ct, cnt in sorted(conflict_types.items()):
            print(f"  {ct:<35} {cnt:>8,d}")

    print("\nGOVERNANCE & HUMAN APPROVAL VERIFICATION:")
    all_approval_required = all(r.approval_required for r in rec_result.recommendations)
    all_execution_forbidden = all(not r.execution_allowed for r in rec_result.recommendations)
    all_draft = all(r.status.value == "DRAFT" for r in rec_result.recommendations)
    all_human_review = all(r.requires_human_review for r in rec_result.recommendations)
    print(f"  All recommendations enforce approval_required=True: {all_approval_required}")
    print(f"  All recommendations enforce execution_allowed=False: {all_execution_forbidden}")
    print(f"  All recommendations are strictly DRAFT:            {all_draft}")
    print(f"  All recommendations flag requires_human_review=True: {all_human_review}")
    print("============================================================\n")


if __name__ == "__main__":
    run_benchmark()
