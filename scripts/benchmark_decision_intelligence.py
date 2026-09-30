"""Canonical Benchmark Script for Decision Intelligence & Action Planning (Phase 6H).

Transforms Phase 6G business recommendations into structured decision packages on the
canonical enterprise synthetic dataset in data/sample/ (773,555 sales rows, 292,500 inventory rows).

Reports ONLY aggregate distributions:
- Total decision packages
- Packages by recommendation type
- Packages by decision status
- Packages by highest risk severity
- Packages by option count
- Packages requiring human review (100%)
- Packages with operational conflicts
- Packages with insufficient information
- Packages by confidence provenance

DO NOT output:
- Top recommendations / best options
- Winner option
- Ranked SKUs or Warehouses
- Ranked financial leaderboards
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
from commerce_ai.decision_intelligence.service import DecisionIntelligenceService
from commerce_ai.decision_intelligence.schemas import (
    DecisionIntelligenceConfig,
    DecisionStatus,
    RiskSeverity,
)
from commerce_ai.recommendations.business_schemas import (
    RecommendationPriority,
    RecommendationType,
)


def run_benchmark():
    data_dir = Path("data/sample")
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        sys.exit(1)

    print("============================================================")
    print("PHASE 6H — DECISION INTELLIGENCE & ACTION PLANNING")
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
    print(f"Emitted {rec_result.total_recommendations:,d} recommendations in {t_rec:.2f}s\n")

    # Step 4: Run Decision Intelligence Service (Phase 6H)
    print("Step 4: Running Decision Intelligence & Action Planning (Phase 6H)...")
    t4 = time.perf_counter()
    decision_service = DecisionIntelligenceService()
    dec_result = decision_service.package_recommendations(
        recommendations=rec_result.recommendations,
        conflicts=rec_result.conflicts,
        as_of_date=rec_result.as_of_date,
    )
    t_dec = time.perf_counter() - t4
    total_time = t_cd + t_impact + t_rec + t_dec

    print("============================================================")
    print("CANONICAL BENCHMARK RESULTS (PHASE 6H)")
    print("============================================================")
    print(f"Total Sales Rows Processed:             {len(sales_df):>15,d}")
    print(f"Total Inventory Snapshots:              {len(inventory_df):>15,d}")
    print(f"Total Primary Records (SKU x WH):       {len(cd_result.records):>15,d}")
    print(f"Total Cross-Domain Signals:             {len(cd_result.signals):>15,d}")
    print(f"Total Business Recommendations Packaged:{len(rec_result.recommendations):>15,d}")
    print(f"Total Decision Packages Generated:      {dec_result.total_decisions:>15,d}")
    print(f"Packages Requiring Human Review:        {dec_result.human_review_count:>15,d} (100.0%)")
    print(f"Packages with Operational Conflicts:    {dec_result.conflict_count:>15,d}")
    print(f"Packages with Insufficient Info:        {dec_result.insufficient_information_count:>15,d}")
    print(f"Phase 6E Runtime:                       {t_cd:>15.2f} s")
    print(f"Phase 6F Runtime:                       {t_impact:>15.2f} s")
    print(f"Phase 6G Runtime:                       {t_rec:>15.2f} s")
    print(f"Phase 6H Runtime:                       {t_dec:>15.2f} s")
    print(f"Total Pipeline Runtime:                 {total_time:>15.2f} s")
    print("------------------------------------------------------------")

    print("\nDECISION PACKAGE DISTRIBUTION BY RECOMMENDATION TYPE:")
    for rec_type, count in sorted(dec_result.decisions_by_type.items(), key=lambda x: x[1], reverse=True):
        pct = (count / dec_result.total_decisions * 100.0) if dec_result.total_decisions > 0 else 0.0
        print(f"  {rec_type:<35} {count:>8,d} ({pct:>5.1f}%)")

    print("\nDECISION PACKAGE DISTRIBUTION BY STATUS:")
    for status, count in sorted(dec_result.decisions_by_status.items()):
        pct = (count / dec_result.total_decisions * 100.0) if dec_result.total_decisions > 0 else 0.0
        print(f"  {status:<15} {count:>8,d} ({pct:>5.1f}%)")

    print("\nDECISION PACKAGE DISTRIBUTION BY HIGHEST RISK SEVERITY:")
    for sev in [RiskSeverity.CRITICAL, RiskSeverity.HIGH, RiskSeverity.MEDIUM, RiskSeverity.LOW]:
        count = dec_result.decisions_by_risk_severity.get(sev.value, 0)
        pct = (count / dec_result.total_decisions * 100.0) if dec_result.total_decisions > 0 else 0.0
        print(f"  {sev.value:<15} {count:>8,d} ({pct:>5.1f}%)")

    print("\nDECISION OPTION COUNT DISTRIBUTION:")
    for opt_cnt, count in sorted(dec_result.decisions_by_option_count.items()):
        pct = (count / dec_result.total_decisions * 100.0) if dec_result.total_decisions > 0 else 0.0
        print(f"  {opt_cnt} options:                         {count:>8,d} ({pct:>5.1f}%)")

    print("\nDECISION CONFIDENCE PROVENANCE DISTRIBUTION:")
    for conf, count in sorted(dec_result.decisions_by_confidence.items(), key=lambda x: x[1], reverse=True):
        pct = (count / dec_result.total_decisions * 100.0) if dec_result.total_decisions > 0 else 0.0
        print(f"  {conf:<25} {count:>8,d} ({pct:>5.1f}%)")

    print("\nGOVERNANCE & HUMAN APPROVAL VERIFICATION:")
    all_approval_required = all(p.approval_required for p in dec_result.decision_packages)
    all_execution_forbidden = all(not p.execution_allowed for p in dec_result.decision_packages)
    all_pending = all(p.decision_status == DecisionStatus.PENDING_REVIEW for p in dec_result.decision_packages)
    all_human_review = all(p.requires_human_review for p in dec_result.decision_packages)
    all_no_winner = all(p.selected_option is None for p in dec_result.decision_packages)
    all_opt_no_winner = all(
        all(not opt.recommended_by_engine for opt in p.decision_options)
        for p in dec_result.decision_packages
    )
    print(f"  All packages enforce approval_required=True:   {all_approval_required}")
    print(f"  All packages enforce execution_allowed=False:  {all_execution_forbidden}")
    print(f"  All packages enforce selected_option=None:     {all_no_winner} (NO WINNER DECLARED)")
    print(f"  No options flagged as recommended winner:      {all_opt_no_winner}")
    print(f"  All packages are strictly PENDING_REVIEW:      {all_pending}")
    print(f"  All packages flag requires_human_review=True:  {all_human_review}")
    print("============================================================\n")


if __name__ == "__main__":
    run_benchmark()
