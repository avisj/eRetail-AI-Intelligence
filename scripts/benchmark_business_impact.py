"""Canonical Benchmark Script for Business Impact & Opportunity Quantification (Phase 6F).

Evaluates the deterministic Business Impact quantification pipeline against the canonical
enterprise synthetic dataset in data/sample/ (773,555 sales rows, 292,500 inventory rows).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import pandas as pd

from commerce_ai.cross_domain.service import CrossDomainIntelligenceService
from commerce_ai.business_impact.service import BusinessImpactService
from commerce_ai.business_impact.schemas import (
    BusinessImpactConfig,
    ImpactCategory,
    ImpactConfidence,
    ImpactType,
)


def run_benchmark():
    data_dir = Path("data/sample")
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found.")
        sys.exit(1)

    print("============================================================")
    print("PHASE 6F — BUSINESS IMPACT & OPPORTUNITY QUANTIFICATION")
    print("CANONICAL DATASET BENCHMARK")
    print("============================================================")
    print(f"Loading canonical datasets from {data_dir.resolve()}...")

    t0 = time.perf_counter()
    products_df = pd.read_csv(data_dir / "products.csv")
    warehouses_df = pd.read_csv(data_dir / "warehouses.csv")
    sales_df = pd.read_csv(data_dir / "sales.csv")
    inventory_df = pd.read_csv(data_dir / "inventory.csv")
    returns_df = pd.read_csv(data_dir / "returns.csv") if (data_dir / "returns.csv").exists() else None
    t_load = time.perf_counter() - t0

    print(f"Loaded:")
    print(f"  - Products:   {len(products_df):>10,d} rows")
    print(f"  - Warehouses: {len(warehouses_df):>10,d} rows")
    print(f"  - Sales:      {len(sales_df):>10,d} rows")
    print(f"  - Inventory:  {len(inventory_df):>10,d} rows")
    if returns_df is not None:
        print(f"  - Returns:    {len(returns_df):>10,d} rows")
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
    total_time = t_cd + t_impact

    summary = impact_result.portfolio_summary
    records = impact_result.impact_records

    print("============================================================")
    print("CANONICAL BENCHMARK RESULTS (PHASE 6F)")
    print("============================================================")
    print(f"Total Sales Rows Processed:             {len(sales_df):>15,d}")
    print(f"Total Inventory Snapshots:              {len(inventory_df):>15,d}")
    print(f"Total Primary Records (SKU x WH):       {len(cd_result.records):>15,d}")
    print(f"Total Evaluated Signals:                {summary.total_signal_count:>15,d}")
    print(f"Total Impact Records Generated:         {summary.impact_record_count:>15,d}")
    print(f"Phase 6E Runtime:                       {t_cd:>15.2f} s")
    print(f"Phase 6F Quantification Runtime:        {t_impact:>15.2f} s")
    print(f"Total Pipeline Runtime:                 {total_time:>15.2f} s")
    print("------------------------------------------------------------")
    print("EXPOSURE TOTALS & DOUBLE-COUNTING PROTECTION:")
    print(f"  Gross Signal Exposure:                ${summary.gross_signal_exposure:>15,.2f}")
    if summary.deduplicated_exposure is not None:
        print(f"  Deduplicated Capital Exposure:        ${summary.deduplicated_exposure:>15,.2f}")
        overlap = summary.gross_signal_exposure - summary.deduplicated_exposure
        overlap_pct = (overlap / summary.gross_signal_exposure * 100.0) if summary.gross_signal_exposure > 0 else 0.0
        print(f"  Overlapping Exposure Deduplicated:    ${overlap:>15,.2f} ({overlap_pct:.1f}%)")
        print(f"  Deduplication Status:                 {summary.deduplication_status}")
    print("------------------------------------------------------------")
    print("EXPOSURE PROVENANCE BREAKDOWN:")
    print(f"  Direct Observed Exposure:             ${summary.direct_observed_exposure:>15,.2f}")
    print(f"  Estimated Exposure:                   ${summary.estimated_exposure:>15,.2f}")
    print(f"  Model-Based Exposure:                 ${summary.model_based_exposure:>15,.2f}")
    print(f"  Insufficient Data (Unquantified):     {summary.insufficient_data_count:>15,d} records")
    print("------------------------------------------------------------")
    print("CURRENCY ISOLATION BREAKDOWN:")
    for curr, amt in sorted(summary.currency_breakdown.items()):
        print(f"  {curr:<10}: ${amt:>15,.2f}")
    print("------------------------------------------------------------")
    print("IMPACT RECORDS BY CATEGORY:")
    for cat, cnt in sorted(summary.impact_category_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / summary.impact_record_count * 100.0) if summary.impact_record_count > 0 else 0.0
        print(f"  {cat:<25}: {cnt:>6,d} ({pct:>5.1f}%)")
    print("------------------------------------------------------------")
    print("IMPACT RECORDS BY TYPE:")
    for itype, cnt in sorted(summary.impact_type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / summary.impact_record_count * 100.0) if summary.impact_record_count > 0 else 0.0
        print(f"  {itype:<32}: {cnt:>6,d} ({pct:>5.1f}%)")
    print("------------------------------------------------------------")
    print("IMPACT RECORDS BY SEVERITY:")
    for sev, cnt in sorted(summary.severity_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / summary.impact_record_count * 100.0) if summary.impact_record_count > 0 else 0.0
        print(f"  {sev:<15}: {cnt:>6,d} ({pct:>5.1f}%)")
    print("------------------------------------------------------------")
    print("EXPOSURE TOTALS BY CATEGORY ($):")
    for cat, val in sorted(summary.exposure_totals_by_category.items(), key=lambda x: x[1], reverse=True):
        pct = (val / summary.gross_signal_exposure * 100.0) if summary.gross_signal_exposure > 0 else 0.0
        print(f"  {cat:<25}: ${val:>15,.2f} ({pct:>5.1f}%)")
    print("------------------------------------------------------------")
    print("EXPOSURE TOTALS BY IMPACT TYPE ($):")
    for itype, val in sorted(summary.exposure_totals_by_type.items(), key=lambda x: x[1], reverse=True):
        pct = (val / summary.gross_signal_exposure * 100.0) if summary.gross_signal_exposure > 0 else 0.0
        print(f"  {itype:<32}: ${val:>15,.2f} ({pct:>5.1f}%)")
    print("------------------------------------------------------------")
    print("CALCULATION STATUS DISTRIBUTION:")
    for stat, cnt in sorted(summary.calculation_status_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / summary.impact_record_count * 100.0) if summary.impact_record_count > 0 else 0.0
        print(f"  {stat:<25}: {cnt:>6,d} ({pct:>5.1f}%)")
    if summary.deduplicated_physical_capital_exposure is not None:
        print("------------------------------------------------------------")
        print("PHYSICAL INVENTORY CAPITAL DEDUPLICATION:")
        gross_inv = summary.exposure_totals_by_category.get("INVENTORY", 0.0)
        dedup_inv = summary.deduplicated_physical_capital_exposure
        overlap_inv = gross_inv - dedup_inv
        overlap_inv_pct = (overlap_inv / gross_inv * 100.0) if gross_inv > 0 else 0.0
        print(f"  Gross Inventory Exposure:             ${gross_inv:>15,.2f}")
        print(f"  Deduplicated Physical Capital:        ${dedup_inv:>15,.2f}")
        print(f"  Physical Overlap Deduplicated:        ${overlap_inv:>15,.2f} ({overlap_inv_pct:.1f}%)")
    print("============================================================")


if __name__ == "__main__":
    run_benchmark()
