#!/usr/bin/env python3
"""CLI Script to validate datasets and output a Data Quality Report."""

import argparse
import json
import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from commerce_ai.config.settings import settings
from commerce_ai.data.loaders import CommerceDataLoader
from commerce_ai.data.validators import generate_data_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate datasets and generate a Data Quality Report.")
    parser.add_argument("--dir", type=str, default=str(settings.sample_dir), help="Directory containing CSV files")
    parser.add_argument("--json", action="store_true", help="Output raw JSON report")

    args = parser.parse_args()
    data_dir = Path(args.dir)

    print("=" * 60)
    print("AI Commerce Intelligence Platform — Data Quality Auditor")
    print("=" * 60)
    print(f"Auditing directory: {data_dir.resolve()}\n")

    loader = CommerceDataLoader()
    try:
        datasets = loader.load_all(data_dir, validate=False, validate_relational=False)
    except Exception as exc:
        print(f"Error loading datasets: {exc}")
        sys.exit(1)

    if not datasets:
        print(f"No CSV files found in {data_dir.resolve()}")
        sys.exit(1)

    report = generate_data_quality_report(datasets)

    if args.json:
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["status"] == "VALID" else 1)

    print(f"Validation Status: {'✅ PASSED (VALID)' if report['status'] == 'VALID' else '❌ FAILED (INVALID)'}")
    print(f"Total Errors:      {report['total_errors']}")
    print(f"Total Warnings:    {report['total_warnings']}")
    print("\nEntity Breakdown:")
    for ent, stats in report["entities"].items():
        print(f"  • {ent:12s} : {stats['rows']:,} rows, {stats['columns']} cols, "
              f"{stats['missing_values']} missing values, {stats['duplicate_primary_keys']} duplicate PKs")

    if "dimensions" in report:
        dims = report["dimensions"]
        print("\nCatalog Dimensions:")
        print(f"  • Unique SKUs:       {dims.get('unique_skus', 0):,}")
        print(f"  • Unique Warehouses:  {dims.get('unique_warehouses', 0):,}")
        print(f"  • Unique Channels:    {dims.get('unique_channels', 0):,}")

    if "date_ranges" in report:
        print("\nDate Ranges:")
        for ent, dr in report["date_ranges"].items():
            print(f"  • {ent:10s}: {dr['min_date']} to {dr['max_date']} ({dr['total_days']} days)")

    if "sales_summary" in report:
        s = report["sales_summary"]
        print("\nSales Aggregate Summary:")
        print(f"  • Total Volume:      {s['total_sales_quantity']:,} units")
        print(f"  • Total Revenue:     ${s['total_revenue']:,.2f} {s['currency']}")
        print(f"  • Avg Order Value:   ${s['average_order_value']:,.2f}")
        print(f"  • Total Sales Orders:{s['total_transactions']:,}")

    if report["validation_errors"]:
        print("\nValidation Errors Detected:")
        for idx, err in enumerate(report["validation_errors"], start=1):
            print(f"  [{idx}] [{err['entity']}] Field: {err.get('field')} - {err['issue']}")
        if "validation_errors_truncated" in report:
            print(f"  ... and {report['validation_errors_truncated']} more errors.")

    print("=" * 60)
    sys.exit(0 if report["status"] == "VALID" else 1)


if __name__ == "__main__":
    main()
