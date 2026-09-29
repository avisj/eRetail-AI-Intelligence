#!/usr/bin/env python3
"""CLI Script to generate synthetic ecommerce dataset."""

import argparse
import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from commerce_ai.config.settings import settings
from commerce_ai.data.generators import GeneratorConfig, SyntheticDataGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ecommerce dataset.")
    parser.add_argument("--skus", type=int, default=settings.gen_skus, help="Number of SKUs (default: 500)")
    parser.add_argument("--warehouses", type=int, default=settings.gen_warehouses, help="Number of warehouses (default: 5)")
    parser.add_argument("--channels", type=int, default=settings.gen_channels, help="Number of channels (default: 4)")
    parser.add_argument("--suppliers", type=int, default=settings.gen_suppliers, help="Number of suppliers (default: 20)")
    parser.add_argument("--days", type=int, default=settings.gen_historical_days, help="Historical days (default: 730)")
    parser.add_argument("--seed", type=int, default=settings.gen_seed, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=str(settings.sample_dir), help="Output directory for CSVs")
    parser.add_argument("--with-anomalies", action="store_true", help="Inject controlled anomalies for testing")

    args = parser.parse_args()

    out_path = Path(args.output)
    print("=" * 60)
    print("AI Commerce Intelligence Platform — Data Generator")
    print("=" * 60)
    print(f"SKUs:            {args.skus}")
    print(f"Warehouses:      {args.warehouses}")
    print(f"Channels:        {args.channels}")
    print(f"Suppliers:       {args.suppliers}")
    print(f"Historical Days: {args.days}")
    print(f"Random Seed:     {args.seed}")
    print(f"Output Path:     {out_path.resolve()}")
    print("-" * 60)
    print("Simulating commerce dynamics (sales, purchases, inventory, returns)...")

    config = GeneratorConfig(
        number_of_skus=args.skus,
        number_of_warehouses=args.warehouses,
        number_of_channels=args.channels,
        number_of_suppliers=args.suppliers,
        historical_days=args.days,
        random_seed=args.seed,
        include_controlled_anomalies=args.with_anomalies,
    )
    generator = SyntheticDataGenerator(config)
    datasets = generator.generate_and_save(out_path)

    print("\nDataset Generation Complete! Created files:")
    for name, df in datasets.items():
        file_path = out_path / f"{name}.csv"
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {name:12s} : {len(df):8,d} rows | {len(df.columns):2d} cols | {size_mb:6.2f} MB")

    print("=" * 60)


if __name__ == "__main__":
    main()
