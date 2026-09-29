"""Data Ingestion and Loading Utilities.

Provides robust CSV dataset loading with:
- Strict path and existence validation
- Type-aware parsing
- Automated schema validation using CommerceDataValidator
- Clear, actionable error messages
- Complete isolation from downstream business logic
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd

from commerce_ai.data.validators import CommerceDataValidator, REQUIRED_COLUMNS


class DataLoadError(Exception):
    """Raised when data loading fails due to malformed content or unreadable files."""


class SchemaValidationError(Exception):
    """Raised when loaded data fails schema contract enforcement."""


class CommerceDataLoader:
    """Loads and verifies CSV datasets for the AI Commerce Platform."""

    def __init__(self, validator: Optional[CommerceDataValidator] = None):
        self.validator = validator or CommerceDataValidator()

    def load_csv(
        self,
        file_path: Union[str, Path],
        entity_name: Optional[str] = None,
        validate: bool = True,
    ) -> pd.DataFrame:
        """Load a CSV file into a pandas DataFrame and optionally validate against schema.

        Args:
            file_path: Path to the CSV file.
            entity_name: Conceptual entity name (e.g. 'sales', 'inventory').
                         Inferred from filename stem if not provided.
            validate: Whether to run CommerceDataValidator on the loaded DataFrame.

        Returns:
            pd.DataFrame: Loaded and validated DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            DataLoadError: If pandas fails to parse the CSV.
            SchemaValidationError: If validation fails and validate=True.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found at path: {path.resolve()}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path.resolve()}")

        inferred_entity = entity_name or path.stem.lower()

        # Date columns to parse automatically
        date_cols = []
        if inferred_entity == "sales":
            date_cols = ["date"]
        elif inferred_entity == "inventory":
            date_cols = ["snapshot_date"]
        elif inferred_entity == "purchases":
            date_cols = ["order_date", "expected_delivery_date", "actual_delivery_date"]
        elif inferred_entity == "returns":
            date_cols = ["return_date"]

        try:
            # First pass: check headers
            header_sample = pd.read_csv(path, nrows=0)
            parse_dates = [c for c in date_cols if c in header_sample.columns]

            df = pd.read_csv(
                path,
                parse_dates=parse_dates if parse_dates else False,
                low_memory=False,
            )
        except Exception as exc:
            raise DataLoadError(f"Failed to parse CSV file '{path.name}': {exc}") from exc

        if df.empty:
            raise DataLoadError(f"File '{path.name}' is empty (0 records loaded).")

        if validate:
            res = self.validator.validate_entity(inferred_entity, df)
            if not res.valid:
                error_summary = "; ".join([e.issue for e in res.errors[:5]])
                raise SchemaValidationError(
                    f"Schema validation failed for '{inferred_entity}' loaded from {path.name}: {error_summary}"
                )

        return df

    def load_all(
        self,
        data_dir: Union[str, Path],
        validate: bool = True,
        validate_relational: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load all standard CSV files from a directory into a dictionary of DataFrames.

        Expected files: sales.csv, inventory.csv, products.csv, warehouses.csv,
                        purchases.csv, returns.csv, channels.csv, suppliers.csv.

        Args:
            data_dir: Directory containing CSV files.
            validate: Validate individual entity schemas.
            validate_relational: Validate foreign keys across datasets.

        Returns:
            Dict[str, pd.DataFrame]: Mapping of entity names to DataFrames.
        """
        directory = Path(data_dir)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Data directory not found: {directory.resolve()}")

        loaded_datasets: Dict[str, pd.DataFrame] = {}
        for entity in REQUIRED_COLUMNS.keys():
            csv_path = directory / f"{entity}.csv"
            if csv_path.exists():
                loaded_datasets[entity] = self.load_csv(
                    file_path=csv_path,
                    entity_name=entity,
                    validate=validate,
                )

        if validate_relational and loaded_datasets:
            rel_res = self.validator.validate_relational_integrity(loaded_datasets)
            if not rel_res.valid:
                error_summary = "; ".join([e.issue for e in rel_res.errors[:5]])
                raise SchemaValidationError(f"Relational integrity check failed: {error_summary}")

        return loaded_datasets


# Convenience functional APIs
def load_csv(
    file_path: Union[str, Path],
    entity_name: Optional[str] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper for CommerceDataLoader.load_csv."""
    return CommerceDataLoader().load_csv(file_path=file_path, entity_name=entity_name, validate=validate)


def load_dataset(
    data_dir: Union[str, Path],
    validate: bool = True,
    validate_relational: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Convenience wrapper for CommerceDataLoader.load_all."""
    return CommerceDataLoader().load_all(
        data_dir=data_dir,
        validate=validate,
        validate_relational=validate_relational,
    )
