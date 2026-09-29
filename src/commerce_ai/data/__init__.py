"""Data package for standard commerce contracts, validation, loaders, and generators."""

from commerce_ai.data.schemas import (
    BaseCommerceModel,
    SaleRecord,
    InventorySnapshot,
    Product,
    Warehouse,
    PurchaseOrder,
    ReturnRecord,
    Channel,
    Supplier,
)
from commerce_ai.data.validators import (
    CommerceDataValidator,
    ValidationErrorDetail,
    ValidationResult,
    generate_data_quality_report,
)
from commerce_ai.data.loaders import (
    CommerceDataLoader,
    DataLoadError,
    SchemaValidationError,
    load_csv,
    load_dataset,
)
from commerce_ai.data.generators import (
    GeneratorConfig,
    SyntheticDataGenerator,
    generate_sample_dataset,
)

__all__ = [
    "BaseCommerceModel",
    "SaleRecord",
    "InventorySnapshot",
    "Product",
    "Warehouse",
    "PurchaseOrder",
    "ReturnRecord",
    "Channel",
    "Supplier",
    "CommerceDataValidator",
    "ValidationErrorDetail",
    "ValidationResult",
    "generate_data_quality_report",
    "CommerceDataLoader",
    "DataLoadError",
    "SchemaValidationError",
    "load_csv",
    "load_dataset",
    "GeneratorConfig",
    "SyntheticDataGenerator",
    "generate_sample_dataset",
]
