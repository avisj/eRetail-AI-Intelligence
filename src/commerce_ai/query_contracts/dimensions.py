"""Business Dimensions, Compatibility Rules, and Combinations (Phase 7B).

Defines dimension groupings, supported combinations, and validation checks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set
from commerce_ai.query_contracts.enums import BusinessDimension, BusinessDomain

# Valid dimension sets by domain
DOMAIN_SUPPORTED_DIMENSIONS: Dict[BusinessDomain, Set[BusinessDimension]] = {
    BusinessDomain.SALES: {
        BusinessDimension.SKU,
        BusinessDimension.WAREHOUSE,
        BusinessDimension.CHANNEL,
        BusinessDimension.CATEGORY,
        BusinessDimension.BRAND,
        BusinessDimension.DATE,
        BusinessDimension.WEEK,
        BusinessDimension.MONTH,
        BusinessDimension.QUARTER,
    },
    BusinessDomain.FINANCIAL: {
        BusinessDimension.SKU,
        BusinessDimension.WAREHOUSE,
        BusinessDimension.CHANNEL,
        BusinessDimension.CATEGORY,
        BusinessDimension.BRAND,
        BusinessDimension.DATE,
        BusinessDimension.WEEK,
        BusinessDimension.MONTH,
    },
    BusinessDomain.INVENTORY: {
        BusinessDimension.SKU,
        BusinessDimension.WAREHOUSE,
        BusinessDimension.CATEGORY,
        BusinessDimension.BRAND,
        BusinessDimension.SUPPLIER,
        BusinessDimension.VELOCITY_TIER,
    },
    BusinessDomain.DEMAND: {
        BusinessDimension.SKU,
        BusinessDimension.CATEGORY,
        BusinessDimension.BRAND,
        BusinessDimension.ABC_CLASS,
        BusinessDimension.XYZ_CLASS,
        BusinessDimension.DATE,
        BusinessDimension.WEEK,
        BusinessDimension.MONTH,
    },
    BusinessDomain.FORECASTING: {
        BusinessDimension.SKU,
        BusinessDimension.DATE,
        BusinessDimension.WEEK,
        BusinessDimension.MONTH,
    },
    BusinessDomain.RETURNS: {
        BusinessDimension.SKU,
        BusinessDimension.CATEGORY,
        BusinessDimension.BRAND,
        BusinessDimension.CHANNEL,
        BusinessDimension.DATE,
        BusinessDimension.WEEK,
        BusinessDimension.MONTH,
    },
    BusinessDomain.OPERATIONS: {
        BusinessDimension.SKU,
        BusinessDimension.WAREHOUSE,
        BusinessDimension.SUPPLIER,
    },
    BusinessDomain.BUSINESS_IMPACT: {
        BusinessDimension.CATEGORY,
        BusinessDimension.BRAND,
        BusinessDimension.WAREHOUSE,
    },
    BusinessDomain.RECOMMENDATIONS: {
        BusinessDimension.SKU,
        BusinessDimension.WAREHOUSE,
        BusinessDimension.CHANNEL,
    },
    BusinessDomain.DECISIONS: {
        BusinessDimension.SKU,
        BusinessDimension.WAREHOUSE,
    },
    BusinessDomain.DATA_QUALITY: {
        BusinessDimension.DATE,
        BusinessDimension.CATEGORY,
    },
    BusinessDomain.CROSS_DOMAIN: {
        BusinessDimension.SKU,
        BusinessDimension.WAREHOUSE,
        BusinessDimension.CHANNEL,
        BusinessDimension.CATEGORY,
        BusinessDimension.BRAND,
    },
}

# Incompatible cross-dimensional combinations
INCOMPATIBLE_DIMENSION_PAIRS: List[Set[BusinessDimension]] = [
    {BusinessDimension.DATE, BusinessDimension.MONTH},  # Redundant time grains
    {BusinessDimension.WEEK, BusinessDimension.MONTH},   # Redundant time grains
    {BusinessDimension.DATE, BusinessDimension.WEEK},    # Redundant time grains
]


def is_dimension_supported(dimension: BusinessDimension, domain: BusinessDomain) -> bool:
    """Check if a specific dimension is supported within a domain."""
    supported = DOMAIN_SUPPORTED_DIMENSIONS.get(domain, set())
    return dimension in supported


def validate_dimensions(
    dimensions: List[BusinessDimension],
    domain: BusinessDomain,
) -> List[str]:
    """Validate a set of dimensions against domain rules and pairwise compatibility.

    Returns a list of error messages (empty if completely valid).
    """
    errors: List[str] = []
    supported = DOMAIN_SUPPORTED_DIMENSIONS.get(domain, set())

    # Check domain support
    for dim in dimensions:
        if dim not in supported:
            errors.append(f"Dimension '{dim.value}' is not supported in domain '{domain.value}'.")

    # Check pairwise conflicts
    dim_set = set(dimensions)
    for pair in INCOMPATIBLE_DIMENSION_PAIRS:
        if pair.issubset(dim_set):
            conflict_names = ", ".join(d.value for d in pair)
            errors.append(f"Incompatible dimension pair requested: [{conflict_names}]. Choose a single time grain.")

    return errors
