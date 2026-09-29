"""Tests for ABC and XYZ Classification Engine."""

import pandas as pd
import pytest

from commerce_ai.analytics.abc_xyz import (
    ABCConfig,
    XYZConfig,
    calculate_abc_classification,
    calculate_xyz_classification,
    calculate_abc_xyz_matrix,
)


@pytest.fixture
def sample_portfolio():
    """Generates a synthetic portfolio with distinct volume and volatility characteristics:
    - SKU_HIGH_STEADY: high revenue ($1,000/day), steady demand (CV ~ 0.1) -> Class AX
    - SKU_HIGH_VOLATILE: high revenue ($800/day), erratic spikes (CV ~ 1.5) -> Class AZ
    - SKU_MED_STEADY: medium revenue ($150/day), steady (CV ~ 0.3) -> Class BX
    - SKU_LOW_ERRATIC: low revenue ($10/day), intermittent (CV ~ 2.0) -> Class CZ
    """
    records = []
    # 20 days of data
    for d in range(20):
        records.append({"date": f"2025-01-{d+1:02d}", "sku_id": "SKU_HIGH_STEADY", "revenue": 1000.0, "units_sold": 100})
        # Erratic: 0 or 200 units
        erratic_qty = 200 if d % 2 == 0 else 0
        records.append({"date": f"2025-01-{d+1:02d}", "sku_id": "SKU_HIGH_VOLATILE", "revenue": erratic_qty * 10.0, "units_sold": erratic_qty})
        records.append({"date": f"2025-01-{d+1:02d}", "sku_id": "SKU_MED_STEADY", "revenue": 150.0, "units_sold": 15})
        records.append({"date": f"2025-01-{d+1:02d}", "sku_id": "SKU_LOW_ERRATIC", "revenue": 10.0 if d == 0 else 0.0, "units_sold": 1 if d == 0 else 0})

    return pd.DataFrame(records)


class TestABCXYZClassification:
    def test_abc_classification_and_pareto(self, sample_portfolio):
        cfg = ABCConfig(a_threshold=0.70, b_threshold=0.95)
        abc_df = calculate_abc_classification(sample_portfolio, config=cfg)

        assert len(abc_df) == 4
        # SKU_HIGH_STEADY generates highest revenue (20,000 / ~43,000 = ~46%)
        assert abc_df["sku_id"].iloc[0] == "SKU_HIGH_STEADY"
        assert abc_df["abc_class"].iloc[0] == "A"
        # Low revenue SKU is in class C
        low_sku = abc_df[abc_df["sku_id"] == "SKU_LOW_ERRATIC"]
        assert low_sku["abc_class"].iloc[0] == "C"

    def test_xyz_classification_volatility(self, sample_portfolio):
        cfg = XYZConfig(x_cv_threshold=0.30, y_cv_threshold=0.80)
        xyz_df = calculate_xyz_classification(sample_portfolio, config=cfg)

        # SKU_HIGH_STEADY has CV = 0.0 -> Class X
        steady_sku = xyz_df[xyz_df["sku_id"] == "SKU_HIGH_STEADY"]
        assert steady_sku["xyz_class"].iloc[0] == "X"
        assert steady_sku["coefficient_of_variation"].iloc[0] == 0.0

        # SKU_HIGH_VOLATILE has alternating 0 and 200 -> High CV -> Class Z
        volatile_sku = xyz_df[xyz_df["sku_id"] == "SKU_HIGH_VOLATILE"]
        assert volatile_sku["xyz_class"].iloc[0] == "Z"
        assert volatile_sku["coefficient_of_variation"].iloc[0] > 0.80

    def test_combined_abc_xyz_matrix(self, sample_portfolio):
        sku_df, summary, crosstab = calculate_abc_xyz_matrix(sample_portfolio)

        assert "abc_xyz_class" in sku_df.columns
        assert set(crosstab.index) == {"A", "B", "C", "Total"}
        assert set(crosstab.columns) == {"X", "Y", "Z", "Total"}
        assert crosstab.loc["Total", "Total"] == 4
