"""Data Contracts and Schema Definitions for Returns Intelligence (Phase 5A).

Defines standardized Pydantic models for:
- Return records and configuration
- Descriptive investigation flags
- Metric records across dimensions (SKU, Channel, Warehouse, SKU×Channel, SKU×Warehouse)
- Reason breakdown and time series points
- Data quality audits and consolidated result containers
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class InvestigationFlag(str, Enum):
    """Deterministic, descriptive investigation flags (no ML predictions)."""

    HIGH_RETURN_RATE = "HIGH_RETURN_RATE"
    HIGH_RETURN_VOLUME = "HIGH_RETURN_VOLUME"
    RETURN_RATE_INCREASING = "RETURN_RATE_INCREASING"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    NO_RETURN_DATA = "NO_RETURN_DATA"
    ZERO_SALES_RECORDED = "ZERO_SALES_RECORDED"


class ReturnReasonBreakdown(BaseModel):
    """Categorized summary of return reasons."""

    model_config = ConfigDict(extra="allow")

    reason: str = Field(..., description="Observed return reason category")
    returned_units: int = Field(..., ge=0, description="Total units returned under this reason")
    return_count: int = Field(..., ge=0, description="Number of return transactions under this reason")
    percentage_of_units: float = Field(..., ge=0.0, le=100.0, description="Share of total returned units (%)")
    percentage_of_returns: float = Field(..., ge=0.0, le=100.0, description="Share of total return transactions (%)")
    estimated_return_value: Optional[float] = Field(default=None, ge=0.0, description="Estimated total return monetary valuation")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["percentage_of_units"] = round(self.percentage_of_units, 2)
        d["percentage_of_returns"] = round(self.percentage_of_returns, 2)
        if self.estimated_return_value is not None:
            d["estimated_return_value"] = round(self.estimated_return_value, 2)
        return d


class ReturnMetricRecord(BaseModel):
    """Standardized factual metric record across analytical dimensions."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    dimension: str = Field(..., description="Dimension: SKU, CHANNEL, WAREHOUSE, SKU_CHANNEL, SKU_WAREHOUSE")
    key: str = Field(..., description="Unique dimension identifier (e.g. SKU_001, CH_01, WH_01, SKU_001:CH_01)")
    sku_id: Optional[str] = Field(default=None, description="SKU identifier if applicable")
    channel_id: Optional[str] = Field(default=None, description="Channel identifier if applicable")
    warehouse_id: Optional[str] = Field(default=None, description="Warehouse identifier if applicable")
    sold_units: int = Field(default=0, ge=0, description="Total sales units sold in period")
    returned_units: int = Field(default=0, ge=0, description="Total units returned in period")
    return_rate: Optional[float] = Field(default=None, ge=0.0, description="Unit return rate: returned_units / sold_units")
    return_count: int = Field(default=0, ge=0, description="Number of distinct return events/records")
    order_count: int = Field(default=0, ge=0, description="Number of distinct sales orders")
    order_return_rate: Optional[float] = Field(default=None, ge=0.0, description="Order return rate: returned_orders / total_orders")
    sales_value: Optional[float] = Field(default=None, ge=0.0, description="Total sales monetary valuation in period")
    return_value: Optional[float] = Field(default=None, ge=0.0, description="Total return monetary valuation in period")
    revenue_return_rate: Optional[float] = Field(default=None, ge=0.0, description="Revenue return rate: return_value / sales_value")
    top_return_reason: Optional[str] = Field(default=None, description="Most frequent return reason category")
    is_sufficient_sample: bool = Field(default=True, description="True if sales volume satisfies minimum threshold")
    investigation_flags: List[str] = Field(default_factory=list, description="Descriptive audit/investigation flags")
    return_rate_change: Optional[float] = Field(default=None, description="Change in return rate vs previous period")
    trend: Optional[str] = Field(default=None, description="Trend direction: INCREASING, DECREASING, STABLE, INSUFFICIENT_DATA")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["return_rate"] = round(self.return_rate, 4) if self.return_rate is not None else None
        d["order_return_rate"] = round(self.order_return_rate, 4) if self.order_return_rate is not None else None
        d["sales_value"] = round(self.sales_value, 2) if self.sales_value is not None else None
        d["return_value"] = round(self.return_value, 2) if self.return_value is not None else None
        d["revenue_return_rate"] = round(self.revenue_return_rate, 4) if self.revenue_return_rate is not None else None
        d["return_rate_change"] = round(self.return_rate_change, 4) if self.return_rate_change is not None else None
        return d


class ReturnTimeSeriesPoint(BaseModel):
    """Time-series aggregation point for returns and sales."""

    model_config = ConfigDict(extra="allow")

    period: str = Field(..., description="Time period bucket (YYYY-MM-DD, YYYY-Www, YYYY-MM)")
    sold_units: int = Field(default=0, ge=0, description="Total units sold in period")
    returned_units: int = Field(default=0, ge=0, description="Total units returned in period")
    return_rate: Optional[float] = Field(default=None, ge=0.0, description="Unit return rate in period")
    return_count: int = Field(default=0, ge=0, description="Return transaction count in period")
    sales_value: Optional[float] = Field(default=None, ge=0.0, description="Sales monetary valuation in period")
    return_value: Optional[float] = Field(default=None, ge=0.0, description="Return monetary valuation in period")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["return_rate"] = round(self.return_rate, 4) if self.return_rate is not None else None
        d["sales_value"] = round(self.sales_value, 2) if self.sales_value is not None else None
        d["return_value"] = round(self.return_value, 2) if self.return_value is not None else None
        return d


class ReturnsDataQualityReport(BaseModel):
    """Quality and integrity audit report for returns data."""

    model_config = ConfigDict(extra="allow")

    total_records: int = Field(default=0, ge=0)
    missing_sku_count: int = Field(default=0, ge=0)
    missing_date_count: int = Field(default=0, ge=0)
    missing_order_id_count: int = Field(default=0, ge=0)
    invalid_quantity_count: int = Field(default=0, ge=0)
    unknown_sku_count: int = Field(default=0, ge=0)
    unknown_warehouse_count: int = Field(default=0, ge=0)
    unknown_channel_count: int = Field(default=0, ge=0)
    duplicate_record_count: int = Field(default=0, ge=0)
    unknown_reason_count: int = Field(default=0, ge=0)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    is_clean: bool = Field(default=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ReturnsConfig(BaseModel):
    """Configurable thresholds and settings for returns intelligence analytics."""

    model_config = ConfigDict(extra="forbid")

    min_sold_units_threshold: int = Field(
        default=30,
        ge=1,
        description="Minimum sales denominator required for reliable return-rate classification",
    )
    min_return_count_threshold: int = Field(
        default=5,
        ge=1,
        description="Minimum return event sample size for rate reliability",
    )
    high_return_rate_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Return rate threshold (e.g. 15%) triggering HIGH_RETURN_RATE investigation flag",
    )
    high_return_volume_threshold: int = Field(
        default=50,
        ge=1,
        description="Absolute returned units threshold triggering HIGH_RETURN_VOLUME investigation flag",
    )
    increasing_rate_delta_threshold: float = Field(
        default=0.05,
        ge=0.0,
        description="Period-over-period return rate increase (e.g. +5%) triggering RETURN_RATE_INCREASING flag",
    )
    known_reasons: Optional[List[str]] = Field(
        default=None,
        description="Optional list of documented return reason categories for domain validation",
    )


class ReturnsSummary(BaseModel):
    """High-level portfolio returns executive summary."""

    model_config = ConfigDict(extra="allow")

    total_sold_units: int = Field(default=0, ge=0)
    total_returned_units: int = Field(default=0, ge=0)
    overall_unit_return_rate: Optional[float] = Field(default=None, ge=0.0)
    total_sales_value: Optional[float] = Field(default=None, ge=0.0)
    total_return_value: Optional[float] = Field(default=None, ge=0.0)
    overall_revenue_return_rate: Optional[float] = Field(default=None, ge=0.0)
    total_return_events: int = Field(default=0, ge=0)
    total_orders: int = Field(default=0, ge=0)
    overall_order_return_rate: Optional[float] = Field(default=None, ge=0.0)
    top_return_reason: Optional[str] = None
    as_of_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["overall_unit_return_rate"] = round(self.overall_unit_return_rate, 4) if self.overall_unit_return_rate is not None else None
        d["overall_revenue_return_rate"] = round(self.overall_revenue_return_rate, 4) if self.overall_revenue_return_rate is not None else None
        d["overall_order_return_rate"] = round(self.overall_order_return_rate, 4) if self.overall_order_return_rate is not None else None
        d["total_sales_value"] = round(self.total_sales_value, 2) if self.total_sales_value is not None else None
        d["total_return_value"] = round(self.total_return_value, 2) if self.total_return_value is not None else None
        return d


class ReturnsIntelligenceResult(BaseModel):
    """Consolidated returns intelligence result bundle (Phase 5A)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary: ReturnsSummary
    sku_metrics: List[ReturnMetricRecord] = Field(default_factory=list)
    channel_metrics: List[ReturnMetricRecord] = Field(default_factory=list)
    warehouse_metrics: List[ReturnMetricRecord] = Field(default_factory=list)
    reason_metrics: List[ReturnReasonBreakdown] = Field(default_factory=list)
    time_series: List[ReturnTimeSeriesPoint] = Field(default_factory=list)
    sku_channel_metrics: List[ReturnMetricRecord] = Field(default_factory=list)
    sku_warehouse_metrics: List[ReturnMetricRecord] = Field(default_factory=list)
    investigation_flags: Dict[str, List[str]] = Field(default_factory=dict)
    data_quality: ReturnsDataQualityReport = Field(default_factory=ReturnsDataQualityReport)
    as_of_date: Optional[str] = None

    def sku_to_dataframe(self) -> pd.DataFrame:
        """Export SKU returns metrics as a flat pandas DataFrame."""
        if not self.sku_metrics:
            return pd.DataFrame()
        return pd.DataFrame([m.to_dict() for m in self.sku_metrics])

    def channel_to_dataframe(self) -> pd.DataFrame:
        """Export channel returns metrics as a flat pandas DataFrame."""
        if not self.channel_metrics:
            return pd.DataFrame()
        return pd.DataFrame([m.to_dict() for m in self.channel_metrics])

    def warehouse_to_dataframe(self) -> pd.DataFrame:
        """Export warehouse returns metrics as a flat pandas DataFrame."""
        if not self.warehouse_metrics:
            return pd.DataFrame()
        return pd.DataFrame([m.to_dict() for m in self.warehouse_metrics])

    def reasons_to_dataframe(self) -> pd.DataFrame:
        """Export reason breakdowns as a flat pandas DataFrame."""
        if not self.reason_metrics:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.reason_metrics])

    def time_series_to_dataframe(self) -> pd.DataFrame:
        """Export time series metrics as a flat pandas DataFrame."""
        if not self.time_series:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.time_series])

    def sku_channel_to_dataframe(self) -> pd.DataFrame:
        """Export SKU × Channel metrics as a flat pandas DataFrame."""
        if not self.sku_channel_metrics:
            return pd.DataFrame()
        return pd.DataFrame([m.to_dict() for m in self.sku_channel_metrics])

    def sku_warehouse_to_dataframe(self) -> pd.DataFrame:
        """Export SKU × Warehouse metrics as a flat pandas DataFrame."""
        if not self.sku_warehouse_metrics:
            return pd.DataFrame()
        return pd.DataFrame([m.to_dict() for m in self.sku_warehouse_metrics])


# =====================================================================
# Phase 5B — Return Anomaly Detection Schemas
# =====================================================================


class AnomalySeverity(str, Enum):
    """Severity classification for detected return anomalies."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class AnomalyDirection(str, Enum):
    """Direction of the observed deviation relative to baseline."""

    INCREASE = "INCREASE"
    DECREASE = "DECREASE"
    NONE = "NONE"


class AnomalyType(str, Enum):
    """Taxonomy of return anomaly categories."""

    RETURN_RATE_SPIKE = "RETURN_RATE_SPIKE"
    RETURN_RATE_DROP = "RETURN_RATE_DROP"
    RETURN_VOLUME_SPIKE = "RETURN_VOLUME_SPIKE"
    RETURN_VOLUME_DROP = "RETURN_VOLUME_DROP"
    RETURN_REASON_SHIFT = "RETURN_REASON_SHIFT"
    CHANNEL_RETURN_SHIFT = "CHANNEL_RETURN_SHIFT"
    WAREHOUSE_RETURN_SHIFT = "WAREHOUSE_RETURN_SHIFT"
    SKU_CHANNEL_RETURN_SHIFT = "SKU_CHANNEL_RETURN_SHIFT"
    SKU_WAREHOUSE_RETURN_SHIFT = "SKU_WAREHOUSE_RETURN_SHIFT"
    BASELINE_SHIFT = "BASELINE_SHIFT"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    NO_ANOMALY = "NO_ANOMALY"


class ReturnAnomalyConfig(BaseModel):
    """Configuration and thresholds for return anomaly detection."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    z_score_threshold: float = Field(
        default=2.0,
        ge=0.0,
        description="Minimum absolute z-score triggering an anomaly flag",
    )
    critical_z_score: float = Field(
        default=3.0,
        ge=0.0,
        description="Threshold for CRITICAL severity (|z| >= critical_z_score)",
    )
    high_z_score: float = Field(
        default=2.5,
        ge=0.0,
        description="Threshold for HIGH severity (high_z_score <= |z| < critical_z_score)",
    )
    medium_z_score: float = Field(
        default=2.0,
        ge=0.0,
        description="Threshold for MEDIUM severity (medium_z_score <= |z| < high_z_score)",
    )
    low_z_score: float = Field(
        default=1.5,
        ge=0.0,
        description="Threshold for LOW severity (low_z_score <= |z| < medium_z_score)",
    )
    min_sold_units: int = Field(
        default=30,
        ge=1,
        alias="minimum_sold_units",
        description="Minimum sold units sample required for reliable return rate calculation",
    )
    min_return_count: int = Field(
        default=5,
        ge=1,
        alias="minimum_return_count",
        description="Minimum return event sample size for rate reliability",
    )
    min_history_periods: int = Field(
        default=3,
        ge=1,
        alias="minimum_history_periods",
        description="Minimum historical periods required to construct a baseline distribution",
    )
    use_mad: bool = Field(
        default=False,
        description="If True, uses Median Absolute Deviation (MAD) modified z-scores instead of sample std",
    )
    reason_shift_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum change in reason percentage share (e.g. 0.15 = 15 pp) triggering a shift flag",
    )
    rate_delta_threshold: float = Field(
        default=0.05,
        ge=0.0,
        description="Minimum absolute change in return rate for shift classification",
    )
    volume_spike_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        description="Volume spike multiplier relative to baseline average returned units",
    )

    @property
    def minimum_sold_units(self) -> int:
        return self.min_sold_units

    @property
    def minimum_return_count(self) -> int:
        return self.min_return_count

    @property
    def minimum_history_periods(self) -> int:
        return self.min_history_periods


class ReturnAnomaly(BaseModel):
    """Standardized factual return anomaly record."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    anomaly_id: str = Field(..., description="Deterministic SHA-256 identifier for this anomaly")
    dimension: str = Field(..., description="Analytical dimension (e.g. SKU, CHANNEL, WAREHOUSE, REASON, TIME_SERIES)")
    entity_id: str = Field(..., description="Dimension entity key (e.g. SKU_001, CH_01, DEFECTIVE)")
    anomaly_type: AnomalyType = Field(..., description="Classified anomaly type")
    severity: AnomalySeverity = Field(..., description="Assessed severity level")
    direction: AnomalyDirection = Field(..., description="Deviation direction (INCREASE, DECREASE, NONE)")
    current_value: float = Field(..., description="Observed metric value in current period")
    baseline_value: float = Field(..., description="Baseline reference metric value (mean/median)")
    metric_name: str = Field(default="return_rate", description="Evaluated metric name")
    z_score: Optional[float] = Field(default=None, description="Standard or modified z-score")
    mad_score: Optional[float] = Field(default=None, description="Median Absolute Deviation score if computed")
    percentage_change: Optional[float] = Field(default=None, description="Percentage change relative to baseline")
    sample_size: int = Field(default=0, ge=0, description="Sold units or relevant sample volume")
    is_sufficient_sample: bool = Field(default=True, description="True if sample size meets minimum thresholds")
    period: Optional[str] = Field(default=None, description="Evaluated time period identifier")
    as_of_date: Optional[str] = Field(default=None, description="Evaluation cutoff date")
    rationale: str = Field(default="", description="Deterministic, explainable factual summary")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual details")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["anomaly_type"] = self.anomaly_type.value if hasattr(self.anomaly_type, "value") else str(self.anomaly_type)
        d["severity"] = self.severity.value if hasattr(self.severity, "value") else str(self.severity)
        d["direction"] = self.direction.value if hasattr(self.direction, "value") else str(self.direction)
        d["current_value"] = round(self.current_value, 4)
        d["baseline_value"] = round(self.baseline_value, 4)
        if self.z_score is not None:
            d["z_score"] = round(self.z_score, 4)
        if self.mad_score is not None:
            d["mad_score"] = round(self.mad_score, 4)
        if self.percentage_change is not None:
            d["percentage_change"] = round(self.percentage_change, 4)
        return d


class ReturnAnomalyResult(BaseModel):
    """Consolidated container for return anomaly detection results."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    anomalies: List[ReturnAnomaly] = Field(default_factory=list)
    total_anomalies: int = Field(default=0, ge=0)
    critical_count: int = Field(default=0, ge=0)
    high_count: int = Field(default=0, ge=0)
    medium_count: int = Field(default=0, ge=0)
    low_count: int = Field(default=0, ge=0)
    as_of_date: Optional[str] = None
    config: Optional[ReturnAnomalyConfig] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Export all detected anomalies to a flat pandas DataFrame."""
        if not self.anomalies:
            return pd.DataFrame(columns=[
                "anomaly_id", "dimension", "entity_id", "anomaly_type", "severity",
                "direction", "current_value", "baseline_value", "metric_name",
                "z_score", "mad_score", "percentage_change", "sample_size",
                "is_sufficient_sample", "period", "as_of_date", "rationale"
            ])
        return pd.DataFrame([a.to_dict() for a in self.anomalies])

    def filter_by_severity(self, severity: Union[str, AnomalySeverity]) -> List[ReturnAnomaly]:
        """Filter anomalies by severity."""
        target = (severity.value if isinstance(severity, AnomalySeverity) else str(severity)).strip().upper()
        return [
            a for a in self.anomalies
            if (a.severity.value if hasattr(a.severity, "value") else str(a.severity)).strip().upper() == target
        ]

    def filter_by_dimension(self, dimension: str) -> List[ReturnAnomaly]:
        """Filter anomalies by dimension (e.g. SKU, CHANNEL, WAREHOUSE)."""
        dim_norm = dimension.strip().upper()
        return [a for a in self.anomalies if a.dimension.strip().upper() == dim_norm]

    def filter_by_type(self, anomaly_type: Union[str, AnomalyType]) -> List[ReturnAnomaly]:
        """Filter anomalies by anomaly type."""
        target = (anomaly_type.value if isinstance(anomaly_type, AnomalyType) else str(anomaly_type)).strip().upper()
        return [
            a for a in self.anomalies
            if (a.anomaly_type.value if hasattr(a.anomaly_type, "value") else str(a.anomaly_type)).strip().upper() == target
        ]


# =====================================================================
# Phase 5C-1 — Return Prediction Dataset & Label Engineering Schemas
# =====================================================================


class FeatureDefinition(BaseModel):
    """Metadata specification documenting a predictive feature."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Unique feature identifier in feature matrix X")
    data_type: str = Field(..., description="Data type representation (e.g. float, int, category)")
    source: str = Field(..., description="Source entity/table (sales, products, calendar, historical returns)")
    calculation: str = Field(..., description="Precise calculation methodology")
    prediction_time_availability: str = Field(..., description="Availability confirmation at order creation")
    leakage_risk_assessment: str = Field(..., description="Evaluation and mitigation of target leakage risk")


class ReturnPredictionRecord(BaseModel):
    """Normalized order-line level prediction record."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    prediction_id: str = Field(..., description="Unique prediction record identifier")
    sale_id: str = Field(..., description="Primary transaction line ID")
    order_id: str = Field(..., description="Order identifier")
    sku_id: str = Field(..., description="Product SKU ID")
    warehouse_id: str = Field(..., description="Fulfillment facility ID")
    channel_id: str = Field(..., description="Sales channel ID")
    prediction_date: str = Field(..., description="Prediction evaluation timestamp or ISO date")
    target_returned: int = Field(..., ge=0, le=1, description="Binary classification target: 1 if returned, 0 if not")
    target_return_quantity: int = Field(default=0, ge=0, description="Total units returned under this order line")
    target_return_date: Optional[str] = Field(default=None, description="Internal labeling return date, forbidden in features")


class ReturnPredictionConfig(BaseModel):
    """Configuration settings for supervised return dataset building."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    return_window_days: int = Field(
        default=30,
        ge=1,
        description="Maximum calendar days post-sale during which a return counts toward target=1",
    )
    prediction_cutoff_policy: str = Field(
        default="exclude_immature",
        description="Policy for handling orders placed within return_window_days of dataset end: exclude_immature, include_all",
    )
    minimum_history_days: int = Field(
        default=0,
        ge=0,
        alias="min_history_days",
        description="Minimum platform operation days before generating training records",
    )
    temporal_train_ratio: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Fraction of earliest chronological data allocated to training",
    )
    temporal_validation_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        alias="temporal_val_ratio",
        description="Fraction of chronological data allocated to validation",
    )
    temporal_test_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Fraction of chronological data allocated to holdout testing",
    )
    missing_value_policy: str = Field(
        default="fallback_hierarchy",
        description="Strategy for unresolved historical features: fallback_hierarchy, nan, zero",
    )
    as_of_date: Optional[Union[str, date, datetime]] = Field(
        default=None,
        description="Optional evaluation cutoff date for strict anti-leakage filtering",
    )

    @property
    def temporal_val_ratio(self) -> float:
        return self.temporal_validation_ratio

    @property
    def minimum_history(self) -> int:
        return self.minimum_history_days


class ReturnPredictionQualityReport(BaseModel):
    """Data quality and integrity audit report for return prediction dataset."""

    model_config = ConfigDict(extra="allow")

    total_input_sales: int = Field(default=0, ge=0)
    total_prediction_rows: int = Field(default=0, ge=0)
    duplicate_prediction_rows: int = Field(default=0, ge=0)
    missing_order_id_count: int = Field(default=0, ge=0)
    missing_sku_count: int = Field(default=0, ge=0)
    missing_sale_date_count: int = Field(default=0, ge=0)
    invalid_quantity_count: int = Field(default=0, ge=0)
    invalid_price_count: int = Field(default=0, ge=0)
    impossible_return_quantity_count: int = Field(default=0, ge=0)
    unmapped_returns_count: int = Field(default=0, ge=0)
    unknown_sku_count: int = Field(default=0, ge=0)
    unknown_warehouse_count: int = Field(default=0, ge=0)
    unknown_channel_count: int = Field(default=0, ge=0)
    immature_orders_excluded: int = Field(default=0, ge=0)
    leakage_violations: List[str] = Field(default_factory=list)
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    is_clean: bool = Field(default=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class TemporalSplitInfo(BaseModel):
    """Information detailing chronological train / validation / test boundaries."""

    model_config = ConfigDict(extra="allow")

    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    train_row_count: int = 0
    val_start_date: Optional[str] = None
    val_end_date: Optional[str] = None
    val_row_count: int = 0
    test_start_date: Optional[str] = None
    test_end_date: Optional[str] = None
    test_row_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# =====================================================================
# Phase 5C-2A — Return Prediction ML Modeling & Evaluation Schemas
# =====================================================================


class LogisticRegressionModelConfig(BaseModel):
    """Configuration hyperparameters for Logistic Regression baseline."""

    model_config = ConfigDict(extra="allow")

    C: float = Field(default=1.0, gt=0, description="Inverse of regularization strength")
    penalty: str = Field(default="l2", description="Regularization penalty specification")
    solver: str = Field(default="lbfgs", description="Optimization algorithm solver")
    max_iter: int = Field(default=1000, ge=100, description="Maximum solver iterations")
    class_weight: Optional[Union[str, Dict[int, float]]] = Field(
        default="balanced",
        description="Class weighting strategy ('balanced', None, or custom weights)",
    )
    random_state: int = Field(default=42, description="Deterministic random state seed")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class LightGBMModelConfig(BaseModel):
    """Configuration hyperparameters for LightGBM baseline."""

    model_config = ConfigDict(extra="allow")

    n_estimators: int = Field(default=100, ge=10, description="Number of boosting iterations")
    learning_rate: float = Field(default=0.05, gt=0, le=1.0, description="Boosting learning rate")
    num_leaves: int = Field(default=31, ge=2, description="Maximum tree leaves for base learners")
    max_depth: int = Field(default=6, description="Maximum tree depth (-1 for no limit)")
    subsample: float = Field(default=0.8, gt=0, le=1.0, description="Row subsample ratio")
    colsample_bytree: float = Field(default=0.8, gt=0, le=1.0, description="Column subsample ratio")
    min_child_samples: int = Field(default=20, ge=1, description="Minimum samples per leaf")
    class_weight: Optional[Union[str, Dict[int, float]]] = Field(
        default="balanced",
        description="Class weighting strategy ('balanced', None, or custom weights)",
    )
    random_state: int = Field(default=42, description="Deterministic random state seed")
    early_stopping_rounds: Optional[int] = Field(
        default=10,
        description="Validation early stopping rounds (None to disable)",
    )
    verbose: int = Field(default=-1, description="Verbosity level for LightGBM solver")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ModelEvaluationMetrics(BaseModel):
    """Structured evaluation metrics across train, validation, or test partitions."""

    model_config = ConfigDict(extra="allow")

    split: str = Field(..., description="Dataset split evaluated: 'train', 'validation', or 'test'")
    roc_auc: Optional[float] = Field(default=None, description="Area under ROC curve")
    pr_auc: Optional[float] = Field(default=None, description="Area under Precision-Recall curve")
    brier_score: Optional[float] = Field(default=None, description="Brier score probability calibration metric")
    precision: Optional[float] = Field(default=None, description="Positive class precision at threshold")
    recall: Optional[float] = Field(default=None, description="Positive class recall at threshold")
    f1_score: Optional[float] = Field(default=None, description="Harmonic mean of precision and recall at threshold")
    threshold: float = Field(default=0.50, description="Decision threshold applied for binary classification")
    confusion_matrix: Optional[Dict[str, int]] = Field(
        default=None,
        description="Confusion matrix counts: tp, fp, fn, tn",
    )
    row_count: int = Field(default=0, ge=0, description="Number of evaluated rows in split")
    positive_count: int = Field(default=0, ge=0, description="Observed positive class count (returned=1)")
    negative_count: int = Field(default=0, ge=0, description="Observed negative class count (returned=0)")
    positive_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Empirical positive class fraction")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ThresholdAnalysisPoint(BaseModel):
    """Classification metrics at a specific candidate probability threshold."""

    model_config = ConfigDict(extra="allow")

    threshold: float = Field(..., ge=0.0, le=1.0, description="Evaluated probability decision cutoff")
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision at threshold")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall at threshold")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1-score at threshold")
    tp: int = Field(default=0, ge=0, description="True positives")
    fp: int = Field(default=0, ge=0, description="False positives")
    fn: int = Field(default=0, ge=0, description="False negatives")
    tn: int = Field(default=0, ge=0, description="True negatives")
    predicted_positive_count: int = Field(default=0, ge=0, description="Predicted positive count")
    predicted_positive_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction predicted positive")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class SegmentEvaluationMetric(BaseModel):
    """Performance metrics evaluated on a specific cohort or entity slice."""

    model_config = ConfigDict(extra="allow")

    dimension: str = Field(..., description="Evaluated segment dimension (e.g. sku_id, channel_id, warehouse_id)")
    segment_value: str = Field(..., description="Specific categorical level or slice value")
    sample_size: int = Field(..., ge=0, description="Total evaluated rows in segment")
    positive_count: int = Field(default=0, ge=0, description="Observed positive return count in segment")
    positive_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Empirical return rate in segment")
    is_sufficient_sample: bool = Field(default=True, description="Whether sample size meets minimum reliability threshold")
    brier_score: Optional[float] = Field(default=None, description="Brier score for segment probabilities")
    roc_auc: Optional[float] = Field(default=None, description="ROC-AUC within segment (if both classes present)")
    pr_auc: Optional[float] = Field(default=None, description="PR-AUC within segment (if both classes present)")
    precision: Optional[float] = Field(default=None, description="Precision at threshold within segment")
    recall: Optional[float] = Field(default=None, description="Recall at threshold within segment")
    f1_score: Optional[float] = Field(default=None, description="F1-score at threshold within segment")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ReturnPredictionOutputRecord(BaseModel):
    """Individual prediction result combining order metadata and calibrated probability."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    prediction_id: str = Field(..., description="Unique prediction row identifier")
    sale_id: str = Field(..., description="Primary sales transaction line ID")
    order_id: str = Field(..., description="Sales order ID")
    sku_id: str = Field(..., description="Product SKU ID")
    warehouse_id: str = Field(..., description="Fulfillment facility ID")
    channel_id: str = Field(..., description="Sales channel ID")
    prediction_date: str = Field(..., description="Evaluation order date (YYYY-MM-DD)")
    actual_target: Optional[int] = Field(default=None, ge=0, le=1, description="Ground truth return outcome if known")
    predicted_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated probability of return in [0, 1]")
    predicted_class: int = Field(..., ge=0, le=1, description="Binary classification at decision threshold")
    model_name: str = Field(..., description="Name of the estimating ML model")
    threshold: float = Field(default=0.50, description="Probability decision threshold applied")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ReturnModelMetadata(BaseModel):
    """Comprehensive, serializable metadata cataloging a trained model baseline."""

    model_config = ConfigDict(extra="allow")

    model_name: str = Field(..., description="Name of the model (e.g. 'LogisticRegression', 'LightGBM')")
    model_version: str = Field(default="1.0.0", description="Model release/version string")
    training_timestamp: str = Field(..., description="ISO 8601 timestamp of training execution")
    feature_names: List[str] = Field(default_factory=list, description="Ordered list of feature column names used")
    feature_count: int = Field(default=0, ge=0, description="Total number of input features in X")
    training_row_count: int = Field(default=0, ge=0, description="Rows in training partition")
    validation_row_count: int = Field(default=0, ge=0, description="Rows in validation partition")
    test_row_count: int = Field(default=0, ge=0, description="Rows in holdout test partition")
    positive_count: int = Field(default=0, ge=0, description="Training positive class count")
    negative_count: int = Field(default=0, ge=0, description="Training negative class count")
    class_weight_strategy: str = Field(..., description="Explicit class imbalance weighting strategy used")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Configuration hyperparameters")
    random_seed: int = Field(default=42, description="Random seed used for deterministic execution")
    threshold_used: float = Field(default=0.50, description="Reporting probability decision threshold")
    train_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Training partition metrics")
    validation_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Validation partition metrics")
    test_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Holdout test partition metrics")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# =====================================================================
# Phase 5C-2B — Return Probability Calibration & Risk Layer Schemas
# =====================================================================


class ReturnRiskBand(str, Enum):
    """Categorical risk band classifications for calibrated return risk."""

    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class CalibrationMethod(str, Enum):
    """Supported probability calibration methodologies."""

    NONE = "none"
    SIGMOID = "sigmoid"  # Platt scaling / logistic calibration
    ISOTONIC = "isotonic"  # Non-parametric isotonic regression


class CalibrationBin(BaseModel):
    """Reliability curve bin capturing predicted vs observed probabilities."""

    model_config = ConfigDict(extra="allow")

    bin_index: int = Field(..., ge=0, description="Zero-based bin ordering index")
    lower_bound: float = Field(..., ge=0.0, le=1.0, description="Bin lower probability threshold")
    upper_bound: float = Field(..., ge=0.0, le=1.0, description="Bin upper probability threshold")
    sample_count: int = Field(default=0, ge=0, description="Observations within this probability bin")
    mean_predicted_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean confidence in bin")
    observed_return_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Empirical return frequency in bin")
    absolute_error: float = Field(default=0.0, ge=0.0, le=1.0, description="|mean_predicted - observed_rate|")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class CalibrationMetrics(BaseModel):
    """Comprehensive calibration evaluation metrics and reliability curve."""

    model_config = ConfigDict(extra="allow")

    method: str = Field(..., description="Calibration method evaluated ('raw', 'sigmoid', 'isotonic')")
    brier_score: float = Field(..., ge=0.0, le=1.0, description="Mean squared probability error")
    log_loss: float = Field(..., ge=0.0, description="Cross-entropy / logarithmic loss")
    ece: float = Field(..., ge=0.0, le=1.0, description="Expected Calibration Error (weighted bin error)")
    mce: float = Field(..., ge=0.0, le=1.0, description="Maximum Calibration Error across non-empty bins")
    mean_predicted_probability: float = Field(..., ge=0.0, le=1.0, description="Mean predicted probability")
    observed_positive_rate: float = Field(..., ge=0.0, le=1.0, description="Empirical ground truth return rate")
    roc_auc: Optional[float] = Field(default=None, description="ROC-AUC of probabilities")
    pr_auc: Optional[float] = Field(default=None, description="PR-AUC of probabilities")
    bins: List[CalibrationBin] = Field(default_factory=list, description="Reliability curve bin details")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class RiskBandBoundary(BaseModel):
    """Specification of a single probability interval mapped to a risk band."""

    model_config = ConfigDict(extra="allow")

    band: ReturnRiskBand = Field(..., description="Risk band classification enum")
    lower_bound: float = Field(..., ge=0.0, le=1.0, description="Inclusive lower probability bound")
    upper_bound: float = Field(..., ge=0.0, le=1.0, description="Upper probability bound")
    description: str = Field(default="", description="Operational description of this risk tier")


class RiskPolicyConfig(BaseModel):
    """Configurable risk classification policy mapping probabilities to risk bands."""

    model_config = ConfigDict(extra="allow")

    policy_version: str = Field(default="1.0.0-prototype", description="Policy version identifier")
    boundaries: List[RiskBandBoundary] = Field(
        default_factory=lambda: [
            RiskBandBoundary(band=ReturnRiskBand.VERY_LOW, lower_bound=0.00, upper_bound=0.10, description="Very low return hazard; standard fulfillment"),
            RiskBandBoundary(band=ReturnRiskBand.LOW, lower_bound=0.10, upper_bound=0.25, description="Low return hazard; routine logistics"),
            RiskBandBoundary(band=ReturnRiskBand.MEDIUM, lower_bound=0.25, upper_bound=0.50, description="Moderate return hazard; sizing/fit advisory candidate"),
            RiskBandBoundary(band=ReturnRiskBand.HIGH, lower_bound=0.50, upper_bound=0.75, description="Elevated return hazard; inspection or routing candidate"),
            RiskBandBoundary(band=ReturnRiskBand.VERY_HIGH, lower_bound=0.75, upper_bound=1.00, description="Critical return hazard; targeted post-purchase intervention"),
        ],
        description="Ordered list of non-overlapping probability intervals spanning [0, 1]",
    )

    @field_validator("boundaries")
    @classmethod
    def validate_boundaries(cls, v: List[RiskBandBoundary]) -> List[RiskBandBoundary]:
        if not v:
            raise ValueError("Risk policy must contain at least one boundary interval.")
        # Check individual boundary validity
        for b in v:
            if b.lower_bound < 0.0 or b.upper_bound > 1.0:
                raise ValueError(f"Boundary bounds must be in [0.0, 1.0], got [{b.lower_bound}, {b.upper_bound}]")
            if b.lower_bound >= b.upper_bound:
                raise ValueError(f"Invalid boundary: lower_bound ({b.lower_bound}) >= upper_bound ({b.upper_bound})")

        # Sort by lower bound
        sorted_bounds = sorted(v, key=lambda b: b.lower_bound)
        if sorted_bounds[0].lower_bound > 0.0:
            raise ValueError(f"First boundary must start at 0.0, got {sorted_bounds[0].lower_bound}")
        if sorted_bounds[-1].upper_bound < 1.0:
            raise ValueError(f"Last boundary must end at 1.0, got {sorted_bounds[-1].upper_bound}")

        for i in range(len(sorted_bounds) - 1):
            curr_b = sorted_bounds[i]
            next_b = sorted_bounds[i + 1]
            if curr_b.upper_bound > next_b.lower_bound + 1e-6:
                raise ValueError(
                    f"Overlapping risk boundaries: {curr_b.band} ({curr_b.lower_bound}-{curr_b.upper_bound}) "
                    f"overlaps with {next_b.band} ({next_b.lower_bound}-{next_b.upper_bound})"
                )
            if curr_b.upper_bound < next_b.lower_bound - 1e-6:
                raise ValueError(
                    f"Gap in risk boundaries between {curr_b.band} (ends at {curr_b.upper_bound}) "
                    f"and {next_b.band} (starts at {next_b.lower_bound})"
                )
        return sorted_bounds

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ReturnRiskResult(BaseModel):
    """Business-facing risk prediction combining raw/calibrated probabilities and risk band."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    prediction_id: str = Field(..., description="Unique prediction row identifier")
    sale_id: str = Field(..., description="Primary sales transaction line ID")
    order_id: str = Field(..., description="Sales order ID")
    sku_id: str = Field(..., description="Product SKU ID")
    warehouse_id: str = Field(..., description="Fulfillment facility ID")
    channel_id: str = Field(..., description="Sales channel ID")
    prediction_date: str = Field(..., description="Evaluation order date (YYYY-MM-DD)")
    model_name: str = Field(..., description="Name of the estimating ML model")
    raw_probability: float = Field(..., ge=0.0, le=1.0, description="Uncalibrated output probability from ML model")
    calibrated_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated return probability")
    risk_band: ReturnRiskBand = Field(..., description="Assigned categorical risk band")
    risk_policy_version: str = Field(..., description="Version of the risk policy used for classification")
    calibration_method: str = Field(..., description="Calibration method applied ('none', 'sigmoid', 'isotonic')")
    interpretation: str = Field(..., description="Deterministic, non-causal statistical interpretation string")
    actual_target: Optional[int] = Field(default=None, ge=0, le=1, description="Ground truth return outcome if known")
    threshold: float = Field(default=0.50, description="Binary decision threshold")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class RiskBandEvaluation(BaseModel):
    """Empirical audit and validation metrics for a specific risk band."""

    model_config = ConfigDict(extra="allow")

    risk_band: str = Field(..., description="Evaluated risk band name")
    record_count: int = Field(default=0, ge=0, description="Total observations classified into this risk band")
    percentage_of_records: float = Field(default=0.0, ge=0.0, le=100.0, description="Share of total cohort in this band")
    actual_returned_count: int = Field(default=0, ge=0, description="Observed returns in this band")
    observed_return_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Empirical return frequency")
    average_calibrated_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean calibrated confidence")
    calibration_error: float = Field(default=0.0, ge=0.0, le=1.0, description="|mean_calibrated - observed_rate|")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# =====================================================================
# Phase 5C-3 — Return Intervention / Cost-Utility Engine Schemas
# =====================================================================


class FinancialInputStatus(str, Enum):
    """Classification of financial input completeness for cost-utility analysis."""

    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"
    INSUFFICIENT_FINANCIAL_INPUTS = "INSUFFICIENT_FINANCIAL_INPUTS"


class InterventionDecision(str, Enum):
    """Categorical intervention decisions emitted by the cost-utility engine."""

    INTERVENTION_INDICATED = "INTERVENTION_INDICATED"
    NO_INTERVENTION_INDICATED = "NO_INTERVENTION_INDICATED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class InterventionRecommendationCode(str, Enum):
    """Actionable recommendation codes for return-risk interventions."""

    REVIEW_RETURN_RISK = "REVIEW_RETURN_RISK"
    REVIEW_HIGH_VALUE_RETURN_EXPOSURE = "REVIEW_HIGH_VALUE_RETURN_EXPOSURE"
    NO_INTERVENTION_INDICATED = "NO_INTERVENTION_INDICATED"
    INSUFFICIENT_FINANCIAL_INPUTS = "INSUFFICIENT_FINANCIAL_INPUTS"
    INSUFFICIENT_PROBABILITY_DATA = "INSUFFICIENT_PROBABILITY_DATA"
    NEGATIVE_MARGIN_REVIEW = "NEGATIVE_MARGIN_REVIEW"
    DATA_QUALITY_ERROR = "DATA_QUALITY_ERROR"


class InterventionPolicyConfig(BaseModel):
    """Configurable business policy determining economic justification for return interventions.

    All thresholds represent prototype configuration rules, NOT business-approved
    or empirically verified corporate policies.
    """

    model_config = ConfigDict(extra="allow")

    policy_version: str = Field(default="1.0.0-prototype", description="Policy version tracking identifier")
    minimum_probability: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Minimum calibrated return probability to consider intervention"
    )
    minimum_expected_exposure: float = Field(
        default=25.0, ge=0.0, description="Minimum expected monetary loss ($) required to trigger intervention"
    )
    high_exposure_threshold: float = Field(
        default=75.0, ge=0.0, description="Monetary threshold ($) qualifying order for HIGH_VALUE review"
    )
    minimum_margin_at_risk: float = Field(
        default=0.0, ge=0.0, description="Optional minimum margin ($) at risk for intervention"
    )
    require_return_costs: bool = Field(
        default=False,
        description="If True, missing return-specific costs force INSUFFICIENT_DATA; if False, revenue/margin proxies are used",
    )
    default_return_shipping_cost: Optional[float] = Field(
        default=None, ge=0.0, description="Optional default shipping cost per return if unobserved"
    )
    default_handling_cost: Optional[float] = Field(
        default=None, ge=0.0, description="Optional default handling cost per return if unobserved"
    )
    default_restocking_cost: Optional[float] = Field(
        default=None, ge=0.0, description="Optional default restocking cost per return if unobserved"
    )
    return_cost_model: str = Field(
        default="item_revenue",
        description="Fallback return impact model: 'item_revenue' (full revenue at risk), 'gross_margin', or 'strict'",
    )
    description: str = Field(
        default="Prototype return intervention and cost-utility policy",
        description="Operational context for this policy specification",
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ReturnInterventionRecommendation(BaseModel):
    """Deterministic return-risk intervention recommendation for a specific order line."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    intervention_id: str = Field(..., description="Deterministic unique SHA-256 recommendation identifier")
    sale_id: str = Field(..., description="Sales transaction line ID")
    order_id: str = Field(..., description="Sales order ID")
    sku_id: str = Field(..., description="Product SKU ID")
    warehouse_id: str = Field(..., description="Fulfillment facility ID")
    channel_id: str = Field(..., description="Sales channel ID")
    prediction_date: Optional[str] = Field(default=None, description="Transaction order date (YYYY-MM-DD)")
    calibrated_return_probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Calibrated probability of return"
    )
    raw_probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Uncalibrated model output probability"
    )
    risk_band: Optional[str] = Field(default=None, description="Assigned risk band (e.g. VERY_LOW, LOW, etc.)")
    quantity: Optional[int] = Field(default=None, description="Sold order-line quantity")
    unit_price: Optional[float] = Field(default=None, description="Unit sales price in currency")
    discount: Optional[float] = Field(default=None, description="Line discount in currency")
    revenue: Optional[float] = Field(default=None, description="Observed/realized net line revenue")
    unit_cost: Optional[float] = Field(default=None, description="Product standard procurement unit cost")
    estimated_product_cost: Optional[float] = Field(
        default=None, description="Calculated cost of goods: quantity * unit_cost"
    )
    estimated_gross_margin: Optional[float] = Field(
        default=None, description="Calculated gross margin: revenue - estimated_product_cost"
    )
    estimated_return_cost: Optional[float] = Field(
        default=None, description="Estimated reverse logistics processing costs (shipping, handling, restocking)"
    )
    estimated_return_impact: Optional[float] = Field(
        default=None, description="Estimated total monetary loss if order line is returned"
    )
    expected_return_exposure: Optional[float] = Field(
        default=None, description="Expected economic loss: calibrated_probability * estimated_return_impact"
    )
    financial_status: FinancialInputStatus = Field(
        ..., description="Data completeness status: SUFFICIENT, PARTIAL, INSUFFICIENT"
    )
    decision: InterventionDecision = Field(
        ..., description="Analytical intervention decision outcome"
    )
    recommendation_code: InterventionRecommendationCode = Field(
        ..., description="Specific operational recommendation code"
    )
    rationale: str = Field(..., description="Deterministic, human-readable statistical rationale")
    missing_inputs: List[str] = Field(
        default_factory=list, description="Explicit list of unobserved or missing financial/probability inputs"
    )
    policy_version: str = Field(..., description="Policy configuration version applied")
    as_of_date: Optional[str] = Field(default=None, description="Cutoff date applied for leakage protection")
    generated_at: str = Field(..., description="ISO 8601 generation timestamp")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in [
            "calibrated_return_probability",
            "raw_probability",
            "unit_price",
            "discount",
            "revenue",
            "unit_cost",
            "estimated_product_cost",
            "estimated_gross_margin",
            "estimated_return_cost",
            "estimated_return_impact",
            "expected_return_exposure",
        ]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4 if "prob" in k else 2)
        return d


class PortfolioInterventionSummary(BaseModel):
    """Portfolio-level aggregation of return intervention opportunities and financial exposure."""

    model_config = ConfigDict(extra="allow")

    total_order_lines_analyzed: int = Field(default=0, ge=0, description="Total order lines evaluated")
    sufficient_financial_records: int = Field(default=0, ge=0, description="Records with complete financial data")
    partial_financial_records: int = Field(default=0, ge=0, description="Records with partial financial data")
    insufficient_financial_records: int = Field(default=0, ge=0, description="Records with insufficient financial data")
    intervention_indicated_count: int = Field(default=0, ge=0, description="Count of INTERVENTION_INDICATED records")
    no_intervention_count: int = Field(default=0, ge=0, description="Count of NO_INTERVENTION_INDICATED records")
    review_required_count: int = Field(default=0, ge=0, description="Count of REVIEW_REQUIRED records")
    insufficient_data_count: int = Field(default=0, ge=0, description="Count of INSUFFICIENT_DATA records")
    total_revenue_represented: Optional[float] = Field(
        default=None, description="Total revenue of evaluated lines (None if unobserved)"
    )
    total_estimated_return_impact: Optional[float] = Field(
        default=None, description="Sum of estimated return impacts where available (None if unobserved)"
    )
    total_expected_return_exposure: Optional[float] = Field(
        default=None, description="Sum of expected return exposure where available (None if unobserved)"
    )
    breakdown_by_sku: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Summary grouped by SKU")
    breakdown_by_channel: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Summary grouped by Channel")
    breakdown_by_warehouse: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Summary grouped by Warehouse")
    breakdown_by_risk_band: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Summary grouped by Risk Band")
    breakdown_by_decision: Dict[str, int] = Field(default_factory=dict, description="Record counts grouped by decision")
    policy_version: str = Field(default="", description="Policy configuration version used")
    generated_at: str = Field(default="", description="ISO 8601 evaluation timestamp")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        for k in ["total_revenue_represented", "total_estimated_return_impact", "total_expected_return_exposure"]:
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 2)
        return d


