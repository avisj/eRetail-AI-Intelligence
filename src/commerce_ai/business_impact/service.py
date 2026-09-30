"""Business Impact & Opportunity Quantification Service (Phase 6F).

Orchestrates:
1. Converting Phase 6E cross-domain signals into quantified financial and operational exposures
2. Double-counting protection (distinguishing gross signal exposure from deduplicated capital exposure)
3. Currency isolation (strictly prohibiting naive cross-currency addition without FX conversion)
4. Historical point-in-time safety (honoring as_of_date across all calculations)
5. Auditability and traceability from impact records back to root signals
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.cross_domain.schemas import (
    CrossDomainBusinessRecord,
    CrossDomainSignal,
    SignalCategory,
    SignalSeverity,
)
from commerce_ai.business_impact.schemas import (
    BusinessImpactConfig,
    BusinessImpactPortfolioSummary,
    BusinessImpactRecord,
    BusinessImpactResult,
    CalculationStatus,
    ImpactCategory,
    ImpactConfidence,
    ImpactType,
)
from commerce_ai.business_impact.quantification import (
    calculate_physical_capital_exposure,
    deduplicate_portfolio_exposure,
    generate_impact_id,
    quantify_signal_impact,
)


class BusinessImpactService:
    """Service providing auditable, deterministic business impact and exposure quantification."""

    def __init__(self, config: Optional[BusinessImpactConfig] = None):
        self.config = config or BusinessImpactConfig()

    def _resolve_config(
        self,
        config: Optional[BusinessImpactConfig] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> BusinessImpactConfig:
        cfg = config or self.config
        if as_of_date is not None:
            as_of_str = as_of_date.isoformat() if hasattr(as_of_date, "isoformat") else str(as_of_date)
            cfg = cfg.model_copy(update={"as_of_date": as_of_str})
        return cfg

    def calculate_impact_for_signal(
        self,
        signal: CrossDomainSignal,
        record: Optional[CrossDomainBusinessRecord] = None,
        config: Optional[BusinessImpactConfig] = None,
    ) -> BusinessImpactRecord:
        """Quantify exposure for a single cross-domain signal."""
        cfg = self._resolve_config(config)
        return quantify_signal_impact(signal=signal, record=record, config=cfg)

    def calculate_signal_impact(
        self,
        signal: CrossDomainSignal,
        record: Optional[CrossDomainBusinessRecord] = None,
        config: Optional[BusinessImpactConfig] = None,
    ) -> BusinessImpactRecord:
        """Convenience alias for calculate_impact_for_signal."""
        return self.calculate_impact_for_signal(signal=signal, record=record, config=config)

    def calculate_all_impacts(
        self,
        signals: List[CrossDomainSignal],
        records: Optional[List[CrossDomainBusinessRecord]] = None,
        config: Optional[BusinessImpactConfig] = None,
    ) -> List[BusinessImpactRecord]:
        """Quantify impacts for a collection of signals mapped against business records."""
        cfg = self._resolve_config(config)

        # Build fast lookup map for business records
        record_map: Dict[Tuple[str, Optional[str], Optional[str]], CrossDomainBusinessRecord] = {}
        if records:
            for r in records:
                key = (r.sku_id, r.warehouse_id, r.channel_id)
                record_map[key] = r

        impact_records: List[BusinessImpactRecord] = []
        for sig in signals:
            key = (sig.sku_id or "", sig.warehouse_id, sig.channel_id)
            rec = record_map.get(key)
            if rec is None and sig.sku_id:
                # Try finding by sku_id alone if warehouse_id wasn't exact match
                rec = next((r for r in (records or []) if r.sku_id == sig.sku_id), None)

            imp = self.calculate_impact_for_signal(signal=sig, record=rec, config=cfg)
            impact_records.append(imp)

        return impact_records

    def calculate_currency_summary(
        self,
        records: List[BusinessImpactRecord],
    ) -> Dict[str, float]:
        """Group and sum gross exposure values strictly by ISO currency code."""
        breakdown: Dict[str, float] = {}
        for r in records:
            if r.exposure_value is not None:
                curr = r.exposure_currency or "USD"
                breakdown[curr] = round(breakdown.get(curr, 0.0) + float(r.exposure_value), 2)
        return breakdown

    def build_portfolio_summary(
        self,
        impact_records: List[BusinessImpactRecord],
        total_signals: Optional[int] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[BusinessImpactConfig] = None,
    ) -> BusinessImpactPortfolioSummary:
        """Calculate network-wide portfolio summary metrics and double-counting deduplication."""
        cfg = self._resolve_config(config, as_of_date)
        n_sigs = total_signals if total_signals is not None else len(impact_records)
        n_records = len(impact_records)

        # Currency Isolation Audit
        currency_breakdown = self.calculate_currency_summary(impact_records)
        unique_currencies = list(currency_breakdown.keys())
        if len(unique_currencies) > 1 and not cfg.allow_multi_currency:
            raise ValueError(
                f"Mixed currency detected: {unique_currencies}. "
                f"Portfolio aggregation without explicit FX conversion is disabled."
            )

        reporting_curr = unique_currencies[0] if len(unique_currencies) == 1 else cfg.default_currency

        # Breakdown by Category, Type, Severity & Status
        cat_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        sev_counts: Dict[str, int] = {}
        calc_status_counts: Dict[str, int] = {}
        cat_exposure: Dict[str, float] = {}
        type_exposure: Dict[str, float] = {}
        direct_exp = 0.0
        est_exp = 0.0
        model_exp = 0.0
        insufficient_cnt = 0
        gross_exp = 0.0

        for r in impact_records:
            cat_k = r.impact_category.value if isinstance(r.impact_category, ImpactCategory) else str(r.impact_category)
            cat_counts[cat_k] = cat_counts.get(cat_k, 0) + 1

            type_k = r.impact_type.value if isinstance(r.impact_type, ImpactType) else str(r.impact_type)
            type_counts[type_k] = type_counts.get(type_k, 0) + 1

            sev_k = r.severity.value if isinstance(r.severity, SignalSeverity) else str(r.severity)
            sev_counts[sev_k] = sev_counts.get(sev_k, 0) + 1

            stat_k = r.calculation_status.value if isinstance(r.calculation_status, CalculationStatus) else str(r.calculation_status)
            calc_status_counts[stat_k] = calc_status_counts.get(stat_k, 0) + 1

            if r.exposure_value is not None:
                val = float(r.exposure_value)
                gross_exp += val
                cat_exposure[cat_k] = round(cat_exposure.get(cat_k, 0.0) + val, 2)
                type_exposure[type_k] = round(type_exposure.get(type_k, 0.0) + val, 2)
                if r.confidence_status == ImpactConfidence.DIRECT_OBSERVED:
                    direct_exp += val
                elif r.confidence_status == ImpactConfidence.ESTIMATED:
                    est_exp += val
                elif r.confidence_status == ImpactConfidence.MODEL_BASED:
                    model_exp += val
            else:
                insufficient_cnt += 1

        # Double-counting protection
        deduped_exp, dedup_status = deduplicate_portfolio_exposure(impact_records)
        _, dedup_phys_cap, _ = calculate_physical_capital_exposure(impact_records)

        return BusinessImpactPortfolioSummary(
            total_signal_count=n_sigs,
            impact_record_count=n_records,
            currency_breakdown=currency_breakdown,
            impact_category_counts=cat_counts,
            impact_type_counts=type_counts,
            severity_counts=sev_counts,
            exposure_totals_by_category=cat_exposure,
            exposure_totals_by_type=type_exposure,
            calculation_status_counts=calc_status_counts,
            direct_observed_exposure=round(direct_exp, 2),
            estimated_exposure=round(est_exp, 2),
            model_based_exposure=round(model_exp, 2),
            insufficient_data_count=insufficient_cnt,
            gross_signal_exposure=round(gross_exp, 2),
            deduplicated_exposure=deduped_exp,
            deduplicated_physical_capital_exposure=dedup_phys_cap,
            deduplication_status=dedup_status,
            currency=reporting_curr,
            as_of_date=str(cfg.as_of_date) if cfg.as_of_date is not None else None,
        )

    def quantify_portfolio(
        self,
        signals: List[CrossDomainSignal],
        records: Optional[List[CrossDomainBusinessRecord]] = None,
        as_of_date: Optional[Union[str, date, datetime]] = None,
        config: Optional[BusinessImpactConfig] = None,
    ) -> BusinessImpactResult:
        """Run the comprehensive end-to-end business impact quantification pipeline."""
        cfg = self._resolve_config(config, as_of_date)

        impacts = self.calculate_all_impacts(signals=signals, records=records, config=cfg)
        summary = self.build_portfolio_summary(
            impact_records=impacts,
            total_signals=len(signals),
            as_of_date=cfg.as_of_date,
            config=cfg,
        )

        return BusinessImpactResult(
            impact_records=impacts,
            portfolio_summary=summary,
            as_of_date=str(cfg.as_of_date) if cfg.as_of_date is not None else None,
        )
