"""Configurable Model Selection and Segment-Level Routing Engine.

Enables automated, objective selection of champion forecasting models based on
cross-validated error metrics (WAPE, RMSE, Bias), operational constraints,
and segment-specific demand characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class SelectionCriteria:
    """Configurable criteria for champion model selection."""

    primary_metric: str = "WAPE (%)"
    secondary_metric: str = "Bias (%)"
    max_failure_rate: float = 0.05
    max_runtime_seconds: Optional[float] = None
    prefer_simpler_baseline_if_within_pct: float = 2.0  # % margin within which simple baseline wins


@dataclass
class ModelSelectionPolicy:
    """Champion model routing policy across segments."""

    global_champion: str
    segment_champions: Dict[str, str] = field(default_factory=dict)
    sku_champions: Dict[str, str] = field(default_factory=dict)
    rationale: Dict[str, Any] = field(default_factory=dict)

    def get_model_for_entity(
        self,
        sku_id: Optional[str] = None,
        segment: Optional[str] = None,
    ) -> str:
        """Route to appropriate model based on hierarchy: SKU > Segment > Global."""
        if sku_id and sku_id in self.sku_champions:
            return self.sku_champions[sku_id]
        if segment and segment in self.segment_champions:
            return self.segment_champions[segment]
        return self.global_champion


class ModelSelector:
    """Automated model selection engine."""

    def __init__(self, criteria: Optional[SelectionCriteria] = None):
        self.criteria = criteria or SelectionCriteria()

    def select_best_model(self, benchmark_table: pd.DataFrame) -> str:
        """Select single overall champion model from benchmark table.

        Args:
            benchmark_table: Standard benchmark DataFrame from ForecastEvaluator.generate_benchmark_table.

        Returns:
            str: Name of the winning model.
        """
        if benchmark_table.empty:
            raise ValueError("Benchmark table is empty; cannot select best model.")

        df = benchmark_table.copy()

        # Filter out unavailable or failed models
        if "Status" in df.columns:
            valid_df = df[df["Status"].isin(["SUCCESS", "PARTIAL_FAIL"])].copy()
            if valid_df.empty:
                # Fallback to any model present
                return str(df["Model"].iloc[0])
        else:
            valid_df = df.copy()

        # Filter by failures if failure counts available
        if "Failures" in valid_df.columns:
            valid_df = valid_df[valid_df["Failures"] == 0]
            if valid_df.empty:
                valid_df = df.copy()

        primary = self.criteria.primary_metric
        if primary not in valid_df.columns:
            primary = "WAPE (%)" if "WAPE (%)" in valid_df.columns else "wape"

        # Exclude NaNs in primary metric
        valid_df = valid_df.dropna(subset=[primary])
        if valid_df.empty:
            return str(df["Model"].iloc[0])

        # Sort by primary metric ascending
        valid_df = valid_df.sort_values(by=primary, ascending=True).reset_index(drop=True)
        best_candidate = valid_df.iloc[0]
        best_model = str(best_candidate["Model"])
        best_score = float(best_candidate[primary])

        # Parsimony principle: If a baseline model is within `prefer_simpler_baseline_if_within_pct`, prefer it
        simple_baselines = ["Seasonal Naive", "Moving Average (7d)", "Naive", "Exponential Smoothing"]
        margin = self.criteria.prefer_simpler_baseline_if_within_pct
        for _, row in valid_df.iterrows():
            m_name = str(row["Model"])
            m_score = float(row[primary])
            if m_name in simple_baselines and m_name != best_model:
                if (m_score - best_score) <= margin:
                    return m_name

        return best_model

    def build_segment_policy(
        self,
        global_benchmark: pd.DataFrame,
        segment_eval_df: Optional[pd.DataFrame] = None,
        sku_eval_df: Optional[pd.DataFrame] = None,
        segment_col: str = "intermittency_class",
    ) -> ModelSelectionPolicy:
        """Construct hierarchical model selection policy.

        Args:
            global_benchmark: Master benchmark table across all models.
            segment_eval_df: Evaluation DataFrame grouped by model and segment (e.g. intermittency_class).
            sku_eval_df: Evaluation DataFrame grouped by model and sku_id.
            segment_col: Column name identifying segment category.

        Returns:
            ModelSelectionPolicy with segment-specific champion routes.
        """
        global_champ = self.select_best_model(global_benchmark)
        segment_champs: Dict[str, str] = {}
        sku_champs: Dict[str, str] = {}
        rationale: Dict[str, Any] = {"global": global_champ}

        # Determine segment winners
        if segment_eval_df is not None and not segment_eval_df.empty and segment_col in segment_eval_df.columns:
            for seg, grp in segment_eval_df.groupby(segment_col):
                metric = "wape" if "wape" in grp.columns else "WAPE (%)"
                valid_grp = grp.dropna(subset=[metric]).sort_values(by=metric, ascending=True)
                if not valid_grp.empty:
                    winner = str(valid_grp.iloc[0]["model_name"])
                    segment_champs[str(seg)] = winner
                    rationale[f"segment_{seg}"] = {
                        "winner": winner,
                        "wape": float(valid_grp.iloc[0][metric]),
                    }

        # Determine SKU winners if sufficient volume
        if sku_eval_df is not None and not sku_eval_df.empty and "sku_id" in sku_eval_df.columns:
            for sku, grp in sku_eval_df.groupby("sku_id"):
                metric = "wape" if "wape" in grp.columns else "WAPE (%)"
                valid_grp = grp.dropna(subset=[metric]).sort_values(by=metric, ascending=True)
                if not valid_grp.empty:
                    sku_champs[str(sku)] = str(valid_grp.iloc[0]["model_name"])

        return ModelSelectionPolicy(
            global_champion=global_champ,
            segment_champions=segment_champs,
            sku_champions=sku_champs,
            rationale=rationale,
        )
