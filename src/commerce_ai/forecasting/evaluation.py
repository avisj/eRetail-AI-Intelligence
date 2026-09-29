"""Forecast Evaluation Engine and Hierarchical Benchmarking.

Computes industry standard metrics (MAE, RMSE, MAPE, WAPE, Bias, Coverage)
across global and multi-level hierarchical dimensions (SKU, Warehouse,
Category, ABC/XYZ, Intermittency type).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from commerce_ai.analytics.metrics import mae, rmse, mape, wape, bias


class ForecastEvaluator:
    """Multi-level hierarchical evaluation and benchmarking engine."""

    @staticmethod
    def evaluate_slice(
        actual: Union[pd.Series, np.ndarray, list],
        predicted: Union[pd.Series, np.ndarray, list],
        lower_bound: Optional[Union[pd.Series, np.ndarray, list]] = None,
        upper_bound: Optional[Union[pd.Series, np.ndarray, list]] = None,
    ) -> Dict[str, float]:
        """Compute standard metrics on a single array pair."""
        y_act = np.asarray(actual, dtype=float)
        y_prd = np.asarray(predicted, dtype=float)

        if len(y_act) == 0:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "mape": 0.0,
                "wape": 0.0,
                "bias": 0.0,
                "coverage": 0.0,
                "total_actual": 0.0,
                "total_forecast": 0.0,
                "count": 0,
            }

        m_val = round(mae(y_act, y_prd), 4)
        r_val = round(rmse(y_act, y_prd), 4)
        mp_val = round(mape(y_act, y_prd), 2)
        w_val = round(wape(y_act, y_prd), 2)
        b_val = round(bias(y_act, y_prd), 2)

        cov_val = 0.0
        if lower_bound is not None and upper_bound is not None:
            low = np.asarray(lower_bound, dtype=float)
            high = np.asarray(upper_bound, dtype=float)
            valid_mask = (~np.isnan(low)) & (~np.isnan(high))
            if np.any(valid_mask):
                inside = (y_act[valid_mask] >= low[valid_mask]) & (y_act[valid_mask] <= high[valid_mask])
                cov_val = round(float(np.mean(inside) * 100.0), 2)

        return {
            "mae": m_val,
            "rmse": r_val,
            "mape": mp_val,
            "wape": w_val,
            "bias": b_val,
            "coverage": cov_val,
            "total_actual": round(float(np.sum(y_act)), 2),
            "total_forecast": round(float(np.sum(y_prd)), 2),
            "count": int(len(y_act)),
        }

    def evaluate_predictions(
        self,
        predictions_df: pd.DataFrame,
        group_by: Optional[List[str]] = None,
        actual_col: str = "actual_units",
        pred_col: str = "forecast_units",
    ) -> pd.DataFrame:
        """Evaluate predictions DataFrame grouped by given dimensions.

        Args:
            predictions_df: DataFrame containing actual and forecast columns.
            group_by: List of grouping columns (e.g. ['model_name', 'sku_id']).
                      If None, groups by ['model_name'].
            actual_col: Name of column containing ground-truth actuals.
            pred_col: Name of column containing predictions.

        Returns:
            pd.DataFrame: Aggregated metrics per group.
        """
        if predictions_df.empty:
            return pd.DataFrame()

        groups = group_by or ["model_name"]
        rows = []

        grouped = predictions_df.groupby(groups)
        for name, group in grouped:
            if not isinstance(name, tuple):
                name = (name,)

            key_dict = {col: name[i] for i, col in enumerate(groups)}
            metrics = self.evaluate_slice(
                actual=group[actual_col],
                predicted=group[pred_col],
                lower_bound=group.get("lower_bound"),
                upper_bound=group.get("upper_bound"),
            )
            rows.append({**key_dict, **metrics})

        res_df = pd.DataFrame(rows)
        if "wape" in res_df.columns:
            res_df = res_df.sort_values("wape").reset_index(drop=True)
        return res_df

    def generate_benchmark_table(
        self,
        predictions_df: pd.DataFrame,
        diagnostics_df: Optional[pd.DataFrame] = None,
        actual_col: str = "actual_units",
        pred_col: str = "forecast_units",
    ) -> pd.DataFrame:
        """Generate standardized master benchmark table.

        Format:
        Model | MAE | RMSE | WAPE (%) | Bias (%) | Runtime (s) | Failures | Status
        """
        if predictions_df.empty and (diagnostics_df is None or diagnostics_df.empty):
            return pd.DataFrame(columns=[
                "Model", "MAE", "RMSE", "WAPE (%)", "Bias (%)", "Runtime (s)", "Failures", "Status"
            ])

        # Evaluate model-level predictions
        eval_df = self.evaluate_predictions(
            predictions_df,
            group_by=["model_name"],
            actual_col=actual_col,
            pred_col=pred_col,
        )

        # Compute runtime and failure counts per model from diagnostics
        runtime_map: Dict[str, float] = {}
        failure_map: Dict[str, int] = {}
        status_map: Dict[str, str] = {}

        if diagnostics_df is not None and not diagnostics_df.empty:
            for model_name, diag_group in diagnostics_df.groupby("model_name"):
                runtime_map[model_name] = round(float(diag_group["runtime_seconds"].sum()), 4)
                fails = int((diag_group["status"] != "SUCCESS").sum())
                failure_map[model_name] = fails
                
                # Check status
                statuses = diag_group["status"].unique()
                if "TIMESFM_UNAVAILABLE" in statuses:
                    status_map[model_name] = "UNAVAILABLE"
                elif fails > 0 and fails == len(diag_group):
                    status_map[model_name] = "FAILED"
                elif fails > 0:
                    status_map[model_name] = "PARTIAL_FAIL"
                else:
                    status_map[model_name] = "SUCCESS"

        # Combine
        records = []
        models_from_eval = list(eval_df["model_name"]) if not eval_df.empty else []
        models_from_diag = list(diagnostics_df["model_name"].unique()) if diagnostics_df is not None and not diagnostics_df.empty else []
        all_models = sorted(list(set(models_from_eval + models_from_diag)))

        for m_name in all_models:
            m_eval = eval_df[eval_df["model_name"] == m_name] if not eval_df.empty else pd.DataFrame()
            if not m_eval.empty:
                r = m_eval.iloc[0]
                records.append({
                    "Model": m_name,
                    "MAE": r["mae"],
                    "RMSE": r["rmse"],
                    "WAPE (%)": r["wape"],
                    "Bias (%)": r["bias"],
                    "Runtime (s)": runtime_map.get(m_name, 0.0),
                    "Failures": failure_map.get(m_name, 0),
                    "Status": status_map.get(m_name, "SUCCESS"),
                })
            else:
                # Model failed completely or was unavailable
                records.append({
                    "Model": m_name,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "WAPE (%)": np.nan,
                    "Bias (%)": np.nan,
                    "Runtime (s)": runtime_map.get(m_name, 0.0),
                    "Failures": failure_map.get(m_name, 1),
                    "Status": status_map.get(m_name, "UNAVAILABLE"),
                })

        benchmark_df = pd.DataFrame(records)
        if "WAPE (%)" in benchmark_df.columns:
            benchmark_df = benchmark_df.sort_values(
                by=["Status", "WAPE (%)"],
                ascending=[True, True],
                na_position="last",
            ).reset_index(drop=True)

        return benchmark_df

    def evaluate_hierarchical(
        self,
        predictions_df: pd.DataFrame,
        entity_metadata: Optional[pd.DataFrame] = None,
        actual_col: str = "actual_units",
        pred_col: str = "forecast_units",
    ) -> Dict[str, pd.DataFrame]:
        """Perform hierarchical evaluations across multiple dimensions."""
        if predictions_df.empty:
            return {}

        df = predictions_df.copy()
        if entity_metadata is not None and not entity_metadata.empty:
            merge_cols = [c for c in ["sku_id", "warehouse_id"] if c in entity_metadata.columns and c in df.columns]
            if merge_cols:
                extra_cols = [c for c in entity_metadata.columns if c not in df.columns]
                df = pd.merge(df, entity_metadata[merge_cols + extra_cols].drop_duplicates(), on=merge_cols, how="left")

        results = {
            "overall": self.evaluate_predictions(df, group_by=["model_name"], actual_col=actual_col, pred_col=pred_col),
        }

        if "fold_index" in df.columns:
            results["by_fold"] = self.evaluate_predictions(df, group_by=["model_name", "fold_index"], actual_col=actual_col, pred_col=pred_col)

        if "sku_id" in df.columns:
            results["by_sku"] = self.evaluate_predictions(df, group_by=["model_name", "sku_id"], actual_col=actual_col, pred_col=pred_col)

        if "warehouse_id" in df.columns:
            results["by_warehouse"] = self.evaluate_predictions(df, group_by=["model_name", "warehouse_id"], actual_col=actual_col, pred_col=pred_col)

        if "category" in df.columns:
            results["by_category"] = self.evaluate_predictions(df, group_by=["model_name", "category"], actual_col=actual_col, pred_col=pred_col)

        if "intermittency_class" in df.columns:
            results["by_intermittency"] = self.evaluate_predictions(df, group_by=["model_name", "intermittency_class"], actual_col=actual_col, pred_col=pred_col)

        if "abc_xyz_class" in df.columns:
            results["by_abc_xyz"] = self.evaluate_predictions(df, group_by=["model_name", "abc_xyz_class"], actual_col=actual_col, pred_col=pred_col)

        return results
