"""Return Prediction Dataset & Label Engineering Engine (Phase 5C-1).

Creates the leakage-safe, point-in-time supervised learning dataset for future return prediction:
- Evaluates at the order-line grain: (order_id, sku_id) / sale_id
- Constructs binary classification target: returned = 1 or 0 within return_window_days
- Handles partial returns and return quantity tracking
- Enforces strict chronological point-in-time feature extraction (no lookahead leakage)
- Generates order-line, product, calendar, and historical rate features
- Implements cold-start fallback hierarchy
- Provides chronological temporal splitting (train / val / test) without shuffling
- Audits dataset quality, missing references, and target imbalance
- Strictly isolates features X, target y, and metadata
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from commerce_ai.returns.schemas import (
    FeatureDefinition,
    ReturnPredictionConfig,
    ReturnPredictionQualityReport,
    ReturnPredictionRecord,
    TemporalSplitInfo,
)
from commerce_ai.returns.analytics import filter_by_as_of_date


# =====================================================================
# Forbidden Leakage Columns Definition
# =====================================================================

FORBIDDEN_LEAKAGE_COLUMNS: Set[str] = {
    "return_date",
    "returned_quantity",
    "return_quantity",
    "return_reason",
    "reason",
    "refund_amount",
    "refund",
    "target_returned",
    "target_return_quantity",
    "target_return_date",
    "return_id",
    "lag_days",
    "is_returned",
}


# =====================================================================
# Return Prediction Dataset Container
# =====================================================================


class ReturnPredictionDataset:
    """Container holding the supervised tabular return prediction dataset."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: pd.DataFrame,
        feature_definitions: Dict[str, FeatureDefinition],
        quality_report: ReturnPredictionQualityReport,
        temporal_split_info: TemporalSplitInfo,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
    ):
        self.X = X
        self.y = y
        self.metadata = metadata
        self.feature_definitions = feature_definitions
        self.quality_report = quality_report
        self.temporal_split_info = temporal_split_info
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        # Strict validation upon instantiation
        self.verify_no_leakage()

    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Retrieve training feature matrix X and target vector y."""
        if not self.train_indices or self.X.empty:
            return self.X.iloc[0:0].copy(), self.y.iloc[0:0].copy()
        return self.X.iloc[self.train_indices].copy(), self.y.iloc[self.train_indices].copy()

    def get_val_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Retrieve validation feature matrix X and target vector y."""
        if not self.val_indices or self.X.empty:
            return self.X.iloc[0:0].copy(), self.y.iloc[0:0].copy()
        return self.X.iloc[self.val_indices].copy(), self.y.iloc[self.val_indices].copy()

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Retrieve holdout test feature matrix X and target vector y."""
        if not self.test_indices or self.X.empty:
            return self.X.iloc[0:0].copy(), self.y.iloc[0:0].copy()
        return self.X.iloc[self.test_indices].copy(), self.y.iloc[self.test_indices].copy()

    def get_imbalance_stats(self) -> Dict[str, Any]:
        """Report class distribution and target imbalance metrics."""
        if self.y.empty:
            return {
                "positive_count": 0,
                "negative_count": 0,
                "positive_rate": 0.0,
                "total_count": 0,
            }
        pos = int((self.y == 1).sum())
        neg = int((self.y == 0).sum())
        tot = pos + neg
        rate = round(pos / tot, 4) if tot > 0 else 0.0
        return {
            "positive_count": pos,
            "negative_count": neg,
            "positive_rate": rate,
            "total_count": tot,
        }

    def verify_no_leakage(self) -> bool:
        """Verify that X contains no forbidden columns and splits are temporally valid."""
        # 1. Check forbidden column names in feature matrix
        x_cols_lower = {c.lower() for c in self.X.columns}
        intersection = x_cols_lower.intersection(FORBIDDEN_LEAKAGE_COLUMNS)
        if intersection:
            raise ValueError(f"Target leakage detected! Feature matrix X contains forbidden columns: {intersection}")

        # 2. Check disjoint split indices
        train_s = set(self.train_indices)
        val_s = set(self.val_indices)
        test_s = set(self.test_indices)

        if train_s & val_s:
            raise ValueError("Temporal split integrity violated: Train and Validation indices overlap!")
        if train_s & test_s:
            raise ValueError("Temporal split integrity violated: Train and Test indices overlap!")
        if val_s & test_s:
            raise ValueError("Temporal split integrity violated: Validation and Test indices overlap!")

        # 3. Check chronological ordering across splits if populated
        if not self.metadata.empty and "prediction_date" in self.metadata.columns:
            dates = pd.to_datetime(self.metadata["prediction_date"]).dt.date
            if self.train_indices and self.val_indices:
                max_train_date = dates.iloc[self.train_indices].max()
                min_val_date = dates.iloc[self.val_indices].min()
                if max_train_date > min_val_date:
                    raise ValueError(
                        f"Temporal sequence violated! Max train date ({max_train_date}) > Min val date ({min_val_date})"
                    )

            if self.val_indices and self.test_indices:
                max_val_date = dates.iloc[self.val_indices].max()
                min_test_date = dates.iloc[self.test_indices].min()
                if max_val_date > min_test_date:
                    raise ValueError(
                        f"Temporal sequence violated! Max val date ({max_val_date}) > Min test date ({min_test_date})"
                    )

        return True


# =====================================================================
# Dataset Builder Engine
# =====================================================================


class ReturnPredictionDatasetBuilder:
    """Builder class constructing the supervised tabular return prediction dataset."""

    def __init__(self, config: Optional[ReturnPredictionConfig] = None):
        self.config = config or ReturnPredictionConfig()

    def build(
        self,
        sales: pd.DataFrame,
        returns: pd.DataFrame,
        products: Optional[pd.DataFrame] = None,
        warehouses: Optional[pd.DataFrame] = None,
        channels: Optional[pd.DataFrame] = None,
        config: Optional[ReturnPredictionConfig] = None,
    ) -> ReturnPredictionDataset:
        """Construct the supervised prediction dataset.

        Args:
            sales: Transaction line items DataFrame.
            returns: Processed customer returns DataFrame.
            products: Optional product master DataFrame.
            warehouses: Optional warehouse master DataFrame.
            channels: Optional channel master DataFrame.
            config: Optional ReturnPredictionConfig overrides.

        Returns:
            ReturnPredictionDataset with separated X, y, metadata, and temporal splits.
        """
        cfg = config or self.config
        report = ReturnPredictionQualityReport()

        if sales is None or sales.empty:
            return self._create_empty_dataset(report)

        report.total_input_sales = len(sales)

        # 1. Apply anti-leakage date cutoff
        sales_df = filter_by_as_of_date(sales, "date", cfg.as_of_date)
        returns_df = filter_by_as_of_date(returns, "return_date", cfg.as_of_date) if returns is not None else pd.DataFrame()

        # 2. Audit sales data quality
        valid_sales, report = self._audit_sales_data(
            sales_df=sales_df,
            products=products,
            warehouses=warehouses,
            channels=channels,
            report=report,
        )

        if valid_sales.empty:
            return self._create_empty_dataset(report)

        # 3. Construct target labels
        labeled_sales, report = self._engineer_target_labels(
            sales_df=valid_sales,
            returns_df=returns_df,
            cfg=cfg,
            report=report,
        )

        if labeled_sales.empty:
            return self._create_empty_dataset(report)

        # 4. Chronological sorting
        labeled_sales = labeled_sales.sort_values(by=["date", "sale_id"]).reset_index(drop=True)
        report.total_prediction_rows = len(labeled_sales)

        # 5. Point-in-Time Historical Return Features
        hist_features = self._compute_point_in_time_history(
            sales_df=valid_sales,
            returns_df=returns_df,
            products_df=products,
            eval_sales=labeled_sales,
            cfg=cfg,
        )

        # 6. Feature Matrix X & Metadata construction
        X, metadata, feature_defs = self._assemble_features_and_metadata(
            sales_df=labeled_sales,
            hist_features=hist_features,
            products_df=products,
        )

        y = labeled_sales["target_returned"].astype(int)

        # 7. Chronological Temporal Split
        train_idx, val_idx, test_idx, split_info = self._compute_temporal_split(
            labeled_sales=labeled_sales,
            cfg=cfg,
        )

        return ReturnPredictionDataset(
            X=X,
            y=y,
            metadata=metadata,
            feature_definitions=feature_defs,
            quality_report=report,
            temporal_split_info=split_info,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
        )

    def _audit_sales_data(
        self,
        sales_df: pd.DataFrame,
        products: Optional[pd.DataFrame],
        warehouses: Optional[pd.DataFrame],
        channels: Optional[pd.DataFrame],
        report: ReturnPredictionQualityReport,
    ) -> Tuple[pd.DataFrame, ReturnPredictionQualityReport]:
        """Validate sales rows for primary keys, foreign keys, and positive quantities."""
        df = sales_df.copy()

        # Parse date
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        missing_date = df["date_parsed"].isna()
        report.missing_sale_date_count = int(missing_date.sum())
        if report.missing_sale_date_count > 0:
            report.issues.append({"field": "date", "issue": "Missing or unparseable sales date", "count": report.missing_sale_date_count})

        # Missing order_id
        missing_order = df["order_id"].isna() | (df["order_id"].astype(str).str.strip() == "")
        report.missing_order_id_count = int(missing_order.sum())
        if report.missing_order_id_count > 0:
            report.issues.append({"field": "order_id", "issue": "Missing order_id", "count": report.missing_order_id_count})

        # Missing sku_id
        missing_sku = df["sku_id"].isna() | (df["sku_id"].astype(str).str.strip() == "")
        report.missing_sku_count = int(missing_sku.sum())
        if report.missing_sku_count > 0:
            report.issues.append({"field": "sku_id", "issue": "Missing sku_id", "count": report.missing_sku_count})

        # Invalid quantity (<= 0 or NaN)
        qty_num = pd.to_numeric(df["quantity"], errors="coerce")
        invalid_qty = qty_num.isna() | (qty_num <= 0)
        report.invalid_quantity_count = int(invalid_qty.sum())
        if report.invalid_quantity_count > 0:
            report.issues.append({"field": "quantity", "issue": "Non-positive or null quantity", "count": report.invalid_quantity_count})

        # Invalid price (< 0 or NaN)
        price_num = pd.to_numeric(df["unit_price"], errors="coerce")
        invalid_price = price_num.isna() | (price_num < 0)
        report.invalid_price_count = int(invalid_price.sum())
        if report.invalid_price_count > 0:
            report.issues.append({"field": "unit_price", "issue": "Negative or null unit price", "count": report.invalid_price_count})

        # Duplicate prediction rows
        if "sale_id" in df.columns:
            dup_sales = int(df["sale_id"].dropna().duplicated().sum())
            report.duplicate_prediction_rows = dup_sales
            if dup_sales > 0:
                report.issues.append({"field": "sale_id", "issue": "Duplicate sale_id records", "count": dup_sales})

        # Referential checks
        if products is not None and not products.empty and "sku_id" in products.columns:
            valid_skus = set(products["sku_id"].dropna().astype(str))
            unknown_skus = int((~df["sku_id"].astype(str).isin(valid_skus)).sum())
            report.unknown_sku_count = unknown_skus
            if unknown_skus > 0:
                report.issues.append({"field": "sku_id", "issue": "Unknown sku_id not in products master", "count": unknown_skus})

        if warehouses is not None and not warehouses.empty and "warehouse_id" in warehouses.columns:
            valid_whs = set(warehouses["warehouse_id"].dropna().astype(str))
            unknown_whs = int((~df["warehouse_id"].astype(str).isin(valid_whs)).sum())
            report.unknown_warehouse_count = unknown_whs
            if unknown_whs > 0:
                report.issues.append({"field": "warehouse_id", "issue": "Unknown warehouse_id not in warehouses master", "count": unknown_whs})

        if channels is not None and not channels.empty and "channel_id" in channels.columns:
            valid_chs = set(channels["channel_id"].dropna().astype(str))
            unknown_chs = int((~df["channel_id"].astype(str).isin(valid_chs)).sum())
            report.unknown_channel_count = unknown_chs
            if unknown_chs > 0:
                report.issues.append({"field": "channel_id", "issue": "Unknown channel_id not in channels master", "count": unknown_chs})

        # Keep valid rows
        valid_mask = (~missing_date) & (~missing_order) & (~missing_sku) & (~invalid_qty) & (~invalid_price)
        report.is_clean = len(report.issues) == 0

        clean_df = df[valid_mask].copy()
        clean_df["date"] = clean_df["date_parsed"]
        clean_df.drop(columns=["date_parsed"], inplace=True)
        return clean_df, report

    def _engineer_target_labels(
        self,
        sales_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        cfg: ReturnPredictionConfig,
        report: ReturnPredictionQualityReport,
    ) -> Tuple[pd.DataFrame, ReturnPredictionQualityReport]:
        """Map returns to order lines and construct target_returned and target_return_quantity."""
        df = sales_df.copy()

        # If returns is empty, all targets are 0
        if returns_df is None or returns_df.empty:
            df["target_returned"] = 0
            df["target_return_quantity"] = 0
            df["target_return_date"] = None
            return df, report

        ret = returns_df.copy()
        ret["return_date_parsed"] = pd.to_datetime(ret["return_date"], errors="coerce").dt.date
        valid_ret = ret[ret["return_date_parsed"].notna()].copy()

        # Check for unmapped returns
        valid_sale_orders = set(df["order_id"].dropna().astype(str))
        unmapped_returns = int((~valid_ret["order_id"].astype(str).isin(valid_sale_orders)).sum())
        report.unmapped_returns_count = unmapped_returns
        if unmapped_returns > 0:
            report.issues.append({"field": "returns.order_id", "issue": "Returns with no matching sales order", "count": unmapped_returns})

        # Aggregate returns by (order_id, sku_id)
        ret_agg = valid_ret.groupby(["order_id", "sku_id"]).agg(
            ret_qty=("quantity", "sum"),
            earliest_return_date=("return_date_parsed", "min"),
        ).reset_index()

        # Merge returns onto sales
        merged = pd.merge(df, ret_agg, on=["order_id", "sku_id"], how="left")

        # Compute return window lag
        merged["sale_date"] = pd.to_datetime(merged["date"]).dt.date
        merged["lag_days"] = np.where(
            merged["earliest_return_date"].notna(),
            (pd.to_datetime(merged["earliest_return_date"]) - pd.to_datetime(merged["sale_date"])).dt.days,
            np.nan,
        )

        # Flag impossible return lag (< 0)
        impossible_lag = (merged["lag_days"] < 0).sum()
        if impossible_lag > 0:
            report.leakage_violations.append(f"{impossible_lag} returns occurred prior to order date!")

        # Flag return quantity > sold quantity
        impossible_ret_qty = (merged["ret_qty"] > merged["quantity"]).sum()
        report.impossible_return_quantity_count = int(impossible_ret_qty)
        if impossible_ret_qty > 0:
            report.issues.append({"field": "quantity", "issue": "Return quantity exceeds sold quantity", "count": int(impossible_ret_qty)})

        # Primary binary label: returned = 1 if lag_days >= 0 and <= return_window_days
        in_window = (merged["lag_days"] >= 0) & (merged["lag_days"] <= cfg.return_window_days) & (merged["ret_qty"] > 0)
        merged["target_returned"] = np.where(in_window, 1, 0)
        merged["target_return_quantity"] = np.where(in_window, np.minimum(merged["ret_qty"].fillna(0), merged["quantity"]).astype(int), 0)
        merged["target_return_date"] = np.where(in_window, merged["earliest_return_date"].astype(str), None)

        # Policy on immature orders near the end of the observation period
        if cfg.prediction_cutoff_policy == "exclude_immature":
            max_date = merged["sale_date"].max()
            min_date = merged["sale_date"].min()
            # Only apply if dataset span exceeds return_window_days
            if (max_date - min_date) >= timedelta(days=cfg.return_window_days):
                immature_mask = (merged["target_returned"] == 0) & ((max_date - merged["sale_date"]) < timedelta(days=cfg.return_window_days))
                immature_count = int(immature_mask.sum())
                if immature_count > 0:
                    report.immature_orders_excluded = immature_count
                    merged = merged[~immature_mask].copy()

        # Clean up temporary columns
        merged.drop(columns=["ret_qty", "earliest_return_date", "sale_date", "lag_days"], inplace=True, errors="ignore")
        return merged, report

    def _compute_point_in_time_history(
        self,
        sales_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame],
        eval_sales: pd.DataFrame,
        cfg: ReturnPredictionConfig,
    ) -> pd.DataFrame:
        """Compute point-in-time historical return rates strictly prior to each order date."""
        # Convert dates to standard date objects
        s_df = sales_df.copy()
        s_df["date_obj"] = pd.to_datetime(s_df["date"]).dt.date

        r_df = (
            returns_df.copy()
            if returns_df is not None and not returns_df.empty
            else pd.DataFrame(columns=["return_date", "sku_id", "channel_id", "warehouse_id", "quantity"])
        )
        if not r_df.empty:
            r_df["date_obj"] = pd.to_datetime(r_df["return_date"]).dt.date
        else:
            r_df["date_obj"] = pd.Series(dtype=object)

        for col_name in ["channel_id", "warehouse_id", "sku_id", "quantity"]:
            if col_name not in r_df.columns:
                r_df[col_name] = "UNKNOWN" if col_name != "quantity" else 0
            if col_name not in s_df.columns:
                s_df[col_name] = "UNKNOWN" if col_name != "quantity" else 0

        # Map categories if products available
        cat_map: Dict[str, str] = {}
        if products_df is not None and not products_df.empty and "sku_id" in products_df.columns and "category_id" in products_df.columns:
            cat_map = dict(zip(products_df["sku_id"].astype(str), products_df["category_id"].astype(str)))

        s_df["category_id"] = s_df["sku_id"].astype(str).map(cat_map).fillna("UNKNOWN")
        if not r_df.empty:
            r_df["category_id"] = r_df["sku_id"].astype(str).map(cat_map).fillna("UNKNOWN")

        eval_dates_parsed = [d if isinstance(d, date) else pd.to_datetime(d).date() for d in eval_sales["date"].unique()]
        eval_dates = sorted(eval_dates_parsed)
        if not eval_dates:
            return pd.DataFrame(index=eval_sales.index)

        # All unique dates across sales, returns, and eval
        all_dates = sorted(list(set(s_df["date_obj"]).union(set(r_df["date_obj"] if not r_df.empty else [])).union(set(eval_dates))))
        date_index = pd.Index(all_dates, name="date_obj")

        # Helper to compute cumulative sum shifted by 1 date
        def build_cum_shifted(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
            if df.empty or key_col not in df.columns:
                return pd.DataFrame(0, index=date_index, columns=[])
            daily = df.groupby(["date_obj", key_col])["quantity"].sum().unstack(fill_value=0)
            daily = daily.reindex(index=date_index, fill_value=0)
            cum = daily.cumsum().shift(1, fill_value=0)
            return cum

        # 1. Global
        s_glob = s_df.groupby("date_obj")["quantity"].sum().reindex(date_index, fill_value=0).cumsum().shift(1, fill_value=0)
        r_glob = (
            r_df.groupby("date_obj")["quantity"].sum().reindex(date_index, fill_value=0).cumsum().shift(1, fill_value=0)
            if not r_df.empty
            else pd.Series(0, index=date_index)
        )
        global_rate = np.divide(
            r_glob.values,
            s_glob.values,
            out=np.zeros(len(date_index), dtype=float),
            where=(s_glob.values >= 10),
        )
        glob_dict = dict(zip(date_index, global_rate))

        # 2. Category
        cum_s_cat = build_cum_shifted(s_df, "category_id")
        cum_r_cat = build_cum_shifted(r_df, "category_id")
        all_cats = list(set(cum_s_cat.columns).union(set(cum_r_cat.columns)))
        cum_s_cat = cum_s_cat.reindex(columns=all_cats, fill_value=0)
        cum_r_cat = cum_r_cat.reindex(columns=all_cats, fill_value=0)

        # 3. SKU
        cum_s_sku = build_cum_shifted(s_df, "sku_id")
        cum_r_sku = build_cum_shifted(r_df, "sku_id")
        all_skus = list(set(cum_s_sku.columns).union(set(cum_r_sku.columns)))
        cum_s_sku = cum_s_sku.reindex(columns=all_skus, fill_value=0)
        cum_r_sku = cum_r_sku.reindex(columns=all_skus, fill_value=0)

        # 4. Channel
        cum_s_ch = build_cum_shifted(s_df, "channel_id")
        cum_r_ch = build_cum_shifted(r_df, "channel_id")
        all_chs = list(set(cum_s_ch.columns).union(set(cum_r_ch.columns)))
        cum_s_ch = cum_s_ch.reindex(columns=all_chs, fill_value=0)
        cum_r_ch = cum_r_ch.reindex(columns=all_chs, fill_value=0)

        # 5. Warehouse
        cum_s_wh = build_cum_shifted(s_df, "warehouse_id")
        cum_r_wh = build_cum_shifted(r_df, "warehouse_id")
        all_whs = list(set(cum_s_wh.columns).union(set(cum_r_wh.columns)))
        cum_s_wh = cum_s_wh.reindex(columns=all_whs, fill_value=0)
        cum_r_wh = cum_r_wh.reindex(columns=all_whs, fill_value=0)

        # 6. SKU x Channel
        s_df_sc = s_df.copy()
        s_df_sc["sc_key"] = s_df_sc["sku_id"].astype(str) + ":" + s_df_sc["channel_id"].astype(str)
        r_df_sc = r_df.copy()
        if not r_df_sc.empty:
            r_df_sc["sc_key"] = r_df_sc["sku_id"].astype(str) + ":" + r_df_sc["channel_id"].astype(str)
        cum_s_sc = build_cum_shifted(s_df_sc, "sc_key")
        cum_r_sc = build_cum_shifted(r_df_sc, "sc_key")
        all_sc = list(set(cum_s_sc.columns).union(set(cum_r_sc.columns)))
        cum_s_sc = cum_s_sc.reindex(columns=all_sc, fill_value=0)
        cum_r_sc = cum_r_sc.reindex(columns=all_sc, fill_value=0)

        # 7. SKU x Warehouse
        s_df_sw = s_df.copy()
        s_df_sw["sw_key"] = s_df_sw["sku_id"].astype(str) + ":" + s_df_sw["warehouse_id"].astype(str)
        r_df_sw = r_df.copy()
        if not r_df_sw.empty:
            r_df_sw["sw_key"] = r_df_sw["sku_id"].astype(str) + ":" + r_df_sw["warehouse_id"].astype(str)
        cum_s_sw = build_cum_shifted(s_df_sw, "sw_key")
        cum_r_sw = build_cum_shifted(r_df_sw, "sw_key")
        all_sw = list(set(cum_s_sw.columns).union(set(cum_r_sw.columns)))
        cum_s_sw = cum_s_sw.reindex(columns=all_sw, fill_value=0)
        cum_r_sw = cum_r_sw.reindex(columns=all_sw, fill_value=0)

        # Export to Dict for O(1) lookup
        dict_s_cat = cum_s_cat.stack().to_dict()
        dict_r_cat = cum_r_cat.stack().to_dict()
        dict_s_sku = cum_s_sku.stack().to_dict()
        dict_r_sku = cum_r_sku.stack().to_dict()
        dict_s_ch = cum_s_ch.stack().to_dict()
        dict_r_ch = cum_r_ch.stack().to_dict()
        dict_s_wh = cum_s_wh.stack().to_dict()
        dict_r_wh = cum_r_wh.stack().to_dict()
        dict_s_sc = cum_s_sc.stack().to_dict()
        dict_r_sc = cum_r_sc.stack().to_dict()
        dict_s_sw = cum_s_sw.stack().to_dict()
        dict_r_sw = cum_r_sw.stack().to_dict()

        # Prepare eval keys
        eval_df = eval_sales.copy()
        e_date = [d if isinstance(d, date) else pd.to_datetime(d).date() for d in eval_df["date"]]
        e_sku = eval_df["sku_id"].astype(str).tolist()
        e_cat = [cat_map.get(s, "UNKNOWN") for s in e_sku]
        e_ch = eval_df["channel_id"].astype(str).tolist()
        e_wh = eval_df["warehouse_id"].astype(str).tolist()
        e_sc = [f"{s}:{c}" for s, c in zip(e_sku, e_ch)]
        e_sw = [f"{s}:{w}" for s, w in zip(e_sku, e_wh)]

        # Vectorized lookups and calculations
        g_rate = np.array([glob_dict.get(d, 0.0) for d in e_date], dtype=float)

        cat_s = np.array([dict_s_cat.get((d, k), 0) for d, k in zip(e_date, e_cat)], dtype=int)
        cat_r = np.array([dict_r_cat.get((d, k), 0) for d, k in zip(e_date, e_cat)], dtype=int)
        cat_rate = np.where(cat_s >= 10, np.divide(cat_r, cat_s, out=np.zeros_like(cat_r, dtype=float), where=(cat_s >= 10)), g_rate)

        sku_s = np.array([dict_s_sku.get((d, k), 0) for d, k in zip(e_date, e_sku)], dtype=int)
        sku_r = np.array([dict_r_sku.get((d, k), 0) for d, k in zip(e_date, e_sku)], dtype=int)
        sku_rate_raw = np.divide(sku_r, sku_s, out=np.zeros_like(sku_r, dtype=float), where=(sku_s >= 10))
        sku_rate_resolved = np.where(sku_s >= 10, sku_rate_raw, cat_rate)
        is_cold_sku = np.where(sku_s < 10, 1, 0)

        ch_s = np.array([dict_s_ch.get((d, k), 0) for d, k in zip(e_date, e_ch)], dtype=int)
        ch_r = np.array([dict_r_ch.get((d, k), 0) for d, k in zip(e_date, e_ch)], dtype=int)
        ch_rate = np.where(ch_s >= 10, np.divide(ch_r, ch_s, out=np.zeros_like(ch_r, dtype=float), where=(ch_s >= 10)), g_rate)
        is_cold_ch = np.where(ch_s < 10, 1, 0)

        wh_s = np.array([dict_s_wh.get((d, k), 0) for d, k in zip(e_date, e_wh)], dtype=int)
        wh_r = np.array([dict_r_wh.get((d, k), 0) for d, k in zip(e_date, e_wh)], dtype=int)
        wh_rate = np.where(wh_s >= 10, np.divide(wh_r, wh_s, out=np.zeros_like(wh_r, dtype=float), where=(wh_s >= 10)), g_rate)

        sc_s = np.array([dict_s_sc.get((d, k), 0) for d, k in zip(e_date, e_sc)], dtype=int)
        sc_r = np.array([dict_r_sc.get((d, k), 0) for d, k in zip(e_date, e_sc)], dtype=int)
        sc_rate_raw = np.divide(sc_r, sc_s, out=np.zeros_like(sc_r, dtype=float), where=(sc_s >= 10))
        sc_rate_resolved = np.where(sc_s >= 10, sc_rate_raw, sku_rate_resolved)
        is_cold_sc = np.where(sc_s < 10, 1, 0)

        sw_s = np.array([dict_s_sw.get((d, k), 0) for d, k in zip(e_date, e_sw)], dtype=int)
        sw_r = np.array([dict_r_sw.get((d, k), 0) for d, k in zip(e_date, e_sw)], dtype=int)
        sw_rate_raw = np.divide(sw_r, sw_s, out=np.zeros_like(sw_r, dtype=float), where=(sw_s >= 10))
        sw_rate_resolved = np.where(sw_s >= 10, sw_rate_raw, sku_rate_resolved)

        hist_df = pd.DataFrame(index=eval_sales.index)
        hist_df["hist_sku_return_rate"] = np.round(sku_rate_resolved, 4)
        hist_df["hist_sku_sold_units"] = sku_s
        hist_df["hist_sku_returned_units"] = sku_r
        hist_df["hist_channel_return_rate"] = np.round(ch_rate, 4)
        hist_df["hist_warehouse_return_rate"] = np.round(wh_rate, 4)
        hist_df["hist_sku_channel_return_rate"] = np.round(sc_rate_resolved, 4)
        hist_df["hist_sku_warehouse_return_rate"] = np.round(sw_rate_resolved, 4)
        hist_df["is_cold_start_sku"] = is_cold_sku
        hist_df["is_cold_start_channel"] = is_cold_ch
        hist_df["is_cold_start_combination"] = is_cold_sc

        return hist_df

    def _assemble_features_and_metadata(
        self,
        sales_df: pd.DataFrame,
        hist_features: pd.DataFrame,
        products_df: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, FeatureDefinition]]:
        """Combine order-line, product, calendar, and historical rate features into X."""
        df = sales_df.copy()

        # Product lookup maps
        prod_map: Dict[str, Dict[str, Any]] = {}
        if products_df is not None and not products_df.empty and "sku_id" in products_df.columns:
            for _, p_row in products_df.iterrows():
                sku = str(p_row["sku_id"])
                prod_map[sku] = {
                    "category_id": str(p_row.get("category_id", "UNKNOWN")),
                    "brand": str(p_row.get("brand", "UNKNOWN")),
                    "unit_cost": float(p_row.get("unit_cost", 0.0)),
                    "selling_price": float(p_row.get("selling_price", 0.0)),
                    "velocity_tier": str(p_row.get("velocity_tier", "UNKNOWN")),
                }

        # Vectorized Calendar features
        dates = pd.to_datetime(df["date"])
        day_of_week = dates.dt.dayofweek.values
        day_of_month = dates.dt.day.values
        month = dates.dt.month.values
        quarter = dates.dt.quarter.values
        is_weekend = np.isin(day_of_week, [5, 6]).astype(int)

        qty = df["quantity"].astype(int).values
        unit_p = df["unit_price"].astype(float).values
        discount = df["discount"].fillna(0.0).astype(float).values if "discount" in df.columns else np.zeros(len(df))
        gross = qty * unit_p
        discount_rate = np.where(gross > 0, discount / gross, 0.0)
        if "revenue" in df.columns:
            rev_numeric = pd.to_numeric(df["revenue"], errors="coerce")
            fallback_rev = pd.Series(gross - discount, index=df.index)
            revenue = rev_numeric.fillna(fallback_rev).astype(float).values
        else:
            revenue = gross - discount
        revenue = np.maximum(0.0, revenue)

        # Product features lookup
        skus_str = df["sku_id"].astype(str).tolist()
        unit_p_list = df["unit_price"].astype(float).tolist()
        cat_ids = [prod_map.get(s, {}).get("category_id", "UNKNOWN") for s in skus_str]
        brands = [prod_map.get(s, {}).get("brand", "UNKNOWN") for s in skus_str]
        unit_costs = [round(prod_map.get(s, {}).get("unit_cost", 0.0), 2) for s in skus_str]
        std_selling_prices = [round(prod_map.get(s, {}).get("selling_price", p), 2) for s, p in zip(skus_str, unit_p_list)]
        margins = [
            round((sp - uc) / sp, 4) if sp > 0 else 0.0
            for sp, uc in zip(std_selling_prices, unit_costs)
        ]
        velocity_tiers = [prod_map.get(s, {}).get("velocity_tier", "UNKNOWN") for s in skus_str]

        # Assemble Feature Matrix X
        X = pd.DataFrame(index=df.index)
        X["quantity"] = qty
        X["unit_price"] = np.round(unit_p, 2)
        X["discount"] = np.round(discount, 2)
        X["discount_rate"] = np.round(discount_rate, 4)
        X["revenue"] = np.round(revenue, 2)
        X["channel_id"] = df["channel_id"].astype(str).values
        X["warehouse_id"] = df["warehouse_id"].astype(str).values

        X["category_id"] = cat_ids
        X["brand"] = brands
        X["unit_cost"] = unit_costs
        X["selling_price"] = std_selling_prices
        X["gross_margin_rate"] = margins
        X["velocity_tier"] = velocity_tiers

        X["day_of_week"] = day_of_week
        X["day_of_month"] = day_of_month
        X["month"] = month
        X["quarter"] = quarter
        X["is_weekend"] = is_weekend

        # Merge historical point-in-time features
        for col in [
            "hist_sku_return_rate",
            "hist_sku_sold_units",
            "hist_sku_returned_units",
            "hist_channel_return_rate",
            "hist_warehouse_return_rate",
            "hist_sku_channel_return_rate",
            "hist_sku_warehouse_return_rate",
            "is_cold_start_sku",
            "is_cold_start_channel",
            "is_cold_start_combination",
        ]:
            if col in hist_features.columns:
                X[col] = hist_features[col].values
            else:
                X[col] = 0.0 if "rate" in col else 0

        # Assemble Metadata
        meta = pd.DataFrame(index=df.index)
        meta["prediction_id"] = ["PRED-" + str(s) for s in df["sale_id"]]
        meta["sale_id"] = df["sale_id"].astype(str).values
        meta["order_id"] = df["order_id"].astype(str).values
        meta["sku_id"] = df["sku_id"].astype(str).values
        meta["warehouse_id"] = df["warehouse_id"].astype(str).values
        meta["channel_id"] = df["channel_id"].astype(str).values
        meta["prediction_date"] = [str(d) for d in df["date"]]
        meta["target_returned"] = df["target_returned"].astype(int).values
        meta["target_return_quantity"] = df["target_return_quantity"].astype(int).values
        meta["target_return_date"] = df["target_return_date"].values

        # Build feature definitions
        feature_defs = self._create_feature_definitions()

        return X, meta, feature_defs

    def _compute_temporal_split(
        self,
        labeled_sales: pd.DataFrame,
        cfg: ReturnPredictionConfig,
    ) -> Tuple[List[int], List[int], List[int], TemporalSplitInfo]:
        """Compute chronological, non-overlapping train/validation/test index splits."""
        n = len(labeled_sales)
        if n == 0:
            return [], [], [], TemporalSplitInfo()

        n_train = int(n * cfg.temporal_train_ratio)
        n_val = int(n * cfg.temporal_val_ratio)

        train_indices = list(range(0, n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n))

        dates = [str(d) for d in labeled_sales["date"]]
        split_info = TemporalSplitInfo(
            train_start_date=dates[train_indices[0]] if train_indices else None,
            train_end_date=dates[train_indices[-1]] if train_indices else None,
            train_row_count=len(train_indices),
            val_start_date=dates[val_indices[0]] if val_indices else None,
            val_end_date=dates[val_indices[-1]] if val_indices else None,
            val_row_count=len(val_indices),
            test_start_date=dates[test_indices[0]] if test_indices else None,
            test_end_date=dates[test_indices[-1]] if test_indices else None,
            test_row_count=len(test_indices),
        )

        return train_indices, val_indices, test_indices, split_info

    def _create_empty_dataset(self, report: ReturnPredictionQualityReport) -> ReturnPredictionDataset:
        """Create empty container when no valid records exist."""
        return ReturnPredictionDataset(
            X=pd.DataFrame(),
            y=pd.Series(dtype=int),
            metadata=pd.DataFrame(),
            feature_definitions=self._create_feature_definitions(),
            quality_report=report,
            temporal_split_info=TemporalSplitInfo(),
            train_indices=[],
            val_indices=[],
            test_indices=[],
        )

    def _create_feature_definitions(self) -> Dict[str, FeatureDefinition]:
        """Generate metadata documentation for all engineered features."""
        defs = {
            "quantity": FeatureDefinition(
                name="quantity",
                data_type="int",
                source="sales",
                calculation="Sold order-line quantity",
                prediction_time_availability="Known at checkout / order placement",
                leakage_risk_assessment="None; pure transaction attribute",
            ),
            "unit_price": FeatureDefinition(
                name="unit_price",
                data_type="float",
                source="sales",
                calculation="Unit sale price in currency",
                prediction_time_availability="Known at checkout",
                leakage_risk_assessment="None; pricing is fixed at purchase",
            ),
            "discount": FeatureDefinition(
                name="discount",
                data_type="float",
                source="sales",
                calculation="Promotional discount amount applied to line",
                prediction_time_availability="Known at checkout",
                leakage_risk_assessment="None; fixed at purchase",
            ),
            "discount_rate": FeatureDefinition(
                name="discount_rate",
                data_type="float",
                source="sales",
                calculation="discount / (quantity * unit_price)",
                prediction_time_availability="Known at checkout",
                leakage_risk_assessment="None; derived from order line",
            ),
            "revenue": FeatureDefinition(
                name="revenue",
                data_type="float",
                source="sales",
                calculation="Realized net revenue (quantity * unit_price - discount)",
                prediction_time_availability="Known at checkout",
                leakage_risk_assessment="None; realized at transaction time",
            ),
            "channel_id": FeatureDefinition(
                name="channel_id",
                data_type="category",
                source="sales",
                calculation="Sales channel identifier (Web, App, Marketplace)",
                prediction_time_availability="Known at checkout",
                leakage_risk_assessment="None; transaction channel",
            ),
            "warehouse_id": FeatureDefinition(
                name="warehouse_id",
                data_type="category",
                source="sales",
                calculation="Fulfillment warehouse identifier",
                prediction_time_availability="Assigned at order routing / fulfillment",
                leakage_risk_assessment="None; operational facility assignment",
            ),
            "category_id": FeatureDefinition(
                name="category_id",
                data_type="category",
                source="products",
                calculation="Product merchandise category",
                prediction_time_availability="Catalog master static field",
                leakage_risk_assessment="None; static product metadata",
            ),
            "brand": FeatureDefinition(
                name="brand",
                data_type="category",
                source="products",
                calculation="Commercial product brand",
                prediction_time_availability="Catalog master static field",
                leakage_risk_assessment="None; static product metadata",
            ),
            "unit_cost": FeatureDefinition(
                name="unit_cost",
                data_type="float",
                source="products",
                calculation="Standard unit procurement/manufacturing cost",
                prediction_time_availability="Catalog master static field",
                leakage_risk_assessment="None; known before sale",
            ),
            "selling_price": FeatureDefinition(
                name="selling_price",
                data_type="float",
                source="products",
                calculation="Standard retail catalog price",
                prediction_time_availability="Catalog master static field",
                leakage_risk_assessment="None; catalog price",
            ),
            "gross_margin_rate": FeatureDefinition(
                name="gross_margin_rate",
                data_type="float",
                source="products",
                calculation="(selling_price - unit_cost) / selling_price",
                prediction_time_availability="Known at checkout",
                leakage_risk_assessment="None; pre-order economic ratio",
            ),
            "velocity_tier": FeatureDefinition(
                name="velocity_tier",
                data_type="category",
                source="products",
                calculation="Catalog velocity classification tier (HIGH, MEDIUM, SLOW)",
                prediction_time_availability="Catalog master static field",
                leakage_risk_assessment="None; static product profile",
            ),
            "day_of_week": FeatureDefinition(
                name="day_of_week",
                data_type="int",
                source="calendar",
                calculation="Day of week (0=Monday, 6=Sunday)",
                prediction_time_availability="Known at order date",
                leakage_risk_assessment="None; deterministic calendar attribute",
            ),
            "day_of_month": FeatureDefinition(
                name="day_of_month",
                data_type="int",
                source="calendar",
                calculation="Day of month (1-31)",
                prediction_time_availability="Known at order date",
                leakage_risk_assessment="None; calendar attribute",
            ),
            "month": FeatureDefinition(
                name="month",
                data_type="int",
                source="calendar",
                calculation="Month of year (1-12)",
                prediction_time_availability="Known at order date",
                leakage_risk_assessment="None; calendar attribute",
            ),
            "quarter": FeatureDefinition(
                name="quarter",
                data_type="int",
                source="calendar",
                calculation="Calendar quarter (1-4)",
                prediction_time_availability="Known at order date",
                leakage_risk_assessment="None; calendar attribute",
            ),
            "is_weekend": FeatureDefinition(
                name="is_weekend",
                data_type="int",
                source="calendar",
                calculation="Binary indicator: 1 if Saturday or Sunday, else 0",
                prediction_time_availability="Known at order date",
                leakage_risk_assessment="None; calendar attribute",
            ),
            "hist_sku_return_rate": FeatureDefinition(
                name="hist_sku_return_rate",
                data_type="float",
                source="historical sales + returns",
                calculation="Cumulative returned units / sold units for SKU strictly prior to order date",
                prediction_time_availability="Known prior to order date",
                leakage_risk_assessment="Protected by point-in-time chronological date filtering (< date)",
            ),
            "hist_sku_sold_units": FeatureDefinition(
                name="hist_sku_sold_units",
                data_type="int",
                source="historical sales",
                calculation="Cumulative units sold for SKU strictly prior to order date",
                prediction_time_availability="Known prior to order date",
                leakage_risk_assessment="Protected by point-in-time date filtering (< date)",
            ),
            "hist_sku_returned_units": FeatureDefinition(
                name="hist_sku_returned_units",
                data_type="int",
                source="historical returns",
                calculation="Cumulative units returned for SKU strictly prior to order date",
                prediction_time_availability="Known prior to order date",
                leakage_risk_assessment="Protected by point-in-time date filtering (< date)",
            ),
            "hist_channel_return_rate": FeatureDefinition(
                name="hist_channel_return_rate",
                data_type="float",
                source="historical sales + returns",
                calculation="Cumulative return rate for channel strictly prior to order date",
                prediction_time_availability="Known prior to order date",
                leakage_risk_assessment="Protected by point-in-time date filtering (< date)",
            ),
            "hist_warehouse_return_rate": FeatureDefinition(
                name="hist_warehouse_return_rate",
                data_type="float",
                source="historical sales + returns",
                calculation="Cumulative return rate for warehouse facility strictly prior to order date",
                prediction_time_availability="Known prior to order date",
                leakage_risk_assessment="Protected by point-in-time date filtering (< date)",
            ),
            "hist_sku_channel_return_rate": FeatureDefinition(
                name="hist_sku_channel_return_rate",
                data_type="float",
                source="historical sales + returns",
                calculation="Cumulative return rate for (SKU, Channel) pair strictly prior to order date",
                prediction_time_availability="Known prior to order date",
                leakage_risk_assessment="Protected by point-in-time date filtering (< date) and fallback hierarchy",
            ),
            "hist_sku_warehouse_return_rate": FeatureDefinition(
                name="hist_sku_warehouse_return_rate",
                data_type="float",
                source="historical sales + returns",
                calculation="Cumulative return rate for (SKU, Warehouse) pair strictly prior to order date",
                prediction_time_availability="Known prior to order date",
                leakage_risk_assessment="Protected by point-in-time date filtering (< date) and fallback hierarchy",
            ),
            "is_cold_start_sku": FeatureDefinition(
                name="is_cold_start_sku",
                data_type="int",
                source="historical sales",
                calculation="Binary indicator: 1 if prior SKU sales < 10 units, else 0",
                prediction_time_availability="Known prior to order date",
                leakage_risk_assessment="Protected by point-in-time date filtering (< date)",
            ),
            "is_cold_start_channel": FeatureDefinition(
                name="is_cold_start_channel",
                data_type="int",
                source="historical sales",
                calculation="Binary indicator: 1 if prior channel sales < 10 units, else 0",
                prediction_time_availability="Known prior to order date",
                leakage_risk_assessment="Protected by point-in-time date filtering (< date)",
            ),
            "is_cold_start_combination": FeatureDefinition(
                name="is_cold_start_combination",
                data_type="int",
                source="historical sales",
                calculation="Binary indicator: 1 if prior (SKU, Channel) sales < 10 units, else 0",
                prediction_time_availability="Known prior to order date",
                leakage_risk_assessment="Protected by point-in-time date filtering (< date)",
            ),
        }
        return defs
