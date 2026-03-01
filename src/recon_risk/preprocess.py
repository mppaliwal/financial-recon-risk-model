from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .labeling import ResolutionMapper
from .logging_utils import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Transforms raw rows into model-ready table without leakage."""

    def __init__(self, resolution_mapper: ResolutionMapper) -> None:
        self.mapper = resolution_mapper

    @staticmethod
    def _canonicalize_categories(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["deal_type"] = out["deal_type"].replace(
            {
                "Securitization": "SEC",
                "Securitised": "SEC",
                "Listed": "SEC",
                "Repo": "OTC",
            }
        )
        return out

    def preprocess(self, raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
        logger.info("Preprocess start rows=%s cols=%s", len(raw), len(raw.columns))
        df = self._canonicalize_categories(raw.copy())
        duplicate_full_row_count = int(df.duplicated().sum())

        for col in ["trade_date", "raised_on", "review_due", "resolution_date"]:
            if col not in df.columns:
                df[col] = pd.NaT
        for col in ["trade_date", "raised_on", "review_due", "resolution_date"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")

        pre_key_dedup_rows = len(df)
        df = df.sort_values("raised_on").drop_duplicates(
            subset=["deal_id", "raised_on", "break_field", "template", "team"],
            keep="last",
        )
        duplicate_business_key_removed = int(pre_key_dedup_rows - len(df))

        if "resolution_via" not in df.columns:
            df["resolution_via"] = None
        df["y"] = self.mapper.to_label_series(df["resolution_via"])
        df["label_status"] = np.where(df["y"].isna(), "unknown", "known")

        df["age_at_raise_days"] = (df["raised_on"] - df["trade_date"]).dt.days
        df["due_in_days"] = (df["review_due"] - df["raised_on"]).dt.days

        for col in ["confirm_age_hours", "field_count_mismatch", "is_new_trade"]:
            if col not in df.columns:
                df[col] = np.nan

        def _safe_median(series: pd.Series, fallback: float) -> float:
            med = series.median()
            return float(med) if pd.notna(med) else float(fallback)

        defaults = {
            "age_at_raise_days": _safe_median(df["age_at_raise_days"], 0.0),
            "due_in_days": _safe_median(df["due_in_days"], 0.0),
            "confirm_age_hours": _safe_median(df["confirm_age_hours"], 24.0),
            "field_count_mismatch": 1,
            "is_new_trade": 1,
        }
        for col, val in defaults.items():
            df[col] = df[col].fillna(val)

        # Business scope: only new deals are in-model scope.
        df["is_new_trade"] = pd.to_numeric(df["is_new_trade"], errors="coerce").fillna(1).astype(int)
        pre_filter_rows = len(df)
        df = df[df["is_new_trade"] == 1].copy()
        filtered_non_new = int(pre_filter_rows - len(df))

        df["query_comments"] = df["query_comments"].fillna("")

        for col, fallback in [
            ("product_type", "UNKNOWN"),
            ("desk", "UNKNOWN"),
            ("book", "UNKNOWN"),
            ("counterparty_tier", "UNKNOWN"),
            ("legal_entity", "UNKNOWN"),
            ("booking_entity", "UNKNOWN"),
            ("notional_bucket", "UNKNOWN"),
            ("confirm_status_at_raise", "UNKNOWN"),
            ("ops_queue", "UNKNOWN"),
            ("asset_class", "Equity"),
        ]:
            if col not in df.columns:
                df[col] = fallback
            df[col] = df[col].fillna(fallback)

        known = df[df["label_status"] == "known"]
        meta = {
            "n_rows": int(len(df)),
            "n_known_labels": int(len(known)),
            "n_unknown_labels": int((df["label_status"] == "unknown").sum()),
            "high_risk_rate_known": float(known["y"].mean()) if len(known) else None,
            "filtered_non_new_rows": filtered_non_new,
            "duplicate_full_row_count": duplicate_full_row_count,
            "duplicate_business_key_removed": duplicate_business_key_removed,
        }
        logger.info(
            "Preprocess done rows=%s known=%s unknown=%s filtered_non_new=%s",
            meta["n_rows"],
            meta["n_known_labels"],
            meta["n_unknown_labels"],
            meta["filtered_non_new_rows"],
        )
        return df, meta
