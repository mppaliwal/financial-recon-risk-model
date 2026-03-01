from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import ResolutionConfig
from .contracts import HARD_REQUIRED_COLUMNS_SCORE, OPTIONAL_MODEL_COLUMNS, schema_gaps
from .labeling import ResolutionMapper
from .logging_utils import get_logger
from .modeling import RiskModelFactory
from .preprocess import DataPreprocessor

logger = get_logger(__name__)


class ScoringRuntime:
    """In-memory scoring runtime for low-latency API inference."""

    def __init__(
        self,
        *,
        model_path: Path | str,
        threshold_path: Path | str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold_path = Path(threshold_path) if threshold_path else None

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        self.default_threshold = 0.5
        if self.threshold_path and self.threshold_path.exists():
            with open(self.threshold_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.default_threshold = float(payload.get("threshold_top_10pct", 0.5))

        self.spec = RiskModelFactory.feature_spec()
        self.feature_cols = list(self.spec["categorical"]) + [str(self.spec["text"])] + list(self.spec["numeric"])
        self.preprocessor = DataPreprocessor(ResolutionMapper(ResolutionConfig()))

        logger.info(
            "API scoring runtime initialized model=%s threshold=%.4f",
            self.model_path,
            self.default_threshold,
        )

    def score_dataframe(
        self,
        df: pd.DataFrame,
        *,
        threshold_override: float | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        gaps = schema_gaps(
            df,
            hard_required=HARD_REQUIRED_COLUMNS_SCORE,
            optional=OPTIONAL_MODEL_COLUMNS,
        )
        if gaps["hard_missing"]:
            raise ValueError("Missing hard-required columns: " + ", ".join(gaps["hard_missing"]))
        if gaps["optional_missing"]:
            logger.warning(
                "Optional columns missing in scoring input; defaults will be applied: %s",
                gaps["optional_missing"],
            )

        prepared, meta = self.preprocessor.preprocess(df)
        if prepared.empty:
            out = prepared.copy()
            out["risk_score"] = []
            out["risk_flag"] = []
            out["risk_rank"] = []
            summary = {
                "threshold": float(
                    threshold_override if threshold_override is not None else self.default_threshold
                ),
                "rows": 0,
                "flagged_rows": 0,
                "flag_rate": 0.0,
                "avg_risk_score": 0.0,
                "filtered_non_new_rows": int(meta.get("filtered_non_new_rows", 0)),
            }
            return out, summary

        x = prepared[self.feature_cols]
        risk_score = self.model.predict_proba(x)[:, 1]

        threshold = float(self.default_threshold if threshold_override is None else threshold_override)
        out = prepared.copy()
        out["risk_score"] = risk_score
        out["risk_flag"] = (out["risk_score"] >= threshold).astype(int)
        out["risk_rank"] = out["risk_score"].rank(method="dense", ascending=False).astype(int)
        for internal_col in ["y", "label_status"]:
            if internal_col in out.columns:
                out = out.drop(columns=[internal_col])
        out = out.sort_values("risk_score", ascending=False).reset_index(drop=True)

        summary = {
            "threshold": float(threshold),
            "rows": int(len(out)),
            "flagged_rows": int(out["risk_flag"].sum()),
            "flag_rate": float(out["risk_flag"].mean()) if len(out) else 0.0,
            "avg_risk_score": float(out["risk_score"].mean()) if len(out) else 0.0,
            "filtered_non_new_rows": int(meta.get("filtered_non_new_rows", 0)),
        }
        return out, summary
