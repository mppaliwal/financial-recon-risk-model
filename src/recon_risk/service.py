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


def score_dataframe(
    df: pd.DataFrame,
    model_path: Path | str,
    threshold_path: Path | str | None = None,
    threshold_override: float | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    logger.info("Scoring start rows=%s", len(df))
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

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    mapper = ResolutionMapper(ResolutionConfig())
    preprocessor = DataPreprocessor(mapper)
    prepared, meta = preprocessor.preprocess(df)

    if prepared.empty:
        out = prepared.copy()
        out["risk_score"] = []
        out["risk_flag"] = []
        out["risk_rank"] = []
        summary = {
            "threshold": float(threshold_override if threshold_override is not None else 0.5),
            "rows": 0,
            "flagged_rows": 0,
            "flag_rate": 0.0,
            "avg_risk_score": 0.0,
            "filtered_non_new_rows": int(meta.get("filtered_non_new_rows", 0)),
        }
        logger.warning("Scoring skipped: no in-scope rows after is_new_trade filter.")
        return out, summary

    spec = RiskModelFactory.feature_spec()
    feature_cols = list(spec["categorical"]) + [str(spec["text"])] + list(spec["numeric"])
    x = prepared[feature_cols]
    risk_score = model.predict_proba(x)[:, 1]

    threshold = 0.5
    if threshold_path:
        with open(threshold_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        threshold = float(payload.get("threshold_top_10pct", 0.5))
    if threshold_override is not None:
        threshold = float(threshold_override)

    out = prepared.copy()
    out["risk_score"] = risk_score
    out["risk_flag"] = (out["risk_score"] >= threshold).astype(int)
    out["risk_rank"] = out["risk_score"].rank(method="dense", ascending=False).astype(int)
    out = out.sort_values("risk_score", ascending=False).reset_index(drop=True)

    summary = {
        "threshold": float(threshold),
        "rows": int(len(out)),
        "flagged_rows": int(out["risk_flag"].sum()),
        "flag_rate": float(out["risk_flag"].mean()) if len(out) else 0.0,
        "avg_risk_score": float(out["risk_score"].mean()) if len(out) else 0.0,
        "filtered_non_new_rows": int(meta.get("filtered_non_new_rows", 0)),
    }
    logger.info(
        "Scoring done rows=%s flagged=%s flag_rate=%.4f",
        summary["rows"],
        summary["flagged_rows"],
        summary["flag_rate"],
    )
    return out, summary
