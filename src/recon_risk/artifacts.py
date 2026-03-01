from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from .config import PathsConfig, TrainingConfig


class ArtifactStore:
    """Persists datasets, reports, and trained model artifacts."""

    def __init__(self, paths: PathsConfig) -> None:
        self.paths = paths
        self.paths.data_out.mkdir(parents=True, exist_ok=True)
        self.paths.report_out.mkdir(parents=True, exist_ok=True)
        self.paths.model_out.mkdir(parents=True, exist_ok=True)

    def save_dataset(self, df: pd.DataFrame) -> Path:
        out = self.paths.data_out / "recon_breaks_processed.csv"
        df.to_csv(out, index=False)
        return out

    def save_eda(self, df: pd.DataFrame) -> Path:
        known = df[df["label_status"] == "known"]
        payload = {
            "row_count_total": int(len(df)),
            "deal_count_total": int(df["deal_id"].nunique()),
            "known_label_rows": int(len(known)),
            "unknown_label_rows": int((df["label_status"] == "unknown").sum()),
            "known_high_risk_rate": float(known["y"].mean()) if len(known) else None,
            "team_distribution": df["team"].value_counts(dropna=False).to_dict(),
            "deal_type_distribution": df["deal_type"].value_counts(dropna=False).to_dict(),
            "template_distribution": df["template"].value_counts(dropna=False).to_dict(),
            "break_field_distribution": df["break_field"].value_counts(dropna=False).to_dict(),
            "null_counts": df.isna().sum().sort_values(ascending=False).head(20).to_dict(),
        }
        out = self.paths.report_out / "eda_summary.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out

    def save_model_bundle(self, model: Pipeline, metrics: Dict[str, object], threshold: float) -> Dict[str, Path]:
        model_path = self.paths.model_out / "risk_model.pkl"
        metrics_path = self.paths.model_out / "metrics.json"
        threshold_path = self.paths.model_out / "threshold.json"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(threshold_path, "w", encoding="utf-8") as f:
            json.dump({"threshold_top_10pct": threshold}, f, indent=2)

        return {
            "model_path": model_path,
            "metrics_path": metrics_path,
            "threshold_path": threshold_path,
        }

    def save_baseline_config(
        self,
        *,
        training_config: TrainingConfig,
        feature_spec: Dict[str, object],
    ) -> Path:
        out = self.paths.model_out / "baseline_config.json"
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_family": "logistic_regression",
            "training_config": {
                "top_k_frac": float(training_config.top_k_frac),
                "split_train": float(training_config.split_train),
                "split_val": float(training_config.split_val),
                "test_reporting_topk": list(training_config.test_reporting_topk),
                "primary_kpi": training_config.primary_kpi,
                "recall_guardrail": float(training_config.recall_guardrail),
            },
            "feature_spec": feature_spec,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out

    def save_run_metadata(
        self,
        *,
        input_csv: Path,
        row_count: int,
        known_rows: int,
        unknown_rows: int,
        git_commit: str,
    ) -> Path:
        out = self.paths.model_out / "run_metadata.json"
        payload = {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_csv": str(input_csv),
            "row_count": int(row_count),
            "known_label_rows": int(known_rows),
            "unknown_label_rows": int(unknown_rows),
            "git_commit": git_commit,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out
