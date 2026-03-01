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

    def _model_dir(self, model_name: str) -> Path:
        model_dir = self.paths.model_out / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

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

    def save_model_bundle(
        self,
        model: Pipeline,
        metrics: Dict[str, object],
        threshold: float,
        *,
        model_name: str,
    ) -> Dict[str, Path]:
        model_dir = self._model_dir(model_name)
        model_path = model_dir / "risk_model.pkl"
        metrics_path = model_dir / "metrics.json"
        threshold_path = model_dir / "threshold.json"

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
        model_dir = self._model_dir(training_config.model_name)
        out = model_dir / "baseline_config.json"
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_family": training_config.model_name,
            "training_config": {
                "model_name": training_config.model_name,
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
        model_name: str,
    ) -> Path:
        model_dir = self._model_dir(model_name)
        out = model_dir / "run_metadata.json"
        payload = {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_csv": str(input_csv),
            "model_name": model_name,
            "row_count": int(row_count),
            "known_label_rows": int(known_rows),
            "unknown_label_rows": int(unknown_rows),
            "git_commit": git_commit,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out

    def save_model_comparison(self) -> Path:
        out = self.paths.model_out / "model_comparison.json"

        def _score_key(item: Dict[str, object]) -> tuple:
            tm = item.get("test_metrics", {})
            recall_pass = bool(item.get("recall_guardrail_pass", False))
            return (
                1 if recall_pass else 0,
                float(tm.get("precision", 0.0)),
                float(tm.get("pr_auc", 0.0)),
                float(tm.get("roc_auc", 0.0)),
            )

        models: list[Dict[str, object]] = []
        for model_dir in sorted(self.paths.model_out.iterdir()) if self.paths.model_out.exists() else []:
            if not model_dir.is_dir():
                continue
            metrics_path = model_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            policy = metrics.get("promotion_policy", {})
            check = metrics.get("promotion_check", {})
            models.append(
                {
                    "model_name": model_dir.name,
                    "metrics_path": str(metrics_path),
                    "promotion_policy": policy,
                    "test_metrics": metrics.get("test_metrics", {}),
                    "recall_guardrail_pass": bool(check.get("recall_guardrail_pass", False)),
                }
            )

        ranked = sorted(models, key=_score_key, reverse=True)
        champion = ranked[0] if ranked else None
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ranking_rule": {
                "order": [
                    "recall_guardrail_pass",
                    "test_metrics.precision",
                    "test_metrics.pr_auc",
                    "test_metrics.roc_auc",
                ]
            },
            "champion": champion,
            "models": ranked,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out
