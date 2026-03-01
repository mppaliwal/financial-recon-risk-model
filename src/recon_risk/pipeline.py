from __future__ import annotations

from pathlib import Path
import subprocess
from time import perf_counter
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from .artifacts import ArtifactStore
from .config import TrainingConfig
from .diagnostics import CollinearityDiagnostics
from .evaluation import Evaluator
from .ingestion import CsvIngestor
from .logging_utils import get_logger
from .modeling import RiskModelFactory
from .preprocess import DataPreprocessor
from .splitter import DealTimeSplitter

logger = get_logger(__name__)


class ReconRiskPipeline:
    def __init__(
        self,
        *,
        ingestor: CsvIngestor,
        preprocessor: DataPreprocessor,
        splitter: DealTimeSplitter,
        model_factory: RiskModelFactory,
        evaluator: Evaluator,
        store: ArtifactStore,
        training_config: TrainingConfig,
    ) -> None:
        self.ingestor = ingestor
        self.preprocessor = preprocessor
        self.splitter = splitter
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.store = store
        self.training_config = training_config

    def _feature_columns(self) -> List[str]:
        spec = self.model_factory.feature_spec()
        return list(spec["categorical"]) + [str(spec["text"])] + list(spec["numeric"])

    @staticmethod
    def _git_commit_hash() -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                .strip()
            )
        except Exception:
            return "unknown"

    def _train_evaluate_model(
        self, df: pd.DataFrame
    ) -> Tuple[Pipeline, Dict[str, object], float, Dict[str, object]]:
        known = df[df["label_status"] == "known"].copy()
        if known.empty:
            raise ValueError(
                "No known-label rows available after preprocessing/filtering. "
                "Check resolution_via mapping and is_new_trade scope."
            )
        split = self.splitter.split(known)

        features = self._feature_columns()
        x_train = known.loc[split["train"], features]
        y_train = known.loc[split["train"], "y"].astype(int).to_numpy()
        x_val = known.loc[split["val"], features]
        y_val = known.loc[split["val"], "y"].astype(int).to_numpy()
        x_test = known.loc[split["test"], features]
        y_test = known.loc[split["test"], "y"].astype(int).to_numpy()

        collinearity_report = CollinearityDiagnostics.generate_report(
            x_train[self.model_factory.feature_spec()["numeric"]]
        )

        tuning_report: Dict[str, object] | None = None
        if self.training_config.model_name in {"xgboost", "random_forest"}:
            pos = int(y_train.sum())
            neg = int(len(y_train) - pos)
            scale_pos_weight = float(neg / max(pos, 1))

            best = None
            trials: List[Dict[str, object]] = []
            if self.training_config.model_name == "xgboost":
                candidates = self.model_factory.xgboost_candidates()
            else:
                candidates = self.model_factory.random_forest_candidates()

            for params in candidates:
                p = dict(params)
                if self.training_config.model_name == "xgboost":
                    p["scale_pos_weight"] = scale_pos_weight
                candidate = self.model_factory.build(self.training_config.model_name, model_params=p)
                candidate.fit(x_train, y_train)
                cand_val_prob = candidate.predict_proba(x_val)[:, 1]
                cand_thr = self.evaluator.choose_threshold(cand_val_prob, self.training_config.top_k_frac)
                cand_val_metrics = self.evaluator.metrics(y_val, cand_val_prob, cand_thr)
                recall_pass = bool(cand_val_metrics["recall"] >= self.training_config.recall_guardrail)
                candidate_score = (
                    1 if recall_pass else 0,
                    float(cand_val_metrics["precision"]),
                    float(cand_val_metrics["pr_auc"]),
                    float(cand_val_metrics["recall"]),
                )
                trials.append(
                    {
                        "params": p,
                        "validation_metrics": cand_val_metrics,
                        "recall_guardrail_pass": recall_pass,
                    }
                )
                if best is None or candidate_score > best["score"]:
                    best = {
                        "score": candidate_score,
                        "model": candidate,
                        "val_prob": cand_val_prob,
                        "threshold": cand_thr,
                        "params": p,
                        "validation_metrics": cand_val_metrics,
                        "recall_guardrail_pass": recall_pass,
                    }

            assert best is not None
            model = best["model"]
            val_prob = best["val_prob"]
            threshold = float(best["threshold"])
            test_prob = model.predict_proba(x_test)[:, 1]
            tuning_report = {
                "model_name": self.training_config.model_name,
                "selected_params": best["params"],
                "selected_validation_metrics": best["validation_metrics"],
                "selected_recall_guardrail_pass": best["recall_guardrail_pass"],
                "candidate_count": len(trials),
                "candidates": trials,
            }
            logger.info(
                "%s tuning selected params=%s val_precision=%.4f val_recall=%.4f",
                self.training_config.model_name,
                best["params"],
                best["validation_metrics"]["precision"],
                best["validation_metrics"]["recall"],
            )
        else:
            model = self.model_factory.build(self.training_config.model_name)
            model.fit(x_train, y_train)
            val_prob = model.predict_proba(x_val)[:, 1]
            threshold = self.evaluator.choose_threshold(val_prob, self.training_config.top_k_frac)
            test_prob = model.predict_proba(x_test)[:, 1]

        report: Dict[str, object] = {
            "split_sizes": {
                "train_rows": int(len(x_train)),
                "val_rows": int(len(x_val)),
                "test_rows": int(len(x_test)),
                "train_deals": int(known.loc[split["train"], "deal_id"].nunique()),
                "val_deals": int(known.loc[split["val"], "deal_id"].nunique()),
                "test_deals": int(known.loc[split["test"], "deal_id"].nunique()),
            },
            "validation_metrics": self.evaluator.metrics(y_val, val_prob, threshold),
            "test_metrics": self.evaluator.metrics(y_test, test_prob, threshold),
        }
        report["promotion_policy"] = {
            "model_name": self.training_config.model_name,
            "primary_kpi": self.training_config.primary_kpi,
            "top_k_frac": float(self.training_config.top_k_frac),
            "recall_guardrail": float(self.training_config.recall_guardrail),
        }
        report["promotion_check"] = {
            "precision_at_top_k": float(report["test_metrics"]["precision"]),
            "recall_at_top_k": float(report["test_metrics"]["recall"]),
            "recall_guardrail_pass": bool(
                report["test_metrics"]["recall"] >= self.training_config.recall_guardrail
            ),
        }
        for frac in self.training_config.test_reporting_topk:
            thr = self.evaluator.choose_threshold(val_prob, frac)
            report[f"test_top_{int(frac * 100)}pct"] = self.evaluator.metrics(y_test, test_prob, thr)
        if tuning_report is not None:
            report[f"{self.training_config.model_name}_tuning"] = tuning_report
        return model, report, threshold, collinearity_report

    def execute_training_from_csv(self, input_csv: Path) -> Dict[str, object]:
        t0 = perf_counter()
        df, ingestion_report = self.ingestor.load_training_csv(input_csv)
        logger.info(
            "Ingestion complete rows=%s cols=%s optional_missing=%s",
            ingestion_report.row_count,
            ingestion_report.column_count,
            len(ingestion_report.missing_optional),
        )

        t1 = perf_counter()
        df, meta = self.preprocessor.preprocess(df)
        logger.info(
            "Preprocess complete known=%s unknown=%s",
            meta.get("n_known_labels"),
            meta.get("n_unknown_labels"),
        )

        t2 = perf_counter()
        model, metrics, threshold, collinearity_report = self._train_evaluate_model(df)
        metrics["collinearity_checks"] = collinearity_report
        logger.info(
            "Train/Eval complete test_pr_auc=%.4f test_roc_auc=%.4f",
            metrics["test_metrics"]["pr_auc"],
            metrics["test_metrics"]["roc_auc"],
        )

        t3 = perf_counter()
        dataset_path = self.store.save_dataset(df)
        eda_path = self.store.save_eda(df)
        bundle_paths = self.store.save_model_bundle(
            model,
            metrics,
            threshold,
            model_name=self.training_config.model_name,
        )
        baseline_config_path = self.store.save_baseline_config(
            training_config=self.training_config,
            feature_spec=self.model_factory.feature_spec(),
        )
        run_metadata_path = self.store.save_run_metadata(
            input_csv=input_csv,
            row_count=len(df),
            known_rows=meta.get("n_known_labels", 0),
            unknown_rows=meta.get("n_unknown_labels", 0),
            git_commit=self._git_commit_hash(),
            model_name=self.training_config.model_name,
        )
        model_comparison_path = self.store.save_model_comparison()
        t4 = perf_counter()
        logger.info(
            "Artifacts saved dataset=%s metrics=%s",
            dataset_path,
            bundle_paths["metrics_path"],
        )

        timings = {
            "ingestion_sec": round(t1 - t0, 4),
            "preprocess_sec": round(t2 - t1, 4),
            "train_eval_sec": round(t3 - t2, 4),
            "artifact_save_sec": round(t4 - t3, 4),
            "total_sec": round(t4 - t0, 4),
        }
        return {
            "dataset_path": str(dataset_path),
            "meta": meta,
            "ingestion_report": ingestion_report.model_dump(),
            "eda_path": str(eda_path),
            "model_path": str(bundle_paths["model_path"]),
            "metrics_path": str(bundle_paths["metrics_path"]),
            "threshold_path": str(bundle_paths["threshold_path"]),
            "baseline_config_path": str(baseline_config_path),
            "run_metadata_path": str(run_metadata_path),
            "model_comparison_path": str(model_comparison_path),
            "test_metrics": metrics["test_metrics"],
            "collinearity_checks": collinearity_report,
            "timings_sec": timings,
        }
