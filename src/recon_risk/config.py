from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ResolutionConfig:
    high_risk: frozenset = frozenset(
        {
            "cpf amend",
            "tx amend",
            "warforge amend",
            "termsheet amend",
            "risk amend",
            "deal closure",
            "de escalated",
        }
    )
    low_risk: frozenset = frozenset(
        {
            "confirmation amend",
            "system limitaion",
            "legacy closure",
            "signed off",
        }
    )


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "logistic_regression"
    top_k_frac: float = 0.10
    split_train: float = 0.70
    split_val: float = 0.15
    test_reporting_topk: Tuple[float, ...] = (0.05, 0.10, 0.20)
    primary_kpi: str = "precision_at_top_k"
    recall_guardrail: float = 0.20


@dataclass(frozen=True)
class PathsConfig:
    data_out: Path = Path("data")
    report_out: Path = Path("reports")
    model_out: Path = Path("artifacts")
