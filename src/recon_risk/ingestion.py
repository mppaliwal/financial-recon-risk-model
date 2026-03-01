from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contracts import (
    HARD_REQUIRED_COLUMNS_TRAIN,
    OPTIONAL_MODEL_COLUMNS,
    SchemaValidationReport,
    TrainingRequest,
    schema_gaps,
)
from .logging_utils import get_logger

logger = get_logger(__name__)


class CsvIngestor:
    """Loads CSV and validates schema contracts before training."""

    def load_training_csv(self, input_csv: Path | str) -> tuple[pd.DataFrame, SchemaValidationReport]:
        request = TrainingRequest(input_csv=Path(input_csv))
        df = pd.read_csv(request.input_csv)

        gaps = schema_gaps(df, HARD_REQUIRED_COLUMNS_TRAIN, OPTIONAL_MODEL_COLUMNS)
        report = SchemaValidationReport(
            mode="train",
            row_count=int(len(df)),
            column_count=int(len(df.columns)),
            missing_hard_required=gaps["hard_missing"],
            missing_optional=gaps["optional_missing"],
        )

        if report.missing_hard_required:
            logger.error("Hard-required schema columns missing: %s", report.missing_hard_required)
            raise ValueError(
                "Missing hard-required columns: " + ", ".join(report.missing_hard_required)
            )

        if report.missing_optional:
            logger.warning(
                "Optional columns missing; defaults will be applied: %s",
                report.missing_optional,
            )
        return df, report

