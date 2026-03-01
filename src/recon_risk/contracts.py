from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from pydantic import BaseModel, Field, field_validator


HARD_REQUIRED_COLUMNS_TRAIN = [
    "deal_id",
    "deal_type",
    "template",
    "team",
    "break_field",
    "trade_date",
    "raised_on",
    "review_due",
    "query_comments",
    "resolution_via",
]

HARD_REQUIRED_COLUMNS_SCORE = [
    "deal_id",
    "deal_type",
    "template",
    "team",
    "break_field",
    "trade_date",
    "raised_on",
    "review_due",
    "query_comments",
]

OPTIONAL_MODEL_COLUMNS = [
    "asset_class",
    "product_type",
    "desk",
    "book",
    "counterparty_tier",
    "legal_entity",
    "booking_entity",
    "notional_bucket",
    "is_new_trade",
    "confirm_status_at_raise",
    "confirm_age_hours",
    "ops_queue",
    "field_count_mismatch",
]


class TrainingRequest(BaseModel):
    input_csv: Path

    @field_validator("input_csv")
    @classmethod
    def _validate_input_csv(cls, value: Path) -> Path:
        if not value.exists() or not value.is_file():
            raise ValueError(f"Input CSV not found: {value}")
        if value.suffix.lower() != ".csv":
            raise ValueError(f"Input must be a .csv file: {value}")
        return value


class SchemaValidationReport(BaseModel):
    mode: str
    row_count: int
    column_count: int
    missing_hard_required: List[str] = Field(default_factory=list)
    missing_optional: List[str] = Field(default_factory=list)


def schema_gaps(df: pd.DataFrame, hard_required: List[str], optional: List[str]) -> dict[str, List[str]]:
    cols = set(df.columns)
    hard_missing = [c for c in hard_required if c not in cols]
    optional_missing = [c for c in optional if c not in cols]
    return {
        "hard_missing": hard_missing,
        "optional_missing": optional_missing,
    }

