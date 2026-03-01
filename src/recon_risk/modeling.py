from __future__ import annotations

from typing import Dict, List

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RiskModelFactory:
    """Builds model pipeline from feature contracts."""

    @staticmethod
    def feature_spec() -> Dict[str, List[str] | str]:
        return {
            "categorical": [
                "deal_type",
                "asset_class",
                "product_type",
                "template",
                "team",
                "break_field",
                "desk",
                "book",
                "counterparty_tier",
                "legal_entity",
                "booking_entity",
                "notional_bucket",
                "confirm_status_at_raise",
                "ops_queue",
            ],
            "numeric": [
                "age_at_raise_days",
                "due_in_days",
                "confirm_age_hours",
                "field_count_mismatch",
                "is_new_trade",
            ],
            "text": "query_comments",
        }

    def build(self) -> Pipeline:
        spec = self.feature_spec()
        pre = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler(with_mean=False)),
                        ]
                    ),
                    spec["numeric"],
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    spec["categorical"],
                ),
                (
                    "txt",
                    TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, max_features=3500),
                    spec["text"],
                ),
            ],
            remainder="drop",
        )
        clf = LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            solver="saga",
            random_state=42,
        )
        return Pipeline(steps=[("pre", pre), ("clf", clf)])
