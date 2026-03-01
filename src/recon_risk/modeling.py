from __future__ import annotations

from typing import Dict, List

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
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

    @staticmethod
    def supported_models() -> List[str]:
        return ["logistic_regression", "xgboost", "random_forest"]

    @staticmethod
    def xgboost_candidates() -> List[Dict[str, object]]:
        # Small, pragmatic search space for fast challenger iteration.
        return [
            {
                "n_estimators": 250,
                "max_depth": 4,
                "learning_rate": 0.06,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_weight": 3,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 450,
                "max_depth": 5,
                "learning_rate": 0.04,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "min_child_weight": 5,
                "reg_lambda": 1.5,
            },
            {
                "n_estimators": 320,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.80,
                "colsample_bytree": 0.85,
                "min_child_weight": 6,
                "reg_lambda": 2.0,
            },
            {
                "n_estimators": 380,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "min_child_weight": 2,
                "reg_lambda": 0.8,
            },
            {
                "n_estimators": 520,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "min_child_weight": 8,
                "reg_lambda": 2.5,
            },
            {
                "n_estimators": 700,
                "max_depth": 3,
                "learning_rate": 0.025,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "min_child_weight": 4,
                "reg_lambda": 2.0,
            },
            {
                "n_estimators": 280,
                "max_depth": 7,
                "learning_rate": 0.06,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "min_child_weight": 10,
                "reg_lambda": 3.0,
            },
            {
                "n_estimators": 420,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.95,
                "colsample_bytree": 0.95,
                "min_child_weight": 2,
                "reg_lambda": 0.6,
            },
        ]

    @staticmethod
    def random_forest_candidates() -> List[Dict[str, object]]:
        return [
            {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced_subsample",
            },
            {
                "n_estimators": 500,
                "max_depth": 16,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "class_weight": "balanced_subsample",
            },
            {
                "n_estimators": 700,
                "max_depth": 12,
                "min_samples_split": 6,
                "min_samples_leaf": 3,
                "class_weight": "balanced_subsample",
            },
            {
                "n_estimators": 450,
                "max_depth": 10,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "class_weight": "balanced",
            },
            {
                "n_estimators": 800,
                "max_depth": 8,
                "min_samples_split": 12,
                "min_samples_leaf": 6,
                "class_weight": "balanced",
            },
            {
                "n_estimators": 350,
                "max_depth": None,
                "min_samples_split": 8,
                "min_samples_leaf": 5,
                "class_weight": "balanced_subsample",
            },
        ]

    @staticmethod
    def _build_classifier(model_name: str, model_params: Dict[str, object] | None = None):
        model_params = model_params or {}
        if model_name == "logistic_regression":
            return LogisticRegression(
                max_iter=1200,
                class_weight="balanced",
                solver="saga",
                random_state=42,
            )
        if model_name == "xgboost":
            try:
                from xgboost import XGBClassifier
            except Exception as exc:
                raise ImportError(
                    "xgboost is required for model_name='xgboost'. "
                    "Install it with `pip install xgboost`."
                ) from exc
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=int(model_params.get("n_estimators", 350)),
                max_depth=int(model_params.get("max_depth", 5)),
                learning_rate=float(model_params.get("learning_rate", 0.05)),
                subsample=float(model_params.get("subsample", 0.85)),
                colsample_bytree=float(model_params.get("colsample_bytree", 0.85)),
                reg_lambda=float(model_params.get("reg_lambda", 1.0)),
                min_child_weight=float(model_params.get("min_child_weight", 3)),
                scale_pos_weight=float(model_params.get("scale_pos_weight", 1.0)),
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )
        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=int(model_params.get("n_estimators", 500)),
                max_depth=(
                    None if model_params.get("max_depth", None) is None
                    else int(model_params.get("max_depth"))
                ),
                min_samples_split=int(model_params.get("min_samples_split", 2)),
                min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
                class_weight=model_params.get("class_weight", "balanced_subsample"),
                n_jobs=-1,
                random_state=42,
            )
        raise ValueError(
            f"Unsupported model_name={model_name!r}. "
            f"Supported: {', '.join(RiskModelFactory.supported_models())}"
        )

    def build(
        self,
        model_name: str = "logistic_regression",
        model_params: Dict[str, object] | None = None,
    ) -> Pipeline:
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
        clf = self._build_classifier(model_name, model_params=model_params)
        return Pipeline(steps=[("pre", pre), ("clf", clf)])
