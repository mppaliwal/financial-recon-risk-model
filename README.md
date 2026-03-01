# Recon Break Risk Scoring Pipeline

This repository contains a training pipeline for **Financial Reconciliation Break Risk Analysis**.
It trains a model that predicts whether a raised break is likely to become **material/high-risk**,
using only information available at the time the break is raised.

The code is organized for programmatic use (for example from a future Streamlit app), not CLI flags.

## Visual Flow

For an end-to-end flow diagram (Train + Score paths), see:
- `docs/flow_visualization.md`

## What This Pipeline Does

1. Loads a break-level CSV dataset.
2. Preprocesses/cleans the data and creates leakage-safe engineered features.
3. Builds deterministic labels from `resolution_via`.
4. Splits data by `deal_id` with chronological ordering to avoid leakage.
5. Trains a baseline risk model (Logistic Regression with mixed tabular + text features).
6. Evaluates model performance and selects an operating threshold (top-k capacity style).
7. Saves processed data, EDA summary, model, metrics, and threshold artifacts.

## Production-Grade Core Additions

- **Pydantic input contract validation**:
  - `src/recon_risk/contracts.py`
  - `src/recon_risk/ingestion.py`
  - validates training input path and schema before preprocessing starts.

- **Structured logging**:
  - `src/recon_risk/logging_utils.py`
  - stage-level logs for ingestion, preprocessing, training/evaluation, scoring, and artifact persistence.
  - default log file: `logs/recon_risk.log`

- **Training run metadata**:
  - training response includes `ingestion_report` and stage `timings_sec`.

## Repository Structure

- `src/recon_risk/`: Core package (domain, preprocessing, model, pipeline, service utilities)
- `src/recon_risk/contracts.py`: Pydantic request + schema contracts (hard-required/optional fields)
- `src/recon_risk/ingestion.py`: Input CSV loading + contract validation
- `src/recon_risk/logging_utils.py`: Central logging setup
- `src/recon_risk/diagnostics.py`: Collinearity diagnostics (correlation + VIF-style)
- `apps/streamlit_app.py`: Main Streamlit application UI
- `scripts/run_training.py`: Script entrypoint for training from CSV
- `data/`, `reports/`, `artifacts/`: Runtime outputs
- `tests/`: Placeholder for unit/integration tests
- `pyproject.toml`: Package metadata and `src` layout config

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Required Input Data

Each row should represent one break.

Hard-required columns (run is blocked if missing):
- `deal_id`
- `deal_type`
- `template`
- `team`
- `break_field`
- `trade_date`
- `raised_on`
- `review_due`
- `query_comments`
- `resolution_via` (training only, used for deterministic labeling)

Optional model columns (allowed missing; defaults are applied and shown in UI warnings):
- `asset_class`
- `product_type`
- `desk`
- `book`
- `counterparty_tier`
- `legal_entity`
- `booking_entity`
- `notional_bucket`
- `confirm_status_at_raise`
- `confirm_age_hours`
- `ops_queue`
- `field_count_mismatch`

`resolution_date` is optional traceability metadata and not used as a model feature.

## Labeling Logic

Target `y` is deterministic from `resolution_via`:

- `y = 1` for high-risk outcomes:
  - `cpf amend`
  - `tx amend`
  - `warforge amend`
  - `termsheet amend`
  - `risk amend`
  - `deal closure`
  - `de escalated`

- `y = 0` for low-risk outcomes:
  - `confirmation amend`
  - `system limitaion`
  - `legacy closure`
  - `signed off`

Rows with any other `resolution_via` are treated as `unknown` labels and excluded from supervised training/evaluation.

## Train/Validation/Test Strategy

- Split unit: `deal_id` (never split rows of the same deal across train/val/test).
- Time logic: deals sorted by earliest `raised_on`.
- Default split:
  - 70% deals train
  - 15% deals validation
  - 15% deals test

This simulates forward-looking deployment and prevents leakage from repeated deal context.

## Model and Features

Baseline model:
- Logistic Regression (`class_weight="balanced"`, solver `saga`)

Feature families:
- Categorical: deal/product/ops context fields
- Numeric: age/SLA/time and related numeric fields
- Text: `query_comments` via TF-IDF (1-2 grams)

Thresholding:
- Uses validation predictions to choose a top-k threshold (default top 10% risk flags).
- Also reports metrics at top 5% and top 20% for capacity planning.

## How to Run

### Option 1: Runner script

```bash
python scripts/run_training.py path/to/breaks.csv
```

Or set environment variable:

```bash
export RECON_INPUT_CSV=path/to/breaks.csv
python scripts/run_training.py
```

### Option 2: Streamlit UI

```bash
python -m streamlit run apps/streamlit_app.py
```

UI capabilities:
- Train from uploaded CSV
- Validate required schema
- Show run diagnostics (ingestion report, stage timings, collinearity alerts)
- Save and download artifacts (processed data, metrics, threshold, model)
- Score new uploaded CSVs with saved or uploaded model artifacts
- Download scored CSV with `risk_score`, `risk_flag`, and `risk_rank`

### Option 3: Programmatic (recommended for app integration)

```python
from recon_risk.app import run_training_from_csv

summary = run_training_from_csv(input_csv="data/breaks.csv")
print(summary)
```

Custom configuration example:

```python
from pathlib import Path
from recon_risk.app import run_training_from_csv
from recon_risk.config import PathsConfig, TrainingConfig

summary = run_training_from_csv(
    input_csv="data/breaks.csv",
    training_config=TrainingConfig(top_k_frac=0.10),
    paths_config=PathsConfig(
        data_out=Path("data"),
        report_out=Path("reports"),
        model_out=Path("artifacts"),
    ),
)
```

## Outputs

After a successful run:

- `data/recon_breaks_processed.csv`
  - Cleaned and feature-engineered dataset used for modeling.

- `reports/eda_summary.json`
  - High-level EDA snapshot (counts, label availability, distributions, null profile).

- `artifacts/risk_model.pkl`
  - Trained sklearn pipeline (preprocessing + model).

- `artifacts/metrics.json`
  - Validation/test metrics, top-k operating-point reports, and collinearity checks.

- `artifacts/threshold.json`
  - Saved top-10% threshold for scoring workflows.

- `artifacts/baseline_config.json`
  - Frozen baseline training policy and feature spec snapshot for fair champion/challenger comparison.

- `artifacts/run_metadata.json`
  - Run timestamp, input source, known/unknown label counts, and git commit hash (if available).

- `logs/recon_risk.log`
  - Stage-level application logs (ingestion, preprocess, train/eval, scoring, persistence).

## Notes on Leakage and Safety

- `resolution_via` is used only for labeling, not as an input feature.
- `resolution_date` is not used as a predictor.
- Split is deal-grouped and chronological to reduce optimistic bias.
- Business scope filter: only rows with `is_new_trade = 1` are kept for training/scoring.

## Typical Next Steps

1. Replace Logistic Regression with challenger models (XGBoost/LightGBM/CatBoost).
2. Add model selection logic based on your business objective (for example precision@top-k).
3. Add scoring service method (`score_dataframe`) for frontend upload flows.
4. Add explainability layer (feature importance / SHAP) for operational review.
5. Add unit tests for labeling, split integrity, and preprocessing contracts.

## Troubleshooting

- If training fails due to missing columns:
  - Ensure required core columns are present and correctly named.

- If no labeled rows are available:
  - Check `resolution_via` values and mapping categories.

- If date parsing is poor:
  - Ensure `trade_date`, `raised_on`, `review_due`, `resolution_date` are valid date strings.

- If model quality is unstable:
  - Verify that `deal_id` granularity and time ordering are correct.
  - Check label prevalence and unknown-label share in `reports/eda_summary.json`.
