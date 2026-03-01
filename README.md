# Recon Break Risk Scoring

Predicts which reconciliation breaks are likely to become high-risk/material, using only data available when the break is raised.

For detailed flow diagrams, see `docs/flow_visualization.md`.

## 1) Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2) Input Data Contract

Each row = one break.

### Required for training
- `deal_id`
- `deal_type`
- `template`
- `team`
- `break_field`
- `trade_date`
- `raised_on`
- `review_due`
- `query_comments`
- `resolution_via`

### Required for scoring
- Same as above, except `resolution_via` is not required.

### Optional model fields
- `asset_class`
- `product_type`
- `desk`
- `book`
- `counterparty_tier`
- `legal_entity`
- `booking_entity`
- `notional_bucket`
- `is_new_trade`
- `confirm_status_at_raise`
- `confirm_age_hours`
- `ops_queue`
- `field_count_mismatch`

Missing optional fields are defaulted during preprocessing.

## 3) Labeling (Deterministic)

Target `y` is derived from `resolution_via`:

- High-risk (`y=1`): `cpf amend`, `tx amend`, `warforge amend`, `termsheet amend`, `risk amend`, `deal closure`, `de escalated`
- Low-risk (`y=0`): `confirmation amend`, `system limitaion`, `legacy closure`, `signed off`
- Anything else: unknown label (excluded from supervised train/eval)

## 4) Training and Model Selection

- Split: grouped by `deal_id`, chronological by earliest `raised_on`
- Default split: 70% train / 15% val / 15% test
- Threshold policy: top-k from validation (default `top_k_frac=0.10`)
- Supported models:
  - `logistic_regression` (baseline)
  - `xgboost` (challenger)
  - `random_forest` (challenger)
- Promotion policy:
  - primary KPI: `precision_at_top_k`
  - recall guardrail: `>= 0.20`

## 5) Run Options

### A) Streamlit (Admin / DS)

```bash
APP_MODE=admin python -m streamlit run apps/streamlit_app.py
```

Capabilities:
- Train models
- Compare champion/challengers
- Score CSVs
- Download artifacts

### B) Streamlit (Ops)

```bash
APP_MODE=ops CHAMPION_MODEL_NAME=logistic_regression python -m streamlit run apps/streamlit_app.py
```

Capabilities:
- Score-only UI
- Uses champion model artifacts

### C) Training script

```bash
RECON_MODEL_NAME=logistic_regression python scripts/run_training.py path/to/train.csv
RECON_MODEL_NAME=xgboost python scripts/run_training.py path/to/train.csv
RECON_MODEL_NAME=random_forest python scripts/run_training.py path/to/train.csv
```

### D) Scoring API (service-to-service)

```bash
CHAMPION_MODEL_NAME=logistic_regression python scripts/run_api.py
```

Health:

```bash
curl http://localhost:8000/health
```

JSON scoring:

```bash
curl -X POST "http://localhost:8000/v1/score" \
  -H "Content-Type: application/json" \
  -d '{"records":[{"deal_id":"D1","deal_type":"OTC","template":"Structured","team":"TSG","break_field":"notional","trade_date":"2025-01-10","raised_on":"2025-01-11","review_due":"2025-01-13","query_comments":"ops follow-up in progress"}]}'
```

## 6) Key Artifacts

- `artifacts/<model_name>/risk_model.pkl`
- `artifacts/<model_name>/threshold.json`
- `artifacts/<model_name>/metrics.json`
- `artifacts/<model_name>/baseline_config.json`
- `artifacts/<model_name>/run_metadata.json`
- `artifacts/model_comparison.json`
- `data/recon_breaks_processed.csv`
- `reports/eda_summary.json`
- `logs/recon_risk.log`

## 7) Docker (API)

```bash
docker build -f Dockerfile.api -t recon-risk-api:latest .
docker run --rm -p 8000:8000 -e CHAMPION_MODEL_NAME=logistic_regression recon-risk-api:latest
```

## 8) Notes

- `resolution_via` is used for training label derivation only (not as model feature).
- `resolution_date` is not used as a predictor.
- Scoring output contains `risk_score`, `risk_flag`, `risk_rank` and excludes internal training columns.
