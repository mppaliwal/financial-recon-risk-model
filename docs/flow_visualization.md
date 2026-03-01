# Recon Risk Pipeline Map

```mermaid
flowchart TD
    classDef step fill:#d9f2d9,stroke:#2f9e44,color:#1b4332,stroke-width:1px;
    classDef decision fill:#74c69d,stroke:#2d6a4f,color:#081c15,stroke-width:1px;
    classDef deploy fill:#ffe8cc,stroke:#f08c00,color:#7c2d12,stroke-width:1px;

    A["Problem Statement<br/>`README.md`"]:::step --> B["Data Collection / Upload<br/>`apps/streamlit_app.py`<br/>`st.file_uploader(...)`"]:::step
    B --> B1["Input Contract Validation<br/>`src/recon_risk/contracts.py` + `src/recon_risk/ingestion.py`"]:::step
    B1 --> C["Data Preprocessing<br/>`src/recon_risk/preprocess.py`<br/>`DataPreprocessor.preprocess(...)`"]:::step
    C --> D["Choose Model + Feature Spec<br/>`src/recon_risk/modeling.py`<br/>`RiskModelFactory.build()`"]:::step

    subgraph TRAIN_LOOP["Training / Validation Loop"]
      direction LR
      E["Train Model<br/>`src/recon_risk/pipeline.py`<br/>`_train_evaluate_model(...)`"]:::step
      F["Validation + Metrics + Collinearity Check<br/>`src/recon_risk/evaluation.py` + `src/recon_risk/diagnostics.py`"]:::step
      G{"Goal Met?<br/>PR-AUC / Precision@Top-k<br/>`artifacts/metrics.json`"}:::decision
      H["Tune / Reconfigure<br/>`apps/streamlit_app.py`<br/>`top_k_frac` + settings"]:::step
      E --> F --> G
      G -- "No" --> H --> E
    end

    D --> E
    G -- "Yes" --> I["Persist Artifacts<br/>`src/recon_risk/artifacts.py`<br/>`save_model_bundle(...)`"]:::deploy
    I --> J["Score New Data<br/>`src/recon_risk/service.py`<br/>`score_dataframe(...)`"]:::deploy
```

---

## Quick Legend

- `UI`: user interaction
- `Orchestration`: run control
- `Core`: preprocessing/model/evaluation
- `Output`: saved artifacts

---

## Train Path (End-to-End)

```mermaid
flowchart LR
    %% ---------- Styles ----------
    classDef ui fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e,stroke-width:1px;
    classDef orchestration fill:#ede9fe,stroke:#7c3aed,color:#4c1d95,stroke-width:1px;
    classDef core fill:#ecfccb,stroke:#65a30d,color:#365314,stroke-width:1px;
    classDef output fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-width:1px;

    %% ---------- Nodes ----------
    A["UI Upload CSV<br/>`apps/streamlit_app.py`"]:::ui
    B["Run Train<br/>`src/recon_risk/app.py`<br/>`run_training_from_csv(...)`"]:::orchestration
    C["Validate + Read CSV<br/>`src/recon_risk/ingestion.py`<br/>`CsvIngestor.load_training_csv(...)`"]:::orchestration
    D["Preprocess + labels<br/>`src/recon_risk/preprocess.py`<br/>+ `src/recon_risk/labeling.py`"]:::core
    E["Deal-grouped time split<br/>`src/recon_risk/splitter.py`"]:::core
    F["Build model pipeline<br/>`src/recon_risk/modeling.py`"]:::core
    G["Fit + predict + threshold<br/>`src/recon_risk/evaluation.py`"]:::core
    H["Save outputs<br/>`src/recon_risk/artifacts.py`"]:::output
    I["`data/recon_breaks_processed.csv`"]:::output
    J["`reports/eda_summary.json`"]:::output
    K["`artifacts/risk_model.pkl`"]:::output
    L["`artifacts/metrics.json`"]:::output
    M["`artifacts/threshold.json`"]:::output

    %% ---------- Edges ----------
    A --> B --> C --> D --> E --> F --> G --> H
    H --> I
    H --> J
    H --> K
    H --> L
    H --> M
```

---

## Score Path (Using Saved Model)

```mermaid
flowchart LR
    classDef ui fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e,stroke-width:1px;
    classDef core fill:#ecfccb,stroke:#65a30d,color:#365314,stroke-width:1px;
    classDef output fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-width:1px;

    A["UI Upload CSV<br/>`apps/streamlit_app.py`"]:::ui
    B["Score service<br/>`src/recon_risk/service.py`<br/>`score_dataframe(...)`"]:::core
    C["Preprocess inference-safe<br/>`src/recon_risk/preprocess.py`"]:::core
    D["Load model<br/>`artifacts/risk_model.pkl`"]:::output
    E["Load threshold<br/>`artifacts/threshold.json`"]:::output
    F["Predict `risk_score` + apply `risk_flag` + `risk_rank`"]:::core
    G["Download scored CSV<br/>from Streamlit UI"]:::output

    A --> B
    B --> C
    B --> D
    B --> E
    C --> F
    D --> F
    E --> F
    F --> G
```

---

## Step Cards (Minimal)

| Step | What happens | File |
|---|---|---|
| 1 | CSV upload and schema check | [`apps/streamlit_app.py`](../apps/streamlit_app.py) |
| 2 | Build pipeline and run orchestration | [`src/recon_risk/app.py`](../src/recon_risk/app.py) |
| 3 | Validate input contract and load CSV | [`src/recon_risk/contracts.py`](../src/recon_risk/contracts.py), [`src/recon_risk/ingestion.py`](../src/recon_risk/ingestion.py) |
| 4 | Preprocess + deterministic labels | [`src/recon_risk/preprocess.py`](../src/recon_risk/preprocess.py), [`src/recon_risk/labeling.py`](../src/recon_risk/labeling.py) |
| 5 | Grouped chronological split (`deal_id`) | [`src/recon_risk/splitter.py`](../src/recon_risk/splitter.py) |
| 6 | Build + fit Logistic Regression pipeline | [`src/recon_risk/modeling.py`](../src/recon_risk/modeling.py) |
| 7 | Select top-k threshold and compute metrics | [`src/recon_risk/evaluation.py`](../src/recon_risk/evaluation.py) |
| 7b | Run numeric collinearity diagnostics (corr + VIF-style) | [`src/recon_risk/diagnostics.py`](../src/recon_risk/diagnostics.py) |
| 8 | Save model/report artifacts and run metadata | [`src/recon_risk/artifacts.py`](../src/recon_risk/artifacts.py), [`src/recon_risk/pipeline.py`](../src/recon_risk/pipeline.py) |
| 9 | Score new CSV using saved artifacts | [`src/recon_risk/service.py`](../src/recon_risk/service.py) |

---

## Quick File Links

- UI: [`apps/streamlit_app.py`](../apps/streamlit_app.py)
- App entry: [`src/recon_risk/app.py`](../src/recon_risk/app.py)
- Contracts: [`src/recon_risk/contracts.py`](../src/recon_risk/contracts.py)
- Ingestion: [`src/recon_risk/ingestion.py`](../src/recon_risk/ingestion.py)
- Pipeline orchestration: [`src/recon_risk/pipeline.py`](../src/recon_risk/pipeline.py)
- Preprocessing: [`src/recon_risk/preprocess.py`](../src/recon_risk/preprocess.py)
- Label mapping: [`src/recon_risk/labeling.py`](../src/recon_risk/labeling.py)
- Split logic: [`src/recon_risk/splitter.py`](../src/recon_risk/splitter.py)
- Model factory: [`src/recon_risk/modeling.py`](../src/recon_risk/modeling.py)
- Evaluation: [`src/recon_risk/evaluation.py`](../src/recon_risk/evaluation.py)
- Logging: [`src/recon_risk/logging_utils.py`](../src/recon_risk/logging_utils.py)
- Artifact writing: [`src/recon_risk/artifacts.py`](../src/recon_risk/artifacts.py)
- Scoring service: [`src/recon_risk/service.py`](../src/recon_risk/service.py)
