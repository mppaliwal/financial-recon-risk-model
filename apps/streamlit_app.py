from __future__ import annotations
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from recon_risk.app import run_training_from_csv
from recon_risk.config import PathsConfig, TrainingConfig
from recon_risk.contracts import (
    HARD_REQUIRED_COLUMNS_SCORE,
    HARD_REQUIRED_COLUMNS_TRAIN,
    OPTIONAL_MODEL_COLUMNS,
    schema_gaps,
)
from recon_risk.logging_utils import setup_logging
from recon_risk.modeling import RiskModelFactory
from recon_risk.service import score_dataframe

setup_logging()

APP_MODE = os.getenv("APP_MODE", "admin").strip().lower()
if APP_MODE not in {"admin", "ops"}:
    APP_MODE = "admin"
CHAMPION_MODEL_NAME = os.getenv("CHAMPION_MODEL_NAME", "logistic_regression").strip().lower()
if CHAMPION_MODEL_NAME not in RiskModelFactory.supported_models():
    CHAMPION_MODEL_NAME = "logistic_regression"


st.set_page_config(
    page_title="Recon Break Risk Studio",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(1200px 500px at 10% -10%, #dbeafe 0%, transparent 60%),
                    radial-gradient(1000px 420px at 90% -20%, #fee2e2 0%, transparent 55%),
                    #f8fafc;
    }
    .hero {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        margin-bottom: 1rem;
    }
    .muted {
        color: #475569;
        font-size: 0.92rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h2 style="margin:0;">Recon Break Risk Studio</h2>
  <p style="margin:0.35rem 0 0 0;">
    Train, evaluate, and score reconciliation breaks from CSV files.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Run Settings")
    if APP_MODE == "admin":
        model_name = st.selectbox(
            "Model",
            options=RiskModelFactory.supported_models(),
            index=0,
            help="Choose baseline or challenger model for training.",
        )
        top_k_frac = st.slider(
            "Top-k Threshold Fraction", min_value=0.01, max_value=0.50, value=0.10, step=0.01
        )
    else:
        model_name = CHAMPION_MODEL_NAME
        top_k_frac = 0.10
        st.info(f"Ops mode enabled. Champion model: `{model_name}`")
    data_out = st.text_input("Processed Data Folder", value="data")
    report_out = st.text_input("Report Folder", value="reports")
    model_out = st.text_input("Model Folder", value="artifacts")

if APP_MODE == "admin":
    tabs = st.tabs(["Train Model", "Score Breaks", "Run Summary"])
    train_tab, score_tab, summary_tab = tabs
else:
    tabs = st.tabs(["Score Breaks"])
    score_tab = tabs[0]

if APP_MODE == "admin":
    with train_tab:
        st.markdown("#### Upload training CSV")
        train_file = st.file_uploader(
            "Training dataset",
            type=["csv"],
            key="train_csv",
            help="Must include `resolution_via` for deterministic labeling.",
        )

        if train_file is not None:
            df_train = pd.read_csv(train_file)
            gaps = schema_gaps(
                df_train,
                hard_required=HARD_REQUIRED_COLUMNS_TRAIN,
                optional=OPTIONAL_MODEL_COLUMNS,
            )
            hard_miss = gaps["hard_missing"]
            optional_miss = gaps["optional_missing"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(df_train):,}")
            c2.metric("Columns", f"{len(df_train.columns):,}")
            c3.metric("Missing Hard-Required", len(hard_miss))

            if hard_miss:
                st.error("Missing hard-required columns: " + ", ".join(hard_miss))
            else:
                st.success("Schema check passed.")
                if optional_miss:
                    st.warning(
                        "Optional model columns missing (defaults will be applied): "
                        + ", ".join(optional_miss)
                    )

            with st.expander("Preview Data", expanded=False):
                st.dataframe(df_train.head(30), use_container_width=True)

            if st.button("Train Model", type="primary", disabled=bool(hard_miss)):
                with st.spinner("Training pipeline running..."):
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
                        df_train.to_csv(tf.name, index=False)
                        input_path = tf.name

                    summary = run_training_from_csv(
                        input_csv=input_path,
                        training_config=TrainingConfig(model_name=model_name, top_k_frac=top_k_frac),
                        paths_config=PathsConfig(
                            data_out=Path(data_out),
                            report_out=Path(report_out),
                            model_out=Path(model_out),
                        ),
                    )
                    st.session_state["train_summary"] = summary

                st.success("Training completed.")
        train_summary = st.session_state.get("train_summary")
        if train_summary:
            tm = train_summary.get("test_metrics", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PR-AUC", f"{tm.get('pr_auc', 0):.4f}")
            m2.metric("ROC-AUC", f"{tm.get('roc_auc', 0):.4f}")
            m3.metric("Precision", f"{tm.get('precision', 0):.4f}")
            m4.metric("Recall", f"{tm.get('recall', 0):.4f}")

            st.markdown("#### Run Diagnostics")
            ingestion = train_summary.get("ingestion_report", {})
            timings = train_summary.get("timings_sec", {})
            coll = train_summary.get("collinearity_checks", {})

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Ingest Rows", f"{ingestion.get('row_count', 0):,}")
            d2.metric("Ingest Cols", f"{ingestion.get('column_count', 0):,}")
            d3.metric("Hard Missing", len(ingestion.get("missing_hard_required", [])))
            d4.metric("Optional Missing", len(ingestion.get("missing_optional", [])))

            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("Ingestion (s)", f"{timings.get('ingestion_sec', 0):.3f}")
            t2.metric("Preprocess (s)", f"{timings.get('preprocess_sec', 0):.3f}")
            t3.metric("Train+Eval (s)", f"{timings.get('train_eval_sec', 0):.3f}")
            t4.metric("Save (s)", f"{timings.get('artifact_save_sec', 0):.3f}")
            t5.metric("Total (s)", f"{timings.get('total_sec', 0):.3f}")

            high_corr = coll.get("high_correlation_pairs", [])
            high_vif = coll.get("high_vif_features", [])
            c1, c2 = st.columns(2)
            c1.metric("High Corr Pairs", len(high_corr))
            c2.metric("High VIF Features", len(high_vif))

            with st.expander("Collinearity Details", expanded=False):
                st.write("High correlation pairs (|corr| >= threshold)")
                st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
                st.write("High VIF features (>= threshold)")
                st.dataframe(pd.DataFrame(high_vif), use_container_width=True)

            st.markdown("#### Download Artifacts")
            for label, key in [
                ("Processed CSV", "dataset_path"),
                ("EDA Summary", "eda_path"),
                ("Metrics JSON", "metrics_path"),
                ("Threshold JSON", "threshold_path"),
                ("Baseline Config", "baseline_config_path"),
                ("Run Metadata", "run_metadata_path"),
                ("Model Comparison", "model_comparison_path"),
                ("Model PKL", "model_path"),
            ]:
                path = train_summary.get(key)
                if path and Path(path).exists():
                    mode = "rb" if path.endswith(".pkl") else "r"
                    with open(path, mode) as f:
                        data = f.read()
                    st.download_button(
                        label=f"Download {label}",
                        data=data,
                        file_name=Path(path).name,
                        mime="application/octet-stream" if path.endswith(".pkl") else "text/plain",
                        key=f"dl_{key}",
                    )

with score_tab:
    st.markdown("#### Upload scoring CSV")
    score_file = st.file_uploader("Scoring dataset", type=["csv"], key="score_csv")
    st.markdown('<p class="muted">Use existing model artifacts or upload your own model + threshold files.</p>', unsafe_allow_html=True)

    source = st.radio("Model Source", ["Use Artifacts Folder", "Upload Files"], horizontal=True)

    model_path_input = Path(model_out) / model_name / "risk_model.pkl"
    threshold_path_input = Path(model_out) / model_name / "threshold.json"

    model_path = None
    threshold_path = None
    threshold_override = st.checkbox("Override threshold manually")
    threshold_val = st.slider("Manual threshold", 0.01, 0.99, 0.50, 0.01, disabled=not threshold_override)

    if source == "Use Artifacts Folder":
        model_path = Path(st.text_input("Model file", value=str(model_path_input), key="model_path_input"))
        threshold_path = Path(
            st.text_input("Threshold file", value=str(threshold_path_input), key="threshold_path_input")
        )
    else:
        model_upload = st.file_uploader("Upload model `.pkl`", type=["pkl"], key="upload_model")
        threshold_upload = st.file_uploader("Upload threshold `.json`", type=["json"], key="upload_threshold")
        if model_upload:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as mf:
                mf.write(model_upload.getvalue())
                model_path = Path(mf.name)
        if threshold_upload:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                tf.write(threshold_upload.getvalue())
                threshold_path = Path(tf.name)

    if score_file is not None:
        df_score = pd.read_csv(score_file)
        gaps = schema_gaps(
            df_score,
            hard_required=HARD_REQUIRED_COLUMNS_SCORE,
            optional=OPTIONAL_MODEL_COLUMNS,
        )
        hard_miss = gaps["hard_missing"]
        optional_miss = gaps["optional_missing"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df_score):,}")
        c2.metric("Columns", f"{len(df_score.columns):,}")
        c3.metric("Missing Hard-Required", len(hard_miss))

        if hard_miss:
            st.error("Missing hard-required columns: " + ", ".join(hard_miss))
        else:
            st.success("Schema check passed.")
            if optional_miss:
                st.warning(
                    "Optional model columns missing (defaults will be applied): "
                    + ", ".join(optional_miss)
                )

        can_score = (
            not hard_miss
            and model_path is not None
            and Path(model_path).exists()
            and (threshold_override or (threshold_path is not None and Path(threshold_path).exists()))
        )
        if st.button("Score Breaks", type="primary", disabled=not can_score):
            with st.spinner("Scoring in progress..."):
                scored_df, score_summary = score_dataframe(
                    df_score,
                    model_path=model_path,
                    threshold_path=None if threshold_override else threshold_path,
                    threshold_override=threshold_val if threshold_override else None,
                )
                st.session_state["score_summary"] = score_summary
                st.session_state["scored_df"] = scored_df

            st.success("Scoring completed.")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Rows", f"{score_summary['rows']:,}")
            s2.metric("Flagged", f"{score_summary['flagged_rows']:,}")
            s3.metric("Flag Rate", f"{score_summary['flag_rate']:.2%}")
            s4.metric("Avg Risk", f"{score_summary['avg_risk_score']:.4f}")

    if "scored_df" in st.session_state:
        st.markdown("#### Highest-Risk Breaks")
        st.dataframe(st.session_state["scored_df"].head(100), use_container_width=True)

        csv_out = st.session_state["scored_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Scored CSV",
            data=csv_out,
            file_name="recon_breaks_scored.csv",
            mime="text/csv",
        )

if APP_MODE == "admin":
    with summary_tab:
        st.markdown("#### Latest Run Details")
        train_summary = st.session_state.get("train_summary")
        score_summary = st.session_state.get("score_summary")

        if train_summary:
            st.markdown("##### Training Summary")
            st.json(train_summary)
        else:
            st.info("No training run in this session yet.")

        if score_summary:
            st.markdown("##### Scoring Summary")
            st.json(score_summary)
        else:
            st.info("No scoring run in this session yet.")
