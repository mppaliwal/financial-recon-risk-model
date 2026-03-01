from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from recon_risk.api_runtime import ScoringRuntime
from recon_risk.logging_utils import get_logger
from recon_risk.logging_utils import setup_logging

setup_logging()
logger = get_logger(__name__)


class ScoreRequest(BaseModel):
    records: list[dict[str, Any]] = Field(default_factory=list)
    threshold_override: float | None = Field(default=None, ge=0.0, le=1.0)


class ScoreResponse(BaseModel):
    model_name: str
    summary: dict[str, float]
    predictions: list[dict[str, Any]]


MODEL_ROOT = Path(os.getenv("MODEL_ROOT", "artifacts"))
MODEL_NAME = os.getenv("CHAMPION_MODEL_NAME", "logistic_regression").strip().lower() or "logistic_regression"
MAX_SCORE_ROWS = int(os.getenv("API_MAX_SCORE_ROWS", "10000"))

MODEL_PATH = MODEL_ROOT / MODEL_NAME / "risk_model.pkl"
THRESHOLD_PATH = MODEL_ROOT / MODEL_NAME / "threshold.json"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

runtime = ScoringRuntime(
    model_path=MODEL_PATH,
    threshold_path=THRESHOLD_PATH if THRESHOLD_PATH.exists() else None,
)

app = FastAPI(
    title="Recon Break Risk Scoring API",
    version="1.0.0",
    description="Scores reconciliation breaks and returns risk ranking for downstream services.",
)


def _records_for_response(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    clean = clean.astype(object).where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "threshold_path": str(THRESHOLD_PATH) if THRESHOLD_PATH.exists() else None,
        "max_score_rows": MAX_SCORE_ROWS,
    }


@app.post("/v1/score", response_model=ScoreResponse)
def score_json(request: ScoreRequest) -> ScoreResponse:
    if not request.records:
        raise HTTPException(status_code=400, detail="`records` must contain at least one row.")
    if len(request.records) > MAX_SCORE_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Too many rows: {len(request.records)}. Max allowed: {MAX_SCORE_ROWS}",
        )
    df = pd.DataFrame(request.records)
    try:
        scored, summary = runtime.score_dataframe(df, threshold_override=request.threshold_override)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected scoring failure for /v1/score")
        raise HTTPException(status_code=500, detail="Internal scoring error.") from exc
    return ScoreResponse(
        model_name=MODEL_NAME,
        summary=summary,
        predictions=_records_for_response(scored),
    )


@app.post("/v1/score_csv", response_model=ScoreResponse)
async def score_csv(
    file: UploadFile = File(...),
    threshold_override: float | None = None,
) -> ScoreResponse:
    if file.filename is None or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")
    raw = await file.read()
    try:
        df = pd.read_csv(StringIO(raw.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid CSV content.") from exc
    if len(df) > MAX_SCORE_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Too many rows: {len(df)}. Max allowed: {MAX_SCORE_ROWS}",
        )
    try:
        scored, summary = runtime.score_dataframe(df, threshold_override=threshold_override)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected scoring failure for /v1/score_csv")
        raise HTTPException(status_code=500, detail="Internal scoring error.") from exc
    return ScoreResponse(
        model_name=MODEL_NAME,
        summary=summary,
        predictions=_records_for_response(scored),
    )
