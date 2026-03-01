from __future__ import annotations

from pathlib import Path
from typing import Optional

from .artifacts import ArtifactStore
from .config import PathsConfig, ResolutionConfig, TrainingConfig
from .evaluation import Evaluator
from .ingestion import CsvIngestor
from .labeling import ResolutionMapper
from .logging_utils import get_logger, setup_logging
from .modeling import RiskModelFactory
from .pipeline import ReconRiskPipeline
from .preprocess import DataPreprocessor
from .splitter import DealTimeSplitter

logger = get_logger(__name__)


def build_pipeline(
    *,
    training_config: Optional[TrainingConfig] = None,
    paths_config: Optional[PathsConfig] = None,
    resolution_config: Optional[ResolutionConfig] = None,
) -> ReconRiskPipeline:
    resolution_cfg = resolution_config or ResolutionConfig()
    training_cfg = training_config or TrainingConfig()
    paths_cfg = paths_config or PathsConfig()

    mapper = ResolutionMapper(resolution_cfg)
    return ReconRiskPipeline(
        ingestor=CsvIngestor(),
        preprocessor=DataPreprocessor(mapper),
        splitter=DealTimeSplitter(training_cfg),
        model_factory=RiskModelFactory(),
        evaluator=Evaluator(),
        store=ArtifactStore(paths_cfg),
        training_config=training_cfg,
    )


def run_training_from_csv(
    input_csv: Path | str,
    *,
    training_config: Optional[TrainingConfig] = None,
    paths_config: Optional[PathsConfig] = None,
    resolution_config: Optional[ResolutionConfig] = None,
) -> dict:
    setup_logging()
    logger.info("Starting training run for input_csv=%s", input_csv)
    pipeline = build_pipeline(
        training_config=training_config,
        paths_config=paths_config,
        resolution_config=resolution_config,
    )
    result = pipeline.execute_training_from_csv(Path(input_csv))
    logger.info("Training run completed. model_path=%s", result.get("model_path"))
    return result
