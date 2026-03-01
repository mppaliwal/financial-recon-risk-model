from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResolutionConfig


class ResolutionMapper:
    """Deterministic outcome-based label mapper."""

    def __init__(self, config: ResolutionConfig) -> None:
        self.config = config

    def to_label_series(self, resolution_via: pd.Series) -> pd.Series:
        rv = (
            resolution_via.fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )
        y = pd.Series(np.nan, index=resolution_via.index, dtype="float")
        y[rv.isin(self.config.high_risk)] = 1.0
        y[rv.isin(self.config.low_risk)] = 0.0
        return y
