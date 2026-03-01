from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import TrainingConfig


class DealTimeSplitter:
    """Grouped, chronological split at deal level."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def split(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        anchors = (
            df.groupby("deal_id", as_index=False)["raised_on"]
            .min()
            .sort_values("raised_on")
            .reset_index(drop=True)
        )
        n = len(anchors)
        if n < 3:
            raise ValueError(
                f"Need at least 3 unique deals for train/val/test split, found {n}."
            )

        i_train = int(n * self.config.split_train)
        i_val = int(n * (self.config.split_train + self.config.split_val))
        i_train = max(1, min(i_train, n - 2))
        i_val = max(i_train + 1, min(i_val, n - 1))

        train_deals = set(anchors.iloc[:i_train]["deal_id"])
        val_deals = set(anchors.iloc[i_train:i_val]["deal_id"])
        test_deals = set(anchors.iloc[i_val:]["deal_id"])

        return {
            "train": df.index[df["deal_id"].isin(train_deals)].to_numpy(),
            "val": df.index[df["deal_id"].isin(val_deals)].to_numpy(),
            "test": df.index[df["deal_id"].isin(test_deals)].to_numpy(),
        }
