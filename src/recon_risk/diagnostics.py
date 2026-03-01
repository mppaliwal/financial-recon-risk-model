from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class CollinearityDiagnostics:
    """Computes lightweight collinearity diagnostics for numeric features."""

    @staticmethod
    def _safe_vif(x: pd.DataFrame) -> Dict[str, float]:
        cols = list(x.columns)
        if len(cols) < 2:
            return {c: 1.0 for c in cols}

        values = x.astype(float).to_numpy()
        vifs: Dict[str, float] = {}

        for i, col in enumerate(cols):
            y = values[:, i]
            x_other = np.delete(values, i, axis=1)
            if x_other.shape[1] == 0:
                vifs[col] = 1.0
                continue

            model = LinearRegression()
            model.fit(x_other, y)
            r2 = float(model.score(x_other, y))
            if r2 >= 0.999999:
                vifs[col] = float("inf")
            else:
                vifs[col] = 1.0 / max(1e-12, (1.0 - r2))
        return vifs

    @staticmethod
    def generate_report(
        x_numeric: pd.DataFrame,
        corr_threshold: float = 0.85,
        vif_threshold: float = 10.0,
    ) -> Dict[str, object]:
        x = x_numeric.copy()
        x = x.replace([np.inf, -np.inf], np.nan).fillna(x.median(numeric_only=True))
        x = x.loc[:, x.nunique(dropna=False) > 1]

        if x.shape[1] == 0:
            return {
                "num_features_checked": 0,
                "high_correlation_pairs": [],
                "vif": {},
                "high_vif_features": [],
            }

        corr = x.corr(numeric_only=True).abs()
        high_corr: List[Dict[str, object]] = []
        for a, b in combinations(corr.columns, 2):
            cval = float(corr.loc[a, b])
            if cval >= corr_threshold:
                high_corr.append({"feature_a": a, "feature_b": b, "abs_corr": cval})
        high_corr.sort(key=lambda d: d["abs_corr"], reverse=True)

        vif = CollinearityDiagnostics._safe_vif(x)
        high_vif = [{"feature": k, "vif": float(v)} for k, v in vif.items() if v >= vif_threshold]
        high_vif.sort(key=lambda d: d["vif"], reverse=True)

        return {
            "num_features_checked": int(x.shape[1]),
            "corr_threshold": float(corr_threshold),
            "vif_threshold": float(vif_threshold),
            "high_correlation_pairs": high_corr,
            "vif": {k: (float(v) if np.isfinite(v) else "inf") for k, v in vif.items()},
            "high_vif_features": high_vif,
        }

