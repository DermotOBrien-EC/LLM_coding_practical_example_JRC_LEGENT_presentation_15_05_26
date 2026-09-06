"""Empirical prediction intervals from backtest residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def residual_quantiles(
    log_errors: pd.DataFrame, levels: tuple[float, ...] = (0.05, 0.95)
) -> pd.DataFrame:
    """Quantiles of log-scale error by hours-ahead, pooled across folds.

    Working in logs keeps the interval multiplicative, which matches how the
    error actually scales with load level.
    """
    return log_errors.quantile(list(levels)).T


def apply_intervals(point: pd.Series, quantiles: pd.DataFrame) -> pd.DataFrame:
    """Attach multiplicative interval bounds to a point forecast."""
    steps = np.arange(len(point))
    out = pd.DataFrame({"forecast_mw": point.to_numpy()}, index=point.index)
    for level in quantiles.columns:
        factor = np.exp(quantiles[level].reindex(steps).to_numpy())
        out[f"q{int(level * 100):02d}_mw"] = out["forecast_mw"].to_numpy() * factor
    return out
