"""SARIMA - the classical statistical entry in the bake-off.

A SARIMA model explains a series by its own recent past plus a repeating
seasonal pattern. Hourly electricity load has two strong cycles: a daily one
(24 hours) and a weekly one (168 hours). Putting the weekly cycle straight
into the seasonal term is not practical - a seasonal period of 168 makes the
model's internal state so large that fitting takes hours - so we do what the
textbooks suggest and handle the two cycles separately:

  * variant "daily"  - SARIMA with a 24-hour seasonal term only, fitted to the
    load itself. Weekly structure has to emerge from the ordinary AR terms.
  * variant "weekly-diff" - first subtract the value 168 hours earlier from
    every observation (that removes the weekly cycle by hand), then fit a
    SARIMA with a 24-hour seasonal term to what is left, then add the weekly
    offset back to get the forecast.

Both variants, several (p,d,q)(P,D,Q,24) orders, and two lengths of training
window are put to the validation set, and whichever scores best there is the
one refitted on train+validation and used on the test week.

Why a training window at all? Fitting a state-space model to all 43,824
training hours is very slow and buys little: the parameters of an hourly load
model are estimated well from one or two recent years, and older data mostly
reflects a different level of demand. The window length is therefore treated
as a hyperparameter and chosen on the validation set like everything else.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (
    HORIZON,
    TEST_START,
    VALIDATION_ORIGINS,
    ModelForecast,
    history_before,
    horizon_index,
    mape,
)

WEEK: int = 168


@dataclass(frozen=True)
class SarimaConfig:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    variant: str          # "daily" or "weekly-diff"
    window_weeks: int     # how much recent history to fit on

    def as_dict(self) -> dict[str, object]:
        return {
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "variant": self.variant,
            "window_weeks": self.window_weeks,
        }


# A deliberately small grid. Each entry costs seven model fits during
# selection, so breadth is bought with wall-clock time.
CANDIDATES: tuple[SarimaConfig, ...] = (
    SarimaConfig((2, 0, 1), (1, 1, 1, 24), "daily", 26),
    SarimaConfig((3, 1, 1), (1, 1, 1, 24), "daily", 26),
    SarimaConfig((2, 0, 1), (1, 1, 1, 24), "weekly-diff", 26),
    SarimaConfig((2, 0, 1), (1, 0, 1, 24), "weekly-diff", 26),
    SarimaConfig((3, 0, 1), (2, 0, 2, 24), "weekly-diff", 26),
    SarimaConfig((3, 0, 2), (2, 0, 1, 24), "weekly-diff", 26),
    SarimaConfig((2, 0, 2), (1, 0, 1, 24), "weekly-diff", 52),
)


def _fit(values: np.ndarray, cfg: SarimaConfig) -> object:
    """Fit one SARIMAX on the given target values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            values,
            order=cfg.order,
            seasonal_order=cfg.seasonal_order,
            trend=None,
            # simple_differencing must stay False. With it set to True,
            # statsmodels fits the differenced series and get_forecast then
            # returns forecasts of the DIFFERENCED series, not of load. That
            # silently turned this model into the naive baseline during
            # development: the weekly-diff variant scored MAPE 7.12%, which
            # was exactly the seasonal-naive score to two decimal places,
            # because the near-zero differenced forecast was simply being
            # added back to last week's load.
            simple_differencing=False,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=100, method="lbfgs")


def _forecast_with_intervals(
    history: pd.Series,
    cfg: SarimaConfig,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Return the point forecast and analytic quantiles, in megawatts."""
    window = history.iloc[-cfg.window_weeks * WEEK :]

    if cfg.variant == "weekly-diff":
        # Remove the weekly cycle by hand before fitting.
        target = window.to_numpy(dtype=float)[WEEK:] - window.to_numpy(dtype=float)[:-WEEK]
        offset = history.iloc[-WEEK:].to_numpy(dtype=float)
        if horizon != WEEK:
            raise ValueError("the weekly-diff variant is written for a 168-hour horizon")
    else:
        target = window.to_numpy(dtype=float)
        offset = np.zeros(horizon)

    fitted = _fit(target, cfg)
    result = fitted.get_forecast(steps=horizon)
    mean = np.asarray(result.predicted_mean, dtype=float) + offset

    quantiles: dict[float, np.ndarray] = {}
    for q_lo, q_hi, alpha in ((0.025, 0.975, 0.05), (0.1, 0.9, 0.20)):
        band = np.asarray(result.conf_int(alpha=alpha), dtype=float)
        quantiles[q_lo] = band[:, 0] + offset
        quantiles[q_hi] = band[:, 1] + offset
    quantiles[0.5] = mean  # Gaussian state space: the mean is the median
    return mean, quantiles


def run(
    series: pd.Series,
    origins: Sequence[pd.Timestamp] = VALIDATION_ORIGINS,
    candidates: Sequence[SarimaConfig] = CANDIDATES,
) -> ModelForecast:
    started = time.perf_counter()

    selection: list[dict[str, object]] = []
    best: tuple[float, SarimaConfig] | None = None
    for cfg in candidates:
        scores: list[float] = []
        failed = ""
        for origin in origins:
            history = history_before(series, origin)
            try:
                mean, _ = _forecast_with_intervals(history, cfg)
            except Exception as exc:  # a bad order can fail to converge
                failed = f"{type(exc).__name__}: {exc}"
                break
            scores.append(mape(series.loc[horizon_index(origin)].to_numpy(), mean))
        if failed:
            selection.append({**cfg.as_dict(), "val_mape_pct": None, "error": failed})
            continue
        mean_score = float(np.mean(scores))
        selection.append(
            {
                **cfg.as_dict(),
                "val_mape_pct": mean_score,
                "per_origin_mape_pct": [round(s, 3) for s in scores],
            }
        )
        if best is None or mean_score < best[0]:
            best = (mean_score, cfg)

    if best is None:
        raise RuntimeError("every SARIMA candidate failed to fit")
    val_mape, best_cfg = best

    # Refit the winner on train + validation and forecast the test week.
    history = history_before(series, TEST_START)
    mean, quantile_arrays = _forecast_with_intervals(history, best_cfg)
    index = horizon_index(TEST_START)

    return ModelForecast(
        name="sarima",
        point=pd.Series(mean, index=index, name="sarima"),
        quantiles={q: pd.Series(v, index=index) for q, v in quantile_arrays.items()},
        hyperparameters={
            **best_cfg.as_dict(),
            "n_training_hours_final_fit": int(best_cfg.window_weeks * WEEK),
            "intervals": "analytic Gaussian from the state-space model",
        },
        runtime_seconds=time.perf_counter() - started,
        validation_mape=val_mape,
        selection_log=selection,
    )
