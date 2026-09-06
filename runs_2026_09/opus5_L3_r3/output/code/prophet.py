"""Prophet: an additive curve-fitting model with explicit holiday effects.

Prophet does not look at the recent past at all. It fits a slowly bending
trend plus a set of repeating shapes (a daily shape, a weekly shape, a yearly
shape) plus a bump for each public holiday, and then evaluates that formula
at the timestamps you ask about. That makes it the natural candidate for the
one question this test window is built around: does the model know that
1 January is not a normal Wednesday?

Hyperparameters swept on the validation weeks:

- `changepoint_prior_scale` - how freely the trend is allowed to bend. Too
  small and the model cannot follow a slow drift in demand; too large and it
  chases noise and extrapolates a spurious slope into the forecast.
- `seasonality_mode` - whether the daily and weekly shapes are added to the
  trend (additive) or multiply it (multiplicative).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

# Import order matters on the next two lines. Importing `common` first runs a
# guard that puts the real `prophet` package into Python's module table, so
# that when darts reaches for it below it does not pick up this file instead.
from common import (  # isort: skip
    HORIZON,
    QUANTILE_LEVELS,
    SEED,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
    ForecastResult,
    from_darts_quantiles,
    mape,
    to_darts,
    validation_origins,
)
from darts.models import Prophet as DartsProphet  # isort: skip  (see note above)

CANDIDATE_CHANGEPOINT_PRIOR: tuple[float, ...] = (0.01, 0.05, 0.5)
CANDIDATE_SEASONALITY_MODE: tuple[str, ...] = ("additive", "multiplicative")

N_SAMPLES: int = 500  # posterior draws used to build the prediction intervals


def _build(changepoint_prior_scale: float, seasonality_mode: str) -> DartsProphet:
    """One Prophet with German federal holidays and all three seasonalities."""
    return DartsProphet(
        country_holidays="DE",
        random_state=SEED,
        suppress_stdout_stderror=True,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        uncertainty_samples=N_SAMPLES,
    )


def _select(series: pd.Series) -> tuple[float, str, list[dict[str, Any]]]:
    """Pick the configuration with the lowest mean MAPE over the validation weeks.

    Because Prophet is not autoregressive, one fit on the training data can
    be evaluated on every validation week at once: predicting hour 2000 of
    the validation window does not require having seen hour 1999. The scoring
    is still done week by week so that it matches how the model is judged on
    the test window.
    """
    train = to_darts(series.loc[:TRAIN_END])
    n_val = len(series.loc[VAL_START:VAL_END])
    origins = validation_origins()
    log: list[dict[str, Any]] = []

    for prior in CANDIDATE_CHANGEPOINT_PRIOR:
        for mode in CANDIDATE_SEASONALITY_MODE:
            t0 = time.perf_counter()
            model = _build(prior, mode)
            model.fit(train)
            pred = model.predict(n_val, num_samples=1)
            predicted = pd.Series(
                pred.values(copy=False).ravel().astype("float64"),
                index=series.loc[VAL_START:VAL_END].index,
            )
            scores = [
                mape(
                    series.loc[o : o + pd.Timedelta(hours=HORIZON - 1)].to_numpy(),
                    predicted.loc[o : o + pd.Timedelta(hours=HORIZON - 1)].to_numpy(),
                )
                for o in origins
            ]
            log.append(
                {
                    "changepoint_prior_scale": prior,
                    "seasonality_mode": mode,
                    "val_mape_pct": float(np.mean(scores)),
                    "fit_seconds": round(time.perf_counter() - t0, 1),
                }
            )

    best = min(log, key=lambda row: row["val_mape_pct"])
    return float(best["changepoint_prior_scale"]), str(best["seasonality_mode"]), log


def run(series: pd.Series) -> ForecastResult:
    """Select on validation, refit on train+validation, forecast the test week."""
    start = time.perf_counter()

    prior, mode, log = _select(series)

    model = _build(prior, mode)
    model.fit(to_darts(series.loc[:VAL_END]))

    test_index = series.loc[TEST_START:TEST_END].index
    sampled = model.predict(HORIZON, num_samples=N_SAMPLES)
    quantiles = from_darts_quantiles(sampled, test_index, QUANTILE_LEVELS)

    # The point forecast is Prophet's own deterministic curve (`yhat`), not
    # the median of the draws, so that it matches what a user of the library
    # would report.
    deterministic = model.predict(HORIZON, num_samples=1)
    point = pd.Series(deterministic.values(copy=False).ravel().astype("float64"), index=test_index)

    return ForecastResult(
        name="prophet",
        point=point,
        quantiles=quantiles,
        hyperparameters={
            "changepoint_prior_scale": prior,
            "seasonality_mode": mode,
            "country_holidays": "DE",
            "seasonalities": ["daily", "weekly", "yearly"],
            "uncertainty_samples": N_SAMPLES,
            "selection_metric": "mean MAPE over 13 held-out validation weeks",
            "interval_method": "quantiles of Prophet posterior draws",
        },
        runtime_seconds=time.perf_counter() - start,
        selection_log=log,
        extra={"components": _decompose(model, series)},
    )


def _decompose(model: DartsProphet, series: pd.Series) -> dict[str, pd.DataFrame]:
    """Evaluate the fitted Prophet's own components so they can be plotted later.

    Done here rather than in the plotting code because each model runs in its
    own process (LightGBM and PyTorch cannot share one), and a fitted Prophet
    object does not need to survive that boundary. Three small tables do.
    """
    underlying = model.model
    history = series.loc[:VAL_END]
    naive_index = history.index.tz_convert("UTC").tz_localize(None)

    frames: dict[str, pd.DataFrame] = {}
    for key, dates in (
        ("trend", pd.date_range(naive_index[0], naive_index[-1], freq="D")),
        ("weekly", pd.date_range("2019-01-07", periods=168, freq="h")),  # a Monday
        ("yearly", pd.date_range("2019-01-01", periods=365, freq="D")),
        ("holiday", pd.date_range("2019-12-20", "2020-01-10 23:00", freq="h")),
    ):
        predicted = underlying.predict(pd.DataFrame({"ds": dates}))
        frames[key] = predicted[["ds", "trend", "weekly", "yearly", "daily", "holidays"]].copy()
    return frames
