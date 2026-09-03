"""Prophet: an additive curve-fitting model with German public holidays.

Prophet describes the load as a sum of pieces that a human can read off a
chart: a slowly moving trend, a repeating daily shape, a repeating weekly
shape, a repeating yearly shape, and a bump (usually a dip, for electricity)
on public holidays. It never looks at recent load values, only at the date
and time, which makes it a pure "calendar model". That is its strength on
1 January and its weakness on an ordinary Tuesday, where knowing yesterday's
load would help a lot.

Two things are chosen on the validation window:

- `changepoint_prior_scale`: how freely the trend is allowed to bend. Too
  large and it chases noise; too small and it cannot follow the slow decline
  in German load over these years.
- `seasonality_mode`: whether the daily and weekly swings are added to the
  trend in megawatts (additive) or scale with its level (multiplicative).

Because Prophet's forecast is a function of the timestamp alone, forecasting
the validation quarter as one block gives exactly the same numbers as
forecasting it in thirteen 168-hour blocks without refitting. We therefore
score the whole block once, which is cheaper and identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file has to be called prophet.py (one file per model), but that name
# also belongs to the pip-installed Prophet package. If this directory stays
# first on the import path, darts looks for Prophet and finds this file
# instead. Pushing the directory to the end fixes that while still letting us
# import common. The orchestrator loads this file under a different module
# name for the same reason.
_HERE = Path(__file__).resolve().parent


def _points_here(entry: str) -> bool:
    try:
        return Path(entry or ".").resolve() == _HERE
    except OSError:
        return False


sys.path[:] = [entry for entry in sys.path if not _points_here(entry)] + [str(_HERE)]

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet

import common as c

# Prophet's Stan backend is chatty; the messages carry no information here.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

PARAM_GRID: list[dict[str, Any]] = [
    {"changepoint_prior_scale": cps, "seasonality_mode": mode}
    for cps in (0.01, 0.05, 0.5)
    for mode in ("additive", "multiplicative")
]

FIXED_KWARGS: dict[str, Any] = {
    "country_holidays": "DE",
    "daily_seasonality": True,
    "weekly_seasonality": True,
    "yearly_seasonality": True,
}


def _to_darts(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(c.to_naive_index(series))


def _build(params: dict[str, Any]) -> Prophet:
    return Prophet(**FIXED_KWARGS, **params)


def select_hyperparameters(series: pd.Series) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    """Fit each candidate on train only and score it on the validation quarter."""
    train = _to_darts(series.loc[: c.TRAIN_END])
    actual = series.loc[c.VAL_START : c.VAL_END]

    trace: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None

    for params in PARAM_GRID:
        model = _build(params)
        model.fit(train)
        prediction = model.predict(n=len(actual)).to_series()
        score = c.mape(actual.to_numpy(), prediction.to_numpy())
        trace.append({**params, "validation_mape_pct": score})
        if best is None or score < best[0]:
            best = (score, params)

    assert best is not None
    return best[1], best[0], trace


def run(series: pd.Series) -> c.ForecastResult:
    """Select on validation, refit on train+validation, forecast the test week."""
    start = time.time()

    best_params, best_score, trace = select_hyperparameters(series)

    fitting_data = _to_darts(c.train_plus_val(series))
    model = _build(best_params)
    model.fit(fitting_data)

    test = series.loc[c.TEST_START : c.TEST_END]

    point = model.predict(n=c.HORIZON).to_series()
    point.index = test.index
    point.name = "prophet"

    # Prophet's uncertainty comes from simulating future trend paths plus
    # observation noise, so we draw samples and read the quantiles off them.
    samples = model.predict(n=c.HORIZON, num_samples=1000).all_values()[:, 0, :]
    quantiles = c.quantile_frame(test.index, samples)

    # Components for the decomposition figure, taken from the fitted
    # underlying Prophet object over the whole fitting period plus the test
    # week, so the reader can see trend, weekly and yearly shapes.
    inner = model.model
    future = inner.make_future_dataframe(periods=c.HORIZON, freq="h", include_history=True)
    components = inner.predict(future)
    keep = [col for col in ("ds", "trend", "weekly", "yearly", "daily", "holidays") if col in components]

    return c.ForecastResult(
        name="prophet",
        point_forecast=point,
        runtime_seconds=time.time() - start,
        hyperparameters={**best_params, **FIXED_KWARGS, "n_samples_for_intervals": 1000},
        quantile_forecast=quantiles,
        validation_mape_pct=best_score,
        extras={"components": components[keep], "grid_search": trace},
    )


if __name__ == "__main__":
    result = run(c.load_load_series())
    result.save()
    print(result.name, result.validation_mape_pct, result.runtime_seconds)
