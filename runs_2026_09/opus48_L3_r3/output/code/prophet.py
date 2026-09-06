"""Prophet: an additive curve-fitting forecaster with holidays.

Prophet explains a series as a sum of smooth pieces: a slow trend plus repeating
daily, weekly and yearly shapes, plus bumps on named holidays. It is designed to
be robust and to handle calendar effects out of the box, which is exactly why it
is interesting here: it is the only model in the study that is explicitly told
which days are German public holidays, so New Year's Day should not surprise it.

We let Prophet learn daily, weekly and yearly seasonality, and we hand it the
German nationwide holiday calendar. The one thing we tune on the validation week
is whether the seasonal effects add to the trend (additive) or scale it
(multiplicative).

Prediction intervals come from Prophet's own uncertainty sampling.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet

import common as c

# The two seasonality modes we compare on the validation week.
SEASONALITY_MODES: list[str] = ["additive", "multiplicative"]

# How many sample paths to draw for the prediction intervals.
N_SAMPLES: int = 500


def _to_ts(series: pd.Series) -> TimeSeries:
    """Wrap a pandas Series as a darts TimeSeries with an explicit hourly step."""
    return TimeSeries.from_series(series.asfreq("h"), freq="h")


def _build(mode: str) -> Prophet:
    """Create a Prophet model with German holidays and all three seasonalities."""
    return Prophet(
        country_holidays="DE",
        seasonality_mode=mode,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )


def run(series: pd.Series) -> c.ModelResult:
    """Select the seasonality mode on validation, refit on train+val, forecast."""
    start = time.perf_counter()

    train_ts = _to_ts(c.train_series(series))
    val_target = c.val_series(series).iloc[: c.HORIZON].to_numpy()

    # --- Step 1: choose additive vs multiplicative by validation error. -----
    val_mapes: dict[str, float] = {}
    for mode in SEASONALITY_MODES:
        model = _build(mode)
        model.fit(train_ts)
        fc = model.predict(n=c.HORIZON).values().flatten()
        val_mapes[mode] = c.mape(val_target, fc)

    best_mode = min(val_mapes, key=val_mapes.get)

    # --- Step 2: refit chosen mode on train+val. ----------------------------
    tv_ts = _to_ts(c.train_plus_val_series(series))
    final = _build(best_mode)
    final.fit(tv_ts)

    # --- Step 3: forecast the test week, point and interval. ----------------
    point = final.predict(n=c.HORIZON).values().flatten().astype(float)
    prob = final.predict(n=c.HORIZON, num_samples=N_SAMPLES)
    quantiles = {
        q: prob.quantile(q).values().flatten().astype(float)
        for q in c.WINNER_QUANTILES
    }

    runtime = time.perf_counter() - start
    return c.ModelResult(
        name="prophet",
        point=point,
        runtime_seconds=runtime,
        hyperparameters={
            "seasonality_mode": best_mode,
            "country_holidays": "DE",
            "seasonalities": ["daily", "weekly", "yearly"],
            "num_samples": N_SAMPLES,
            "selection_metric": "validation_mape_pct",
            "validation_mape_by_mode": {k: round(v, 3) for k, v in val_mapes.items()},
        },
        quantiles=quantiles,
        extra={"model": final},
    )
