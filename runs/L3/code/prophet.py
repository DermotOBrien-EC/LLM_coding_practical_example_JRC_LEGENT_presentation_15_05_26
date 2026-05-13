"""Prophet forecaster with German national holidays.

Plain language: Prophet decomposes the load into a smooth trend, periodic
seasonalities (daily, weekly, yearly), and a holiday effect for the dates
we hand it. The seasonalities and trend are estimated by Bayesian MAP
optimisation under the hood; we pull samples from the posterior to get
prediction intervals.

We use darts's wrapper so the data exchange with the rest of the
orchestrator is consistent. The wrapper exposes `country_holidays="DE"`,
which loads the same federal holiday set we use elsewhere.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet as DartsProphet

from code.common import N_TEST, ModelForecast, mape

# Prophet is chatty. Silence the cmdstanpy / Prophet INFO logs so the
# orchestrator's stdout stays readable.
for name in ("cmdstanpy", "prophet", "prophet.plot"):
    logging.getLogger(name).setLevel(logging.ERROR)


CANDIDATE_PRIORS: tuple[tuple[float, float], ...] = (
    # (changepoint_prior_scale, seasonality_prior_scale)
    (0.05, 10.0),    # Prophet defaults — trend reasonably smooth
    (0.1, 10.0),     # slightly more flexible trend
    (0.05, 20.0),    # stronger seasonality
)


def _fit_prophet(series: pd.Series, *, changepoint: float, seasonality: float) -> DartsProphet:
    ts = TimeSeries.from_series(series)
    model = DartsProphet(
        country_holidays="DE",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=changepoint,
        seasonality_prior_scale=seasonality,
    )
    model.fit(ts)
    return model


def run_prophet(train: pd.Series, val: pd.Series, trainval: pd.Series) -> ModelForecast:
    """Pick priors on validation, refit on train+val, forecast the test week."""
    started = time.perf_counter()

    val_mapes: dict[str, float] = {}
    best = None
    best_mape = float("inf")
    for cp, sp in CANDIDATE_PRIORS:
        m = _fit_prophet(train, changepoint=cp, seasonality=sp)
        f = m.predict(len(val))
        m_mape = mape(val.to_numpy(), f.values().ravel())
        key = f"cp={cp}, sp={sp}"
        val_mapes[key] = m_mape
        if m_mape < best_mape:
            best_mape = m_mape
            best = (cp, sp)

    cp, sp = best  # type: ignore[misc]
    final = _fit_prophet(trainval, changepoint=cp, seasonality=sp)
    pred = final.predict(N_TEST, num_samples=500)
    samples = pred.all_values(copy=False)[:, 0, :]  # (168, n_samples)
    point = samples.mean(axis=1)
    q10 = np.quantile(samples, 0.10, axis=1)
    q50 = np.quantile(samples, 0.50, axis=1)
    q90 = np.quantile(samples, 0.90, axis=1)
    lower80 = q10
    upper80 = q90
    lower95 = np.quantile(samples, 0.025, axis=1)
    upper95 = np.quantile(samples, 0.975, axis=1)

    return ModelForecast(
        name="prophet",
        point=point,
        q10=q10,
        q50=q50,
        q90=q90,
        lower80=lower80,
        upper80=upper80,
        lower95=lower95,
        upper95=upper95,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters={
            "changepoint_prior_scale": cp,
            "seasonality_prior_scale": sp,
            "country_holidays": "DE",
            "validation_mape_by_priors": val_mapes,
            "selected_validation_mape": best_mape,
            "num_samples": 500,
        },
    )
