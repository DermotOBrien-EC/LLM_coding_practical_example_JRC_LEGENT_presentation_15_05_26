"""Prophet with German public holidays.

Prophet takes a completely different view from SARIMA. Instead of modelling
how each hour depends on the previous hours, it decomposes the series into
pieces you can name: a slowly bending trend, a daily shape, a weekly shape, a
yearly shape, and a bump (or dip) on each public holiday. Adding
`country_holidays="DE"` is what makes it the natural candidate for beating
everything else on 1 January - it has an explicit "this is a holiday" term
fitted from the five previous New Year's Days in the training data.

The knobs we tune on the validation set are how flexible the trend is allowed
to be (`changepoint_prior_scale`), how strong the seasonal terms may get
(`seasonality_prior_scale`), and whether the seasonal effects multiply the
trend or add to it. Load is roughly a fixed shape scaled up and down, so the
multiplicative option is a genuine contender rather than a formality.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet as DartsProphet

from common import (
    HORIZON,
    SEED,
    TEST_START,
    VALIDATION_ORIGINS,
    ModelForecast,
    history_before,
    horizon_index,
    mape,
)

QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.5, 0.9, 0.975)
N_SAMPLES: int = 500

for noisy in ("prophet", "cmdstanpy", "prophet.models"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


CANDIDATES: tuple[dict[str, object], ...] = (
    {"seasonality_mode": "additive", "changepoint_prior_scale": 0.05,
     "seasonality_prior_scale": 10.0},
    {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.05,
     "seasonality_prior_scale": 10.0},
    {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.01,
     "seasonality_prior_scale": 10.0},
    {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.05,
     "seasonality_prior_scale": 25.0},
)


def _to_darts(history: pd.Series) -> TimeSeries:
    """Prophet wants naive timestamps; the series is UTC throughout anyway."""
    return TimeSeries.from_series(history.tz_localize(None), freq="h")


def _build(params: dict[str, object]) -> DartsProphet:
    return DartsProphet(
        country_holidays="DE",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        **params,
    )


def _fit_and_forecast(
    history: pd.Series,
    params: dict[str, object],
    horizon: int = HORIZON,
    num_samples: int = 1,
) -> tuple[np.ndarray, dict[float, np.ndarray] | None, DartsProphet]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build(params)
        model.fit(_to_darts(history))
        if num_samples <= 1:
            forecast = model.predict(n=horizon)
            return forecast.values().ravel().astype(float), None, model
        forecast = model.predict(n=horizon, num_samples=num_samples)
    samples = forecast.all_values()[:, 0, :]  # shape (horizon, num_samples)
    point = samples.mean(axis=1).astype(float)
    quantiles = {q: np.quantile(samples, q, axis=1).astype(float) for q in QUANTILES}
    return point, quantiles, model


def decomposition(model: DartsProphet, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Trend / weekly / yearly / daily / holiday parts over a given period."""
    inner = model.model
    future = pd.DataFrame({"ds": index.tz_localize(None)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parts = inner.predict(future)
    keep = [c for c in ("trend", "daily", "weekly", "yearly", "holidays") if c in parts]
    out = parts[keep].copy()
    out.index = index
    return out


def run(
    series: pd.Series,
    origins: Sequence[pd.Timestamp] = VALIDATION_ORIGINS,
    candidates: Sequence[dict[str, object]] = CANDIDATES,
) -> tuple[ModelForecast, dict[str, object]]:
    started = time.perf_counter()

    selection: list[dict[str, object]] = []
    best: tuple[float, dict[str, object]] | None = None
    for params in candidates:
        scores: list[float] = []
        for origin in origins:
            point, _, _ = _fit_and_forecast(history_before(series, origin), params)
            scores.append(mape(series.loc[horizon_index(origin)].to_numpy(), point))
        mean_score = float(np.mean(scores))
        selection.append(
            {
                **params,
                "val_mape_pct": mean_score,
                "per_origin_mape_pct": [round(s, 3) for s in scores],
            }
        )
        if best is None or mean_score < best[0]:
            best = (mean_score, dict(params))

    assert best is not None
    val_mape, best_params = best

    np.random.seed(SEED)
    history = history_before(series, TEST_START)
    point, quantile_arrays, model = _fit_and_forecast(
        history, best_params, num_samples=N_SAMPLES
    )
    index = horizon_index(TEST_START)

    # A full year of components for the decomposition figure.
    comp_index = pd.date_range("2019-01-01", "2019-12-31 23:00", freq="h", tz="UTC")
    components = decomposition(model, comp_index)

    assert quantile_arrays is not None
    return (
        ModelForecast(
            name="prophet",
            point=pd.Series(point, index=index, name="prophet"),
            quantiles={
                q: pd.Series(v, index=index) for q, v in quantile_arrays.items()
            },
            hyperparameters={
                **best_params,
                "country_holidays": "DE",
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": True,
                "n_uncertainty_samples": N_SAMPLES,
            },
            runtime_seconds=time.perf_counter() - started,
            validation_mape=val_mape,
            selection_log=selection,
        ),
        {"components": components},
    )
