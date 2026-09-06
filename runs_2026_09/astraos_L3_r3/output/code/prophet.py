from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from darts.models import Prophet

from .common import QUANTILES, SEED, TRAIN_END, VALIDATION_START, ForecastResult, as_darts, mape

CANDIDATES = [0.01, 0.05]


def make_model(changepoint_prior_scale: float) -> Prophet:
    return Prophet(
        country_holidays="DE",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=changepoint_prior_scale,
        holidays_prior_scale=10.0,
        seasonality_prior_scale=10.0,
        uncertainty_samples=0,
        random_state=SEED,
    )


def run(pretest: pd.Series) -> ForecastResult:
    start = perf_counter()
    train, validation = pretest.loc[:TRAIN_END], pretest.loc[VALIDATION_START:]
    records = []
    for prior in CANDIDATES:
        model = make_model(prior)
        model.fit(as_darts(train))
        # Prophet is a calendar regression, so new load context cannot update a fixed fit.
        prediction = model.predict(len(validation)).univariate_values()
        value = mape(validation.to_numpy(), prediction)
        records.append({"changepoint_prior_scale": prior, "mape_validation_pct": value})
        print(f"prophet validation prior={prior}: {value:.4f}%", flush=True)
    selected = min(records, key=lambda record: record["mape_validation_pct"])[
        "changepoint_prior_scale"
    ]
    final = make_model(selected)
    final.fit(as_darts(pretest))
    point = final.predict(168).univariate_values()
    samples = final.predict(168, num_samples=1000).all_values()[:, 0, :]
    quantiles = np.quantile(samples, QUANTILES, axis=1).T
    daily_dates = pd.date_range("2015-01-01 12:00", "2019-12-31 12:00", freq="D")
    daily = final.model.predict(pd.DataFrame({"ds": daily_dates}))
    week_dates = pd.date_range("2019-01-07", periods=168, freq="h")
    weekly = final.model.predict(pd.DataFrame({"ds": week_dates}))
    return ForecastResult(
        name="prophet",
        point=point,
        quantiles=quantiles,
        runtime_seconds=perf_counter() - start,
        hyperparameters={
            "country_holidays": "DE",
            "daily_seasonality": True,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
            "seasonality_mode": "additive",
            "changepoint_prior_scale": selected,
            "holidays_prior_scale": 10.0,
            "seasonality_prior_scale": 10.0,
            "posterior_predictive_samples": 1000,
            "seed": SEED,
            "mcmc_samples": 0,
            "implementation": "darts.models.Prophet",
        },
        validation=records,
        diagnostics={
            "daily_components": {
                "dates": daily_dates.strftime("%Y-%m-%d").tolist(),
                "trend": daily.trend.tolist(),
                "yearly": daily.yearly.tolist(),
            },
            "weekly_components": {
                "hours_since_monday": list(range(168)),
                "weekly": weekly.weekly.tolist(),
            },
            "interval_caveat": "MAP parameter fit; predictive noise and simulated trend changes, not full posterior parameter uncertainty.",
        },
    )
