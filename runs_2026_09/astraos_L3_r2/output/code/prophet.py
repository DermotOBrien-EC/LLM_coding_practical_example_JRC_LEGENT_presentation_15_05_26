from __future__ import annotations

from time import perf_counter
from typing import Any
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet
from .common import ModelResult, QUANTILES, ROOT, SEED, TRAIN_END, mape


def fit(series: pd.Series, config: dict[str, Any]) -> Prophet:
    model = Prophet(
        country_holidays="DE",
        yearly_seasonality=10,
        seasonality_mode="additive",
        uncertainty_samples=1000,
        suppress_stdout_stderror=True,
        random_state=SEED,
        **config,
    )
    model.fit(TimeSeries.from_series(series))
    return model


def run(pretest: pd.Series) -> ModelResult:
    start = perf_counter()
    train = pretest.loc[: TRAIN_END - pd.Timedelta(hours=1)]
    val = pretest.loc[TRAIN_END:]
    candidates = [
        {"changepoint_prior_scale": 0.01, "daily_seasonality": 4, "weekly_seasonality": 3},
        {"changepoint_prior_scale": 0.05, "daily_seasonality": 10, "weekly_seasonality": 6},
    ]
    validation: list[dict[str, Any]] = []
    for config in candidates:
        model = fit(train, config)
        # Prophet has no recent-load state: all origins use the same fitted calendar curve.
        prediction = model.predict(len(val)).univariate_values()
        validation.append({"configuration": config, "mape_pct": mape(val.to_numpy(), prediction)})
    best = min(validation, key=lambda item: item["mape_pct"])["configuration"]
    model = fit(pretest, best)
    point = model.predict(168).univariate_values()
    samples = model.predict(168, num_samples=1000, random_state=SEED).all_values()[:, 0, :]
    quantiles = np.quantile(samples, QUANTILES, axis=1).T
    # Save component curves as numbers so plotting never requires a second fit.
    (ROOT / "artifacts").mkdir(exist_ok=True)
    for name, index in {
        "trend": pd.date_range("2015-01-01", "2020-01-07", freq="D"),
        "weekly": pd.date_range("2020-01-06", periods=168, freq="h"),
        "yearly": pd.date_range("2019-01-01", periods=365, freq="D"),
    }.items():
        raw = model.model.predict(pd.DataFrame({"ds": index}))
        raw[["ds", name]].to_csv(ROOT / "artifacts" / f"prophet_{name}.csv", index=False)
    return ModelResult(
        "prophet",
        point,
        quantiles,
        perf_counter() - start,
        {
            **best,
            "yearly_seasonality": 10,
            "country_holidays": "DE",
            "seasonality_mode": "additive",
            "uncertainty_samples": 1000,
            "random_state": SEED,
            "mcmc_samples": 0,
        },
        validation,
    )
