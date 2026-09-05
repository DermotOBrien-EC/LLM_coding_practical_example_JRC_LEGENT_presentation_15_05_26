"""Prophet with German public holidays.

Prophet writes the load as a sum of a slowly bending trend, a yearly wave,
a weekly wave, a daily wave and a bump for each public holiday. The waves
are sums of a few sines and cosines, so the model is small, fast and easy
to read. Its known weakness on electricity data is that a single daily
wave cannot look different on a Saturday and on a Tuesday; the candidate
configurations therefore include a version with separate weekday and
weekend daily waves (Prophet's "conditional seasonality"), driven by a
0/1 flag derived from the timestamp.

Everything runs through the darts wrapper, as required, with
`country_holidays="DE"` supplying the holiday calendar.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import common  # noqa: F401  (must come before darts: it loads the real prophet package)
from darts import TimeSeries
from darts.models import Prophet

from common import (
    HORIZON,
    QUANTILES,
    SEED,
    ForecastResult,
    Stopwatch,
    block_mape,
    calendar_features,
    log,
    sort_quantiles,
    to_darts,
    validation_blocks,
)

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

N_SAMPLES: int = 1000

CONDITIONAL_DAILY: list[dict[str, Any]] = [
    {
        "name": "daily_weekday",
        "seasonal_periods": 24,
        "fourier_order": 10,
        "condition_name": "is_weekday",
    },
    {
        "name": "daily_weekend",
        "seasonal_periods": 24,
        "fourier_order": 10,
        "condition_name": "is_weekend",
    },
]
CANDIDATES: dict[str, dict[str, Any]] = {
    "prophet_defaults": {
        "add_seasonalities": None,
        "kwargs": {"seasonality_mode": "additive", "changepoint_prior_scale": 0.05},
    },
    "conditional_daily_additive": {
        "add_seasonalities": CONDITIONAL_DAILY,
        "kwargs": {
            "daily_seasonality": False,
            "weekly_seasonality": 6,
            "yearly_seasonality": 10,
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.05,
        },
    },
    "conditional_daily_multiplicative": {
        "add_seasonalities": CONDITIONAL_DAILY,
        "kwargs": {
            "daily_seasonality": False,
            "weekly_seasonality": 6,
            "yearly_seasonality": 10,
            "seasonality_mode": "multiplicative",
            "changepoint_prior_scale": 0.05,
        },
    },
    "conditional_daily_additive_stiff_trend": {
        "add_seasonalities": CONDITIONAL_DAILY,
        "kwargs": {
            "daily_seasonality": False,
            "weekly_seasonality": 6,
            "yearly_seasonality": 10,
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.01,
        },
    },
}


def condition_covariates(index: pd.DatetimeIndex) -> TimeSeries:
    """0/1 weekday and weekend flags that switch the conditional daily waves."""
    weekend = calendar_features(index)["is_weekend"].to_numpy()
    frame = pd.DataFrame(
        {"is_weekday": (weekend == 0).astype(float), "is_weekend": (weekend == 1).astype(float)},
        index=index,
    )
    return TimeSeries.from_dataframe(frame, freq="h")


def build(name: str) -> Prophet:
    cfg = CANDIDATES[name]
    return Prophet(
        add_seasonalities=cfg["add_seasonalities"],
        country_holidays="DE",
        random_state=SEED,
        **cfg["kwargs"],
    )


def components(model: Prophet, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Trend, seasonal and holiday parts of the fitted model over `index`."""
    weekend = calendar_features(index)["is_weekend"].to_numpy()
    future = pd.DataFrame({"ds": index, "is_weekday": weekend == 0, "is_weekend": weekend == 1})
    out = model.model.predict(future)
    out.index = index
    keep = [
        c
        for c in out.columns
        if (c in ("trend", "weekly", "yearly", "holidays", "daily") or c.startswith("daily_"))
        and not c.endswith(("_lower", "_upper"))
    ]
    return out[keep]


def run(train: pd.Series, val: pd.Series, test_index: pd.DatetimeIndex) -> ForecastResult:
    with Stopwatch() as sw:
        history = pd.concat([train, val])
        full_index = pd.date_range(train.index[0], test_index[-1], freq="h")
        covariates = condition_covariates(full_index)
        blocks = validation_blocks(val.index)

        # Stage 1: each candidate is fitted on the training window and
        # scored on the 13 validation weeks. Prophet's forecast is a pure
        # function of the calendar, so one long forecast over the
        # validation window equals the 13 separate week-ahead forecasts.
        sweep_results: list[dict[str, Any]] = []
        for name in CANDIDATES:
            model = build(name)
            model.fit(to_darts(train), future_covariates=covariates)
            fc = model.predict(n=len(val), future_covariates=covariates)
            fc_series = pd.Series(fc.values().ravel(), index=val.index)
            val_mape = block_mape(val, [fc_series.loc[block] for block in blocks])
            sweep_results.append(
                {
                    "candidate": name,
                    **CANDIDATES[name]["kwargs"],
                    "val_mape_pct": round(val_mape, 4),
                }
            )
            log(f"prophet sweep {name}: val MAPE {val_mape:.3f} %")
        best = min(sweep_results, key=lambda r: r["val_mape_pct"])
        chosen = str(best["candidate"])

        # Stage 2: refit on train + validation, forecast the test week with
        # a point forecast and a cloud of sampled futures for the intervals.
        model = build(chosen)
        model.fit(to_darts(history), future_covariates=covariates)
        point_ts = model.predict(n=HORIZON, future_covariates=covariates)
        point = pd.Series(point_ts.values().ravel(), index=test_index)
        sample_ts = model.predict(n=HORIZON, future_covariates=covariates, num_samples=N_SAMPLES)
        samples = sample_ts.all_values()[:, 0, :]
        quantiles = sort_quantiles(
            {q: pd.Series(np.quantile(samples, q, axis=1), index=test_index) for q in QUANTILES}
        )
        comps = components(model, full_index)

    return ForecastResult(
        name="prophet",
        point=point,
        quantiles=quantiles,
        runtime_seconds=sw.seconds,
        hyperparameters={
            "candidate": chosen,
            **CANDIDATES[chosen]["kwargs"],
            "conditional_daily_seasonality": CANDIDATES[chosen]["add_seasonalities"] is not None,
            "country_holidays": "DE",
            "growth": "linear",
            "n_samples_for_intervals": N_SAMPLES,
        },
        validation={"sweep": sweep_results, "chosen": chosen, "val_mape_pct": best["val_mape_pct"]},
        extras={"components": comps},
    )
