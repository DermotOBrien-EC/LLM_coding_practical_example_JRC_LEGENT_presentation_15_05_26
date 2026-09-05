"""SARIMA: the classical statistical benchmark.

A seasonal ARIMA describes each hour's load as a weighted combination of
recent hours, recent forecast errors, and the same hour on previous days
(daily season, s = 24). A weekly season (s = 168) is not practical in the
state-space form, so the weekly rhythm is offered to the model as a small
set of sine/cosine regressors, and public holidays as a 0/1 regressor.
Both are functions of the timestamp only. Whether the model uses them at
all is decided on the validation window, like the ARIMA orders.

Fitting is done on the most recent few weeks of history (four or eight,
chosen on the validation window) rather than all 4.75 years: exact
maximum likelihood on 41,000 hourly points with a daily season takes tens
of minutes per fit, and an ARIMA has short memory anyway, so distant
history adds cost but almost no information.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from common import (
    HORIZON,
    QUANTILES,
    ForecastResult,
    Stopwatch,
    block_mape,
    calendar_features,
    log,
    sort_quantiles,
    validation_blocks,
)

WEEKLY_HARMONICS: int = 3
PHASE_ORIGIN = pd.Timestamp("2015-01-05 00:00")  # a Monday, fixes the phase of the weekly waves

SWEEP: dict[str, list[Any]] = {
    "window_weeks": [4, 8],
    "regressors": ["none", "weekly", "weekly+holiday"],
    "order": [(1, 0, 1), (2, 0, 1), (1, 1, 1)],
    "seasonal_order": [(0, 1, 1, 24), (1, 1, 1, 24)],
}


def regressors(index: pd.DatetimeIndex, kind: str) -> pd.DataFrame | None:
    """Weekly sine/cosine waves and a holiday flag for the given hours."""
    if kind == "none":
        return None
    hours = ((index - PHASE_ORIGIN).total_seconds() / 3600.0).to_numpy()
    cols: dict[str, np.ndarray] = {}
    for k in range(1, WEEKLY_HARMONICS + 1):
        cols[f"weekly_sin{k}"] = np.sin(2.0 * np.pi * k * hours / 168.0)
        cols[f"weekly_cos{k}"] = np.cos(2.0 * np.pi * k * hours / 168.0)
    if kind == "weekly+holiday":
        cols["holiday"] = calendar_features(index)["is_public_holiday_de"].to_numpy(dtype=float)
    return pd.DataFrame(cols, index=index)


def fit(y: pd.Series, config: dict[str, Any]) -> SARIMAXResults:
    """Estimate the model on the last `window_weeks` weeks of `y`."""
    window = y.iloc[-HORIZON * config["window_weeks"] :]
    model = SARIMAX(
        window,
        exog=regressors(window.index, config["regressors"]),
        order=config["order"],
        seasonal_order=config["seasonal_order"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=200)


def forecast_from(
    fitted: SARIMAXResults, history: pd.Series, index: pd.DatetimeIndex, config: dict[str, Any]
) -> pd.DataFrame:
    """Re-run the fitted model over a new history window and forecast `index`.

    `apply` keeps the estimated coefficients and only re-runs the Kalman
    filter over the new window, which is exactly the "same model, later
    origin" situation the rolling validation needs.
    """
    window = history.iloc[-HORIZON * config["window_weeks"] :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        applied = fitted.apply(window, exog=regressors(window.index, config["regressors"]))
        pred = applied.get_forecast(len(index), exog=regressors(index, config["regressors"]))
        frame = pred.summary_frame()
    frame.index = index
    return frame


def run(train: pd.Series, val: pd.Series, test_index: pd.DatetimeIndex) -> ForecastResult:
    with Stopwatch() as sw:
        blocks = validation_blocks(val.index)
        history = pd.concat([train, val])

        # Stage 1: every candidate is estimated on the end of the training
        # window and then rolled through the 13 validation weeks.
        sweep_results: list[dict[str, Any]] = []
        for combo in itertools.product(*SWEEP.values()):
            config = dict(zip(SWEEP.keys(), combo))
            try:
                fitted = fit(train, config)
                forecasts = [
                    forecast_from(
                        fitted, history.loc[: block[0] - pd.Timedelta(hours=1)], block, config
                    )["mean"]
                    for block in blocks
                ]
                val_mape = block_mape(val, forecasts)
            except (ValueError, np.linalg.LinAlgError) as exc:
                log(f"sarima sweep {config}: failed ({exc})")
                val_mape = float("inf")
            sweep_results.append({**config, "val_mape_pct": round(val_mape, 4)})
            log(f"sarima sweep {config}: val MAPE {val_mape:.3f} %")
        best = min(sweep_results, key=lambda r: r["val_mape_pct"])
        chosen = {k: best[k] for k in SWEEP}

        # Stage 2: refit on train + validation and forecast the test week.
        fitted = fit(history, chosen)
        frame = forecast_from(fitted, history, test_index, chosen)
        point = frame["mean"]
        se = frame["mean_se"]
        quantiles = sort_quantiles({q: point + norm.ppf(q) * se for q in QUANTILES})

    return ForecastResult(
        name="sarima",
        point=point,
        quantiles=quantiles,
        runtime_seconds=sw.seconds,
        hyperparameters={
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in chosen.items()},
            "weekly_fourier_harmonics": WEEKLY_HARMONICS if chosen["regressors"] != "none" else 0,
            "estimation": "exact maximum likelihood (statsmodels SARIMAX)",
            "intervals": "analytic Gaussian forecast variance",
            "aic_refit": float(fitted.aic),
        },
        validation={"sweep": sweep_results, "chosen": chosen, "val_mape_pct": best["val_mape_pct"]},
    )
