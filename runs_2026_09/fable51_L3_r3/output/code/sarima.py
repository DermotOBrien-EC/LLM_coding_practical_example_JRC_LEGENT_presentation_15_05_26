"""SARIMA: the classical statistical benchmark, via statsmodels SARIMAX.

The model is "regression with SARIMA errors": a level shift for each
weekday and for public holidays (both known in advance from the calendar),
with the remaining hour-to-hour structure captured by an ARIMA process
that has a 24-hour seasonal component. Weekly seasonality (period 168)
is not practical to fit directly, so the weekday shifts carry it.

Fitting one candidate on the full 4.75-year history costs about three
minutes on this machine, so the model is estimated on a rolling window of
the most recent 52 weeks instead. That is the usual way SARIMA is run in
practice: its coefficients are constant, so old data adds cost, not skill.
The window keeps the whole study reproducible in under half an hour.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (
    HORIZON,
    ForecastResult,
    Timer,
    calendar_features,
    rolling_validation_mape,
    validation_origins,
)

WINDOW_HOURS = 52 * HORIZON  # 8,736 hours = 52 weeks
Order = tuple[int, int, int]
SeasonalOrder = tuple[int, int, int, int]
# (p, d, q)(P, D, Q, 24). All use daily seasonal differencing; the
# non-seasonal part varies from a small ARMA to a differenced ARIMA.
CANDIDATES: list[tuple[Order, SeasonalOrder]] = [
    ((1, 0, 1), (1, 1, 1, 24)),
    ((2, 0, 2), (1, 1, 1, 24)),
    ((1, 1, 1), (1, 1, 1, 24)),
    ((2, 0, 1), (0, 1, 1, 24)),
    ((3, 0, 2), (1, 1, 1, 24)),
    ((4, 0, 2), (1, 1, 1, 24)),
]
EXOG_COLUMNS = [f"dow_{d}" for d in range(6)] + ["is_public_holiday_de"]


def exog_for(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Weekday shift dummies (Sunday is the baseline) and the holiday flag."""
    cal = calendar_features(index)
    frame = pd.DataFrame(index=index)
    for d in range(6):
        frame[f"dow_{d}"] = (cal["day_of_week"] == d).astype(float)
    frame["is_public_holiday_de"] = cal["is_public_holiday_de"].astype(float)
    return frame


def fit(history: pd.Series, order: Order, seasonal_order: SeasonalOrder) -> Any:
    window = history.iloc[-WINDOW_HOURS:]
    window.index = window.index.tz_localize(None)
    window = window.asfreq("h")
    exog = exog_for(history.index[-WINDOW_HOURS:])
    exog.index = window.index
    model = sm.tsa.SARIMAX(window, exog=exog, order=order, seasonal_order=seasonal_order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=100)


def forecast(results: Any, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.DataFrame]:
    """Point forecast and 80 % / 95 % interval bounds for the hours in `index`."""
    exog = exog_for(index)
    exog.index = index.tz_localize(None)
    pred = results.get_forecast(len(index), exog=exog)
    point = pd.Series(np.asarray(pred.predicted_mean), index=index, name="sarima")
    ci80 = np.asarray(pred.conf_int(alpha=0.20))
    ci95 = np.asarray(pred.conf_int(alpha=0.05))
    bounds = pd.DataFrame(
        {0.025: ci95[:, 0], 0.1: ci80[:, 0], 0.5: point.values, 0.9: ci80[:, 1], 0.975: ci95[:, 1]},
        index=index,
    )
    return point, bounds


def validation_forecasts(results: Any, val: pd.Series, origins: list[pd.Timestamp], horizon: int) -> list[pd.Series]:
    """Roll through the validation weeks: forecast one week, then feed the
    week's actual values into the filter (no re-estimation) and repeat."""
    forecasts = []
    current = results
    for origin in origins:
        window = val[origin : origin + pd.Timedelta(hours=horizon - 1)]
        forecasts.append(forecast(current, window.index)[0])
        new_exog = exog_for(window.index)
        new_exog.index = window.index.tz_localize(None)
        new_endog = window.copy()
        new_endog.index = new_exog.index
        new_endog = new_endog.asfreq("h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            current = current.extend(new_endog, exog=new_exog)
    return forecasts


def run(train: pd.Series, val: pd.Series, horizon: int = HORIZON) -> ForecastResult:
    with Timer() as timer:
        origins = validation_origins(val, horizon)
        scores: list[tuple[float, Order, SeasonalOrder, float]] = []
        for order, seasonal_order in CANDIDATES:
            results = fit(train, order, seasonal_order)
            fcs = validation_forecasts(results, val, origins, horizon)
            score = rolling_validation_mape(val, fcs, origins)
            scores.append((score, order, seasonal_order, float(results.aic)))
            print(f"  sarima {order}{seasonal_order}: validation MAPE {score:.2f} % (AIC {results.aic:.0f})", flush=True)
        best_score, order, seasonal_order, _ = min(scores, key=lambda s: s[0])

        history = pd.concat([train, val])
        final = fit(history, order, seasonal_order)
        index = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
        point, bounds = forecast(final, index)
        quantiles = {tau: bounds[tau] for tau in bounds.columns}

    return ForecastResult(
        name="sarima",
        point=point,
        quantiles=quantiles,
        runtime_seconds=timer.seconds,
        hyperparameters={
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "estimation_window_hours": WINDOW_HOURS,
            "exogenous_regressors": EXOG_COLUMNS,
            "aic_final_fit": float(final.aic),
            "intervals": "analytic Gaussian forecast variance",
            "candidates_scored": len(CANDIDATES),
        },
        validation_mape_pct=best_score,
        extras={"validation_scores": scores, "final_params": final.params.to_dict()},
    )


if __name__ == "__main__":
    from common import load_series, split

    train, val, _ = split(load_series())
    result = run(train, val)
    print(f"validation MAPE {result.validation_mape_pct:.2f} %  runtime {result.runtime_seconds:.1f} s")
    print(result.point.head())
