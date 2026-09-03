"""SARIMA: the classical statistical entry.

A seasonal ARIMA model says the next value is a weighted sum of recent
values and recent forecast errors, after the repeating daily pattern has
been differenced away. It is the textbook approach and it is the one that
comes with honest analytic prediction intervals for free.

Three practical decisions had to be made, all of them recorded here because
they shape the result:

1. **Seasonal period.** Hourly load has two rhythms, daily (24) and weekly
   (168). A seasonal order of 168 is not computable at this sample size, so
   the seasonal part is set to 24 and a seasonal difference removes the
   daily cycle. That leaves the weekly rhythm, in particular the drop at
   weekends, for the model to find on its own.

2. **Weekly Fourier terms.** As an alternative to hoping the weekly rhythm
   emerges from differencing, we offer the model three pairs of sine and
   cosine waves with a 168-hour period as external regressors. These come
   from the clock alone, not from any other dataset. Whether they help is
   decided on the validation window, not by us.

3. **Training window length.** Fitting a state-space model by maximum
   likelihood on all 41,616 training hours takes many minutes per candidate,
   which makes a real hyperparameter search impossible. Instead the length
   of the recent window used for fitting is itself treated as a
   hyperparameter and chosen on validation. This is standard practice for
   ARIMA-type models, which are local by construction: after seasonal
   differencing, load from 2015 tells you very little about January 2020.

Candidates are scored the same way as every other model: the validation
quarter is forecast in back-to-back 168-hour blocks. Between blocks the
observed values are appended to the fitted model without re-estimating any
coefficients, which is exactly how such a model is operated in production.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX

import common as c

warnings.filterwarnings("ignore")

WEEKLY_HARMONICS: int = 3
MAX_ITER: int = 50

# (p, d, q) x (P, D, Q, 24) candidates. All of them difference once at lag 24
# to remove the daily cycle; they differ in how much short-memory structure
# they carry.
ORDER_CANDIDATES: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
    ((2, 0, 1), (1, 1, 1, 24)),
    ((3, 0, 2), (1, 1, 1, 24)),
    ((2, 1, 2), (1, 1, 1, 24)),
]
WINDOW_WEEKS_CANDIDATES: tuple[int, ...] = (26, 52)
USE_FOURIER_CANDIDATES: tuple[bool, ...] = (False, True)


def _exog(index: pd.DatetimeIndex, use_fourier: bool) -> pd.DataFrame | None:
    """Weekly sine/cosine regressors, or nothing at all."""
    if not use_fourier:
        return None
    return c.fourier_terms(index, period_hours=168.0, n_harmonics=WEEKLY_HARMONICS, prefix="week")


def _fit(endog: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int], use_fourier: bool) -> Any:
    model = SARIMAX(
        endog,
        exog=_exog(endog.index, use_fourier),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=MAX_ITER)


def _forecast(results: Any, index: pd.DatetimeIndex, use_fourier: bool) -> tuple[pd.Series, pd.Series]:
    """Mean forecast and its standard error, both indexed by timestamp."""
    forecast = results.get_forecast(steps=len(index), exog=_exog(index, use_fourier))
    mean = pd.Series(np.asarray(forecast.predicted_mean, dtype=float), index=index)
    stderr = pd.Series(np.asarray(forecast.se_mean, dtype=float), index=index)
    return mean, stderr


def _score_candidate(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    window_weeks: int,
    use_fourier: bool,
) -> float:
    """Validation MAPE over the thirteen 168-hour blocks."""
    train_window = series.loc[: c.TRAIN_END].iloc[-window_weeks * 168 :]
    results = _fit(train_window, order, seasonal_order, use_fourier)

    errors: list[np.ndarray] = []
    for origin in c.validation_origins():
        block = pd.date_range(origin, periods=c.HORIZON, freq="h", tz="UTC")
        mean, _ = _forecast(results, block, use_fourier)
        actual = series.loc[block]
        errors.append(np.abs((actual.to_numpy() - mean.to_numpy()) / actual.to_numpy()))
        # Observe the block, then move on without re-estimating coefficients.
        results = results.append(actual, exog=_exog(block, use_fourier), refit=False)

    return float(np.mean(np.concatenate(errors)) * 100.0)


def select_hyperparameters(series: pd.Series) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    """Search orders, window length and the Fourier switch on validation."""
    trace: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None

    for order, seasonal_order in ORDER_CANDIDATES:
        for window_weeks in WINDOW_WEEKS_CANDIDATES:
            for use_fourier in USE_FOURIER_CANDIDATES:
                started = time.time()
                try:
                    score = _score_candidate(series, order, seasonal_order, window_weeks, use_fourier)
                except Exception as error:  # a candidate that will not converge is simply out
                    trace.append(
                        {
                            "order": list(order),
                            "seasonal_order": list(seasonal_order),
                            "window_weeks": window_weeks,
                            "weekly_fourier": use_fourier,
                            "validation_mape_pct": None,
                            "error": str(error)[:200],
                        }
                    )
                    continue

                config = {
                    "order": list(order),
                    "seasonal_order": list(seasonal_order),
                    "window_weeks": window_weeks,
                    "weekly_fourier": use_fourier,
                }
                trace.append({**config, "validation_mape_pct": score, "seconds": round(time.time() - started, 1)})
                print(f"  {config} -> {score:.3f}%", flush=True)
                if best is None or score < best[0]:
                    best = (score, config)

    if best is None:
        raise RuntimeError("No SARIMA candidate converged.")
    return best[1], best[0], trace


def run(series: pd.Series) -> c.ForecastResult:
    """Select on validation, refit on the chosen window ending 2019-12-31."""
    start = time.time()

    best_config, best_score, trace = select_hyperparameters(series)

    order = tuple(best_config["order"])
    seasonal_order = tuple(best_config["seasonal_order"])
    window_weeks = int(best_config["window_weeks"])
    use_fourier = bool(best_config["weekly_fourier"])

    # Refit on train + validation, using the selected window length, so the
    # final model ends exactly at the hour before the test week starts.
    fitting_data = c.train_plus_val(series).iloc[-window_weeks * 168 :]
    results = _fit(fitting_data, order, seasonal_order, use_fourier)

    test = series.loc[c.TEST_START : c.TEST_END]
    mean, stderr = _forecast(results, test.index, use_fourier)
    mean.name = "sarima"

    # The state-space model gives a Gaussian predictive distribution, so the
    # quantiles are analytic rather than simulated.
    quantiles = pd.DataFrame(
        {str(q): mean.to_numpy() + norm.ppf(q) * stderr.to_numpy() for q in c.QUANTILES},
        index=test.index,
    )

    return c.ForecastResult(
        name="sarima",
        point_forecast=mean,
        runtime_seconds=time.time() - start,
        hyperparameters={
            **best_config,
            "seasonal_period": 24,
            "n_fitting_observations": int(len(fitting_data)),
            "fitting_window_start": str(fitting_data.index[0]),
            "weekly_fourier_harmonics": WEEKLY_HARMONICS if use_fourier else 0,
            "interval_method": "analytic Gaussian from the state-space model",
            "aic": float(results.aic),
        },
        quantile_forecast=quantiles,
        validation_mape_pct=best_score,
        extras={"grid_search": trace},
    )


if __name__ == "__main__":
    result = run(c.load_load_series())
    result.save()
    print(result.name, result.validation_mape_pct, result.runtime_seconds)
