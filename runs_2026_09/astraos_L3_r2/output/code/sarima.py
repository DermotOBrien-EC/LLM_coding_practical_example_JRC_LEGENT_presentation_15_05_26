from __future__ import annotations

from time import perf_counter
from typing import Any
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .common import ModelResult, TRAIN_END, gaussian_quantiles, mape, validation_blocks


def weekly_terms(index: pd.DatetimeIndex) -> np.ndarray:
    phase = np.asarray(index.dayofweek * 24 + index.hour, dtype=float) / 168
    return np.column_stack(
        [function(2 * np.pi * k * phase) for k in (1, 2, 3) for function in (np.sin, np.cos)]
    )


def fit(series: pd.Series, order: tuple[int, int, int], maxiter: int = 60) -> Any:
    model = SARIMAX(
        series / 1000.0,
        exog=weekly_terms(series.index),
        order=order,
        seasonal_order=(1, 0, 0, 24),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    return model.fit(disp=False, maxiter=maxiter, method="lbfgs", cov_type="none", low_memory=False)


def run(pretest: pd.Series) -> ModelResult:
    start = perf_counter()
    train = pretest.loc[: TRAIN_END - pd.Timedelta(hours=1)]
    validation: list[dict[str, Any]] = []
    for order in [(1, 0, 0), (2, 0, 0)]:
        result = fit(train, order)
        convergence = bool(result.mle_retvals.get("converged", False))
        actuals, predictions = [], []
        for _, target in validation_blocks(pretest):
            forecast = result.get_forecast(len(target), exog=weekly_terms(target.index))
            predictions.extend(np.asarray(forecast.predicted_mean) * 1000)
            actuals.extend(target.to_numpy())
            result = result.extend(target / 1000, exog=weekly_terms(target.index))
        validation.append(
            {
                "order": list(order),
                "mape_pct": mape(np.array(actuals), np.array(predictions)),
                "converged": convergence,
            }
        )
    eligible = [item for item in validation if item["converged"]]
    if not eligible:
        raise RuntimeError("Neither SARIMA training candidate converged")
    best = min(eligible, key=lambda item: item["mape_pct"])
    final = fit(pretest, tuple(best["order"]), maxiter=180)
    index = pd.date_range("2020-01-01", periods=168, freq="h")
    forecast = final.get_forecast(168, exog=weekly_terms(index))
    mean = np.asarray(forecast.predicted_mean) * 1000
    sigma = np.asarray(forecast.se_mean) * 1000
    return ModelResult(
        "sarima",
        mean,
        gaussian_quantiles(mean, sigma),
        perf_counter() - start,
        {
            "order": best["order"],
            "seasonal_order": [1, 0, 0, 24],
            "trend": "c",
            "weekly_fourier_harmonics": 3,
            "candidate_maxiter": 60,
            "refit_maxiter": 180,
            "selection": "lowest validation MAPE among converged candidates",
            "intervals": "Gaussian state-space forecast, fixed estimated parameters",
        },
        validation,
        {
            "refit_converged": bool(final.mle_retvals.get("converged", False)),
            "refit_iterations": int(final.mle_retvals.get("iterations", -1)),
        },
    )
