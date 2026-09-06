from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import ForecastResult, elapsed_seconds, mape_pct, normal_interval, timer


def _fit(values: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            values.astype(float),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            simple_differencing=True,
        )
        return model.fit(disp=False, maxiter=80)


def forecast(train: pd.Series, validation: pd.Series, train_validation: pd.Series, test: pd.Series) -> ForecastResult:
    start = timer()
    candidates = [
        ((1, 0, 1), (1, 1, 1, 24)),
        ((2, 0, 1), (1, 1, 1, 24)),
        ((1, 0, 2), (1, 1, 1, 24)),
        ((1, 1, 1), (1, 0, 1, 24)),
        ((2, 1, 1), (1, 0, 1, 24)),
    ]
    validation_scores: list[dict[str, object]] = []
    best_order: tuple[int, int, int] | None = None
    best_seasonal_order: tuple[int, int, int, int] | None = None
    best_score = float("inf")
    for order, seasonal_order in candidates:
        try:
            fitted = _fit(train, order, seasonal_order)
            pred = fitted.get_forecast(steps=len(validation)).predicted_mean.to_numpy(dtype=float)
            score = mape_pct(validation.to_numpy(), pred)
            validation_scores.append({"order": order, "seasonal_order": seasonal_order, "validation_mape_pct": score})
            if score < best_score:
                best_score = score
                best_order = order
                best_seasonal_order = seasonal_order
        except Exception as exc:
            validation_scores.append({"order": order, "seasonal_order": seasonal_order, "error": repr(exc)})
    if best_order is None or best_seasonal_order is None:
        raise RuntimeError("SARIMA validation search produced no fitted model")

    fitted = _fit(train_validation, best_order, best_seasonal_order)
    forecast_result = fitted.get_forecast(steps=len(test))
    frame = forecast_result.summary_frame(alpha=0.05)
    point = forecast_result.predicted_mean.to_numpy(dtype=float)
    se = frame["mean_se"].to_numpy(dtype=float)
    q025, q975 = normal_interval(point, se, 1.959963984540054)
    q10, q90 = normal_interval(point, se, 1.2815515655446004)
    q25, q75 = normal_interval(point, se, 0.6744897501960817)
    return ForecastResult(
        name="sarima",
        point=point,
        runtime_seconds=elapsed_seconds(start),
        hyperparameters={
            "selected_order": best_order,
            "selected_seasonal_order": best_seasonal_order,
            "validation_mape_pct": best_score,
            "validation_grid": validation_scores,
        },
        validation_mape_pct=best_score,
        q025=q025,
        q10=q10,
        q25=q25,
        q50=point,
        q75=q75,
        q90=q90,
        q975=q975,
    )
