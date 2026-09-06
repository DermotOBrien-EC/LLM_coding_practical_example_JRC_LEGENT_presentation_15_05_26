from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import ModelResult, QUANTILES, mape, validation_blocks

SCALE = 10000.0


def fit_model(y: pd.Series, order: tuple[int, int, int], seasonal: tuple[int, int, int, int]) -> tuple[Any, list[str]]:
    weekly_difference = (y.diff(168).iloc[168:] / SCALE).asfreq("h")
    model = SARIMAX(weekly_difference, order=order, seasonal_order=seasonal,
                    trend="n", enforce_stationarity=True, enforce_invertibility=True,
                    concentrate_scale=True)
    # extend() needs predicted states; keep them but omit unused smoothing arrays.
    model.ssm.set_conserve_memory(memory_no_smoothing=True, memory_no_filtered=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = model.fit(disp=False, maxiter=60, low_memory=False)
        if not fit.mle_retvals.get("converged", False):
            fit = model.fit(start_params=fit.params, disp=False, maxiter=120, low_memory=False)
    messages = list(dict.fromkeys(str(w.message) for w in caught))
    if not fit.mle_retvals.get("converged", False):
        raise RuntimeError(f"SARIMA optimizer did not converge: {order}, {seasonal}")
    return fit, messages


def run(train: pd.Series, pretest: pd.Series, future: pd.DatetimeIndex) -> ModelResult:
    start = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    forecasts: list[np.ndarray] = []
    warnings_seen: list[str] = []
    for order, seasonal in [((1, 0, 0), (1, 0, 0, 24)), ((2, 0, 0), (1, 0, 0, 24))]:
        fit, messages = fit_model(train, order, seasonal)
        warnings_seen.extend(messages)
        prediction: list[np.ndarray] = []
        for history, target in validation_blocks(pretest, len(train)):
            baseline = history.loc[target.index - pd.Timedelta(hours=168)].to_numpy()
            prediction.append(fit.get_forecast(len(target)).predicted_mean.to_numpy() * SCALE + baseline)
            # Only the state is updated as each validation block becomes history.
            observed_difference = pd.Series((target.to_numpy() - baseline) / SCALE, index=target.index).asfreq("h")
            fit = fit.extend(observed_difference)
        point = np.concatenate(prediction)
        score = mape(pretest.iloc[len(train):].to_numpy(), point)
        candidates.append({"order": list(order), "seasonal_order": list(seasonal),
                           "validation_mape_pct": score, "converged": True})
        forecasts.append(point)
        print(f"sarima {order}{seasonal} validation MAPE={score:.3f}%", flush=True)
    chosen = int(np.argmin([c["validation_mape_pct"] for c in candidates]))
    config = candidates[chosen]
    fit, messages = fit_model(pretest, tuple(config["order"]), tuple(config["seasonal_order"]))
    warnings_seen.extend(messages)
    distribution = fit.get_forecast(len(future))
    baseline = pretest.loc[future - pd.Timedelta(hours=168)].to_numpy()
    point = distribution.predicted_mean.to_numpy() * SCALE + baseline
    se = np.asarray(distribution.se_mean) * SCALE
    q = point[:, None] + se[:, None] * norm.ppf(QUANTILES)[None, :]
    params = {"order": config["order"], "seasonal_order": config["seasonal_order"],
              "weekly_pre_difference_hours": 168, "fixed_numeric_scale_mw": SCALE,
              "trend": "none", "concentrate_scale": True, "final_converged": True,
              "uncertainty": "Gaussian forecast-error variance; fixed fitted parameters"}
    return ModelResult("sarima", point, q, forecasts[chosen], params, candidates,
                       time.perf_counter() - start, {"optimizer_warnings": sorted(set(warnings_seen))})
