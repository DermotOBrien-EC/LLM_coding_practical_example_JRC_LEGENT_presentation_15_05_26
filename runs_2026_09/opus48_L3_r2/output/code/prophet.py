"""Prophet with German public holidays, via darts.

Prophet is a "curve-fitting" forecaster: it explains the series as a slow trend
plus a stack of repeating seasonal shapes (daily, weekly, yearly) plus known
holiday effects. It does not use lags of the load; it just learns what a typical
Tuesday in January at 8am looks like and projects that forward. Because it is
told about German public holidays, it has a fair chance on New Year's Day, where
demand collapses to a Sunday-like level.

We let Prophet quantify its own uncertainty by sampling from its posterior
predictive distribution, which gives us the prediction intervals.
"""

from __future__ import annotations

import logging
import time

import numpy as np

import common as C

# Prophet (via cmdstanpy) is chatty; quiet it down.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# We tune the seasonality mode and how freely the trend may bend.
GRID: list[dict] = [
    {"seasonality_mode": "additive", "changepoint_prior_scale": 0.05},
    {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.05},
    {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.5},
]

NUM_SAMPLES = 500


def _make_model(params: dict):
    C.ensure_real_prophet()
    from darts.models import Prophet

    return Prophet(
        country_holidays="DE",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode=params["seasonality_mode"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
    )


def run(bundle: C.DataBundle) -> C.ForecastResult:
    start = time.time()
    C.set_global_seeds()

    train_ts = C.to_darts(C.slice_inclusive(bundle.full, C.TRAIN_START, C.TRAIN_END))
    val_actual = bundle.val_select.to_numpy()

    # --- Select settings on the first validation week ------------------------
    best_params, best_val = None, np.inf
    for params in GRID:
        try:
            m = _make_model(params)
            m.fit(train_ts)
            pred = m.predict(C.HORIZON)  # deterministic mean is enough to score
            score = C.mape(val_actual, pred.values(copy=False).flatten())
        except Exception:
            score = np.inf
        if score < best_val:
            best_val, best_params = score, params

    if best_params is None:
        best_params = GRID[1]

    # --- Refit on Train+Validation and forecast the test week ----------------
    trainval_ts = C.to_darts(C.slice_inclusive(bundle.full, C.TRAIN_START, C.VAL_END))
    model = _make_model(best_params)
    model.fit(trainval_ts)
    pred = model.predict(C.HORIZON, num_samples=NUM_SAMPLES)
    point, quantiles = C.darts_quantiles(pred)

    runtime = time.time() - start
    return C.ForecastResult(
        name="prophet",
        point=point,
        runtime_seconds=runtime,
        hyperparameters=best_params,
        val_mape=float(best_val),
        quantiles=quantiles,
    )


if __name__ == "__main__":
    b = C.build_bundle()
    res = run(b)
    print("prophet params:", res.hyperparameters, "val MAPE:", res.val_mape)
    print("prophet test MAPE:", C.mape(b.test.to_numpy(), res.point))
