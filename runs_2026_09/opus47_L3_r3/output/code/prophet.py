"""Prophet with daily, weekly and yearly seasonality plus German holidays.

Uses darts' Prophet wrapper so the API stays consistent with the rest of the
darts models in the study. We pass `country_holidays='DE'` and let Prophet's
own MAP-fit choose the harmonic amplitudes; there are no hyperparameters worth
sweeping in a first-pass univariate setup.

Prediction intervals come from Prophet's Monte-Carlo trajectories
(`num_samples`), from which we take the 10, 90, 2.5 and 97.5 percentiles.
"""

from __future__ import annotations

# Careful: this file is called `prophet.py`, which would shadow the real
# `prophet` package on sys.path. Force-load the real prophet package into
# `sys.modules['prophet']` before darts tries `import prophet` on its own,
# then rebind our own module under a different key so subsequent imports of
# `prophet` continue to hit the real package.
import importlib.util as _il
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_VENV_SITE = _HERE.parent.parent.parent / ".venv" / "lib" / "python3.12" / "site-packages" / "prophet"
if "prophet" in _sys.modules and getattr(_sys.modules["prophet"], "__file__", "").endswith("prophet.py"):
    _sys.modules["_forecast_prophet_module"] = _sys.modules.pop("prophet")
_spec = _il.spec_from_file_location(
    "prophet",
    str(_VENV_SITE / "__init__.py"),
    submodule_search_locations=[str(_VENV_SITE)],
)
_real = _il.module_from_spec(_spec)
_sys.modules["prophet"] = _real
_spec.loader.exec_module(_real)

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet as DartsProphet

from common import RANDOM_SEED, Split, mape, to_hourly_naive

# Prophet is very chatty; silence its logging so the transcript stays clean.
for name in ("cmdstanpy", "prophet", "prophet.plot", "prophet.models"):
    logging.getLogger(name).setLevel(logging.ERROR)


@dataclass
class ProphetResult:
    forecast: pd.Series
    lower_80: pd.Series
    upper_80: pd.Series
    lower_95: pd.Series
    upper_95: pd.Series
    runtime_seconds: float
    hyperparameters: dict[str, object]
    trend: pd.Series | None = None
    weekly: pd.Series | None = None
    yearly: pd.Series | None = None
    fitted_model: object | None = None


def _as_timeseries(series: pd.Series) -> TimeSeries:
    naive = to_hourly_naive(series)
    return TimeSeries.from_series(naive)


def run_prophet(split: Split) -> ProphetResult:
    t0 = time.perf_counter()
    # Validation is essentially a sanity check for Prophet: with country_holidays
    # and yearly/weekly/daily seasonality on, there is little to tune. We still
    # score Val to report the pre-refit metric alongside test.
    train_ts = _as_timeseries(split.train)
    val_ts = _as_timeseries(split.val)

    hp: dict[str, object] = {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": True,
        "country_holidays": "DE",
    }
    prov = DartsProphet(**hp, random_state=RANDOM_SEED)
    prov.fit(train_ts)
    val_forecast = prov.predict(n=len(val_ts))
    val_mape = mape(val_ts.values().flatten(), val_forecast.values().flatten())

    # Refit on Train + Validation with the same configuration.
    full_ts = _as_timeseries(split.train_plus_val)
    model = DartsProphet(**hp, random_state=RANDOM_SEED)
    model.fit(full_ts)
    # Ask for samples so we can carve out empirical intervals.
    samples = model.predict(n=168, num_samples=500)
    values = samples.all_values()  # (n_time, n_components, n_samples)
    arr = values[:, 0, :]  # (n_time, n_samples)
    point = np.median(arr, axis=1)
    q10 = np.quantile(arr, 0.10, axis=1)
    q90 = np.quantile(arr, 0.90, axis=1)
    q025 = np.quantile(arr, 0.025, axis=1)
    q975 = np.quantile(arr, 0.975, axis=1)

    idx = split.test.index
    forecast = pd.Series(point, index=idx, name="prophet")
    lower_80 = pd.Series(q10, index=idx, name="q10")
    upper_80 = pd.Series(q90, index=idx, name="q90")
    lower_95 = pd.Series(q025, index=idx, name="q025")
    upper_95 = pd.Series(q975, index=idx, name="q975")

    # Pull out the components for figure 08 (decomposition), if we need it.
    trend = weekly = yearly = None
    try:
        # darts Prophet exposes the underlying prophet in `.model`
        underlying = getattr(model, "model", None)
        if underlying is not None:
            future = underlying.make_future_dataframe(periods=168, freq="h", include_history=False)
            comps = underlying.predict(future)
            comps = comps.set_index(pd.DatetimeIndex(comps["ds"]))
            trend = pd.Series(comps["trend"].values, index=idx, name="trend")
            if "weekly" in comps:
                weekly = pd.Series(comps["weekly"].values, index=idx, name="weekly")
            if "yearly" in comps:
                yearly = pd.Series(comps["yearly"].values, index=idx, name="yearly")
    except Exception:
        pass

    runtime = time.perf_counter() - t0
    hp["validation_mape_pct"] = float(val_mape)
    return ProphetResult(
        forecast=forecast,
        lower_80=lower_80,
        upper_80=upper_80,
        lower_95=lower_95,
        upper_95=upper_95,
        runtime_seconds=runtime,
        hyperparameters=hp,
        trend=trend,
        weekly=weekly,
        yearly=yearly,
        fitted_model=model,
    )
