"""Prophet with German public holidays, via darts.

Prophet is a "curve-fitting" forecaster: it explains a series as a slow trend
plus a stack of repeating shapes (a daily shape, a weekly shape, a yearly
shape) plus bumps on named holidays. That last part is why it is in this
bake-off: unlike the naive baseline or a plain SARIMA, Prophet can be told that
Jan 1 is New Year's Day and behaves differently from an ordinary Wednesday.

We let Prophet fit daily, weekly and yearly shapes and add the German federal
holiday calendar. The one knob we tune on the validation window is whether the
seasonal shapes add to the trend (additive) or scale it (multiplicative).
Prediction intervals come from Prophet's own posterior samples.
"""

from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import pandas as pd

import common as c

warnings.filterwarnings("ignore")
for _name in ("cmdstanpy", "prophet", "pytorch_lightning"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from darts import TimeSeries          # noqa: E402
from darts.models import Prophet      # noqa: E402

QUANTILE_LEVELS: list[float] = [0.025, 0.1, 0.5, 0.9, 0.975]
SEASONALITY_MODES: list[str] = ["additive", "multiplicative"]
NUM_SAMPLES: int = 1000


def _make_model(mode: str) -> Prophet:
    return Prophet(
        country_holidays="DE",
        seasonality_mode=mode,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        random_state=c.SEED,
    )


def _to_ts(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(series, freq="h")


def run(splits: c.Splits) -> c.ModelResult:
    start = time.time()

    train_ts = _to_ts(splits.train)
    val_ts = _to_ts(splits.val)

    # ---- pick additive vs multiplicative on validation MAPE ------------
    best_mode: str | None = None
    best_val_mape = np.inf
    sweep: list[dict] = []
    for mode in SEASONALITY_MODES:
        model = _make_model(mode)
        model.fit(train_ts)
        val_fc = model.predict(len(val_ts), num_samples=1)
        val_pred = pd.Series(val_fc.values().ravel(), index=splits.val.index)
        val_mape = c.mape(splits.val.to_numpy(), val_pred.to_numpy())
        sweep.append({"seasonality_mode": mode, "val_mape": round(val_mape, 4)})
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_mode = mode

    assert best_mode is not None

    # ---- refit on Train+Val and forecast the test week -----------------
    trainval_ts = _to_ts(splits.trainval)
    model = _make_model(best_mode)
    model.fit(trainval_ts)
    sampled = model.predict(len(splits.test), num_samples=NUM_SAMPLES)

    point = pd.Series(sampled.quantile(0.5).values().ravel(), index=splits.test.index, name="prophet")
    quantiles: dict[float, pd.Series] = {}
    for q in QUANTILE_LEVELS:
        quantiles[q] = pd.Series(sampled.quantile(q).values().ravel(), index=splits.test.index, name=f"prophet_q{q}")

    components = _extract_components(model)

    return c.ModelResult(
        name="prophet",
        point=point,
        runtime_s=time.time() - start,
        hyperparameters={
            "seasonality_mode": best_mode,
            "country_holidays": "DE",
            "seasonalities": ["daily", "weekly", "yearly"],
            "num_samples": NUM_SAMPLES,
            "selected_by_val_mape": round(best_val_mape, 4),
            "sweep": sweep,
        },
        quantiles=quantiles,
        extra={"components": components},
    )


def _extract_components(model: Prophet) -> dict[str, list]:
    """Pull compact, picklable trend/weekly/yearly shapes from a fitted model.

    Used only for the decomposition figure (drawn if Prophet ranks in the top
    two). We store small summarised arrays rather than the whole fitted object
    so results cache cleanly to disk.
    """
    inner = model.model  # the underlying prophet.Prophet
    future = inner.make_future_dataframe(periods=0, freq="h", include_history=True)
    comp = inner.predict(future)
    comp = comp.assign(ds=pd.to_datetime(comp["ds"]))
    daily_trend = comp.set_index("ds")["trend"].resample("D").mean()
    hour_of_week = comp["ds"].dt.dayofweek * 24 + comp["ds"].dt.hour
    weekly = comp.groupby(hour_of_week)["weekly"].mean()
    yearly = comp.groupby(comp["ds"].dt.dayofyear)["yearly"].mean()
    return {
        "trend_dates": [d.strftime("%Y-%m-%d") for d in daily_trend.index],
        "trend_values": daily_trend.to_numpy().tolist(),
        "weekly_hour_of_week": weekly.index.to_numpy().tolist(),
        "weekly_values": weekly.to_numpy().tolist(),
        "yearly_day_of_year": yearly.index.to_numpy().tolist(),
        "yearly_values": yearly.to_numpy().tolist(),
    }
