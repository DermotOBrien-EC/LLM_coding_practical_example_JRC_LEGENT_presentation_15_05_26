"""Prophet with German public holidays.

Prophet fits a smooth trend plus daily, weekly and yearly seasonal curves
and a fixed effect per holiday. It is a curve fit, not an autoregression:
the forecast for a given hour does not depend on the last observed hours,
only on the calendar. That makes it robust to a strange final week of
history but blind to whatever the last few days are doing.

Four settings are compared on validation: additive versus multiplicative
seasonality, and how flexible the trend is allowed to be. Because a
Prophet forecast does not depend on the forecast origin, all seven
validation weeks are scored from a single fit on Train.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# This file has to be called prophet.py (the task's layout), which shadows
# the real `prophet` package whenever this directory sits first on sys.path.
# Import the real package with this directory temporarily hidden so that
# darts can find it, then restore the path.
_CODE_DIR = Path(__file__).resolve().parent
_saved_path = list(sys.path)
sys.path = [p for p in sys.path if Path(p or ".").resolve() != _CODE_DIR]
sys.modules.pop("prophet", None)
import prophet as _real_prophet  # noqa: E402,F401

sys.path = _saved_path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from darts import TimeSeries  # noqa: E402
from darts.models import Prophet  # noqa: E402

import common  # noqa: E402

CANDIDATES: list[dict[str, Any]] = [
    {"seasonality_mode": "additive", "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0},
    {"seasonality_mode": "additive", "changepoint_prior_scale": 0.01, "seasonality_prior_scale": 10.0},
    {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0},
    {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.01, "seasonality_prior_scale": 10.0},
]
N_SAMPLES: int = 1000


def build_model(cand: dict[str, Any]) -> Prophet:
    return Prophet(
        country_holidays="DE",
        random_state=common.SEED,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        **cand,
    )


def _to_ts(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(series.astype(np.float64))


def run(s: pd.Series) -> common.ModelResult:
    t0 = time.perf_counter()
    train = common.train_slice(s)
    val = common.val_slice(s)
    table: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        tc = time.perf_counter()
        model = build_model(cand)
        model.fit(_to_ts(train))
        fc = model.predict(len(val)).to_series()
        fc.index = val.index
        per_origin = [common.mape(s.loc[common.horizon_index(o)], fc) for o in common.VALIDATION_ORIGINS]
        row = {
            **cand,
            "val_mape_pct": float(np.mean(per_origin)),
            "val_mape_by_origin_pct": [round(v, 3) for v in per_origin],
            "val_mape_full_window_pct": common.mape(val, fc),
            "fit_seconds": round(time.perf_counter() - tc, 1),
        }
        table.append(row)
        print(f"prophet: {cand} val MAPE {row['val_mape_pct']:.2f} % ({row['fit_seconds']} s)")
    table = sorted(table, key=lambda r: r["val_mape_pct"])
    chosen = {k: table[0][k] for k in CANDIDATES[0]}

    # Refit on Train + Validation and forecast the test week with samples.
    common.set_seeds()
    model = build_model(chosen)
    model.fit(_to_ts(common.train_val_slice(s)))
    idx = common.horizon_index(common.TEST_START)
    point = model.predict(common.HORIZON).to_series()
    point.index = idx
    samples = model.predict(common.HORIZON, num_samples=N_SAMPLES).all_values()[:, 0, :]
    quantiles = {q: pd.Series(np.quantile(samples, q, axis=1), index=idx) for q in common.QUANTILES}
    quantiles = common.sort_quantiles(quantiles)

    # Fitted components for the decomposition figure (only drawn if Prophet
    # finishes in the top two). Stored at a coarse resolution to keep the
    # cache small: trend per day, weekly curve over one week, yearly curve
    # over one year.
    prophet_model = model.model
    frame = pd.DataFrame({"ds": common.train_val_slice(s).index})
    comp = prophet_model.predict(frame)
    comp = comp.set_index(pd.DatetimeIndex(comp["ds"]))
    trend_daily = comp["trend"].resample("D").mean()
    week = comp.loc["2019-01-07":"2019-01-13 23:00", "weekly"]
    year = comp.loc["2019-01-01":"2019-12-31 23:00", "yearly"].resample("D").mean()
    extra = {
        "trend_daily": {"index": [str(t.date()) for t in trend_daily.index], "values": trend_daily.round(1).tolist()},
        "weekly": {"index": [str(t) for t in week.index], "values": week.round(1).tolist()},
        "yearly_daily": {"index": [str(t.date()) for t in year.index], "values": year.round(1).tolist()},
        "holiday_effect_jan1_mw": float(comp.loc["2019-01-01", "holidays"].mean()) if "holidays" in comp else None,
        "seasonality_mode": chosen["seasonality_mode"],
    }
    return common.ModelResult(
        name="prophet",
        point=point,
        quantiles=quantiles,
        hyperparameters={
            **chosen,
            "country_holidays": "DE",
            "seasonalities": ["daily", "weekly", "yearly"],
            "uncertainty_samples": N_SAMPLES,
            "n_candidates": len(CANDIDATES),
        },
        runtime_seconds=time.perf_counter() - t0,
        validation_mape_pct=table[0]["val_mape_pct"],
        validation_table=table,
        notes="Intervals from 1000 Prophet predictive samples (trend uncertainty plus observation noise).",
        extra=extra,
    )


if __name__ == "__main__":
    common.silence_warnings()
    s = common.load_series()
    result = run(s)
    result.save()
    print(f"prophet: test MAPE {common.mape(common.test_slice(s), result.point):.2f} %, runtime {result.runtime_seconds:.0f} s")
