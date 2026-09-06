from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet as DartsProphet

from common import FINAL_TRAIN_END, MODEL_COLORS, ModelOutput, TEST_END, TEST_START, TRAIN_END, TRAIN_START, VAL_END, VAL_START, mape_pct

CANDIDATES = [
    {"seasonality_mode": "additive", "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0},
    {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0},
    {"seasonality_mode": "additive", "changepoint_prior_scale": 0.1, "seasonality_prior_scale": 10.0},
]


def to_darts(series: pd.Series) -> TimeSeries:
    clean = series.copy()
    clean.index = pd.DatetimeIndex(clean.index).tz_localize(None)
    clean = clean.asfreq("h")
    return TimeSeries.from_series(clean)


def make_model(params: dict[str, object]) -> DartsProphet:
    return DartsProphet(
        country_holidays="DE",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        random_state=42,
        suppress_stdout_stderror=True,
        **params,
    )


def series_from_darts(prediction: TimeSeries, index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(prediction.values(copy=False).reshape(-1), index=index, name="forecast")


def select_params(series: pd.Series) -> dict[str, object]:
    train = series.loc[TRAIN_START:TRAIN_END]
    val = series.loc[VAL_START:VAL_END]
    train_ts = to_darts(train)
    best = CANDIDATES[0]
    best_score = float("inf")
    for params in CANDIDATES:
        model = make_model(params)
        model.fit(train_ts)
        forecast = series_from_darts(model.predict(len(val)), val.index)
        score = mape_pct(val, forecast)
        if score < best_score:
            best = params
            best_score = score
    return {**best, "validation_mape_pct": best_score, "implementation": "darts_prophet_with_country_holidays_de"}


def quantiles_from_samples(samples: TimeSeries, index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    values = samples.all_values(copy=False)[:, 0, :]
    quantile_values = np.quantile(values, [0.025, 0.1, 0.5, 0.9, 0.975], axis=1)
    keys = ["q025", "q10", "q50", "q90", "q975"]
    return {key: pd.Series(quantile_values[i], index=index) for i, key in enumerate(keys)}


def run(series: pd.Series) -> ModelOutput:
    selected = select_params(series)
    params = {k: v for k, v in selected.items() if k not in {"validation_mape_pct", "implementation"}}
    final_train = series.loc[TRAIN_START:FINAL_TRAIN_END]
    test_index = pd.date_range(TEST_START, TEST_END, freq="h")
    model = make_model(params)
    model.fit(to_darts(final_train))
    forecast_ts = model.predict(len(test_index))
    forecast = series_from_darts(forecast_ts, test_index)
    samples = model.predict(len(test_index), num_samples=500, random_state=42)
    qs = quantiles_from_samples(samples, test_index)
    return ModelOutput(
        name="prophet",
        forecast=forecast,
        runtime_seconds=0.0,
        hyperparameters=selected,
        lower_80=qs["q10"],
        upper_80=qs["q90"],
        lower_95=qs["q025"],
        upper_95=qs["q975"],
        q10=qs["q10"],
        q50=qs["q50"],
        q90=qs["q90"],
        fitted_model=model,
    )


def plot_prophet_decomposition(series: pd.Series, hyperparameters: dict[str, object], output_path: Path) -> None:
    params = {k: v for k, v in hyperparameters.items() if k not in {"validation_mape_pct", "implementation"}}
    model = make_model(params)
    model.fit(to_darts(series.loc[TRAIN_START:FINAL_TRAIN_END]))
    prediction = model.predict(len(pd.date_range(TEST_START, TEST_END, freq="h")))
    prophet_forecast = model.model.predict(pd.DataFrame({"ds": prediction.time_index}))
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    component_specs = [("trend", "Trend (MW)"), ("weekly", "Weekly component (MW)"), ("yearly", "Yearly component (MW)")]
    for ax, (column, label) in zip(axes, component_specs):
        ax.plot(pd.to_datetime(prophet_forecast["ds"]), prophet_forecast[column], color=MODEL_COLORS["prophet"], linewidth=1.6)
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[-1].set_xlabel("Time (UTC)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
