from __future__ import annotations

import time
from typing import Any

import holidays
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import ModelResult, QUANTILES, SEED, mape, validation_blocks

FEATURES = ["hour", "day_of_week", "month", "is_weekend", "is_public_holiday_de",
            "lag_24h", "lag_168h", "lag_8760h", "rolling_mean_24h", "rolling_std_24h",
            "rolling_mean_168h", "rolling_std_168h"]


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_localize("UTC").tz_convert("Europe/Berlin")
    holiday_dates = holidays.Germany(years=range(local.year.min(), local.year.max() + 1))
    return pd.DataFrame({
        "hour": local.hour, "day_of_week": local.dayofweek, "month": local.month,
        "is_weekend": (local.dayofweek >= 5).astype(int),
        "is_public_holiday_de": [int(day in holiday_dates) for day in local.date],
    }, index=index)


def feature_frame(y: pd.Series) -> pd.DataFrame:
    x = calendar(y.index)
    for hours in (24, 168, 8760):
        x[f"lag_{hours}h"] = y.shift(hours)
    # Excluding the current target is stricter than merely excluding later rows.
    past = y.shift(1)
    for hours in (24, 168):
        x[f"rolling_mean_{hours}h"] = past.rolling(hours).mean()
        x[f"rolling_std_{hours}h"] = past.rolling(hours).std(ddof=1)
    return x[FEATURES]


def recursive_forecast(models: dict[float, Any], history: pd.Series,
                       future: pd.DatetimeIndex) -> np.ndarray:
    if len(history) < 8760 or future[0] != history.index[-1] + pd.Timedelta(hours=1):
        raise ValueError("Forecast must follow a complete history with all lags available.")
    if not future.equals(pd.date_range(future[0], periods=len(future), freq="h")):
        raise ValueError("Forecast timestamps must be contiguous hourly instants.")
    known: list[float] = history.to_list()
    time_values = calendar(future).to_numpy(dtype=float)
    quantiles = sorted(models)
    result = np.empty((len(future), len(quantiles)))
    for i in range(len(future)):
        row = list(time_values[i]) + [known[-lag] for lag in (24, 168, 8760)]
        for hours in (24, 168):
            window = np.asarray(known[-hours:])
            row.extend([float(window.mean()), float(window.std(ddof=1))])
        x = pd.DataFrame([row], columns=FEATURES)
        prediction = np.array([models[q].predict(x)[0] for q in quantiles])
        median_index = quantiles.index(0.5)
        median = prediction[median_index]
        result[i, :median_index] = np.minimum(np.sort(prediction[:median_index]), median)
        result[i, median_index] = median
        result[i, median_index + 1:] = np.maximum(np.sort(prediction[median_index + 1:]), median)
        known.append(float(median))
    return result


def estimator(leaves: int, quantile: float) -> LGBMRegressor:
    return LGBMRegressor(objective="quantile", alpha=quantile, n_estimators=300,
                         learning_rate=0.05, num_leaves=leaves, min_child_samples=60,
                         reg_lambda=1.0, random_state=SEED, n_jobs=1, verbosity=-1,
                         deterministic=True, force_col_wise=True)


def run(train: pd.Series, pretest: pd.Series, future: pd.DatetimeIndex) -> ModelResult:
    start = time.perf_counter()
    x = feature_frame(train).iloc[8760:]
    y = train.iloc[8760:]
    candidates: list[dict[str, Any]] = []
    forecasts: list[np.ndarray] = []
    for leaves in (15, 31):
        model = estimator(leaves, 0.5).fit(x, y)
        prediction = np.concatenate([recursive_forecast({0.5: model}, history, target.index)[:, 0]
                                     for history, target in validation_blocks(pretest, len(train))])
        score = mape(pretest.iloc[len(train):].to_numpy(), prediction)
        candidates.append({"num_leaves": leaves, "validation_mape_pct": score})
        forecasts.append(prediction)
        print(f"lightgbm leaves={leaves} validation MAPE={score:.3f}%", flush=True)
    chosen = int(np.argmin([c["validation_mape_pct"] for c in candidates]))
    leaves = candidates[chosen]["num_leaves"]
    final_x = feature_frame(pretest).iloc[8760:]
    models = {q: estimator(leaves, q).fit(final_x, pretest.iloc[8760:]) for q in QUANTILES}
    quantiles = recursive_forecast(models, pretest, future)
    importance = dict(zip(FEATURES, models[0.5].booster_.feature_importance(importance_type="gain").tolist()))
    params = {"num_leaves": leaves, "n_estimators": 300, "learning_rate": 0.05,
              "min_child_samples": 60, "reg_lambda": 1.0, "quantiles": QUANTILES,
              "seed": SEED, "features": FEATURES, "calendar_timezone": "Europe/Berlin",
              "training_feature_warmup_hours": 8760, "final_training_rows": len(final_x),
              "forecast_method": "recursive median, median-preserving lower/upper quantile rearrangement",
              "uncertainty": "conditional quantiles on recursive median predictor path"}
    return ModelResult("lightgbm", quantiles[:, 2], quantiles, forecasts[chosen], params,
                       candidates, time.perf_counter() - start, {"feature_importance_gain": importance})
