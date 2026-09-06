"""LightGBM on hand-built calendar, lag and rolling-window features.

The idea is the opposite of the deep-learning models: instead of asking a
network to discover structure, we tell the model directly what we know matters
for electricity load - what hour it is, what day of the week, whether it is a
public holiday, what the load was yesterday and a week ago, and what the
recent average has been. A gradient-boosted tree ensemble then learns how
those combine.

An important honesty point about the forecast itself. The test task is a
single 168-hour-ahead forecast made on 31 December 2019. A `lag_24h` feature
for, say, 7 January is the load on 6 January - which is inside the test week
and therefore not knowable at forecast time. So we forecast *recursively*:
the model predicts hour 1, that prediction is fed back in as history, it
predicts hour 2, and so on for 168 steps. No test observation is ever used as
an input. This makes LightGBM comparable to the other five models, all of
which also produce a genuine 168-hour-ahead forecast. We also compute the
easier "teacher-forced" variant (real lags supplied) purely as a diagnostic,
and report it separately in transcript.md so the gap is visible.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import (
    HORIZON,
    SEED,
    TEST_START,
    VALIDATION_ORIGINS,
    ModelForecast,
    calendar_frame,
    history_before,
    horizon_index,
    mape,
)

# The quantiles we need for the 80% band (0.1/0.9), the 95% band
# (0.025/0.975) and the median.
QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.5, 0.9, 0.975)

SHORT_LAGS: tuple[int, ...] = (24, 168)
ROLLING_WINDOWS: tuple[int, ...] = (24, 168)


def feature_names(yearly_lag_hours: int) -> list[str]:
    names = ["hour", "day_of_week", "month", "is_weekend", "is_public_holiday_de"]
    names += [f"lag_{h}h" for h in SHORT_LAGS]
    names.append(f"lag_{yearly_lag_hours}h")
    for w in ROLLING_WINDOWS:
        names += [f"roll_mean_{w}h", f"roll_std_{w}h"]
    return names


def build_features(series: pd.Series, yearly_lag_hours: int) -> pd.DataFrame:
    """Turn a load series into the model's feature table.

    Every lag and every rolling statistic is shifted by at least one hour, so
    a row never contains information from its own timestamp or later. That is
    the whole trick to not fooling yourself with a leaky feature.
    """
    frame = calendar_frame(series.index)
    for lag in SHORT_LAGS:
        frame[f"lag_{lag}h"] = series.shift(lag)
    frame[f"lag_{yearly_lag_hours}h"] = series.shift(yearly_lag_hours)
    past = series.shift(1)  # strictly earlier than the row's own timestamp
    for w in ROLLING_WINDOWS:
        frame[f"roll_mean_{w}h"] = past.rolling(w).mean()
        frame[f"roll_std_{w}h"] = past.rolling(w).std()
    frame["target"] = series.to_numpy()
    return frame


def _training_matrix(
    history: pd.Series, yearly_lag_hours: int
) -> tuple[pd.DataFrame, pd.Series]:
    frame = build_features(history, yearly_lag_hours).dropna()
    cols = feature_names(yearly_lag_hours)
    return frame[cols], frame["target"]


def _row_features(
    buffer: pd.Series, stamp: pd.Timestamp, yearly_lag_hours: int
) -> np.ndarray:
    """Features for one future hour, from whatever history we have so far."""
    cal = calendar_frame(pd.DatetimeIndex([stamp], tz="UTC")).iloc[0]
    values = [
        cal["hour"], cal["day_of_week"], cal["month"],
        cal["is_weekend"], cal["is_public_holiday_de"],
    ]
    for lag in SHORT_LAGS:
        values.append(buffer.loc[stamp - pd.Timedelta(hours=lag)])
    values.append(buffer.loc[stamp - pd.Timedelta(hours=yearly_lag_hours)])
    for w in ROLLING_WINDOWS:
        recent = buffer.iloc[-w:]
        values.append(recent.mean())
        values.append(recent.std(ddof=1))
    return np.asarray(values, dtype=float)


def recursive_forecast(
    models: dict[str, LGBMRegressor],
    history: pd.Series,
    origin: pd.Timestamp,
    yearly_lag_hours: int,
    horizon: int = HORIZON,
) -> dict[str, np.ndarray]:
    """Roll the point model forward 168 times, reading quantiles off the way.

    `models` must contain "point"; any other keys are quantile models that are
    evaluated on the same feature rows, so all outputs describe one single
    forecast trajectory.
    """
    buffer = history.copy()
    stamps = horizon_index(origin, horizon)
    out: dict[str, list[float]] = {k: [] for k in models}
    for stamp in stamps:
        x = _row_features(buffer, stamp, yearly_lag_hours).reshape(1, -1)
        for key, model in models.items():
            out[key].append(float(model.predict(x)[0]))
        buffer = pd.concat(
            [buffer, pd.Series([out["point"][-1]], index=[stamp])]
        )
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def teacher_forced_forecast(
    model: LGBMRegressor,
    series: pd.Series,
    origin: pd.Timestamp,
    yearly_lag_hours: int,
    horizon: int = HORIZON,
) -> np.ndarray:
    """Diagnostic only: let the model see the real recent load at every step."""
    frame = build_features(series, yearly_lag_hours)
    rows = frame.loc[horizon_index(origin, horizon), feature_names(yearly_lag_hours)]
    return np.asarray(model.predict(rows), dtype=float)


def _make_point_model(params: dict[str, object]) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        **params,
    )


def _make_quantile_model(params: dict[str, object], q: float) -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile",
        alpha=q,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        **params,
    )


CANDIDATES: tuple[dict[str, object], ...] = (
    {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
    {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20},
    {"n_estimators": 800, "learning_rate": 0.03, "num_leaves": 127, "min_child_samples": 40},
    {"n_estimators": 1500, "learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 40},
)

YEARLY_LAG_CHOICES: tuple[int, ...] = (8736, 8760)  # 364 days (day-aligned) vs 365


def run(
    series: pd.Series,
    origins: Sequence[pd.Timestamp] = VALIDATION_ORIGINS,
) -> ModelForecast:
    started = time.perf_counter()

    selection: list[dict[str, object]] = []
    best: tuple[float, dict[str, object], int] | None = None
    for yearly_lag in YEARLY_LAG_CHOICES:
        for params in CANDIDATES:
            scores: list[float] = []
            for origin in origins:
                history = history_before(series, origin)
                x, y = _training_matrix(history, yearly_lag)
                model = _make_point_model(params).fit(x, y)
                pred = recursive_forecast(
                    {"point": model}, history, origin, yearly_lag
                )["point"]
                actual = series.loc[horizon_index(origin)].to_numpy()
                scores.append(mape(actual, pred))
            mean_score = float(np.mean(scores))
            selection.append(
                {
                    "yearly_lag_hours": yearly_lag,
                    **params,
                    "val_mape_pct": mean_score,
                    "per_origin_mape_pct": [round(s, 3) for s in scores],
                }
            )
            if best is None or mean_score < best[0]:
                best = (mean_score, dict(params), yearly_lag)

    assert best is not None
    val_mape, best_params, best_lag = best

    # Refit the chosen configuration on train + validation, then forecast test.
    history = history_before(series, TEST_START)
    x, y = _training_matrix(history, best_lag)
    point_model = _make_point_model(best_params).fit(x, y)
    models: dict[str, LGBMRegressor] = {"point": point_model}
    for q in QUANTILES:
        models[f"q{q}"] = _make_quantile_model(best_params, q).fit(x, y)

    paths = recursive_forecast(models, history, TEST_START, best_lag)
    index = horizon_index(TEST_START)
    point = pd.Series(paths["point"], index=index, name="lightgbm")

    # Quantile crossing is possible because the five quantile models are fitted
    # independently; sorting each hour's quantiles fixes it without changing
    # any individual model.
    stacked = np.vstack([paths[f"q{q}"] for q in QUANTILES])
    stacked = np.sort(stacked, axis=0)
    quantiles = {
        q: pd.Series(stacked[i], index=index) for i, q in enumerate(QUANTILES)
    }

    teacher = teacher_forced_forecast(point_model, series, TEST_START, best_lag)

    importance = pd.Series(
        point_model.booster_.feature_importance(importance_type="gain"),
        index=feature_names(best_lag),
    ).sort_values(ascending=False)

    return ModelForecast(
        name="lightgbm",
        point=point,
        quantiles=quantiles,
        hyperparameters={
            **best_params,
            "yearly_lag_hours": best_lag,
            "objective": "regression (L2) for the point path",
            "quantile_objective_alphas": list(QUANTILES),
            "forecast_mode": "recursive 168-step",
            "features": feature_names(best_lag),
        },
        runtime_seconds=time.perf_counter() - started,
        validation_mape=val_mape,
        selection_log=selection,
        notes="teacher-forced diagnostic reported separately",
    ), {
        "teacher_forced": pd.Series(teacher, index=index),
        "feature_importance_gain": importance,
    }
