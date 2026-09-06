from __future__ import annotations

from time import perf_counter
from typing import Any
import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from .common import ModelResult, QUANTILES, SEED, TRAIN_END, mape, validation_blocks

HOLIDAYS = holidays.Germany(years=range(2015, 2022))
FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday_de",
    "lag_24h",
    "lag_168h",
    "lag_8736h",
    "mean_24h",
    "std_24h",
    "mean_168h",
    "std_168h",
]


def feature_frame(series: pd.Series) -> pd.DataFrame:
    index = series.index
    frame = pd.DataFrame(
        {
            "hour": index.hour,
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "is_public_holiday_de": [int(t.date() in HOLIDAYS) for t in index],
        },
        index=index,
    )
    for lag in (24, 168, 8736):
        frame[f"lag_{lag}h"] = series.shift(lag)
    past = series.shift(1)
    for window in (24, 168):
        frame[f"mean_{window}h"] = past.rolling(window).mean()
        frame[f"std_{window}h"] = past.rolling(window).std(ddof=1)
    return frame[FEATURES]


def recursive_features(history: list[float], timestamp: pd.Timestamp) -> np.ndarray:
    if len(history) < 8736:
        raise ValueError("Insufficient pre-origin history")
    return np.array(
        [
            timestamp.hour,
            timestamp.dayofweek,
            timestamp.month,
            int(timestamp.dayofweek >= 5),
            int(timestamp.date() in HOLIDAYS),
            history[-24],
            history[-168],
            history[-8736],
            np.mean(history[-24:]),
            np.std(history[-24:], ddof=1),
            np.mean(history[-168:]),
            np.std(history[-168:], ddof=1),
        ],
        dtype=float,
    )


def fit(series: pd.Series, config: dict[str, Any], quantile: float = 0.5) -> lgb.LGBMRegressor:
    features = feature_frame(series)
    valid = features.notna().all(axis=1)
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        verbosity=-1,
        random_state=SEED,
        n_jobs=4,
        deterministic=True,
        force_col_wise=True,
        learning_rate=0.05,
        min_child_samples=60,
        **config,
    )
    model.fit(features.loc[valid], series.loc[valid])
    return model


def forecast(
    model: lgb.LGBMRegressor, history: pd.Series, timestamps: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    state = history.to_list()
    design: list[np.ndarray] = []
    predictions: list[float] = []
    for timestamp in timestamps:
        row = recursive_features(state, timestamp)
        # The state advances with forecasts, never with observations inside the horizon.
        prediction = float(model.booster_.predict(row[None, :], num_threads=1)[0])
        state.append(prediction)
        design.append(row)
        predictions.append(prediction)
    return np.array(predictions), np.array(design)


def sample_paths(
    models: dict[float, lgb.LGBMRegressor],
    history: pd.Series,
    timestamps: pd.DatetimeIndex,
    n_paths: int = 1000,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(SEED)
    state = np.empty((n_paths, 8736 + len(timestamps)))
    state[:, :8736] = history.iloc[-8736:].to_numpy()[None, :]
    z_knots = norm.ppf(QUANTILES)
    crossed = 0
    for step, timestamp in enumerate(timestamps):
        end = 8736 + step
        calendar = [
            timestamp.hour,
            timestamp.dayofweek,
            timestamp.month,
            int(timestamp.dayofweek >= 5),
            int(timestamp.date() in HOLIDAYS),
        ]
        design = np.column_stack(
            [
                *[np.full(n_paths, x) for x in calendar],
                state[:, end - 24],
                state[:, end - 168],
                state[:, end - 8736],
                state[:, end - 24 : end].mean(axis=1),
                state[:, end - 24 : end].std(axis=1, ddof=1),
                state[:, end - 168 : end].mean(axis=1),
                state[:, end - 168 : end].std(axis=1, ddof=1),
            ]
        )
        raw = np.column_stack(
            [models[float(q)].booster_.predict(design, num_threads=1) for q in QUANTILES]
        )
        crossed += int(np.any(np.diff(raw, axis=1) < 0, axis=1).sum())
        ordered = np.sort(raw, axis=1)
        # Linear interpolation on a Gaussian-quantile axis also defines explicit tails.
        z = rng.standard_normal(n_paths)
        lower = np.clip(np.searchsorted(z_knots, z) - 1, 0, 3)
        rows = np.arange(n_paths)
        fraction = (z - z_knots[lower]) / (z_knots[lower + 1] - z_knots[lower])
        draw = ordered[rows, lower] + fraction * (ordered[rows, lower + 1] - ordered[rows, lower])
        state[:, end] = draw
    return state[:, 8736:].T, crossed


def run(pretest: pd.Series) -> ModelResult:
    start = perf_counter()
    train = pretest.loc[: TRAIN_END - pd.Timedelta(hours=1)]
    candidates = [{"num_leaves": 15, "n_estimators": 200}, {"num_leaves": 31, "n_estimators": 400}]
    validation: list[dict[str, Any]] = []
    for config in candidates:
        model = fit(train, config)
        actuals, forecasts = [], []
        for history, target in validation_blocks(pretest):
            prediction, _ = forecast(model, history, target.index)
            actuals.extend(target.to_numpy())
            forecasts.extend(prediction)
        validation.append(
            {"configuration": config, "mape_pct": mape(np.array(actuals), np.array(forecasts))}
        )
    best = min(validation, key=lambda item: item["mape_pct"])["configuration"]
    final_models = {float(q): fit(pretest, best, float(q)) for q in QUANTILES}
    timestamps = pd.date_range("2020-01-01", periods=168, freq="h")
    point, _ = forecast(final_models[0.5], pretest, timestamps)
    paths, crossed = sample_paths(final_models, pretest, timestamps)
    quantiles = np.quantile(paths, QUANTILES, axis=1).T
    importance = final_models[0.5].booster_.feature_importance(importance_type="gain")
    return ModelResult(
        "lightgbm",
        point,
        quantiles,
        perf_counter() - start,
        {
            **best,
            "objective": "quantile",
            "quantiles": QUANTILES.tolist(),
            "learning_rate": 0.05,
            "min_child_samples": 60,
            "seed": SEED,
            "features": FEATURES,
            "annual_lag_hours": 8736,
            "point": "recursive conditional-median path",
            "quantile_recursion": "1000 simulated load paths",
            "n_paths": 1000,
            "inverse_cdf": "monotone rearrangement then linear interpolation/extrapolation on normal quantile axis",
            "innovation_dependence": "independent conditional draws",
        },
        validation,
        {
            "feature_gain": dict(zip(FEATURES, importance.tolist())),
            "crossed_path_hours_before_rearrangement": crossed,
            "point_vs_marginal_median_max_mw": float(np.max(np.abs(point - quantiles[:, 2]))),
            "train_rows_after_lag_warmup": len(train) - 8736,
            "refit_rows_after_lag_warmup": len(pretest) - 8736,
        },
    )
