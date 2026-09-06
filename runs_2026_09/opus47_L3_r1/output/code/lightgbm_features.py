"""LightGBM on engineered calendar and lag features.

Features per row t:
  - hour, day_of_week, month, is_weekend, is_public_holiday_de
  - lag_24h, lag_168h, lag_8760h (all shifted from the actual series)
  - roll_mean_24h, roll_std_24h, roll_mean_168h, roll_std_168h
    (rolling stats over t-1 .. t-N, causal)

Because the model consumes only lags and rolling windows computed from
observed values, the 168-hour test window can be forecast in one shot: for
each test hour the required lags all lie strictly before 2020-01-01.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import holidays as hol
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import RANDOM_SEED, Splits, mape

FEATURE_COLS: list[str] = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday_de",
    "lag_24h",
    "lag_168h",
    "lag_8760h",
    "roll_mean_24h",
    "roll_std_24h",
    "roll_mean_168h",
    "roll_std_168h",
]


def build_frame(series: pd.Series) -> pd.DataFrame:
    """Return a DataFrame of features + target aligned with series.index.

    Rolling windows are computed CAUSALLY (shift 1 before rolling) so a row
    at time t sees only observations strictly before t. Rows with any NaN
    lag are dropped by the caller after building the frame.
    """
    idx = series.index
    de_hols = hol.country_holidays("DE", years=range(idx.min().year, idx.max().year + 1))
    dates = idx.tz_convert(None).normalize()  # naive dates for holidays lookup
    is_hol = np.array([d.date() in de_hols for d in dates], dtype=int)

    df = pd.DataFrame(index=idx)
    df["y"] = series.astype(float)
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_public_holiday_de"] = is_hol
    df["lag_24h"] = series.shift(24)
    df["lag_168h"] = series.shift(168)
    df["lag_8760h"] = series.shift(8760)
    # Causal rolling stats.
    lagged = series.shift(1)
    df["roll_mean_24h"] = lagged.rolling(24, min_periods=24).mean()
    df["roll_std_24h"] = lagged.rolling(24, min_periods=24).std()
    df["roll_mean_168h"] = lagged.rolling(168, min_periods=168).mean()
    df["roll_std_168h"] = lagged.rolling(168, min_periods=168).std()
    return df


CANDIDATE_HP: list[dict[str, Any]] = [
    dict(n_estimators=800, learning_rate=0.05, num_leaves=63, min_child_samples=20),
    dict(n_estimators=1200, learning_rate=0.03, num_leaves=127, min_child_samples=20),
    dict(n_estimators=500, learning_rate=0.08, num_leaves=31, min_child_samples=50),
]


@dataclass
class LGBMResult:
    """Container returned to the orchestrator."""

    mean: pd.Series
    intervals: pd.DataFrame
    feature_importances: pd.Series
    hyperparameters: dict[str, Any]


def _fit_predict_mean(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    hp: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, LGBMRegressor]:
    model = LGBMRegressor(
        objective="regression",
        random_state=seed,
        verbose=-1,
        **hp,
    )
    model.fit(frame.loc[train_mask, FEATURE_COLS], frame.loc[train_mask, "y"])
    pred = model.predict(frame.loc[test_mask, FEATURE_COLS])
    return pred, model


def _fit_predict_quantile(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    hp: dict[str, Any],
    alpha: float,
    seed: int,
) -> np.ndarray:
    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        random_state=seed,
        verbose=-1,
        **hp,
    )
    model.fit(frame.loc[train_mask, FEATURE_COLS], frame.loc[train_mask, "y"])
    return model.predict(frame.loc[test_mask, FEATURE_COLS])


def forecast(splits: Splits) -> LGBMResult:
    frame = build_frame(splits.series).dropna(subset=FEATURE_COLS + ["y"]).copy()

    train_mask = frame.index <= splits.train.index[-1]
    val_mask = frame.index.isin(splits.val.index)
    train_val_mask = frame.index <= splits.val.index[-1]
    test_mask = frame.index.isin(splits.test.index)

    # Small hyperparameter sweep on validation MAPE.
    best_hp = None
    best_mape = float("inf")
    for hp in CANDIDATE_HP:
        pred, _ = _fit_predict_mean(frame, train_mask, val_mask, hp, RANDOM_SEED)
        m = mape(frame.loc[val_mask, "y"].to_numpy(), pred)
        if m < best_mape:
            best_mape = m
            best_hp = hp
    assert best_hp is not None

    # Refit on train+val at best hyperparameters, then predict test.
    pred_test, model_final = _fit_predict_mean(
        frame, train_val_mask, test_mask, best_hp, RANDOM_SEED
    )
    mean = pd.Series(pred_test, index=frame.index[test_mask], name="lightgbm")

    # Quantile fits for 80% and 95% intervals.
    q10 = _fit_predict_quantile(frame, train_val_mask, test_mask, best_hp, 0.10, RANDOM_SEED)
    q90 = _fit_predict_quantile(frame, train_val_mask, test_mask, best_hp, 0.90, RANDOM_SEED)
    q025 = _fit_predict_quantile(frame, train_val_mask, test_mask, best_hp, 0.025, RANDOM_SEED)
    q975 = _fit_predict_quantile(frame, train_val_mask, test_mask, best_hp, 0.975, RANDOM_SEED)
    intervals = pd.DataFrame({
        "lower_80": q10,
        "upper_80": q90,
        "lower_95": q025,
        "upper_95": q975,
    }, index=frame.index[test_mask])

    fi = pd.Series(
        model_final.feature_importances_, index=FEATURE_COLS, name="gain"
    ).sort_values(ascending=False)
    return LGBMResult(
        mean=mean,
        intervals=intervals,
        feature_importances=fi,
        hyperparameters={**best_hp, "val_mape_pct": best_mape},
    )
