"""Forecast models for a fixed 168-hour ahead horizon.

Every model exposes ``fit(history)`` / ``predict(index)``. All predictor
information is either calendar-derived (knowable indefinitely ahead) or a lag
of at least ``horizon`` hours, so a forecast for the target week uses no
observation at or after the forecast origin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast.data import build_calendar

HORIZON = 168


class Forecaster(ABC):
    """Common interface: fit on history, predict for a future hourly index."""

    name: str

    @abstractmethod
    def fit(self, history: pd.Series) -> "Forecaster": ...

    @abstractmethod
    def predict(self, index: pd.DatetimeIndex) -> pd.Series: ...


@dataclass
class SeasonalNaive(Forecaster):
    """Repeat the load observed ``lag_hours`` earlier."""

    lag_hours: int = HORIZON
    name: str = "seasonal_naive_168h"
    _history: pd.Series = field(init=False, repr=False)

    def fit(self, history: pd.Series) -> "SeasonalNaive":
        self._history = history
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        source = index - pd.Timedelta(hours=self.lag_hours)
        return pd.Series(self._history.reindex(source).to_numpy(), index=index, name=self.name)


@dataclass
class YearAgoNaive(Forecaster):
    """Same hour 364 days earlier (preserves weekday), rescaled to the recent level.

    The scale factor is the ratio of mean load over the 8-to-4 weeks before the
    forecast origin to the same calendar window a year earlier. That window is
    deliberately chosen to end four weeks before the origin so the Christmas
    shutdown does not contaminate the level estimate.
    """

    name: str = "year_ago_naive_364d"
    _history: pd.Series = field(init=False, repr=False)
    _scale: float = field(init=False, default=1.0)

    def fit(self, history: pd.Series) -> "YearAgoNaive":
        self._history = history
        end = history.index[-1]
        recent = history.loc[end - pd.Timedelta(days=56) : end - pd.Timedelta(days=28)]
        prior = history.loc[end - pd.Timedelta(days=56 + 364) : end - pd.Timedelta(days=28 + 364)]
        self._scale = float(recent.mean() / prior.mean()) if len(prior) else 1.0
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        source = index - pd.Timedelta(days=364)
        values = self._history.reindex(source).to_numpy() * self._scale
        return pd.Series(values, index=index, name=self.name)


@dataclass
class HolidayAdjustedNaive(Forecaster):
    """Seasonal naive corrected by the historical Jan-week / Christmas-week ratio.

    The week preceding a first-week-of-January forecast is always the Christmas
    shutdown, which makes the plain lag-168 baseline biased low by a stable
    amount. This model estimates that bias per hour-of-week from prior years.
    """

    name: str = "holiday_adjusted_naive"
    _history: pd.Series = field(init=False, repr=False)
    _ratio: pd.Series = field(init=False, repr=False)

    def fit(self, history: pd.Series) -> "HolidayAdjustedNaive":
        self._history = history
        cal = build_calendar(history.index)
        # Ratio of each hour to the same hour one week earlier, measured only on
        # target-week hours of previous years (year-end offset 0..6).
        lagged = history.shift(HORIZON)
        target_like = cal["yearend_day"].between(0, 6) | (
            (cal["month"] == 1) & (cal["doy"].between(1, 7))
        )
        ratio = (history / lagged)[target_like & lagged.notna()]
        hour_of_week = cal["hour_of_week"][ratio.index]
        by_how = ratio.groupby(hour_of_week.to_numpy()).median()
        self._ratio = by_how.reindex(range(168)).fillna(float(ratio.median()))
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        base = self._history.reindex(index - pd.Timedelta(hours=HORIZON)).to_numpy()
        how = build_calendar(index)["hour_of_week"].to_numpy()
        return pd.Series(base * self._ratio.reindex(how).to_numpy(), index=index, name=self.name)


def _design_matrix(cal: pd.DataFrame) -> pd.DataFrame:
    """Build the linear model's design matrix from calendar attributes."""
    parts: list[pd.DataFrame] = []
    parts.append(
        pd.get_dummies(cal["hour_of_week"], prefix="how")
        .reindex(columns=[f"how_{i}" for i in range(168)], fill_value=0)
        .astype(float)
        .iloc[:, 1:]
    )
    # Holidays reshape the daily profile, so interact the holiday population
    # weight with hour-of-day rather than adding a flat level shift.
    hour_dummies = (
        pd.get_dummies(cal["hour"], prefix="h")
        .reindex(columns=[f"h_{i}" for i in range(24)], fill_value=0)
        .astype(float)
    )
    parts.append(hour_dummies.mul(cal["hol_weight"].to_numpy(), axis=0).add_prefix("holx_"))
    parts.append(
        pd.DataFrame(
            {
                "hol_prev": cal["hol_weight_prev"].to_numpy(),
                "hol_next": cal["hol_weight_next"].to_numpy(),
            },
            index=cal.index,
        )
    )
    ye = cal["yearend_day"].where(cal["yearend_day"] != 99, np.nan)
    ye_dummies = (
        pd.get_dummies(ye, prefix="ye")
        .reindex(columns=[f"ye_{float(i)}" for i in range(-8, 6)], fill_value=0)
        .astype(float)
    )
    parts.append(ye_dummies)
    parts.append(cal[[c for c in cal.columns if c.startswith(("yr_sin", "yr_cos"))]])
    parts.append(cal[["trend_years"]])
    design = pd.concat(parts, axis=1)
    design.insert(0, "const", 1.0)
    return design


@dataclass
class LinearProfileModel(Forecaster):
    """Ridge-regularised OLS on log load with hour-of-week and holiday profiles."""

    ridge: float = 1.0
    name: str = "linear_profile"
    _coef: np.ndarray = field(init=False, repr=False)
    _columns: list[str] = field(init=False, repr=False)

    def fit(self, history: pd.Series) -> "LinearProfileModel":
        design = _design_matrix(build_calendar(history.index))
        self._columns = list(design.columns)
        x = design.to_numpy(dtype=float)
        y = np.log(history.to_numpy(dtype=float))
        penalty = np.eye(x.shape[1]) * self.ridge
        penalty[0, 0] = 0.0  # never shrink the intercept
        self._coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        design = _design_matrix(build_calendar(index)).reindex(
            columns=self._columns, fill_value=0.0
        )
        values = np.exp(design.to_numpy(dtype=float) @ self._coef)
        return pd.Series(values, index=index, name=self.name)


def _lag_features(
    series: pd.Series, index: pd.DatetimeIndex, min_lag: int = HORIZON
) -> pd.DataFrame:
    """Lagged load features, none drawing on data newer than ``min_lag`` hours.

    ``series`` may cover the forecast index; only shifted values are read, so a
    forecast at horizon ``min_lag`` never sees an unobserved value.
    """
    if min_lag < HORIZON:
        raise ValueError("min_lag must be at least the forecast horizon")
    base = series.reindex(
        pd.date_range(series.index[0], max(series.index[-1], index[-1]), freq="h", tz="UTC")
    )
    out = pd.DataFrame(index=index)
    for lag in (min_lag, min_lag + 24, min_lag + 168, min_lag + 336):
        out[f"lag_{lag}"] = base.shift(lag).reindex(index).to_numpy()
    for days in (364, 371, 365):
        out[f"lag_{days}d"] = base.shift(days * 24).reindex(index).to_numpy()
    # Level anchors: a 4-week mean ending at the origin, and a cleaner 8-to-4
    # week mean that excludes the Christmas shutdown for a January forecast.
    shifted = base.shift(min_lag)
    out["roll_28d"] = shifted.rolling(24 * 28, min_periods=24 * 7).mean().reindex(index).to_numpy()
    out["roll_28d_lag28d"] = (
        base.shift(min_lag + 24 * 28)
        .rolling(24 * 28, min_periods=24 * 7)
        .mean()
        .reindex(index)
        .to_numpy()
    )
    out["lag_ratio_yoy"] = (
        out["roll_28d_lag28d"]
        / base.shift(min_lag + 24 * 28 + 364 * 24)
        .rolling(24 * 28, min_periods=24 * 7)
        .mean()
        .reindex(index)
        .to_numpy()
    )
    return out


@dataclass
class LightGBMModel(Forecaster):
    """Gradient-boosted trees on log load with calendar and long-lag features."""

    n_estimators: int = 900
    learning_rate: float = 0.04
    num_leaves: int = 64
    min_child_samples: int = 40
    seed: int = 7
    name: str = "lightgbm"
    _model: lgb.LGBMRegressor = field(init=False, repr=False)
    _history: pd.Series = field(init=False, repr=False)
    _columns: list[str] = field(init=False, repr=False)

    CATEGORICAL = ["hour", "dow", "month"]

    def _features(self, series: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
        cal = build_calendar(index)
        cal = cal.drop(columns=[c for c in cal.columns if c.startswith(("yr_sin", "yr_cos"))])
        frame = pd.concat([cal, _lag_features(series, index)], axis=1)
        # Ratios to the level anchor let the trees split on shape, not level.
        for lag in ("lag_168", "lag_336"):
            frame[f"{lag}_over_roll"] = frame[lag] / frame["roll_28d"]
        for col in self.CATEGORICAL:
            frame[col] = frame[col].astype("category")
        return frame

    def fit(self, history: pd.Series) -> "LightGBMModel":
        self._history = history
        frame = self._features(history, history.index)
        target = np.log(history.to_numpy(dtype=float))
        usable = frame["lag_336"].notna().to_numpy() & frame["roll_28d"].notna().to_numpy()
        self._columns = list(frame.columns)
        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=self.seed,
            verbose=-1,
        )
        self._model.fit(frame[usable], target[usable], categorical_feature=self.CATEGORICAL)
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        frame = self._features(self._history, index).reindex(columns=self._columns)
        values = np.exp(self._model.predict(frame))
        return pd.Series(values, index=index, name=self.name)


@dataclass
class Ensemble(Forecaster):
    """Weighted geometric mean of member forecasts."""

    members: list[Forecaster]
    weights: list[float]
    name: str = "ensemble"

    def fit(self, history: pd.Series) -> "Ensemble":
        for member in self.members:
            member.fit(history)
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        total = sum(self.weights)
        logs = [w * np.log(m.predict(index).to_numpy()) for m, w in zip(self.members, self.weights)]
        return pd.Series(np.exp(np.sum(logs, axis=0) / total), index=index, name=self.name)
