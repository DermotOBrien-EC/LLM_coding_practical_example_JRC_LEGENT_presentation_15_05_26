from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import holidays
import numpy as np
import pandas as pd

LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
TIME_COLUMN = "utc_timestamp"

TRAIN_START = pd.Timestamp("2015-01-01 00:00:00")
TRAIN_END = pd.Timestamp("2019-09-30 23:00:00")
VAL_START = pd.Timestamp("2019-10-01 00:00:00")
VAL_END = pd.Timestamp("2019-12-31 23:00:00")
TEST_START = pd.Timestamp("2020-01-01 00:00:00")
TEST_END = pd.Timestamp("2020-01-07 23:00:00")
FINAL_TRAIN_END = VAL_END
SEED = 42

MODEL_ORDER = ["naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"]
MODEL_LABELS = {
    "naive": "Seasonal naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer",
}
MODEL_COLORS = {
    "naive": "#2a78d6",
    "sarima": "#eb6834",
    "prophet": "#1baf7a",
    "lightgbm": "#eda100",
    "nbeats": "#e87ba4",
    "patchtst": "#008300",
}


@dataclass(slots=True)
class ModelOutput:
    name: str
    forecast: pd.Series
    runtime_seconds: float
    hyperparameters: dict[str, object]
    lower_80: pd.Series | None = None
    upper_80: pd.Series | None = None
    lower_95: pd.Series | None = None
    upper_95: pd.Series | None = None
    q10: pd.Series | None = None
    q50: pd.Series | None = None
    q90: pd.Series | None = None
    feature_importance: pd.Series | None = None
    fitted_model: object | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MetricsRow:
    name: str
    mape_test_pct: float
    rmse_test_mw: float
    mae_test_mw: float
    mape_jan1_pct: float
    mape_jan2_to_jan7_pct: float
    runtime_seconds: float
    hyperparameters: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mape_test_pct": self.mape_test_pct,
            "rmse_test_mw": self.rmse_test_mw,
            "mae_test_mw": self.mae_test_mw,
            "mape_jan1_pct": self.mape_jan1_pct,
            "mape_jan2_to_jan7_pct": self.mape_jan2_to_jan7_pct,
            "runtime_seconds": self.runtime_seconds,
            "hyperparameters": self.hyperparameters,
        }


def timed(call: Callable[[], ModelOutput]) -> ModelOutput:
    start = perf_counter()
    output = call()
    measured = perf_counter() - start
    output.runtime_seconds = measured
    return output


def load_data(path: str | Path = "opsd_de_load.csv") -> pd.DataFrame:
    raw = pd.read_csv(path)
    expected_columns = [TIME_COLUMN, LOAD_COLUMN]
    if raw.columns.tolist() != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, got {raw.columns.tolist()}")
    raw[TIME_COLUMN] = pd.to_datetime(raw[TIME_COLUMN], utc=True).dt.tz_convert(None)
    raw = raw.set_index(TIME_COLUMN).sort_index()
    if len(raw) != 50_400:
        raise ValueError(f"Expected 50,400 rows, got {len(raw):,}")
    if raw[LOAD_COLUMN].isna().any():
        raise ValueError("Load column contains NaN values")
    expected_index = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
    if not raw.index.equals(expected_index):
        raise ValueError("Hourly timestamp index has gaps or duplicates")
    if raw.index[0] != pd.Timestamp("2015-01-01 00:00:00"):
        raise ValueError(f"Unexpected first timestamp {raw.index[0]}")
    if raw.index[-1] != pd.Timestamp("2020-09-30 23:00:00"):
        raise ValueError(f"Unexpected last timestamp {raw.index[-1]}")
    return raw


def load_series(df: pd.DataFrame) -> pd.Series:
    return df[LOAD_COLUMN].astype(float)


def slice_range(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return series.loc[start:end]


def train_series(series: pd.Series) -> pd.Series:
    return slice_range(series, TRAIN_START, TRAIN_END)


def validation_series(series: pd.Series) -> pd.Series:
    return slice_range(series, VAL_START, VAL_END)


def final_training_series(series: pd.Series) -> pd.Series:
    return slice_range(series, TRAIN_START, FINAL_TRAIN_END)


def test_series(series: pd.Series) -> pd.Series:
    return slice_range(series, TEST_START, TEST_END)


def mape_pct(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual_arr - pred_arr) / actual_arr)) * 100.0)


def rmse_mw(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))


def mae_mw(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual_arr - pred_arr)))


def evaluate_output(output: ModelOutput, actual: pd.Series) -> MetricsRow:
    forecast = output.forecast.reindex(actual.index)
    jan1 = actual.index.date == pd.Timestamp("2020-01-01").date()
    rest = ~jan1
    return MetricsRow(
        name=output.name,
        mape_test_pct=mape_pct(actual, forecast),
        rmse_test_mw=rmse_mw(actual, forecast),
        mae_test_mw=mae_mw(actual, forecast),
        mape_jan1_pct=mape_pct(actual.loc[jan1], forecast.loc[jan1]),
        mape_jan2_to_jan7_pct=mape_pct(actual.loc[rest], forecast.loc[rest]),
        runtime_seconds=float(output.runtime_seconds),
        hyperparameters=output.hyperparameters,
    )


def per_day_mape(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    aligned = pd.DataFrame({"actual": actual, "forecast": forecast.reindex(actual.index)})
    return aligned.groupby(aligned.index.date).apply(lambda x: mape_pct(x["actual"], x["forecast"]))


def coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    lower_aligned = lower.reindex(actual.index)
    upper_aligned = upper.reindex(actual.index)
    inside = (actual >= lower_aligned) & (actual <= upper_aligned)
    return float(inside.mean())


def pinball_loss(actual: pd.Series, predicted_quantile: pd.Series, q: float) -> float:
    y = actual.to_numpy(dtype=float)
    pred = predicted_quantile.reindex(actual.index).to_numpy(dtype=float)
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def holiday_indicator(index: pd.DatetimeIndex) -> np.ndarray:
    years = sorted({int(ts.year) for ts in index})
    de_holidays = holidays.Germany(years=years)
    return np.array([ts.date() in de_holidays for ts in index], dtype=bool)


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    features = pd.DataFrame(index=index)
    features["hour"] = index.hour.astype(int)
    features["day_of_week"] = index.dayofweek.astype(int)
    features["month"] = index.month.astype(int)
    features["is_weekend"] = (index.dayofweek >= 5).astype(int)
    features["is_public_holiday_de"] = holiday_indicator(index).astype(int)
    return features


def supervised_feature_frame(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"load": series.astype(float)})
    features = time_features(series.index)
    features["lag_24h"] = df["load"].shift(24)
    features["lag_168h"] = df["load"].shift(168)
    features["lag_8760h"] = df["load"].shift(8_736)
    shifted = df["load"].shift(1)
    features["rolling_mean_24h"] = shifted.rolling(24).mean()
    features["rolling_std_24h"] = shifted.rolling(24).std()
    features["rolling_mean_168h"] = shifted.rolling(168).mean()
    features["rolling_std_168h"] = shifted.rolling(168).std()
    features["load"] = df["load"]
    return features.dropna()


def recursive_feature_row(history: pd.Series, timestamp: pd.Timestamp) -> pd.DataFrame:
    values = history.astype(float)
    row = time_features(pd.DatetimeIndex([timestamp]))
    row["lag_24h"] = float(values.loc[timestamp - pd.Timedelta(hours=24)])
    row["lag_168h"] = float(values.loc[timestamp - pd.Timedelta(hours=168)])
    row["lag_8760h"] = float(values.loc[timestamp - pd.Timedelta(hours=8_736)])
    prior = values.loc[: timestamp - pd.Timedelta(hours=1)]
    row["rolling_mean_24h"] = float(prior.tail(24).mean())
    row["rolling_std_24h"] = float(prior.tail(24).std())
    row["rolling_mean_168h"] = float(prior.tail(168).mean())
    row["rolling_std_168h"] = float(prior.tail(168).std())
    return row


def recursive_forecast(model: object, history: pd.Series, horizon_index: pd.DatetimeIndex) -> pd.Series:
    mutable_history = history.copy().astype(float)
    predictions: list[float] = []
    for timestamp in horizon_index:
        row = recursive_feature_row(mutable_history, timestamp)
        pred = float(model.predict(row)[0])
        predictions.append(pred)
        mutable_history.loc[timestamp] = pred
    return pd.Series(predictions, index=horizon_index, name="forecast")


def residuals(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    return actual - forecast.reindex(actual.index)


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
