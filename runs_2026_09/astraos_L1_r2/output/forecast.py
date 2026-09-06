from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.easter import easter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

CUTOFF = pd.Timestamp("2020-01-01T00:00:00Z")
HORIZON = 168
TARGET = "DE_load_actual_entsoe_transparency"
MODELS = ("weekly_naive", "annual_weekday_naive", "calendar_ridge", "boosted_trees", "blend")
SELECTION_ORIGINS = (
    "2017-01-01", "2018-01-01", "2019-01-01",
    "2019-09-04", "2019-10-02", "2019-10-23",
)
CALIBRATION_ORIGINS = (
    "2019-11-06", "2019-11-20", "2019-12-04", "2019-12-18", "2019-12-25",
)


@dataclass(frozen=True)
class Metrics:
    mae_mw: float
    rmse_mw: float
    mape_percent: float


def forecast_index(origin: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(origin, periods=HORIZON, freq="h")


def load_history(path: Path, cutoff: pd.Timestamp) -> pd.Series:
    raw = pd.read_csv(path, dtype=str)
    timestamps = pd.to_datetime(raw["utc_timestamp"], utc=True)
    keep = timestamps < cutoff
    # Filter before parsing target values so future actuals cannot enter analysis.
    values = pd.to_numeric(raw.loc[keep, TARGET], errors="raise").to_numpy(dtype=float)
    history = pd.Series(values, index=pd.DatetimeIndex(timestamps[keep]), name="load_mw")
    if history.empty or not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError("Training timestamps must be nonempty, unique, and sorted")
    expected = pd.date_range(history.index[0], cutoff - pd.Timedelta(hours=1), freq="h")
    if not history.index.equals(expected):
        raise ValueError("Training data must be a complete hourly series ending at the cutoff")
    if not np.isfinite(history.to_numpy()).all() or (history <= 0).any():
        raise ValueError("Training load must be finite and positive")
    return history


def national_holidays(year: int) -> dict[date, str]:
    e = easter(year)
    holidays: dict[date, str] = {
        date(year, 1, 1): "new_year",
        e - timedelta(days=2): "good_friday",
        e + timedelta(days=1): "easter_monday",
        date(year, 5, 1): "labour_day",
        e + timedelta(days=39): "ascension",
        e + timedelta(days=50): "whit_monday",
        date(year, 10, 3): "unity_day",
        date(year, 12, 25): "christmas",
        date(year, 12, 26): "boxing_day",
    }
    if year == 2017:
        holidays[date(year, 10, 31)] = "reformation_500"
    return holidays


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert("Europe/Berlin")
    holiday_map: dict[date, str] = {}
    for year in np.unique(local.year):
        holiday_map.update(national_holidays(int(year)))
    frame = pd.DataFrame(index=index)
    frame["hour"] = local.hour
    frame["dow"] = local.dayofweek
    frame["month"] = local.month
    frame["day"] = local.day
    frame["day_of_year"] = local.dayofyear
    frame["year"] = local.year
    frame["holiday"] = [holiday_map.get(d, "none") for d in local.date]
    frame["effective_dow"] = np.where(frame.holiday != "none", 6, frame.dow)
    frame["christmas_eve"] = ((frame.month == 12) & (frame.day == 24)).astype(int)
    frame["new_year_eve"] = ((frame.month == 12) & (frame.day == 31)).astype(int)
    frame["regional_epiphany"] = ((frame.month == 1) & (frame.day == 6)).astype(int)
    frame["winter_break"] = (
        ((frame.month == 12) & (frame.day >= 21))
        | ((frame.month == 1) & (frame.day <= 6))
    ).astype(int)
    frame["utc_offset"] = [int(t.utcoffset().total_seconds() // 3600) for t in local]
    return frame


def calendar_records(index: pd.DatetimeIndex) -> list[dict[str, float]]:
    frame = calendar_frame(index)
    records: list[dict[str, float]] = []
    for t, r in zip(index, frame.itertuples(index=False), strict=True):
        hour = r.hour
        dow = r.effective_dow
        phase = 2 * np.pi * (r.day_of_year - 1 + hour / 24) / 365.2425
        rec: dict[str, float] = {
            f"hour_week={dow}_{hour}": 1.0,
            "trend": (t - pd.Timestamp("2015-01-01T00:00Z")).total_seconds() / (86400 * 365.2425),
        }
        for k in range(1, 5):
            s, c = float(np.sin(k * phase)), float(np.cos(k * phase))
            rec[f"annual_sin={k}_{hour}"] = s
            rec[f"annual_cos={k}_{hour}"] = c
            rec[f"annual_workday_sin={k}"] = s * (dow < 5)
            rec[f"annual_workday_cos={k}"] = c * (dow < 5)
        if r.holiday != "none":
            rec[f"holiday={r.holiday}"] = 1.0
            rec[f"holiday_hour={r.holiday}_{hour}"] = 1.0
        if r.christmas_eve or r.new_year_eve:
            tag = "christmas_eve" if r.christmas_eve else "new_year_eve"
            rec[f"special={tag}"] = 1.0
            rec[f"special_hour={tag}_{hour}"] = 1.0
        if (r.month == 12 and r.day >= 20) or (r.month == 1 and r.day <= 10):
            tag = f"{r.month:02d}{r.day:02d}_{int(r.dow < 5)}"
            rec[f"winter_date={tag}"] = 1.0
            rec[f"winter_date_block={tag}_{hour // 4}"] = 1.0
        if r.regional_epiphany:
            rec["regional_epiphany"] = 1.0
            rec[f"regional_epiphany_hour={hour}"] = 1.0
        records.append(rec)
    return records


def build_tree_features(
    index: pd.DatetimeIndex, history: pd.Series, origin: pd.Timestamp
) -> pd.DataFrame:
    history = history.loc[history.index < origin]
    frame = calendar_frame(index)
    holiday_codes = {"none": 0, "new_year": 1, "christmas": 2, "boxing_day": 3,
                     "good_friday": 4, "easter_monday": 5, "labour_day": 6,
                     "ascension": 7, "whit_monday": 8, "unity_day": 9, "reformation_500": 10}
    frame["holiday"] = frame.holiday.map(holiday_codes)
    for lag in (168, 336, 504, 672, 8736, 8760, 8784):
        source = index - pd.Timedelta(hours=lag)
        frame[f"lag_{lag}"] = history.reindex(source).to_numpy()
        if lag in (168, 336, 8736, 8760):
            previous_calendar = calendar_frame(source)
            for name in ("holiday", "dow", "winter_break", "christmas_eve", "new_year_eve"):
                values = previous_calendar[name]
                if name == "holiday":
                    values = values.map(holiday_codes)
                frame[f"lag_{lag}_{name}"] = values.to_numpy()
    # Every rolling window ends at t-168, never at a target-period observation.
    for window in (24, 168):
        past = history.rolling(window, min_periods=window).mean()
        frame[f"past_mean_{window}"] = past.reindex(index - pd.Timedelta(hours=168)).to_numpy()
    phase = 2 * np.pi * frame.day_of_year / 365.2425
    frame["annual_sin"] = np.sin(phase)
    frame["annual_cos"] = np.cos(phase)
    return frame.astype(float)


def predict_candidates(history: pd.Series, origin: pd.Timestamp) -> dict[str, np.ndarray]:
    history = history.loc[history.index < origin]
    target = forecast_index(origin)
    predictions: dict[str, np.ndarray] = {
        "weekly_naive": history.reindex(target - pd.Timedelta(hours=168)).to_numpy(),
        "annual_weekday_naive": history.reindex(target - pd.Timedelta(hours=8736)).to_numpy(),
    }
    train = history.loc[history.index >= origin - pd.DateOffset(years=3)]
    encoder = DictVectorizer(sparse=True)
    matrix = encoder.fit_transform(calendar_records(train.index))
    ridge = Ridge(alpha=2.0, solver="lsqr", tol=1e-6)
    ridge.fit(matrix, train.to_numpy())
    ridge_pred = ridge.predict(encoder.transform(calendar_records(target)))
    # Recent non-holiday residuals adjust the level without treating Christmas as normal demand.
    recent_index = train.index[-42 * 24:]
    recent_calendar = calendar_frame(recent_index)
    normal = (recent_calendar.holiday == "none") & (recent_calendar.winter_break == 0)
    normal_index = recent_index[normal.to_numpy()]
    if len(normal_index):
        fitted = ridge.predict(encoder.transform(calendar_records(normal_index)))
        ridge_pred += np.median(history.loc[normal_index].to_numpy() - fitted)
    predictions["calendar_ridge"] = ridge_pred
    tree = HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=250, max_leaf_nodes=23,
        min_samples_leaf=30, l2_regularization=10.0,
        early_stopping=False, random_state=2020,
    )
    tree.fit(build_tree_features(train.index, history, origin), train.to_numpy())
    predictions["boosted_trees"] = tree.predict(build_tree_features(target, history, origin))
    predictions["blend"] = 0.5 * predictions["calendar_ridge"] + 0.5 * predictions["boosted_trees"]
    if not all(np.isfinite(p).all() and len(p) == HORIZON for p in predictions.values()):
        raise ValueError("A model returned incomplete or non-finite predictions")
    return predictions


def metrics(actual: np.ndarray, predicted: np.ndarray) -> Metrics:
    error = actual - predicted
    return Metrics(
        mae_mw=float(np.mean(np.abs(error))),
        rmse_mw=float(np.sqrt(np.mean(error ** 2))),
        mape_percent=float(np.mean(np.abs(error) / actual) * 100),
    )


def make_forecast_table(
    index: pd.DatetimeIndex, predicted: np.ndarray, width80: float, width95: float
) -> pd.DataFrame:
    if len(index) != HORIZON or len(predicted) != HORIZON:
        raise ValueError("The forecast must contain exactly 168 hourly values")
    if not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError("Predictions must be finite and positive")
    if not 0 <= width80 <= width95 or not np.isfinite(width95):
        raise ValueError("Invalid uncertainty widths")
    return pd.DataFrame({
        "utc_timestamp": index,
        "local_timestamp": index.tz_convert("Europe/Berlin"),
        "forecast_mw": np.round(predicted, 1),
        "lower_80_mw": np.round(np.maximum(0, predicted - width80), 1),
        "upper_80_mw": np.round(predicted + width80, 1),
        "lower_95_mw": np.round(np.maximum(0, predicted - width95), 1),
        "upper_95_mw": np.round(predicted + width95, 1),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe German hourly load forecast")
    parser.add_argument("--input", type=Path, default=Path("opsd_de_load.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    history = load_history(args.input, CUTOFF)
    score_rows: list[dict[str, str | float]] = []
    backtest_rows: list[pd.DataFrame] = []
    for start in SELECTION_ORIGINS:
        origin = pd.Timestamp(start, tz="UTC")
        index = forecast_index(origin)
        actual = history.reindex(index).to_numpy()
        print(f"Selection backtest {start}", flush=True)
        with threadpool_limits(limits=2):
            candidates = predict_candidates(history, origin)
        for model, predicted in candidates.items():
            score_rows.append({"origin": start, "group": "new_year" if origin.month == 1 else "autumn",
                               "model": model, **asdict(metrics(actual, predicted))})
            backtest_rows.append(pd.DataFrame({"origin": start, "stage": "selection", "model": model,
                                              "utc_timestamp": index, "actual_mw": actual,
                                              "prediction_mw": predicted}))
    scores = pd.DataFrame(score_rows)
    mean_groups = scores.groupby(["model", "group"]).mae_mw.mean().unstack()
    # New Year is the target regime; ordinary autumn weeks provide a secondary sanity check.
    mean_groups["selection_score_mw"] = 0.75 * mean_groups.new_year + 0.25 * mean_groups.autumn
    selected = str(mean_groups.selection_score_mw.idxmin())
    print(f"Selected model: {selected}\n{mean_groups.to_string()}", flush=True)
    errors: list[np.ndarray] = []
    for start in CALIBRATION_ORIGINS:
        origin = pd.Timestamp(start, tz="UTC")
        index = forecast_index(origin)
        actual = history.reindex(index).to_numpy()
        print(f"Independent calibration backtest {start}", flush=True)
        with threadpool_limits(limits=2):
            predicted = predict_candidates(history, origin)[selected]
        errors.append(actual - predicted)
        backtest_rows.append(pd.DataFrame({"origin": start, "stage": "calibration", "model": selected,
                                          "utc_timestamp": index, "actual_mw": actual,
                                          "prediction_mw": predicted}))
        score_rows.append({"origin": start, "group": "calibration", "model": selected,
                           **asdict(metrics(actual, predicted))})
    calibration_errors = np.concatenate(errors)
    width80, width95 = np.quantile(np.abs(calibration_errors), [0.8, 0.95], method="higher")
    print("Fitting final forecast using observations strictly before 2020", flush=True)
    with threadpool_limits(limits=2):
        predicted = predict_candidates(history, CUTOFF)[selected]
    result = make_forecast_table(forecast_index(CUTOFF), predicted, float(width80), float(width95))
    result.to_csv(args.output / "forecast.csv", index=False)
    daily = result.assign(date_utc=result.utc_timestamp.dt.strftime("%Y-%m-%d")).groupby("date_utc")
    summary = daily.forecast_mw.agg(mean_mw="mean", minimum_mw="min", maximum_mw="max", energy_mwh="sum")
    summary.round(1).to_csv(args.output / "daily_summary.csv")
    pd.DataFrame(score_rows).to_csv(args.output / "backtest_metrics.csv", index=False)
    pd.concat(backtest_rows, ignore_index=True).to_csv(args.output / "backtest_predictions.csv", index=False)
    mean_groups.to_csv(args.output / "model_selection.csv")
    peak = result.loc[result.forecast_mw.idxmax()]
    minimum = result.loc[result.forecast_mw.idxmin()]
    metadata = {
        "input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "training_start_utc": str(history.index.min()), "training_end_utc": str(history.index.max()),
        "historical_rows": len(history), "fit_window_years": 3,
        "forecast_start_utc": str(CUTOFF), "forecast_end_utc": str(result.utc_timestamp.iloc[-1]),
        "forecast_hours": HORIZON, "unit": "MW (assumed from OPSD ENTSO-E load column)",
        "selected_model": selected, "selection_weight_new_year": 0.75,
        "selection_weight_autumn": 0.25,
        "selection_origins": SELECTION_ORIGINS, "calibration_origins": CALIBRATION_ORIGINS,
        "empirical_interval_half_width_80_mw": float(width80),
        "empirical_interval_half_width_95_mw": float(width95),
        "interval_caveat": "Pointwise empirical error bands, not guaranteed coverage or simultaneous week bands. Five calibration weeks, with serially correlated errors and no weather forecast.",
        "mean_load_mw": float(result.forecast_mw.mean()),
        "total_energy_mwh": float(result.forecast_mw.sum()),
        "peak_load_mw": float(peak.forecast_mw), "peak_utc": str(peak.utc_timestamp),
        "minimum_load_mw": float(minimum.forecast_mw), "minimum_utc": str(minimum.utc_timestamp),
        "target_actuals_used": False,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
