from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path('/Users/doob/dev/energy_forecast_ws/760623/project/runs/760623')
INPUT = ROOT / 'opsd_de_load.csv'
OUTPUT = ROOT / 'forecast_jan2020_week1.csv'
BERLIN = ZoneInfo('Europe/Berlin')
BASE = datetime(2015, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Row:
    utc: datetime
    local: datetime
    load: float


@dataclass(frozen=True)
class Fit:
    beta: np.ndarray
    scale: np.ndarray


names: list[str] = []


def add_name(name: str) -> int:
    names.append(name)
    return len(names) - 1


add_name('intercept')
add_name('trend_years')
for hour in range(24):
    add_name(f'hour={hour}')
for dow in range(7):
    add_name(f'dow={dow}')
for dow in range(7):
    for hour in range(24):
        add_name(f'dow={dow}:hour={hour}')
for harmonic in range(1, 7):
    add_name(f'ann_sin{harmonic}')
    add_name(f'ann_cos{harmonic}')
for day in range(1, 8):
    for hour in range(24):
        add_name(f'jan{day}:hour={hour}')
FEATURE_COUNT = len(names)


def read_rows() -> list[Row]:
    rows: list[Row] = []
    with INPUT.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            utc = datetime.fromisoformat(record['utc_timestamp'])
            load = float(record['DE_load_actual_entsoe_transparency'])
            rows.append(Row(utc=utc, local=utc.astimezone(BERLIN), load=load))
    return rows


def year_fraction(local_time: datetime) -> float:
    start = datetime(local_time.year, 1, 1, tzinfo=BERLIN)
    end = datetime(local_time.year + 1, 1, 1, tzinfo=BERLIN)
    return (local_time - start).total_seconds() / (end - start).total_seconds()


def features(utc: datetime) -> dict[int, float]:
    local_time = utc.astimezone(BERLIN)
    hour = local_time.hour
    dow = local_time.weekday()
    result: dict[int, float] = {
        0: 1.0,
        1: (utc - BASE).total_seconds() / (365.25 * 24 * 3600),
        2 + hour: 1.0,
        2 + 24 + dow: 1.0,
        2 + 24 + 7 + dow * 24 + hour: 1.0,
    }
    harmonic_start = 2 + 24 + 7 + 168
    fraction = year_fraction(local_time)
    for harmonic in range(1, 7):
        angle = 2 * math.pi * harmonic * fraction
        offset = harmonic_start + (harmonic - 1) * 2
        result[offset] = math.sin(angle)
        result[offset + 1] = math.cos(angle)
    jan_start = harmonic_start + 12
    if local_time.month == 1 and 1 <= local_time.day <= 7:
        result[jan_start + (local_time.day - 1) * 24 + hour] = 1.0
    return result


def design_matrix(times: list[datetime]) -> np.ndarray:
    matrix = np.zeros((len(times), FEATURE_COUNT), dtype=float)
    for row_index, utc in enumerate(times):
        for feature_index, value in features(utc).items():
            matrix[row_index, feature_index] = value
    return matrix


def fit_model(rows: list[Row], alpha: float) -> Fit:
    x = design_matrix([row.utc for row in rows])
    y = np.log(np.array([row.load for row in rows], dtype=float))
    scale = np.ones(FEATURE_COUNT, dtype=float)
    scale[1:] = np.sqrt((x[:, 1:] ** 2).mean(axis=0))
    scale[scale == 0] = 1.0
    xs = x / scale
    penalty = np.eye(FEATURE_COUNT, dtype=float) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xs.T @ xs + penalty, xs.T @ y)
    return Fit(beta=beta, scale=scale)


def predict(fit: Fit, times: list[datetime]) -> np.ndarray:
    x = design_matrix(times) / fit.scale
    return np.exp(x @ fit.beta)


def evaluate_alpha(rows: list[Row], alpha: float) -> tuple[float, float]:
    absolute_percentage_errors: list[float] = []
    squared_errors: list[float] = []
    for year in (2017, 2018, 2019):
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=7)
        train_rows = [row for row in rows if row.utc < start]
        times = [start + timedelta(hours=hour) for hour in range(168)]
        actual = np.array([row.load for row in rows if start <= row.utc < end], dtype=float)
        forecast = predict(fit_model(train_rows, alpha), times)
        absolute_percentage_errors.extend(np.abs((forecast - actual) / actual).tolist())
        squared_errors.extend(((forecast - actual) ** 2).tolist())
    return float(np.mean(absolute_percentage_errors) * 100), float(math.sqrt(np.mean(squared_errors)))


def main() -> None:
    rows = read_rows()
    alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    scores = [(alpha, *evaluate_alpha(rows, alpha)) for alpha in alphas]
    best_alpha, best_mape, best_rmse = min(scores, key=lambda item: item[1])
    target_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    target_end = target_start + timedelta(days=7)
    train_rows = [row for row in rows if row.utc < target_start]
    target_times = [target_start + timedelta(hours=hour) for hour in range(168)]
    forecast = predict(fit_model(train_rows, best_alpha), target_times)
    actual_by_time = {row.utc: row.load for row in rows if target_start <= row.utc < target_end}
    with OUTPUT.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['utc_timestamp', 'forecast_de_load_mw'])
        for utc, value in zip(target_times, forecast, strict=True):
            writer.writerow([utc.isoformat(sep=' '), round(float(value), 1)])
    actual = np.array([actual_by_time[utc] for utc in target_times], dtype=float)
    mape = float(np.mean(np.abs((forecast - actual) / actual)) * 100)
    rmse = float(math.sqrt(np.mean((forecast - actual) ** 2)))
    print(f'rows={len(rows)} train_rows={len(train_rows)} output={OUTPUT}')
    print('backtest_scores alpha,mape_percent,rmse_mw')
    for alpha, mape_score, rmse_score in scores:
        print(f'{alpha:g},{mape_score:.3f},{rmse_score:.1f}')
    print(f'best_alpha={best_alpha:g} backtest_mape_percent={best_mape:.3f} backtest_rmse_mw={best_rmse:.1f}')
    print(f'jan2020_actual_mape_percent={mape:.3f} jan2020_actual_rmse_mw={rmse:.1f}')
    print('first_rows')
    for utc, value in list(zip(target_times, forecast, strict=True))[:8]:
        print(f'{utc.isoformat(sep=" ")},{round(float(value), 1)}')


if __name__ == '__main__':
    main()
