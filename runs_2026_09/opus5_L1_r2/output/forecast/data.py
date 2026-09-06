"""Loading and calendar feature construction for German hourly load."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

LOAD_COL = "DE_load_actual_entsoe_transparency"
TZ = "Europe/Berlin"

# 2019 population by federal state, millions (Destatis). Used to weight the
# regionally-observed holidays: Epiphany (6 Jan) is a public holiday only in
# BW/BY/ST, so it depresses national load partially rather than fully.
STATE_POPULATION_M: dict[str, float] = {
    "BW": 11.10,
    "BY": 13.08,
    "BE": 3.67,
    "BB": 2.52,
    "HB": 0.68,
    "HH": 1.85,
    "HE": 6.27,
    "MV": 1.61,
    "NI": 7.99,
    "NW": 17.93,
    "RP": 4.09,
    "SL": 0.99,
    "SN": 4.07,
    "ST": 2.20,
    "SH": 2.90,
    "TH": 2.14,
}


def load_series(csv_path: Path) -> pd.Series:
    """Read the OPSD extract as a UTC-indexed hourly series, validated for gaps."""
    frame = pd.read_csv(csv_path, parse_dates=["utc_timestamp"])
    series = frame.set_index("utc_timestamp")[LOAD_COL].astype(float)
    series.index = pd.DatetimeIndex(series.index).tz_convert("UTC")
    series = series.sort_index()

    if not series.index.is_unique:
        raise ValueError("duplicate timestamps in source data")
    expected = pd.date_range(series.index[0], series.index[-1], freq="h", tz="UTC")
    if len(expected.difference(series.index)):
        raise ValueError(f"{len(expected.difference(series.index))} missing hours in source data")
    if series.isna().any():
        raise ValueError(f"{int(series.isna().sum())} NaN load values in source data")
    series.name = "load_mw"
    return series


def holiday_weight(days: pd.DatetimeIndex) -> pd.Series:
    """Population share of Germany for which each calendar day is a public holiday.

    Returns 1.0 for nationwide holidays, 0.0 for ordinary days, and an
    intermediate share for regionally-observed ones such as Epiphany.
    """
    years = sorted({int(y) for y in days.year} | {int(y) + 1 for y in days.year})
    total = sum(STATE_POPULATION_M.values())
    weights = pd.Series(0.0, index=days)
    for state, population in STATE_POPULATION_M.items():
        cal = holidays.Germany(subdiv=state, years=years)
        is_hol = np.fromiter((d.date() in cal for d in days), dtype=bool, count=len(days))
        weights += np.where(is_hol, population / total, 0.0)
    return weights.clip(0.0, 1.0)


@dataclass(frozen=True)
class CalendarFrame:
    """Per-hour calendar attributes in German local time."""

    frame: pd.DataFrame


def build_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build local-time calendar features for a UTC hourly index.

    Every column here is knowable arbitrarily far ahead, so the whole frame is
    valid for a forecast horizon of any length.
    """
    local = index.tz_convert(TZ)
    days = pd.DatetimeIndex(local.normalize().tz_localize(None))
    # Pad by a day either side so the previous/next-day lookups below always
    # resolve, including at the first and last day of the series.
    unique_days = pd.date_range(
        days.min() - pd.Timedelta(days=1), days.max() + pd.Timedelta(days=1), freq="D"
    )
    day_weight = holiday_weight(unique_days)

    out = pd.DataFrame(index=index)
    out["hour"] = local.hour
    out["dow"] = local.dayofweek
    out["hour_of_week"] = local.dayofweek * 24 + local.hour
    out["month"] = local.month
    out["doy"] = local.dayofyear
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)

    out["hol_weight"] = day_weight.reindex(days).to_numpy()
    out["hol_weight_prev"] = day_weight.reindex(days - pd.Timedelta(days=1)).to_numpy()
    out["hol_weight_next"] = day_weight.reindex(days + pd.Timedelta(days=1)).to_numpy()
    out["is_national_holiday"] = (out["hol_weight"] > 0.95).astype(int)

    # Year-end shutdown: 24 Dec to 6 Jan behaves as its own regime, distinct
    # from the individual public holidays inside it.
    month = local.month
    day = local.day
    in_yearend = ((month == 12) & (day >= 24)) | ((month == 1) & (day <= 6))
    out["is_yearend"] = in_yearend.astype(int)
    # Signed day offset from 1 January, clipped to the shutdown window.
    offset = np.where(month == 12, day - 32, np.where(month == 1, day - 1, 99))
    out["yearend_day"] = np.where(in_yearend, offset, 99)

    # Smooth annual and weekly cycles for the linear model.
    doy_frac = 2.0 * np.pi * local.dayofyear.to_numpy() / 365.25
    for k in (1, 2, 3):
        out[f"yr_sin{k}"] = np.sin(k * doy_frac)
        out[f"yr_cos{k}"] = np.cos(k * doy_frac)

    out["trend_years"] = (index - index[0]).total_seconds() / (365.25 * 24 * 3600)
    return out
