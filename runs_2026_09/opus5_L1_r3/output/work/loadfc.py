"""German hourly load forecasting: calendar-driven models and backtest harness."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

TZ = "Europe/Berlin"


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
def easter_sunday(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return pd.Timestamp(year=year, month=month, day=day + 1)


def buss_und_bettag(year: int) -> pd.Timestamp:
    """Wednesday before 23 November."""
    d = pd.Timestamp(year, 11, 22)
    while d.dayofweek != 2:
        d -= pd.Timedelta(days=1)
    return d


def german_holidays(year: int) -> dict[pd.Timestamp, float]:
    """Map date -> share of national load affected (1.0 = nationwide holiday).

    Regional shares are rough population weights; they matter only as a
    learned level shift, not as a precise demographic claim.
    """
    e = easter_sunday(year)
    hol: dict[pd.Timestamp, float] = {
        pd.Timestamp(year, 1, 1): 1.0,        # Neujahr
        e - pd.Timedelta(days=2): 1.0,        # Karfreitag
        e + pd.Timedelta(days=1): 1.0,        # Ostermontag
        pd.Timestamp(year, 5, 1): 1.0,        # Tag der Arbeit
        e + pd.Timedelta(days=39): 1.0,       # Christi Himmelfahrt
        e + pd.Timedelta(days=50): 1.0,       # Pfingstmontag
        pd.Timestamp(year, 10, 3): 1.0,       # Tag der Deutschen Einheit
        pd.Timestamp(year, 12, 25): 1.0,
        pd.Timestamp(year, 12, 26): 1.0,
        pd.Timestamp(year, 1, 6): 0.30,       # Heilige Drei Koenige: BW, BY, ST
        e + pd.Timedelta(days=60): 0.50,      # Fronleichnam
        pd.Timestamp(year, 8, 15): 0.15,      # Mariae Himmelfahrt: SL, BY (partial)
        pd.Timestamp(year, 11, 1): 0.45,      # Allerheiligen
        buss_und_bettag(year): 0.05,          # Sachsen only
    }
    # Reformationstag: 5 eastern states, nationwide for the 2017 anniversary,
    # then 9 states after Hamburg, Bremen, Niedersachsen and Schleswig-Holstein
    # adopted it in 2018.
    hol[pd.Timestamp(year, 10, 31)] = 1.0 if year == 2017 else (0.31 if year >= 2018 else 0.15)
    if year >= 2019:
        hol[pd.Timestamp(year, 3, 8)] = 0.045      # Frauentag, Berlin, from 2019
    if year == 2019:
        hol[pd.Timestamp(year, 9, 20)] = 0.026     # Weltkindertag, Thueringen, 2019 only
    return hol


def holiday_share(dates: pd.DatetimeIndex) -> np.ndarray:
    table: dict[pd.Timestamp, float] = {}
    for y in range(int(dates.year.min()) - 1, int(dates.year.max()) + 2):
        table.update(german_holidays(y))
    naive = pd.DatetimeIndex(dates.tz_localize(None).normalize())
    return np.array([table.get(d, 0.0) for d in naive], dtype=float)


def _daily_off_grid(dates: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Padded daily calendar with an off-day flag, independent of which rows the
    caller kept. A season-windowed training index has gaps; neighbour lookups
    must not read those gaps as working days."""
    days = pd.DatetimeIndex(pd.Series(dates.tz_localize(None).normalize()).unique())
    grid = pd.date_range(days.min() - pd.Timedelta(days=10),
                         days.max() + pd.Timedelta(days=10), freq="D")
    table: dict[pd.Timestamp, float] = {}
    for y in range(grid.year.min() - 1, grid.year.max() + 2):
        table.update(german_holidays(y))
    is_off = np.array([(d.dayofweek >= 5) or (table.get(d, 0.0) >= 0.9) for d in grid])
    return grid, is_off


def bridge_flag(dates: pd.DatetimeIndex) -> np.ndarray:
    """Working day with an off day on both sides, read off the full calendar."""
    grid, is_off = _daily_off_grid(dates)
    br = (~is_off) & np.roll(is_off, 1) & np.roll(is_off, -1)
    lut = pd.Series(br, index=grid)
    return lut.reindex(pd.DatetimeIndex(dates.tz_localize(None).normalize())).to_numpy()


def work_run(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Length of the consecutive working-day run containing each day, and the
    1-based position within it. Off days (weekend / national holiday) get 0.

    A short run wedged between two off blocks is a bridge period: people take
    it as leave and load drops well below a normal working day.
    """
    grid, is_off = _daily_off_grid(dates)
    run_len = np.zeros(len(grid), dtype=int)
    run_pos = np.zeros(len(grid), dtype=int)
    i = 0
    while i < len(grid):
        if is_off[i]:
            i += 1
            continue
        j = i
        while j < len(grid) and not is_off[j]:
            j += 1
        run_len[i:j] = j - i
        run_pos[i:j] = np.arange(1, j - i + 1)
        i = j
    lut_len = pd.Series(run_len, index=grid)
    lut_pos = pd.Series(run_pos, index=grid)
    key = pd.DatetimeIndex(dates.tz_localize(None).normalize())
    return lut_len.reindex(key).to_numpy(), lut_pos.reindex(key).to_numpy()


def season_distance(idx: pd.DatetimeIndex, anchor: pd.Timestamp) -> np.ndarray:
    """Calendar-exact days to the nearest occurrence of the anchor's month/day.

    A day-of-year circle mis-measures across leap years: in 2016, day 339 is 31
    calendar days before 4 January, not 30.
    """
    loc = pd.DatetimeIndex(idx.tz_convert(TZ).tz_localize(None)).normalize()
    best = None
    for y in range(int(loc.year.min()) - 1, int(loc.year.max()) + 2):
        d = np.abs((loc - pd.Timestamp(y, anchor.month, anchor.day)).days.to_numpy())
        best = d if best is None else np.minimum(best, d)
    return best


def turn_of_year_offset(dates: pd.DatetimeIndex, lo: int = -11, hi: int = 10) -> np.ndarray:
    """Signed day offset from the nearest 1 January; 99 outside [lo, hi]."""
    naive = pd.DatetimeIndex(dates.tz_localize(None).normalize())
    ny_prev = pd.to_datetime({"year": naive.year, "month": 1, "day": 1})
    ny_next = pd.to_datetime({"year": naive.year + 1, "month": 1, "day": 1})
    off_prev = (naive - pd.DatetimeIndex(ny_prev)).days
    off_next = (naive - pd.DatetimeIndex(ny_next)).days
    off = np.where(np.abs(off_prev) <= np.abs(off_next), off_prev, off_next)
    return np.where((off >= lo) & (off <= hi), off, 99).astype(int)


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
TEMP_KNOTS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)


def weather_features(temp: pd.Series) -> pd.DataFrame:
    """Derived temperature features on the full contiguous hourly series.

    Smoothing must be computed before any train/target split, otherwise the
    horizon's smoothed temperature restarts from nothing.
    """
    t = temp.astype(float).sort_index()
    w = pd.DataFrame(index=t.index)
    w["temp"] = t
    w["temp_s24"] = t.ewm(halflife=24).mean()
    w["temp_s72"] = t.ewm(halflife=72).mean()
    w["temp_min24"] = t.rolling(24, min_periods=1).min()
    w["hdd15"] = np.clip(15.0 - t, 0, None)
    w["cdd18"] = np.clip(t - 18.0, 0, None)
    w["hdd15_s24"] = np.clip(15.0 - w["temp_s24"], 0, None)
    for k in TEMP_KNOTS:
        w[f"hinge_{k:g}"] = np.clip(t - k, 0, None)
    return w


# --------------------------------------------------------------------------- #
# Feature frame
# --------------------------------------------------------------------------- #
def build_features(index_utc: pd.DatetimeIndex, wx: pd.DataFrame | None = None) -> pd.DataFrame:
    loc = index_utc.tz_convert(TZ)
    share = holiday_share(loc)
    dow = loc.dayofweek.to_numpy()
    hour = loc.hour.to_numpy()

    # Effective day type: a full national holiday behaves like a Sunday.
    eff = dow.astype(object).astype(str)
    eff = np.where(share >= 0.9, "HOL", np.array([str(x) for x in dow]))

    doy = loc.dayofyear.to_numpy()
    ang = 2 * np.pi * doy / 365.25
    f = pd.DataFrame(index=index_utc)
    f["hour"] = hour
    f["dow"] = dow
    f["eff_daytype"] = eff
    f["hour_of_week"] = dow * 24 + hour
    f["hol_share"] = share
    f["is_hol"] = (share >= 0.9).astype(int)
    f["toy"] = turn_of_year_offset(loc)
    f["month"] = loc.month.to_numpy()
    f["doy"] = doy
    f["trend"] = (index_utc - pd.Timestamp("2015-01-01", tz="UTC")).days / 365.25
    for k in (1, 2, 3, 4):
        f[f"sin{k}"] = np.sin(k * ang)
        f[f"cos{k}"] = np.cos(k * ang)
    f["bridge"] = bridge_flag(loc).astype(int)
    rl, rp = work_run(loc)
    f["run_len"] = np.clip(rl, 0, 6)
    f["run_pos"] = np.clip(rp, 0, 6)
    if wx is not None:
        aligned = wx.reindex(index_utc)
        if aligned.isna().any().any():
            raise ValueError("weather features do not cover the requested index")
        for c in aligned.columns:
            f[c] = aligned[c].to_numpy()
    return f


def design_matrix(f: pd.DataFrame, toy_weekend: bool = False, toy_block: bool = False,
                  weather: bool = False, work_run_feat: bool = False) -> pd.DataFrame:
    """Ridge design: day-type x hour shape, month x hour season, ToY day levels."""
    parts = [
        pd.get_dummies(f["eff_daytype"].astype(str) + "_h" + f["hour"].astype(str), prefix="dt"),
        pd.get_dummies(f["month"].astype(str) + "_h" + f["hour"].astype(str), prefix="mh"),
        pd.get_dummies(f["toy"].astype(str), prefix="toy"),
        pd.get_dummies(f["hol_share"].round(2).astype(str), prefix="hs"),
    ]
    if work_run_feat:
        # (run length, position) pair: supported all year by bridges around
        # Easter, Ascension, May 1 and Unity Day, not only by New Year.
        pair = f["run_len"].astype(str) + "p" + f["run_pos"].astype(str)
        parts.append(pd.get_dummies(pair, prefix="run"))
    if toy_weekend:
        wk = np.where(f["dow"].to_numpy() >= 5, "we", "wd")
        parts.append(pd.get_dummies(f["toy"].astype(str) + "_" + wk, prefix="toywe"))
    if toy_block:
        blk = (f["hour"].to_numpy() // 6).astype(str)
        parts.append(pd.get_dummies(f["toy"].astype(str) + "_b" + blk, prefix="toyb"))
    x = pd.concat(parts, axis=1).astype(float)
    x["bridge"] = f["bridge"].astype(float)
    x["trend"] = f["trend"]
    for k in (1, 2, 3, 4):
        x[f"sin{k}"] = f[f"sin{k}"]
        x[f"cos{k}"] = f[f"cos{k}"]
    if weather:
        basis = ["temp"] + [f"hinge_{k:g}" for k in TEMP_KNOTS] + ["temp_s24", "temp_s72",
                                                                  "hdd15_s24", "temp_min24"]
        blk = (f["hour"].to_numpy() // 6)
        for c in basis:
            v = f[c].to_numpy(dtype=float)
            x[c] = v
            for b in range(4):
                x[f"{c}_b{b}"] = v * (blk == b)
        we = (f["dow"].to_numpy() >= 5).astype(float)
        for c in ("temp", "hdd15_s24"):
            x[f"{c}_we"] = f[c].to_numpy(dtype=float) * we
    return x


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass
class Fit:
    name: str
    predict: object
    columns: list[str] = field(default_factory=list)


def _level_correction(actual: pd.Series, fitted: pd.Series, origin: pd.Timestamp,
                      days: int = 56, exclude_toy: bool = True) -> float:
    """Ratio bias correction from a recent clean window ending at the origin."""
    win = slice(origin - pd.Timedelta(days=days), origin - pd.Timedelta(hours=1))
    a, p = actual.loc[win], fitted.loc[win]
    if exclude_toy:
        keep = turn_of_year_offset(a.index.tz_convert(TZ)) == 99
        a, p = a[keep], p[keep]
    if len(a) < 24 * 7:
        return 1.0
    return float(a.mean() / p.mean())


def sample_weights(index: pd.DatetimeIndex, halflife_years: float | None) -> np.ndarray | None:
    if halflife_years is None:
        return None
    age = (index[-1] - index).days / 365.25
    return 0.5 ** (age / halflife_years)


def fit_ridge(y: pd.Series, f_train: pd.DataFrame, alpha: float = 3.0,
              halflife: float | None = None, **design_kw) -> tuple[Ridge, list[str]]:
    x = design_matrix(f_train, **design_kw)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x.to_numpy(), np.log(y.to_numpy()),
              sample_weight=sample_weights(f_train.index, halflife))
    return model, list(x.columns)


def predict_ridge(model: Ridge, cols: list[str], f: pd.DataFrame, **design_kw) -> pd.Series:
    x = design_matrix(f, **design_kw).reindex(columns=cols, fill_value=0.0)
    return pd.Series(np.exp(model.predict(x.to_numpy())), index=f.index)


GBM_BASE = ["hour", "dow", "toy", "month", "doy", "hol_share", "is_hol", "bridge",
            "run_len", "run_pos", "trend",
            "sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "sin4", "cos4"]
GBM_WX = ["temp", "temp_s24", "temp_s72", "temp_min24", "hdd15", "cdd18", "hdd15_s24"]


def gbm_cols(weather: bool) -> list[str]:
    return GBM_BASE + (GBM_WX if weather else [])


def _gbm_cat(cols: list[str]) -> list[bool]:
    return [c in ("hour", "dow", "toy", "month") for c in cols]


def fit_gbm(y: pd.Series, f_train: pd.DataFrame, seed: int = 0,
            halflife: float | None = None,
            weather: bool = False) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=400, learning_rate=0.06,
        max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=False, max_bins=128,
        categorical_features=_gbm_cat(gbm_cols(weather)), random_state=seed,
    )
    cols = gbm_cols(weather)
    model.fit(f_train[cols].to_numpy(dtype=float), np.log(y.to_numpy()),
              sample_weight=sample_weights(f_train.index, halflife))
    model._cols = cols
    return model


def predict_gbm(model: HistGradientBoostingRegressor, f: pd.DataFrame) -> pd.Series:
    return pd.Series(np.exp(model.predict(f[model._cols].to_numpy(dtype=float))), index=f.index)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    a, p = actual.to_numpy(dtype=float), pred.reindex(actual.index).to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    err = p - a
    a_s = actual[ok]
    p_s = pred.reindex(actual.index)[ok]
    daily_a = a_s.tz_convert(TZ).resample("D").max()
    daily_p = p_s.tz_convert(TZ).resample("D").max()
    return {
        "n_hours": int(ok.sum()),
        "MAPE_%": float(np.mean(np.abs(err / a)) * 100),
        "MAE_MW": float(np.mean(np.abs(err))),
        "RMSE_MW": float(np.sqrt(np.mean(err**2))),
        "bias_MW": float(np.mean(err)),
        "peak_MAPE_%": float(np.mean(np.abs((daily_p - daily_a) / daily_a)) * 100),
    }
