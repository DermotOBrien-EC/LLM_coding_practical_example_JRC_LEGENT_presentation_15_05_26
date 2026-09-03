"""Hourly German electricity load forecasting: features, models, evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd

TZ = "Europe/Berlin"
CSV = "opsd_de_load.csv"

# Population share of states observing Epiphany (BY 13.1M, BW 11.1M, ST 2.2M of 83.2M).
EPIPHANY_WEIGHT = 0.317
_REGIONAL_SUBDIVS = ("BY", "BW", "ST")


def load_series(path: str = CSV) -> pd.Series:
    """Hourly load in MW, indexed in Europe/Berlin local time."""
    df = pd.read_csv(path, parse_dates=["utc_timestamp"])
    s = df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"]
    s.index = pd.DatetimeIndex(s.index).tz_convert(TZ)
    s = s.sort_index()
    s.name = "load"
    return s


def _regional_holiday_weight(dates: pd.DatetimeIndex, years: list[int]) -> np.ndarray:
    """Fraction of population on a regional (non-national) public holiday."""
    nat = holidays.Germany(years=years)
    per_state = {sd: holidays.Germany(years=years, subdiv=sd) for sd in _REGIONAL_SUBDIVS}
    out = np.zeros(len(dates))
    for i, ts in enumerate(dates):
        d = ts.date()
        if d in nat:
            continue
        hit = sum(1 for sd in _REGIONAL_SUBDIVS if d in per_state[sd])
        if hit:
            out[i] = EPIPHANY_WEIGHT * hit / len(_REGIONAL_SUBDIVS)
    return out


def turn_of_year_offset(dates: pd.DatetimeIndex) -> np.ndarray:
    """Days from the nearest Jan 1, as float; NaN outside the Dec 18 - Jan 12 season."""
    out = np.full(len(dates), np.nan)
    for i, ts in enumerate(dates):
        if ts.month == 12 and ts.day >= 18:
            out[i] = (ts.normalize() - pd.Timestamp(ts.year + 1, 1, 1, tz=ts.tz)).days
        elif ts.month == 1 and ts.day <= 12:
            out[i] = (ts.normalize() - pd.Timestamp(ts.year, 1, 1, tz=ts.tz)).days
    return out


def build_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Purely calendar-derived features: computable arbitrarily far ahead, no leakage."""
    years = sorted({int(y) for y in index.year} | {int(index.year.max()) + 1})
    nat = holidays.Germany(years=years)
    is_nat = np.array([ts.date() in nat for ts in index], dtype=float)

    f = pd.DataFrame(index=index)
    f["hour"] = index.hour
    f["dow"] = index.dayofweek
    f["doy"] = index.dayofyear
    f["year"] = index.year
    f["is_holiday"] = is_nat
    f["reg_holiday"] = _regional_holiday_weight(index, years)
    # Day type: Mon..Sun = 0..6, national public holiday = 7.
    f["daytype"] = np.where(is_nat > 0, 7, index.dayofweek)
    f["toy"] = turn_of_year_offset(index)
    # Day before / after a national holiday (bridge-day behaviour).
    hol_days = {d for d in nat}
    f["pre_holiday"] = [
        float((ts.normalize() + pd.Timedelta(days=1)).date() in hol_days) for ts in index
    ]
    f["post_holiday"] = [
        float((ts.normalize() - pd.Timedelta(days=1)).date() in hol_days) for ts in index
    ]
    return f


def _onehot(codes: pd.Series, levels: list[int], prefix: str) -> pd.DataFrame:
    cols = {f"{prefix}_{lv}": (codes == lv).astype(float) for lv in levels}
    return pd.DataFrame(cols, index=codes.index)


HOUR_BLOCKS = 6  # 4-hour blocks
TOY_LEVELS = list(range(-14, 12))


def design_matrix(f: pd.DataFrame) -> pd.DataFrame:
    """Ridge design matrix on calendar features (log-load target)."""
    idx = f.index
    parts: list[pd.DataFrame] = []

    # Core profile: daytype x hour-of-day.
    dt_hour = f["daytype"].astype(int) * 24 + f["hour"].astype(int)
    parts.append(_onehot(dt_hour, list(range(8 * 24)), "dth"))

    # Annual seasonality, plus its interaction with time of day (winter vs summer shape).
    ang = 2 * np.pi * f["doy"].to_numpy() / 365.25
    ann: dict[str, np.ndarray] = {}
    for k in (1, 2, 3):
        ann[f"ann_sin{k}"] = np.sin(k * ang)
        ann[f"ann_cos{k}"] = np.cos(k * ang)
    parts.append(pd.DataFrame(ann, index=idx))
    hblock = (f["hour"].astype(int) // 4).astype(int)
    hb = _onehot(hblock, list(range(HOUR_BLOCKS)), "hb")
    for k in (1, 2):
        for trig in ("sin", "cos"):
            base = ann[f"ann_{trig}{k}"]
            inter = hb.mul(base, axis=0)
            inter.columns = [f"{c}_x_ann_{trig}{k}" for c in hb.columns]
            parts.append(inter)

    # Turn-of-year: day offset x hour block (the crux for a first-week-of-January forecast).
    toy = f["toy"]
    toy_code = toy.fillna(99).astype(int)
    toy_oh = _onehot(toy_code, TOY_LEVELS, "toy")
    parts.append(toy_oh)
    toy_cols = {
        f"toy_{lv}_hb{b}": toy_oh[f"toy_{lv}"] * hb[f"hb_{b}"]
        for lv in TOY_LEVELS
        for b in range(HOUR_BLOCKS)
    }
    parts.append(pd.DataFrame(toy_cols, index=idx))

    # Regional holidays, bridge days, linear trend.
    misc = pd.DataFrame(index=idx)
    misc["reg_holiday"] = f["reg_holiday"]
    for b in range(HOUR_BLOCKS):
        misc[f"reg_holiday_hb{b}"] = f["reg_holiday"] * hb[f"hb_{b}"]
    misc["pre_holiday"] = f["pre_holiday"]
    misc["post_holiday"] = f["post_holiday"]
    misc["trend"] = (f["year"].to_numpy() - 2017.0) + (f["doy"].to_numpy() - 183.0) / 365.25
    parts.append(misc)

    X = pd.concat(parts, axis=1)
    return X.loc[:, X.std(axis=0).to_numpy() > 0]


@dataclass
class RidgeLog:
    """Ridge regression on log load with column standardisation."""

    alpha: float = 3.0
    _cols: list[str] | None = None
    _mu: np.ndarray | None = None
    _sd: np.ndarray | None = None
    _beta: np.ndarray | None = None
    _b0: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series, w: np.ndarray | None = None) -> RidgeLog:
        self._cols = list(X.columns)
        A = X.to_numpy(dtype=float)
        n = len(A)
        if w is None:
            w = np.ones(n)
        w = w * (n / w.sum())
        self._mu = (A * w[:, None]).sum(axis=0) / n
        sd = np.sqrt(((A - self._mu) ** 2 * w[:, None]).sum(axis=0) / n)
        sd[sd == 0] = 1.0
        self._sd = sd
        Z = (A - self._mu) / sd
        t = np.log(y.to_numpy(dtype=float))
        self._b0 = float((t * w).sum() / n)
        Zw = Z * w[:, None]
        G = Z.T @ Zw + self.alpha * n / 1000.0 * np.eye(Z.shape[1])
        self._beta = np.linalg.solve(G, Zw.T @ (t - self._b0))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._cols is not None and self._beta is not None
        A = X.reindex(columns=self._cols, fill_value=0.0).to_numpy(dtype=float)
        Z = (A - self._mu) / self._sd
        return np.exp(self._b0 + Z @ self._beta)


def metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    a = actual.to_numpy(dtype=float)
    p = pred.reindex(actual.index).to_numpy(dtype=float)
    err = p - a
    return {
        "MAPE_%": float(np.mean(np.abs(err / a)) * 100),
        "MAE_MW": float(np.mean(np.abs(err))),
        "RMSE_MW": float(np.sqrt(np.mean(err**2))),
        "Bias_MW": float(np.mean(err)),
        "peak_err_%": float((p.max() - a.max()) / a.max() * 100),
    }
