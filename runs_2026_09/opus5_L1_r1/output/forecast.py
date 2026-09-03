"""Final forecast: German hourly load, 1-7 January 2020, trained only on 2015-2019."""

from __future__ import annotations

import numpy as np
import pandas as pd

import fastridge
import loadfc

ORIGIN = pd.Timestamp(2020, 1, 1, tz=loadfc.TZ)
HOURS = 168
ALPHA = 30.0  # selected on 449 held-out winter origins (tune.py); see tuning_results.csv
REF_DAYS = 45
REF_END_OFFSET_DAYS = 14


def _level_offset(
    s: pd.Series, A: np.ndarray, fit: fastridge.RidgeFit, origin: pd.Timestamp
) -> float:
    """Mean log residual over a recent window that ends before the Christmas period."""
    end = origin - pd.Timedelta(days=REF_END_OFFSET_DAYS)
    start = end - pd.Timedelta(days=REF_DAYS)
    mask = (s.index >= start) & (s.index <= end)
    pos = np.flatnonzero(mask)
    resid = np.log(s.to_numpy(float)[pos]) - np.log(fit.predict(A[pos]))
    return float(resid.mean())


def forecast_at(
    s: pd.Series, A: np.ndarray, path: fastridge.MomentPath, origin: pd.Timestamp
) -> pd.Series | None:
    """168h-ahead forecast using only load observed strictly before `origin`."""
    n = int((s.index < origin).sum())
    lo = int(np.searchsorted(s.index, origin))
    if lo + HOURS > len(s) or n < 24 * 400:
        return None
    fit = fastridge.fit_from_moments(path.upto(n), ALPHA)
    rows = A[lo : lo + HOURS]
    pred = fit.predict(rows) * np.exp(_level_offset(s, A, fit, origin))
    return pd.Series(pred, index=s.index[lo : lo + HOURS], dtype=float)


def main() -> None:
    s = loadfc.load_series()
    X = loadfc.design_matrix(loadfc.build_features(s.index))
    A = X.to_numpy(float)
    t = np.log(s.to_numpy(float))

    # Past winter origins, ascending, for the empirical error distribution.
    hist = [
        ts
        for ts in pd.date_range("2016-04-01", "2019-12-01", freq="D", tz=loadfc.TZ)
        if ts.month in (11, 12, 1, 2)
    ]
    path = fastridge.MomentPath(A, t)
    errs = []
    for o in hist:
        p = forecast_at(s, A, path, o)
        if p is None:
            continue
        a = s.reindex(p.index)
        errs.append(np.log(a.to_numpy()) - np.log(p.to_numpy()))
    E = np.array(errs)
    day = np.arange(HOURS) // 24
    bands = {
        q: np.array([np.quantile(E[:, day == d].ravel(), q) for d in range(7)])
        for q in (0.1, 0.9)
    }
    print(f"Empirical error bands from {len(E)} past winter origins (log scale, by horizon day):")
    for d in range(7):
        print(f"  day {d + 1}: q10 {bands[0.1][d]:+.4f}  q90 {bands[0.9][d]:+.4f}")

    point = forecast_at(s, A, path, ORIGIN)
    assert point is not None
    actual = s.reindex(point.index)

    out = pd.DataFrame(
        {
            "timestamp_local": point.index.strftime("%Y-%m-%d %H:%M"),
            "timestamp_utc": point.index.tz_convert("UTC").strftime("%Y-%m-%d %H:%M"),
            "forecast_MW": point.to_numpy().round(0),
            "p10_MW": (point.to_numpy() * np.exp(bands[0.1][day])).round(0),
            "p90_MW": (point.to_numpy() * np.exp(bands[0.9][day])).round(0),
            "actual_MW": actual.to_numpy(),
        }
    )
    out.to_csv("forecast_2020_jan_w1.csv", index=False)

    print("\n=== Out-of-sample accuracy, 1-7 Jan 2020 (168h ahead, trained through 2019) ===")
    for k, v in loadfc.metrics(actual, point).items():
        print(f"  {k:12s} {v:10.2f}")
    cov = ((out["actual_MW"] >= out["p10_MW"]) & (out["actual_MW"] <= out["p90_MW"])).mean()
    print(f"  {'coverage_%':12s} {cov * 100:10.1f}   (nominal 80%)")

    daily = pd.DataFrame(
        {
            "forecast_mean": point.groupby(point.index.date).mean(),
            "actual_mean": actual.groupby(actual.index.date).mean(),
            "forecast_peak": point.groupby(point.index.date).max(),
            "actual_peak": actual.groupby(actual.index.date).max(),
        }
    )
    daily["err_MW"] = daily["forecast_mean"] - daily["actual_mean"]
    daily["err_%"] = (daily["forecast_mean"] / daily["actual_mean"] - 1) * 100
    daily["MAPE_%"] = (
        (point - actual).abs().div(actual).groupby(point.index.date).mean() * 100
    )
    daily.index = [f"{d:%a %d %b}" for d in pd.to_datetime(daily.index)]
    print("\n=== Daily summary (MW) ===")
    print(daily.round(1).to_string())
    print("\nWrote forecast_2020_jan_w1.csv")


if __name__ == "__main__":
    main()
