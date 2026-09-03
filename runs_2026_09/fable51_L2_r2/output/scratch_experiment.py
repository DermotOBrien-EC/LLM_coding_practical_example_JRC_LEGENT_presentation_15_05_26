"""Scratch: compare feature variants on Jan-week backtests for 2017, 2018, 2019 (never 2020)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import forecast_load as fl

series = fl.load_series(fl.DATA_PATH)
base = fl.build_features(series)

local_dates = pd.Index(series.index.tz_convert(fl.LOCAL_TZ).normalize())
years = sorted({d.year for d in local_dates} | {2021})
import holidays  # noqa: E402

nat = holidays.country_holidays("DE", years=years)
hol_days = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in nat)).tz_localize(fl.LOCAL_TZ)
naive_dates = local_dates.tz_localize(None)
hol_naive = hol_days.tz_localize(None)
pos = hol_naive.searchsorted(naive_dates)
nxt = hol_naive[np.minimum(pos, len(hol_naive) - 1)]
prv = hol_naive[np.maximum(pos - 1, 0)]
days_to_next = np.clip((nxt - naive_dates).days.to_numpy(), 0, 10)
days_since_prev = np.clip((naive_dates - prv).days.to_numpy(), 0, 10)

variants: dict[str, pd.DataFrame] = {}
variants["A current"] = base
variants["B no day_of_year"] = base.drop(columns=["day_of_year"])
c = base.copy()
c["days_to_next_holiday"] = np.asarray(days_to_next, dtype=float)
c["days_since_prev_holiday"] = np.asarray(days_since_prev, dtype=float)
variants["C + holiday distance"] = c
variants["D B+C"] = c.drop(columns=["day_of_year"])
e = c.drop(columns=["day_of_year", "lag_52w"])
variants["E D minus lag_52w"] = e

starts = [pd.Timestamp(f"{y}-01-01", tz="UTC") for y in (2017, 2018, 2019)]
for name, feats in variants.items():
    rows = []
    for st in starts:
        fc = fl.fit_and_forecast(series, feats, st)
        act = series.loc[fc.index]
        m = fl.Metrics.from_series(act, fc)
        rows.append((st.year, m.mape_pct, float((fc - act).mean())))
    avg = np.mean([r[1] for r in rows])
    detail = "  ".join(f"{y}: MAPE {p:.2f}% bias {b:+.0f}" for y, p, b in rows)
    print(f"{name:<22} avg MAPE {avg:.2f}%   {detail}")
