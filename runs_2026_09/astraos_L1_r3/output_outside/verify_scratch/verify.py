from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path("/Users/doob/dev/energy_forecast_ws/c6af1f/project/runs/c6af1f")
sys.path.insert(0, str(RUN))
import holidays  # noqa: E402

from forecast_load import CUTOFF, FOLDS, HORIZON, LAGS, calendar_features, load_history  # noqa: E402

history = load_history(RUN / "opsd_de_load.csv")
assert history.index.max() < CUTOFF
print("history rows", len(history), "min", history.min(), "max", history.max())
print("last obs", history.index.max())

# forecast.csv alignment
fc = pd.read_csv(RUN / "forecast.csv")
ts = pd.DatetimeIndex(pd.to_datetime(fc["utc_timestamp"], utc=True))
expected = pd.date_range("2020-01-01", periods=168, freq="h", tz="UTC")
print("forecast rows", len(fc), "cols", list(fc.columns), "aligned", ts.equals(expected))
v = fc["forecast_load_mw"].to_numpy()
print("finite", np.isfinite(v).all(), "min", v.min(), "max", v.max(), "mean", v.mean(), "TWh", v.sum() / 1e6)
print("argmin", ts[v.argmin()], "argmax", ts[v.argmax()])
print("within hist range", history.min() <= v.min(), v.max() <= history.max())

# daily summary recompute
ds = pd.read_csv(RUN / "daily_summary.csv")
daily = pd.Series(v, index=ts).resample("D").agg(["mean", "min", "max"]).round(2)
print("daily match", np.allclose(daily.to_numpy(), ds[["mean_load_mw", "minimum_load_mw", "maximum_load_mw"]].to_numpy()))

# backtest predictions
bp = pd.read_csv(RUN / "backtest_predictions.csv")
bts = pd.DatetimeIndex(pd.to_datetime(bp["utc_timestamp"], utc=True))
print("bp rows", len(bp), "max ts", bts.max(), "no 2020", bool((bts < CUTOFF).all()))
# actuals match history for pre-2020
hist_aligned = history.reindex(bts).to_numpy()
print("actuals match history", np.allclose(hist_aligned, bp["actual_load_mw"].to_numpy()))

# recompute metrics
bm = pd.read_csv(RUN / "backtest_metrics.csv")
rows = []
for (fs, ft, m), g in bp.groupby(["fold_start_utc", "fold_type", "model"], sort=False):
    assert len(g) == HORIZON
    e = g["forecast_load_mw"].to_numpy() - g["actual_load_mw"].to_numpy()
    a = g["actual_load_mw"].to_numpy()
    rows.append(
        {
            "fold_start_utc": pd.Timestamp(fs).isoformat(),
            "fold_type": ft,
            "model": m,
            "mae_mw": np.mean(np.abs(e)),
            "rmse_mw": np.sqrt(np.mean(e**2)),
            "mape_pct": 100 * np.mean(np.abs(e) / a),
            "bias_mw": np.mean(e),
        }
    )
rc = pd.DataFrame(rows)
merged = bm.merge(rc, on=["fold_start_utc", "fold_type", "model"], suffixes=("", "_re"))
print("metrics rows", len(bm), "merged", len(merged))
for c in ["mae_mw", "rmse_mw", "mape_pct", "bias_mw"]:
    print(c, "max abs diff", (merged[c] - merged[c + "_re"]).abs().max())

# blend = mean of the two lgbm
piv = bp.pivot_table(index=["fold_start_utc", "utc_timestamp"], columns="model", values="forecast_load_mw")
print("blend is mean", np.allclose(piv["equal_blend"], (piv["calendar_lgbm"] + piv["lag_calendar_lgbm"]) / 2))
# weekly naive = history lag 168 (fold-local), four-week median check
wn_ok = True
fwm_ok = True
for fs, g in bp[bp.model == "weekly_naive"].groupby("fold_start_utc"):
    t = pd.DatetimeIndex(pd.to_datetime(g["utc_timestamp"], utc=True))
    wn_ok &= np.allclose(history.reindex(t - pd.Timedelta(hours=168)).to_numpy(), g["forecast_load_mw"].to_numpy())
for fs, g in bp[bp.model == "four_week_median"].groupby("fold_start_utc"):
    t = pd.DatetimeIndex(pd.to_datetime(g["utc_timestamp"], utc=True))
    stack = np.column_stack([history.reindex(t - pd.Timedelta(hours=l)).to_numpy() for l in LAGS[:4]])
    fwm_ok &= np.allclose(np.median(stack, axis=1), g["forecast_load_mw"].to_numpy())
print("weekly naive ok", wn_ok, "four week median ok", fwm_ok)

# selection
jan = bm[bm.fold_type == "new_year"].groupby("model")["mae_mw"].mean().sort_values()
print(jan)
best_base = jan[["weekly_naive", "four_week_median", "scaled_annual_naive"]].min()
print("skill", 100 * (1 - jan["equal_blend"] / best_base))
rmse_mean = bm[bm.fold_type == "new_year"].groupby("model")["rmse_mw"].mean()
print("rmse mean", rmse_mean.round(2).to_dict())

# holiday features on target week and history
cf = calendar_features(expected)
loc = expected.tz_convert("Europe/Berlin")
print(pd.DataFrame({"local": loc, "hol": cf.national_holiday.to_numpy(), "kind": cf.holiday_kind.to_numpy(), "epi": cf.epiphany.to_numpy(), "nyd": cf.new_year_distance.to_numpy(), "xd": cf.christmas_distance.to_numpy(), "wd": cf.weekday.to_numpy(), "hr": cf.hour.to_numpy()}).iloc[::24].to_string())
print("first row local", loc[0], "last row local", loc[-1])
nat = holidays.country_holidays("DE", years=range(2014, 2021), language="en_US")
for d, n in sorted(nat.items()):
    if d.year in (2017, 2019, 2020):
        print(d, n)
# historical flagged holiday days count per year
hf = calendar_features(history.index)
hd = pd.Series(hf.national_holiday.to_numpy(), index=history.index.tz_convert("Europe/Berlin")).resample("D").max()
print("holiday days per year", hd.groupby(hd.index.year).sum().to_dict())

# fold origins vs cutoff
for f in FOLDS:
    s = pd.Timestamp(f.start, tz="UTC")
    print(f.start, f.kind, "end", (s + pd.Timedelta(hours=167)).isoformat(), "weekday", s.tz_convert("Europe/Berlin").day_name())
print("target start weekday", CUTOFF.tz_convert("Europe/Berlin").day_name())

# reference: Jan 1-7 daily means in 2019 (pre-2020) for plausibility
for y in (2017, 2018, 2019):
    w = history.loc[f"{y}-01-01":f"{y}-01-07 23:00"]
    print(y, "week mean", round(w.mean()), "daily", w.resample("D").mean().round(0).astype(int).tolist())
