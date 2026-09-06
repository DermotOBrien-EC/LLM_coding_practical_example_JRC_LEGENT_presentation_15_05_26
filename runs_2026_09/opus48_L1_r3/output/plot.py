from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

df = pd.read_csv("forecast_jan2020.csv", parse_dates=["utc_timestamp"])
local = df["utc_timestamp"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
yhat = df["forecast_MW"].to_numpy() / 1000.0
yact = df["actual_MW"].to_numpy() / 1000.0
print("n:", len(local), "range:", local.iloc[0], "->", local.iloc[-1])

fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(12, 6), height_ratios=[3, 1], sharex=True,
    gridspec_kw={"hspace": 0.18})

handles = []
for day, col, lab in [("2020-01-01", "#c0392b", "Jan 1 New Year (national holiday)"),
                      ("2020-01-06", "#8e44ad", "Jan 6 Epiphany (regional holiday)")]:
    s = pd.Timestamp(day)
    ax.axvspan(s, s + pd.Timedelta(days=1), color=col, alpha=0.10)
    handles.append(mpatches.Patch(color=col, alpha=0.30, label=lab))

l1, = ax.plot(local, yact, color="#1b2a4a", lw=1.8, label="Actual")
l2, = ax.plot(local, yhat, color="#e07a2b", lw=1.8, ls="--", label="Forecast")
ax.set_ylabel("Load (GW)")
ax.set_title("German electricity load: forecast vs actual, 1-7 Jan 2020 (hourly, CET)")
ax.legend(handles=[l1, l2] + handles, loc="lower right", frameon=False, fontsize=8)
ax.grid(alpha=0.25)

err = yhat - yact
ax2.bar(local, err, width=1 / 26, color=np.where(err >= 0, "#e07a2b", "#2b7ae0"))
ax2.axhline(0, color="k", lw=0.6)
ax2.set_ylabel("Error (GW)")
ax2.grid(alpha=0.25)
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d"))
ax.set_xlim(pd.Timestamp("2019-12-31 23:00"), pd.Timestamp("2020-01-08 01:00"))

fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.09)
fig.savefig("forecast_jan2020.png", dpi=110)
print("wrote forecast_jan2020.png")
