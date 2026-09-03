"""Chart: 168h-ahead load forecast vs actual, first week of January 2020."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
ACTUAL = "#2a78d6"   # categorical slot 1
FORECAST = "#eb6834"  # categorical slot 2

df = pd.read_csv("forecast_2020_jan_w1.csv", parse_dates=["timestamp_local"])
t = df["timestamp_local"]
mape = float(np.mean(np.abs(df["forecast_MW"] / df["actual_MW"] - 1)) * 100)
mae = float(np.mean(np.abs(df["forecast_MW"] - df["actual_MW"])))

fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(12, 7), height_ratios=[3, 1], sharex=True,
    gridspec_kw={"hspace": 0.12}, facecolor=SURFACE,
)
for a in (ax, ax2):
    a.set_facecolor(SURFACE)
    a.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    a.set_axisbelow(True)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_color("#d8d7d1")
    a.tick_params(colors=INK_2, labelsize=9, length=0)

# New Year's Day (public holiday) shaded as context, not as a series.
hol_end = t.iloc[0] + pd.Timedelta(days=1)
for a in (ax, ax2):
    a.axvspan(t.iloc[0], hol_end, color="#f0efe9", zorder=0)
ax.text(t.iloc[0] + pd.Timedelta(hours=12), ax.get_ylim()[0], "", ha="center")

ax.fill_between(t, df["p10_MW"], df["p90_MW"], color=FORECAST, alpha=0.16, linewidth=0,
                label="Forecast 80% interval")
ax.plot(t, df["actual_MW"], color=ACTUAL, linewidth=2, label="Actual", zorder=3)
ax.plot(t, df["forecast_MW"], color=FORECAST, linewidth=2, label="Forecast", zorder=4)

ax.set_ylabel("Load (MW)", color=INK_2, fontsize=10)
ax.set_title(
    "German electricity load, 1-7 January 2020: 168-hour-ahead forecast vs actual",
    color=INK, fontsize=13, fontweight="semibold", loc="left", pad=16,
)
ax.text(
    0, 1.015,
    f"Trained on 2015-2019 only  ·  MAPE {mape:.2f}%  ·  MAE {mae:,.0f} MW  ·  shaded day = New Year's Day",
    transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom",
)
leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5, ncol=3,
                bbox_to_anchor=(0.0, 0.99))
for txt in leg.get_texts():
    txt.set_color(INK_2)

# Direct label on the week's peak, the number a dispatcher actually cares about.
pk = int(df["actual_MW"].idxmax())
ax.annotate(
    f"peak {df['actual_MW'][pk]:,.0f} MW\nforecast {df['forecast_MW'][pk]:,.0f} MW",
    xy=(t[pk], df["actual_MW"][pk]), xytext=(-8, 14), textcoords="offset points",
    ha="right", fontsize=9, color=INK_2,
)

err = df["forecast_MW"] - df["actual_MW"]
ax2.axhline(0, color="#d8d7d1", linewidth=1)
ax2.fill_between(t, 0, err, color=FORECAST, alpha=0.35, linewidth=0)
ax2.plot(t, err, color=FORECAST, linewidth=1.2)
ax2.set_ylabel("Error (MW)", color=INK_2, fontsize=10)

ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%-d Jan"))
ax2.set_xlim(t.iloc[0], t.iloc[-1])
fig.savefig("forecast_2020_jan_w1.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)
print("wrote forecast_2020_jan_w1.png")
