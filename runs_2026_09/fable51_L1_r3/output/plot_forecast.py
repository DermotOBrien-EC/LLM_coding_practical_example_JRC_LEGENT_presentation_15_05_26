from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

TZ = "Europe/Berlin"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
ACTUAL = "#2a78d6"
FORECAST = "#eb6834"

df = pd.read_csv("forecast_jan2020_week1.csv", index_col=0)
df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)
err = df["lightgbm"] - df["actual"]
mape = (err.abs() / df["actual"]).mean() * 100
mae = err.abs().mean()

fig, ax = plt.subplots(figsize=(12, 5), dpi=150, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.plot(df.index, df["actual"] / 1000, color=ACTUAL, lw=2, label="Actual")
ax.plot(df.index, df["lightgbm"] / 1000, color=FORECAST, lw=2, label="Forecast (LightGBM)")
ax.text(df.index[-1], df["actual"].iloc[-1] / 1000, "  Actual", color=INK, va="center", fontsize=9)
ax.text(df.index[-1], df["lightgbm"].iloc[-1] / 1000, "  Forecast", color=INK, va="center", fontsize=9)

for d in pd.date_range("2020-01-01", "2020-01-08", tz=TZ):
    ax.axvline(d, color=GRID, lw=0.8, zorder=0)
ax.set_title(
    f"German hourly electricity load, 1-7 January 2020\n"
    f"168 h ahead forecast trained through 31 Dec 2019. MAE {mae/1000:.1f} GW, MAPE {mape:.1f} %",
    loc="left", color=INK, fontsize=11,
)
ax.set_ylabel("Load (GW)", color=INK2)
ax.xaxis.set_major_locator(mdates.DayLocator(tz=TZ))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %-d Jan", tz=TZ))
ax.tick_params(colors=INK2, length=0)
ax.grid(axis="y", color=GRID, lw=0.8)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(frameon=False, loc="upper left", labelcolor=INK)
ax.set_xlim(df.index[0], df.index[-1] + pd.Timedelta(hours=14))
fig.tight_layout()
fig.savefig("forecast_jan2020_week1.png", facecolor=SURFACE)
print("wrote forecast_jan2020_week1.png")
