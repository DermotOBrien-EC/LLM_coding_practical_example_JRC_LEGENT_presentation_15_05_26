from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
BLUE = "#2a78d6"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the forecast without loading any actuals")
    parser.add_argument("--directory", type=Path, default=Path("outputs"))
    parser.add_argument("--print-texture", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(args.directory / "forecast.csv")
    timestamp = pd.to_datetime(data.utc_timestamp, utc=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": SECONDARY,
        "xtick.color": SECONDARY, "ytick.color": SECONDARY,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
    })
    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.19, top=0.77)
    fig.text(0.075, 0.94, "German electricity load: 1–7 January 2020", fontsize=20, weight="bold")
    fig.text(0.075, 0.875, "Hourly forecast from a calendar-aware model blend | Forecast origin: 1 Jan 2020, 00:00 UTC",
             fontsize=10.5, color=SECONDARY)
    ax.fill_between(timestamp, data.lower_95_mw / 1000, data.upper_95_mw / 1000,
                    color=BLUE, alpha=0.09, linewidth=0,
                    hatch="/" if args.print_texture else None, label="Approx. 95% error band")
    ax.fill_between(timestamp, data.lower_80_mw / 1000, data.upper_80_mw / 1000,
                    color=BLUE, alpha=0.16, linewidth=0,
                    hatch="\\" if args.print_texture else None, label="Approx. 80% error band")
    ax.plot(timestamp, data.forecast_mw / 1000, color=BLUE, linewidth=1.5,
            solid_capstyle="round", solid_joinstyle="round", zorder=3)
    peak = int(data.forecast_mw.idxmax())
    ax.scatter(timestamp.iloc[peak], data.forecast_mw.iloc[peak] / 1000,
               s=40, color=BLUE, edgecolors=SURFACE, linewidth=1.5, zorder=4)
    ax.annotate(f"Peak {data.forecast_mw.iloc[peak] / 1000:.1f} GW",
                (timestamp.iloc[peak], data.forecast_mw.iloc[peak] / 1000),
                xytext=(-15, 35), textcoords="offset points", ha="right", color=INK,
                arrowprops={"arrowstyle": "-", "color": SECONDARY, "lw": 0.65})
    ax.set_ylabel("Load (GW)", labelpad=12)
    ax.set_ylim(30, 85)
    ax.set_yticks([30, 40, 50, 60, 70, 80])
    ax.set_xlim(timestamp.iloc[0], timestamp.iloc[0] + pd.Timedelta(days=7))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[12], tz=timezone.utc))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d Jan\n%a", tz=timezone.utc))
    ax.tick_params(axis="both", which="both", length=0, pad=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, ncol=2)
    fig.text(0.075, 0.075, "Time: UTC (Germany is UTC+1). Shading uses errors from five held-out 2019 weeks; coverage is not guaranteed.",
             color=SECONDARY, fontsize=9)
    fig.text(0.075, 0.035, "Input: opsd_de_load.csv, observations before 2020 only. Complete hourly values and bands: forecast.csv.",
             color=SECONDARY, fontsize=9)
    suffix = "_print" if args.print_texture else ""
    fig.savefig(args.directory / f"forecast{suffix}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
