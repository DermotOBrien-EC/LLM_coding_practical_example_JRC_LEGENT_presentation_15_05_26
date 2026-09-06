"""Forecast vs actual chart, light and dark, from the validated palette."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  grid="#e3e2df", s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
    "dark":  dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                  grid="#343431", s1="#3987e5", s2="#d95926", s3="#199e70"),
}

SUMM = json.load(open("out/summary.json"))
M_SEL = SUMM["metrics_2020"]["primary (selected)"]
M_ALT = SUMM["metrics_2020"]["alt full-history"]

d = pd.read_pickle("work/plot_data.pkl")
preds, actual, loc = d["preds"], d["actual"], d["loc"]
lo, hi = d["lo"], d["hi"]
x = loc.to_pydatetime()
sel = preds["primary (selected)"].to_numpy()
alt = preds["alt full-history"].to_numpy()
act = actual.to_numpy()


def draw(theme: str, path: str) -> None:
    c = THEMES[theme]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 8.6), sharex=True, height_ratios=[2.4, 1],
        gridspec_kw=dict(hspace=0.13))
    fig.patch.set_facecolor(c["surface"])
    for a in (ax, ax2):
        a.set_facecolor(c["surface"])
        a.grid(True, color=c["grid"], linewidth=0.8, alpha=0.9)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(c["grid"])
        a.tick_params(colors=c["secondary"], labelsize=10, length=0)

    ax.fill_between(x, lo / 1000, hi / 1000, color=c["s2"], alpha=0.13, linewidth=0,
                    label="80% interval (selected)", zorder=1)
    ax.plot(x, act / 1000, color=c["s1"], linewidth=2, label="Actual", zorder=4)
    ax.plot(x, sel / 1000, color=c["s2"], linewidth=2, label="Selected model", zorder=3)
    ax.plot(x, alt / 1000, color=c["s3"], linewidth=2, label="Runner-up: full history",
            zorder=2)
    ax.set_ylabel("Load (GW)", color=c["secondary"], fontsize=11)
    ax.set_title("German hourly electricity load, 1–7 January 2020",
                 color=c["primary"], fontsize=15.5, fontweight="bold", loc="left", pad=44)
    ax.text(0, 1.075, "Trained on 2015–2019 only. Both models were chosen on 2017–2019 "
                      "backtests, blind to this week.",
            transform=ax.transAxes, color=c["secondary"], fontsize=10.5, va="bottom")
    ax.text(0, 1.017, f"Selected model: MAPE {M_SEL['MAPE_%']:.2f}%, "
                      f"MAE {M_SEL['MAE_MW']:,.0f} MW    ·    "
                      f"Runner-up: MAPE {M_ALT['MAPE_%']:.2f}%, "
                      f"MAE {M_ALT['MAE_MW']:,.0f} MW",
            transform=ax.transAxes, color=c["secondary"], fontsize=10.5, va="bottom")
    ax.annotate("Actual", (x[41], act[41] / 1000), textcoords="offset points",
                xytext=(-6, -17), color=c["s1"], fontsize=10.5, fontweight="bold", ha="right")
    ax.annotate("Selected", (x[62], sel[62] / 1000), textcoords="offset points",
                xytext=(8, 6), color=c["s2"], fontsize=10.5, fontweight="bold")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=10, ncol=4)
    for t in leg.get_texts():
        t.set_color(c["secondary"])

    ax2.axhline(0, color=c["grid"], linewidth=1.2)
    ax2.plot(x, (alt - act) / 1000, color=c["s3"], linewidth=2, label="Runner-up error")
    ax2.plot(x, (sel - act) / 1000, color=c["s2"], linewidth=2, label="Selected error")
    ax2.set_ylabel("Error (GW)", color=c["secondary"], fontsize=11)
    leg2 = ax2.legend(loc="upper right", frameon=False, fontsize=10, ncol=2)
    for t in leg2.get_texts():
        t.set_color(c["secondary"])

    for a in (ax, ax2):
        a.axvspan(x[24], x[72], color=c["s2"], alpha=0.05, zorder=0)
    ax.annotate("The selected model runs high all week; both miss most on\n"
                "2–3 Jan, when only two working days sit between New Year\n"
                "and the weekend. That never occurs in 2015–2019.",
                xy=(x[58], (sel[58] + act[58]) / 2000), xytext=(x[1], 76.5),
                color=c["secondary"], fontsize=9.5, va="top",
                arrowprops=dict(arrowstyle="->", color=c["secondary"], linewidth=1,
                                connectionstyle="arc3,rad=-0.15"))

    ax2.xaxis.set_major_locator(mdates.DayLocator(tz=loc.tz))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=loc.tz))
    ax2.set_xlim(x[0], x[-1])
    ax2.set_xlabel("Local time (Europe/Berlin)", color=c["secondary"], fontsize=11)
    fig.savefig(path, dpi=150, facecolor=c["surface"], bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


draw("light", "out/forecast_2020_week1.png")
draw("dark", "out/forecast_2020_week1_dark.png")
