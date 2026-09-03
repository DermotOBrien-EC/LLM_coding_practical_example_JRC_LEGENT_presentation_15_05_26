"""Backtest: forecast Jan 1-7 of each year using only data available beforehand."""

from __future__ import annotations

import sys

import pandas as pd

import loadfc
import models

HOURS = 168


def window(year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(year, 1, 1, tz=loadfc.TZ)
    return pd.date_range(start, periods=HOURS, freq="h", tz=loadfc.TZ)


def run(years: list[int], with_lgbm: bool = True) -> pd.DataFrame:
    s = loadfc.load_series()
    rows = []
    preds: dict[tuple[int, str], pd.Series] = {}
    for y in years:
        tgt = window(y)
        train = s.loc[s.index < tgt[0]]
        actual = s.reindex(tgt)
        cand: dict[str, pd.Series] = {
            "seasonal_naive_168h": models.seasonal_naive(train, tgt),
            "prior_years_calendar": models.prior_years_calendar(train, tgt),
            "ridge": models.ridge(train, tgt, alpha=30.0, level_adjust=True),
            "ridge_nolevel": models.ridge(train, tgt, alpha=30.0, level_adjust=False),
        }
        if with_lgbm:
            cand["lgbm"] = models.lgbm(train, tgt, level_adjust=True)
            cand["ensemble_ridge_lgbm"] = (cand["ridge"] + cand["lgbm"]) / 2
        for name, p in cand.items():
            preds[(y, name)] = p
            if p.isna().any():
                continue
            rows.append({"year": y, "model": name, "train_h": len(train), **loadfc.metrics(actual, p)})
    return pd.DataFrame(rows), preds


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2017, 2018, 2019]
    res, _ = run(years)
    pd.set_option("display.width", 200)
    piv = res.pivot_table(index="model", columns="year", values="MAPE_%").round(2)
    piv["mean"] = piv.mean(axis=1).round(2)
    print("\n=== MAPE %% by target year (Jan 1-7) ===")
    print(piv.sort_values("mean"))
    print("\n=== full metrics ===")
    print(res.set_index(["year", "model"]).round(2).to_string())
