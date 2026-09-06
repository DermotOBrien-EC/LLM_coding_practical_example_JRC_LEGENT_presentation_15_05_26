from __future__ import annotations
import sys; sys.path.insert(0, ".")
from pathlib import Path
import pandas as pd
from forecast.data import load_series
from forecast.evaluate import backtest, january_windows
from forecast.models import (HolidayAdjustedNaive, LightGBMModel, LinearProfileModel,
                             SeasonalNaive, YearAgoNaive)

pd.set_option("display.width", 200)
s = load_series(Path("opsd_de_load.csv"))

factories = {
    "seasonal_naive_168h": SeasonalNaive,
    "year_ago_naive_364d": YearAgoNaive,
    "holiday_adjusted_naive": HolidayAdjustedNaive,
    "linear_profile": LinearProfileModel,
    "lightgbm": LightGBMModel,
}
# Selection folds: prior January weeks only. 2020 is untouched here.
res = backtest(s, factories, january_windows([2017, 2018, 2019]))
print("=== per-fold (validation only, 2020 held out) ===")
print(res.set_index(["window", "model"]).round(2).to_string())
print("\n=== mean over validation folds ===")
agg = res.groupby("model")[["MAE", "RMSE", "MAPE", "sMAPE", "BIAS"]].mean().sort_values("MAPE")
print(agg.round(2).to_string())
res.to_csv("outputs/validation_folds.csv", index=False)
