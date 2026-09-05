"""SARIMA: the classical statistical benchmark.

A seasonal ARIMA with a 24-hour season is fitted with statsmodels' SARIMAX.
Two families are compared, and validation picks between them:

* "daily": SARIMA(p,d,q)(P,1,Q)_24 on the raw load, optionally with two
  calendar regressors (Saturday, Sunday-or-holiday) so that the weekly
  shape is not entirely lost.
* "weekly-differenced": the load minus its value one week earlier is
  modelled by ARMA(p,q)(P,0,Q)_24, optionally with the week-on-week change
  of the holiday flag as a regressor. This is the task's suggestion of
  letting the weekly structure emerge from differencing: the forecast is
  last week's load plus a modelled correction, so it starts from the
  seasonal-naive baseline and adds the daily-seasonal ARMA on top.

Within each family the ARIMA orders, the length of the recent-history
window the model is fitted on, and the regressor choice are selected on
the validation window.

Why a recent window rather than all 4.75 years: an hourly SARIMA is a
local model. Its forecast a week ahead depends on the last few weeks of
data and on a handful of coefficients; fitting those coefficients on
41,000 hours takes tens of minutes per candidate for no gain, because the
seasonal differencing removes exactly the structure that the long history
would inform. The window length is therefore treated as a hyperparameter
and chosen on validation like everything else.

Validation follows the shared protocol: every candidate is fitted once on
data ending 30 September 2019, and the fitted coefficients are then
applied (no re-estimation) to the history before each of the seven
validation origins to produce a 168-hour forecast.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.sarimax import SARIMAX

import common

SEASON: int = 24
WEEK: int = 168
SCALE: float = 1000.0  # fit in gigawatts for a better-conditioned optimiser

Order = tuple[tuple[int, int, int], tuple[int, int, int, int]]
DAILY_ORDERS: list[Order] = [
    ((1, 0, 1), (0, 1, 1, SEASON)),
    ((2, 0, 2), (0, 1, 1, SEASON)),
    ((1, 0, 1), (1, 1, 1, SEASON)),
]
WEEKLY_DIFF_ORDERS: list[Order] = [
    ((1, 0, 1), (0, 0, 1, SEASON)),
    ((1, 0, 1), (1, 0, 1, SEASON)),
    ((2, 0, 2), (1, 0, 1, SEASON)),
]
# The 26-week window exists so that the holiday regressor can be estimated:
# the 12 weeks before 30 September 2019 contain no German public holiday,
# whereas 26 weeks reach back to Easter, 1 May, Ascension and Whit Monday.
# The daily family with seasonal differencing is several times slower per
# fit and is not tried at 26 weeks.
WINDOW_WEEKS_BY_FAMILY: dict[str, list[int]] = {
    "daily": [6, 12],
    "weekly-differenced": [6, 12, 26],
}
EXOG_BY_FAMILY: dict[str, list[str]] = {
    "daily": ["none", "calendar"],
    "weekly-differenced": ["none", "holiday"],
}


@dataclass(frozen=True)
class Candidate:
    family: str
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    window_weeks: int
    exog: str

    @property
    def weekly_diff(self) -> bool:
        return self.family == "weekly-differenced"

    def label(self) -> str:
        return f"{self.family} {self.order}{self.seasonal_order} window={self.window_weeks}w exog={self.exog}"


def all_candidates() -> list[Candidate]:
    out: list[Candidate] = []
    for family, orders in (("daily", DAILY_ORDERS), ("weekly-differenced", WEEKLY_DIFF_ORDERS)):
        for order, seasonal in orders:
            for weeks in WINDOW_WEEKS_BY_FAMILY[family]:
                for exog in EXOG_BY_FAMILY[family]:
                    out.append(Candidate(family, order, seasonal, weeks, exog))
    return out


def calendar_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Two 0/1 regressors from the local calendar: Saturday, Sunday or holiday."""
    cal = common.calendar_features(index)
    out = pd.DataFrame(index=index)
    out["saturday"] = (cal["day_of_week"] == 5).astype(float)
    out["sunday_or_holiday"] = ((cal["day_of_week"] == 6) | (cal["is_public_holiday_de"] == 1)).astype(float)
    return out


def holiday_diff_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Holiday flag at t minus the holiday flag one week earlier."""
    now = common.is_german_holiday(index).astype(float)
    before = common.is_german_holiday(index - pd.Timedelta(hours=WEEK)).astype(float)
    return pd.DataFrame({"holiday_change": now - before}, index=index)


def _exog_for(index: pd.DatetimeIndex, exog: str) -> pd.DataFrame | None:
    if exog == "calendar":
        return calendar_exog(index)
    if exog == "holiday":
        return holiday_diff_exog(index)
    return None


def _endog(history: pd.Series, cand: Candidate) -> pd.Series:
    """The series the ARIMA is fitted on: the last `window_weeks` weeks, in GW,
    week-on-week differenced for the weekly-differenced family."""
    y = history / SCALE
    if cand.weekly_diff:
        y = y - y.shift(WEEK)
    return y.iloc[-cand.window_weeks * WEEK :]


def fit_model(history: pd.Series, cand: Candidate) -> Any:
    y = _endog(history, cand)
    return SARIMAX(y, exog=_exog_for(y.index, cand.exog), order=cand.order, seasonal_order=cand.seasonal_order).fit(
        disp=False, maxiter=200, method="lbfgs"
    )


def forecast_with(res: Any, history: pd.Series, origin: pd.Timestamp, cand: Candidate) -> pd.DataFrame:
    """Apply already-estimated coefficients to `history` and forecast 168 h.

    Returns the mean plus the 80 % and 95 % interval bounds, in MW.
    """
    y = _endog(history, cand)
    applied = res.apply(endog=y, exog=_exog_for(y.index, cand.exog), refit=False)
    idx = common.horizon_index(origin)
    fc = applied.get_forecast(common.HORIZON, exog=_exog_for(idx, cand.exog))
    f80 = fc.summary_frame(alpha=0.2)
    f95 = fc.summary_frame(alpha=0.05)
    # For the differenced family the forecast is a change relative to the
    # load one week earlier, which is fully observed because the horizon is
    # exactly one week. The interval width is unaffected by adding it back.
    base = history.reindex(idx - pd.Timedelta(hours=WEEK)).to_numpy() / SCALE if cand.weekly_diff else np.zeros(common.HORIZON)
    out = pd.DataFrame(index=idx)
    out["point"] = (f80["mean"].to_numpy() + base) * SCALE
    out["q0.1"] = (f80["mean_ci_lower"].to_numpy() + base) * SCALE
    out["q0.9"] = (f80["mean_ci_upper"].to_numpy() + base) * SCALE
    out["q0.025"] = (f95["mean_ci_lower"].to_numpy() + base) * SCALE
    out["q0.975"] = (f95["mean_ci_upper"].to_numpy() + base) * SCALE
    return out


def _evaluate_candidate(s: pd.Series, cand: Candidate) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        res = fit_model(common.train_slice(s), cand)
        scores = []
        for origin in common.VALIDATION_ORIGINS:
            fc = forecast_with(res, common.history_before(s, origin), origin, cand)
            scores.append(common.mape(s.loc[fc.index], fc["point"]))
        val_mape = float(np.mean(scores))
        aic = float(res.aic)
    except Exception as err:  # a candidate that fails to converge is simply not selected
        val_mape, aic, scores = None, None, []
        print(f"sarima: candidate {cand.label()} failed: {err}")
    return {
        **asdict(cand),
        "val_mape_pct": val_mape,
        "val_mape_by_origin_pct": [round(v, 3) for v in scores],
        "aic": aic,
        "fit_seconds": round(time.perf_counter() - t0, 1),
    }


def run(s: pd.Series) -> common.ModelResult:
    t0 = time.perf_counter()
    grid = all_candidates()
    table = Parallel(n_jobs=-1)(delayed(_evaluate_candidate)(s, c) for c in grid)
    table = sorted(table, key=lambda r: (r["val_mape_pct"] is None, r["val_mape_pct"] or 0.0))
    best = table[0]
    cand = Candidate(best["family"], tuple(best["order"]), tuple(best["seasonal_order"]), best["window_weeks"], best["exog"])
    print(f"sarima: selected {cand.label()} val MAPE {best['val_mape_pct']:.2f} %")

    # Final fit on the window ending 31 December 2019 (Train + Validation).
    history = common.history_before(s, common.TEST_START)
    res = fit_model(history, cand)
    fc = forecast_with(res, history, common.TEST_START, cand)
    quantiles = {0.025: fc["q0.025"], 0.1: fc["q0.1"], 0.5: fc["point"], 0.9: fc["q0.9"], 0.975: fc["q0.975"]}
    return common.ModelResult(
        name="sarima",
        point=fc["point"],
        quantiles=common.sort_quantiles(quantiles),
        hyperparameters={
            "family": cand.family,
            "order": list(cand.order),
            "seasonal_order": list(cand.seasonal_order),
            "weekly_difference_lag_hours": WEEK if cand.weekly_diff else 0,
            "fit_window_weeks": cand.window_weeks,
            "exogenous": cand.exog,
            "fit_window_end": str(history.index[-1]),
            "final_aic": float(res.aic),
            "n_candidates": len(grid),
        },
        runtime_seconds=time.perf_counter() - t0,
        validation_mape_pct=best["val_mape_pct"],
        validation_table=table,
        notes="Intervals are the analytic Gaussian forecast intervals from the state-space model.",
    )


if __name__ == "__main__":
    common.silence_warnings()
    s = common.load_series()
    result = run(s)
    result.save()
    print(f"sarima: test MAPE {common.mape(common.test_slice(s), result.point):.2f} %, runtime {result.runtime_seconds:.0f} s")
