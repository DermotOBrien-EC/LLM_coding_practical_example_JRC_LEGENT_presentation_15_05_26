"""Does a general 'workday island' feature help? Decided on the BACKTEST weeks.

German Brueckentag effect: when a midweek public holiday strands a short run of
working days against the weekend, much of the workforce takes the whole block
off and load stays low on days that are not themselves holidays. The shipped
``is_bridge_day`` only fires on days *adjacent* to a holiday, so it flags
Thu 2 Jan 2020 but misses Fri 3 Jan, the worst day of the target-week forecast.

Accept/reject is decided on the 2017-2019 January weeks only. The 2020 number is
computed and printed for transparency but takes no part in the decision.

Run one variant per invocation (each is four model fits):

    python experiment_bridge.py baseline
    python experiment_bridge.py workday-run
    python experiment_bridge.py --report
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import forecast_load as fl

RESULTS: Path = Path(__file__).parent / "outputs" / "experiment_bridge.json"
ORIGINS: list[str] = list(fl.BACKTEST_STARTS) + [fl.TARGET_START]

_ORIGINAL_CALENDAR = fl.calendar_features


def workday_run_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Length and position of the contiguous run of working days around each day."""
    days = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in index.date}))
    span = pd.date_range(days.min() - pd.Timedelta(days=10), days.max() + pd.Timedelta(days=10))
    years = list(range(int(span.year.min()), int(span.year.max()) + 1))
    national = fl.holidays.country_holidays("DE", years=years)
    working = np.array([d.dayofweek < 5 and d.date() not in national for d in span])

    run_len = np.zeros(len(span))
    days_into = np.zeros(len(span))
    days_left = np.zeros(len(span))
    i = 0
    while i < len(span):
        if not working[i]:
            i += 1
            continue
        j = i
        while j < len(span) and working[j]:
            j += 1
        for k in range(i, j):
            run_len[k], days_into[k], days_left[k] = j - i, k - i + 1, j - k
        i = j

    table = pd.DataFrame(
        {"workday_run_len": run_len, "days_into_run": days_into, "days_left_in_run": days_left},
        index=pd.Index(span.date),
    )
    return table.loc[index.date].set_index(index)


def _patched_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.concat([_ORIGINAL_CALENDAR(index), workday_run_features(index)], axis=1)


def run_variant(variant: str) -> None:
    fl.calendar_features = _patched_calendar if variant == "workday-run" else _ORIGINAL_CALENDAR  # type: ignore[assignment]
    load = fl.load_series()
    store = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for start in ORIGINS:
        week = fl.forecast_week(load, start)
        store.setdefault(start, {})[variant] = asdict(week.model)
        print(f"  {variant:<12} {start}  MAPE {week.model.mape_pct:5.2f}%", flush=True)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(store, indent=2) + "\n")


def report() -> None:
    store = json.loads(RESULTS.read_text())
    print(f"{'week':<14}{'baseline':>12}{'+workday-run':>15}{'change':>10}   role")
    print("-" * 66)
    for start in ORIGINS:
        base = store[start]["baseline"]["mape_pct"]
        new = store[start]["workday-run"]["mape_pct"]
        role = "decision set" if start in fl.BACKTEST_STARTS else "TARGET (excluded from decision)"
        print(f"{start:<14}{base:>11.2f}%{new:>14.2f}%{new - base:>+10.2f}   {role}")

    base_mean = float(np.mean([store[s]["baseline"]["mape_pct"] for s in fl.BACKTEST_STARTS]))
    new_mean = float(np.mean([store[s]["workday-run"]["mape_pct"] for s in fl.BACKTEST_STARTS]))
    print(f"\nbacktest mean MAPE  baseline {base_mean:.2f}%  ->  +workday-run {new_mean:.2f}%")
    print("VERDICT:", "ACCEPT the feature" if new_mean < base_mean else "REJECT — keep baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", choices=["baseline", "workday-run"])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report()
    elif args.variant:
        run_variant(args.variant)
    else:
        parser.error("give a variant or --report")
