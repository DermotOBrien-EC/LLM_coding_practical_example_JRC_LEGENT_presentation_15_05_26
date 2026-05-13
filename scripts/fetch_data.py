"""Fetch the German hourly load subset from Open Power System Data.

Reads only the columns we need from the 130 MB OPSD CSV (streaming via pandas),
filters to 2015-01-01 to 2020-09-30, writes data/opsd_de_load.csv.

This script is idempotent: running it again overwrites the same CSV.

Usage:
    uv run python scripts/fetch_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

URL = (
    "https://data.open-power-system-data.org/time_series/latest/"
    "time_series_60min_singleindex.csv"
)
COLUMNS = ["utc_timestamp", "DE_load_actual_entsoe_transparency"]
START = "2015-01-01 00:00"
END = "2020-09-30 23:00"
OUT = Path(__file__).resolve().parent.parent / "data" / "opsd_de_load.csv"


def main() -> int:
    df = pd.read_csv(URL, usecols=COLUMNS, parse_dates=["utc_timestamp"])
    df = df[(df["utc_timestamp"] >= START) & (df["utc_timestamp"] <= END)]
    df = df.sort_values("utc_timestamp").reset_index(drop=True)

    n_nan = int(df["DE_load_actual_entsoe_transparency"].isna().sum())
    if n_nan > 0:
        print(
            f"WARNING: {n_nan} NaN values in load column. Phase-1 agents "
            "must pick an explicit imputation strategy (see L3 prompt).",
            file=sys.stderr,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    expected_min = 50_000
    if len(df) < expected_min:
        print(
            f"ERROR: only {len(df)} rows; expected at least {expected_min}",
            file=sys.stderr,
        )
        return 1

    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"Range: {df['utc_timestamp'].min()} to {df['utc_timestamp'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
