from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path("/Users/doob/dev/energy_forecast_ws/c6af1f/project/runs/c6af1f")
sys.path.insert(0, str(RUN))
from forecast_load import load_history  # noqa: E402

history = load_history(RUN / "opsd_de_load.csv")
bp = pd.read_csv(RUN / "backtest_predictions.csv")
ok = True
for fs, g in bp[bp.model == "scaled_annual_naive"].groupby("fold_start_utc"):
    start = pd.Timestamp(fs)
    t = pd.DatetimeIndex(pd.to_datetime(g["utc_timestamp"], utc=True))
    analog = history.reindex(t - pd.Timedelta(hours=8736)).to_numpy()
    recent = pd.date_range(start - pd.Timedelta(hours=672), periods=672, freq="h")
    r = history.reindex(recent).mean()
    a = history.reindex(recent - pd.Timedelta(hours=8736)).mean()
    mine = analog * r / a
    ok &= np.allclose(mine, g["forecast_load_mw"].to_numpy())
    print(fs, "ratio", round(r / a, 4), "match", np.allclose(mine, g["forecast_load_mw"].to_numpy()))
print("scaled annual ok", ok)

# lag-8736 analog dates for the target week (calendar status of the analog)
t = pd.date_range("2020-01-01", periods=168, freq="h", tz="UTC")
for lag in (8736, 8760, 8784):
    src = (t - pd.Timedelta(hours=lag)).tz_convert("Europe/Berlin")
    print(lag, "analog local start", src[0], "end", src[-1])
