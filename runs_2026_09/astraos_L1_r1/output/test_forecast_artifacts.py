from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_model import load_history, run, target_hours

ROOT = Path(__file__).parent


def test_delivered_forecast_has_correct_hours_and_bounds() -> None:
    forecast = pd.read_csv(ROOT / "forecast.csv")
    timestamps = pd.DatetimeIndex(pd.to_datetime(forecast["utc_timestamp"], utc=True))
    assert timestamps.equals(target_hours("2020-01-01"))
    values = forecast[["forecast_mw", "lower_90_mw", "upper_90_mw"]]
    assert np.isfinite(values.to_numpy()).all()
    assert (values > 0).all().all()
    assert (forecast["lower_90_mw"] < forecast["forecast_mw"]).all()
    assert (forecast["upper_90_mw"] > forecast["forecast_mw"]).all()
    local_as_utc = pd.DatetimeIndex(pd.to_datetime(forecast["germany_timestamp"], utc=True))
    assert timestamps.equals(local_as_utc)


def test_backtest_metrics_and_selected_model_recompute() -> None:
    validation = pd.read_csv(ROOT / "backtest_predictions.csv")
    metrics = pd.read_csv(ROOT / "backtest_metrics.csv")
    history = load_history(ROOT / "opsd_de_load.csv", "2020-01-01")
    for (year, model), frame in validation.groupby(["year", "model"]):
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["utc_timestamp"], utc=True))
        assert timestamps.equals(target_hours(f"{year}-01-01"))
        assert np.allclose(history.reindex(timestamps), frame["actual_mw"], atol=0.001, rtol=0)
        error = frame["forecast_mw"] - frame["actual_mw"]
        reported = metrics.loc[(metrics["year"] == year) & (metrics["model"] == model)].iloc[0]
        assert abs(error.abs().mean() - reported["mae_mw"]) < 0.002
        assert abs(np.sqrt(np.mean(error**2)) - reported["rmse_mw"]) < 0.002
    comparison = metrics.groupby("model")["mae_mw"].mean()
    metadata = json.loads((ROOT / "forecast_metadata.json").read_text())
    assert comparison.idxmin() == metadata["selected_model"]


def test_full_pipeline_is_unchanged_when_future_values_are_poisoned(tmp_path: Path) -> None:
    original = pd.read_csv(ROOT / "opsd_de_load.csv", dtype=str)
    future = pd.to_datetime(original["utc_timestamp"], utc=True) >= pd.Timestamp(
        "2020-01-01", tz="UTC"
    )
    assert future.sum() > 168
    original.loc[future, "DE_load_actual_entsoe_transparency"] = "UNAVAILABLE_FUTURE"
    poisoned_path = tmp_path / "poisoned.csv"
    original.to_csv(poisoned_path, index=False)
    output_dir = tmp_path / "rerun"
    rerun_metadata = run(poisoned_path, output_dir)
    original_metadata = json.loads((ROOT / "forecast_metadata.json").read_text())
    assert rerun_metadata["training_sha256"] == original_metadata["training_sha256"]
    for filename in (
        "forecast.csv",
        "daily_forecast.csv",
        "model_comparison.csv",
        "backtest_metrics.csv",
        "backtest_predictions.csv",
    ):
        assert (output_dir / filename).read_bytes() == (ROOT / filename).read_bytes()
