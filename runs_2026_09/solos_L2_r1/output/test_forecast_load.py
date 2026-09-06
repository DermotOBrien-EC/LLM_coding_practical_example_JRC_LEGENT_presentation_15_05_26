from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from forecast_load import (
    FORECAST_START,
    HORIZON_HOURS,
    MAX_FORECAST_HOURS,
    build_features,
    calculate_metrics,
    forecast_index,
    format_forecast_period,
)


class ForecastLoadTests(unittest.TestCase):
    def test_forecast_interval_contains_exactly_seven_days(self) -> None:
        index = forecast_index(FORECAST_START, HORIZON_HOURS)

        self.assertEqual(len(index), 168)
        self.assertEqual(index[0], pd.Timestamp("2020-01-01 00:00:00", tz="UTC"))
        self.assertEqual(index[-1], pd.Timestamp("2020-01-07 23:00:00", tz="UTC"))

    def test_horizon_longer_than_shortest_lag_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            forecast_index(FORECAST_START, MAX_FORECAST_HOURS + 1)

    def test_start_must_be_aligned_to_an_hour(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            forecast_index(pd.Timestamp("2020-01-01 00:30:00", tz="UTC"), HORIZON_HOURS)

    def test_target_values_do_not_leak_into_target_features(self) -> None:
        index = pd.date_range("2018-01-01", periods=18_000, freq="h", tz="UTC")
        load = pd.Series(
            50_000.0 + 4_000.0 * np.sin(2.0 * np.pi * np.arange(len(index)) / 24.0),
            index=index,
            name="load_mw",
        )
        start = index[10_000]
        target = forecast_index(start, HORIZON_HOURS)

        original_features = build_features(load).loc[target]
        changed_load = load.copy()
        changed_load.loc[target] += 100_000.0
        changed_features = build_features(changed_load).loc[target]

        pd.testing.assert_frame_equal(original_features, changed_features)

    def test_metrics_match_known_values_and_benchmarks(self) -> None:
        index = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
        results = pd.DataFrame(
            {
                "actual_mw": [100.0, 200.0, 400.0],
                "forecast_mw": [110.0, 180.0, 430.0],
                "weekly_naive_mw": [90.0, 210.0, 360.0],
                "annual_naive_mw": [105.0, 190.0, 420.0],
            },
            index=index,
        )
        results["error_mw"] = results["forecast_mw"] - results["actual_mw"]
        results["absolute_error_mw"] = results["error_mw"].abs()
        results["absolute_percentage_error_percent"] = (
            results["absolute_error_mw"] / results["actual_mw"] * 100.0
        )

        metrics = calculate_metrics(
            results,
            training_rows=1_000,
            training_start=pd.Timestamp("2018-01-01", tz="UTC"),
            training_end=pd.Timestamp("2019-12-31 23:00:00", tz="UTC"),
        )

        self.assertAlmostEqual(metrics.mae_mw, 20.0)
        self.assertAlmostEqual(metrics.rmse_mw, np.sqrt(1400.0 / 3.0))
        self.assertAlmostEqual(metrics.mape_percent, 9.1666666667)
        self.assertAlmostEqual(metrics.mean_error_mw, 20.0 / 3.0)
        self.assertAlmostEqual(metrics.weekly_naive_mae_mw, 20.0)
        self.assertAlmostEqual(metrics.annual_naive_mae_mw, 35.0 / 3.0)
        self.assertAlmostEqual(metrics.mae_improvement_over_weekly_naive_percent, 0.0)
        self.assertAlmostEqual(
            metrics.mae_improvement_over_annual_naive_percent,
            (35.0 / 3.0 - 20.0) / (35.0 / 3.0) * 100.0,
        )

    def test_plot_period_label_uses_supplied_dates(self) -> None:
        march_index = pd.date_range("2020-03-01", periods=48, freq="h", tz="UTC")

        self.assertEqual(format_forecast_period(march_index), "1–2 March 2020 (UTC)")


if __name__ == "__main__":
    unittest.main()
