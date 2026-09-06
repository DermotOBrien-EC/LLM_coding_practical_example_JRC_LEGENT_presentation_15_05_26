from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from common import QUANTILES, load_data, mape, metrics, quantile_frame, validation_blocks
from lightgbm_features import FEATURES, feature_frame, recursive_forecast


class StudyTests(unittest.TestCase):
    def test_data_splits_and_validation_cover_every_hour(self) -> None:
        data = load_data()
        train = data.loc[:"2019-09-30 23:00"]
        pretest = data.loc[:"2019-12-31 23:00"]
        self.assertEqual(len(data), 50400)
        self.assertEqual(len(train), 41616)
        self.assertEqual(len(pretest), 43824)
        blocks = validation_blocks(pretest, len(train))
        self.assertEqual([len(x) for _, x in blocks], [168] * 13 + [24])
        self.assertEqual(sum(len(x) for _, x in blocks), 2208)
        for history, future in blocks:
            self.assertLess(history.index[-1], future.index[0])

    def test_metrics_known_values(self) -> None:
        index = pd.date_range("2020-01-01", periods=168, freq="h")
        actual = pd.Series(np.full(168, 100.0), index=index)
        result = metrics(actual, np.full(168, 110.0))
        for key in ("mape_test_pct", "mae_test_mw", "rmse_test_mw", "mape_jan1_pct", "mape_jan2_to_jan7_pct"):
            self.assertAlmostEqual(result[key], 10.0)
        self.assertAlmostEqual(mape(np.array([100.0, 200.0]), np.array([110.0, 180.0])), 10.0)

    def test_training_features_exclude_current_and_future_load(self) -> None:
        index = pd.date_range("2018-01-01", periods=9200, freq="h")
        y = pd.Series(np.arange(9200, dtype=float) + 10000, index=index)
        before = feature_frame(y).iloc[9000].copy()
        y.iloc[9000:] = 9999999
        after = feature_frame(y).iloc[9000]
        np.testing.assert_allclose(before.to_numpy(float), after.to_numpy(float))
        self.assertEqual(before["lag_24h"], 18976.0)
        self.assertEqual(before["lag_8760h"], 10240.0)
        self.assertAlmostEqual(before["rolling_mean_24h"], np.mean(np.arange(8976, 9000) + 10000))

    def test_recursive_lags_use_predictions_beyond_origin(self) -> None:
        class EchoLag:
            def predict(self, x: pd.DataFrame) -> np.ndarray:
                return x["lag_24h"].to_numpy() + 10

        index = pd.date_range("2018-01-01", periods=9000, freq="h")
        y = pd.Series(np.arange(9000, dtype=float) + 10000, index=index)
        future = pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=168, freq="h")
        result = recursive_forecast({0.5: EchoLag()}, y, future)
        self.assertEqual(result.shape, (168, 1))
        self.assertEqual(result[0, 0], y.iloc[-24] + 10)
        self.assertEqual(result[24, 0], result[0, 0] + 10)
        self.assertEqual(result[48, 0], result[24, 0] + 10)

    def test_crossing_correction_preserves_selected_median_path(self) -> None:
        class FixedForecast:
            def __init__(self, value: float) -> None:
                self.value = value

            def predict(self, x: pd.DataFrame) -> np.ndarray:
                return np.full(len(x), self.value)

        index = pd.date_range("2018-01-01", periods=9000, freq="h")
        history = pd.Series(np.full(9000, 10000.0), index=index)
        future = pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=168, freq="h")
        models = {q: FixedForecast(value) for q, value in zip(QUANTILES, [20000, 19000, 12000, 5000, 4000])}
        predicted = recursive_forecast(models, history, future)
        np.testing.assert_array_equal(predicted[:, 2], np.full(168, 12000.0))
        self.assertTrue((np.diff(predicted, axis=1) >= 0).all())

    def test_features_have_federal_holidays_and_local_calendar(self) -> None:
        index = pd.date_range("2018-01-01", "2020-01-07", freq="h")
        y = pd.Series(np.full(len(index), 50000.0), index=index)
        x = feature_frame(y)
        self.assertEqual(list(x.columns), FEATURES)
        self.assertEqual(x.loc["2020-01-01 00:00", "is_public_holiday_de"], 1)
        self.assertEqual(x.loc["2020-01-06 12:00", "is_public_holiday_de"], 0)
        self.assertEqual(x.loc["2020-01-01 00:00", "hour"], 1)

    def test_quantile_crossing_is_rearranged(self) -> None:
        q = quantile_frame(np.array([[5.0, 2.0, 4.0, 3.0, 1.0]]))
        self.assertEqual(list(q.columns), [f"q{x:g}" for x in QUANTILES])
        np.testing.assert_array_equal(q.iloc[0], np.arange(1, 6))


if __name__ == "__main__":
    unittest.main()
