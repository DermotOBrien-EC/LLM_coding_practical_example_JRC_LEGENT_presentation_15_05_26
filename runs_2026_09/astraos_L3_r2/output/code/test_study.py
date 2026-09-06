from __future__ import annotations

# ruff: noqa: E402
# Match the executable's package bootstrap before importing study modules.

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
sys.path = [p for p in sys.path if Path(p or ".").resolve() != CODE]
spec = importlib.util.spec_from_file_location("load_bakeoff", CODE / "__init__.py")
package = importlib.util.module_from_spec(spec)
sys.modules["load_bakeoff"] = package
spec.loader.exec_module(package)

import numpy as np
import pandas as pd
from load_bakeoff.common import gaussian_quantiles, load_data, mape, validation_blocks
from load_bakeoff.lightgbm_features import feature_frame, recursive_features


class StudyTests(unittest.TestCase):
    def test_data_contract(self) -> None:
        data = load_data()
        self.assertEqual(len(data), 50400)
        self.assertEqual(len(data.loc[:"2019-09-30"]), 41616)
        self.assertEqual(len(data.loc["2019-10-01":"2019-12-31"]), 2208)
        self.assertEqual(len(data.loc["2020-01-01":"2020-01-07"]), 168)

    def test_validation_partitions_once(self) -> None:
        data = load_data().loc[:"2019-12-31"]
        blocks = validation_blocks(data)
        self.assertEqual([len(target) for _, target in blocks], [168] * 13 + [24])
        self.assertEqual(sum(len(target) for _, target in blocks), 2208)
        for history, target in blocks:
            self.assertEqual(history.index[-1] + pd.Timedelta(hours=1), target.index[0])

    def test_metric_units(self) -> None:
        self.assertAlmostEqual(mape(np.array([100.0, 200.0]), np.array([90.0, 180.0])), 10.0)
        q = gaussian_quantiles(np.array([100.0]), np.array([10.0]))
        self.assertAlmostEqual(q[0, 2], 100.0)
        self.assertTrue(np.all(np.diff(q, axis=1) > 0))

    def test_features_do_not_observe_current_or_future_target(self) -> None:
        index = pd.date_range("2017-01-01", periods=9000, freq="h")
        values = pd.Series(np.arange(9000, dtype=float) + 1000, index=index)
        before = feature_frame(values)
        altered = values.copy()
        altered.iloc[8900:] = -1e9
        after = feature_frame(altered)
        np.testing.assert_allclose(before.loc[index[8900]], after.loc[index[8900]])
        self.assertEqual(before.loc[index[8900], "mean_24h"], values.iloc[8876:8900].mean())
        self.assertEqual(before.loc[index[8900], "lag_8736h"], values.iloc[164])

    def test_recursive_and_training_features_match_at_origin(self) -> None:
        data = load_data().loc[:"2019-12-31"]
        row = recursive_features(data.to_list(), pd.Timestamp("2020-01-01"))
        extended = pd.concat([data, pd.Series([999999.0], index=[pd.Timestamp("2020-01-01")])])
        expected = feature_frame(extended).iloc[-1].to_numpy()
        np.testing.assert_allclose(row, expected, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
