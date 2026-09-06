from __future__ import annotations

import json
import runpy
from typing import Any

runpy.run_path("code/forecast.py", run_name="study_bootstrap")

import numpy as np
from study.common import QUANTILES, ROOT, TEST_INDEX, TEST_START, load_data
from study.lightgbm_features import fit_models, recursive_forecast


class Recorder:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.values: list[float] = []

    def predict(self, row: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(row)
        self.values.append(float(prediction[0]))
        return prediction


metadata = json.loads((ROOT / "artifacts/lightgbm_run.json").read_text())
config = {key: metadata["hyperparameters"][key] for key in ["num_leaves", "n_estimators"]}
series = load_data()
history = series.loc[series.index < TEST_START]
models = {q: Recorder(model) for q, model in fit_models(history, config, QUANTILES).items()}
forecast = recursive_forecast(models, history, TEST_INDEX)
raw = np.column_stack([models[q].values for q in QUANTILES])
changed = int(np.count_nonzero(raw[:, 2] != forecast[:, 2]))
output = {"median_trajectory_changed_by_interval_sorting": changed, "hours_checked": 168,
          "interpretation": "Sorting can change the median; candidate validation and final refit must both fit the same five quantiles."}
(ROOT / "artifacts/quantile_path_check.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output))
