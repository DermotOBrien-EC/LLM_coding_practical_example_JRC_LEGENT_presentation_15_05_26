"""N-BEATS: a deep network of stacked fully-connected blocks.

The network reads the last 168 hours and writes the next 168 hours in one
shot, learning the daily and weekly shapes from the 41,000 sliding
windows of the training period. It sees only the load itself, no
calendar, so a holiday looks to it like an unexplained dip in the input.

The darts default stack configuration is kept (30 generic stacks, width
256). Only the learning rate is searched, and the number of epochs is set
by early stopping on validation MAPE. The shared training loop lives in
common.run_darts_torch_model.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from darts.models import NBEATSModel

import common

FIXED: dict[str, Any] = {
    "generic_architecture": True,
    "num_stacks": 30,
    "num_blocks": 1,
    "num_layers": 4,
    "layer_widths": 256,
    "expansion_coefficient_dim": 5,
}
CANDIDATES: list[dict[str, Any]] = [{"lr": 1e-3}, {"lr": 3e-4}]


def run(s: pd.Series) -> common.ModelResult:
    result = common.run_darts_torch_model("nbeats", NBEATSModel, FIXED, CANDIDATES, s)
    result.notes = "Quantiles predicted directly by the network (quantile-regression head), sorted per hour to remove crossings."
    return result


if __name__ == "__main__":
    common.silence_warnings()
    s = common.load_series()
    result = run(s)
    result.save()
    print(f"nbeats: test MAPE {common.mape(common.test_slice(s), result.point):.2f} %, runtime {result.runtime_seconds:.0f} s")
