"""N-BEATS: a deep stack of plain feed-forward blocks, one week in, one week out.

The model reads the last 168 hours and writes the next 168 in one shot.
It sees no calendar information at all, so it must infer where it is in
the week from the shape of the input window, and it cannot know that the
first test day is a public holiday.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from darts.models import NBEATSModel
from pytorch_lightning.callbacks import Callback

from common import HORIZON, ForecastResult
from torch_common import common_model_kwargs, select_refit_forecast

CANDIDATES: list[dict[str, Any]] = [
    {"num_stacks": 30, "num_blocks": 1, "num_layers": 4, "layer_widths": 256, "lr": 1e-3},
    {"num_stacks": 10, "num_blocks": 1, "num_layers": 4, "layer_widths": 512, "lr": 1e-3},
]


def make_model(params: dict[str, Any], n_epochs: int, callbacks: list[Callback]) -> NBEATSModel:
    return NBEATSModel(
        generic_architecture=True,
        num_stacks=params["num_stacks"],
        num_blocks=params["num_blocks"],
        num_layers=params["num_layers"],
        layer_widths=params["layer_widths"],
        model_name="nbeats",
        **common_model_kwargs(n_epochs, callbacks, params["lr"]),
    )


def run(train: pd.Series, val: pd.Series, horizon: int = HORIZON) -> ForecastResult:
    return select_refit_forecast("nbeats", CANDIDATES, make_model, train, val, horizon)


if __name__ == "__main__":
    from common import load_series, split

    train, val, _ = split(load_series())
    result = run(train, val)
    print(f"validation MAPE {result.validation_mape_pct:.2f} %  runtime {result.runtime_seconds:.1f} s")
    print(result.point.head())
