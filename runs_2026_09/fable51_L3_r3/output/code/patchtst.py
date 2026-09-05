"""TSMixer, standing in for PatchTST.

darts 0.41.0 does not ship a PatchTSTModel, so this slot uses the
TSMixerModel the task names as its first alternative. TSMixer is an
all-MLP architecture that alternately mixes information across time steps
and across features; like N-BEATS it is univariate here and sees no
calendar information. The slot keeps the name "patchtst" in the outputs so
the metrics schema matches the study design; the substitution is recorded
in transcript.md.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from darts.models import TSMixerModel
from pytorch_lightning.callbacks import Callback

from common import HORIZON, ForecastResult
from torch_common import common_model_kwargs, select_refit_forecast

CANDIDATES: list[dict[str, Any]] = [
    {"hidden_size": 64, "ff_size": 64, "num_blocks": 2, "dropout": 0.1, "lr": 1e-3},
    {"hidden_size": 128, "ff_size": 128, "num_blocks": 3, "dropout": 0.1, "lr": 1e-3},
]


def make_model(params: dict[str, Any], n_epochs: int, callbacks: list[Callback]) -> TSMixerModel:
    return TSMixerModel(
        hidden_size=params["hidden_size"],
        ff_size=params["ff_size"],
        num_blocks=params["num_blocks"],
        dropout=params["dropout"],
        model_name="tsmixer",
        **common_model_kwargs(n_epochs, callbacks, params["lr"]),
    )


def run(train: pd.Series, val: pd.Series, horizon: int = HORIZON) -> ForecastResult:
    return select_refit_forecast("patchtst", CANDIDATES, make_model, train, val, horizon)


if __name__ == "__main__":
    from common import load_series, split

    train, val, _ = split(load_series())
    result = run(train, val)
    print(f"validation MAPE {result.validation_mape_pct:.2f} %  runtime {result.runtime_seconds:.1f} s")
    print(result.point.head())
