"""TSMixer in the "PatchTST" slot of the bake-off.

The installed darts (0.41.0) does not ship PatchTSTModel. The task names
darts.models.TSMixerModel as the alternative for this slot, so that is
what runs here. TSMixer is a recent (2023) deep model that alternates
mixing across time steps with mixing across features, using small
fully-connected layers instead of attention. It is not a transformer;
this substitution is recorded in transcript.md. The darts TransformerModel
was not used because it is a vanilla encoder-decoder that is slow and
weak at 168-step direct forecasting.

Same protocol as N-BEATS: 168 hours in, 168 hours out, learning rate
searched on validation, epoch count from early stopping on validation
MAPE, refit on Train+Val, quantile-regression head for the intervals.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from darts.models import TSMixerModel

import common

FIXED: dict[str, Any] = {
    "hidden_size": 64,
    "ff_size": 64,
    "num_blocks": 2,
    "dropout": 0.1,
}
CANDIDATES: list[dict[str, Any]] = [{"lr": 1e-3}, {"lr": 3e-4}]


def run(s: pd.Series) -> common.ModelResult:
    result = common.run_darts_torch_model("patchtst", TSMixerModel, FIXED, CANDIDATES, s)
    result.hyperparameters["substitution"] = "darts.models.TSMixerModel (PatchTSTModel not available in darts 0.41.0)"
    result.notes = "TSMixer substituted for PatchTST. Quantiles predicted directly by the network, sorted per hour to remove crossings."
    return result


if __name__ == "__main__":
    common.silence_warnings()
    s = common.load_series()
    result = run(s)
    result.save()
    print(f"patchtst (TSMixer): test MAPE {common.mape(common.test_slice(s), result.point):.2f} %, runtime {result.runtime_seconds:.0f} s")
