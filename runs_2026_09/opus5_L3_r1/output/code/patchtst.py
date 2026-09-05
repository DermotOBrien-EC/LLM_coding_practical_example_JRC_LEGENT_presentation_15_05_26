"""The sixth entry: a modern deep architecture, univariate, same setup as N-BEATS.

The study asks for PatchTST. The darts version installed here (0.41.0) does
not expose a `PatchTSTModel`; the classes it offers in that family are
`TSMixerModel`, `TFTModel` and `TransformerModel`. The study text allows
`darts.models.TSMixerModel` as the alternative, and that is what this file
uses. The substitution is recorded in transcript.md as well as here, and the
class actually used is written into metrics.json under this model's
hyperparameters, so the slot name cannot mislead anyone.

TSMixer is a close relative of PatchTST in spirit. Both give up on
recurrence and instead mix information across time positions with cheap
layers: PatchTST with attention over patches of the series, TSMixer with
fully connected mixing along the time axis. For a univariate hourly series
with a one-week context window the two behave similarly, and neither is
allowed to see a calendar.

Everything else is identical to N-BEATS on purpose (context window, horizon,
epoch budget, early stopping, quantile loss, sampling for intervals, and the
retrain-on-train-plus-validation step), so the difference between the two
deep entries is architecture and nothing else.

Stages, as for N-BEATS:

    python patchtst.py sweep 0
    python patchtst.py sweep 1
    python patchtst.py final
    python patchtst.py
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import pandas as pd
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression

import common as c

warnings.filterwarnings("ignore")

NAME: str = "patchtst"
MODEL_CLASS_NAME: str = "darts.models.TSMixerModel"
REQUESTED_MODEL: str = "darts.models.PatchTSTModel"

INPUT_CHUNK: int = 168
OUTPUT_CHUNK: int = 168
MAX_EPOCHS: int = 22
PATIENCE: int = 4
N_SAMPLES: int = 500

CANDIDATES: list[dict[str, Any]] = [
    {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "hidden_size": 64,
        "dropout": 0.1,
        "input_chunk_length": INPUT_CHUNK,
    },
    {
        "learning_rate": 5e-4,
        "batch_size": 128,
        "hidden_size": 128,
        "dropout": 0.1,
        "input_chunk_length": INPUT_CHUNK,
    },
]


def build(params: dict[str, Any], n_epochs: int, patience: int | None, work_dir: str) -> TSMixerModel:
    return TSMixerModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        hidden_size=int(params["hidden_size"]),
        dropout=float(params["dropout"]),
        n_epochs=n_epochs,
        batch_size=int(params["batch_size"]),
        optimizer_kwargs={"lr": float(params["learning_rate"])},
        likelihood=QuantileRegression(quantiles=list(c.QUANTILES)),
        random_state=c.SEED,
        model_name="tsmixer",
        work_dir=work_dir,
        save_checkpoints=False,
        force_reset=True,
        pl_trainer_kwargs=c.torch_trainer_kwargs(patience),
    )


EXTRA_HYPERPARAMETERS: dict[str, Any] = {
    "requested_model": REQUESTED_MODEL,
    "model_class_used": MODEL_CLASS_NAME,
    "substitution_reason": "darts 0.41.0 does not expose PatchTSTModel",
    "output_chunk_length": OUTPUT_CHUNK,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
    "early_stopping_monitor": "validation loss (quantile loss)",
    "likelihood": "QuantileRegression",
    "scaler": "MinMax, fitted on the fitting window only",
}


def sweep(series: pd.Series, index: int) -> dict[str, Any]:
    """Train and score one candidate configuration."""
    return c.deep_sweep_stage(NAME, build, CANDIDATES[index], index, series, MAX_EPOCHS, PATIENCE)


def final(series: pd.Series) -> c.ForecastResult:
    """Retrain the winning candidate on train+validation and forecast the test week."""
    return c.deep_final_stage(NAME, build, CANDIDATES, series, N_SAMPLES, EXTRA_HYPERPARAMETERS)


def run(series: pd.Series) -> c.ForecastResult:
    """Every stage, in order."""
    for index in range(len(CANDIDATES)):
        sweep(series, index)
    return final(series)


if __name__ == "__main__":
    load = c.load_load_series()
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        record = sweep(load, int(sys.argv[2]))
        print(NAME, "candidate", record["candidate_index"], record)
    elif len(sys.argv) > 1 and sys.argv[1] == "final":
        result = final(load)
        result.save()
        print(NAME, result.validation_mape_pct, result.runtime_seconds)
    else:
        result = run(load)
        result.save()
        print(NAME, result.validation_mape_pct, result.runtime_seconds)
