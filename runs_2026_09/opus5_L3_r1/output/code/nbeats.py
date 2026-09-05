"""N-BEATS: a deep network that only ever sees past load values.

N-BEATS takes a fixed window of recent history (here one week, 168 hours)
and maps it straight to the next week, using a stack of fully connected
blocks. Each block predicts part of the answer and passes what it could not
explain on to the next block. There are no calendar features and no lags
chosen by a human: whatever daily and weekly structure the network uses, it
had to find in the numbers alone.

That is what makes the comparison interesting. LightGBM is told that Sunday
is different and that 1 January is a public holiday. N-BEATS is told
nothing, and has to infer the weekday pattern from a 168-hour context window
that happens to contain exactly one of every weekday.

The model is trained with a quantile-regression loss, so it produces a whole
predictive distribution rather than a single line, and the prediction
intervals come from sampling it.

Two configurations are compared on the validation window. Within each one
the number of epochs is decided by early stopping on the validation loss;
the winning configuration is then retrained from scratch on train plus
validation for that same number of epochs.

The work is split into stages so that each stage is a short, restartable
job:

    python nbeats.py sweep 0    # train and score the first candidate
    python nbeats.py sweep 1    # train and score the second candidate
    python nbeats.py final      # retrain the winner and forecast the test week
    python nbeats.py            # all of the above in order
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import pandas as pd
from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression

import common as c

warnings.filterwarnings("ignore")

NAME: str = "nbeats"
INPUT_CHUNK: int = 168  # one week of context
OUTPUT_CHUNK: int = 168  # the whole forecast horizon, produced in one shot
MAX_EPOCHS: int = 22
PATIENCE: int = 4
N_SAMPLES: int = 500

CANDIDATES: list[dict[str, Any]] = [
    {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "dropout": 0.0,
        "input_chunk_length": INPUT_CHUNK,
    },
    {
        "learning_rate": 5e-4,
        "batch_size": 128,
        "dropout": 0.1,
        "input_chunk_length": INPUT_CHUNK,
    },
]


def build(params: dict[str, Any], n_epochs: int, patience: int | None, work_dir: str) -> NBEATSModel:
    """An N-BEATS model with the darts default (generic) stack configuration."""
    return NBEATSModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        generic_architecture=True,
        n_epochs=n_epochs,
        batch_size=int(params["batch_size"]),
        dropout=float(params["dropout"]),
        optimizer_kwargs={"lr": float(params["learning_rate"])},
        likelihood=QuantileRegression(quantiles=list(c.QUANTILES)),
        random_state=c.SEED,
        model_name=NAME,
        work_dir=work_dir,
        save_checkpoints=False,
        force_reset=True,
        pl_trainer_kwargs=c.torch_trainer_kwargs(patience),
    )


EXTRA_HYPERPARAMETERS: dict[str, Any] = {
    "output_chunk_length": OUTPUT_CHUNK,
    "generic_architecture": True,
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
