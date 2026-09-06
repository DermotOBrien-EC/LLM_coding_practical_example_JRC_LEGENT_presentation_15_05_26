"""N-BEATS: a deep model that learns the shape of a week from scratch.

N-BEATS reads a block of 168 consecutive hours and emits the next 168 hours
in one shot. It is a stack of fully connected blocks, each of which tries to
explain part of what the previous blocks left over. There is no calendar
input at all: it is given nothing but the numbers, so anything it knows about
weekends or holidays it has had to infer from the shape of the past week it
is looking at.

That makes it an interesting counterweight to Prophet and LightGBM. It has
the most capacity of the six and the least prior knowledge.

The forecast is probabilistic: instead of one number per hour the network
predicts a set of quantiles, so the prediction interval comes out of the same
fit rather than from a separate assumption about the error distribution.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from darts.models import NBEATSModel
from darts.utils.likelihood_models.torch import QuantileRegression

from common import (
    HORIZON,
    QUANTILE_LEVELS,
    SEED,
    ForecastResult,
    select_and_refit_torch,
    trainer_kwargs,
)

# Only the optimiser settings are swept. The architecture stays at the darts
# default generic stack configuration, as the study brief asks.
CANDIDATES: tuple[dict[str, Any], ...] = (
    {"learning_rate": 1e-3, "batch_size": 1024},
    {"learning_rate": 3e-4, "batch_size": 1024},
)

MAX_EPOCHS: int = 30
PATIENCE: int = 5


def _factory(params: dict[str, Any], n_epochs: int, callbacks: list[Any]) -> NBEATSModel:
    """Build a fresh, unfitted N-BEATS with the given optimiser settings."""
    return NBEATSModel(
        input_chunk_length=HORIZON,
        output_chunk_length=HORIZON,
        generic_architecture=True,  # darts default stacks
        n_epochs=n_epochs,
        batch_size=int(params["batch_size"]),
        optimizer_kwargs={"lr": float(params["learning_rate"])},
        likelihood=QuantileRegression(list(QUANTILE_LEVELS)),
        random_state=SEED,
        pl_trainer_kwargs=trainer_kwargs(callbacks),
        force_reset=True,
        save_checkpoints=False,
    )


def run(series: pd.Series) -> ForecastResult:
    """Select on validation, refit on train+validation, forecast the test week."""
    return select_and_refit_torch(
        name="nbeats",
        factory=_factory,
        series=series,
        candidates=CANDIDATES,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
    )
