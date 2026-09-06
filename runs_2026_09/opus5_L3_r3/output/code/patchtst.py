"""The "PatchTST slot": TSMixer, because PatchTST is not in this darts build.

SUBSTITUTION NOTE. The study brief asks for `darts.models.PatchTSTModel`
"or `darts.models.TSMixerModel`", whichever the installed darts version
exposes. darts 0.41.0 in this environment ships `TSMixerModel` but not
`PatchTSTModel`, so TSMixer fills the slot. The substitution is recorded in
`transcript.md` as well as here.

TSMixer is not a transformer: it has no attention. It is a stack of small
fully connected layers applied alternately along the time axis and the
feature axis. For a univariate series like this one the feature axis has
length one, so what it really is here is a learned mixing of the 168 input
hours. It plays the same role in the bake-off that PatchTST would have
played: a modern deep architecture, different in kind from N-BEATS, trained
on the same context length and the same horizon.

Like N-BEATS it sees only the numbers. No calendar, no holiday flag.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from darts.models import TSMixerModel
from darts.utils.likelihood_models.torch import QuantileRegression

from common import (
    HORIZON,
    QUANTILE_LEVELS,
    SEED,
    ForecastResult,
    select_and_refit_torch,
    trainer_kwargs,
)

SUBSTITUTION: str = (
    "darts 0.41.0 does not expose PatchTSTModel; TSMixerModel was used instead, "
    "as permitted by the study brief."
)

CANDIDATES: tuple[dict[str, Any], ...] = (
    {"learning_rate": 1e-3, "batch_size": 1024},
    {"learning_rate": 3e-4, "batch_size": 1024},
)

MAX_EPOCHS: int = 30
PATIENCE: int = 5


def _factory(params: dict[str, Any], n_epochs: int, callbacks: list[Any]) -> TSMixerModel:
    """Build a fresh, unfitted TSMixer with the given optimiser settings."""
    return TSMixerModel(
        input_chunk_length=HORIZON,
        output_chunk_length=HORIZON,
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
    result = select_and_refit_torch(
        name="patchtst",
        factory=_factory,
        series=series,
        candidates=CANDIDATES,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
    )
    result.hyperparameters["substitution"] = SUBSTITUTION
    result.hyperparameters["darts_class"] = "TSMixerModel"
    return result
