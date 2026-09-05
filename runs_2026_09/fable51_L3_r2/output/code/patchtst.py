"""PatchTST slot: TSMixer, the transformer-era univariate forecaster.

The installed darts (0.41.0) does not ship a PatchTST model, so the slot
is filled with darts' TSMixerModel, the first of the two options named in
the study brief. TSMixer comes from the same "patch the last window and
mix across time and features" family as PatchTST, but it replaces the
attention layers with alternating time-mixing and feature-mixing
multilayer perceptrons. Like N-BEATS it reads the last 168 hours and
writes the next 168, sees no calendar, and predicts five quantiles.

The width of the network is chosen on the validation weeks; the epoch
count found by early stopping is reused for the refit on train +
validation. The substitution is recorded in transcript.md.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import pandas as pd
from darts.models import TSMixerModel
from torchmetrics import MeanAbsolutePercentageError

from common import (
    HORIZON,
    SEED,
    ForecastResult,
    Stopwatch,
    block_mape,
    cached_sweep,
    fit_with_early_stopping,
    log,
    make_quantile_likelihood,
    predict_quantiles,
    to_darts,
    torch_trainer_kwargs,
    validation_blocks,
)

warnings.filterwarnings("ignore")

MAX_EPOCHS: int = 30
PATIENCE: int = 5
BATCH_SIZE: int = 512
LEARNING_RATE: float = 1e-3
CANDIDATES: dict[str, dict[str, Any]] = {
    "small": {"hidden_size": 64, "ff_size": 64, "num_blocks": 2, "dropout": 0.1},
    "large": {"hidden_size": 128, "ff_size": 256, "num_blocks": 4, "dropout": 0.2},
}


def build(name: str, trainer_kwargs: dict[str, Any], n_epochs: int) -> TSMixerModel:
    return TSMixerModel(
        input_chunk_length=HORIZON,
        output_chunk_length=HORIZON,
        **CANDIDATES[name],
        n_epochs=n_epochs,
        batch_size=BATCH_SIZE,
        random_state=SEED,
        likelihood=make_quantile_likelihood(),
        torch_metrics=MeanAbsolutePercentageError(),
        optimizer_kwargs={"lr": LEARNING_RATE},
        pl_trainer_kwargs={**torch_trainer_kwargs(), **trainer_kwargs},
    )


def run(train: pd.Series, val: pd.Series, test_index: pd.DatetimeIndex) -> ForecastResult:
    with Stopwatch() as sw:
        scale = float(train.mean())
        train_s, val_s = train / scale, val / scale
        history_s = pd.concat([train_s, val_s])
        blocks = validation_blocks(val.index)

        def evaluate(name: str) -> dict[str, Any]:
            model, info = fit_with_early_stopping(
                lambda tk: build(name, tk, MAX_EPOCHS), train_s, val_s, MAX_EPOCHS, PATIENCE
            )
            forecasts = [
                predict_quantiles(
                    model, history_s.loc[: block[0] - pd.Timedelta(hours=1)], HORIZON, scale
                )[0.5]
                for block in blocks
            ]
            val_mape = block_mape(val, forecasts)
            log(f"tsmixer sweep {name}: best epoch {info['best_epoch']}, val MAPE {val_mape:.3f} %")
            return {
                "candidate": name,
                **CANDIDATES[name],
                **info,
                "val_mape_pct": round(val_mape, 4),
            }

        sweep_results: list[dict[str, Any]] = []
        for name in CANDIDATES:
            sweep_results.append(cached_sweep(f"tsmixer_{name}", functools.partial(evaluate, name)))
        best = min(sweep_results, key=lambda r: r["val_mape_pct"])
        chosen = str(best["candidate"])
        n_epochs = max(1, int(best["best_epoch"]))

        final = build(chosen, {}, n_epochs)
        final.fit(to_darts(history_s), verbose=False)
        quantiles = predict_quantiles(final, history_s, HORIZON, scale)
        for q in quantiles:
            quantiles[q].index = test_index

    return ForecastResult(
        name="patchtst",
        point=quantiles[0.5],
        quantiles=quantiles,
        runtime_seconds=sw.seconds,
        hyperparameters={
            "model_class": "darts.models.TSMixerModel (PatchTSTModel not available in darts 0.41.0)",
            "candidate": chosen,
            **CANDIDATES[chosen],
            "input_chunk_length": HORIZON,
            "output_chunk_length": HORIZON,
            "learning_rate": LEARNING_RATE,
            "epochs_refit": n_epochs,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "likelihood": "quantile regression (0.025, 0.1, 0.5, 0.9, 0.975)",
            "accelerator": torch_trainer_kwargs()["accelerator"],
            "scaling": "divide by training mean",
        },
        validation={"sweep": sweep_results, "chosen": chosen, "val_mape_pct": best["val_mape_pct"]},
    )
