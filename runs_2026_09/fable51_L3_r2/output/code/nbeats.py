"""N-BEATS: a deep, purely feed-forward neural forecaster.

N-BEATS looks at the last week of hourly load (168 numbers) and writes
out the next week (168 numbers) through a deep stack of fully connected
blocks; each block explains a part of the input and hands the remainder
to the next one. It sees no calendar and no holiday flag, only the
recent shape of the series, so it is the strongest "pattern continuation"
model in the study. The darts implementation is used with its default
stack layout; the network's last layer predicts five quantiles of the
load, which give both the point forecast (the median) and the intervals.

Training stops early when the validation MAPE stops improving. The
learning rate is chosen on the validation weeks, and the number of epochs
found there is reused for the final refit on train + validation, because
that refit has no validation window left to stop on.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import pandas as pd
from darts.models import NBEATSModel
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
LEARNING_RATES: tuple[float, ...] = (1e-3, 3e-4)
ARCHITECTURE: dict[str, Any] = {
    "input_chunk_length": HORIZON,
    "output_chunk_length": HORIZON,
    "generic_architecture": True,
    "num_stacks": 30,
    "num_blocks": 1,
    "num_layers": 4,
    "layer_widths": 256,
    "expansion_coefficient_dim": 5,
}


def build(lr: float, trainer_kwargs: dict[str, Any], n_epochs: int) -> NBEATSModel:
    return NBEATSModel(
        **ARCHITECTURE,
        n_epochs=n_epochs,
        batch_size=BATCH_SIZE,
        random_state=SEED,
        likelihood=make_quantile_likelihood(),
        torch_metrics=MeanAbsolutePercentageError(),
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={**torch_trainer_kwargs(), **trainer_kwargs},
    )


def run(train: pd.Series, val: pd.Series, test_index: pd.DatetimeIndex) -> ForecastResult:
    with Stopwatch() as sw:
        # The network trains on the load divided by its training mean, so
        # values sit near 1. MAPE is unchanged by this rescaling.
        scale = float(train.mean())
        train_s, val_s = train / scale, val / scale
        history_s = pd.concat([train_s, val_s])
        blocks = validation_blocks(val.index)

        def evaluate(lr: float) -> dict[str, Any]:
            model, info = fit_with_early_stopping(
                lambda tk: build(lr, tk, MAX_EPOCHS), train_s, val_s, MAX_EPOCHS, PATIENCE
            )
            forecasts = [
                predict_quantiles(
                    model, history_s.loc[: block[0] - pd.Timedelta(hours=1)], HORIZON, scale
                )[0.5]
                for block in blocks
            ]
            val_mape = block_mape(val, forecasts)
            log(f"nbeats sweep lr={lr}: best epoch {info['best_epoch']}, val MAPE {val_mape:.3f} %")
            return {"lr": lr, **info, "val_mape_pct": round(val_mape, 4)}

        sweep_results: list[dict[str, Any]] = []
        for lr in LEARNING_RATES:
            sweep_results.append(cached_sweep(f"nbeats_lr{lr:g}", functools.partial(evaluate, lr)))
        best = min(sweep_results, key=lambda r: r["val_mape_pct"])
        n_epochs = max(1, int(best["best_epoch"]))

        # Refit on train + validation for the number of epochs that was
        # best on the validation weeks, then forecast the test week.
        final = build(float(best["lr"]), {}, n_epochs)
        final.fit(to_darts(history_s), verbose=False)
        quantiles = predict_quantiles(final, history_s, HORIZON, scale)
        for q in quantiles:
            quantiles[q].index = test_index

    return ForecastResult(
        name="nbeats",
        point=quantiles[0.5],
        quantiles=quantiles,
        runtime_seconds=sw.seconds,
        hyperparameters={
            **ARCHITECTURE,
            "learning_rate": best["lr"],
            "epochs_refit": n_epochs,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "likelihood": "quantile regression (0.025, 0.1, 0.5, 0.9, 0.975)",
            "accelerator": torch_trainer_kwargs()["accelerator"],
            "scaling": "divide by training mean",
        },
        validation={
            "sweep": sweep_results,
            "chosen_lr": best["lr"],
            "val_mape_pct": best["val_mape_pct"],
        },
    )
