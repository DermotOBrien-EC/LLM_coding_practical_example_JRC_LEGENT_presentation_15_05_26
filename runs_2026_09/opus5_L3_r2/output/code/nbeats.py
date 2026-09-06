"""N-BEATS - a deep network that only ever sees the load series.

N-BEATS is a stack of fully connected blocks. Each block looks at the last
168 hours, produces a forecast for the next 168, and passes on what it could
not explain to the next block, which tries again. Nothing about calendars or
holidays is given to it; if it is going to know that Sundays are quiet it has
to work that out from the numbers alone. That is exactly what makes it an
interesting entry: it is the "let the model find the structure" end of the
complexity gradient.

Following the prompt we keep the published default stack (30 stacks, four
256-wide layers each) and tune only how the network is trained - the learning
rate and the batch size - on the validation set. Training stops early when
the validation loss stops improving, and the number of epochs that took is
then reused for the final refit on train + validation.

The forecast is probabilistic: instead of one number per hour the network
predicts a set of quantiles, so we can draw samples from it and read off
prediction intervals.
"""

from __future__ import annotations

import time
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel

from common import (
    HORIZON,
    SEED,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    VALIDATION_ORIGINS,
    ModelForecast,
    darts_series,
    best_epoch,
    fitted_epochs,
    history_before,
    horizon_index,
    mape,
    quantile_likelihood,
    torch_trainer_kwargs,
    window,
)

INPUT_CHUNK: int = 168
OUTPUT_CHUNK: int = 168
MAX_EPOCHS: int = 30
PATIENCE: int = 5
N_SAMPLES: int = 500

# The quantiles the network is asked to learn. The five we report are in here;
# the extra middle ones just help the loss shape the distribution.
TRAIN_QUANTILES: tuple[float, ...] = (
    0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975,
)
REPORT_QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.5, 0.9, 0.975)

CANDIDATES: tuple[dict[str, object], ...] = (
    {"learning_rate": 1e-3, "batch_size": 512},
    {"learning_rate": 5e-4, "batch_size": 256},
)


def _build(params: dict[str, object], max_epochs: int, patience: int | None) -> NBEATSModel:
    return NBEATSModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        # published default stack configuration, left alone on purpose
        generic_architecture=True,
        num_stacks=30,
        num_blocks=1,
        num_layers=4,
        layer_widths=256,
        batch_size=int(params["batch_size"]),
        optimizer_kwargs={"lr": float(params["learning_rate"])},
        likelihood=quantile_likelihood(TRAIN_QUANTILES),
        random_state=SEED,
        model_name="nbeats_bakeoff",
        force_reset=True,
        save_checkpoints=False,
        pl_trainer_kwargs=torch_trainer_kwargs(max_epochs, patience),
    )


def _predict_from(
    model: NBEATSModel, scaler: Scaler, history: pd.Series, num_samples: int = 1
):
    scaled = scaler.transform(darts_series(history))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = model.predict(n=HORIZON, series=scaled, num_samples=num_samples)
    return scaler.inverse_transform(forecast)


def run(
    series: pd.Series,
    origins: Sequence[pd.Timestamp] = VALIDATION_ORIGINS,
    candidates: Sequence[dict[str, object]] = CANDIDATES,
) -> ModelForecast:
    started = time.perf_counter()

    train = window(series, TRAIN_START, TRAIN_END)
    val = window(series, VAL_START, VAL_END)

    # The scaler is fitted on the training data only, so nothing about the
    # validation or test period leaks into how the inputs are normalised.
    selection_scaler = Scaler()
    train_scaled = selection_scaler.fit_transform(darts_series(train))
    val_scaled = selection_scaler.transform(darts_series(val))

    selection: list[dict[str, object]] = []
    best: tuple[float, dict[str, object], int] | None = None
    for params in candidates:
        model = _build(params, MAX_EPOCHS, PATIENCE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_scaled, val_series=val_scaled, verbose=False)
        epochs_run = fitted_epochs(model)
        epochs_best = best_epoch(model, MAX_EPOCHS, PATIENCE)

        scores: list[float] = []
        for origin in origins:
            forecast = _predict_from(model, selection_scaler, history_before(series, origin))
            predicted = forecast.values().ravel().astype(float)
            scores.append(mape(series.loc[horizon_index(origin)].to_numpy(), predicted))
        mean_score = float(np.mean(scores))
        selection.append(
            {
                **params,
                "epochs_run": epochs_run,
                "epochs_best": epochs_best,
                "val_mape_pct": mean_score,
                "per_origin_mape_pct": [round(s, 3) for s in scores],
            }
        )
        if best is None or mean_score < best[0]:
            best = (mean_score, dict(params), epochs_best)

    assert best is not None
    val_mape, best_params, best_epochs = best
    final_epochs = max(1, best_epochs)

    # Refit on train + validation for the epoch count the validation run
    # settled on, then forecast the test week.
    history = history_before(series, TEST_START)
    final_scaler = Scaler()
    history_scaled = final_scaler.fit_transform(darts_series(history))
    final_model = _build(best_params, final_epochs, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(history_scaled, verbose=False)

    forecast = _predict_from(final_model, final_scaler, history, num_samples=N_SAMPLES)
    samples = forecast.all_values()[:, 0, :]
    index = horizon_index(TEST_START)
    point = pd.Series(samples.mean(axis=1).astype(float), index=index, name="nbeats")
    quantiles = {
        q: pd.Series(np.quantile(samples, q, axis=1).astype(float), index=index)
        for q in REPORT_QUANTILES
    }

    return ModelForecast(
        name="nbeats",
        point=point,
        quantiles=quantiles,
        hyperparameters={
            **best_params,
            "input_chunk_length": INPUT_CHUNK,
            "output_chunk_length": OUTPUT_CHUNK,
            "architecture": "generic default (30 stacks, 1 block, 4 layers, width 256)",
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "epochs_used_for_final_refit": final_epochs,
            "likelihood": f"QuantileRegression{list(TRAIN_QUANTILES)}",
            "n_prediction_samples": N_SAMPLES,
        },
        runtime_seconds=time.perf_counter() - started,
        validation_mape=val_mape,
        selection_log=selection,
    )
