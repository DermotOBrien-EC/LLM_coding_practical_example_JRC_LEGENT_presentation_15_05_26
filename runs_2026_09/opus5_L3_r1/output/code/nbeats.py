"""N-BEATS: a deep network that only ever sees past load values.

N-BEATS takes a fixed window of recent history (here one week, 168 hours)
and maps it straight to the next week, using a stack of fully connected
blocks. Each block predicts part of the answer and passes what it could not
explain to the next block. There are no calendar features and no lags chosen
by a human: whatever daily and weekly structure the network uses, it had to
find in the numbers.

That is the interesting comparison. LightGBM is told that Sunday is
different and that 1 January is a holiday. N-BEATS is told nothing, and has
to infer the weekday pattern from a 168-hour context window that happens to
contain exactly one of every weekday.

The model is trained with a quantile-regression loss, so it produces a full
predictive distribution rather than a single line, and prediction intervals
come from sampling it.

Two configurations are compared on the validation window. Within each
configuration the number of epochs is decided by early stopping on the
validation loss, then the winning configuration is retrained from scratch on
train plus validation for that same number of epochs. Retraining is
necessary because the final model must use the December data, and early
stopping cannot be used at that point without holding some of it back.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression

import common as c

warnings.filterwarnings("ignore")

INPUT_CHUNK: int = 168  # one week of context
OUTPUT_CHUNK: int = 168  # the whole forecast horizon in one shot
MAX_EPOCHS: int = 30
PATIENCE: int = 4
N_SAMPLES: int = 500

CANDIDATES: list[dict[str, Any]] = [
    {"learning_rate": 1e-3, "batch_size": 128, "dropout": 0.0},
    {"learning_rate": 5e-4, "batch_size": 128, "dropout": 0.1},
]


def _build(params: dict[str, Any], n_epochs: int, patience: int | None, work_dir: str) -> NBEATSModel:
    """Create an N-BEATS model with the darts default stack configuration."""
    return NBEATSModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        generic_architecture=True,  # the darts default stack setup
        n_epochs=n_epochs,
        batch_size=int(params["batch_size"]),
        dropout=float(params["dropout"]),
        optimizer_kwargs={"lr": float(params["learning_rate"])},
        likelihood=QuantileRegression(quantiles=list(c.QUANTILES)),
        random_state=c.SEED,
        model_name="nbeats",
        work_dir=work_dir,
        save_checkpoints=False,
        force_reset=True,
        pl_trainer_kwargs=c.torch_trainer_kwargs(patience),
    )


def select_hyperparameters(series: pd.Series) -> tuple[dict[str, Any], float, int, list[dict[str, Any]]]:
    """Train each candidate on the training years, score it on validation."""
    train = c.to_darts_series(series.loc[: c.TRAIN_END])
    full = c.to_darts_series(c.train_plus_val(series))

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    full_scaled = scaler.transform(full)
    val_scaled = full_scaled.drop_before(pd.Timestamp(c.VAL_START.tz_localize(None)) - pd.Timedelta(hours=INPUT_CHUNK + 1))

    trace: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], int] | None = None

    for params in CANDIDATES:
        c.set_seeds()
        model = _build(params, MAX_EPOCHS, PATIENCE, work_dir=str(c.ARTIFACT_DIR))
        model.fit(train_scaled, val_series=val_scaled, verbose=False)
        epochs_used = int(model.trainer.current_epoch)

        score = c.rolling_origin_mape_torch(model, full_scaled, scaler, series)
        trace.append({**params, "epochs_used": epochs_used, "validation_mape_pct": score})
        print(f"  {params} epochs={epochs_used} -> {score:.3f}%", flush=True)
        if best is None or score < best[0]:
            best = (score, params, epochs_used)

    assert best is not None
    return best[1], best[0], best[2], trace


def run(series: pd.Series) -> c.ForecastResult:
    """Select on validation, retrain on train+validation, forecast the test week."""
    start = time.time()

    best_params, best_score, epochs_used, trace = select_hyperparameters(series)

    fitting_data = c.to_darts_series(c.train_plus_val(series))
    scaler = Scaler()
    fitting_scaled = scaler.fit_transform(fitting_data)

    c.set_seeds()
    model = _build(best_params, max(epochs_used, 1), patience=None, work_dir=str(c.ARTIFACT_DIR))
    model.fit(fitting_scaled, verbose=False)

    prediction = scaler.inverse_transform(model.predict(n=c.HORIZON, num_samples=N_SAMPLES))
    samples = np.asarray(prediction.all_values(), dtype=float)[:, 0, :]

    test = series.loc[c.TEST_START : c.TEST_END]
    quantiles = c.quantile_frame(test.index, samples)
    # With a quantile-regression loss the natural point forecast is the
    # predicted median, not the mean of the sample cloud.
    point = pd.Series(quantiles["0.5"].to_numpy(), index=test.index, name="nbeats")

    return c.ForecastResult(
        name="nbeats",
        point_forecast=point,
        runtime_seconds=time.time() - start,
        hyperparameters={
            **best_params,
            "input_chunk_length": INPUT_CHUNK,
            "output_chunk_length": OUTPUT_CHUNK,
            "generic_architecture": True,
            "max_epochs": MAX_EPOCHS,
            "epochs_used_after_early_stopping": epochs_used,
            "early_stopping_patience": PATIENCE,
            "likelihood": "QuantileRegression",
            "n_samples_for_intervals": N_SAMPLES,
            "scaler": "MinMax on the fitting window only",
        },
        quantile_forecast=quantiles,
        validation_mape_pct=best_score,
        extras={"grid_search": trace},
    )


if __name__ == "__main__":
    result = run(c.load_load_series())
    result.save()
    print(result.name, result.validation_mape_pct, result.runtime_seconds)
