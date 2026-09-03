"""The sixth entry: a modern deep architecture, univariate, same setup as N-BEATS.

The study asks for PatchTST. The darts version installed here (0.41.0) does
not expose a `PatchTSTModel`; the classes it offers in that family are
`TSMixerModel`, `TFTModel` and `TransformerModel`. The study text allows
`darts.models.TSMixerModel` as the alternative, and that is what this file
uses. The substitution is recorded in transcript.md as well as here, and the
model class actually used is written into metrics.json under the model's
hyperparameters so nobody can be misled by the slot name.

TSMixer is a close relative of PatchTST in spirit: both abandon recurrence
and instead mix information across time positions with simple, cheap layers,
PatchTST with attention over patches of the series and TSMixer with
fully-connected mixing across the time axis. For a univariate hourly series
with a one-week context window the two behave similarly, and neither gets to
see a calendar.

Everything else -- context window, horizon, epoch budget, early stopping,
quantile loss, sampling for intervals, and the retrain-on-train-plus-
validation step -- is identical to N-BEATS, so the comparison between the two
deep models is about architecture and nothing else.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from darts.dataprocessing.transformers import Scaler
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression

import common as c

warnings.filterwarnings("ignore")

MODEL_CLASS_NAME: str = "darts.models.TSMixerModel"
REQUESTED_MODEL: str = "darts.models.PatchTSTModel"

INPUT_CHUNK: int = 168
OUTPUT_CHUNK: int = 168
MAX_EPOCHS: int = 30
PATIENCE: int = 4
N_SAMPLES: int = 500

CANDIDATES: list[dict[str, Any]] = [
    {"learning_rate": 1e-3, "batch_size": 128, "hidden_size": 64, "dropout": 0.1},
    {"learning_rate": 5e-4, "batch_size": 128, "hidden_size": 128, "dropout": 0.1},
]


def _build(params: dict[str, Any], n_epochs: int, patience: int | None, work_dir: str) -> TSMixerModel:
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
    point = pd.Series(quantiles["0.5"].to_numpy(), index=test.index, name="patchtst")

    return c.ForecastResult(
        name="patchtst",
        point_forecast=point,
        runtime_seconds=time.time() - start,
        hyperparameters={
            **best_params,
            "requested_model": REQUESTED_MODEL,
            "model_class_used": MODEL_CLASS_NAME,
            "substitution_reason": "darts 0.41.0 does not expose PatchTSTModel",
            "input_chunk_length": INPUT_CHUNK,
            "output_chunk_length": OUTPUT_CHUNK,
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
