"""The "PatchTST" slot in the bake-off - substituted with TSMixer.

SUBSTITUTION NOTE. The prompt asks for PatchTST via `darts.models.TSMixerModel`
or `darts.models.PatchTSTModel` "if the installed darts version exposes it".
The installed version is darts 0.41.0, and it does not expose a
`PatchTSTModel` (verified by listing `darts.models`; the catalogue offers
BlockRNNModel, DLinearModel, NBEATSModel, NHiTSModel, NLinearModel, TCNModel,
TFTModel, TSMixerModel, TiDEModel and TransformerModel). We therefore use
`TSMixerModel`, which the prompt names as the acceptable first option. It is
the closest relative of PatchTST in the catalogue: same long-horizon
forecasting family, same idea of mixing information along the time axis with
lightweight layers rather than one big attention matrix. The substitution is
recorded again in transcript.md. Throughout the outputs this model keeps the
name "patchtst" so it lines up with the schema the prompt fixed.

Like N-BEATS it sees nothing but the load series, and it is trained to predict
a set of quantiles so that it can produce prediction intervals.
"""

from __future__ import annotations

import time
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from darts.dataprocessing.transformers import Scaler
from darts.models import TSMixerModel

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

SUBSTITUTED_FOR: str = "PatchTST"
ACTUAL_MODEL: str = "darts.models.TSMixerModel"

INPUT_CHUNK: int = 168
OUTPUT_CHUNK: int = 168
MAX_EPOCHS: int = 30
PATIENCE: int = 5
N_SAMPLES: int = 500

TRAIN_QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
REPORT_QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.5, 0.9, 0.975)

CANDIDATES: tuple[dict[str, object], ...] = (
    {"learning_rate": 1e-3, "batch_size": 512, "hidden_size": 64},
    {"learning_rate": 5e-4, "batch_size": 256, "hidden_size": 128},
)


def _build(params: dict[str, object], max_epochs: int, patience: int | None) -> TSMixerModel:
    return TSMixerModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        hidden_size=int(params["hidden_size"]),
        batch_size=int(params["batch_size"]),
        optimizer_kwargs={"lr": float(params["learning_rate"])},
        likelihood=quantile_likelihood(TRAIN_QUANTILES),
        random_state=SEED,
        model_name="tsmixer_bakeoff",
        force_reset=True,
        save_checkpoints=False,
        pl_trainer_kwargs=torch_trainer_kwargs(max_epochs, patience),
    )


def _predict_from(
    model: TSMixerModel, scaler: Scaler, history: pd.Series, num_samples: int = 1
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
    point = pd.Series(samples.mean(axis=1).astype(float), index=index, name="patchtst")
    quantiles = {
        q: pd.Series(np.quantile(samples, q, axis=1).astype(float), index=index)
        for q in REPORT_QUANTILES
    }

    return ModelForecast(
        name="patchtst",
        point=point,
        quantiles=quantiles,
        hyperparameters={
            **best_params,
            "substituted_model": ACTUAL_MODEL,
            "requested_model": SUBSTITUTED_FOR,
            "substitution_reason": "darts 0.41.0 does not expose PatchTSTModel",
            "input_chunk_length": INPUT_CHUNK,
            "output_chunk_length": OUTPUT_CHUNK,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "epochs_used_for_final_refit": final_epochs,
            "likelihood": f"QuantileRegression{list(TRAIN_QUANTILES)}",
            "n_prediction_samples": N_SAMPLES,
        },
        runtime_seconds=time.perf_counter() - started,
        validation_mape=val_mape,
        selection_log=selection,
        notes=f"{ACTUAL_MODEL} substituted for {SUBSTITUTED_FOR}",
    )
