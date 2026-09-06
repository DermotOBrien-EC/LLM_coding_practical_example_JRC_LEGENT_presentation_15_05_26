from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from common import ForecastResult, SEED, elapsed_seconds, mape_pct, timer


def _series(values: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(values.astype("float32"), fill_missing_dates=False, freq="h")


def _scaled(values: pd.Series, center: float, scale: float) -> pd.Series:
    return ((values.astype(float) - center) / scale).astype("float32")


def _build_model(dropout: float) -> NBEATSModel:
    callback = EarlyStopping(monitor="val_loss", patience=4, mode="min", min_delta=0.0001)
    return NBEATSModel(
        input_chunk_length=168,
        output_chunk_length=168,
        generic_architecture=True,
        num_stacks=4,
        num_blocks=1,
        num_layers=3,
        layer_widths=128,
        dropout=dropout,
        likelihood=QuantileRegression(quantiles=[0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]),
        random_state=SEED,
        n_epochs=30,
        batch_size=64,
        optimizer_kwargs={"lr": 1e-3},
        pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False, "callbacks": [callback]},
        force_reset=True,
        save_checkpoints=False,
    )


def forecast(train: pd.Series, validation: pd.Series, train_validation: pd.Series, test: pd.Series) -> ForecastResult:
    start = timer()
    torch.set_num_threads(4)
    center = float(train.mean())
    scale = float(train.std())
    train_scaled = _scaled(train, center, scale)
    validation_scaled = _scaled(validation, center, scale)
    train_validation_scaled = _scaled(train_validation, center, scale)
    val_slice = train_validation_scaled.iloc[-(168 + len(validation)) :]

    candidates = [{"dropout": 0.0}, {"dropout": 0.1}]
    best_params: dict[str, float] | None = None
    best_score = float("inf")
    validation_scores: list[dict[str, float]] = []
    for params in candidates:
        model = _build_model(**params)
        model.fit(_series(train_scaled), val_series=_series(validation_scaled), verbose=False)
        pred_scaled = model.predict(len(validation), series=_series(train_scaled), num_samples=1, verbose=False, random_state=SEED)
        pred = pred_scaled.values(copy=False).reshape(-1).astype(float) * scale + center
        score = mape_pct(validation.to_numpy(), pred)
        validation_scores.append({**params, "validation_mape_pct": score})
        if score < best_score:
            best_score = score
            best_params = params
    if best_params is None:
        raise RuntimeError("N-BEATS validation search produced no fitted model")

    model = _build_model(**best_params)
    model.fit(_series(train_validation_scaled), val_series=_series(val_slice), verbose=False)
    pred_series = model.predict(len(test), series=_series(train_validation_scaled), num_samples=300, verbose=False, random_state=SEED)
    samples = pred_series.all_values(copy=False)[:, 0, :].astype(float) * scale + center
    q025, q10, q25, q50, q75, q90, q975 = np.quantile(samples, [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975], axis=1)
    point = q50.astype(float)
    return ForecastResult(
        name="nbeats",
        point=point,
        runtime_seconds=elapsed_seconds(start),
        hyperparameters={"selected": best_params, "validation_mape_pct": best_score, "validation_grid": validation_scores},
        validation_mape_pct=best_score,
        q025=q025,
        q10=q10,
        q25=q25,
        q50=q50,
        q75=q75,
        q90=q90,
        q975=q975,
    )
