from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping

from common import FINAL_TRAIN_END, ModelOutput, TEST_END, TEST_START, TRAIN_END, TRAIN_START, VAL_END, VAL_START, SEED, mape_pct

EPOCHS = 20
CANDIDATES = [
    {"num_stacks": 4, "num_blocks": 1, "num_layers": 2, "layer_widths": 128, "dropout": 0.05},
]


def to_darts(series: pd.Series) -> TimeSeries:
    clean = series.copy().asfreq("h")
    return TimeSeries.from_series(clean)


def make_model(params: dict[str, object], model_name: str, work_dir: Path, use_early_stopping: bool) -> NBEATSModel:
    torch.manual_seed(SEED)
    callbacks = [EarlyStopping(monitor="val_loss", patience=3, min_delta=0.0001, mode="min")] if use_early_stopping else []
    return NBEATSModel(
        input_chunk_length=168,
        output_chunk_length=168,
        n_epochs=EPOCHS,
        batch_size=256,
        random_state=SEED,
        likelihood=QuantileRegression(quantiles=[0.025, 0.1, 0.5, 0.9, 0.975]),
        optimizer_kwargs={"lr": 1e-3},
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "callbacks": callbacks,
            "logger": False,
        },
        model_name=model_name,
        work_dir=str(work_dir),
        force_reset=True,
        save_checkpoints=False,
        **params,
    )


def select_params(series: pd.Series) -> dict[str, object]:
    train = series.loc[TRAIN_START:TRAIN_END]
    val = series.loc[VAL_START:VAL_END]
    train_ts = to_darts(train)
    val_ts = to_darts(val)
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_ts)
    val_scaled = scaler.transform(val_ts)
    work_dir = Path("code") / ".darts_nbeats_select"
    best = CANDIDATES[0]
    best_score = float("inf")
    for i, params in enumerate(CANDIDATES):
        model = make_model(params, f"nbeats_select_{i}", work_dir, use_early_stopping=True)
        model.fit(train_scaled, val_series=val_scaled, verbose=False)
        pred_scaled = model.predict(len(val), series=train_scaled, num_samples=1, verbose=False)
        pred = scaler.inverse_transform(pred_scaled).to_series()
        pred.index = val.index
        score = mape_pct(val, pred)
        if score < best_score:
            best = params
            best_score = score
    return {**best, "validation_mape_pct": best_score, "epochs": EPOCHS, "early_stopping_patience": 3}


def quantiles_from_samples(samples: TimeSeries, scaler: Scaler, index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    samples_unscaled = scaler.inverse_transform(samples)
    values = samples_unscaled.all_values(copy=False)[:, 0, :]
    quantile_values = np.quantile(values, [0.025, 0.1, 0.5, 0.9, 0.975], axis=1)
    keys = ["q025", "q10", "q50", "q90", "q975"]
    return {key: pd.Series(quantile_values[i], index=index) for i, key in enumerate(keys)}


def run(series: pd.Series) -> ModelOutput:
    selected = select_params(series)
    params = {k: v for k, v in selected.items() if k not in {"validation_mape_pct", "epochs", "early_stopping_patience"}}
    final_train = series.loc[TRAIN_START:FINAL_TRAIN_END]
    test_index = pd.date_range(TEST_START, TEST_END, freq="h")
    train_ts = to_darts(final_train)
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_ts)
    model = make_model(params, "nbeats_final", Path("code") / ".darts_nbeats_final", use_early_stopping=False)
    model.fit(train_scaled, val_series=train_scaled[-336:], verbose=False)
    samples = model.predict(len(test_index), series=train_scaled, num_samples=100, verbose=False, random_state=SEED)
    qs = quantiles_from_samples(samples, scaler, test_index)
    forecast = qs["q50"].rename("forecast")
    return ModelOutput(
        name="nbeats",
        forecast=forecast,
        runtime_seconds=0.0,
        hyperparameters=selected,
        lower_80=qs["q10"],
        upper_80=qs["q90"],
        lower_95=qs["q025"],
        upper_95=qs["q975"],
        q10=qs["q10"],
        q50=qs["q50"],
        q90=qs["q90"],
        fitted_model=model,
    )
